import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request, status
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response, StreamingResponse

from enterprise_ai_assistant.api.schemas import (
    ActionListResponse,
    AssistantResponse,
    ChatRequest,
    ConfirmationRequest,
    ConversationHistoryResponse,
    ConversationMessage,
    DemoAuthRequest,
    DemoAuthResponse,
    DevTokenRequest,
    HealthResponse,
    InputRequest,
    MemoryListResponse,
    TokenResponse,
)
from enterprise_ai_assistant.core.config import Settings, get_settings
from enterprise_ai_assistant.core.metrics import BUDGET_REJECTIONS, REGISTRY
from enterprise_ai_assistant.core.models import (
    PendingConfirmation,
    PendingInput,
    TaskStatus,
)
from enterprise_ai_assistant.core.observability import LLMUsageTracker
from enterprise_ai_assistant.core.runs import (
    END_SENTINEL,
    HEARTBEAT_SENTINEL,
    DisconnectMode,
    Publisher,
    Run,
    RunConflictError,
    RunManager,
    RunStatus,
    StreamGap,
)
from enterprise_ai_assistant.core.security import (
    CurrentIdentity,
    CurrentUser,
    Identity,
    create_access_token,
)
from enterprise_ai_assistant.repositories.users import (
    DemoUser,
    DemoUserRepository,
    conversation_id_for,
    normalize_name,
)

router = APIRouter(prefix="/api/v1")

_NODE_PROGRESS = {
    "recall": "正在读取你的历史偏好",
    "understand": "正在结合会话上下文理解你的请求",
    "direct_respond": "正在生成回复",
    "plan": "正在拆解任务并分析依赖关系",
    "select_task": "Supervisor 正在选择合适的专业 Agent",
    "initialize": "正在初始化专业 Agent",
    "decide": "专业 Agent 正在分析字段并选择工具",
    "confirm_tool": "工具调用需要人工确认",
    "execute_tool": "正在调用企业工具",
    "respond": "专业 Agent 正在生成回答",
    "apply_domain_result": "正在归并专业 Agent 的处理结果",
    "remember": "正在整理本轮值得记住的信息",
}

_SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _config(
    conversation_id: UUID, user_id: str, tracker: LLMUsageTracker | None = None
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "configurable": {"thread_id": str(conversation_id)},
        "tags": ["enterprise-assistant", "multi-agent"],
        "metadata": {"conversation_id": str(conversation_id), "user_id": user_id},
        "recursion_limit": 100,
    }
    if tracker is not None:
        config["callbacks"] = [tracker]
    return config


def _runs(app: Any) -> RunManager:
    manager: RunManager = app.state.runs
    return manager


def _budget_key(conversation_id: UUID) -> str:
    return f"budget:tokens:{conversation_id}"


async def _enforce_token_budget(app: Any, conversation_id: UUID, settings: Settings) -> None:
    """会话累计 token 超过预算时拒绝新一轮请求。

    预算计数仅用于成本护栏，Redis 不可用时放行而不是阻断业务。
    """
    if settings.conversation_token_budget <= 0:
        return
    try:
        spent = await app.state.redis.get(_budget_key(conversation_id))
    except Exception:
        app.state.logger.warning(
            "token_budget_check_failed", conversation_id=str(conversation_id)
        )
        return
    if spent is not None and int(spent) >= settings.conversation_token_budget:
        BUDGET_REJECTIONS.inc()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="当前会话的用量已达上限，请开启新会话",
        )


async def _record_usage(
    app: Any,
    conversation_id: UUID,
    user_id: str,
    tracker: LLMUsageTracker,
    settings: Settings,
) -> None:
    tracker.log_summary(conversation_id=str(conversation_id), user_id=user_id)
    if settings.conversation_token_budget <= 0 or tracker.total_tokens == 0:
        return
    try:
        key = _budget_key(conversation_id)
        await app.state.redis.incrby(key, tracker.total_tokens)
        await app.state.redis.expire(key, settings.conversation_budget_ttl_hours * 3600)
    except Exception:
        app.state.logger.warning(
            "token_budget_update_failed", conversation_id=str(conversation_id)
        )


def _initial_state(payload: ChatRequest, identity: Identity) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=payload.message)],
        "user_id": identity.user_id,
        # 每轮都刷新称呼：令牌是它的真相来源，改名后不必等检查点失效。
        "user_name": identity.name,
        "conversation_id": payload.conversation_id,
        "request_id": payload.request_id,
    }


async def _validate_chat_turn(
    app: Any, conversation_id: UUID, user_id: str, request_id: UUID | None = None
) -> None:
    # 同一会话同时只允许一次执行：并发写同一个 thread 会让检查点互相覆盖。
    # 重复提交同一个 request_id 属于客户端重试，交给 RunManager 幂等处理。
    active = _runs(app).active(conversation_id)
    if active is not None and active.request_id != request_id:
        raise HTTPException(status_code=409, detail="当前会话仍在执行中")
    snapshot = await app.state.graph.aget_state(_config(conversation_id, user_id))
    if not snapshot.values:
        return
    if snapshot.values.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="会话不存在")
    if _pending_confirmation(snapshot) is not None:
        raise HTTPException(status_code=409, detail="当前会话仍有待确认操作")
    # 挂在追问上的线程不能再发普通消息：那会被当成新的一次 invoke，中断点连同
    # 整个任务 DAG 一起丢掉。回答要走 /input，放弃则清空会话。
    if _pending_input(snapshot) is not None:
        raise HTTPException(status_code=409, detail="当前会话有待补充的问题")


def _pending_confirmation(snapshot: Any) -> PendingConfirmation | None:
    """从公开的 interrupt 契约读取确认信息，不依赖子图内部节点名。"""
    for item in getattr(snapshot, "interrupts", ()):
        try:
            return PendingConfirmation.model_validate(item.value)
        except (AttributeError, ValueError):
            continue
    return None


def _pending_input(snapshot: Any) -> PendingInput | None:
    """从同一个 interrupt 通道读取追问。

    两种中断共用这个通道，靠 payload 里的 kind 区分，不靠"哪个模型 validate
    得过"——那种判别会随字段增减悄悄失效。
    """
    for item in getattr(snapshot, "interrupts", ()):
        value = getattr(item, "value", None)
        if not isinstance(value, Mapping) or value.get("kind") != "input":
            continue
        try:
            return PendingInput.model_validate(value)
        except ValueError:
            continue
    return None


def _encode_sse(event: str, data: Any, event_id: int | None = None) -> str:
    """编码一条带类型的 SSE 事件；JSON 可明确表示换行符和 Unicode 字符。

    `event_id` 会写入 `id:` 字段，客户端重连时通过 `Last-Event-ID` 带回，
    服务端据此只补发缺失的增量。
    """
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    prefix = f"id: {event_id}\n" if event_id is not None else ""
    return f"{prefix}event: {event}\ndata: {payload}\n\n"


def _message_text_delta(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


async def _response(app: Any, conversation_id: UUID, user_id: str) -> AssistantResponse:
    snapshot = await app.state.graph.aget_state(_config(conversation_id, user_id))
    values = snapshot.values
    if not values or values.get("user_id") != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
    pending = _pending_confirmation(snapshot)
    awaiting = _pending_input(snapshot)
    tasks = values.get("tasks", [])
    if pending:
        workflow_status = "waiting_confirmation"
        tasks = [
            task.model_copy(update={"status": TaskStatus.WAITING_CONFIRMATION})
            if task.id == pending.task_id
            else task
            for task in tasks
        ]
    elif awaiting:
        # 子图挂在 interrupt 上，父图里这个任务还是 running；面板要显示"待补充"。
        workflow_status = "waiting_input"
        tasks = [
            task.model_copy(update={"status": TaskStatus.WAITING_INPUT})
            if task.id == awaiting.task_id
            else task
            for task in tasks
        ]
    elif any(task.status.value == "waiting_input" for task in tasks):
        workflow_status = "waiting_input"
    elif any(task.status.value == "failed" for task in tasks):
        workflow_status = "failed"
    elif any(task.status.value == "rejected" for task in tasks):
        workflow_status = "rejected"
    elif not str(values.get("last_answer", "")).strip():
        workflow_status = "failed"
    else:
        workflow_status = "completed"
    return AssistantResponse(
        conversation_id=conversation_id,
        status=workflow_status,
        answer=values.get("last_answer", ""),
        user_goal=values.get("user_goal", ""),
        tasks=tasks,
        artifacts=values.get("artifacts", {}),
        tool_results=values.get("tool_results", []),
        pending_confirmation=pending,
        pending_input=awaiting,
    )


def _run_status(response: AssistantResponse) -> RunStatus:
    """把工作流的终态映射成运行状态。

    停在人工确认或等待补充输入上不是失败，而是一个可继续的中断点。
    """
    if response.status in ("waiting_confirmation", "waiting_input"):
        return RunStatus.interrupted
    if response.status in ("failed", "rejected"):
        return RunStatus.failed
    return RunStatus.completed


async def _execute_run(
    app: Any,
    run: Run,
    publish: Publisher,
    graph_input: dict[str, Any] | Command[Any],
    conversation_id: UUID,
    user_id: str,
) -> RunStatus:
    """在后台任务里执行工作流，把进度、回答增量和终态快照写入事件通道。

    规划器的结构化输出不会作为模型词元暴露。客户端接收稳定的工作流事件，
    只有标记为 user-visible 的领域回答会按模型原生增量发送。最终 `done` 事件是
    权威数据来源。

    这里不感知任何 HTTP 连接：订阅者来去与执行无关。
    """
    settings = get_settings()
    tracker = LLMUsageTracker(settings)
    await publish(
        "metadata", {"conversation_id": str(conversation_id), "run_id": run.run_id}
    )
    try:
        active_message_ids: set[str] = set()
        async for part in app.state.graph.astream(
            graph_input,
            _config(conversation_id, user_id, tracker),
            # tasks 而不是 updates：updates 在节点**跑完**之后才 emit，用它发
            # "正在生成回复"就意味着回答早已流完才显示这句话。tasks 会在任务开始
            # 时先发一次，进度文案才对得上正在发生的事。
            stream_mode=["messages", "tasks"],
            subgraphs=True,
            version="v2",
        ):
            if part["type"] == "messages":
                chunk, metadata = part["data"]
                if "user-visible" not in metadata.get("tags", []):
                    continue
                content = _message_text_delta(chunk.content)
                if not content:
                    continue
                message_id = str(chunk.id or metadata.get("task_id") or "answer")
                if message_id not in active_message_ids:
                    active_message_ids.add(message_id)
                    await publish(
                        "answer_start",
                        {
                            "message_id": message_id,
                            "agent": metadata.get("agent"),
                            "task_id": metadata.get("task_id"),
                        },
                    )
                await publish(
                    "token",
                    {
                        "message_id": message_id,
                        "agent": metadata.get("agent"),
                        "task_id": metadata.get("task_id"),
                        "content": content,
                    },
                )
            elif part["type"] == "tasks":
                task = part["data"]
                # 同一个任务开始和结束各发一次，结束那次带 result/error。只认开始。
                if "result" in task or "error" in task:
                    continue
                node_name = str(task.get("name", ""))
                message = _NODE_PROGRESS.get(node_name)
                if message:
                    await publish("progress", {"node": node_name, "message": message})

        response = await _response(app, conversation_id, user_id)
        await publish("done", response.model_dump(mode="json"))
        return _run_status(response)
    except asyncio.CancelledError:
        app.state.logger.info(
            "run_cancelled", run_id=run.run_id, conversation_id=str(conversation_id)
        )
        raise
    except Exception:
        app.state.logger.exception(
            "graph_stream_failed", run_id=run.run_id, conversation_id=str(conversation_id)
        )
        await publish("error", {"message": "智能助手执行失败，请稍后重试"})
        return RunStatus.failed
    finally:
        await _record_usage(app, conversation_id, user_id, tracker, settings)


def _disconnect_mode(requested: str | None, settings: Settings) -> DisconnectMode:
    return DisconnectMode(requested or settings.run_on_disconnect)


async def _start_run(
    app: Any,
    graph_input: dict[str, Any] | Command[Any],
    conversation_id: UUID,
    user_id: str,
    *,
    request_id: UUID | None = None,
    on_disconnect: DisconnectMode = DisconnectMode.continue_,
) -> Run:
    async def runner(run: Run, publish: Publisher) -> RunStatus:
        return await _execute_run(app, run, publish, graph_input, conversation_id, user_id)

    try:
        return await _runs(app).start(
            conversation_id=conversation_id,
            user_id=user_id,
            runner=runner,
            request_id=request_id,
            on_disconnect=on_disconnect,
        )
    except RunConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _last_seq(request: Request) -> int | None:
    """解析断线重连游标：优先用标准的 Last-Event-ID 头，兼容查询参数。"""
    raw = request.headers.get("Last-Event-ID") or request.query_params.get("last_event_id")
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


async def _subscribe_sse(
    request: Request, run: Run, *, apply_on_disconnect: bool
) -> AsyncIterator[str]:
    """把后台运行的事件转成 SSE 帧。

    这个生成器只负责传输：客户端断开时最多按 `on_disconnect` 策略取消运行，
    默认让执行继续，用户重连后还能接着看。观察者（重连订阅）永远传
    `apply_on_disconnect=False`——只读的旁观者不该因为关掉页面就终止别人的执行。
    """
    manager = _runs(request.app)
    try:
        async for item in manager.subscribe(run, last_seq=_last_seq(request)):
            if isinstance(item, StreamGap):
                # 缓冲窗口已经滚过客户端的游标，回放会缺片段；让客户端改用
                # 会话快照重建，而不是把残缺的增量当成完整回答。
                yield _encode_sse(
                    "gap",
                    {
                        "code": "stream_replay_gap",
                        "run_id": run.run_id,
                        "requested_seq": item.requested_seq,
                        "earliest_available_seq": item.earliest_available_seq,
                        "latest_available_seq": item.latest_available_seq,
                        "recovery": "reload_conversation",
                    },
                )
                return
            if item is HEARTBEAT_SENTINEL:
                yield ": heartbeat\n\n"
                continue
            if item is END_SENTINEL:
                return
            if await request.is_disconnected():
                break
            yield _encode_sse(item.event, item.data, event_id=item.seq)
    finally:
        if (
            apply_on_disconnect
            and run.on_disconnect is DisconnectMode.cancel
            and run.status is RunStatus.running
        ):
            manager.request_cancel(run.run_id)


async def _snapshot_sse(response: AssistantResponse) -> AsyncIterator[str]:
    """没有可订阅的运行时，直接给一份终态快照并结束。"""
    yield _encode_sse("done", response.model_dump(mode="json"))


def _stream_response(request: Request, run: Run, *, apply_on_disconnect: bool) -> StreamingResponse:
    return StreamingResponse(
        _subscribe_sse(request, run, apply_on_disconnect=apply_on_disconnect),
        media_type="text/event-stream",
        headers={
            **_SSE_HEADERS,
            # 客户端据此知道重连该访问哪个运行资源。
            "Content-Location": f"/api/v1/conversations/{run.conversation_id}/stream",
        },
    )


@router.post("/chat", response_model=AssistantResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    identity: CurrentIdentity,
) -> AssistantResponse:
    settings = get_settings()
    user_id = identity.user_id
    await _enforce_token_budget(request.app, payload.conversation_id, settings)
    await _validate_chat_turn(
        request.app, payload.conversation_id, user_id, payload.request_id
    )
    run = await _start_run(
        request.app,
        _initial_state(payload, identity),
        payload.conversation_id,
        user_id,
        request_id=payload.request_id,
    )
    if run.task is not None:
        # asyncio.wait 不会把本请求的取消传导给后台任务：调用方断开时，
        # 执行仍然跑完并落检查点。
        await asyncio.wait([run.task])
    if run.status is RunStatus.failed:
        raise HTTPException(status_code=502, detail="智能助手执行失败，请稍后重试")
    return await _response(request.app, payload.conversation_id, user_id)


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    identity: CurrentIdentity,
) -> StreamingResponse:
    """聊天输入使用 POST，因此该 SSE 接口由流式 fetch 消费。"""
    settings = get_settings()
    user_id = identity.user_id
    await _enforce_token_budget(request.app, payload.conversation_id, settings)
    await _validate_chat_turn(
        request.app, payload.conversation_id, user_id, payload.request_id
    )
    run = await _start_run(
        request.app,
        _initial_state(payload, identity),
        payload.conversation_id,
        user_id,
        request_id=payload.request_id,
        on_disconnect=_disconnect_mode(payload.on_disconnect, settings),
    )
    return _stream_response(request, run, apply_on_disconnect=True)


#: 确认决定在会话历史里的留痕。用 SystemMessage 而不是 HumanMessage：它不是用户
#: 说的话，`_conversation()` 只挑 human/ai，因此这条记录进得了历史、进不了模型上下文。
#: 决定本身对下一轮的指代消解并非必需——领域回答里已经写明操作是否执行。
_DECISION_TEXTS = {True: "你确认执行了这个操作", False: "你取消了这个操作"}


def _decision_message(approved: bool) -> SystemMessage:
    return SystemMessage(
        content=_DECISION_TEXTS[approved], additional_kwargs={"kind": "decision"}
    )


def _resume_command(payload: ConfirmationRequest) -> Command[Any]:
    # update 和 resume 一起发：卡片一关，对话里必须留下这个决定，否则回头看只剩
    # 一句没头没尾的回答。追加发生在子图恢复之前，所以它排在本轮回答的前面。
    return Command(
        resume={
            "confirmation_id": str(payload.confirmation_id),
            "approved": payload.approved,
            "comment": payload.comment,
        },
        update={"messages": [_decision_message(payload.approved)]},
    )


def _resume_input_command(payload: InputRequest, awaiting: PendingInput) -> Command[Any]:
    """恢复追问，并把这一问一答补进会话历史。

    追问本身由领域子图的 respond 产出，而子图此刻挂在 interrupt 上还没返回，父图
    也就没有把它并进 messages——用户看到的那句话只活在本轮的流式输出里，刷新之后
    对话中间凭空少一条，看起来像自己无缘无故答了"培训"。所以这里把提问和回答成对
    写进去。

    回答用 HumanMessage：它就是用户说的话，下一轮的 Context Supervisor 要靠它消解
    指代。确认决定那条用的是 SystemMessage，因为那不是用户说的，两者刻意不同。
    """
    return Command(
        resume={"input_id": str(payload.input_id), "text": payload.text},
        update={
            "messages": [
                AIMessage(content=awaiting.question),
                HumanMessage(content=payload.text),
            ]
        },
    )


async def _validate_input(
    app: Any, conversation_id: UUID, user_id: str, payload: InputRequest
) -> PendingInput:
    """校验这条回答对得上当前挂着的那次提问，并把提问原文交回给调用方。"""
    snapshot = await app.state.graph.aget_state(_config(conversation_id, user_id))
    if not snapshot.values or snapshot.values.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="会话不存在")
    awaiting = _pending_input(snapshot)
    if awaiting is None:
        raise HTTPException(status_code=409, detail="当前会话没有待补充的问题")
    if awaiting.input_id != payload.input_id:
        raise HTTPException(status_code=409, detail="该问题已过期，请刷新后重试")
    return awaiting


async def _validate_confirmation(
    app: Any, conversation_id: UUID, user_id: str, payload: ConfirmationRequest
) -> None:
    snapshot = await app.state.graph.aget_state(_config(conversation_id, user_id))
    if not snapshot.values or snapshot.values.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="会话不存在")
    pending = _pending_confirmation(snapshot)
    if pending is None:
        raise HTTPException(status_code=409, detail="当前会话没有待确认操作")
    if pending.confirmation_id != payload.confirmation_id:
        raise HTTPException(status_code=409, detail="确认请求已过期，请刷新后重试")


@router.post("/conversations/{conversation_id}/confirm", response_model=AssistantResponse)
async def confirm(
    conversation_id: UUID,
    payload: ConfirmationRequest,
    request: Request,
    user_id: CurrentUser,
) -> AssistantResponse:
    await _validate_confirmation(request.app, conversation_id, user_id, payload)
    run = await _start_run(
        request.app, _resume_command(payload), conversation_id, user_id
    )
    if run.task is not None:
        await asyncio.wait([run.task])
    return await _response(request.app, conversation_id, user_id)


@router.post("/conversations/{conversation_id}/confirm/stream")
async def confirm_stream(
    conversation_id: UUID,
    payload: ConfirmationRequest,
    request: Request,
    user_id: CurrentUser,
) -> StreamingResponse:
    """恢复持久化的人工确认中断，并流式发送剩余任务。"""
    settings = get_settings()
    await _validate_confirmation(request.app, conversation_id, user_id, payload)
    run = await _start_run(
        request.app,
        _resume_command(payload),
        conversation_id,
        user_id,
        on_disconnect=_disconnect_mode(payload.on_disconnect, settings),
    )
    return _stream_response(request, run, apply_on_disconnect=True)


@router.post("/conversations/{conversation_id}/input/stream")
async def provide_input_stream(
    conversation_id: UUID,
    payload: InputRequest,
    request: Request,
    user_id: CurrentUser,
) -> StreamingResponse:
    """回答领域任务的追问，并继续流式执行剩余任务。

    走的是和人工确认同一条恢复路径：任务 DAG 留在检查点里，补完字段原地继续，
    而不是把回答当成新的一轮重新规划——那会让同一请求里尚未执行的任务被新计划
    覆盖掉。
    """
    settings = get_settings()
    awaiting = await _validate_input(request.app, conversation_id, user_id, payload)
    run = await _start_run(
        request.app,
        _resume_input_command(payload, awaiting),
        conversation_id,
        user_id,
        on_disconnect=_disconnect_mode(payload.on_disconnect, settings),
    )
    return _stream_response(request, run, apply_on_disconnect=True)


@router.get("/conversations/{conversation_id}/stream")
async def attach_stream(
    conversation_id: UUID,
    request: Request,
    user_id: CurrentUser,
) -> StreamingResponse:
    """重新订阅会话当前的执行。

    这是断线恢复入口：带上 `Last-Event-ID` 就只补发缺失的增量。没有可订阅的
    运行时（早已结束或被回收）直接返回一份终态快照。
    这里是只读旁观，断开不会影响后台执行。
    """
    run = _runs(request.app).latest(conversation_id)
    if run is not None and run.user_id != user_id:
        raise HTTPException(status_code=404, detail="会话不存在")
    if run is None:
        snapshot = await _response(request.app, conversation_id, user_id)
        return StreamingResponse(
            _snapshot_sse(snapshot), media_type="text/event-stream", headers=_SSE_HEADERS
        )
    return _stream_response(request, run, apply_on_disconnect=False)


@router.get(
    "/conversations/{conversation_id}/messages", response_model=ConversationHistoryResponse
)
async def list_conversation_messages(
    conversation_id: UUID,
    request: Request,
    user_id: CurrentUser,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    before: Annotated[int | None, Query(ge=0)] = None,
) -> ConversationHistoryResponse:
    """按页取回会话消息，默认给最近的一页。

    演示用户长期停在同一个会话里，首屏把整段历史铺出来既慢又没必要；客户端拿最近
    一页，需要时用 before 往前翻。注意分页只减少传输量：检查点仍然整体反序列化，
    真要压这部分成本得在状态层面回收历史，而不是在这个接口上。
    """
    snapshot = await request.app.state.graph.aget_state(_config(conversation_id, user_id))
    values = snapshot.values
    if not values:
        # 还没说过话的新会话不是错误，返回空页让前端直接进入对话界面。
        return ConversationHistoryResponse(messages=[], has_more=False)
    if values.get("user_id") != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")

    turns: list[ConversationMessage] = []
    for message in values.get("messages", []):
        role: Literal["user", "assistant", "decision"]
        if message.additional_kwargs.get("kind") == "decision":
            role = "decision"
        elif message.type == "human":
            role = "user"
        elif message.type == "ai":
            role = "assistant"
        else:
            continue
        text = _message_text_delta(message.content).strip()
        if not text:
            continue
        turns.append(ConversationMessage(index=len(turns), role=role, text=text))

    end = len(turns) if before is None else min(before, len(turns))
    start = max(0, end - limit)
    return ConversationHistoryResponse(messages=turns[start:end], has_more=start > 0)


@router.get("/conversations/{conversation_id}", response_model=AssistantResponse)
async def get_conversation(
    conversation_id: UUID,
    request: Request,
    user_id: CurrentUser,
) -> AssistantResponse:
    response = await _response(request.app, conversation_id, user_id)
    run = _runs(request.app).active(conversation_id)
    if run is not None and run.user_id == user_id:
        # 执行还在继续，检查点里的半成品状态不能被当成失败。
        return response.model_copy(update={"status": "running", "run_id": run.run_id})
    return response


def _demo_users(request: Request, settings: Settings) -> DemoUserRepository:
    """取演示名册；接口未开启时一律 404，不泄露它的存在。"""
    repository: DemoUserRepository | None = getattr(request.app.state, "demo_users", None)
    if repository is None or not settings.demo_login_enabled or settings.app_env != "development":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="接口不可用")
    return repository


def _demo_auth_response(user: DemoUser, settings: Settings, *, created: bool) -> DemoAuthResponse:
    token, expires_in = create_access_token(
        user.user_id, settings, display_name=user.display_name
    )
    return DemoAuthResponse(
        access_token=token,
        expires_in=expires_in,
        user_id=user.user_id,
        display_name=user.display_name,
        conversation_id=conversation_id_for(user.user_id),
        created=created,
    )


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def clear_conversation(
    conversation_id: UUID, request: Request, user_id: CurrentUser
) -> Response:
    """清空会话，让演示可以重来一遍。

    演示用户的 conversation_id 由名字派生、换不掉，一旦聊歪了就没有别的退路；
    历史也会随轮次一直增长，检查点越读越慢。删掉整个 thread 是最干净的重置。

    已经不存在的会话同样返回 204：调用方要的是"清空"这个结果，重复调用不该报错。
    """
    snapshot = await request.app.state.graph.aget_state(_config(conversation_id, user_id))
    if snapshot.values and snapshot.values.get("user_id") != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
    if _runs(request.app).active(conversation_id) is not None:
        raise HTTPException(status_code=409, detail="当前会话仍在执行中，请稍后再试")
    checkpointer = getattr(request.app.state, "checkpointer", None)
    if checkpointer is not None:
        await checkpointer.adelete_thread(str(conversation_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/auth/login", response_model=DemoAuthResponse)
async def demo_login(payload: DemoAuthRequest, request: Request) -> DemoAuthResponse:
    """用名字进入，返回令牌和这个用户固定的会话 ID。

    名字没见过就顺手建一个，不单独设注册动作：没有凭据的前提下，注册和登录本来就是
    同一件事，把名字占用做成冲突错误挡不住任何冒用，只会在演示现场平添一次点击。

    这不是身份系统——任何人输入同一个名字就能取得这个身份，由 APP_ENV 和
    DEMO_LOGIN_ENABLED 双重开关挡在生产之外。
    """
    settings = get_settings()
    repository = _demo_users(request, settings)
    display_name = normalize_name(payload.name)
    if not display_name:
        raise HTTPException(status_code=422, detail="名字不能为空")
    user, created = await repository.get_or_create(display_name, display_name)
    return _demo_auth_response(user, settings, created=created)


@router.post("/auth/dev-token", response_model=TokenResponse)
async def dev_token(payload: DevTokenRequest) -> TokenResponse:
    """签发本地联调用的访问令牌。

    仅在开发环境且显式开启 DEV_LOGIN_ENABLED 时可用；生产部署应关闭该接口，
    由企业 SSO 颁发令牌。
    """
    settings = get_settings()
    if not settings.dev_login_enabled or settings.app_env != "development":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="接口不可用")
    token, expires_in = create_access_token(payload.user_id, settings)
    return TokenResponse(access_token=token, expires_in=expires_in)


@router.get("/memories", response_model=MemoryListResponse)
async def list_memories(request: Request, user_id: CurrentUser) -> MemoryListResponse:
    """列出当前用户的长期记忆。

    user_id 一律取自访问令牌，不接受查询参数指定：记忆按人隔离，让调用方
    指定身份等于开放跨用户读取。
    """
    settings = get_settings()
    repository = getattr(request.app.state, "memories", None)
    if repository is None or not settings.memory_enabled:
        return MemoryListResponse(memories=[], recent_actions=[])
    return MemoryListResponse(
        memories=await repository.list_memories(user_id, settings.memory_recall_limit),
        recent_actions=await repository.recent_actions(
            user_id, settings.memory_recent_action_limit
        ),
    )


@router.get("/actions", response_model=ActionListResponse)
async def list_actions(
    request: Request,
    user_id: CurrentUser,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> ActionListResponse:
    """列出当前用户提交过的单据。

    和 /memories 不同，这里不看 MEMORY_ENABLED：单据来自 workflow_actions，
    是用户自己办过的事，不因为关掉长期记忆就该从界面上消失。

    user_id 同样只取自访问令牌——单据按人隔离，让调用方指定身份等于开放跨用户读取。
    """
    repository = getattr(request.app.state, "memories", None)
    if repository is None:
        return ActionListResponse(actions=[])
    return ActionListResponse(actions=await repository.recent_actions(user_id, limit))


@router.delete("/memories/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(
    memory_id: UUID, request: Request, user_id: CurrentUser
) -> Response:
    """删除一条记错的记忆。

    没有这个入口，一条错误画像会持续污染该用户之后的每一轮对话。
    """
    repository = getattr(request.app.state, "memories", None)
    if repository is None or not await repository.delete(user_id, memory_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="记忆不存在")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/metrics")
async def metrics() -> Response:
    """Prometheus 抓取端点。与健康检查一样由基础设施访问，不要求业务令牌。"""
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    services: dict[str, str] = {}
    try:
        async with request.app.state.db_pool.acquire() as connection:
            await connection.fetchval("SELECT 1")
        services["postgres"] = "up"
    except Exception:
        services["postgres"] = "down"
    try:
        services["redis"] = "up" if await request.app.state.redis.ping() else "down"
    except Exception:
        services["redis"] = "down"
    try:
        services["milvus"] = (
            "up" if request.app.state.milvus.list_collections() is not None else "down"
        )
    except Exception:
        services["milvus"] = "down"
    overall = "ok" if all(value == "up" for value in services.values()) else "degraded"
    return HealthResponse(status=overall, services=services)
