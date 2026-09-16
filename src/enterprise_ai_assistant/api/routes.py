import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Query, Request, status
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response, StreamingResponse

from enterprise_ai_assistant.api.matters import project_matters
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
    Matter,
    MemoryListResponse,
    MessageFeedbackRequest,
    TokenResponse,
    TurnStep,
)
from enterprise_ai_assistant.core.config import Settings, get_settings
from enterprise_ai_assistant.core.metrics import BUDGET_REJECTIONS, REGISTRY
from enterprise_ai_assistant.core.models import (
    DomainTaskResult,
    PendingConfirmation,
    PlannedTask,
    TaskStatus,
    ToolResult,
)
from enterprise_ai_assistant.core.observability import TRACE_ID_KEY, LLMUsageTracker
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
from enterprise_ai_assistant.graph.workflow import (
    STEPS_KEY,
    TASK_KEY,
    TRACE_KEY,
    decision_message,
    reply_message,
)
from enterprise_ai_assistant.repositories.feedback import FeedbackRepository, MessageFeedback
from enterprise_ai_assistant.repositories.users import (
    DemoUser,
    DemoUserRepository,
    conversation_id_for,
    normalize_name,
)
from enterprise_ai_assistant.services.feedback import FeedbackEvent, FeedbackSync
from enterprise_ai_assistant.tools.registry import TOOL_LABELS

router = APIRouter(prefix="/api/v1")

_NODE_PROGRESS = {
    "recall": "正在读取你的历史偏好",
    "understand": "正在结合会话上下文理解你的请求",
    "plan": "正在拆解任务并分析依赖关系",
    "select_task": "Supervisor 正在选择合适的专业 Agent",
    "initialize": "正在初始化专业 Agent",
    "decide": "专业 Agent 正在分析字段并选择工具",
    "confirm_tool": "工具调用需要人工确认",
    "execute_tool": "正在调用企业工具",
    # 只有工具失败等兜底路径才会进这个节点；常规回答在 decide 里直接写出。
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


def _ip_budget_key(client_ip: str) -> str:
    return f"budget:tokens:ip:{client_ip}"


def _client_ip(request: Request) -> str | None:
    # 只认连接的对端地址，不自己解析 X-Forwarded-For：这个头谁都能伪造，换个值就换一份额度。
    # 部署在反向代理后面时，由 uvicorn 的 --proxy-headers 和 FORWARDED_ALLOW_IPS 按受信代理改写。
    return request.client.host if request.client else None


async def _spent(app: Any, key: str) -> int | None:
    try:
        raw = await app.state.redis.get(key)
    except Exception:
        app.state.logger.warning("token_budget_check_failed", key=key)
        return None
    return None if raw is None else int(raw)


async def _enforce_token_budget(
    app: Any, conversation_id: UUID, settings: Settings, client_ip: str | None = None
) -> None:
    """会话或来源 IP 累计 token 超过预算时拒绝新一轮请求。

    只在新一轮开始前检查，不拦确认卡的恢复：停在确认卡上的事拦下来就办不完了。
    检查发生在执行之前，最后一轮会超出预算一轮的用量。
    预算计数仅用于成本护栏，Redis 不可用时放行而不是阻断业务。
    """
    if settings.conversation_token_budget > 0:
        spent = await _spent(app, _budget_key(conversation_id))
        if spent is not None and spent >= settings.conversation_token_budget:
            BUDGET_REJECTIONS.inc()
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="当前会话的用量已达上限，请开启新会话",
            )
    if settings.ip_token_budget > 0 and client_ip:
        spent = await _spent(app, _ip_budget_key(client_ip))
        if spent is not None and spent >= settings.ip_token_budget:
            BUDGET_REJECTIONS.inc()
            # 开新会话绕不过按 IP 的额度，提示里不能再让人去开新会话。对用户来说和模型服务
            # 额度用完是一回事，用同一句提示。
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=QUOTA_EXHAUSTED_MESSAGE,
            )


async def _record_usage(
    app: Any,
    conversation_id: UUID,
    user_id: str,
    tracker: LLMUsageTracker,
    settings: Settings,
    client_ip: str | None = None,
) -> None:
    tracker.log_summary(conversation_id=str(conversation_id), user_id=user_id)
    if tracker.total_tokens == 0:
        return
    if settings.conversation_token_budget > 0:
        try:
            key = _budget_key(conversation_id)
            await app.state.redis.incrby(key, tracker.total_tokens)
            await app.state.redis.expire(key, settings.conversation_budget_ttl_hours * 3600)
        except Exception:
            app.state.logger.warning(
                "token_budget_update_failed", conversation_id=str(conversation_id)
            )
    if settings.ip_token_budget > 0 and client_ip:
        try:
            key = _ip_budget_key(client_ip)
            total = await app.state.redis.incrby(key, tracker.total_tokens)
            # 固定窗口：只在这个窗口的第一笔用量时设过期。每次都续期的话，持续在用的 IP
            # 计数永远不清零，额度用完后只要隔一会儿再试一次就又续上，等于永久封禁。
            if total == tracker.total_tokens:
                await app.state.redis.expire(key, settings.ip_budget_ttl_hours * 3600)
        except Exception:
            app.state.logger.warning("token_budget_update_failed", client_ip=client_ip)


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


def _pending_interrupt(snapshot: Any) -> tuple[Any, PendingConfirmation] | None:
    """从公开的 interrupt 契约读取确认信息，不依赖子图内部节点名。

    没有依赖的任务并行执行，可能同时停在两张确认卡上。只确认其中一张时，已恢复并跑完的
    分支在同一超步结束前仍挂着原来的中断记录，所以先按任务过滤掉已有结果的，再取第一张：
    界面一次只出一张卡，确认完再出下一张。
    """
    tasks = getattr(snapshot, "tasks", None)
    if tasks:
        items = [
            item
            for task in tasks
            if getattr(task, "result", None) is None
            for item in getattr(task, "interrupts", ())
        ]
    else:
        items = list(getattr(snapshot, "interrupts", ()))
    for item in items:
        try:
            return item, PendingConfirmation.model_validate(item.value)
        except (AttributeError, ValueError):
            continue
    return None


def _pending_confirmation(snapshot: Any) -> PendingConfirmation | None:
    found = _pending_interrupt(snapshot)
    return found[1] if found else None


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


def _steps(tool_results: list[ToolResult], tasks: list[PlannedTask]) -> list[TurnStep]:
    titles = {task.id: task.title for task in tasks}
    # 用户在确认卡上取消的任务不展示步骤。步骤要等任务归并才下发，取消又没有回答，
    # 卡片之前查过的余额之类会孤零零挂在"你取消了"下面，看着像取消之后又做了什么。
    # 查到的内容卡片上方那句话已经说过了。
    rejected = {task.id for task in tasks if task.status == TaskStatus.REJECTED}
    return [
        TurnStep(
            id=f"{item.task_id}:{item.tool}:{item.created_at.isoformat()}",
            task_id=item.task_id,
            label=TOOL_LABELS.get(item.tool, titles.get(item.task_id, item.tool)),
            success=item.success,
        )
        for item in tool_results
        if item.task_id not in rejected
    ]


def _settled_results(snapshot: Any) -> list[DomainTaskResult]:
    """同一批并行任务里已经跑完、但还没归并进状态的结果。

    父图要等整批结束才归并。两个任务都停在确认卡上时，确认完第一张，它的回答已经流出去，
    执行步骤和任务状态却要等第二张确认完才进状态——界面上两个步骤会一起出现在第二次确认
    之后。响应里先把这些结果算进去，步骤就能跟着各自的确认出现。
    """
    results: list[DomainTaskResult] = []
    for task in getattr(snapshot, "tasks", None) or ():
        written = getattr(task, "result", None)
        if not isinstance(written, dict):
            continue
        for item in written.get("domain_results") or ():
            results.append(DomainTaskResult.model_validate(item))
    return results


async def _response(
    app: Any, conversation_id: UUID, user_id: str, *, running: bool | None = None
) -> AssistantResponse:
    """会话当前状态的完整快照。

    running 默认问运行管理器；执行体在图跑完、发 done 之前调用时，运行在管理器里仍标记为
    执行中，要显式传 False。
    """
    if running is None:
        running = _runs(app).active(conversation_id) is not None
    snapshot = await app.state.graph.aget_state(_config(conversation_id, user_id))
    values = snapshot.values
    if not values or values.get("user_id") != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
    pending = _pending_confirmation(snapshot)
    settled = _settled_results(snapshot)
    finished = {
        result.task_id: result.status
        for result in settled
        if result.status != TaskStatus.HANDED_OFF
    }
    tasks = [
        task.model_copy(update={"status": finished[task.id]}) if task.id in finished else task
        for task in values.get("tasks", [])
    ]
    tool_results = [
        *values.get("tool_results", []),
        *(item for result in settled for item in result.tool_results),
    ]
    if pending:
        workflow_status = "waiting_confirmation"
        tasks = [
            task.model_copy(update={"status": TaskStatus.WAITING_CONFIRMATION})
            if task.id == pending.task_id
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
        tool_results=tool_results,
        pending_confirmation=pending,
        matters=project_matters(values, running=running, tasks=tasks, pending=pending),
        steps=_steps(tool_results, tasks),
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


#: 执行失败时的通用提示。
_RUN_FAILED_MESSAGE = "智能助手执行失败，请稍后重试"
#: 模型服务额度耗尽时的提示。这不是"稍后重试"能解决的故障，演示部署里只能找作者充值。
QUOTA_EXHAUSTED_MESSAGE = "模型额度（token）已用完，请联系作者"
#: 模型服务表示额度耗尽的错误码：百炼免费额度用尽（AllocationQuota.*）、百炼欠费、
#: OpenAI 额度不足。
_QUOTA_ERROR_CODES = ("AllocationQuota.", "Arrearage", "insufficient_quota")


def _failure_message(error: BaseException) -> str:
    """把执行异常翻成用户能看懂的一句话。

    openai SDK 的状态码异常带着服务端返回的 code。LangGraph 可能在外面再包一层，
    所以顺着 __cause__ / __context__ 往下找。
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        code = getattr(current, "code", None)
        if isinstance(code, str) and code.startswith(_QUOTA_ERROR_CODES):
            return QUOTA_EXHAUSTED_MESSAGE
        current = current.__cause__ or current.__context__
    return _RUN_FAILED_MESSAGE


@dataclass
class _Answer:
    metadata: dict[str, Any]
    text: str = ""
    finished: bool = False
    # 已经推给前端的字数；此后的增量直接转发。
    sent: int = 0


class _AnswerRelay:
    """把领域 Agent 的回答转成 answer_start / token 事件。

    两件事要在这里处理：

    - 回答来自决策调用：看过工具结果之后，模型可能先写几句再调下一个工具（提交请假前先
      报出余额），这些话照常流出，由子图记进会话。工具调用的增量一出现，这段话就到此为止，
      后面的增量不再转发。文字开头先攒着，攒够 `HOLD_CHARS` 或这次模型调用结束才放行，
      排在后面的回答不会被一段还没开口的话挡住太久。
    - 没有依赖的任务并行执行，几个领域 Agent 会同时写回答。前端把增量追加到最后一个
      气泡，交错转发会把两段话搅在一起。所以同一时刻只转发一段：先开口的先流，其余的
      攒着，前一段结束再依次放出。执行仍是并行的，只有展示排队。

    每段回答按所在节点执行的 checkpoint 命名空间区分，而不是模型给的消息 id：
    标志一次调用结束的 `chunk_position="last"` 那块用的是另一个 id，对不上。
    """

    HOLD_CHARS = 20

    def __init__(self, publish: Publisher) -> None:
        self._publish = publish
        self._answers: dict[str, _Answer] = {}
        # 开口的先后顺序，也是展示顺序。
        self._queue: list[str] = []
        self._suppressed: set[str] = set()
        self._active: str | None = None
        # 同一个任务可能先后有好几段整段推来的话（确认卡前说的、追问的问题），键不能重复。
        self._wholes = 0

    async def chunk(self, chunk: Any, metadata: dict[str, Any]) -> None:
        key = str(
            metadata.get("langgraph_checkpoint_ns") or chunk.id or metadata.get("task_id")
        )
        if key in self._suppressed:
            return
        if getattr(chunk, "tool_call_chunks", None):
            self._suppressed.add(key)
            answer = self._answers.get(key)
            if answer is not None:
                answer.finished = True
                await self._advance()
            return
        answer = self._answers.get(key)
        content = _message_text_delta(chunk.content)
        if answer is None:
            if not content:
                return
            answer = self._answers[key] = _Answer(metadata)
            self._queue.append(key)
        answer.text += content
        if getattr(chunk, "chunk_position", None) == "last":
            answer.finished = True
        await self._advance()

    async def whole(self, text: str, metadata: dict[str, Any]) -> None:
        self._wholes += 1
        key = f"answer:{metadata.get('task_id') or 'task'}:{self._wholes}"
        self._answers[key] = _Answer(metadata, text=text, finished=True)
        self._queue.append(key)
        await self._advance()

    async def flush(self) -> None:
        """执行结束时放出所有还攒着的回答。"""
        for answer in self._answers.values():
            answer.finished = True
        await self._advance()

    async def _advance(self) -> None:
        while self._queue:
            key = self._queue[0]
            answer = self._answers[key]
            if self._active != key:
                if not answer.finished and len(answer.text) < self.HOLD_CHARS:
                    return
                self._active = key
                await self._send(
                    "answer_start",
                    {
                        "message_id": key,
                        "agent": answer.metadata.get("agent"),
                        "task_id": answer.metadata.get("task_id"),
                    },
                )
            if len(answer.text) > answer.sent:
                await self._send(
                    "token",
                    {
                        "message_id": key,
                        "agent": answer.metadata.get("agent"),
                        "task_id": answer.metadata.get("task_id"),
                        "content": answer.text[answer.sent :],
                    },
                )
                answer.sent = len(answer.text)
            if not answer.finished:
                return
            self._queue.pop(0)
            self._active = None

    async def _send(self, event: str, data: dict[str, Any]) -> None:
        await self._publish(event, data)


class _MattersRelay:
    """把事项投影推给前端，内容没变就不推。

    根图每个超步都会落检查点，大多数超步不改计划，原样重复推送只是噪音。
    """

    def __init__(self, publish: Publisher) -> None:
        self._publish = publish
        self._last: list[dict[str, Any]] | None = None

    async def publish(self, matters: list[Matter]) -> None:
        payload = [item.model_dump(mode="json") for item in matters]
        if payload == self._last:
            return
        self._last = payload
        await self._publish("matters", {"matters": payload})


async def _publish_task_done(
    publish: Publisher, relay: "_AnswerRelay", data: dict[str, Any]
) -> None:
    """一个任务归并完成：推送它的执行步骤，前端据此把步骤和这个任务的回答一起呈现。

    先放出还攒着的回答：任务都归并了，它的回答一定已经写完，步骤不能跑到回答前面去。
    """
    await relay.flush()
    tool_results = [ToolResult.model_validate(item) for item in data.get("tool_results") or []]
    await publish(
        "task_done",
        {
            "task_id": data["task_done"],
            "steps": [step.model_dump(mode="json") for step in _steps(tool_results, [])],
        },
    )


async def _execute_run(
    app: Any,
    run: Run,
    publish: Publisher,
    graph_input: dict[str, Any] | Command[Any],
    conversation_id: UUID,
    user_id: str,
    client_ip: str | None = None,
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
    matters = _MattersRelay(publish)
    config = _config(conversation_id, user_id, tracker)
    # 自己定 trace id 作为根 run 的 run_id，并放进 metadata 让节点读到：模型写出的话记下它，
    # 用户对这段话的反馈才能挂回 LangSmith 上的这条 trace。不指定的话 id 由 LangChain 在
    # 内部生成，图里拿不到。
    trace_id = uuid4()
    config["run_id"] = trace_id
    config["metadata"][TRACE_ID_KEY] = str(trace_id)
    try:
        relay = _AnswerRelay(publish)
        async for part in app.state.graph.astream(
            graph_input,
            config,
            # tasks 而不是 updates：updates 在节点**跑完**之后才 emit，用它发
            # "正在生成回复"就意味着回答早已流完才显示这句话。tasks 会在任务开始
            # 时先发一次，进度文案才对得上正在发生的事。
            # custom 承载不经模型的回答（领域 Agent 的追问），messages 流里没有它们。
            # checkpoints 驱动右栏：根图每落一次检查点就按同一个投影函数重算事项。
            stream_mode=["messages", "tasks", "custom", "checkpoints"],
            subgraphs=True,
            version="v2",
        ):
            if part["type"] == "messages":
                chunk, metadata = part["data"]
                if "user-visible" in metadata.get("tags", []):
                    await relay.chunk(chunk, metadata)
            elif part["type"] == "custom":
                data = part["data"]
                if isinstance(data, dict) and isinstance(data.get("answer"), str):
                    await relay.whole(data["answer"], data)
                elif isinstance(data, dict) and isinstance(data.get("task_done"), str):
                    await _publish_task_done(publish, relay, data)
            elif part["type"] == "checkpoints":
                # 子图的检查点只有领域 Agent 的私有状态，计划和事项都在根图上。
                if not part["ns"]:
                    values = part["data"]["values"]
                    await matters.publish(project_matters(values, running=True))
            elif part["type"] == "tasks":
                task = part["data"]
                # 同一个任务开始和结束各发一次，结束那次带 result/error。只认开始。
                if "result" in task or "error" in task:
                    continue
                node_name = str(task.get("name", ""))
                message = _NODE_PROGRESS.get(node_name)
                if message:
                    await publish("progress", {"node": node_name, "message": message})

        await relay.flush()
        # 图已经跑完，但运行在管理器里要等这个函数返回才结束，这里的快照不算执行中。
        response = await _response(app, conversation_id, user_id, running=False)
        await publish("done", response.model_dump(mode="json"))
        return _run_status(response)
    except asyncio.CancelledError:
        app.state.logger.info(
            "run_cancelled", run_id=run.run_id, conversation_id=str(conversation_id)
        )
        raise
    except Exception as error:
        app.state.logger.exception(
            "graph_stream_failed", run_id=run.run_id, conversation_id=str(conversation_id)
        )
        run.error_message = _failure_message(error)
        # 最后一次推给前端的事项还是"处理中"。失败后没有 done，不补一份快照的话，右栏会一直
        # 挂着一件不会再有进展的事。快照本身也可能读不出来（失败的正是数据库），那就不补。
        with contextlib.suppress(Exception):
            snapshot = await _response(app, conversation_id, user_id, running=False)
            await matters.publish(snapshot.matters)
        await publish("error", {"message": run.error_message})
        return RunStatus.failed
    finally:
        await _record_usage(app, conversation_id, user_id, tracker, settings, client_ip)


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
    client_ip: str | None = None,
) -> Run:
    async def runner(run: Run, publish: Publisher) -> RunStatus:
        return await _execute_run(
            app, run, publish, graph_input, conversation_id, user_id, client_ip
        )

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
    await _enforce_token_budget(
        request.app, payload.conversation_id, settings, _client_ip(request)
    )
    await _validate_chat_turn(
        request.app, payload.conversation_id, user_id, payload.request_id
    )
    run = await _start_run(
        request.app,
        _initial_state(payload, identity),
        payload.conversation_id,
        user_id,
        request_id=payload.request_id,
        client_ip=_client_ip(request),
    )
    if run.task is not None:
        # asyncio.wait 不会把本请求的取消传导给后台任务：调用方断开时，
        # 执行仍然跑完并落检查点。
        await asyncio.wait([run.task])
    if run.status is RunStatus.failed:
        raise HTTPException(status_code=502, detail=run.error_message or _RUN_FAILED_MESSAGE)
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
    await _enforce_token_budget(
        request.app, payload.conversation_id, settings, _client_ip(request)
    )
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
        client_ip=_client_ip(request),
    )
    return _stream_response(request, run, apply_on_disconnect=True)


def _history_message(
    index: int,
    role: Literal["user", "assistant", "decision"],
    text: str,
    extra: dict[str, Any],
    message_id: str | None = None,
) -> ConversationMessage:
    """历史里的一条消息，助手回答带上实时画出来时的任务标题和执行步骤。

    这两样实时是从本轮计划和 task_done 里拿的，刷新后计划可能已经换了，只能用写回答时
    记在消息上的那份（`reply_message`）。早先的消息没记，就不带。
    """
    # 记着 trace 的才是模型写的话，才给评价的句柄。
    rated_id = message_id if role == "assistant" and extra.get(TRACE_KEY) else None
    task = extra.get(TASK_KEY) if role == "assistant" else None
    if not isinstance(task, dict):
        return ConversationMessage(index=index, role=role, text=text, message_id=rated_id)
    task_id = str(task.get("id", ""))
    steps = [
        TurnStep(
            id=f"history:{index}:{position}",
            task_id=task_id,
            label=TOOL_LABELS.get(str(step.get("tool")), str(step.get("tool"))),
            success=bool(step.get("success")),
        )
        for position, step in enumerate(extra.get(STEPS_KEY) or [])
        if isinstance(step, dict)
    ]
    return ConversationMessage(
        index=index,
        role=role,
        text=text,
        task_id=task_id,
        title=str(task.get("title", "")) or None,
        steps=steps,
        message_id=rated_id,
    )


def _resume_command(
    payload: ConfirmationRequest,
    interrupt: Any,
    pending: PendingConfirmation,
    task: PlannedTask | None = None,
) -> Command[Any]:
    decision = {
        "confirmation_id": str(payload.confirmation_id),
        "approved": payload.approved,
        "comment": payload.comment,
    }
    # 并行分支同时中断时，LangGraph 要求按中断 id 指明恢复哪一个，给单个值会直接报错。
    interrupt_id = getattr(interrupt, "id", None)
    # update 和 resume 一起发：卡片一关，对话里必须留下这个决定，否则回头看只剩
    # 一句没头没尾的回答。追加发生在子图恢复之前，所以它排在本轮回答的前面。
    # 确认卡之前领域 Agent 说过的话（比如查到的余额）也在这时落进会话，排在决定前面。
    return Command(
        resume={interrupt_id: decision} if interrupt_id else decision,
        update={
            "messages": [
                *(
                    # 沿用生成时定下的 id 和 trace：停在卡上时这句话可能已经被评价过，
                    # 而此刻已经是另一次执行了。
                    reply_message(
                        note.text,
                        note.tools,
                        task=task,
                        trace_id=note.trace_id,
                        message_id=note.id,
                    )
                    for note in pending.notes
                ),
                decision_message(payload.approved, pending.title),
            ]
        },
    )


async def _validate_confirmation(
    app: Any, conversation_id: UUID, user_id: str, payload: ConfirmationRequest
) -> Command[Any]:
    snapshot = await app.state.graph.aget_state(_config(conversation_id, user_id))
    if not snapshot.values or snapshot.values.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="会话不存在")
    found = _pending_interrupt(snapshot)
    if found is None:
        raise HTTPException(status_code=409, detail="当前会话没有待确认操作")
    interrupt, pending = found
    if pending.confirmation_id != payload.confirmation_id:
        raise HTTPException(status_code=409, detail="确认请求已过期，请刷新后重试")
    task = next(
        (item for item in snapshot.values.get("tasks", []) if item.id == pending.task_id), None
    )
    return _resume_command(payload, interrupt, pending, task)


@router.post("/conversations/{conversation_id}/confirm", response_model=AssistantResponse)
async def confirm(
    conversation_id: UUID,
    payload: ConfirmationRequest,
    request: Request,
    user_id: CurrentUser,
) -> AssistantResponse:
    command = await _validate_confirmation(request.app, conversation_id, user_id, payload)
    run = await _start_run(
        request.app, command, conversation_id, user_id, client_ip=_client_ip(request)
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
    command = await _validate_confirmation(request.app, conversation_id, user_id, payload)
    run = await _start_run(
        request.app,
        command,
        conversation_id,
        user_id,
        on_disconnect=_disconnect_mode(payload.on_disconnect, settings),
        client_ip=_client_ip(request),
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
        turns.append(
            _history_message(len(turns), role, text, message.additional_kwargs, message.id)
        )
    # 停在确认卡上时，卡片之前说过的话还在中断里，确认后才进会话。不补上的话，
    # 刷新页面只剩一张卡，用户看不到据以决定的内容（比如余额）。
    pending = _pending_confirmation(snapshot)
    if pending:
        titles = {task.id: task.title for task in values.get("tasks", [])}
        for note in pending.notes:
            turns.append(
                ConversationMessage(
                    index=len(turns),
                    role="assistant",
                    text=note.text,
                    task_id=pending.task_id,
                    title=titles.get(pending.task_id),
                    message_id=note.id if note.trace_id else None,
                )
            )

    end = len(turns) if before is None else min(before, len(turns))
    start = max(0, end - limit)
    page = turns[start:end]
    repository: FeedbackRepository | None = getattr(request.app.state, "feedback", None)
    rated = [item.message_id for item in page if item.message_id]
    if repository is not None and rated:
        ratings = await repository.ratings(user_id, rated)
        page = [
            item.model_copy(update={"feedback": ratings.get(item.message_id)})
            if item.message_id
            else item
            for item in page
        ]
    return ConversationHistoryResponse(messages=page, has_more=start > 0)


@dataclass(frozen=True)
class _RatedMessage:
    text: str
    trace_id: str
    task_id: str | None


def _rated_message(snapshot: Any, message_id: str) -> _RatedMessage | None:
    """在会话里找到可以评价的那条助手消息，包括停在确认卡上、还没写进会话的话。"""
    for message in snapshot.values.get("messages", []):
        if message.type != "ai" or message.id != message_id:
            continue
        trace_id = message.additional_kwargs.get(TRACE_KEY)
        if not trace_id:
            return None
        task = message.additional_kwargs.get(TASK_KEY)
        return _RatedMessage(
            text=_message_text_delta(message.content).strip(),
            trace_id=str(trace_id),
            task_id=str(task.get("id")) if isinstance(task, dict) else None,
        )
    pending = _pending_confirmation(snapshot)
    if pending is None:
        return None
    for note in pending.notes:
        if note.id == message_id and note.trace_id:
            return _RatedMessage(text=note.text, trace_id=note.trace_id, task_id=pending.task_id)
    return None


@router.put(
    "/conversations/{conversation_id}/messages/{message_id}/feedback",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def rate_message(
    conversation_id: UUID,
    message_id: str,
    payload: MessageFeedbackRequest,
    request: Request,
    user_id: CurrentUser,
) -> Response:
    """对一条助手回答点赞或点踩，重复提交覆盖上一次。

    先落本地表再同步 LangSmith：界面上点没点过以本地为准，LangSmith 不可用时不影响评价。
    消息必须在当前用户自己的会话里查得到，否则任何人都能往任意 trace 上刷反馈。
    """
    repository: FeedbackRepository | None = getattr(request.app.state, "feedback", None)
    if repository is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="接口不可用")
    snapshot = await request.app.state.graph.aget_state(_config(conversation_id, user_id))
    if not snapshot.values or snapshot.values.get("user_id") != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
    found = _rated_message(snapshot, message_id)
    if found is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="这条消息不能评价")
    feedback = MessageFeedback(
        user_id=user_id,
        conversation_id=conversation_id,
        message_id=message_id,
        trace_id=found.trace_id,
        rating=payload.rating,
        reasons=list(dict.fromkeys(payload.reasons)),
        comment=payload.comment,
    )
    first = await repository.save(feedback)
    sync: FeedbackSync | None = getattr(request.app.state, "feedback_sync", None)
    if sync is not None:
        sync.submit(
            FeedbackEvent(feedback=feedback, first=first, task_id=found.task_id, text=found.text)
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


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
