import asyncio
import json
from collections.abc import AsyncIterator
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Request, status
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from starlette.responses import StreamingResponse

from enterprise_ai_assistant.api.schemas import (
    AssistantResponse,
    ChatRequest,
    ConfirmationRequest,
    HealthResponse,
)
from enterprise_ai_assistant.core.models import PendingConfirmation, TaskStatus

router = APIRouter(prefix="/api/v1")

_NODE_PROGRESS = {
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
}


def _config(conversation_id: UUID, user_id: str) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": str(conversation_id)},
        "tags": ["enterprise-assistant", "multi-agent"],
        "metadata": {"conversation_id": str(conversation_id), "user_id": user_id},
        "recursion_limit": 100,
    }


def _initial_state(payload: ChatRequest, user_id: str) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=payload.message)],
        "user_id": user_id,
        "conversation_id": payload.conversation_id,
        "request_id": payload.request_id,
    }


async def _validate_chat_turn(
    request: Request, conversation_id: UUID, user_id: str
) -> None:
    snapshot = await request.app.state.graph.aget_state(_config(conversation_id, user_id))
    if not snapshot.values:
        return
    if snapshot.values.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="会话不存在")
    if _pending_confirmation(snapshot) is not None:
        raise HTTPException(status_code=409, detail="当前会话仍有待确认操作")


def _pending_confirmation(snapshot: Any) -> PendingConfirmation | None:
    """从公开的 interrupt 契约读取确认信息，不依赖子图内部节点名。"""
    for item in getattr(snapshot, "interrupts", ()):
        try:
            return PendingConfirmation.model_validate(item.value)
        except (AttributeError, ValueError):
            continue
    return None


def _encode_sse(event: str, data: Any) -> str:
    """编码一条带类型的 SSE 事件；JSON 可明确表示换行符和 Unicode 字符。"""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


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


async def _response(request: Request, conversation_id: UUID, user_id: str) -> AssistantResponse:
    snapshot = await request.app.state.graph.aget_state(_config(conversation_id, user_id))
    values = snapshot.values
    if not values or values.get("user_id") != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
    pending = _pending_confirmation(snapshot)
    tasks = values.get("tasks", [])
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
        tool_results=values.get("tool_results", []),
        pending_confirmation=pending,
    )


async def _stream_graph(
    request: Request,
    graph_input: dict[str, Any] | Command[Any],
    conversation_id: UUID,
    user_id: str,
) -> AsyncIterator[str]:
    """流式发送持久化工作流进度、回答增量和最终状态快照。

    规划器的结构化输出不会作为模型词元暴露。客户端接收稳定的工作流事件，
    只有标记为 user-visible 的领域回答会按模型原生增量发送。最终 `done` 事件是
    权威数据来源。
    """
    yield _encode_sse("metadata", {"conversation_id": str(conversation_id)})
    try:
        active_message_ids: set[str] = set()
        async for part in request.app.state.graph.astream(
            graph_input,
            _config(conversation_id, user_id),
            stream_mode=["messages", "updates"],
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
                    yield _encode_sse(
                        "answer_start",
                        {
                            "message_id": message_id,
                            "agent": metadata.get("agent"),
                            "task_id": metadata.get("task_id"),
                        },
                    )
                if await request.is_disconnected():
                    return
                yield _encode_sse(
                    "token",
                    {
                        "message_id": message_id,
                        "agent": metadata.get("agent"),
                        "task_id": metadata.get("task_id"),
                        "content": content,
                    },
                )
            elif part["type"] == "updates":
                for node_name in part["data"]:
                    message = _NODE_PROGRESS.get(node_name)
                    if message:
                        yield _encode_sse(
                            "progress", {"node": node_name, "message": message}
                        )

        response = await _response(request, conversation_id, user_id)
        yield _encode_sse("done", response.model_dump(mode="json"))
    except asyncio.CancelledError:
        request.app.state.logger.info(
            "sse_client_disconnected", conversation_id=str(conversation_id)
        )
        raise
    except Exception:
        request.app.state.logger.exception(
            "graph_stream_failed", conversation_id=str(conversation_id)
        )
        yield _encode_sse("error", {"message": "智能助手执行失败，请稍后重试"})


@router.post("/chat", response_model=AssistantResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    user_id: Annotated[str, Header(alias="X-User-ID", min_length=1, max_length=128)],
) -> AssistantResponse:
    await _validate_chat_turn(request, payload.conversation_id, user_id)
    initial = _initial_state(payload, user_id)
    try:
        await request.app.state.graph.ainvoke(initial, _config(payload.conversation_id, user_id))
    except Exception as exc:
        request.app.state.logger.exception(
            "graph_invocation_failed", conversation_id=str(payload.conversation_id)
        )
        raise HTTPException(status_code=502, detail="智能助手执行失败，请稍后重试") from exc
    return await _response(request, payload.conversation_id, user_id)


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    user_id: Annotated[str, Header(alias="X-User-ID", min_length=1, max_length=128)],
) -> StreamingResponse:
    """聊天输入使用 POST，因此该 SSE 接口由流式 fetch 消费。"""
    await _validate_chat_turn(request, payload.conversation_id, user_id)
    return StreamingResponse(
        _stream_graph(
            request,
            _initial_state(payload, user_id),
            payload.conversation_id,
            user_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/conversations/{conversation_id}/confirm", response_model=AssistantResponse)
async def confirm(
    conversation_id: UUID,
    payload: ConfirmationRequest,
    request: Request,
    user_id: Annotated[str, Header(alias="X-User-ID", min_length=1, max_length=128)],
) -> AssistantResponse:
    snapshot = await request.app.state.graph.aget_state(_config(conversation_id, user_id))
    if not snapshot.values or snapshot.values.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="会话不存在")
    pending = _pending_confirmation(snapshot)
    if pending is None:
        raise HTTPException(status_code=409, detail="当前会话没有待确认操作")
    if pending.confirmation_id != payload.confirmation_id:
        raise HTTPException(status_code=409, detail="确认请求已过期，请刷新后重试")
    await request.app.state.graph.ainvoke(
        Command(
            resume={
                "confirmation_id": str(payload.confirmation_id),
                "approved": payload.approved,
                "comment": payload.comment,
            }
        ),
        _config(conversation_id, user_id),
    )
    return await _response(request, conversation_id, user_id)


@router.post("/conversations/{conversation_id}/confirm/stream")
async def confirm_stream(
    conversation_id: UUID,
    payload: ConfirmationRequest,
    request: Request,
    user_id: Annotated[str, Header(alias="X-User-ID", min_length=1, max_length=128)],
) -> StreamingResponse:
    """恢复持久化的人工确认中断，并流式发送剩余任务。"""
    snapshot = await request.app.state.graph.aget_state(_config(conversation_id, user_id))
    if not snapshot.values or snapshot.values.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="会话不存在")
    pending = _pending_confirmation(snapshot)
    if pending is None:
        raise HTTPException(status_code=409, detail="当前会话没有待确认操作")
    if pending.confirmation_id != payload.confirmation_id:
        raise HTTPException(status_code=409, detail="确认请求已过期，请刷新后重试")
    command: Command[str] = Command(
        resume={
            "confirmation_id": str(payload.confirmation_id),
            "approved": payload.approved,
            "comment": payload.comment,
        }
    )
    return StreamingResponse(
        _stream_graph(request, command, conversation_id, user_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/conversations/{conversation_id}", response_model=AssistantResponse)
async def get_conversation(
    conversation_id: UUID,
    request: Request,
    user_id: Annotated[str, Header(alias="X-User-ID", min_length=1, max_length=128)],
) -> AssistantResponse:
    return await _response(request, conversation_id, user_id)


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
