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

router = APIRouter(prefix="/api/v1")

_NODE_PROGRESS = {
    "understand": "正在理解你的目标并提取关键信息",
    "plan": "正在拆解任务并分析依赖关系",
    "select_task": "Supervisor 正在选择合适的专业 Agent",
    "travel": "Travel Agent 正在处理差旅任务",
    "expense": "Expense Agent 正在处理报销任务",
    "hr": "HR Agent 正在处理人事任务",
    "policy": "Policy Agent 正在查询企业制度",
    "confirm": "操作需要人工确认",
    "execute_travel": "正在提交差旅申请",
    "execute_expense": "正在提交报销申请",
    "execute_hr": "正在提交请假申请",
    "complete_task": "正在汇总任务执行结果",
}


def _config(conversation_id: UUID, user_id: str) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": str(conversation_id)},
        "tags": ["enterprise-assistant", "multi-agent"],
        "metadata": {"conversation_id": str(conversation_id), "user_id": user_id},
        "recursion_limit": 50,
    }


def _initial_state(payload: ChatRequest, user_id: str) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=payload.message)],
        "user_id": user_id,
        "user_goal": "",
        "tasks": [],
        "slots": {},
        "tool_results": [],
        "current_agent": None,
        "active_task_id": None,
        "pending_confirmation": None,
        "last_answer": "",
    }


def _encode_sse(event: str, data: Any) -> str:
    """编码一条带类型的 SSE 事件；JSON 可明确表示换行符和 Unicode 字符。"""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


async def _response(request: Request, conversation_id: UUID, user_id: str) -> AssistantResponse:
    snapshot = await request.app.state.graph.aget_state(_config(conversation_id, user_id))
    values = snapshot.values
    if not values or values.get("user_id") != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
    pending = values.get("pending_confirmation")
    workflow_status = (
        "waiting_confirmation" if pending and "confirm" in snapshot.next else "completed"
    )
    return AssistantResponse(
        conversation_id=conversation_id,
        status=workflow_status,
        answer=values.get("last_answer", ""),
        user_goal=values.get("user_goal", ""),
        tasks=values.get("tasks", []),
        slots=values.get("slots", {}),
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
    只有面向用户的回答会以增量形式发送。最终的 `done` 事件仍是任务和槽位的
    权威数据来源。
    """
    yield _encode_sse("metadata", {"conversation_id": str(conversation_id)})
    try:
        async for update in request.app.state.graph.astream(
            graph_input,
            _config(conversation_id, user_id),
            stream_mode="updates",
        ):
            for node_name in update:
                message = _NODE_PROGRESS.get(node_name)
                if message:
                    yield _encode_sse("progress", {"node": node_name, "message": message})

        response = await _response(request, conversation_id, user_id)
        yield _encode_sse("answer_start", {})
        for character in response.answer:
            if await request.is_disconnected():
                return
            yield _encode_sse("token", {"content": character})
            # 短暂延迟可让较短且确定的工具回答呈现出可见的流式效果。
            # 后续可在此处透传 LLM 回答节点原生的词元输出时序。
            await asyncio.sleep(0.012)
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
    if "confirm" not in snapshot.next:
        raise HTTPException(status_code=409, detail="当前会话没有待确认操作")
    await request.app.state.graph.ainvoke(
        Command(resume={"approved": payload.approved, "comment": payload.comment}),
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
    if "confirm" not in snapshot.next:
        raise HTTPException(status_code=409, detail="当前会话没有待确认操作")
    command: Command[str] = Command(
        resume={"approved": payload.approved, "comment": payload.comment}
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
