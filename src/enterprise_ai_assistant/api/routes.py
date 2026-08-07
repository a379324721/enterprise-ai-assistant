from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Request, status
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from enterprise_ai_assistant.api.schemas import (
    AssistantResponse,
    ChatRequest,
    ConfirmationRequest,
    HealthResponse,
)

router = APIRouter(prefix="/api/v1")


def _config(conversation_id: UUID, user_id: str) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": str(conversation_id)},
        "tags": ["enterprise-assistant", "multi-agent"],
        "metadata": {"conversation_id": str(conversation_id), "user_id": user_id},
        "recursion_limit": 50,
    }


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


@router.post("/chat", response_model=AssistantResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    user_id: Annotated[str, Header(alias="X-User-ID", min_length=1, max_length=128)],
) -> AssistantResponse:
    initial: dict[str, Any] = {
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
    try:
        await request.app.state.graph.ainvoke(initial, _config(payload.conversation_id, user_id))
    except Exception as exc:
        request.app.state.logger.exception(
            "graph_invocation_failed", conversation_id=str(payload.conversation_id)
        )
        raise HTTPException(status_code=502, detail="智能助手执行失败，请稍后重试") from exc
    return await _response(request, payload.conversation_id, user_id)


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
