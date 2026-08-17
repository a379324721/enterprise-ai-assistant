from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from enterprise_ai_assistant.core.models import PendingConfirmation, PlannedTask, ToolResult


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    conversation_id: UUID = Field(default_factory=uuid4)


class ConfirmationRequest(BaseModel):
    approved: bool
    comment: str | None = Field(default=None, max_length=500)


class AssistantResponse(BaseModel):
    conversation_id: UUID
    status: str
    answer: str
    user_goal: str
    tasks: list[PlannedTask]
    slots: dict[str, Any]
    tool_results: list[ToolResult]
    pending_confirmation: PendingConfirmation | None = None


class HealthResponse(BaseModel):
    status: str
    services: dict[str, str]
