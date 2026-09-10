from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from enterprise_ai_assistant.core.models import PendingConfirmation, PlannedTask, ToolResult


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    conversation_id: UUID = Field(default_factory=uuid4)
    request_id: UUID = Field(default_factory=uuid4)


class ConfirmationRequest(BaseModel):
    confirmation_id: UUID
    approved: bool
    comment: str | None = Field(default=None, max_length=500)


class AssistantResponse(BaseModel):
    conversation_id: UUID
    status: str
    answer: str
    user_goal: str
    tasks: list[PlannedTask]
    artifacts: dict[str, Any]
    tool_results: list[ToolResult]
    pending_confirmation: PendingConfirmation | None = None


class HealthResponse(BaseModel):
    status: str
    services: dict[str, str]


class DevTokenRequest(BaseModel):
    """本地联调用的令牌申请；生产环境由企业 SSO 颁发访问令牌。"""

    user_id: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

