from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from enterprise_ai_assistant.core.models import (
    MemoryRecord,
    PendingConfirmation,
    PlannedTask,
    RecentAction,
    ToolResult,
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    conversation_id: UUID = Field(default_factory=uuid4)
    request_id: UUID = Field(default_factory=uuid4)
    # SSE 断开时是否终止后台执行；留空则采用服务端默认策略。
    on_disconnect: Literal["cancel", "continue"] | None = None


class ConfirmationRequest(BaseModel):
    confirmation_id: UUID
    approved: bool
    comment: str | None = Field(default=None, max_length=500)
    on_disconnect: Literal["cancel", "continue"] | None = None


class AssistantResponse(BaseModel):
    conversation_id: UUID
    status: str
    answer: str
    user_goal: str
    tasks: list[PlannedTask]
    artifacts: dict[str, Any]
    tool_results: list[ToolResult]
    pending_confirmation: PendingConfirmation | None = None
    # 仍在执行时给出当前运行标识，客户端据此重新订阅事件流。
    run_id: str | None = None


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


class MemoryListResponse(BaseModel):
    """用户可见的记忆清单。

    画像可以删除；近期单据只是 workflow_actions 的投影，要作废得走业务流程，
    所以在这里是只读的。
    """

    memories: list[MemoryRecord]
    recent_actions: list[RecentAction]
