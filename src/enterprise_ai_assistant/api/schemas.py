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


class ActionListResponse(BaseModel):
    """当前用户提交过的单据清单。

    数据来自 workflow_actions，与长期记忆开关无关：单据是业务事实，不是画像。
    这张表只知道适配器被调用过，不知道外部系统的审批结果，所以这里没有状态字段——
    界面上凭它显示"审批中"或"已通过"都是幻觉。
    """

    actions: list[RecentAction]


class DemoAuthRequest(BaseModel):
    """演示登录只要一个名字；没有凭据，因此接口受环境开关保护。"""

    name: str = Field(min_length=1, max_length=64)


class DemoAuthResponse(TokenResponse):
    user_id: str
    display_name: str
    # 演示用户固定一个会话，由服务端从 user_id 确定性派生后下发，
    # 客户端换设备或清了本地存储也能回到同一个会话。
    conversation_id: UUID
    # true 表示本次调用完成了注册，前端据此区分"注册成功"和"欢迎回来"。
    created: bool


class ConversationMessage(BaseModel):
    """会话中的一条可见消息。

    index 是过滤掉工具消息后的序号，也是向前翻页的游标；领域子图的内部消息不外泄，
    所以这里只会有用户、助手，以及用户在确认卡片上做的选择（decision）三种角色。
    """

    index: int
    role: Literal["user", "assistant", "decision"]
    text: str


class ConversationHistoryResponse(BaseModel):
    messages: list[ConversationMessage]
    # 还有更早的消息可以继续向前翻。
    has_more: bool
