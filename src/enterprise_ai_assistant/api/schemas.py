from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from enterprise_ai_assistant.core.models import (
    AgentName,
    DraftField,
    MemoryRecord,
    PendingConfirmation,
    PlannedTask,
    RecentAction,
    TaskStatus,
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


class MatterTask(BaseModel):
    id: str
    title: str
    domain: AgentName
    status: TaskStatus


#: in_progress 只在运行执行期间出现：当前计划没有卡住的任务，但还有任务排队或在跑。
#: 规则见 api/matters.py 的 project_matters。
MatterStatus = Literal["waiting_input", "waiting_confirmation", "in_progress", "shelved"]


class Matter(BaseModel):
    """右栏的一张事项卡：一件还没办完的事。

    卡在待补充、待确认上的计划，以及执行中途还有任务排队或在跑的当前计划，才会成为事项。
    办完的计划不出现在这里——提交过的单据已经在"我的单据"里，纯查询也没有需要跟进的状态。
    """

    plan_id: str
    status: MatterStatus
    # 卡住或正在处理的那个任务；卡片标题和字段都来自它，同一计划里的其他任务作为子项列出。
    task_id: str
    title: str
    known_fields: list[DraftField] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    tasks: list[MatterTask] = Field(default_factory=list)


class TurnStep(BaseModel):
    """本轮执行过的一次工具调用，显示在对话流里。"""

    # 前端用它去重：确认前后的两次 done 都会带上同一轮更早的步骤。
    id: str
    task_id: str
    label: str
    success: bool


class AssistantResponse(BaseModel):
    conversation_id: UUID
    status: str
    answer: str
    user_goal: str
    tasks: list[PlannedTask]
    artifacts: dict[str, Any]
    tool_results: list[ToolResult]
    pending_confirmation: PendingConfirmation | None = None
    matters: list[Matter] = Field(default_factory=list)
    steps: list[TurnStep] = Field(default_factory=list)
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

    画像可以删除；近期单据只是 workflow_actions 的投影，要撤销得在对话里走业务流程
    （撤销工具要人工确认），所以在这里是只读的。
    """

    memories: list[MemoryRecord]
    recent_actions: list[RecentAction]


class ActionListResponse(BaseModel):
    """当前用户提交过的单据清单。

    数据来自 workflow_actions，与长期记忆开关无关：单据是业务事实，不是画像。
    这张表只知道适配器被调用过、以及本系统里有没有撤销过，不知道外部系统的审批结果，
    所以只有撤销标记而没有审批状态——界面上凭它显示"审批中"或"已通过"都是幻觉。
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
    # 领域任务的回答才有：所属任务、写回答时的任务标题、回答前展示过的执行步骤。
    # 实时画的时候有这些，刷新后要照原样画回来。
    task_id: str | None = None
    title: str | None = None
    steps: list[TurnStep] = Field(default_factory=list)


class ConversationHistoryResponse(BaseModel):
    messages: list[ConversationMessage]
    # 还有更早的消息可以继续向前翻。
    has_more: bool
