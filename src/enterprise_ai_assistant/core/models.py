from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_CONFIRMATION = "waiting_confirmation"
    WAITING_INPUT = "waiting_input"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"


class AgentName(StrEnum):
    SUPERVISOR = "supervisor"
    TRAVEL = "travel"
    EXPENSE = "expense"
    HR = "hr"
    MEETING = "meeting"
    POLICY = "policy"


class PlannedTask(BaseModel):
    """Planner 只描述领域目标和依赖，不决定字段、工具或风险。"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(min_length=1, max_length=200)
    domain: AgentName
    objective: str = Field(min_length=1, max_length=2000)
    depends_on: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list, max_length=20)
    status: TaskStatus = TaskStatus.PENDING

    @model_validator(mode="after")
    def reject_supervisor_domain(self) -> "PlannedTask":
        if self.domain == AgentName.SUPERVISOR:
            raise ValueError("supervisor cannot execute a domain task")
        return self


class TaskPlan(BaseModel):
    user_goal: str = Field(min_length=1, max_length=8000)
    tasks: list[PlannedTask] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_dependencies(self) -> "TaskPlan":
        ids = {task.id for task in self.tasks}
        if len(ids) != len(self.tasks):
            raise ValueError("task ids must be unique")
        for task in self.tasks:
            if task.id in task.depends_on or not set(task.depends_on) <= ids:
                raise ValueError(f"invalid dependencies for task {task.id}")
        dependencies = {task.id: set(task.depends_on) for task in self.tasks}
        ready = [task_id for task_id, required in dependencies.items() if not required]
        visited: set[str] = set()
        while ready:
            completed = ready.pop()
            if completed in visited:
                continue
            visited.add(completed)
            for task_id, required in dependencies.items():
                if task_id not in visited and required <= visited:
                    ready.append(task_id)
        if visited != ids:
            raise ValueError("task dependencies must form an acyclic graph")
        return self


class TurnRelation(StrEnum):
    """本轮输入与未办完事项之间的关系。"""

    NEW = "new"
    CONTINUE = "continue"
    CANCEL = "cancel"


class OpenTask(BaseModel):
    """交给 Context Supervisor 的待补充任务摘要。

    只有标题和缺失字段的名称，没有任何字段值：Supervisor 需要知道"刚才在问什么"
    才能认出"当天往返""1"这种短回复在补充谁，但拿到值就有了补写领域字段的材料。
    """

    plan_id: str
    task_id: str
    title: str
    domain: AgentName
    missing_fields: list[str] = Field(default_factory=list)
    # false 是当前事项，true 是用户换话题时被搁置的事项。
    shelved: bool = False


class DraftField(BaseModel):
    """领域 Agent 在追问时报告的一个已知字段。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=64)
    value: str = Field(min_length=1, max_length=500)
    # memory 只是建议值，还没被用户确认；dependency 来自前置任务的产物。
    source: Literal["user", "memory", "dependency"] = "user"


class TaskDraft(BaseModel):
    """任务停在待补充时的字段状态。

    下一轮续跑时交还给领域 Agent。领域子图每轮从头构造 domain_messages，
    没有这份草稿就只能指望 Supervisor 把上一轮交代过的字段全部复述进改写后的请求。
    """

    known_fields: list[DraftField] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)


class ShelvedPlan(BaseModel):
    """用户换话题时被搁置的未办完计划，原样保存，恢复时整体换回当前计划。

    不自动过期：半截的事是用户自己的工作，只有用户说"继续"或"不办了"才会离开这里。
    """

    plan_id: str
    user_goal: str
    tasks: list[PlannedTask]
    artifacts: dict[str, Any] = Field(default_factory=dict)
    drafts: dict[str, TaskDraft] = Field(default_factory=dict)


class ContextResolution(BaseModel):
    """Supervisor 对完整会话的解析结果，不包含任何领域业务字段。

    这是理解阶段的唯一出口：执行链路上的节点都只读这里，不再回头看用户原话。
    不需要执行任何任务的轮次，对用户的回复也在这里一并写出——Supervisor 是唯一
    读完整会话的节点，由它说话才知道自己之前说过什么。
    """

    standalone_request: str = Field(min_length=1, max_length=8000)
    intent_summary: str = Field(min_length=1, max_length=1000)
    requires_task_planning: bool
    explicit_constraints: list[str] = Field(default_factory=list)
    referenced_task_ids: list[str] = Field(default_factory=list)
    unresolved_references: list[str] = Field(default_factory=list)
    # 用户本轮使用的语言。下游节点不读原始消息，只能靠这里保持语言一致。
    user_language: str = Field(default="简体中文", min_length=1, max_length=32)
    # 与本次请求相关的记忆 key。Supervisor 只做相关性筛选，不读取也不改写 value，
    # 领域字段的判断仍然只发生在领域子图里。
    relevant_memory_keys: list[str] = Field(default_factory=list, max_length=20)
    # continue 表示本轮在补充上一轮停在待补充的任务：跳过 Planner，原任务续跑。
    # 没有待补充任务时，运行时会忽略这里的 continue。
    turn_relation: TurnRelation = TurnRelation.NEW
    # continue / cancel 指向的事项。补充当前事项时可以留空；恢复或取消被搁置的事项时必填。
    target_plan_id: str | None = None
    # 需要规划时本次请求涉及的业务领域。只有一个领域时跳过 Planner：单领域请求只会
    # 被拆成一个任务，那次模型调用的产出事先就知道。这是意图分类，不是字段抽取。
    domains: list[AgentName] = Field(default_factory=list, max_length=5)
    # 直接对用户的回复。只在本轮不执行任务、也不是取消事项时使用；其余情况运行时忽略它：
    # 执行任务时由领域 Agent 说话，取消事项时运行时按真实处理结果说话。
    reply: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def validate_reply(self) -> "ContextResolution":
        # 这种轮次没有任何别的节点会开口，漏写回复用户就只能收到一个空气泡。
        # 抛错交给结构化输出的重试，而不是在运行时拿兜底文案糊过去。
        if (
            not self.requires_task_planning
            and self.turn_relation != TurnRelation.CANCEL
            and not self.reply.strip()
        ):
            raise ValueError("reply is required when no task runs and nothing is cancelled")
        return self


class MemoryKind(StrEnum):
    """长期记忆只保留稳定属性和偏好；易变的业务事实从 workflow_actions 派生。"""

    PROFILE = "profile"
    PREFERENCE = "preference"


class MemoryCandidate(BaseModel):
    """抽取阶段产出的候选记忆；key 是同类事实的稳定标识，用于覆盖而不是堆积。"""

    kind: MemoryKind
    key: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    value: str = Field(min_length=1, max_length=200)


class MemoryExtraction(BaseModel):
    """一轮会话的记忆抽取结果；没有值得长期保留的信息时返回空列表。"""

    memories: list[MemoryCandidate] = Field(default_factory=list, max_length=10)


class MemoryRecord(BaseModel):
    """已持久化的一条用户记忆。"""

    id: UUID
    kind: MemoryKind
    key: str
    value: str
    source_conversation_id: UUID | None = None
    updated_at: datetime

    def render(self) -> str:
        return f"{self.key}={self.value}"


class RecentAction(BaseModel):
    """从 workflow_actions 派生的近期业务事实；单号的真相来源始终是那张表。"""

    reference_id: str
    action_type: str
    summary: str
    created_at: datetime
    # 白名单字段的结构化形式，供界面自己排版；模型侧只用 summary 那一行。
    fields: dict[str, str] = Field(default_factory=dict)
    # 撤销的单据仍然列出来并标明：直接隐藏的话，用户问"我那张请假呢"时模型只会说没有提交过。
    revoked_at: datetime | None = None

    def render(self) -> str:
        day = self.created_at.date().isoformat()
        # 老数据的 result 里可能没有单号。这时只能整段省略——幂等键不是备选项。
        label = f"{self.action_type} {self.reference_id}" if self.reference_id else self.action_type
        revoked = "（已撤销）" if self.revoked_at else ""
        return f"{label}（{day}）{revoked}：{self.summary}"


class ConfirmationField(BaseModel):
    """确认卡片上的一行。标签取自工具入参契约，值已按契约里的取值标签翻译。"""

    name: str
    label: str
    value: str


class PendingConfirmation(BaseModel):
    confirmation_id: UUID = Field(default_factory=uuid4)
    task_id: str
    action: str
    tool_call_id: str
    # 操作的中文名（TOOL_LABELS）和逐项字段。用户要确认的是"做什么、用什么值"，
    # 工具名和原始 JSON 对用户没有意义。
    title: str
    fields: list[ConfirmationField] = Field(default_factory=list)
    payload: dict[str, Any]
    requested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ToolResult(BaseModel):
    task_id: str
    tool: str
    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DomainTaskRequest(BaseModel):
    """父图交给领域子图的稳定输入契约。"""

    user_id: str = Field(min_length=1, max_length=128)
    # 展示名仅用于回答里的称呼；身份判断一律用 user_id。
    user_name: str = Field(default="", max_length=128)
    conversation_id: UUID
    request_id: UUID
    user_goal: str = Field(min_length=1, max_length=8000)
    task: PlannedTask
    dependency_results: dict[str, Any] = Field(default_factory=dict)
    # 已由理解阶段按相关性筛选并渲染的档案行；只作为字段默认值的建议，
    # 不构成用户已确认的事实。子图不接触未筛选的全量记忆。
    memories: list[str] = Field(default_factory=list, max_length=20)
    recent_actions: list[RecentAction] = Field(default_factory=list)
    # 任务上一轮停在待补充时留下的字段状态；首次执行为 None。
    draft: TaskDraft | None = None
    # 助手此前对用户说过的话，按时间顺序，含本轮排在前面的任务的回答。领域 Agent 据此
    # 保持口径一致：不重复已经问过的问题、不否定别的任务刚说过的话、不重复称呼。
    # 这里只有助手的话。用户原话仍然只经 Supervisor 改写后以 user_goal 进入。
    assistant_replies: list[str] = Field(default_factory=list)


class DomainTaskResult(BaseModel):
    """领域子图返回给父图的稳定输出契约。"""

    task_id: str
    status: TaskStatus
    answer: str = Field(min_length=1)
    artifact: dict[str, Any] | None = None
    tool_results: list[ToolResult] = Field(default_factory=list)
    # 只在 WAITING_INPUT 时有值，由父图按 task_id 保存，续跑时放回 DomainTaskRequest。
    draft: TaskDraft | None = None


class TravelApplication(BaseModel):
    application_id: UUID = Field(default_factory=uuid4)
    user_id: str
    destination: str
    start_date: date
    end_date: date
    purpose: str
    status: str = "submitted"
