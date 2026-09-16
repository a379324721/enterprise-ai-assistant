from collections.abc import Mapping
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
    # 只出现在 DomainTaskResult 上：父图收到后要么改派（任务回到 PENDING），要么判失败，
    # 不会以这个状态存进计划。
    HANDED_OFF = "handed_off"


class AgentName(StrEnum):
    SUPERVISOR = "supervisor"
    TRAVEL = "travel"
    EXPENSE = "expense"
    HR = "hr"
    MEETING = "meeting"
    POLICY = "policy"


class PlannedTask(BaseModel):
    """任务只描述领域目标和依赖，不决定字段、工具或风险。"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(min_length=1, max_length=200)
    domain: AgentName
    objective: str = Field(min_length=1, max_length=2000)
    depends_on: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list, max_length=20)
    status: TaskStatus = TaskStatus.PENDING
    # 这个任务被领域 Agent 转交过的领域，按先后顺序。改派不回到这里的领域，
    # 次数也有上限，否则两个 Agent 可以把同一个任务来回踢。
    handed_off_from: list[AgentName] = Field(default_factory=list)

    @model_validator(mode="after")
    def reject_supervisor_domain(self) -> "PlannedTask":
        if self.domain == AgentName.SUPERVISOR:
            raise ValueError("domain 不能是 supervisor，只能是 travel、expense、hr、meeting、policy")
        return self


class TaskPlan(BaseModel):
    user_goal: str = Field(min_length=1, max_length=8000)
    tasks: list[PlannedTask] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_dependencies(self) -> "TaskPlan":
        ids = {task.id for task in self.tasks}
        if len(ids) != len(self.tasks):
            raise ValueError("tasks 里的 id 不能重复")
        for task in self.tasks:
            if task.id in task.depends_on or not set(task.depends_on) <= ids:
                raise ValueError(
                    f"任务 {task.id} 的 depends_on 只能写 tasks 里其他任务的 id，"
                    "不能写自己，也不能写不存在的 id"
                )
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
            raise ValueError("任务之间的 depends_on 形成了环，前置任务必须能排出先后顺序")
        return self


class TaskOutline(BaseModel):
    """Context Supervisor 在理解结果里直接给出的任务。

    和 PlannedTask 分开定义：状态、成功标准由运行时维护，不该出现在模型要填的结构里。
    """

    title: str = Field(min_length=1, max_length=200)
    domain: AgentName
    objective: str = Field(min_length=1, max_length=2000)
    # 用前置任务的领域指代它，不用 id 或序号。实测不开思考时，字符串 id 数组会被写成
    # ['id": ']，加 schema 说明也过半失败；整数序号则 0 起和 1 起混用，歧义消不掉。
    # 拆分规则本来就是一个领域一个目标，领域足以唯一指代任务，而枚举值模型写得稳。
    # 只能依赖排在前面的任务，环无从出现。任务 id 由运行时按位置生成为 task-N。
    depends_on: list[AgentName] = Field(
        default_factory=list,
        description="依赖的前置任务的领域，只能是排在本任务前面的任务的 domain",
    )


def plan_from_outlines(user_goal: str, outlines: list[TaskOutline]) -> TaskPlan:
    """把 Supervisor 给出的任务换成运行时的计划；依赖非法时抛 ValueError。"""
    tasks: list[PlannedTask] = []
    earlier: dict[AgentName, str] = {}
    for position, outline in enumerate(outlines, start=1):
        missing = [domain for domain in outline.depends_on if domain not in earlier]
        if missing:
            # 这句话会原样反馈给模型让它修正输出，所以写成它能照着改的说明，而不是给人看的断言。
            raise ValueError(
                f"tasks 第 {position} 个任务（{outline.domain.value}）的 depends_on 写了 "
                f"{[item.value for item in missing]}，但本次 tasks 里排在它前面的任务没有这些领域。"
                "depends_on 只能写本次 tasks 里排在它前面的任务的 domain；"
                "前置事项已经办完、不在本次 tasks 里时，depends_on 写空数组"
            )
        task_id = f"task-{position}"
        tasks.append(
            PlannedTask(
                id=task_id,
                title=outline.title,
                domain=outline.domain,
                objective=outline.objective,
                depends_on=[earlier[domain] for domain in dict.fromkeys(outline.depends_on)],
            )
        )
        earlier.setdefault(outline.domain, task_id)
    return TaskPlan(user_goal=user_goal, tasks=tasks)


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


def recover_interrupted(tasks: list[PlannedTask], drafts: Mapping[str, Any]) -> list[PlannedTask]:
    """把没有运行在跑、却停在 RUNNING 的任务解读成它真实所处的状态。

    同一会话同时只有一次执行，没有运行时不可能有任务真的在跑。停在 RUNNING 说明那一轮
    在领域子图里抛了异常（例如模型服务 403）、服务重启或运行被取消，任务状态没来得及写回。
    有草稿说明它本来停在待补充上，放回 WAITING_INPUT；没有草稿说明第一次执行就失败了，
    放回 PENDING。两处必须按同一条规则解读：下一轮 understand 据此续跑，右栏据此展示，
    规则不一致时右栏会显示一件实际上不会被续跑的事。
    """
    return [
        task.model_copy(
            update={
                "status": TaskStatus.WAITING_INPUT if task.id in drafts else TaskStatus.PENDING
            }
        )
        if task.status == TaskStatus.RUNNING
        else task
        for task in tasks
    ]


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
    # continue 表示本轮在补充上一轮停在待补充的任务：不重新规划，原任务续跑。
    # 没有待补充任务时，运行时会忽略这里的 continue。
    turn_relation: TurnRelation = TurnRelation.NEW
    # continue / cancel 指向的事项。补充当前事项时可以留空；恢复或取消被搁置的事项时必填。
    target_plan_id: str | None = None
    # 新的业务请求拆出的任务 DAG。原先由单独的 Planner 调用产出，但它的输入只有这份理解
    # 结果，领域归类也已经在这里做完，多一次调用只多出依赖关系这点信息。并进同一次输出后，
    # 每个需要执行的轮次省一次模型调用。只描述领域、目标和依赖，不是字段抽取。
    # continue 轮次也要求给出：能续跑时运行时忽略它，误报 continue、指认不到事项时照它执行。
    # 仍然留空时运行时退回 Planner。
    tasks: list[TaskOutline] = Field(default_factory=list, max_length=20)
    # 直接对用户的回复。只在本轮不执行任务、也不是取消事项时使用；其余情况运行时忽略它：
    # 执行任务时由领域 Agent 说话，取消事项时运行时按真实处理结果说话。
    reply: str = Field(default="", max_length=4000)

    @property
    def domains(self) -> list[AgentName]:
        return [task.domain for task in self.tasks]

    def plan(self) -> TaskPlan | None:
        if not self.tasks:
            return None
        return plan_from_outlines(self.standalone_request, self.tasks)

    @model_validator(mode="after")
    def validate_tasks(self) -> "ContextResolution":
        # 依赖非法在这里抛错，错误说明会反馈给模型修正（services/planning.py 的
        # _StructuredStage）；放到规划节点才发现的话，模型已经没有机会改，整轮只能失败。
        self.plan()
        return self

    @model_validator(mode="after")
    def validate_reply(self) -> "ContextResolution":
        # 这种轮次没有任何别的节点会开口，漏写回复用户就只能收到一个空气泡。
        # 抛错交给结构化输出的重试，而不是在运行时拿兜底文案糊过去。
        if (
            not self.requires_task_planning
            and self.turn_relation != TurnRelation.CANCEL
            and not self.reply.strip()
        ):
            raise ValueError(
                "requires_task_planning 为 false 且 turn_relation 不是 cancel 时必须写 reply："
                "这一轮没有别的环节会回复用户"
            )
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


class AgentNote(BaseModel):
    """领域 Agent 调工具时顺带对用户说的话，例如提交请假前先报出查到的余额。

    它在工具执行前就已经流给用户，所以要作为独立的助手消息进会话；tools 是说这句话
    之前本任务调用过的工具，渲染成来源标注。
    """

    text: str
    tools: list[str] = Field(default_factory=list)
    # 生成时就定下的消息 id 和所在 trace。停在确认卡上时这句话要到下一次执行才写进会话，
    # 那时的 trace 已经换了；id 也得在写进会话之前就有，历史接口补出来的这条才能被评价。
    id: str = Field(default_factory=lambda: uuid4().hex)
    trace_id: str | None = None


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
    # 确认卡之前领域 Agent 已经对用户说过的话。停在确认卡上时子图的结果还没回到父图，
    # 这些话只能随中断带出来：历史接口据此补上，恢复时和确认决定一起写进会话。
    notes: list[AgentNote] = Field(default_factory=list)


class ToolResult(BaseModel):
    task_id: str
    tool: str
    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DialogueTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


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
    # 最近几条会话原文（用户和助手），按时间顺序，含本轮排在前面的任务刚写下的回答。
    # 原先只给助手说过的话、用户意图只经 Supervisor 改写后的 user_goal 进入，可改写本身会
    # 丢信息或解析错，领域 Agent 没有原话就发现不了；也不知道用户追问的"为什么"指什么。
    # 字段来源的底线改由 prompt 规则和确认卡守住：写工具的每个参数都在卡上逐项给用户过目。
    recent_messages: list[DialogueTurn] = Field(default_factory=list, max_length=50)


class DomainTaskResult(BaseModel):
    """领域子图返回给父图的稳定输出契约。"""

    task_id: str
    status: TaskStatus
    # 转交和用户取消时为空：分错的任务不对用户说话，接手的领域 Agent 会回答；
    # 取消的决定本身已经记在会话里。
    answer: str = ""
    artifact: dict[str, Any] | None = None
    tool_results: list[ToolResult] = Field(default_factory=list)
    # 只在 WAITING_INPUT 时有值，由父图按 task_id 保存，续跑时放回 DomainTaskRequest。
    draft: TaskDraft | None = None
    # 只在 HANDED_OFF 时有值：领域 Agent 认为该接手的领域。
    handoff_to: AgentName | None = None
    # 回答之前已经流给用户、但还没进会话的话。停在确认卡之前说的随 PendingConfirmation 走了，不在这里。
    notes: list[AgentNote] = Field(default_factory=list)
    # 写出 answer 的那次执行的 trace。并行分支里先办完的一支要等另一支确认后才归并，
    # 归并时的 trace 不是生成回答的那个。
    trace_id: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "DomainTaskResult":
        if self.status == TaskStatus.HANDED_OFF:
            if self.handoff_to is None:
                raise ValueError("a handed-off result must name the target domain")
        elif self.status != TaskStatus.REJECTED and not self.answer.strip():
            raise ValueError("answer is required unless the task is handed off or rejected")
        return self


class TravelApplication(BaseModel):
    application_id: UUID = Field(default_factory=uuid4)
    user_id: str
    destination: str
    start_date: date
    end_date: date
    purpose: str
    status: str = "submitted"
