from collections.abc import Mapping
from typing import Annotated, Any, NotRequired, TypedDict
from uuid import UUID, uuid4

from langgraph.graph.message import add_messages

from enterprise_ai_assistant.core.models import (
    AgentNote,
    DomainTaskRequest,
    DomainTaskResult,
    MemoryRecord,
    PendingConfirmation,
    Plan,
    PlannedTask,
    RecentAction,
    TaskDraft,
    ToolResult,
)


def _collect_results(
    left: list[DomainTaskResult] | None, right: list[DomainTaskResult] | None
) -> list[DomainTaskResult]:
    if right is None:
        return []
    return [*(left or []), *right]


class DomainTaskInput(TypedDict):
    """并行派发给单个领域任务分支的输入。"""

    domain_request: DomainTaskRequest


class AssistantState(TypedDict):
    """外层调度状态；领域模型只接收为当前任务构造的 domain_messages。"""

    messages: Annotated[list[Any], add_messages]
    user_id: str
    # 用户的展示名，来自令牌的 name 声明。只用于称呼，不参与会话归属和幂等键。
    # 早于该字段的检查点没有它，读取时一律走 get 兜底。
    user_name: NotRequired[str]
    conversation_id: UUID
    request_id: UUID
    # 当前计划平铺存放（plan_id、user_goal、tasks、artifacts、drafts）。整件事一起处理的地方
    # （搁置、换回、取消、投影成事项）经 current_plan / plan_update 当作一个 Plan 读写。不改成
    # 一个 plan 键：已有检查点按这几个键存着，换了键旧会话的待办就读不出来，而新旧两套键并存
    # 只会让每处读取都要兼容两种形状。
    user_goal: str
    tasks: list[PlannedTask]
    artifacts: dict[str, Any]
    tool_results: list[ToolResult]
    current_agent: str | None
    active_task_id: str | None
    last_answer: str
    understanding: NotRequired[dict[str, Any]]
    history_digest: NotRequired[list[str]]
    turn_answers: NotRequired[list[str]]
    # 本批并行派发的任务请求，由 select_task 写入。
    domain_batch: NotRequired[list[DomainTaskRequest]]
    # 本批各分支的结果。并行分支同时写入，需要归并规则；写 None 表示清空。
    domain_results: Annotated[list[DomainTaskResult], _collect_results]
    # recall 节点在每轮开头写入，供领域子图预填字段；不参与检查点以外的持久化。
    memories: NotRequired[list[MemoryRecord]]
    recent_actions: NotRequired[list[RecentAction]]
    # 停在待补充的任务留下的字段草稿，按 task_id 索引。和 tasks、artifacts 一样
    # 跨轮保留到下一次重新规划，续跑时放回 DomainTaskRequest。
    drafts: NotRequired[dict[str, TaskDraft]]
    # 当前计划的标识，Supervisor 用它指认要续跑或取消的事项。早于该字段的检查点没有它。
    plan_id: NotRequired[str]
    # 换话题时被搁置的未办完计划，跨轮保留，不自动过期。
    shelved_plans: NotRequired[list[Plan]]


def current_plan(values: Mapping[str, Any]) -> Plan:
    """把状态里平铺的当前计划读成一个 Plan。"""
    return Plan(
        # 早于 plan_id 的检查点没有这个字段，补一个，写回后搁置时才能被指认。
        plan_id=values.get("plan_id") or uuid4().hex[:8],
        user_goal=values.get("user_goal") or "",
        tasks=list(values.get("tasks") or []),
        artifacts=dict(values.get("artifacts") or {}),
        drafts=dict(values.get("drafts") or {}),
    )


def plan_update(plan: Plan) -> dict[str, Any]:
    """把一个 Plan 写回成当前计划的状态更新。"""
    return {
        "plan_id": plan.plan_id,
        "user_goal": plan.user_goal,
        "tasks": plan.tasks,
        "artifacts": plan.artifacts,
        "drafts": plan.drafts,
    }


class DomainTaskState(TypedDict):
    """单次领域任务子图状态；只有 request/result 与父图共享。"""

    domain_request: DomainTaskRequest
    domain_result: DomainTaskResult | None
    domain_messages: list[Any]
    domain_iterations: int
    pending_confirmation: NotRequired[PendingConfirmation | None]
    pending_tool_call: NotRequired[dict[str, Any] | None]
    confirmation_approved: NotRequired[bool]
    domain_waiting_input: bool
    domain_rejected: bool
    domain_failed: bool
    domain_retry_required: bool
    domain_tool_executed: bool
    # 不需要再调模型就已经确定的回答：决策调用看完工具结果写出的文字，
    # 或 request_information 给出的问题。为空时由兜底回答调用生成。
    domain_answer: NotRequired[str]
    # 调工具时顺带流给用户、还没交给父图的话；弹确认卡时移进 PendingConfirmation。
    domain_notes: NotRequired[list[AgentNote]]
    # 领域 Agent 调用转交工具时指定的接手领域。
    domain_handoff_to: NotRequired[str | None]
    artifact: NotRequired[dict[str, Any] | None]
    domain_draft: NotRequired[TaskDraft | None]
    domain_tool_results: list[ToolResult]
