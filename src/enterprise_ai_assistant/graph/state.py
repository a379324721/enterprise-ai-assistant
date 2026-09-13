from typing import Annotated, Any, NotRequired, TypedDict
from uuid import UUID

from langgraph.graph.message import add_messages

from enterprise_ai_assistant.core.models import (
    DomainTaskRequest,
    DomainTaskResult,
    MemoryRecord,
    PendingConfirmation,
    PlannedTask,
    RecentAction,
    ShelvedPlan,
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
    shelved_plans: NotRequired[list[ShelvedPlan]]


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
    # 领域 Agent 调用转交工具时指定的接手领域。
    domain_handoff_to: NotRequired[str | None]
    artifact: NotRequired[dict[str, Any] | None]
    domain_draft: NotRequired[TaskDraft | None]
    domain_tool_results: list[ToolResult]
