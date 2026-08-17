from typing import Annotated, Any, NotRequired, TypedDict
from uuid import UUID

from langgraph.graph.message import add_messages

from enterprise_ai_assistant.core.models import PendingConfirmation, PlannedTask, ToolResult


class AssistantState(TypedDict):
    """外层调度状态；领域模型只接收为当前任务构造的 domain_messages。"""

    messages: Annotated[list[Any], add_messages]
    user_id: str
    conversation_id: UUID
    request_id: UUID
    user_goal: str
    tasks: list[PlannedTask]
    artifacts: dict[str, Any]
    tool_results: list[ToolResult]
    current_agent: str | None
    active_task_id: str | None
    pending_confirmation: PendingConfirmation | None
    pending_tool_call: NotRequired[dict[str, Any] | None]
    last_answer: str
    understanding: NotRequired[dict[str, Any]]
    confirmation_approved: NotRequired[bool]
    domain_messages: NotRequired[list[Any]]
    domain_iterations: NotRequired[int]
    domain_waiting_input: NotRequired[bool]
    domain_rejected: NotRequired[bool]
    domain_failed: NotRequired[bool]
    domain_retry_required: NotRequired[bool]
    turn_answers: NotRequired[list[str]]
