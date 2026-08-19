from typing import Annotated, Any, NotRequired, TypedDict
from uuid import UUID

from langgraph.graph.message import add_messages

from enterprise_ai_assistant.core.models import (
    DomainTaskRequest,
    DomainTaskResult,
    PendingConfirmation,
    PlannedTask,
    ToolResult,
)


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
    last_answer: str
    understanding: NotRequired[dict[str, Any]]
    turn_answers: NotRequired[list[str]]
    domain_request: NotRequired[DomainTaskRequest | None]
    domain_result: NotRequired[DomainTaskResult | None]


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
    artifact: NotRequired[dict[str, Any] | None]
    domain_tool_results: list[ToolResult]
