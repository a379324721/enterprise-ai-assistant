from typing import Annotated, Any, NotRequired, TypedDict

from langgraph.graph.message import add_messages

from enterprise_ai_assistant.core.models import (
    PendingConfirmation,
    PlannedTask,
    TaskRun,
    ToolResult,
)


class AssistantState(TypedDict):
    """Agent 之间共享的唯一契约；Agent 提示词永远不会接收 `messages`。"""

    messages: Annotated[list[Any], add_messages]
    user_id: str
    user_goal: str
    tasks: list[PlannedTask]
    task_history: list[TaskRun]
    slots: dict[str, Any]
    tool_results: list[ToolResult]
    current_agent: str | None
    active_task_id: str | None
    pending_confirmation: PendingConfirmation | None
    last_answer: str
    understanding: NotRequired[dict[str, Any]]
    confirmation_approved: NotRequired[bool]
