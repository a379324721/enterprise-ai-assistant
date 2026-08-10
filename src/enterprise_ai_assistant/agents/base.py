from dataclasses import dataclass, field
from typing import Any, Protocol

from enterprise_ai_assistant.core.models import (
    PendingConfirmation,
    PlannedTask,
    TaskStatus,
    ToolResult,
)


@dataclass
class AgentOutcome:
    answer: str
    slot_updates: dict[str, Any] = field(default_factory=dict)
    tool_result: ToolResult | None = None
    confirmation: PendingConfirmation | None = None
    task_status: TaskStatus | None = None


class DomainAgent(Protocol):
    async def prepare(self, task: PlannedTask, slots: dict[str, Any]) -> AgentOutcome: ...


class WritableDomainAgent(DomainAgent, Protocol):
    async def execute(
        self,
        task: PlannedTask,
        user_id: str,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
    ) -> AgentOutcome: ...
