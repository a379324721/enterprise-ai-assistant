from langsmith import traceable

from enterprise_ai_assistant.core.models import (
    AgentName,
    GoalUnderstanding,
    PlannedTask,
    TaskPlan,
    TaskStatus,
)
from enterprise_ai_assistant.services.capabilities import CapabilityRegistry
from enterprise_ai_assistant.services.planning import PlanningService


class SupervisorAgent:
    """负责理解、拆解和委派任务，但不执行领域写操作。"""

    def __init__(self, planning: PlanningService, capabilities: CapabilityRegistry) -> None:
        self._planning = planning
        self._capabilities = capabilities

    @traceable(name="supervisor-understand", run_type="chain")
    async def understand(self, user_text: str) -> GoalUnderstanding:
        return await self._planning.understand(user_text)

    @traceable(name="supervisor-plan", run_type="chain")
    async def plan(self, understanding: GoalUnderstanding) -> TaskPlan:
        return await self._planning.plan(understanding)

    @traceable(name="supervisor-capability-routing", run_type="chain")
    def route(self, task: PlannedTask) -> AgentName:
        return self._capabilities.select(task)

    @staticmethod
    def next_runnable(tasks: list[PlannedTask]) -> PlannedTask | None:
        completed = {task.id for task in tasks if task.status == TaskStatus.COMPLETED}
        for task in tasks:
            if task.status == TaskStatus.PENDING and set(task.depends_on) <= completed:
                return task
        return None
