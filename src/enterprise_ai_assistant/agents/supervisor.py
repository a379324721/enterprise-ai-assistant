from collections.abc import Sequence

from langsmith import traceable

from enterprise_ai_assistant.core.models import (
    ContextResolution,
    MemoryExtraction,
    OpenTask,
    PlannedTask,
    TaskPlan,
    TaskStatus,
)
from enterprise_ai_assistant.services.planning import PlanningService


class SupervisorAgent:
    """负责理解、拆解和委派任务，但不执行领域写操作。"""

    def __init__(self, planning: PlanningService) -> None:
        self._planning = planning

    @traceable(name="supervisor-understand", run_type="chain")
    async def resolve_context(
        self,
        conversation: list[dict[str, str]],
        memory_keys: Sequence[str] = (),
        open_tasks: Sequence[OpenTask] = (),
        recent_actions: Sequence[str] = (),
        user_name: str = "",
    ) -> ContextResolution:
        return await self._planning.resolve_context(
            conversation, memory_keys, open_tasks, recent_actions, user_name
        )

    async def plan(self, context: ContextResolution) -> TaskPlan:
        """理解结果里已有任务时直接用；没有时才退回单独的 Planner 调用。"""
        return context.plan() or await self._replan(context)

    @traceable(name="supervisor-plan", run_type="chain")
    async def _replan(self, context: ContextResolution) -> TaskPlan:
        return await self._planning.plan(context)

    @traceable(name="supervisor-extract-memories", run_type="chain")
    async def extract_memories(
        self, conversation: list[dict[str, str]], known: list[str]
    ) -> MemoryExtraction:
        return await self._planning.extract_memories(conversation, known)

    @staticmethod
    @traceable(name="supervisor-task-scheduling", run_type="chain")
    def next_runnable(tasks: list[PlannedTask]) -> PlannedTask | None:
        completed = {task.id for task in tasks if task.status == TaskStatus.COMPLETED}
        for task in tasks:
            if task.status == TaskStatus.PENDING and set(task.depends_on) <= completed:
                return task
        return None
