from collections.abc import Sequence

from langchain_core.messages import AIMessage
from langsmith import traceable

from enterprise_ai_assistant.core.models import (
    AgentName,
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
    ) -> ContextResolution:
        return await self._planning.resolve_context(conversation, memory_keys, open_tasks)

    @traceable(name="supervisor-plan", run_type="chain")
    async def plan(self, context: ContextResolution) -> TaskPlan:
        return await self._planning.plan(context)

    @traceable(name="supervisor-direct-response", run_type="chain")
    async def respond_direct(
        self,
        context: ContextResolution,
        memories: Sequence[str] = (),
        recent_actions: Sequence[str] = (),
        user_name: str = "",
        notices: Sequence[str] = (),
    ) -> AIMessage:
        return await self._planning.respond_direct(
            context, memories, recent_actions, user_name, notices
        )

    @traceable(name="supervisor-extract-memories", run_type="chain")
    async def extract_memories(
        self, conversation: list[dict[str, str]], known: list[str]
    ) -> MemoryExtraction:
        return await self._planning.extract_memories(conversation, known)

    @staticmethod
    def single_domain_plan(context: ContextResolution) -> TaskPlan | None:
        """Supervisor 已认定只涉及一个领域时，直接构造单任务计划，省掉一次 Planner 调用。

        Planner 的规则是"一个领域一个目标"，单领域请求交给它也只会拆出一个任务，
        等它的只是一次可预知结果的结构化输出。领域不唯一或没给出时返回 None，走 Planner。
        """
        domains = set(context.domains) - {AgentName.SUPERVISOR}
        if len(domains) != 1 or len(set(context.domains)) != 1:
            return None
        return TaskPlan(
            user_goal=context.standalone_request,
            tasks=[
                PlannedTask(
                    id="task-1",
                    title=context.intent_summary[:200],
                    domain=domains.pop(),
                    objective=context.standalone_request[:2000],
                )
            ],
        )

    @staticmethod
    @traceable(name="supervisor-task-scheduling", run_type="chain")
    def next_runnable(tasks: list[PlannedTask]) -> PlannedTask | None:
        completed = {task.id for task in tasks if task.status == TaskStatus.COMPLETED}
        for task in tasks:
            if task.status == TaskStatus.PENDING and set(task.depends_on) <= completed:
                return task
        return None
