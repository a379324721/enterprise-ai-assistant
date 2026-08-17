import pytest
from pydantic import ValidationError

from enterprise_ai_assistant.core.models import AgentName, PlannedTask, TaskPlan


def task(task_id: str, dependencies: list[str]) -> PlannedTask:
    return PlannedTask(
        id=task_id,
        title=task_id,
        domain=AgentName.TRAVEL,
        objective="处理差旅任务",
        depends_on=dependencies,
    )


def test_task_plan_rejects_dependency_cycles() -> None:
    with pytest.raises(ValidationError, match="acyclic"):
        TaskPlan(
            user_goal="循环计划",
            tasks=[task("task-1", ["task-2"]), task("task-2", ["task-1"])],
        )


def test_task_plan_requires_at_least_one_task() -> None:
    with pytest.raises(ValidationError):
        TaskPlan(user_goal="空计划", tasks=[])
