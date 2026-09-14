import pytest
from pydantic import ValidationError

from enterprise_ai_assistant.core.models import (
    AgentName,
    ContextResolution,
    PlannedTask,
    TaskOutline,
    TaskPlan,
    TaskStatus,
)


def task(task_id: str, dependencies: list[str]) -> PlannedTask:
    return PlannedTask(
        id=task_id,
        title=task_id,
        domain=AgentName.TRAVEL,
        objective="处理差旅任务",
        depends_on=dependencies,
    )


def test_task_plan_rejects_dependency_cycles() -> None:
    with pytest.raises(ValidationError, match="形成了环"):
        TaskPlan(
            user_goal="循环计划",
            tasks=[task("task-1", ["task-2"]), task("task-2", ["task-1"])],
        )


def test_task_plan_requires_at_least_one_task() -> None:
    with pytest.raises(ValidationError):
        TaskPlan(user_goal="空计划", tasks=[])


def _resolution(tasks: list[TaskOutline]) -> ContextResolution:
    return ContextResolution(
        standalone_request="去上海出差并订会议室",
        intent_summary="出差并订会议室",
        requires_task_planning=True,
        tasks=tasks,
    )


def test_supervisor_tasks_become_the_plan_with_runtime_ids() -> None:
    resolution = _resolution(
        [
            TaskOutline(title="上海出差申请", domain=AgentName.TRAVEL, objective="申请出差"),
            TaskOutline(
                title="预订会议室",
                domain=AgentName.MEETING,
                objective="在出差地订会议室",
                depends_on=[AgentName.TRAVEL],
            ),
        ]
    )

    plan = resolution.plan()

    assert plan is not None
    assert plan.user_goal == "去上海出差并订会议室"
    # 依赖按领域写出，运行时换成前置任务的 id；状态由运行时给，不来自模型。
    assert [(item.id, item.depends_on, item.status) for item in plan.tasks] == [
        ("task-1", [], TaskStatus.PENDING),
        ("task-2", ["task-1"], TaskStatus.PENDING),
    ]
    assert resolution.domains == [AgentName.TRAVEL, AgentName.MEETING]


def test_dependency_on_a_later_or_missing_task_fails_validation_so_the_output_is_retried() -> None:
    # 在理解阶段抛错才会触发结构化输出的重试；拖到规划节点再发现，整轮只能失败。
    with pytest.raises(ValidationError, match="排在它前面的任务没有这些领域"):
        _resolution(
            [
                TaskOutline(
                    title="预订会议室",
                    domain=AgentName.MEETING,
                    objective="订会议室",
                    depends_on=[AgentName.TRAVEL],
                ),
                TaskOutline(title="上海出差申请", domain=AgentName.TRAVEL, objective="申请出差"),
            ]
        )


def test_request_without_tasks_has_no_plan() -> None:
    assert _resolution([]).plan() is None
