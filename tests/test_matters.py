"""右栏事项卡和对话流执行步骤的投影。"""

from datetime import UTC, datetime
from uuid import UUID

from enterprise_ai_assistant.api.routes import _matters, _steps
from enterprise_ai_assistant.core.models import (
    AgentName,
    DraftField,
    PlannedTask,
    ShelvedPlan,
    TaskDraft,
    TaskStatus,
    ToolResult,
)
from enterprise_ai_assistant.repositories.actions import InMemoryActionRepository
from enterprise_ai_assistant.repositories.policies import InMemoryPolicyRepository
from enterprise_ai_assistant.tools import LocalEnterpriseToolProvider, ToolContext
from enterprise_ai_assistant.tools.registry import TOOL_LABELS, DomainToolRegistry


def _task(task_id: str, title: str, status: TaskStatus, domain: AgentName) -> PlannedTask:
    return PlannedTask(id=task_id, title=title, domain=domain, objective=title, status=status)


def test_finished_plan_is_not_a_matter() -> None:
    """办完的事已经在"我的单据"里，查询也没有要跟进的状态，都不占右栏。"""
    tasks = [_task("task-1", "查询差旅制度", TaskStatus.COMPLETED, AgentName.TRAVEL)]

    assert _matters({"plan_id": "p-1"}, tasks) == []


def test_waiting_plan_becomes_a_card_focused_on_the_stuck_task() -> None:
    tasks = [
        _task("task-1", "差旅申请", TaskStatus.COMPLETED, AgentName.TRAVEL),
        _task("task-2", "预订会议室", TaskStatus.WAITING_INPUT, AgentName.MEETING),
    ]
    draft = TaskDraft(
        known_fields=[DraftField(name="city", label="地点", value="上海", source="dependency")],
        missing_fields=["会议主题"],
    )

    [card] = _matters({"plan_id": "p-1", "drafts": {"task-2": draft}}, tasks)

    assert (card.status, card.task_id, card.title) == ("waiting_input", "task-2", "预订会议室")
    assert card.known_fields[0].value == "上海"
    assert card.missing_fields == ["会议主题"]
    assert [item.status for item in card.tasks] == [TaskStatus.COMPLETED, TaskStatus.WAITING_INPUT]


def test_shelved_plans_follow_the_current_one_most_recent_first() -> None:
    current = [_task("task-1", "请假申请", TaskStatus.WAITING_CONFIRMATION, AgentName.HR)]
    older = ShelvedPlan(
        plan_id="p-old",
        user_goal="出差",
        tasks=[_task("task-1", "差旅申请", TaskStatus.WAITING_INPUT, AgentName.TRAVEL)],
    )
    newer = ShelvedPlan(
        plan_id="p-new",
        user_goal="订会议室",
        tasks=[_task("task-1", "预订会议室", TaskStatus.WAITING_INPUT, AgentName.MEETING)],
    )

    cards = _matters({"plan_id": "p-cur", "shelved_plans": [older, newer]}, current)

    assert [(card.plan_id, card.status) for card in cards] == [
        ("p-cur", "waiting_confirmation"),
        ("p-new", "shelved"),
        ("p-old", "shelved"),
    ]


def test_steps_use_server_side_tool_labels() -> None:
    at = datetime(2026, 9, 13, tzinfo=UTC)
    results = [
        ToolResult(task_id="task-1", tool="search_travel_policy", success=True, created_at=at),
        ToolResult(task_id="task-1", tool="create_travel_application", success=False, created_at=at),
    ]

    steps = _steps(results, [])

    assert [(step.label, step.success) for step in steps] == [
        ("检索差旅制度", True),
        ("提交差旅申请", False),
    ]
    assert len({step.id for step in steps}) == 2


def test_every_registered_tool_has_a_label() -> None:
    """新增工具忘了起中文名，界面上就会冒出一个英文函数名。"""
    registry = DomainToolRegistry(
        LocalEnterpriseToolProvider(InMemoryActionRepository(), InMemoryPolicyRepository())
    )
    context = ToolContext(
        user_id="u-1",
        conversation_id=UUID("00000000-0000-0000-0000-000000000001"),
        request_id=UUID("00000000-0000-0000-0000-000000000002"),
        task_id="task-1",
    )
    names = {
        item.tool.name
        for agent in AgentName
        if agent != AgentName.SUPERVISOR
        for item in registry.for_agent(agent, context)
    }

    assert names <= set(TOOL_LABELS)
