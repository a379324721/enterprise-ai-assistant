"""右栏事项卡和对话流执行步骤的投影。"""

from datetime import UTC, datetime
from uuid import UUID

from enterprise_ai_assistant.api.matters import project_matters
from enterprise_ai_assistant.api.routes import _steps
from enterprise_ai_assistant.core.models import (
    AgentName,
    ConfirmationField,
    DraftField,
    PendingConfirmation,
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

    assert project_matters({"plan_id": "p-1", "tasks": tasks}, running=False) == []
    assert project_matters({"plan_id": "p-1", "tasks": tasks}, running=True) == []


def test_waiting_plan_becomes_a_card_focused_on_the_stuck_task() -> None:
    tasks = [
        _task("task-1", "差旅申请", TaskStatus.COMPLETED, AgentName.TRAVEL),
        _task("task-2", "预订会议室", TaskStatus.WAITING_INPUT, AgentName.MEETING),
    ]
    draft = TaskDraft(
        known_fields=[DraftField(name="city", label="地点", value="上海", source="dependency")],
        missing_fields=["会议主题"],
    )

    [card] = project_matters(
        {"plan_id": "p-1", "tasks": tasks, "drafts": {"task-2": draft}}, running=False
    )

    assert (card.status, card.task_id, card.title) == ("waiting_input", "task-2", "预订会议室")
    assert card.known_fields[0].value == "上海"
    assert card.missing_fields == ["会议主题"]
    assert [item.status for item in card.tasks] == [TaskStatus.COMPLETED, TaskStatus.WAITING_INPUT]


def test_waiting_confirmation_card_shows_the_arguments_about_to_be_submitted() -> None:
    """草稿停在追问那一刻：补齐字段后进入确认，卡片不能还列着一排"待补充"。"""
    tasks = [
        _task("task-1", "上海出差申请", TaskStatus.WAITING_CONFIRMATION, AgentName.TRAVEL),
        _task("task-2", "预订上海会议室", TaskStatus.PENDING, AgentName.MEETING),
    ]
    stale = TaskDraft(
        known_fields=[DraftField(name="destination", label="目的地", value="上海")],
        missing_fields=["出发地", "结束日期", "出差事由"],
    )
    pending = PendingConfirmation(
        task_id="task-1",
        action="create_travel_application",
        tool_call_id="call-1",
        title="提交差旅申请",
        fields=[
            ConfirmationField(name="origin", label="出发地", value="北京"),
            ConfirmationField(name="destination", label="目的地", value="上海"),
            ConfirmationField(name="purpose", label="出差事由", value="培训"),
        ],
        payload={},
    )

    [card] = project_matters(
        {"plan_id": "p-1", "drafts": {"task-1": stale}},
        running=False,
        tasks=tasks,
        pending=pending,
    )

    assert card.status == "waiting_confirmation"
    assert [(item.label, item.value) for item in card.known_fields] == [
        ("出发地", "北京"),
        ("目的地", "上海"),
        ("出差事由", "培训"),
    ]
    assert card.missing_fields == []


def test_plan_with_work_left_stays_a_card_while_the_run_is_executing() -> None:
    """差旅确认完、会议室还没开始追问的那段时间，卡片不能先消失再冒出来。"""
    tasks = [
        _task("task-1", "上海出差申请", TaskStatus.COMPLETED, AgentName.TRAVEL),
        _task("task-2", "预订上海会议室", TaskStatus.PENDING, AgentName.MEETING),
    ]
    resumed = TaskDraft(
        known_fields=[DraftField(name="date", label="日期", value="2026-09-16")],
        missing_fields=["会议主题"],
    )
    values = {"plan_id": "p-1", "tasks": tasks, "drafts": {"task-2": resumed}}

    [card] = project_matters(values, running=True)

    assert (card.status, card.task_id, card.title) == ("in_progress", "task-2", "预订上海会议室")
    # 已知字段沿用；缺失字段是上一轮追问时的，这时再列出来就过期了。
    assert [item.value for item in card.known_fields] == ["2026-09-16"]
    assert card.missing_fields == []


def test_running_task_is_the_focus_over_queued_ones() -> None:
    tasks = [
        _task("task-1", "查询年假余额", TaskStatus.PENDING, AgentName.HR),
        _task("task-2", "预订会议室", TaskStatus.RUNNING, AgentName.MEETING),
    ]

    [card] = project_matters({"plan_id": "p-1", "tasks": tasks}, running=True)

    assert (card.status, card.task_id) == ("in_progress", "task-2")


def test_leftovers_of_a_failed_run_are_not_shown_as_in_progress() -> None:
    """一轮在领域子图里失败，任务停在 RUNNING。没有运行在跑时不能一直显示处理中，
    要和下一轮 understand 续跑时的解读一致。"""
    first_attempt = [_task("task-1", "上海出差申请", TaskStatus.RUNNING, AgentName.TRAVEL)]
    assert project_matters({"plan_id": "p-1", "tasks": first_attempt}, running=False) == []

    resumed = [_task("task-1", "上海出差申请", TaskStatus.RUNNING, AgentName.TRAVEL)]
    draft = TaskDraft(missing_fields=["出差事由"])
    [card] = project_matters(
        {"plan_id": "p-1", "tasks": resumed, "drafts": {"task-1": draft}}, running=False
    )
    assert (card.status, card.missing_fields) == ("waiting_input", ["出差事由"])


def test_topic_change_keeps_the_shelved_plan_while_the_new_one_runs() -> None:
    """换话题那一轮，被搁置的事项和新计划必须同时在：两者来自同一份状态、同一次投影。"""
    shelved = ShelvedPlan(
        plan_id="p-old",
        user_goal="出差",
        tasks=[_task("task-1", "上海出差申请", TaskStatus.WAITING_INPUT, AgentName.TRAVEL)],
    )
    values = {
        "plan_id": "p-new",
        "tasks": [_task("task-1", "查询年假余额", TaskStatus.RUNNING, AgentName.HR)],
        "shelved_plans": [shelved],
    }

    cards = project_matters(values, running=True)

    assert [(card.plan_id, card.status) for card in cards] == [
        ("p-new", "in_progress"),
        ("p-old", "shelved"),
    ]


def test_shelved_plan_without_a_stuck_task_is_not_a_card() -> None:
    shelved = ShelvedPlan(
        plan_id="p-old",
        user_goal="出差",
        tasks=[_task("task-1", "差旅申请", TaskStatus.PENDING, AgentName.TRAVEL)],
    )

    assert project_matters({"plan_id": "p-cur", "shelved_plans": [shelved]}, running=True) == []


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

    cards = project_matters(
        {"plan_id": "p-cur", "tasks": current, "shelved_plans": [older, newer]}, running=False
    )

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


def test_a_task_the_user_cancelled_shows_no_steps() -> None:
    """取消没有回答，卡片之前查过的步骤会孤零零挂在"你取消了"下面。"""
    at = datetime(2026, 9, 13, tzinfo=UTC)
    results = [
        ToolResult(task_id="task-1", tool="get_leave_balance", success=True, created_at=at),
        ToolResult(task_id="task-2", tool="search_travel_policy", success=True, created_at=at),
    ]
    tasks = [
        _task("task-1", "提交请假申请", TaskStatus.REJECTED, AgentName.HR),
        _task("task-2", "查询差旅制度", TaskStatus.COMPLETED, AgentName.TRAVEL),
    ]

    assert [step.task_id for step in _steps(results, tasks)] == ["task-2"]


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
