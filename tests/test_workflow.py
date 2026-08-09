from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from enterprise_ai_assistant.agents.domain_agents import (
    ExpenseAgent,
    HRAgent,
    PolicyAgent,
    TravelAgent,
)
from enterprise_ai_assistant.agents.supervisor import SupervisorAgent
from enterprise_ai_assistant.core.models import GoalUnderstanding, PlannedTask, TaskPlan
from enterprise_ai_assistant.graph.workflow import Workflow, build_graph
from enterprise_ai_assistant.repositories.actions import InMemoryActionRepository
from enterprise_ai_assistant.repositories.policies import InMemoryPolicyRepository
from enterprise_ai_assistant.services.capabilities import CapabilityRegistry
from enterprise_ai_assistant.services.planning import PlanningService


class StubPlanningService:
    async def understand(
        self, user_text: str, conversation_history: str = ""
    ) -> GoalUnderstanding:
        assert "上海" in user_text
        return GoalUnderstanding(
            normalized_goal="申请上海差旅并在返程后提醒报销",
            inferred_slots={
                "destination": "上海",
                "start_date": "2026-08-10",
                "end_date": "2026-08-14",
                "purpose": "客户交流",
            },
        )

    async def plan(self, understanding: GoalUnderstanding) -> TaskPlan:
        return TaskPlan(
            user_goal=understanding.normalized_goal,
            tasks=[
                PlannedTask(
                    id="task-1",
                    title="创建上海差旅申请",
                    operation="submit",
                    required_capabilities={"travel.application.write"},
                    risk="high",
                ),
                PlannedTask(
                    id="task-2",
                    title="返程后提醒报销",
                    operation="schedule",
                    required_capabilities={"expense.reminder.write"},
                    depends_on=["task-1"],
                ),
            ],
            extracted_slots=understanding.inferred_slots,
        )


class GreetingPlanningService:
    async def understand(
        self, user_text: str, conversation_history: str = ""
    ) -> GoalUnderstanding:
        assert user_text == "你好"
        return GoalUnderstanding(normalized_goal="问候")

    async def plan(self, understanding: GoalUnderstanding) -> TaskPlan:
        return TaskPlan(
            user_goal=understanding.normalized_goal,
            tasks=[],
            direct_answer="你好！有什么企业事务需要我协助吗？",
        )


class MissingTravelPlanningService:
    async def understand(
        self, user_text: str, conversation_history: str = ""
    ) -> GoalUnderstanding:
        return GoalUnderstanding(
            normalized_goal="创建差旅申请",
            inferred_slots={"start_date": "2026-08-11"},
        )

    async def plan(self, understanding: GoalUnderstanding) -> TaskPlan:
        return TaskPlan(
            user_goal=understanding.normalized_goal,
            tasks=[
                PlannedTask(
                    id="task-1",
                    title="创建差旅申请",
                    operation="submit",
                    required_capabilities=["travel.application.write"],
                    risk="high",
                )
            ],
            extracted_slots=understanding.inferred_slots,
        )


class HistoryPlanningService:
    def __init__(self) -> None:
        self.histories: list[str] = []

    async def understand(
        self, user_text: str, conversation_history: str = ""
    ) -> GoalUnderstanding:
        self.histories.append(conversation_history)
        return GoalUnderstanding(normalized_goal=user_text)

    async def plan(self, understanding: GoalUnderstanding) -> TaskPlan:
        return TaskPlan(
            user_goal=understanding.normalized_goal,
            tasks=[],
            direct_answer=f"收到：{understanding.normalized_goal}",
        )


def make_graph(
    planning: PlanningService | None = None,
) -> tuple[Any, InMemoryActionRepository]:
    policies = InMemoryPolicyRepository()
    actions = InMemoryActionRepository()
    supervisor = SupervisorAgent(planning or StubPlanningService(), CapabilityRegistry())
    workflow = Workflow(
        supervisor,
        TravelAgent(policies, actions),
        ExpenseAgent(policies, actions),
        HRAgent(policies, actions),
        PolicyAgent(policies),
    )
    return build_graph(workflow, InMemorySaver()), actions


def initial_state() -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content="下周去上海出差，帮我申请，回来提醒报销")],
        "user_id": "u-1",
        "user_goal": "",
        "tasks": [],
        "slots": {},
        "tool_results": [],
        "current_agent": None,
        "active_task_id": None,
        "pending_confirmation": None,
        "last_answer": "",
    }


@pytest.mark.asyncio
async def test_compound_workflow_interrupt_resume_and_shared_state() -> None:
    graph, actions = make_graph()
    config = {"configurable": {"thread_id": "thread-1"}}
    await graph.ainvoke(initial_state(), config)
    paused = await graph.aget_state(config)
    assert "confirm" in paused.next
    assert paused.values["pending_confirmation"].action == "travel.application.submit"
    assert [task.id for task in paused.values["tasks"]] == ["task-1", "task-2"]
    assert paused.values["tasks"][0].status.value == "waiting_confirmation"

    final = await graph.ainvoke(Command(resume={"approved": True}), config)
    assert [task.status.value for task in final["tasks"]] == ["completed", "completed"]
    assert final["slots"]["travel_application"]["destination"] == "上海"
    assert final["slots"]["expense_reminder"]["travel_reference"] == "u-1:task-1"
    assert len(actions.records) == 1

    # 仓储层的幂等性也能防止工具重试造成重复写入。
    first = next(iter(actions.records.values()))
    repeated = await actions.execute_once(
        idempotency_key="u-1:task-1",
        action_type="travel",
        user_id="u-1",
        payload={"destination": "错误覆盖"},
    )
    assert repeated == first


@pytest.mark.asyncio
async def test_reject_cancels_dependent_task_without_writing() -> None:
    graph, actions = make_graph()
    config = {"configurable": {"thread_id": "thread-2"}}
    await graph.ainvoke(initial_state(), config)
    final = await graph.ainvoke(Command(resume={"approved": False}), config)
    assert [task.status.value for task in final["tasks"]] == ["rejected", "rejected"]
    assert actions.records == {}


@pytest.mark.asyncio
async def test_greeting_returns_direct_answer_without_tasks() -> None:
    graph, actions = make_graph(GreetingPlanningService())
    state = initial_state()
    state["messages"] = [HumanMessage(content="你好")]

    final = await graph.ainvoke(state, {"configurable": {"thread_id": "thread-greeting"}})

    assert final["tasks"] == []
    assert final["last_answer"] == "你好！有什么企业事务需要我协助吗？"
    assert final["messages"][-1].content == final["last_answer"]
    assert actions.records == {}


@pytest.mark.asyncio
async def test_missing_information_keeps_task_waiting_for_input() -> None:
    graph, actions = make_graph(MissingTravelPlanningService())

    final = await graph.ainvoke(
        initial_state(), {"configurable": {"thread_id": "thread-missing-input"}}
    )

    assert final["tasks"][0].status.value == "waiting_input"
    assert final["last_answer"] == "创建差旅申请前还需要补充：出差地点、返回日期、出差事由。"
    assert actions.records == {}


@pytest.mark.asyncio
async def test_follow_up_turn_passes_recent_conversation_to_understanding() -> None:
    planning = HistoryPlanningService()
    graph, _ = make_graph(planning)
    config = {"configurable": {"thread_id": "thread-history"}}
    state = initial_state()
    state["messages"] = [HumanMessage(content="我明天要出差")]

    await graph.ainvoke(state, config)
    final = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="改成后天")],
            "user_id": "u-1",
            "last_answer": "",
            "pending_confirmation": None,
        },
        config,
    )

    assert planning.histories[0] == ""
    assert "用户：我明天要出差" in planning.histories[1]
    assert "助手：收到：我明天要出差" in planning.histories[1]
    assert final["messages"][-1].content == "收到：改成后天"
