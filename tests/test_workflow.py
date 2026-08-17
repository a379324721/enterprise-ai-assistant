from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from enterprise_ai_assistant.agents.supervisor import SupervisorAgent
from enterprise_ai_assistant.core.models import (
    AgentName,
    ContextResolution,
    PlannedTask,
    TaskPlan,
)
from enterprise_ai_assistant.graph.workflow import Workflow, build_graph
from enterprise_ai_assistant.repositories.actions import InMemoryActionRepository
from enterprise_ai_assistant.repositories.policies import InMemoryPolicyRepository
from enterprise_ai_assistant.tools import LocalEnterpriseToolProvider, ToolContext
from enterprise_ai_assistant.tools.registry import DomainToolRegistry, RegisteredTool


class StubPlanningService:
    async def resolve_context(
        self, conversation: list[dict[str, str]]
    ) -> ContextResolution:
        assert "上海" in conversation[-1]["content"]
        return ContextResolution(
            standalone_request="2026-08-10 至 2026-08-14 去上海客户交流，创建差旅并提醒报销",
            intent_summary="申请差旅并设置报销提醒",
        )

    async def plan(self, context: ContextResolution) -> TaskPlan:
        return TaskPlan(
            user_goal=context.standalone_request,
            tasks=[
                PlannedTask(
                    id="task-1",
                    title="创建上海差旅申请",
                    domain=AgentName.TRAVEL,
                    objective="根据请求创建差旅申请",
                    success_criteria=["返回差旅申请编号"],
                ),
                PlannedTask(
                    id="task-2",
                    title="返程后提醒报销",
                    domain=AgentName.EXPENSE,
                    objective="在行程结束日设置报销提醒",
                    depends_on=["task-1"],
                    success_criteria=["返回提醒状态"],
                ),
            ],
        )


class ScriptedRuntime:
    def __init__(
        self, name: AgentName, tools: list[RegisteredTool]
    ) -> None:
        self.name = name
        self.tools = {item.tool.name: item for item in tools}

    def tool(self, name: str) -> RegisteredTool:
        return self.tools[name]

    async def decide(
        self,
        task_objective: str,
        messages: list[BaseMessage],
        *,
        task_id: str,
    ) -> AIMessage:
        del task_objective
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content="")
        if self.name == AgentName.TRAVEL:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "create_travel_application",
                        "args": {
                            "destination": "上海",
                            "start_date": "2026-08-10",
                            "end_date": "2026-08-14",
                            "purpose": "客户交流",
                        },
                        "id": f"{task_id}-travel-call",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "schedule_expense_reminder",
                    "args": {
                        "trigger_date": "2026-08-14",
                        "note": "提醒提交本次差旅费用",
                        "travel_reference": "u-1:task-1:create_travel_application",
                    },
                    "id": f"{task_id}-expense-call",
                    "type": "tool_call",
                }
            ],
        )

    async def respond(
        self,
        task_objective: str,
        messages: list[BaseMessage],
        *,
        task_id: str,
    ) -> AIMessage:
        del task_objective, messages, task_id
        return AIMessage(content=f"{self.name.value} 任务处理完成")

    async def invoke_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self.tools[name].tool.ainvoke(arguments)
        assert isinstance(result, dict)
        return result


class ScriptedRuntimeFactory:
    def __init__(self, registry: DomainToolRegistry) -> None:
        self.registry = registry

    def create(self, agent: AgentName, context: ToolContext) -> ScriptedRuntime:
        return ScriptedRuntime(agent, self.registry.for_agent(agent, context))


def make_graph() -> tuple[Any, InMemoryActionRepository]:
    actions = InMemoryActionRepository()
    provider = LocalEnterpriseToolProvider(actions, InMemoryPolicyRepository())
    workflow = Workflow(
        SupervisorAgent(StubPlanningService()),
        ScriptedRuntimeFactory(DomainToolRegistry(provider)),
    )
    return build_graph(workflow, InMemorySaver()), actions


def initial_state() -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content="下周去上海出差，帮我申请，回来提醒报销")],
        "user_id": "u-1",
        "user_goal": "",
        "tasks": [],
        "artifacts": {},
        "tool_results": [],
        "current_agent": None,
        "active_task_id": None,
        "pending_confirmation": None,
        "last_answer": "",
    }


@pytest.mark.asyncio
async def test_compound_workflow_uses_tools_with_separate_confirmations() -> None:
    graph, actions = make_graph()
    config = {"configurable": {"thread_id": "thread-1"}}

    await graph.ainvoke(initial_state(), config)
    first_pause = await graph.aget_state(config)
    assert "confirm_tool" in first_pause.next
    assert first_pause.values["pending_confirmation"].action == "create_travel_application"

    await graph.ainvoke(Command(resume={"approved": True}), config)
    second_pause = await graph.aget_state(config)
    assert "confirm_tool" in second_pause.next
    assert second_pause.values["pending_confirmation"].action == "schedule_expense_reminder"

    final = await graph.ainvoke(Command(resume={"approved": True}), config)
    assert [task.status.value for task in final["tasks"]] == ["completed", "completed"]
    assert final["artifacts"]["task-1"]["data"]["destination"] == "上海"
    assert final["artifacts"]["task-2"]["data"]["travel_reference"].startswith("u-1")
    assert len(actions.records) == 2


@pytest.mark.asyncio
async def test_rejecting_write_cancels_dependent_task() -> None:
    graph, actions = make_graph()
    config = {"configurable": {"thread_id": "thread-2"}}

    await graph.ainvoke(initial_state(), config)
    final = await graph.ainvoke(Command(resume={"approved": False}), config)

    assert [task.status.value for task in final["tasks"]] == ["rejected", "rejected"]
    assert actions.records == {}
