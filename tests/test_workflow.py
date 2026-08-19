from typing import Any
from uuid import UUID

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from enterprise_ai_assistant.agents.supervisor import SupervisorAgent
from enterprise_ai_assistant.core.models import (
    AgentName,
    ContextResolution,
    DomainTaskRequest,
    PendingConfirmation,
    PlannedTask,
    TaskPlan,
    ToolResult,
)
from enterprise_ai_assistant.graph.domain import DomainTaskWorkflow, build_domain_graph
from enterprise_ai_assistant.graph.state import DomainTaskState
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
            requires_task_planning=True,
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

    async def respond_direct(self, conversation: list[dict[str, str]]) -> AIMessage:
        raise AssertionError(f"not used: {conversation}")


class DirectPlanningService:
    async def resolve_context(
        self, conversation: list[dict[str, str]]
    ) -> ContextResolution:
        assert conversation[-1]["content"] == "你好"
        return ContextResolution(
            standalone_request="你好",
            intent_summary="用户向助手打招呼",
            requires_task_planning=False,
        )

    async def plan(self, context: ContextResolution) -> TaskPlan:
        raise AssertionError(f"direct conversation must not be planned: {context}")

    async def respond_direct(self, conversation: list[dict[str, str]]) -> AIMessage:
        assert conversation[-1]["content"] == "你好"
        return AIMessage(content="你好！有什么企业事务需要我协助？")


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
                        "travel_reference": "travel-reference-1",
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


class InvalidResponseRuntime(ScriptedRuntime):
    async def respond(
        self,
        task_objective: str,
        messages: list[BaseMessage],
        *,
        task_id: str,
    ) -> AIMessage:
        del task_objective, messages, task_id
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_general_policy",
                    "args": {"query": "问候", "limit": 1},
                    "id": "invalid-final-tool-call",
                    "type": "tool_call",
                }
            ],
        )


class InvalidResponseRuntimeFactory(ScriptedRuntimeFactory):
    def create(self, agent: AgentName, context: ToolContext) -> InvalidResponseRuntime:
        return InvalidResponseRuntime(agent, self.registry.for_agent(agent, context))


class InvalidToolRuntime(ScriptedRuntime):
    async def decide(
        self,
        task_objective: str,
        messages: list[BaseMessage],
        *,
        task_id: str,
    ) -> AIMessage:
        del task_objective
        if any(
            isinstance(message, ToolMessage) and message.name == "search_general_policy"
            for message in messages
        ):
            return AIMessage(content="")
        if any(
            isinstance(message, ToolMessage) and message.name == "not_allowed"
            for message in messages
        ):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_general_policy",
                        "args": {"query": "差旅制度", "limit": 1},
                        "id": f"{task_id}-valid-call",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "not_allowed",
                    "args": {},
                    "id": f"{task_id}-invalid-call",
                    "type": "tool_call",
                }
            ],
        )


class InvalidToolRuntimeFactory(ScriptedRuntimeFactory):
    def create(self, agent: AgentName, context: ToolContext) -> InvalidToolRuntime:
        return InvalidToolRuntime(agent, self.registry.for_agent(agent, context))


def make_graph() -> tuple[Any, InMemoryActionRepository]:
    actions = InMemoryActionRepository()
    provider = LocalEnterpriseToolProvider(actions, InMemoryPolicyRepository())
    workflow = Workflow(SupervisorAgent(StubPlanningService()))
    domain_workflow = DomainTaskWorkflow(
        ScriptedRuntimeFactory(DomainToolRegistry(provider))
    )
    return build_graph(workflow, domain_workflow, InMemorySaver()), actions


def initial_state() -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content="下周去上海出差，帮我申请，回来提醒报销")],
        "user_id": "u-1",
        "conversation_id": UUID("00000000-0000-0000-0000-000000000001"),
        "request_id": UUID("00000000-0000-0000-0000-000000000002"),
        "user_goal": "",
        "tasks": [],
        "artifacts": {},
        "tool_results": [],
        "current_agent": None,
        "active_task_id": None,
        "last_answer": "",
    }


@pytest.mark.asyncio
async def test_direct_conversation_skips_planning_and_tools() -> None:
    actions = InMemoryActionRepository()
    provider = LocalEnterpriseToolProvider(actions, InMemoryPolicyRepository())
    workflow = Workflow(SupervisorAgent(DirectPlanningService()))
    domain_workflow = DomainTaskWorkflow(
        ScriptedRuntimeFactory(DomainToolRegistry(provider))
    )
    graph = build_graph(workflow, domain_workflow, InMemorySaver())
    state = initial_state()
    state["messages"] = [HumanMessage(content="你好")]

    final = await graph.ainvoke(state, {"configurable": {"thread_id": "direct-thread"}})

    assert final["last_answer"] == "你好！有什么企业事务需要我协助？"
    assert final["tasks"] == []
    assert actions.records == {}


@pytest.mark.asyncio
async def test_domain_response_rejects_empty_text_and_tool_calls() -> None:
    actions = InMemoryActionRepository()
    provider = LocalEnterpriseToolProvider(actions, InMemoryPolicyRepository())
    workflow = DomainTaskWorkflow(
        InvalidResponseRuntimeFactory(DomainToolRegistry(provider)),
    )
    task = PlannedTask(
            id="task-1",
            title="回答制度问题",
            domain=AgentName.POLICY,
            objective="回答用户的制度问题",
            status="running",
        )
    state: DomainTaskState = {
        "domain_request": DomainTaskRequest(
            user_id="u-1",
            conversation_id=UUID("00000000-0000-0000-0000-000000000001"),
            request_id=UUID("00000000-0000-0000-0000-000000000002"),
            user_goal="回答制度问题",
            task=task,
        ),
        "domain_result": None,
        "domain_messages": [HumanMessage(content="回答制度问题")],
        "domain_iterations": 1,
        "domain_waiting_input": False,
        "domain_rejected": False,
        "domain_failed": False,
        "domain_retry_required": False,
        "tool_results": [],
    }

    with pytest.raises(RuntimeError, match="no user-visible text"):
        await workflow.respond(state)


@pytest.mark.asyncio
async def test_domain_subgraph_recovers_from_tool_outside_allowlist() -> None:
    provider = LocalEnterpriseToolProvider(
        InMemoryActionRepository(), InMemoryPolicyRepository()
    )
    workflow = DomainTaskWorkflow(
        InvalidToolRuntimeFactory(DomainToolRegistry(provider))
    )
    task = PlannedTask(
        id="task-1",
        title="查询制度",
        domain=AgentName.POLICY,
        objective="查询差旅制度",
    )

    final = await build_domain_graph(workflow).ainvoke(
        {
            "domain_request": DomainTaskRequest(
                user_id="u-1",
                conversation_id=UUID("00000000-0000-0000-0000-000000000001"),
                request_id=UUID("00000000-0000-0000-0000-000000000002"),
                user_goal="查询差旅制度",
                task=task,
            ),
            "domain_result": None,
        }
    )

    assert final["domain_result"].status.value == "completed"
    assert final["domain_iterations"] == 3
    assert final["tool_results"][0].tool == "search_general_policy"


@pytest.mark.asyncio
async def test_compound_workflow_uses_tools_with_separate_confirmations() -> None:
    graph, actions = make_graph()
    config = {"configurable": {"thread_id": "thread-1"}}

    await graph.ainvoke(initial_state(), config)
    first_pause = await graph.aget_state(config)
    assert "domain_task" in first_pause.next
    first_confirmation = PendingConfirmation.model_validate(first_pause.interrupts[0].value)
    assert first_confirmation.action == "create_travel_application"
    assert "domain_messages" not in first_pause.values
    assert "pending_confirmation" not in first_pause.values
    nested_pause = await graph.aget_state(config, subgraphs=True)
    nested_state = nested_pause.tasks[0].state
    assert nested_state is not None
    assert nested_state.values["pending_confirmation"].action == "create_travel_application"

    await graph.ainvoke(
        Command(
            resume={
                "confirmation_id": str(first_confirmation.confirmation_id),
                "approved": True,
            }
        ),
        config,
    )
    second_pause = await graph.aget_state(config)
    assert "domain_task" in second_pause.next
    second_confirmation = PendingConfirmation.model_validate(second_pause.interrupts[0].value)
    assert second_confirmation.action == "schedule_expense_reminder"

    final = await graph.ainvoke(
        Command(
            resume={
                "confirmation_id": str(second_confirmation.confirmation_id),
                "approved": True,
            }
        ),
        config,
    )
    assert [task.status.value for task in final["tasks"]] == ["completed", "completed"]
    assert final["artifacts"]["task-1"]["data"]["destination"] == "上海"
    assert final["artifacts"]["task-2"]["data"]["travel_reference"] == "travel-reference-1"
    assert len(actions.records) == 2


@pytest.mark.asyncio
async def test_rejecting_write_cancels_dependent_task() -> None:
    graph, actions = make_graph()
    config = {"configurable": {"thread_id": "thread-2"}}

    await graph.ainvoke(initial_state(), config)
    pause = await graph.aget_state(config)
    confirmation = PendingConfirmation.model_validate(pause.interrupts[0].value)
    final = await graph.ainvoke(
        Command(
            resume={
                "confirmation_id": str(confirmation.confirmation_id),
                "approved": False,
            }
        ),
        config,
    )

    assert [task.status.value for task in final["tasks"]] == ["rejected", "rejected"]
    assert actions.records == {}


def test_blocked_tasks_are_rejected_transitively() -> None:
    tasks = [
        PlannedTask(
            id="task-1",
            title="失败任务",
            domain=AgentName.TRAVEL,
            objective="失败",
            status="failed",
        ),
        PlannedTask(
            id="task-2",
            title="二级依赖",
            domain=AgentName.EXPENSE,
            objective="等待 task-1",
            depends_on=["task-1"],
        ),
        PlannedTask(
            id="task-3",
            title="三级依赖",
            domain=AgentName.HR,
            objective="等待 task-2",
            depends_on=["task-2"],
        ),
    ]

    result = Workflow._reject_blocked_tasks(tasks)

    assert [item.status.value for item in result] == ["failed", "rejected", "rejected"]


@pytest.mark.asyncio
async def test_new_turn_clears_turn_scoped_artifacts_and_tool_results() -> None:
    workflow = Workflow(SupervisorAgent(StubPlanningService()))
    state = initial_state()
    state["artifacts"] = {"task-1": {"stale": True}}
    state["tool_results"] = [
        ToolResult(task_id="task-1", tool="old_tool", success=True)
    ]
    state["last_answer"] = "上一轮回答"

    update = await workflow.understand(state)  # type: ignore[arg-type]

    assert update["artifacts"] == {}
    assert update["tool_results"] == []
    assert update["last_answer"] == ""
