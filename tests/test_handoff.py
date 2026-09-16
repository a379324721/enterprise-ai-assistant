"""领域 Agent 发现任务分错时交还，父图改派给它指出的领域。"""

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from enterprise_ai_assistant.agents.domain_runtime import DomainRuntimeProvider
from enterprise_ai_assistant.agents.supervisor import SupervisorAgent
from enterprise_ai_assistant.api.routes import _execute_run
from enterprise_ai_assistant.core.models import (
    AgentName,
    ContextResolution,
    DomainTaskRequest,
    MemoryExtraction,
    OpenTask,
    PlannedTask,
    TaskOutline,
    TaskPlan,
    TaskStatus,
    ToolResult,
)
from enterprise_ai_assistant.core.runs import MemoryStreamBridge, Run, RunManager
from enterprise_ai_assistant.graph.domain import DomainTaskWorkflow
from enterprise_ai_assistant.graph.state import DomainTaskState
from enterprise_ai_assistant.graph.workflow import HANDOFF_EXHAUSTED_REPLY, Workflow, build_graph
from enterprise_ai_assistant.repositories.actions import InMemoryActionRepository
from enterprise_ai_assistant.repositories.policies import InMemoryPolicyRepository
from enterprise_ai_assistant.services.planning import PlanningService
from enterprise_ai_assistant.tools import LocalEnterpriseToolProvider, ToolContext
from enterprise_ai_assistant.tools.registry import DomainToolRegistry, RegisteredTool

CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000021")


class OnePlan(PlanningService):
    def __init__(self, tasks: list[TaskOutline]) -> None:
        self._tasks = tasks

    async def resolve_context(
        self,
        conversation: list[dict[str, str]],
        memory_keys: Sequence[str] = (),
        open_tasks: Sequence[OpenTask] = (),
        recent_actions: Sequence[str] = (),
        user_name: str = "",
    ) -> ContextResolution:
        del conversation, memory_keys, open_tasks, recent_actions, user_name
        return ContextResolution(
            standalone_request="报销上周打车费",
            intent_summary="报销打车费",
            requires_task_planning=True,
            tasks=self._tasks,
        )

    async def plan(self, context: ContextResolution) -> TaskPlan:
        raise AssertionError("tasks come from the supervisor")

    async def extract_memories(
        self, conversation: list[dict[str, str]], known: list[str]
    ) -> MemoryExtraction:
        return MemoryExtraction()


class HandoffRuntime:
    """按领域查表：表里有去处就转交，没有就查一次制度后作答。"""

    def __init__(
        self, name: AgentName, tools: list[RegisteredTool], routes: dict[AgentName, str]
    ) -> None:
        self.name = name
        self.tools = {item.tool.name: item for item in tools}
        self._routes = routes

    def tool(self, name: str) -> RegisteredTool:
        return self.tools[name]

    async def decide(
        self,
        task_objective: str,
        messages: list[BaseMessage],
        *,
        task_id: str,
        answering: bool = False,
    ) -> AIMessage:
        del task_objective, answering
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content=f"{self.name.value} 办好了")
        if self.name in self._routes:
            call = {
                "name": "handoff_task",
                "args": {"target_domain": self._routes[self.name], "reason": "不属于本领域"},
            }
        else:
            call = {"name": f"search_{self.name.value}_policy", "args": {"query": "制度"}}
        return AIMessage(
            content="", tool_calls=[{**call, "id": f"{task_id}-{uuid4()}", "type": "tool_call"}]
        )

    async def respond(
        self, task_objective: str, messages: list[BaseMessage], *, task_id: str
    ) -> AIMessage:
        raise AssertionError("not used")

    async def invoke_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self.tools[name].tool.ainvoke(arguments)
        assert isinstance(result, dict)
        return result


class HandoffRuntimeFactory(DomainRuntimeProvider):
    def __init__(self, routes: dict[AgentName, str]) -> None:
        provider = LocalEnterpriseToolProvider(
            InMemoryActionRepository(), InMemoryPolicyRepository()
        )
        self._registry = DomainToolRegistry(provider)
        self._routes = routes

    def create(self, agent: AgentName, context: ToolContext) -> HandoffRuntime:
        return HandoffRuntime(agent, self._registry.for_agent(agent, context), self._routes)


async def _run(tasks: list[TaskOutline], routes: dict[AgentName, str]) -> dict[str, Any]:
    graph = build_graph(
        Workflow(SupervisorAgent(OnePlan(tasks))),
        DomainTaskWorkflow(HandoffRuntimeFactory(routes)),
        InMemorySaver(),
    )
    result: dict[str, Any] = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="报销上周打车费 58 元")],
            "user_id": "u-1",
            "conversation_id": CONVERSATION_ID,
            "request_id": uuid4(),
        },
        {"configurable": {"thread_id": str(CONVERSATION_ID)}},
    )
    return result


@pytest.mark.asyncio
async def test_misrouted_task_is_rerouted_without_speaking_for_the_wrong_domain() -> None:
    state = await _run(
        [TaskOutline(title="报销打车费", domain=AgentName.TRAVEL, objective="报销打车费")],
        {AgentName.TRAVEL: "expense"},
    )

    [task] = state["tasks"]
    assert (task.id, task.domain, task.status) == ("task-1", AgentName.EXPENSE, TaskStatus.COMPLETED)
    assert task.handed_off_from == [AgentName.TRAVEL]
    # 分错的那个 Agent 一句话都没说，用户只看到接手方的回答。
    assert state["turn_answers"] == ["expense 办好了"]
    assert [item.tool for item in state["tool_results"]] == ["handoff_task", "search_expense_policy"]


@pytest.mark.asyncio
async def test_task_bounced_back_to_a_visited_domain_fails_with_a_fixed_reply() -> None:
    state = await _run(
        [
            TaskOutline(title="报销打车费", domain=AgentName.TRAVEL, objective="报销打车费"),
            TaskOutline(
                title="订会议室",
                domain=AgentName.MEETING,
                objective="订会议室",
                depends_on=[AgentName.TRAVEL],
            ),
        ],
        {AgentName.TRAVEL: "expense", AgentName.EXPENSE: "travel"},
    )

    assert [(task.domain, task.status) for task in state["tasks"]] == [
        (AgentName.EXPENSE, TaskStatus.FAILED),
        (AgentName.MEETING, TaskStatus.REJECTED),
    ]
    assert state["turn_answers"] == [HANDOFF_EXHAUSTED_REPLY]


def _domain_state(tool_results: list[ToolResult]) -> DomainTaskState:
    return {
        "domain_request": DomainTaskRequest(
            user_id="u-1",
            conversation_id=CONVERSATION_ID,
            request_id=uuid4(),
            user_goal="报销打车费",
            task=PlannedTask(
                id="task-1", title="报销", domain=AgentName.TRAVEL, objective="报销打车费"
            ),
        ),
        "domain_result": None,
        "domain_messages": [],
        "domain_iterations": 0,
        "domain_waiting_input": False,
        "domain_rejected": False,
        "domain_failed": False,
        "domain_retry_required": False,
        "domain_tool_executed": bool(tool_results),
        "domain_tool_results": tool_results,
    }


def test_handoff_to_the_current_domain_is_sent_back_to_the_model() -> None:
    workflow = DomainTaskWorkflow(HandoffRuntimeFactory({}))

    error = workflow._handoff_error(_domain_state([]), {"target_domain": "travel"})

    assert error is not None and "当前领域" in error


def test_handoff_after_a_write_is_refused() -> None:
    workflow = DomainTaskWorkflow(HandoffRuntimeFactory({}))
    written = ToolResult(task_id="task-1", tool="create_travel_application", success=True, data={})

    error = workflow._handoff_error(_domain_state([written]), {"target_domain": "expense"})

    assert error is not None and "写操作" in error


def test_handoff_tool_lists_only_the_other_domains() -> None:
    provider = LocalEnterpriseToolProvider(InMemoryActionRepository(), InMemoryPolicyRepository())
    context = ToolContext(
        user_id="u-1", conversation_id=CONVERSATION_ID, request_id=uuid4(), task_id="task-1"
    )
    [handoff] = [
        item
        for item in DomainToolRegistry(provider).for_agent(AgentName.TRAVEL, context)
        if item.tool.name == "handoff_task"
    ]

    assert "- expense：" in handoff.tool.description
    assert "- travel：" not in handoff.tool.description


@pytest.mark.asyncio
async def test_each_merged_task_announces_its_steps_but_a_handed_off_one_does_not() -> None:
    graph = build_graph(
        Workflow(
            SupervisorAgent(
                OnePlan(
                    [
                        TaskOutline(title="报销打车费", domain=AgentName.TRAVEL, objective="报销"),
                        TaskOutline(
                            title="查会议室制度",
                            domain=AgentName.MEETING,
                            objective="查制度",
                            depends_on=[AgentName.TRAVEL],
                        ),
                    ]
                )
            )
        ),
        DomainTaskWorkflow(HandoffRuntimeFactory({AgentName.TRAVEL: "expense"})),
        InMemorySaver(),
    )

    announced: list[tuple[str, list[str]]] = []
    async for chunk in graph.astream(
        {
            "messages": [HumanMessage(content="报销打车费，再查下会议室制度")],
            "user_id": "u-1",
            "conversation_id": CONVERSATION_ID,
            "request_id": uuid4(),
        },
        {"configurable": {"thread_id": "announce"}},
        stream_mode="custom",
    ):
        if "task_done" in chunk:
            announced.append((chunk["task_done"], [item["tool"] for item in chunk["tool_results"]]))

    # 转交那一次没有回答，不单独宣布；改派后在报销领域办完才宣布，随后才是依赖它的会议室任务。
    assert announced == [
        ("task-1", ["search_expense_policy"]),
        ("task-2", ["search_meeting_policy"]),
    ]


@pytest.mark.asyncio
async def test_side_panel_never_goes_blank_between_dependent_tasks() -> None:
    """真实图上跑一遍：右栏跟着根图检查点走，转交和两个任务交接之间卡片都不能消失。"""
    graph = build_graph(
        Workflow(
            SupervisorAgent(
                OnePlan(
                    [
                        TaskOutline(title="报销打车费", domain=AgentName.TRAVEL, objective="报销"),
                        TaskOutline(
                            title="查会议室制度",
                            domain=AgentName.MEETING,
                            objective="查制度",
                            depends_on=[AgentName.TRAVEL],
                        ),
                    ]
                )
            )
        ),
        DomainTaskWorkflow(HandoffRuntimeFactory({AgentName.TRAVEL: "expense"})),
        InMemorySaver(),
    )
    published: list[tuple[str, Any]] = []

    async def publish(event: str, data: Any) -> None:
        published.append((event, data))

    run = Run(run_id="r-1", conversation_id=CONVERSATION_ID, user_id="u-1")
    app = SimpleNamespace(
        state=SimpleNamespace(
            graph=graph,
            logger=SimpleNamespace(info=print, warning=print, exception=print),
            runs=RunManager(MemoryStreamBridge(), SimpleNamespace()),
        )
    )
    await _execute_run(
        app,
        run,
        publish,
        {
            "messages": [HumanMessage(content="报销打车费，再查下会议室制度")],
            "user_id": "u-1",
            "conversation_id": CONVERSATION_ID,
            "request_id": uuid4(),
        },
        CONVERSATION_ID,
        "u-1",
    )

    cards = [
        [item["task_id"] for item in data["matters"]] for event, data in published if event == "matters"
    ]
    # 第一份是计划还没建出来时的空右栏；从计划出现到全部办完，中间每一份都有卡片。
    assert cards[0] == []
    assert cards[-1] == []
    working = cards[1:-1]
    assert working and all(working)
    assert working[0] == ["task-1"] and working[-1] == ["task-2"]
    done = next(data for event, data in published if event == "done")
    assert done["matters"] == []
