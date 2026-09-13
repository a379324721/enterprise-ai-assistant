"""补充信息的轮次续跑原计划，不重新规划。"""

import json
from collections.abc import Sequence
from typing import Any
from uuid import UUID, uuid4

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from enterprise_ai_assistant.agents.supervisor import SupervisorAgent
from enterprise_ai_assistant.core.models import (
    AgentName,
    ContextResolution,
    OpenTask,
    PlannedTask,
    TaskPlan,
    TaskStatus,
    TurnRelation,
)
from enterprise_ai_assistant.graph.domain import DomainTaskWorkflow
from enterprise_ai_assistant.graph.workflow import Workflow, build_graph
from enterprise_ai_assistant.repositories.actions import InMemoryActionRepository
from enterprise_ai_assistant.repositories.policies import InMemoryPolicyRepository
from enterprise_ai_assistant.tools import LocalEnterpriseToolProvider, ToolContext
from enterprise_ai_assistant.tools.registry import DomainToolRegistry, RegisteredTool

CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000011")


class ScriptedPlanning:
    """按轮次依次返回预设的理解结果，并记录 Supervisor 看到的待补充任务。"""

    def __init__(self, resolutions: list[ContextResolution]) -> None:
        self._resolutions = list(resolutions)
        self.seen_open_tasks: list[list[OpenTask]] = []
        self.plan_calls = 0

    async def resolve_context(
        self,
        conversation: list[dict[str, str]],
        memory_keys: Sequence[str] = (),
        open_tasks: Sequence[OpenTask] = (),
    ) -> ContextResolution:
        del conversation, memory_keys
        self.seen_open_tasks.append(list(open_tasks))
        return self._resolutions.pop(0)

    async def plan(self, context: ContextResolution) -> TaskPlan:
        self.plan_calls += 1
        return TaskPlan(
            user_goal=context.standalone_request,
            tasks=[
                PlannedTask(
                    id="task-1",
                    title="查询差旅制度",
                    domain=AgentName.TRAVEL,
                    objective="查询出差相关制度",
                ),
                PlannedTask(
                    id="task-2",
                    title="查询通用制度",
                    domain=AgentName.POLICY,
                    objective="查询考勤制度",
                    depends_on=["task-1"],
                ),
            ],
        )

    async def respond_direct(
        self,
        context: ContextResolution,
        memories: Sequence[str] = (),
        recent_actions: Sequence[str] = (),
        user_name: str = "",
    ) -> AIMessage:
        del context, memories, recent_actions, user_name
        return AIMessage(content="不客气")


class DraftAwareRuntime:
    """travel 首次执行时追问；带着草稿续跑时改为查询制度。"""

    def __init__(self, name: AgentName, tools: list[RegisteredTool], seen: list[Any]) -> None:
        self.name = name
        self.tools = {item.tool.name: item for item in tools}
        self._seen = seen

    def tool(self, name: str) -> RegisteredTool:
        return self.tools[name]

    async def decide(
        self, task_objective: str, messages: list[BaseMessage], *, task_id: str
    ) -> AIMessage:
        del task_objective
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content="")
        payload = json.loads(str(messages[0].content))
        self._seen.append((task_id, payload.get("previous_draft")))
        if self.name == AgentName.TRAVEL and "previous_draft" not in payload:
            call = {
                "name": "request_information",
                "args": {
                    "missing_fields": ["end_date"],
                    "question": "预计哪天返回？",
                    "known_fields": [
                        {"name": "destination", "label": "目的地", "value": "上海"}
                    ],
                },
            }
        else:
            tool = "search_travel_policy" if self.name == AgentName.TRAVEL else (
                "search_general_policy"
            )
            call = {"name": tool, "args": {"query": "制度"}}
        return AIMessage(
            content="",
            tool_calls=[{**call, "id": f"{task_id}-{uuid4()}", "type": "tool_call"}],
        )

    async def respond(
        self, task_objective: str, messages: list[BaseMessage], *, task_id: str
    ) -> AIMessage:
        del task_objective, messages
        return AIMessage(content=f"{task_id} 的回答")

    async def invoke_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self.tools[name].tool.ainvoke(arguments)
        assert isinstance(result, dict)
        return result


class DraftAwareRuntimeFactory:
    def __init__(self, registry: DomainToolRegistry) -> None:
        self.registry = registry
        self.seen: list[Any] = []

    def create(self, agent: AgentName, context: ToolContext) -> DraftAwareRuntime:
        return DraftAwareRuntime(agent, self.registry.for_agent(agent, context), self.seen)


def _resolution(
    request: str,
    *,
    planning: bool = True,
    relation: TurnRelation = TurnRelation.NEW,
) -> ContextResolution:
    return ContextResolution(
        standalone_request=request,
        intent_summary=request,
        requires_task_planning=planning,
        turn_relation=relation,
    )


def _build(planning: ScriptedPlanning) -> tuple[Any, DraftAwareRuntimeFactory]:
    provider = LocalEnterpriseToolProvider(
        InMemoryActionRepository(), InMemoryPolicyRepository()
    )
    runtimes = DraftAwareRuntimeFactory(DomainToolRegistry(provider))
    graph = build_graph(
        Workflow(SupervisorAgent(planning)), DomainTaskWorkflow(runtimes), InMemorySaver()
    )
    return graph, runtimes


async def _turn(graph: Any, text: str) -> dict[str, Any]:
    state = {
        "messages": [HumanMessage(content=text)],
        "user_id": "u-1",
        "conversation_id": CONVERSATION_ID,
        "request_id": uuid4(),
    }
    result: dict[str, Any] = await graph.ainvoke(
        state, {"configurable": {"thread_id": str(CONVERSATION_ID)}}
    )
    return result


def _statuses(state: dict[str, Any]) -> list[tuple[str, TaskStatus]]:
    return [(task.id, task.status) for task in state["tasks"]]


@pytest.mark.asyncio
async def test_supplement_resumes_the_waiting_task_without_replanning() -> None:
    planning = ScriptedPlanning(
        [
            _resolution("去上海出差并查考勤制度"),
            _resolution(
                "去上海出差当天往返并查考勤制度", relation=TurnRelation.CONTINUE
            ),
        ]
    )
    graph, runtimes = _build(planning)

    first = await _turn(graph, "去上海出差，顺便查下考勤制度")

    assert _statuses(first) == [
        ("task-1", TaskStatus.WAITING_INPUT),
        ("task-2", TaskStatus.PENDING),
    ]
    assert first["drafts"]["task-1"].missing_fields == ["end_date"]

    second = await _turn(graph, "当天往返")

    assert planning.plan_calls == 1
    # Supervisor 只拿到标题和缺失字段名，拿不到字段值。
    assert planning.seen_open_tasks[1] == [
        OpenTask(
            task_id="task-1",
            title="查询差旅制度",
            domain=AgentName.TRAVEL,
            missing_fields=["end_date"],
        )
    ]
    assert _statuses(second) == [
        ("task-1", TaskStatus.COMPLETED),
        ("task-2", TaskStatus.COMPLETED),
    ]
    assert second["user_goal"] == "去上海出差当天往返并查考勤制度"
    # 续跑时领域 Agent 拿回了上一轮的草稿，任务结束后草稿随之清除。
    resumed = [draft for task_id, draft in runtimes.seen if task_id == "task-1"][-1]
    assert resumed["known_fields"][0]["value"] == "上海"
    assert second["drafts"] == {}


@pytest.mark.asyncio
async def test_small_talk_while_waiting_keeps_the_plan() -> None:
    planning = ScriptedPlanning(
        [
            _resolution("去上海出差并查考勤制度"),
            _resolution("用户道谢", planning=False),
            _resolution("去上海出差当天往返", relation=TurnRelation.CONTINUE),
        ]
    )
    graph, _ = _build(planning)

    await _turn(graph, "去上海出差，顺便查下考勤制度")
    chat = await _turn(graph, "好的谢谢")

    assert chat["last_answer"] == "不客气"
    assert _statuses(chat)[0] == ("task-1", TaskStatus.WAITING_INPUT)
    assert "task-1" in chat["drafts"]

    resumed = await _turn(graph, "当天往返")

    assert planning.plan_calls == 1
    assert _statuses(resumed)[0] == ("task-1", TaskStatus.COMPLETED)


@pytest.mark.asyncio
async def test_new_request_while_waiting_replaces_the_plan() -> None:
    planning = ScriptedPlanning(
        [
            _resolution("去上海出差并查考勤制度"),
            _resolution("查询年假制度"),
        ]
    )
    graph, runtimes = _build(planning)

    await _turn(graph, "去上海出差，顺便查下考勤制度")
    second = await _turn(graph, "年假怎么规定的")

    assert planning.plan_calls == 2
    # Planner 惯用 task-1 这类短 id，新计划里的同名任务不能继承旧计划的草稿。
    assert runtimes.seen[-1] == ("task-1", None)
    assert _statuses(second)[0] == ("task-1", TaskStatus.WAITING_INPUT)


@pytest.mark.asyncio
async def test_continue_without_a_waiting_task_falls_back_to_planning() -> None:
    planning = ScriptedPlanning(
        [_resolution("查询差旅制度", relation=TurnRelation.CONTINUE)]
    )
    graph, _ = _build(planning)

    state = await _turn(graph, "差旅制度")

    assert planning.plan_calls == 1
    assert planning.seen_open_tasks == [[]]
    assert _statuses(state)[0] == ("task-1", TaskStatus.WAITING_INPUT)
