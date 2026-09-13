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
        self.seen_notices: list[list[str]] = []
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
        notices: Sequence[str] = (),
    ) -> AIMessage:
        del context, memories, recent_actions, user_name
        self.seen_notices.append(list(notices))
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
        self._seen.append((task_id, payload))
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
    target: str | None = None,
    domains: Sequence[AgentName] = (),
) -> ContextResolution:
    return ContextResolution(
        standalone_request=request,
        intent_summary=request,
        requires_task_planning=planning,
        turn_relation=relation,
        target_plan_id=target,
        domains=list(domains),
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
    assert [item.model_dump(exclude={"plan_id"}) for item in planning.seen_open_tasks[1]] == [
        OpenTask(
            plan_id="",
            task_id="task-1",
            title="查询差旅制度",
            domain=AgentName.TRAVEL,
            missing_fields=["end_date"],
        ).model_dump(exclude={"plan_id"})
    ]
    assert _statuses(second) == [
        ("task-1", TaskStatus.COMPLETED),
        ("task-2", TaskStatus.COMPLETED),
    ]
    # 界面上的目标仍是整件事的目标；本轮补充只交给领域 Agent。
    assert second["user_goal"] == "去上海出差并查考勤制度"
    # 续跑时领域 Agent 拿回了上一轮的草稿，任务结束后草稿随之清除。
    request, resumed = [
        (payload["standalone_request"], payload.get("previous_draft"))
        for task_id, payload in runtimes.seen
        if task_id == "task-1"
    ][-1]
    assert request == "去上海出差当天往返并查考勤制度"
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
async def test_new_request_while_waiting_shelves_the_plan() -> None:
    planning = ScriptedPlanning(
        [
            _resolution("去上海出差并查考勤制度"),
            _resolution("查询年假制度"),
        ]
    )
    graph, runtimes = _build(planning)

    first = await _turn(graph, "去上海出差，顺便查下考勤制度")
    second = await _turn(graph, "年假怎么规定的")

    assert planning.plan_calls == 2
    # Planner 惯用 task-1 这类短 id，新计划里的同名任务不能继承旧计划的草稿。
    assert runtimes.seen[-1][0] == "task-1"
    assert "previous_draft" not in runtimes.seen[-1][1]
    assert _statuses(second)[0] == ("task-1", TaskStatus.WAITING_INPUT)
    [shelved] = second["shelved_plans"]
    assert shelved.plan_id == first["plan_id"] != second["plan_id"]
    assert shelved.user_goal == "去上海出差并查考勤制度"
    assert "task-1" in shelved.drafts


@pytest.mark.asyncio
async def test_resuming_a_shelved_plan_swaps_it_back_in() -> None:
    planning = ScriptedPlanning(
        [
            _resolution("去上海出差并查考勤制度"),
            _resolution("查询年假制度"),
        ]
    )
    graph, runtimes = _build(planning)
    trip = await _turn(graph, "去上海出差，顺便查下考勤制度")
    leave = await _turn(graph, "年假怎么规定的")

    planning._resolutions.append(
        _resolution("继续上海出差申请", relation=TurnRelation.CONTINUE, target=trip["plan_id"])
    )
    resumed = await _turn(graph, "继续刚才的出差申请")

    assert planning.plan_calls == 2
    assert resumed["plan_id"] == trip["plan_id"]
    assert resumed["user_goal"] == "去上海出差并查考勤制度"
    assert _statuses(resumed) == [
        ("task-1", TaskStatus.COMPLETED),
        ("task-2", TaskStatus.COMPLETED),
    ]
    # 草稿随计划一起换回来；被换下去的年假计划还没办完，轮到它搁置。
    assert runtimes.seen[-2][1]["previous_draft"]["known_fields"][0]["value"] == "上海"
    [shelved] = resumed["shelved_plans"]
    assert shelved.plan_id == leave["plan_id"]


@pytest.mark.asyncio
async def test_continue_without_target_resumes_the_only_shelved_plan() -> None:
    planning = ScriptedPlanning(
        [
            _resolution("去上海出差并查考勤制度"),
            _resolution("查询考勤制度", domains=[AgentName.POLICY]),
            _resolution("继续上海出差申请", relation=TurnRelation.CONTINUE),
        ]
    )
    graph, _ = _build(planning)
    trip = await _turn(graph, "去上海出差，顺便查下考勤制度")
    await _turn(graph, "考勤怎么规定的")

    resumed = await _turn(graph, "继续刚才那个")

    assert resumed["plan_id"] == trip["plan_id"]
    assert resumed["shelved_plans"] == []
    assert _statuses(resumed)[0] == ("task-1", TaskStatus.COMPLETED)


@pytest.mark.asyncio
async def test_cancelling_the_current_plan_rejects_unfinished_tasks() -> None:
    planning = ScriptedPlanning(
        [
            _resolution("去上海出差并查考勤制度"),
            _resolution("不出差了", planning=False, relation=TurnRelation.CANCEL),
        ]
    )
    graph, _ = _build(planning)
    await _turn(graph, "去上海出差，顺便查下考勤制度")

    cancelled = await _turn(graph, "算了不出差了")

    assert planning.plan_calls == 1
    assert _statuses(cancelled) == [
        ("task-1", TaskStatus.REJECTED),
        ("task-2", TaskStatus.REJECTED),
    ]
    assert cancelled["drafts"] == {}
    assert planning.seen_notices == [["已放弃尚未提交的事项：查询差旅制度、查询通用制度"]]
    assert cancelled["last_answer"] == "不客气"


@pytest.mark.asyncio
async def test_cancelling_a_shelved_plan_drops_it() -> None:
    planning = ScriptedPlanning(
        [
            _resolution("去上海出差并查考勤制度"),
            _resolution("查询考勤制度", domains=[AgentName.POLICY]),
        ]
    )
    graph, _ = _build(planning)
    trip = await _turn(graph, "去上海出差，顺便查下考勤制度")
    policy = await _turn(graph, "考勤怎么规定的")
    planning._resolutions.append(
        _resolution(
            "不出差了", planning=False, relation=TurnRelation.CANCEL, target=trip["plan_id"]
        )
    )

    cancelled = await _turn(graph, "出差那个不办了")

    assert cancelled["shelved_plans"] == []
    # 当前计划是已经办完的考勤查询，不受影响。
    assert cancelled["plan_id"] == policy["plan_id"]
    assert _statuses(cancelled) == [("task-1", TaskStatus.COMPLETED)]
    assert len(planning.seen_notices[-1]) == 1


@pytest.mark.asyncio
async def test_cancel_with_nothing_unfinished_is_ignored() -> None:
    planning = ScriptedPlanning(
        [_resolution("撤销已提交的差旅申请", planning=False, relation=TurnRelation.CANCEL)]
    )
    graph, _ = _build(planning)

    state = await _turn(graph, "把刚才的差旅申请撤了")

    assert planning.seen_notices == [[]]
    assert state["last_answer"] == "不客气"


@pytest.mark.asyncio
async def test_single_domain_request_skips_the_planner() -> None:
    planning = ScriptedPlanning([_resolution("查询考勤制度", domains=[AgentName.POLICY])])
    graph, _ = _build(planning)

    state = await _turn(graph, "考勤怎么规定的")

    assert planning.plan_calls == 0
    assert [(task.domain, task.status) for task in state["tasks"]] == [
        (AgentName.POLICY, TaskStatus.COMPLETED)
    ]


@pytest.mark.asyncio
async def test_multi_domain_request_still_uses_the_planner() -> None:
    planning = ScriptedPlanning(
        [_resolution("出差并查考勤", domains=[AgentName.TRAVEL, AgentName.POLICY])]
    )
    graph, _ = _build(planning)

    await _turn(graph, "去上海出差，顺便查下考勤制度")

    assert planning.plan_calls == 1


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


class FailOnceRuntimeFactory(DraftAwareRuntimeFactory):
    """续跑时第一次调用模型抛错，模拟模型服务额度耗尽。"""

    def __init__(self, registry: DomainToolRegistry) -> None:
        super().__init__(registry)
        self.fail_next_resume = True

    def create(self, agent: AgentName, context: ToolContext) -> DraftAwareRuntime:
        runtime = super().create(agent, context)
        factory = self
        original = runtime.decide

        async def decide(
            task_objective: str, messages: list[BaseMessage], *, task_id: str
        ) -> AIMessage:
            payload = json.loads(str(messages[0].content))
            if "previous_draft" in payload and factory.fail_next_resume:
                factory.fail_next_resume = False
                raise RuntimeError("Error code: 403 - Free quota exhausted")
            return await original(task_objective, messages, task_id=task_id)

        runtime.decide = decide  # type: ignore[method-assign]
        return runtime


@pytest.mark.asyncio
async def test_a_resume_that_crashed_can_be_resumed_again() -> None:
    planning = ScriptedPlanning(
        [
            _resolution("去上海出差并查考勤制度"),
            _resolution("去上海出差单程", relation=TurnRelation.CONTINUE),
            _resolution("去上海出差单程", relation=TurnRelation.CONTINUE),
        ]
    )
    provider = LocalEnterpriseToolProvider(
        InMemoryActionRepository(), InMemoryPolicyRepository()
    )
    runtimes = FailOnceRuntimeFactory(DomainToolRegistry(provider))
    graph = build_graph(
        Workflow(SupervisorAgent(planning)), DomainTaskWorkflow(runtimes), InMemorySaver()
    )
    await _turn(graph, "去上海出差，顺便查下考勤制度")

    with pytest.raises(RuntimeError, match="403"):
        await _turn(graph, "单程")

    retried = await _turn(graph, "单程")

    # 重发时 Supervisor 仍然看得到那件待补充的事，于是续跑而不是重新规划。
    assert [item.task_id for item in planning.seen_open_tasks[2]] == ["task-1"]
    assert planning.plan_calls == 1
    assert _statuses(retried) == [
        ("task-1", TaskStatus.COMPLETED),
        ("task-2", TaskStatus.COMPLETED),
    ]


@pytest.mark.asyncio
async def test_first_attempt_crash_leaves_the_task_pending_not_waiting() -> None:
    tasks = [
        PlannedTask(
            id="task-1",
            title="差旅",
            domain=AgentName.TRAVEL,
            objective="差旅",
            status=TaskStatus.RUNNING,
        )
    ]

    [recovered] = Workflow._recover_interrupted(tasks, {})

    assert recovered.status == TaskStatus.PENDING
