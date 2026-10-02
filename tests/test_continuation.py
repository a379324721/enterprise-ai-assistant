"""补充信息的轮次续跑原计划，不重新规划。"""

import json
from collections.abc import Sequence
from typing import Any
from uuid import UUID, uuid4

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from enterprise_ai_assistant.agents.domain_runtime import DomainRuntimeProvider
from enterprise_ai_assistant.agents.supervisor import SupervisorAgent
from enterprise_ai_assistant.core.matters import OpenMatter, OpenMatterTask
from enterprise_ai_assistant.core.models import (
    AgentName,
    ContextResolution,
    PlannedTask,
    TaskOutline,
    TaskPlan,
    TaskRevision,
    TaskStatus,
    TurnRelation,
    recover_interrupted,
)
from enterprise_ai_assistant.graph.domain import DomainTaskWorkflow
from enterprise_ai_assistant.graph.workflow import (
    NOTHING_TO_CANCEL_REPLY,
    NOTHING_TO_REVISE_REPLY,
    Workflow,
    build_graph,
)
from enterprise_ai_assistant.repositories.actions import InMemoryActionRepository
from enterprise_ai_assistant.repositories.policies import InMemoryPolicyRepository
from enterprise_ai_assistant.tools import LocalEnterpriseToolProvider, ToolContext
from enterprise_ai_assistant.tools.registry import DomainToolRegistry, RegisteredTool

CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000011")


class ScriptedPlanning:
    """按轮次依次返回预设的理解结果，并记录 Supervisor 看到的待补充任务。"""

    def __init__(self, resolutions: list[ContextResolution]) -> None:
        self._resolutions = list(resolutions)
        self.seen_matters: list[list[OpenMatter]] = []
        self.plan_calls = 0

    async def resolve_context(
        self,
        conversation: list[dict[str, str]],
        memory_keys: Sequence[str] = (),
        matters: Sequence[OpenMatter] = (),
        recent_actions: Sequence[str] = (),
        user_name: str = "",
    ) -> ContextResolution:
        del conversation, memory_keys, recent_actions, user_name
        self.seen_matters.append(list(matters))
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


class DraftAwareRuntime:
    """travel 首次执行时追问；带着草稿续跑时改为查询制度。"""

    def __init__(self, name: AgentName, tools: list[RegisteredTool], seen: list[Any]) -> None:
        self.name = name
        self.tools = {item.tool.name: item for item in tools}
        self._seen = seen

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
            return AIMessage(content="")
        payload = json.loads(str(messages[0].content))
        self._seen.append((task_id, payload))
        if self.name == AgentName.TRAVEL and "previous_draft" not in payload:
            call = {
                "name": "request_information",
                "args": {
                    "missing_fields": ["end_date"],
                    "question": "预计哪天返回？",
                    "known_fields": [{"name": "destination", "label": "目的地", "value": "上海"}],
                },
            }
        else:
            tool = (
                "search_travel_policy"
                if self.name == AgentName.TRAVEL
                else ("search_general_policy")
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


class DraftAwareRuntimeFactory(DomainRuntimeProvider):
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
    revisions: Sequence[TaskRevision] = (),
    task_ids: Sequence[str] = (),
) -> ContextResolution:
    return ContextResolution(
        standalone_request=request,
        intent_summary=request,
        requires_task_planning=planning,
        turn_relation=relation,
        target_plan_id=target,
        tasks=[TaskOutline(title=request, domain=domain, objective=request) for domain in domains],
        revisions=list(revisions),
        target_task_ids=list(task_ids),
        reply=(
            ""
            if planning or relation in {TurnRelation.CANCEL, TurnRelation.REVISE}
            else "不客气"
        ),
    )


def _build(planning: ScriptedPlanning) -> tuple[Any, DraftAwareRuntimeFactory]:
    provider = LocalEnterpriseToolProvider(InMemoryActionRepository(), InMemoryPolicyRepository())
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
        state,
        {"configurable": {"thread_id": str(CONVERSATION_ID)}, "metadata": {"trace_id": "t"}},
    )
    return result


def _statuses(state: dict[str, Any]) -> list[tuple[str, TaskStatus]]:
    return [(task.id, task.status) for task in state["tasks"]]


@pytest.mark.asyncio
async def test_supplement_resumes_the_waiting_task_without_replanning() -> None:
    planning = ScriptedPlanning(
        [
            _resolution("去上海出差并查考勤制度"),
            _resolution("去上海出差当天往返并查考勤制度", relation=TurnRelation.CONTINUE),
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
    # Supervisor 看到的和右栏一样是整件事：排在后面的任务也在，免得用户提到它时认不出来。
    # 只有标题、状态和缺失字段名，拿不到字段值。
    [matter] = planning.seen_matters[1]
    assert matter.model_dump(exclude={"plan_id"}) == {
        "shelved": False,
        "tasks": [
            OpenMatterTask(
                task_id="task-1",
                title="查询差旅制度",
                domain=AgentName.TRAVEL,
                status=TaskStatus.WAITING_INPUT,
                missing_fields=["end_date"],
            ).model_dump(),
            OpenMatterTask(
                task_id="task-2",
                title="查询通用制度",
                domain=AgentName.POLICY,
                status=TaskStatus.PENDING,
            ).model_dump(),
        ],
    }
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
async def test_queued_tasks_keep_the_plan_goal_when_a_resumed_task_finishes() -> None:
    """续跑轮的改写只讲被续跑的那件事，排队的任务要拿计划的总目标。

    否则排队任务看到的"用户请求"里没有自己的事，会判断派错了领域而转交出去，
    接手的领域看任务目标又转回来——实测两个任务各来回一趟，一轮跑了十几分钟。
    """
    planning = ScriptedPlanning(
        [
            _resolution("去上海出差并查考勤制度"),
            _resolution("去上海出差当天往返", relation=TurnRelation.CONTINUE),
        ]
    )
    graph, runtimes = _build(planning)

    await _turn(graph, "去上海出差，顺便查下考勤制度")
    second = await _turn(graph, "当天往返")

    assert _statuses(second) == [
        ("task-1", TaskStatus.COMPLETED),
        ("task-2", TaskStatus.COMPLETED),
    ]
    requests = {task_id: payload["standalone_request"] for task_id, payload in runtimes.seen}
    # 被续跑的任务要本轮的补充，排队的任务要整件事的目标。
    assert requests["task-1"] == "去上海出差当天往返"
    assert requests["task-2"] == "去上海出差并查考勤制度"


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
    # 回复由运行时按实际放弃的事项写出，不用模型的话。
    assert cancelled["last_answer"] == "好的，这件事不办了，已放弃：查询差旅制度、查询通用制度。"


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
    assert cancelled["last_answer"].startswith("好的，这件事不办了")


@pytest.mark.asyncio
async def test_cancel_with_nothing_unfinished_is_ignored() -> None:
    planning = ScriptedPlanning(
        [_resolution("撤销已提交的差旅申请", planning=False, relation=TurnRelation.CANCEL)]
    )
    graph, _ = _build(planning)

    state = await _turn(graph, "把刚才的差旅申请撤了")

    # 指认不到任何未办完的事项：如实说没有可放弃的，不能让用户以为撤掉了。
    assert state["last_answer"] == NOTHING_TO_CANCEL_REPLY
    assert state["messages"][-1].content == NOTHING_TO_CANCEL_REPLY
    # 固定文案不是模型写的，不记 trace，界面上也就不给点赞点踩。
    assert "trace_id" not in state["messages"][-1].additional_kwargs


@pytest.mark.asyncio
async def test_tasks_from_the_supervisor_skip_the_planner() -> None:
    planning = ScriptedPlanning([_resolution("查询考勤制度", domains=[AgentName.POLICY])])
    graph, _ = _build(planning)

    state = await _turn(graph, "考勤怎么规定的")

    assert planning.plan_calls == 0
    assert [(task.domain, task.status) for task in state["tasks"]] == [
        (AgentName.POLICY, TaskStatus.COMPLETED)
    ]


@pytest.mark.asyncio
async def test_multi_domain_tasks_from_the_supervisor_skip_the_planner() -> None:
    planning = ScriptedPlanning(
        [_resolution("出差并查考勤", domains=[AgentName.TRAVEL, AgentName.POLICY])]
    )
    graph, _ = _build(planning)

    state = await _turn(graph, "去上海出差，顺便查下考勤制度")

    assert planning.plan_calls == 0
    assert [(task.id, task.domain) for task in state["tasks"]] == [
        ("task-1", AgentName.TRAVEL),
        ("task-2", AgentName.POLICY),
    ]


@pytest.mark.asyncio
async def test_business_request_without_tasks_falls_back_to_the_planner() -> None:
    planning = ScriptedPlanning([_resolution("查询差旅制度")])
    graph, _ = _build(planning)

    await _turn(graph, "差旅制度")

    assert planning.plan_calls == 1


@pytest.mark.asyncio
async def test_continue_without_a_waiting_task_falls_back_to_planning() -> None:
    planning = ScriptedPlanning([_resolution("查询差旅制度", relation=TurnRelation.CONTINUE)])
    graph, _ = _build(planning)

    state = await _turn(graph, "差旅制度")

    assert planning.plan_calls == 1
    assert planning.seen_matters == [[]]
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
            task_objective: str,
            messages: list[BaseMessage],
            *,
            task_id: str,
            answering: bool = False,
        ) -> AIMessage:
            payload = json.loads(str(messages[0].content))
            if "previous_draft" in payload and factory.fail_next_resume:
                factory.fail_next_resume = False
                raise RuntimeError("Error code: 403 - Free quota exhausted")
            return await original(task_objective, messages, task_id=task_id, answering=answering)

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
    provider = LocalEnterpriseToolProvider(InMemoryActionRepository(), InMemoryPolicyRepository())
    runtimes = FailOnceRuntimeFactory(DomainToolRegistry(provider))
    graph = build_graph(
        Workflow(SupervisorAgent(planning)), DomainTaskWorkflow(runtimes), InMemorySaver()
    )
    await _turn(graph, "去上海出差，顺便查下考勤制度")

    with pytest.raises(RuntimeError, match="403"):
        await _turn(graph, "单程")

    retried = await _turn(graph, "单程")

    # 重发时 Supervisor 仍然看得到那件待补充的事，于是续跑而不是重新规划。
    [matter] = planning.seen_matters[2]
    assert [(task.task_id, task.status) for task in matter.tasks] == [
        ("task-1", TaskStatus.WAITING_INPUT),
        ("task-2", TaskStatus.PENDING),
    ]
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

    [recovered] = recover_interrupted(tasks, {})

    assert recovered.status == TaskStatus.PENDING


@pytest.mark.asyncio
async def test_independent_tasks_wait_while_an_earlier_one_asks_for_input() -> None:
    # 默认串行：并行时两件事会在同一轮里一起追问，用户只答一件，没答的那件下一轮又被原样问一遍。
    planning = ScriptedPlanning(
        [
            _resolution("去上海出差，顺便查考勤制度", domains=[AgentName.TRAVEL, AgentName.POLICY]),
            _resolution("当天往返", relation=TurnRelation.CONTINUE),
        ]
    )
    graph, runtimes = _build(planning)

    first = await _turn(graph, "去上海出差，顺便查考勤制度")

    assert _statuses(first) == [
        ("task-1", TaskStatus.WAITING_INPUT),
        ("task-2", TaskStatus.PENDING),
    ]
    assert [task_id for task_id, _ in runtimes.seen] == ["task-1"]

    second = await _turn(graph, "当天往返")

    assert _statuses(second) == [
        ("task-1", TaskStatus.COMPLETED),
        ("task-2", TaskStatus.COMPLETED),
    ]
    assert [task_id for task_id, _ in runtimes.seen] == ["task-1", "task-1", "task-2"]


# -- 更正排队的任务、只放弃其中几个任务 ------------------------------------------

_OVERTIME = TaskRevision(task_id="task-2", title="查询加班制度", objective="查询加班相关制度")


def _payloads(runtimes: DraftAwareRuntimeFactory, task_id: str) -> list[dict[str, Any]]:
    return [payload for seen_id, payload in runtimes.seen if seen_id == task_id]


@pytest.mark.asyncio
async def test_revising_a_queued_task_changes_it_without_running_anything() -> None:
    planning = ScriptedPlanning(
        [
            _resolution("去上海出差并查考勤制度"),
            _resolution(
                "把考勤制度改成查加班制度",
                planning=False,
                relation=TurnRelation.REVISE,
                revisions=[_OVERTIME],
            ),
        ]
    )
    graph, runtimes = _build(planning)
    first = await _turn(graph, "去上海出差，顺便查下考勤制度")

    revised = await _turn(graph, "考勤那个改成查加班的")

    # 正在追问的差旅不续跑：用户没回答它，续跑只会把同一个问题再问一遍。
    assert _statuses(revised) == _statuses(first)
    assert [task_id for task_id, _ in runtimes.seen] == ["task-1"]
    assert planning.plan_calls == 1
    task = revised["tasks"][1]
    assert (task.title, task.objective) == ("查询加班制度", "查询加班相关制度")
    # 记下的是用户原话，不是改写。
    assert task.supplements == ["考勤那个改成查加班的"]
    assert revised["last_answer"] == "好的，已更新：查询加班制度。"
    assert "trace_id" not in revised["messages"][-1].additional_kwargs


@pytest.mark.asyncio
async def test_revised_task_hands_the_users_words_to_its_agent() -> None:
    planning = ScriptedPlanning(
        [
            _resolution("去上海出差并查考勤制度"),
            _resolution(
                "把考勤制度改成查加班制度",
                planning=False,
                relation=TurnRelation.REVISE,
                revisions=[_OVERTIME],
            ),
            _resolution("当天往返", relation=TurnRelation.CONTINUE),
        ]
    )
    graph, runtimes = _build(planning)
    await _turn(graph, "去上海出差，顺便查下考勤制度")
    await _turn(graph, "考勤那个改成查加班的")

    done = await _turn(graph, "当天往返")

    assert _statuses(done) == [("task-1", TaskStatus.COMPLETED), ("task-2", TaskStatus.COMPLETED)]
    [payload] = _payloads(runtimes, "task-2")
    assert payload["task_supplements"] == ["考勤那个改成查加班的"]
    assert payload["task"]["objective"] == "查询加班相关制度"
    # 只出现在单独的键里，不跟着任务再出现一遍。
    assert "supplements" not in payload["task"]
    # 没被更正过的任务不带这个键。
    assert all("task_supplements" not in item for item in _payloads(runtimes, "task-1"))


@pytest.mark.asyncio
async def test_answering_and_revising_in_one_turn_does_both() -> None:
    planning = ScriptedPlanning(
        [
            _resolution("去上海出差并查考勤制度"),
            _resolution(
                "当天往返，考勤制度改成查加班制度",
                relation=TurnRelation.CONTINUE,
                revisions=[_OVERTIME],
            ),
        ]
    )
    graph, runtimes = _build(planning)
    await _turn(graph, "去上海出差，顺便查下考勤制度")

    done = await _turn(graph, "当天往返，考勤那个改成查加班的")

    assert _statuses(done) == [("task-1", TaskStatus.COMPLETED), ("task-2", TaskStatus.COMPLETED)]
    assert done["tasks"][1].title == "查询加班制度"
    [payload] = _payloads(runtimes, "task-2")
    assert payload["task_supplements"] == ["当天往返，考勤那个改成查加班的"]


@pytest.mark.asyncio
async def test_revising_the_task_being_asked_about_resumes_it() -> None:
    """更正落在正在追问的任务上就是在补充它：只改不跑的话，这句话要等下一轮才被读到。"""
    planning = ScriptedPlanning(
        [
            _resolution("去上海出差并查考勤制度"),
            _resolution(
                "出差改成去杭州",
                planning=False,
                relation=TurnRelation.REVISE,
                revisions=[
                    TaskRevision(task_id="task-1", title="查询杭州差旅制度", objective="查询差旅制度")
                ],
            ),
        ]
    )
    graph, runtimes = _build(planning)
    await _turn(graph, "去上海出差，顺便查下考勤制度")

    resumed = await _turn(graph, "出差改成去杭州")

    assert _statuses(resumed)[0] == ("task-1", TaskStatus.COMPLETED)
    # 正在追问的任务标题不跟着改：它的补充经会话原文和草稿交给领域 Agent。
    assert resumed["tasks"][0].title == "查询差旅制度"
    assert [task_id for task_id, _ in runtimes.seen][:2] == ["task-1", "task-1"]


@pytest.mark.asyncio
async def test_revising_something_that_is_not_queued_says_so() -> None:
    planning = ScriptedPlanning(
        [
            _resolution("去上海出差并查考勤制度"),
            _resolution(
                "改一下",
                planning=False,
                relation=TurnRelation.REVISE,
                revisions=[TaskRevision(task_id="task-9", title="不存在", objective="不存在")],
            ),
        ]
    )
    graph, _ = _build(planning)
    first = await _turn(graph, "去上海出差，顺便查下考勤制度")

    state = await _turn(graph, "改一下")

    assert state["last_answer"] == NOTHING_TO_REVISE_REPLY
    assert state["tasks"] == first["tasks"]


@pytest.mark.asyncio
async def test_revise_with_nothing_open_says_so() -> None:
    planning = ScriptedPlanning(
        [
            _resolution(
                "会议室改成上午",
                planning=False,
                relation=TurnRelation.REVISE,
                revisions=[_OVERTIME],
            )
        ]
    )
    graph, _ = _build(planning)

    state = await _turn(graph, "会议室改成上午")

    assert state["last_answer"] == NOTHING_TO_REVISE_REPLY


@pytest.mark.asyncio
async def test_cancelling_one_queued_task_keeps_the_rest_waiting() -> None:
    planning = ScriptedPlanning(
        [
            _resolution("去上海出差并查考勤制度"),
            _resolution(
                "考勤制度不用查了",
                planning=False,
                relation=TurnRelation.CANCEL,
                task_ids=["task-2"],
            ),
        ]
    )
    graph, runtimes = _build(planning)
    await _turn(graph, "去上海出差，顺便查下考勤制度")

    state = await _turn(graph, "考勤制度不用查了")

    assert _statuses(state) == [
        ("task-1", TaskStatus.WAITING_INPUT),
        ("task-2", TaskStatus.REJECTED),
    ]
    # 还在追问的差旅草稿保留，用户回过头补充时接着用。
    assert state["drafts"]["task-1"].missing_fields == ["end_date"]
    assert state["last_answer"] == "好的，已放弃：查询通用制度。"
    assert [task_id for task_id, _ in runtimes.seen] == ["task-1"]


@pytest.mark.asyncio
async def test_cancelling_a_task_also_drops_the_ones_that_need_it() -> None:
    planning = ScriptedPlanning(
        [
            _resolution("去上海出差并查考勤制度"),
            _resolution(
                "出差不去了", planning=False, relation=TurnRelation.CANCEL, task_ids=["task-1"]
            ),
        ]
    )
    graph, _ = _build(planning)
    await _turn(graph, "去上海出差，顺便查下考勤制度")

    state = await _turn(graph, "出差不去了")

    # task-2 依赖 task-1，前置放弃了它也办不成。
    assert _statuses(state) == [
        ("task-1", TaskStatus.REJECTED),
        ("task-2", TaskStatus.REJECTED),
    ]
    assert state["drafts"] == {}
    assert state["last_answer"] == "好的，这件事不办了，已放弃：查询差旅制度、查询通用制度。"


@pytest.mark.asyncio
async def test_cancelling_the_asked_task_moves_on_to_independent_ones() -> None:
    """放弃正在追问的任务后，不依赖它的任务接着办：停下的话计划里没有等用户的任务，
    右栏不再显示它，也没有哪句话能再指认到它。"""
    planning = ScriptedPlanning(
        [
            _resolution("去上海出差，顺便查考勤制度", domains=[AgentName.TRAVEL, AgentName.POLICY]),
            _resolution(
                "出差不去了", planning=False, relation=TurnRelation.CANCEL, task_ids=["task-1"]
            ),
        ]
    )
    graph, runtimes = _build(planning)
    await _turn(graph, "去上海出差，顺便查考勤制度")

    state = await _turn(graph, "出差不去了")

    assert _statuses(state) == [
        ("task-1", TaskStatus.REJECTED),
        ("task-2", TaskStatus.COMPLETED),
    ]
    assert [task_id for task_id, _ in runtimes.seen] == ["task-1", "task-2"]
    cancelled = "好的，已放弃：去上海出差，顺便查考勤制度。"
    assert state["turn_answers"][0] == cancelled
    assert state["messages"][-2].content == cancelled


@pytest.mark.asyncio
async def test_cancelling_the_last_waiting_task_of_a_shelved_plan_drops_the_whole_plan() -> None:
    planning = ScriptedPlanning(
        [
            _resolution("去上海出差并查考勤制度"),
            _resolution("查询考勤制度", domains=[AgentName.POLICY]),
        ]
    )
    graph, _ = _build(planning)
    trip = await _turn(graph, "去上海出差，顺便查下考勤制度")
    await _turn(graph, "考勤怎么规定的")
    planning._resolutions.append(
        _resolution(
            "出差不去了",
            planning=False,
            relation=TurnRelation.CANCEL,
            target=trip["plan_id"],
            task_ids=["task-1"],
        )
    )

    state = await _turn(graph, "出差那个不去了")

    assert state["shelved_plans"] == []
    assert state["last_answer"] == "好的，这件事不办了，已放弃：查询差旅制度、查询通用制度。"


@pytest.mark.asyncio
async def test_cancelling_a_finished_task_is_not_possible() -> None:
    planning = ScriptedPlanning(
        [
            _resolution("去上海出差，顺便查考勤制度", domains=[AgentName.POLICY, AgentName.TRAVEL]),
        ]
    )
    graph, _ = _build(planning)
    first = await _turn(graph, "查考勤制度，顺便去上海出差")
    assert _statuses(first) == [
        ("task-1", TaskStatus.COMPLETED),
        ("task-2", TaskStatus.WAITING_INPUT),
    ]
    planning._resolutions.append(
        _resolution("考勤不查了", planning=False, relation=TurnRelation.CANCEL, task_ids=["task-1"])
    )

    state = await _turn(graph, "考勤不查了")

    assert state["last_answer"] == NOTHING_TO_CANCEL_REPLY
    assert _statuses(state) == _statuses(first)


@pytest.mark.asyncio
async def test_planner_output_cannot_carry_user_supplements() -> None:
    class SupplementingPlanning(ScriptedPlanning):
        async def plan(self, context: ContextResolution) -> TaskPlan:
            plan = await super().plan(context)
            plan.tasks[0].supplements = ["用户说过去杭州"]
            return plan

    graph, _ = _build(SupplementingPlanning([_resolution("去上海出差并查考勤制度")]))

    state = await _turn(graph, "去上海出差，顺便查下考勤制度")

    assert all(task.supplements == [] for task in state["tasks"])


@pytest.mark.asyncio
async def test_domain_agents_see_the_other_tasks_of_their_plan() -> None:
    """standalone_request 讲的是整件事。不告诉领域 Agent 别的部分有人办，它会替用户指路，
    说"会议室不归这边处理"。"""
    planning = ScriptedPlanning(
        [
            _resolution("去上海出差并查考勤制度"),
            _resolution("当天往返", relation=TurnRelation.CONTINUE),
        ]
    )
    graph, runtimes = _build(planning)
    await _turn(graph, "去上海出差，顺便查下考勤制度")
    await _turn(graph, "当天往返")

    first, resumed = _payloads(runtimes, "task-1")
    [policy] = _payloads(runtimes, "task-2")
    assert first["other_tasks"] == [{"title": "查询通用制度", "status": "pending"}]
    assert resumed["other_tasks"] == first["other_tasks"]
    # 只有标题和状态，不带对方的字段和产物。
    assert policy["other_tasks"] == [{"title": "查询差旅制度", "status": "completed"}]
