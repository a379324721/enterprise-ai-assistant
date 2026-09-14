from collections.abc import Sequence
from typing import Any
from uuid import UUID

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from enterprise_ai_assistant.agents.supervisor import SupervisorAgent
from enterprise_ai_assistant.api import routes
from enterprise_ai_assistant.api.schemas import ConfirmationRequest
from enterprise_ai_assistant.core.models import (
    AgentName,
    ContextResolution,
    DomainTaskRequest,
    OpenTask,
    PendingConfirmation,
    PlannedTask,
    TaskPlan,
    ToolResult,
)
from enterprise_ai_assistant.graph.domain import DomainTaskWorkflow, build_domain_graph
from enterprise_ai_assistant.graph.state import DomainTaskState
from enterprise_ai_assistant.graph.workflow import (
    DIGEST_HEADER,
    Workflow,
    build_graph,
    reply_message,
)
from enterprise_ai_assistant.repositories.actions import InMemoryActionRepository
from enterprise_ai_assistant.repositories.policies import InMemoryPolicyRepository
from enterprise_ai_assistant.tools import LocalEnterpriseToolProvider, ToolContext
from enterprise_ai_assistant.tools.registry import DomainToolRegistry, RegisteredTool


class StubPlanningService:
    async def resolve_context(
        self,
        conversation: list[dict[str, str]],
        memory_keys: Sequence[str] = (),
        open_tasks: Sequence[OpenTask] = (),
        recent_actions: Sequence[str] = (),
        user_name: str = "",
    ) -> ContextResolution:
        assert "上海" in conversation[-1]["content"]
        return ContextResolution(
            standalone_request="2026-08-10 至 2026-08-14 去上海客户交流，创建差旅申请并报销已购高铁票 480 元",
            intent_summary="申请差旅并报销已购票款",
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
                    title="报销已购高铁票",
                    domain=AgentName.EXPENSE,
                    objective="创建关联本次差旅的报销单",
                    depends_on=["task-1"],
                    success_criteria=["返回报销单号"],
                ),
            ],
        )


class DirectPlanningService:
    async def resolve_context(
        self,
        conversation: list[dict[str, str]],
        memory_keys: Sequence[str] = (),
        open_tasks: Sequence[OpenTask] = (),
        recent_actions: Sequence[str] = (),
        user_name: str = "",
    ) -> ContextResolution:
        assert conversation[-1]["content"] == "你好"
        return ContextResolution(
            standalone_request="你好",
            intent_summary="用户向助手打招呼",
            requires_task_planning=False,
            reply="你好！有什么企业事务需要我协助？",
        )

    async def plan(self, context: ContextResolution) -> TaskPlan:
        raise AssertionError(f"direct conversation must not be planned: {context}")


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
        answering: bool = False,
    ) -> AIMessage:
        del task_objective, answering
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content="")
        if self.name == AgentName.TRAVEL:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "create_travel_application",
                        "args": {
                            "origin": "杭州",
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
                    "name": "create_expense_claim",
                    "args": {
                        "expense_type": "交通",
                        "amount": "480.00",
                        "receipt_refs": ["INV-20260810"],
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
        answering: bool = False,
    ) -> AIMessage:
        del task_objective, answering
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
        "messages": [HumanMessage(content="下周去上海出差帮我申请，高铁票 480 已经买了一起报了")],
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

    # 回复就是 Supervisor 写在理解结果里的那句，本轮不再调用任何别的模型。
    assert final["last_answer"] == "你好！有什么企业事务需要我协助？"
    assert final["messages"][-1].content == "你好！有什么企业事务需要我协助？"
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
        "domain_tool_results": [],
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
    assert final["domain_tool_results"][0].tool == "search_general_policy"


class AnsweringRuntime(ScriptedRuntime):
    """先调一次工具，看到结果后在同一个决策调用里直接作答；兜底回答调用不该被用到。"""

    first_call: dict[str, Any] = {}
    answering_flags: list[bool] = []

    async def decide(
        self,
        task_objective: str,
        messages: list[BaseMessage],
        *,
        task_id: str,
        answering: bool = False,
    ) -> AIMessage:
        del task_objective
        type(self).answering_flags.append(answering)
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content="差旅制度里没有查到相关条款，建议咨询行政部。")
        return AIMessage(
            content="",
            tool_calls=[{**type(self).first_call, "id": f"{task_id}-call", "type": "tool_call"}],
        )

    async def respond(
        self, task_objective: str, messages: list[BaseMessage], *, task_id: str
    ) -> AIMessage:
        raise AssertionError("回答应当由决策调用直接给出")


class AnsweringRuntimeFactory(ScriptedRuntimeFactory):
    def create(self, agent: AgentName, context: ToolContext) -> AnsweringRuntime:
        return AnsweringRuntime(agent, self.registry.for_agent(agent, context))


def _policy_request() -> dict[str, Any]:
    return {
        "domain_request": DomainTaskRequest(
            user_id="u-1",
            conversation_id=UUID("00000000-0000-0000-0000-000000000001"),
            request_id=UUID("00000000-0000-0000-0000-000000000002"),
            user_goal="查询差旅制度",
            task=PlannedTask(
                id="task-1", title="查询制度", domain=AgentName.POLICY, objective="查询差旅制度"
            ),
        ),
        "domain_result": None,
    }


@pytest.mark.asyncio
async def test_decision_that_reads_tool_results_answers_without_another_model_call() -> None:
    provider = LocalEnterpriseToolProvider(InMemoryActionRepository(), InMemoryPolicyRepository())
    AnsweringRuntime.first_call = {"name": "search_general_policy", "args": {"query": "差旅"}}
    AnsweringRuntime.answering_flags = []
    workflow = DomainTaskWorkflow(AnsweringRuntimeFactory(DomainToolRegistry(provider)))

    final = await build_domain_graph(workflow).ainvoke(_policy_request())

    result = final["domain_result"]
    assert result.status.value == "completed"
    assert result.answer == "差旅制度里没有查到相关条款，建议咨询行政部。"
    # 只有看过工具结果的那次调用标为对用户可见，第一次决策的输出不会被流出去。
    assert AnsweringRuntime.answering_flags == [False, True]


@pytest.mark.asyncio
async def test_information_request_sends_its_question_without_a_model_call() -> None:
    provider = LocalEnterpriseToolProvider(InMemoryActionRepository(), InMemoryPolicyRepository())
    AnsweringRuntime.first_call = {
        "name": "request_information",
        "args": {"missing_fields": ["结束日期"], "question": "预计哪天返回？"},
    }
    AnsweringRuntime.answering_flags = []
    workflow = DomainTaskWorkflow(AnsweringRuntimeFactory(DomainToolRegistry(provider)))

    custom: list[Any] = []
    final: dict[str, Any] = {}
    async for mode, data in build_domain_graph(workflow).astream(
        _policy_request(), stream_mode=["custom", "values"]
    ):
        if mode == "custom":
            custom.append(data)
        else:
            final = data

    result = final["domain_result"]
    assert result.status.value == "waiting_input"
    assert result.answer == "预计哪天返回？"
    assert AnsweringRuntime.answering_flags == [False]
    # 问题没经过模型，SSE 的 messages 流里没有它，得靠 custom 事件送到前端。
    assert custom == [{"answer": "预计哪天返回？", "agent": "policy", "task_id": "task-1"}]


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
    assert second_confirmation.action == "create_expense_claim"

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
    travel = final["artifacts"]["task-1"]["create_travel_application"]
    claim = final["artifacts"]["task-2"]["create_expense_claim"]
    assert travel["data"]["destination"] == "上海"
    assert claim["data"]["travel_reference"] == "travel-reference-1"
    assert len(final["tool_results"]) == 2
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


class LeavePlanningService:
    async def resolve_context(
        self,
        conversation: list[dict[str, str]],
        memory_keys: Sequence[str] = (),
        open_tasks: Sequence[OpenTask] = (),
        recent_actions: Sequence[str] = (),
        user_name: str = "",
    ) -> ContextResolution:
        return ContextResolution(
            standalone_request="查询剩余年假，并请 2026-09-18 一天年假",
            intent_summary="查年假余额并请假",
            requires_task_planning=True,
            tasks=[
                {"title": "查询年假余额并提交请假申请", "domain": "hr", "objective": "查余额并请假"}
            ],
        )

    async def plan(self, context: ContextResolution) -> TaskPlan:
        raise AssertionError("Supervisor 已经给出任务")


class NarratingLeaveRuntime(ScriptedRuntime):
    """查完余额后在写工具的 message_to_user 里报出余额，确认提交后只说提交结果。"""

    #: 写回答那次决策看到的消息，用来确认它知道自己报过余额。
    answering_context: list[BaseMessage] = []

    async def decide(
        self,
        task_objective: str,
        messages: list[BaseMessage],
        *,
        task_id: str,
        answering: bool = False,
    ) -> AIMessage:
        del task_objective, answering
        called = [message.name for message in messages if isinstance(message, ToolMessage)]
        if not called:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_leave_balance",
                        "args": {"leave_type": "annual"},
                        "id": f"{task_id}-balance",
                        "type": "tool_call",
                    }
                ],
            )
        if called == ["get_leave_balance"]:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_leave_request",
                        "args": {
                            "message_to_user": "你的年假还剩 8 天。",
                            "leave_type": "annual",
                            "start_date": "2026-09-18",
                            "end_date": "2026-09-18",
                        },
                        "id": f"{task_id}-submit",
                        "type": "tool_call",
                    }
                ],
            )
        type(self).answering_context = list(messages)
        return AIMessage(content="请假申请已提交。")


class NarratingLeaveRuntimeFactory(ScriptedRuntimeFactory):
    def create(self, agent: AgentName, context: ToolContext) -> NarratingLeaveRuntime:
        return NarratingLeaveRuntime(agent, self.registry.for_agent(agent, context))


@pytest.mark.asyncio
async def test_what_the_agent_says_before_a_confirmation_lands_in_the_conversation() -> None:
    provider = LocalEnterpriseToolProvider(InMemoryActionRepository(), InMemoryPolicyRepository())
    graph = build_graph(
        Workflow(SupervisorAgent(LeavePlanningService())),
        DomainTaskWorkflow(NarratingLeaveRuntimeFactory(DomainToolRegistry(provider))),
        InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "thread-leave"}}
    state = initial_state()
    state["messages"] = [HumanMessage(content="我还有多少年假？下周五请一天年假")]

    custom = [
        data
        async for _, mode, data in graph.astream(
            state, config, stream_mode=["custom"], subgraphs=True
        )
        if mode == "custom"
    ]
    # 参数里的话不在 messages 流里，要整段推给前端，排在确认卡之前出现在聊天气泡里。
    assert {"answer": "你的年假还剩 8 天。", "agent": "hr", "task_id": "task-1"} in custom
    snapshot = await graph.aget_state(config)
    found = routes._pending_interrupt(snapshot)
    assert found is not None
    interrupt, pending = found
    # 停在卡上时这句话只能随中断带出来；确认卡和业务参数里都没有它。
    assert [(note.text, note.tools) for note in pending.notes] == [
        ("你的年假还剩 8 天。", ["get_leave_balance"])
    ]
    assert "message_to_user" not in pending.payload
    assert "message_to_user" not in [field.name for field in pending.fields]

    command = routes._resume_command(
        ConfirmationRequest(confirmation_id=pending.confirmation_id, approved=True),
        interrupt,
        pending,
    )
    final = await graph.ainvoke(command, config)

    # 顺序和用户看到的一致：余额、确认决定、提交结果；余额只记一次。
    assert [
        (message.type, message.content, message.additional_kwargs.get("tools_called"))
        for message in final["messages"][1:]
    ] == [
        ("ai", "你的年假还剩 8 天。", ["get_leave_balance"]),
        ("system", "你确认了：提交请假申请", None),
        ("ai", "请假申请已提交。", ["get_leave_balance", "submit_leave_request"]),
    ]
    # 写回答时模型看得到自己那次调用里说过的话，不会再报一遍余额。
    assert any(
        isinstance(message, AIMessage)
        and any(
            call["args"].get("message_to_user") == "你的年假还剩 8 天。"
            for call in message.tool_calls
        )
        for message in NarratingLeaveRuntime.answering_context
    )


class NarratingPolicyRuntime(ScriptedRuntime):
    """查了一次没查到，先说一句再换个关键词查，最后作答。"""

    async def decide(
        self,
        task_objective: str,
        messages: list[BaseMessage],
        *,
        task_id: str,
        answering: bool = False,
    ) -> AIMessage:
        del task_objective, answering
        searched = sum(isinstance(message, ToolMessage) for message in messages)
        if searched == 2:
            return AIMessage(content="差旅制度里也没有，建议咨询行政部。")
        return AIMessage(
            content="通用制度里没查到，再查一下差旅制度。" if searched else "",
            tool_calls=[
                {
                    "name": "search_general_policy",
                    "args": {"query": "差旅" if searched else "出差", "limit": 1},
                    "id": f"{task_id}-search-{searched}",
                    "type": "tool_call",
                }
            ],
        )


class NarratingPolicyRuntimeFactory(ScriptedRuntimeFactory):
    def create(self, agent: AgentName, context: ToolContext) -> NarratingPolicyRuntime:
        return NarratingPolicyRuntime(agent, self.registry.for_agent(agent, context))


class ScriptedLeaveRuntime(ScriptedRuntime):
    """按类属性给出的两次决策依次返回，确认后作答。"""

    decisions: list[AIMessage] = []

    async def decide(
        self,
        task_objective: str,
        messages: list[BaseMessage],
        *,
        task_id: str,
        answering: bool = False,
    ) -> AIMessage:
        del task_objective, task_id, answering
        called = sum(isinstance(message, ToolMessage) for message in messages)
        decisions = type(self).decisions
        return decisions[called] if called < len(decisions) else AIMessage(content="已提交。")


class ScriptedLeaveRuntimeFactory(ScriptedRuntimeFactory):
    def create(self, agent: AgentName, context: ToolContext) -> ScriptedLeaveRuntime:
        return ScriptedLeaveRuntime(agent, self.registry.for_agent(agent, context))


def _call(name: str, **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": name, "type": "tool_call"}])


_LEAVE = {"leave_type": "annual", "start_date": "2026-09-18", "end_date": "2026-09-18"}


async def _pause_on_leave_card(
    decisions: list[AIMessage],
) -> tuple[PendingConfirmation, list[Any]]:
    ScriptedLeaveRuntime.decisions = decisions
    provider = LocalEnterpriseToolProvider(InMemoryActionRepository(), InMemoryPolicyRepository())
    graph = build_graph(
        Workflow(SupervisorAgent(LeavePlanningService())),
        DomainTaskWorkflow(ScriptedLeaveRuntimeFactory(DomainToolRegistry(provider))),
        InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "thread-scripted-leave"}}
    state = initial_state()
    state["messages"] = [HumanMessage(content="下周五请一天年假")]
    custom = [
        data
        async for _, mode, data in graph.astream(
            state, config, stream_mode=["custom"], subgraphs=True
        )
        if mode == "custom"
    ]
    found = routes._pending_interrupt(await graph.aget_state(config))
    assert found is not None
    return found[1], custom


@pytest.mark.asyncio
async def test_nothing_is_said_when_submitting_without_looking_anything_up() -> None:
    # 直接提交时模型手里没有任何查询结果，参数里就算写了话也不是查来的，不对外。
    pending, custom = await _pause_on_leave_card(
        [_call("submit_leave_request", message_to_user="好的，这就帮你提交。", **_LEAVE)]
    )

    assert pending.notes == []
    assert custom == []


@pytest.mark.asyncio
async def test_what_was_already_said_in_the_text_is_not_repeated_from_the_arguments() -> None:
    submit = _call("submit_leave_request", message_to_user="你的年假还剩 8 天。", **_LEAVE)
    # 正文已经逐字流出去了，参数里同样意思的话不再推一遍、也不再记一遍。
    submit.content = "年假余额 8 天。"
    pending, custom = await _pause_on_leave_card(
        [_call("get_leave_balance", leave_type="annual"), submit]
    )

    assert [note.text for note in pending.notes] == ["年假余额 8 天。"]
    assert not any("answer" in item for item in custom)


@pytest.mark.asyncio
async def test_an_empty_message_to_user_says_nothing() -> None:
    pending, custom = await _pause_on_leave_card(
        [
            _call("get_leave_balance", leave_type="annual"),
            _call("submit_leave_request", message_to_user="", **_LEAVE),
        ]
    )

    assert pending.notes == []
    assert not any("answer" in item for item in custom)


@pytest.mark.asyncio
async def test_what_the_agent_says_between_read_tools_goes_back_with_the_result() -> None:
    provider = LocalEnterpriseToolProvider(InMemoryActionRepository(), InMemoryPolicyRepository())
    workflow = DomainTaskWorkflow(NarratingPolicyRuntimeFactory(DomainToolRegistry(provider)))

    final = await build_domain_graph(workflow).ainvoke(_policy_request())

    result = final["domain_result"]
    assert [(note.text, note.tools) for note in result.notes] == [
        ("通用制度里没查到，再查一下差旅制度。", ["search_general_policy"])
    ]
    assert result.answer == "差旅制度里也没有，建议咨询行政部。"


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


@pytest.mark.asyncio
async def test_multiple_writes_in_one_task_accumulate_artifacts() -> None:
    """同一任务里连续调用两个写工具时，先前的业务单号不能被后一次产出覆盖。"""
    provider = LocalEnterpriseToolProvider(
        InMemoryActionRepository(), InMemoryPolicyRepository()
    )
    workflow = DomainTaskWorkflow(ScriptedRuntimeFactory(DomainToolRegistry(provider)))
    task = PlannedTask(
        id="task-1",
        title="报销并设置提醒",
        domain=AgentName.EXPENSE,
        objective="创建费用报销单并设置提醒",
    )
    state: DomainTaskState = {
        "domain_request": DomainTaskRequest(
            user_id="u-1",
            conversation_id=UUID("00000000-0000-0000-0000-000000000001"),
            request_id=UUID("00000000-0000-0000-0000-000000000002"),
            user_goal="报销打车费并提醒我",
            task=task,
        ),
        "domain_result": None,
        "domain_messages": [],
        "domain_iterations": 1,
        "domain_waiting_input": False,
        "domain_rejected": False,
        "domain_failed": False,
        "domain_retry_required": False,
        "domain_tool_executed": False,
        "domain_tool_results": [],
        "pending_tool_call": {
            "name": "create_expense_claim",
            "args": {
                "expense_type": "打车",
                "amount": "128.50",
                "receipt_refs": ["receipt-1"],
            },
            "id": "call-claim",
        },
    }

    first = await workflow.execute_tool(state)
    state = {**state, **first, "pending_tool_call": {
        "name": "search_expense_policy",
        "args": {"query": "打车费报销标准", "limit": 1},
        "id": "call-policy",
    }}
    second = await workflow.execute_tool(state)

    artifact = second["artifact"]
    assert set(artifact) == {"create_expense_claim", "search_expense_policy"}
    assert artifact["create_expense_claim"]["data"]["expense_type"] == "打车"


@pytest.mark.asyncio
async def test_failed_tool_does_not_overwrite_successful_artifact() -> None:
    provider = LocalEnterpriseToolProvider(
        InMemoryActionRepository(), InMemoryPolicyRepository()
    )
    workflow = DomainTaskWorkflow(ScriptedRuntimeFactory(DomainToolRegistry(provider)))
    task = PlannedTask(
        id="task-1",
        title="报销",
        domain=AgentName.EXPENSE,
        objective="创建费用报销单",
    )
    state: DomainTaskState = {
        "domain_request": DomainTaskRequest(
            user_id="u-1",
            conversation_id=UUID("00000000-0000-0000-0000-000000000001"),
            request_id=UUID("00000000-0000-0000-0000-000000000002"),
            user_goal="报销打车费",
            task=task,
        ),
        "domain_result": None,
        "domain_messages": [],
        "domain_iterations": 1,
        "domain_waiting_input": False,
        "domain_rejected": False,
        "domain_failed": False,
        "domain_retry_required": False,
        "domain_tool_executed": True,
        "domain_tool_results": [],
        "artifact": {"create_expense_claim": {"tool": "create_expense_claim", "success": True}},
        "pending_tool_call": {
            "name": "search_expense_policy",
            "args": {"query": "", "limit": 1},
            "id": "call-bad",
        },
    }

    update = await workflow.execute_tool(state)

    assert update["domain_failed"] is True
    assert update["artifact"] == {
        "create_expense_claim": {"tool": "create_expense_claim", "success": True}
    }


class RecordingPlanningService:
    """记录每次送进 Context Supervisor 的会话，用于断言 prompt 输入被裁剪。"""

    def __init__(self) -> None:
        self.seen: list[list[dict[str, str]]] = []

    async def resolve_context(
        self,
        conversation: list[dict[str, str]],
        memory_keys: Sequence[str] = (),
        open_tasks: Sequence[OpenTask] = (),
        recent_actions: Sequence[str] = (),
        user_name: str = "",
    ) -> ContextResolution:
        self.seen.append(conversation)
        return ContextResolution(
            standalone_request=f"第{len(self.seen)}轮独立请求",
            intent_summary="记录会话输入",
            requires_task_planning=False,
            reply="好的",
        )

    async def plan(self, context: ContextResolution) -> TaskPlan:
        raise AssertionError(f"not used: {context}")


def _history(turns: int) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for index in range(1, turns + 1):
        messages.append(HumanMessage(content=f"问题{index}"))
        messages.append(AIMessage(content=f"回答{index}"))
    return messages


def _context_state(messages: list[BaseMessage], digest: list[str]) -> Any:
    state = initial_state()
    state["messages"] = messages
    state["history_digest"] = digest
    return state


def test_short_history_is_sent_verbatim() -> None:
    workflow = Workflow(
        SupervisorAgent(RecordingPlanningService()), history_window=12, digest_turns=20
    )

    conversation = workflow._conversation(_context_state(_history(3), ["轮1", "轮2", "轮3"]))

    assert [turn["content"] for turn in conversation] == [
        "问题1", "回答1", "问题2", "回答2", "问题3", "回答3",
    ]
    assert not any(DIGEST_HEADER in turn["content"] for turn in conversation)


def test_assistant_turns_tell_the_model_whether_a_tool_was_called() -> None:
    """Supervisor 分不清哪条回复查过、哪条是凭清单直接答的，被追问时就会谎称查过。"""
    workflow = Workflow(SupervisorAgent(RecordingPlanningService()))
    messages: list[BaseMessage] = [
        HumanMessage(content="我提交过哪些单子"),
        reply_message("您最近提交过差旅申请。", []),
        HumanMessage(content="你再查一下"),
        reply_message("查到一张差旅申请。", ["query_travel_applications"]),
        # 标注上线前写进检查点的旧消息，来源未知，不能冒充成"没调用工具"。
        AIMessage(content="旧回答"),
    ]

    conversation = workflow._conversation(_context_state(messages, []))

    assert [turn["content"] for turn in conversation[1::2]] == [
        "[未调用工具]\n您最近提交过差旅申请。",
        "[调用了：查询差旅申请]\n查到一张差旅申请。",
    ]
    assert conversation[-1]["content"] == "旧回答"


def test_long_history_is_windowed_with_digest_of_dropped_turns() -> None:
    workflow = Workflow(
        SupervisorAgent(RecordingPlanningService()), history_window=4, digest_turns=20
    )
    digest = [f"轮{index}摘要" for index in range(1, 6)]

    conversation = workflow._conversation(_context_state(_history(5), digest))

    assert len(conversation) == 5
    assert [turn["content"] for turn in conversation[1:]] == [
        "问题4", "回答4", "问题5", "回答5",
    ]
    summary = conversation[0]["content"]
    assert summary.startswith(DIGEST_HEADER)
    assert "轮1摘要" in summary and "轮3摘要" in summary
    # 轮 4、5 的原文还在窗口里，不该重复出现在摘要中。
    assert "轮4摘要" not in summary and "轮5摘要" not in summary


def test_digest_alignment_is_stable_across_understand_and_respond() -> None:
    """understand 时当轮摘要尚未入 digest，select_task 时已入，两者不能错位。"""
    workflow = Workflow(
        SupervisorAgent(RecordingPlanningService()), history_window=4, digest_turns=20
    )
    messages = [*_history(4), HumanMessage(content="问题5")]
    before = [f"轮{index}摘要" for index in range(1, 5)]
    after = [*before, "轮5摘要"]

    at_understand = workflow._conversation(_context_state(messages, before))
    at_respond = workflow._conversation(_context_state(messages, after))

    assert at_understand == at_respond
    assert at_understand[0]["content"].endswith("轮3摘要")


@pytest.mark.asyncio
async def test_understand_appends_and_caps_digest() -> None:
    workflow = Workflow(
        SupervisorAgent(RecordingPlanningService()), history_window=4, digest_turns=3
    )
    state = _context_state([HumanMessage(content="问题1")], ["轮1", "轮2", "轮3"])

    update = await workflow.understand(state)

    assert update["history_digest"] == ["轮2", "轮3", "第1轮独立请求"]


@pytest.mark.asyncio
async def test_digest_entry_is_truncated() -> None:
    class LongRequestService(RecordingPlanningService):
        async def resolve_context(
            self,
            conversation: list[dict[str, str]],
            memory_keys: Sequence[str] = (),
            open_tasks: Sequence[OpenTask] = (),
            recent_actions: Sequence[str] = (),
            user_name: str = "",
        ) -> ContextResolution:
            self.seen.append(conversation)
            return ContextResolution(
                standalone_request="长" * 2000,
                intent_summary="超长请求",
                requires_task_planning=False,
                reply="好的",
            )

    workflow = Workflow(SupervisorAgent(LongRequestService()))
    state = _context_state([HumanMessage(content="问题1")], [])

    update = await workflow.understand(state)

    assert len(update["history_digest"][0]) == 240


@pytest.mark.asyncio
async def test_window_disabled_keeps_full_history() -> None:
    workflow = Workflow(SupervisorAgent(RecordingPlanningService()), history_window=0)

    conversation = workflow._conversation(_context_state(_history(20), ["轮1"]))

    assert len(conversation) == 40
