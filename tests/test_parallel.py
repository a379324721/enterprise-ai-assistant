"""没有依赖的任务并行执行；同时停在确认卡上时逐张确认。"""

import asyncio
from collections.abc import Sequence
from typing import Any
from uuid import UUID, uuid4

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from enterprise_ai_assistant.agents.domain_runtime import DomainRuntimeProvider
from enterprise_ai_assistant.agents.supervisor import SupervisorAgent
from enterprise_ai_assistant.api import routes
from enterprise_ai_assistant.api.schemas import ConfirmationRequest
from enterprise_ai_assistant.core.models import (
    AgentName,
    ContextResolution,
    MemoryExtraction,
    OpenTask,
    PlannedTask,
    TaskOutline,
    TaskPlan,
    TaskStatus,
)
from enterprise_ai_assistant.graph.domain import DomainTaskWorkflow
from enterprise_ai_assistant.graph.serde import checkpoint_serializer
from enterprise_ai_assistant.graph.workflow import Workflow, build_graph
from enterprise_ai_assistant.repositories.actions import InMemoryActionRepository
from enterprise_ai_assistant.repositories.policies import InMemoryPolicyRepository
from enterprise_ai_assistant.services.planning import PlanningService
from enterprise_ai_assistant.tools import LocalEnterpriseToolProvider, ToolContext
from enterprise_ai_assistant.tools.registry import DomainToolRegistry, RegisteredTool

CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000031")
CONFIG = {"configurable": {"thread_id": str(CONVERSATION_ID)}}
CALLS: list[AgentName] = []

_WRITES: dict[AgentName, dict[str, Any]] = {
    AgentName.TRAVEL: {
        "name": "create_travel_application",
        "args": {
            "origin": "杭州",
            "destination": "上海",
            "start_date": "2026-08-10",
            "end_date": "2026-08-14",
            "purpose": "客户交流",
        },
    },
    AgentName.EXPENSE: {
        "name": "create_expense_claim",
        "args": {"expense_type": "交通", "amount": "58.00", "receipt_refs": ["INV-1"]},
    },
}


class TwoTasks(PlanningService):
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
            standalone_request="申请上海出差，另外报销打车费 58 元",
            intent_summary="出差并报销",
            requires_task_planning=True,
            tasks=[
                TaskOutline(title="上海出差申请", domain=AgentName.TRAVEL, objective="申请出差"),
                TaskOutline(title="报销打车费", domain=AgentName.EXPENSE, objective="报销"),
            ],
        )

    async def plan(self, context: ContextResolution) -> TaskPlan:
        raise AssertionError("tasks come from the supervisor")

    async def extract_memories(
        self, conversation: list[dict[str, str]], known: list[str]
    ) -> MemoryExtraction:
        return MemoryExtraction()


class RendezvousRuntime:
    """首次决策前必须等到另一个分支也开始决策：串行执行时这里会一直等下去。"""

    def __init__(
        self, name: AgentName, tools: list[RegisteredTool], started: dict[AgentName, asyncio.Event]
    ) -> None:
        self.name = name
        self.tools = {item.tool.name: item for item in tools}
        self._started = started

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
        CALLS.append(self.name)
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content=f"{self.name.value} 已提交")
        self._started[self.name].set()
        other = next(domain for domain in self._started if domain != self.name)
        await asyncio.wait_for(self._started[other].wait(), timeout=2)
        return AIMessage(
            content="",
            tool_calls=[{**_WRITES[self.name], "id": f"{task_id}-call", "type": "tool_call"}],
        )

    async def respond(
        self, task_objective: str, messages: list[BaseMessage], *, task_id: str
    ) -> AIMessage:
        raise AssertionError("not used")

    async def invoke_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self.tools[name].tool.ainvoke(arguments)
        assert isinstance(result, dict)
        return result


class RendezvousFactory(DomainRuntimeProvider):
    def __init__(self) -> None:
        provider = LocalEnterpriseToolProvider(
            InMemoryActionRepository(), InMemoryPolicyRepository()
        )
        self._registry = DomainToolRegistry(provider)
        self.started = {AgentName.TRAVEL: asyncio.Event(), AgentName.EXPENSE: asyncio.Event()}

    def create(self, agent: AgentName, context: ToolContext) -> RendezvousRuntime:
        return RendezvousRuntime(agent, self._registry.for_agent(agent, context), self.started)


async def _confirm(graph: Any) -> str:
    snapshot = await graph.aget_state(CONFIG)
    found = routes._pending_interrupt(snapshot)
    assert found is not None
    interrupt, pending = found
    command = routes._resume_command(
        ConfirmationRequest(confirmation_id=pending.confirmation_id, approved=True),
        interrupt,
        pending,
    )
    await graph.ainvoke(command, CONFIG)
    return pending.action


@pytest.mark.asyncio
async def test_independent_tasks_run_together_and_confirm_one_card_at_a_time() -> None:
    CALLS.clear()
    graph = build_graph(
        Workflow(SupervisorAgent(TwoTasks()), parallel_tasks=True),
        DomainTaskWorkflow(RendezvousFactory()),
        InMemorySaver(),
    )

    await graph.ainvoke(
        {
            "messages": [HumanMessage(content="申请上海出差，另外报销打车费 58 元")],
            "user_id": "u-1",
            "conversation_id": CONVERSATION_ID,
            "request_id": uuid4(),
        },
        CONFIG,
    )
    paused = await graph.aget_state(CONFIG)
    assert len(paused.interrupts) == 2

    # 界面一次只出一张卡；确认完第一张，第二张仍在，且不是刚确认过的那张。
    first = await _confirm(graph)
    # 第一张确认完、第二张还没确认时，已跑完的分支结果还没归并进状态，
    # 但响应里要能看到它的执行步骤和完成状态，否则两个步骤会等到第二次确认后才一起出现。
    between = await graph.aget_state(CONFIG)
    settled = routes._settled_results(between)
    assert [item.tool for result in settled for item in result.tool_results] == [first]
    assert between.values["tool_results"] == []
    second = await _confirm(graph)
    assert routes._settled_results(await graph.aget_state(CONFIG)) == []
    assert {first, second} == {"create_travel_application", "create_expense_claim"}

    final = await graph.aget_state(CONFIG)
    assert routes._pending_interrupt(final) is None
    assert [(task.domain, task.status) for task in final.values["tasks"]] == [
        (AgentName.TRAVEL, TaskStatus.COMPLETED),
        (AgentName.EXPENSE, TaskStatus.COMPLETED),
    ]
    # 分支完成的先后不固定，归并后的回答按计划顺序。
    assert final.values["turn_answers"] == ["travel 已提交", "expense 已提交"]
    # 每个分支一次选工具、一次作答。确认其中一张时另一个分支不会从头重跑、重复调模型。
    assert sorted(CALLS) == sorted([AgentName.TRAVEL, AgentName.EXPENSE] * 2)


@pytest.mark.asyncio
async def test_checkpoints_restore_with_an_explicit_type_allowlist(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """升级到拦截未登记类型的 LangGraph 后，停在确认卡上的会话要能照常恢复。

    显式传允许清单就等于严格模式：清单外的类型会被拦截，而不是只打警告。
    """
    CALLS.clear()
    graph = build_graph(
        Workflow(SupervisorAgent(TwoTasks()), parallel_tasks=True),
        DomainTaskWorkflow(RendezvousFactory()),
        InMemorySaver(serde=checkpoint_serializer()),
    )

    await graph.ainvoke(
        {
            "messages": [HumanMessage(content="申请上海出差，另外报销打车费 58 元")],
            "user_id": "u-1",
            "conversation_id": CONVERSATION_ID,
            "request_id": uuid4(),
        },
        CONFIG,
    )
    await _confirm(graph)
    await _confirm(graph)

    final = await graph.aget_state(CONFIG)
    assert all(isinstance(task, PlannedTask) for task in final.values["tasks"])
    assert [task.status for task in final.values["tasks"]] == [TaskStatus.COMPLETED] * 2
    assert not [record for record in caplog.records if "unregistered" in record.getMessage()]
    assert not [record for record in caplog.records if "blocked" in record.getMessage().lower()]
