from collections.abc import Mapping
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from langsmith import traceable

from enterprise_ai_assistant.agents.base import AgentOutcome, DomainAgent, WritableDomainAgent
from enterprise_ai_assistant.agents.domain_agents import (
    ExpenseAgent,
    HRAgent,
    PolicyAgent,
    TravelAgent,
)
from enterprise_ai_assistant.agents.supervisor import SupervisorAgent
from enterprise_ai_assistant.core.models import (
    AgentName,
    ContextResolution,
    PlannedTask,
    TaskStatus,
)
from enterprise_ai_assistant.graph.state import AssistantState


class Workflow:
    def __init__(
        self,
        supervisor: SupervisorAgent,
        travel: TravelAgent,
        expense: ExpenseAgent,
        hr: HRAgent,
        policy: PolicyAgent,
    ) -> None:
        self.supervisor = supervisor
        self.agents: dict[AgentName, DomainAgent] = {
            AgentName.TRAVEL: travel,
            AgentName.EXPENSE: expense,
            AgentName.HR: hr,
            AgentName.POLICY: policy,
        }
        self.write_agents: dict[AgentName, WritableDomainAgent] = {
            AgentName.TRAVEL: travel,
            AgentName.EXPENSE: expense,
            AgentName.HR: hr,
        }

    @staticmethod
    def _active_task(state: AssistantState) -> PlannedTask:
        task = next((item for item in state["tasks"] if item.id == state["active_task_id"]), None)
        if task is None:
            raise RuntimeError("active task is missing from state")
        return task

    async def understand(self, state: AssistantState) -> dict[str, Any]:
        conversation = [
            {
                "role": "user" if message.type == "human" else "assistant",
                "content": str(message.content),
            }
            for message in state["messages"]
            if message.type in {"human", "ai"}
        ]
        context = await self.supervisor.resolve_context(conversation)
        return {
            "user_goal": context.standalone_request,
            "understanding": context.model_dump(mode="json"),
        }

    async def plan(self, state: AssistantState) -> dict[str, Any]:
        context = ContextResolution.model_validate(state["understanding"])
        plan = await self.supervisor.plan(context)
        return {
            "user_goal": plan.user_goal,
            "tasks": plan.tasks,
            "slots": {**state.get("slots", {}), **plan.extracted_slots},
        }

    async def select_task(self, state: AssistantState) -> dict[str, Any]:
        task = self.supervisor.next_runnable(state["tasks"])
        if task is None:
            return {"active_task_id": None, "current_agent": None}
        agent = self.supervisor.route(task)
        tasks = [
            item.model_copy(update={"status": TaskStatus.RUNNING}) if item.id == task.id else item
            for item in state["tasks"]
        ]
        return {"tasks": tasks, "active_task_id": task.id, "current_agent": agent.value}

    def route_task(self, state: AssistantState) -> str:
        return state["current_agent"] or "done"

    async def _prepare(self, state: AssistantState, agent_name: AgentName) -> dict[str, Any]:
        task = self._active_task(state)
        outcome = await self.agents[agent_name].prepare(task, dict(state["slots"]))
        return self._outcome_update(state, outcome)

    async def travel(self, state: AssistantState) -> dict[str, Any]:
        return await self._prepare(state, AgentName.TRAVEL)

    async def expense(self, state: AssistantState) -> dict[str, Any]:
        return await self._prepare(state, AgentName.EXPENSE)

    async def hr(self, state: AssistantState) -> dict[str, Any]:
        return await self._prepare(state, AgentName.HR)

    async def policy(self, state: AssistantState) -> dict[str, Any]:
        return await self._prepare(state, AgentName.POLICY)

    @traceable(name="human-confirmation", run_type="chain")
    async def confirm(self, state: AssistantState) -> dict[str, Any]:
        pending = state["pending_confirmation"]
        if pending is None:
            raise RuntimeError("confirmation node entered without a pending action")
        decision = interrupt(pending.model_dump(mode="json"))
        approved = bool(decision.get("approved")) if isinstance(decision, Mapping) else False
        if approved:
            return {"confirmation_approved": True}
        tasks = [
            item.model_copy(update={"status": TaskStatus.REJECTED})
            if item.id == pending.task_id
            else item
            for item in state["tasks"]
        ]
        return {
            "tasks": tasks,
            "confirmation_approved": False,
            "pending_confirmation": None,
            "last_answer": "操作已取消，未写入企业系统。",
            "messages": [AIMessage(content="操作已取消，未写入企业系统。")],
        }

    async def _execute(self, state: AssistantState, agent_name: AgentName) -> dict[str, Any]:
        pending = state["pending_confirmation"]
        if pending is None:
            raise RuntimeError("execute node entered without a confirmed action")
        outcome = await self.write_agents[agent_name].execute(
            self._active_task(state), state["user_id"], pending.payload
        )
        update = self._outcome_update(state, outcome)
        update["pending_confirmation"] = None
        return update

    async def execute_travel(self, state: AssistantState) -> dict[str, Any]:
        return await self._execute(state, AgentName.TRAVEL)

    async def execute_expense(self, state: AssistantState) -> dict[str, Any]:
        return await self._execute(state, AgentName.EXPENSE)

    async def execute_hr(self, state: AssistantState) -> dict[str, Any]:
        return await self._execute(state, AgentName.HR)

    def after_prepare(self, state: AssistantState) -> Literal["confirm", "complete_task"]:
        return "confirm" if state["pending_confirmation"] else "complete_task"

    def after_confirm(self, state: AssistantState) -> str:
        return (
            f"execute_{state['current_agent']}"
            if state.get("confirmation_approved")
            else "complete_task"
        )

    async def complete_task(self, state: AssistantState) -> dict[str, Any]:
        active_id = state["active_task_id"]
        tasks = []
        for item in state["tasks"]:
            if item.id == active_id and item.status == TaskStatus.RUNNING:
                item = item.model_copy(update={"status": TaskStatus.COMPLETED})
            tasks.append(item)
        # 取消前置任务已被拒绝的任务，避免 DAG 陷入死锁。
        rejected = {
            item.id for item in tasks if item.status in {TaskStatus.REJECTED, TaskStatus.FAILED}
        }
        tasks = [
            item.model_copy(update={"status": TaskStatus.REJECTED})
            if set(item.depends_on) & rejected and item.status == TaskStatus.PENDING
            else item
            for item in tasks
        ]
        return {
            "tasks": tasks,
            "active_task_id": None,
            "current_agent": None,
            "confirmation_approved": False,
        }

    @staticmethod
    def _outcome_update(state: AssistantState, outcome: AgentOutcome) -> dict[str, Any]:
        update: dict[str, Any] = {
            "last_answer": outcome.answer,
            "messages": [AIMessage(content=outcome.answer)],
            "slots": {**state["slots"], **outcome.slot_updates},
            "pending_confirmation": outcome.confirmation,
        }
        if outcome.tool_result:
            update["tool_results"] = [*state["tool_results"], outcome.tool_result]
        return update


def build_graph(workflow: Workflow, checkpointer: Any) -> Any:
    graph = StateGraph(AssistantState)
    graph.add_node("understand", workflow.understand)
    graph.add_node("plan", workflow.plan)
    graph.add_node("select_task", workflow.select_task)
    for name in ("travel", "expense", "hr", "policy"):
        graph.add_node(name, getattr(workflow, name))
    graph.add_node("confirm", workflow.confirm)
    for name in ("travel", "expense", "hr"):
        graph.add_node(f"execute_{name}", getattr(workflow, f"execute_{name}"))
    graph.add_node("complete_task", workflow.complete_task)

    graph.add_edge(START, "understand")
    graph.add_edge("understand", "plan")
    graph.add_edge("plan", "select_task")
    graph.add_conditional_edges(
        "select_task",
        workflow.route_task,
        {"travel": "travel", "expense": "expense", "hr": "hr", "policy": "policy", "done": END},
    )
    for name in ("travel", "expense", "hr", "policy"):
        graph.add_conditional_edges(name, workflow.after_prepare)
    graph.add_conditional_edges(
        "confirm",
        workflow.after_confirm,
        {
            "execute_travel": "execute_travel",
            "execute_expense": "execute_expense",
            "execute_hr": "execute_hr",
            "complete_task": "complete_task",
        },
    )
    for name in ("execute_travel", "execute_expense", "execute_hr"):
        graph.add_edge(name, "complete_task")
    graph.add_edge("complete_task", "select_task")
    return graph.compile(checkpointer=checkpointer)
