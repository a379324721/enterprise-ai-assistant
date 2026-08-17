import json
from collections.abc import Mapping
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from langsmith import traceable

from enterprise_ai_assistant.agents.domain_runtime import DomainRuntime, DomainRuntimeProvider
from enterprise_ai_assistant.agents.supervisor import SupervisorAgent
from enterprise_ai_assistant.core.models import (
    ContextResolution,
    PendingConfirmation,
    PlannedTask,
    TaskStatus,
    ToolResult,
)
from enterprise_ai_assistant.graph.state import AssistantState
from enterprise_ai_assistant.tools import BusinessToolOutcome, ToolContext, ToolRisk


class Workflow:
    MAX_DOMAIN_ITERATIONS = 8

    def __init__(self, supervisor: SupervisorAgent, domains: DomainRuntimeProvider) -> None:
        self.supervisor = supervisor
        self.domains = domains

    @staticmethod
    def _active_task(state: AssistantState) -> PlannedTask:
        task = next((item for item in state["tasks"] if item.id == state["active_task_id"]), None)
        if task is None:
            raise RuntimeError("active task is missing from state")
        return task

    def _runtime(self, state: AssistantState) -> DomainRuntime:
        task = self._active_task(state)
        return self.domains.create(
            task.domain,
            ToolContext(
                user_id=state["user_id"],
                task_id=task.id,
                idempotency_key=f"{state['user_id']}:{task.id}",
            ),
        )

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
            "pending_confirmation": None,
            "pending_tool_call": None,
        }

    async def plan(self, state: AssistantState) -> dict[str, Any]:
        context = ContextResolution.model_validate(state["understanding"])
        plan = await self.supervisor.plan(context)
        return {"user_goal": plan.user_goal, "tasks": plan.tasks}

    async def select_task(self, state: AssistantState) -> dict[str, Any]:
        task = self.supervisor.next_runnable(state["tasks"])
        if task is None:
            return {"active_task_id": None, "current_agent": None}
        tasks = [
            item.model_copy(update={"status": TaskStatus.RUNNING}) if item.id == task.id else item
            for item in state["tasks"]
        ]
        dependency_results = {
            dependency: state.get("artifacts", {}).get(dependency)
            for dependency in task.depends_on
            if dependency in state.get("artifacts", {})
        }
        domain_input = {
            "standalone_request": state["user_goal"],
            "task": task.model_dump(mode="json"),
            "dependency_results": dependency_results,
        }
        return {
            "tasks": tasks,
            "active_task_id": task.id,
            "current_agent": task.domain.value,
            "domain_messages": [
                HumanMessage(content=json.dumps(domain_input, ensure_ascii=False))
            ],
            "domain_iterations": 0,
            "domain_waiting_input": False,
            "domain_rejected": False,
            "domain_failed": False,
            "pending_tool_call": None,
        }

    def route_task(self, state: AssistantState) -> str:
        return "domain_decide" if state.get("active_task_id") else "done"

    async def domain_decide(self, state: AssistantState) -> dict[str, Any]:
        iterations = state.get("domain_iterations", 0) + 1
        if iterations > self.MAX_DOMAIN_ITERATIONS:
            raise RuntimeError("domain agent exceeded its model-call limit")
        task = self._active_task(state)
        response = await self._runtime(state).decide(
            task.objective,
            list(state.get("domain_messages", [])),
            task_id=task.id,
        )
        if len(response.tool_calls) > 1:
            raise RuntimeError("parallel domain tool calls are not allowed")
        pending: dict[str, Any] | None = None
        if response.tool_calls:
            raw_call = response.tool_calls[0]
            raw_arguments = raw_call.get("args")
            if not isinstance(raw_arguments, Mapping):
                raise RuntimeError("domain tool call arguments must be an object")
            pending = {
                "name": str(raw_call["name"]),
                "args": dict(raw_arguments),
                "id": str(raw_call["id"]),
            }
        confirmation = None
        tasks = state["tasks"]
        if pending:
            registered = self._runtime(state).tool(str(pending["name"]))
            if registered.risk == ToolRisk.WRITE:
                confirmation = PendingConfirmation(
                    task_id=task.id,
                    action=str(pending["name"]),
                    tool_call_id=str(pending["id"]),
                    summary=(
                        f"允许 {task.domain.value} 执行 {pending['name']}："
                        f"{json.dumps(pending['args'], ensure_ascii=False)}"
                    ),
                    payload=dict(pending["args"]),
                )
                tasks = [
                    item.model_copy(update={"status": TaskStatus.WAITING_CONFIRMATION})
                    if item.id == task.id
                    else item
                    for item in tasks
                ]
        return {
            "tasks": tasks,
            "domain_messages": [*state.get("domain_messages", []), response],
            "domain_iterations": iterations,
            "pending_tool_call": pending,
            "pending_confirmation": confirmation,
        }

    def after_decide(self, state: AssistantState) -> Literal["confirm_tool", "execute_tool", "domain_respond"]:
        call = state.get("pending_tool_call")
        if not call:
            return "domain_respond"
        registered = self._runtime(state).tool(str(call["name"]))
        return "confirm_tool" if registered.risk == ToolRisk.WRITE else "execute_tool"

    @traceable(name="tool-confirmation", run_type="chain")
    async def confirm_tool(self, state: AssistantState) -> dict[str, Any]:
        call = state.get("pending_tool_call")
        if not call:
            raise RuntimeError("confirmation entered without a tool call")
        pending = state.get("pending_confirmation")
        if pending is None:
            raise RuntimeError("confirmation details are missing")
        decision = interrupt(pending.model_dump(mode="json"))
        approved = bool(decision.get("approved")) if isinstance(decision, Mapping) else False
        if approved:
            tasks = [
                item.model_copy(update={"status": TaskStatus.RUNNING})
                if item.id == pending.task_id
                else item
                for item in state["tasks"]
            ]
            return {
                "tasks": tasks,
                "pending_confirmation": pending,
                "confirmation_approved": True,
            }
        rejected = ToolMessage(
            content=json.dumps(
                {"success": False, "status": "rejected", "error": "用户拒绝执行工具"},
                ensure_ascii=False,
            ),
            tool_call_id=pending.tool_call_id,
            name=pending.action,
        )
        return {
            "domain_messages": [*state.get("domain_messages", []), rejected],
            "pending_confirmation": None,
            "pending_tool_call": None,
            "confirmation_approved": False,
            "domain_rejected": True,
        }

    def after_confirm(self, state: AssistantState) -> Literal["execute_tool", "domain_respond"]:
        return "execute_tool" if state.get("confirmation_approved") else "domain_respond"

    async def execute_tool(self, state: AssistantState) -> dict[str, Any]:
        call = state.get("pending_tool_call")
        if not call:
            raise RuntimeError("tool execution entered without a tool call")
        task = self._active_task(state)
        name = str(call["name"])
        registered = self._runtime(state).tool(name)
        try:
            raw = await self._runtime(state).invoke_tool(name, dict(call["args"]))
            outcome = BusinessToolOutcome.model_validate(raw)
        except Exception as exc:
            outcome = BusinessToolOutcome(
                tool=name,
                success=False,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        message = ToolMessage(
            content=outcome.model_dump_json(),
            tool_call_id=str(call["id"]),
            name=name,
        )
        audit = ToolResult(
            task_id=task.id,
            tool=name,
            success=outcome.success,
            data=outcome.model_dump(mode="json"),
            error=outcome.error,
        )
        artifacts = dict(state.get("artifacts", {}))
        artifacts[task.id] = outcome.model_dump(mode="json")
        return {
            "domain_messages": [*state.get("domain_messages", []), message],
            "tool_results": [*state.get("tool_results", []), audit],
            "artifacts": artifacts,
            "pending_confirmation": None,
            "pending_tool_call": None,
            "confirmation_approved": False,
            "domain_waiting_input": registered.terminal,
            "domain_failed": not outcome.success,
        }

    def after_execute(self, state: AssistantState) -> Literal["domain_decide", "domain_respond"]:
        return (
            "domain_respond"
            if state.get("domain_waiting_input") or state.get("domain_failed")
            else "domain_decide"
        )

    async def domain_respond(self, state: AssistantState) -> dict[str, Any]:
        task = self._active_task(state)
        response = await self._runtime(state).respond(
            task.objective,
            list(state.get("domain_messages", [])),
            task_id=task.id,
        )
        if state.get("domain_waiting_input"):
            status = TaskStatus.WAITING_INPUT
        elif state.get("domain_rejected"):
            status = TaskStatus.REJECTED
        elif state.get("domain_failed"):
            status = TaskStatus.FAILED
        else:
            status = TaskStatus.COMPLETED
        tasks = [
            item.model_copy(update={"status": status}) if item.id == task.id else item
            for item in state["tasks"]
        ]
        if status in {TaskStatus.REJECTED, TaskStatus.FAILED}:
            blocked = {
                item.id
                for item in tasks
                if item.status in {TaskStatus.REJECTED, TaskStatus.FAILED}
            }
            tasks = [
                item.model_copy(update={"status": TaskStatus.REJECTED})
                if set(item.depends_on) & blocked and item.status == TaskStatus.PENDING
                else item
                for item in tasks
            ]
        return {
            "tasks": tasks,
            "last_answer": str(response.content),
            "messages": [AIMessage(content=response.content)],
            "active_task_id": None,
            "current_agent": None,
            "domain_messages": [*state.get("domain_messages", []), response],
        }

    def after_response(self, state: AssistantState) -> Literal["select_task", "done"]:
        return "done" if any(
            item.status == TaskStatus.WAITING_INPUT for item in state["tasks"]
        ) else "select_task"


def build_graph(workflow: Workflow, checkpointer: Any) -> Any:
    graph = StateGraph(AssistantState)
    graph.add_node("understand", workflow.understand)
    graph.add_node("plan", workflow.plan)
    graph.add_node("select_task", workflow.select_task)
    graph.add_node("domain_decide", workflow.domain_decide)
    graph.add_node("confirm_tool", workflow.confirm_tool)
    graph.add_node("execute_tool", workflow.execute_tool)
    graph.add_node("domain_respond", workflow.domain_respond)

    graph.add_edge(START, "understand")
    graph.add_edge("understand", "plan")
    graph.add_edge("plan", "select_task")
    graph.add_conditional_edges(
        "select_task", workflow.route_task, {"domain_decide": "domain_decide", "done": END}
    )
    graph.add_conditional_edges("domain_decide", workflow.after_decide)
    graph.add_conditional_edges("confirm_tool", workflow.after_confirm)
    graph.add_conditional_edges("execute_tool", workflow.after_execute)
    graph.add_conditional_edges(
        "domain_respond", workflow.after_response, {"select_task": "select_task", "done": END}
    )
    return graph.compile(checkpointer=checkpointer)
