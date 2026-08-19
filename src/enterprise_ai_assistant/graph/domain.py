import json
from collections.abc import Mapping
from typing import Any, Literal

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from langsmith import traceable

from enterprise_ai_assistant.agents.domain_runtime import DomainRuntime, DomainRuntimeProvider
from enterprise_ai_assistant.core.models import (
    DomainTaskRequest,
    DomainTaskResult,
    PendingConfirmation,
    TaskStatus,
    ToolResult,
)
from enterprise_ai_assistant.graph.state import DomainTaskState
from enterprise_ai_assistant.tools import BusinessToolOutcome, ToolContext, ToolRisk

logger = structlog.get_logger()


class DomainTaskWorkflow:
    """执行单个领域任务；内部循环和临时状态不泄漏到调度图。"""

    MAX_ITERATIONS = 8

    def __init__(self, domains: DomainRuntimeProvider) -> None:
        self.domains = domains

    @staticmethod
    def _request(state: DomainTaskState) -> DomainTaskRequest:
        return DomainTaskRequest.model_validate(state["domain_request"])

    def _runtime(self, state: DomainTaskState) -> DomainRuntime:
        request = self._request(state)
        return self.domains.create(
            request.task.domain,
            ToolContext(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                request_id=request.request_id,
                task_id=request.task.id,
            ),
        )

    @staticmethod
    def _answer_text(message: AIMessage) -> str:
        if message.tool_calls:
            return ""
        if isinstance(message.content, str):
            return message.content.strip()
        if not isinstance(message.content, list):
            return ""
        parts: list[str] = []
        for block in message.content:
            if isinstance(block, str):
                parts.append(block)
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                parts.append(block["text"])
        return "".join(parts).strip()

    async def initialize(self, state: DomainTaskState) -> dict[str, Any]:
        request = self._request(state)
        domain_input = {
            "standalone_request": request.user_goal,
            "task": request.task.model_dump(mode="json"),
            "dependency_results": request.dependency_results,
        }
        return {
            "domain_result": None,
            "domain_messages": [
                HumanMessage(content=json.dumps(domain_input, ensure_ascii=False))
            ],
            "domain_iterations": 0,
            "domain_waiting_input": False,
            "domain_rejected": False,
            "domain_failed": False,
            "domain_retry_required": False,
            "domain_tool_executed": False,
            "pending_confirmation": None,
            "pending_tool_call": None,
            "confirmation_approved": False,
            "artifact": None,
            "tool_results": [],
        }

    async def decide(self, state: DomainTaskState) -> dict[str, Any]:
        iterations = state.get("domain_iterations", 0) + 1
        if iterations > self.MAX_ITERATIONS:
            raise RuntimeError("domain agent exceeded its model-call limit")
        request = self._request(state)
        response = await self._runtime(state).decide(
            request.task.objective,
            list(state.get("domain_messages", [])),
            task_id=request.task.id,
        )
        pending: dict[str, Any] | None = None
        registered = None
        validation_messages: list[ToolMessage] = []
        if len(response.tool_calls) > 1:
            for raw_call in response.tool_calls:
                validation_messages.append(
                    ToolMessage(
                        content=json.dumps(
                            {
                                "success": False,
                                "status": "invalid_tool_call",
                                "error": "每次只能调用一个工具",
                            },
                            ensure_ascii=False,
                        ),
                        tool_call_id=str(raw_call["id"]),
                        name=str(raw_call["name"]),
                    )
                )
        elif response.tool_calls:
            raw_call = response.tool_calls[0]
            raw_arguments = raw_call.get("args")
            name = str(raw_call["name"])
            error = None
            if not isinstance(raw_arguments, Mapping):
                error = "工具参数必须是对象"
            else:
                try:
                    registered = self._runtime(state).tool(name)
                except (KeyError, ValueError):
                    error = f"工具 {name} 不在当前领域白名单中"
            if error:
                validation_messages.append(
                    ToolMessage(
                        content=json.dumps(
                            {
                                "success": False,
                                "status": "invalid_tool_call",
                                "error": error,
                            },
                            ensure_ascii=False,
                        ),
                        tool_call_id=str(raw_call["id"]),
                        name=name,
                    )
                )
            else:
                pending = {
                    "name": name,
                    "args": dict(raw_arguments),
                    "id": str(raw_call["id"]),
                }

        retry_required = bool(validation_messages) or (
            pending is None and not state.get("domain_tool_executed", False)
        )
        domain_messages = [
            *state.get("domain_messages", []),
            response,
            *validation_messages,
        ]
        if retry_required:
            domain_messages.append(
                SystemMessage(
                    content=(
                        "运行时校验：当前任务尚未调用任何工具。必须选择一个领域工具，"
                        "或调用 request_information 说明缺失字段。"
                    )
                )
            )

        confirmation = None
        if pending:
            if registered is None:
                raise RuntimeError("validated tool registration is missing")
            if registered.risk == ToolRisk.WRITE:
                confirmation = PendingConfirmation(
                    task_id=request.task.id,
                    action=str(pending["name"]),
                    tool_call_id=str(pending["id"]),
                    summary=(
                        f"允许 {request.task.domain.value} 执行 {pending['name']}："
                        f"{json.dumps(pending['args'], ensure_ascii=False)}"
                    ),
                    payload=dict(pending["args"]),
                )
        return {
            "domain_messages": domain_messages,
            "domain_iterations": iterations,
            "pending_tool_call": pending,
            "pending_confirmation": confirmation,
            "domain_retry_required": retry_required,
        }

    def after_decide(
        self, state: DomainTaskState
    ) -> Literal["confirm_tool", "execute_tool", "respond", "decide"]:
        if state.get("domain_retry_required"):
            return "decide"
        call = state.get("pending_tool_call")
        if not call:
            return "respond"
        registered = self._runtime(state).tool(str(call["name"]))
        return "confirm_tool" if registered.risk == ToolRisk.WRITE else "execute_tool"

    @traceable(name="tool-confirmation", run_type="chain")
    async def confirm_tool(self, state: DomainTaskState) -> dict[str, Any]:
        call = state.get("pending_tool_call")
        if not call:
            raise RuntimeError("confirmation entered without a tool call")
        pending = state.get("pending_confirmation")
        if pending is None:
            raise RuntimeError("confirmation details are missing")
        pending = PendingConfirmation.model_validate(pending)
        decision = interrupt(pending.model_dump(mode="json"))
        confirmation_id = decision.get("confirmation_id") if isinstance(decision, Mapping) else None
        if str(confirmation_id) != str(pending.confirmation_id):
            raise RuntimeError("confirmation id does not match the pending action")
        approved = bool(decision.get("approved")) if isinstance(decision, Mapping) else False
        if approved:
            return {"confirmation_approved": True}

        comment = decision.get("comment") if isinstance(decision, Mapping) else None
        error = "用户拒绝执行工具"
        if isinstance(comment, str) and comment.strip():
            error = f"{error}：{comment.strip()}"
        rejected = ToolMessage(
            content=json.dumps(
                {"success": False, "status": "rejected", "error": error},
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

    def after_confirm(self, state: DomainTaskState) -> Literal["execute_tool", "respond"]:
        return "execute_tool" if state.get("confirmation_approved") else "respond"

    async def execute_tool(self, state: DomainTaskState) -> dict[str, Any]:
        call = state.get("pending_tool_call")
        if not call:
            raise RuntimeError("tool execution entered without a tool call")
        request = self._request(state)
        name = str(call["name"])
        registered = self._runtime(state).tool(name)
        try:
            raw = await self._runtime(state).invoke_tool(name, dict(call["args"]))
            outcome = BusinessToolOutcome.model_validate(raw)
        except Exception:
            logger.exception(
                "domain_tool_failed",
                tool=name,
                task_id=request.task.id,
                request_id=str(request.request_id),
            )
            outcome = BusinessToolOutcome(
                tool=name,
                success=False,
                status="failed",
                error="企业工具执行失败",
            )
        message = ToolMessage(
            content=outcome.model_dump_json(),
            tool_call_id=str(call["id"]),
            name=name,
        )
        audit = ToolResult(
            task_id=request.task.id,
            tool=name,
            success=outcome.success,
            data=outcome.model_dump(mode="json"),
            error=outcome.error,
        )
        return {
            "domain_messages": [*state.get("domain_messages", []), message],
            "tool_results": [*state.get("tool_results", []), audit],
            "artifact": outcome.model_dump(mode="json"),
            "pending_confirmation": None,
            "pending_tool_call": None,
            "confirmation_approved": False,
            "domain_waiting_input": registered.terminal and outcome.success,
            "domain_failed": not outcome.success,
            "domain_retry_required": False,
            "domain_tool_executed": True,
        }

    def after_execute(self, state: DomainTaskState) -> Literal["decide", "respond"]:
        return (
            "respond"
            if state.get("domain_waiting_input") or state.get("domain_failed")
            else "decide"
        )

    async def respond(self, state: DomainTaskState) -> dict[str, Any]:
        request = self._request(state)
        response = await self._runtime(state).respond(
            request.task.objective,
            list(state.get("domain_messages", [])),
            task_id=request.task.id,
        )
        answer = self._answer_text(response)
        if not answer:
            raise RuntimeError("domain responder returned no user-visible text")
        if state.get("domain_waiting_input"):
            status = TaskStatus.WAITING_INPUT
        elif state.get("domain_rejected"):
            status = TaskStatus.REJECTED
        elif state.get("domain_failed"):
            status = TaskStatus.FAILED
        else:
            status = TaskStatus.COMPLETED
        return {
            "domain_messages": [*state.get("domain_messages", []), response],
            "domain_result": DomainTaskResult(
                task_id=request.task.id,
                status=status,
                answer=answer,
                artifact=state.get("artifact"),
                tool_results=list(state.get("tool_results", [])),
            ),
        }


def build_domain_graph(workflow: DomainTaskWorkflow) -> Any:
    graph = StateGraph(DomainTaskState)
    graph.add_node("initialize", workflow.initialize)
    graph.add_node("decide", workflow.decide)
    graph.add_node("confirm_tool", workflow.confirm_tool)
    graph.add_node("execute_tool", workflow.execute_tool)
    graph.add_node("respond", workflow.respond)

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "decide")
    graph.add_conditional_edges("decide", workflow.after_decide)
    graph.add_conditional_edges("confirm_tool", workflow.after_confirm)
    graph.add_conditional_edges("execute_tool", workflow.after_execute)
    graph.add_edge("respond", END)
    return graph.compile()
