import json
import time
from collections.abc import Mapping
from typing import Any, Literal

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from langsmith import traceable

from enterprise_ai_assistant.agents.domain_runtime import DomainRuntime, DomainRuntimeProvider
from enterprise_ai_assistant.core.metrics import (
    CONFIRMATIONS,
    TOOL_DURATION,
    TOOL_INVOCATIONS,
)
from enterprise_ai_assistant.core.models import (
    AgentName,
    AgentNote,
    DomainTaskRequest,
    DomainTaskResult,
    PendingConfirmation,
    TaskDraft,
    TaskStatus,
    ToolResult,
)
from enterprise_ai_assistant.graph.state import DomainTaskState
from enterprise_ai_assistant.tools import BusinessToolOutcome, ToolContext, ToolRisk
from enterprise_ai_assistant.tools.registry import HANDOFF_TOOL

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

    @classmethod
    def _answer_text(cls, message: AIMessage) -> str:
        return "" if message.tool_calls else cls._text(message)

    @staticmethod
    def _text(message: AIMessage) -> str:
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
        domain_input: dict[str, Any] = {
            "standalone_request": request.user_goal,
            "task": request.task.model_dump(mode="json"),
            "dependency_results": request.dependency_results,
        }
        # 记忆单独成键，和用户当前请求区分开：模型必须能分辨哪些是本轮说的、
        # 哪些只是历史档案给出的建议值。
        if request.user_name:
            domain_input["user_name"] = request.user_name
        if request.memories:
            domain_input["user_memory"] = list(request.memories)
        if request.recent_actions:
            domain_input["recent_actions"] = [
                item.render() for item in request.recent_actions
            ]
        # 续跑时带回上一轮追问留下的草稿。和记忆一样单独成键：用户本轮的补充和更正
        # 在 standalone_request 里，两者冲突时以本轮为准。
        if request.draft is not None:
            domain_input["previous_draft"] = request.draft.model_dump(mode="json")
        if request.recent_messages:
            domain_input["recent_messages"] = [
                turn.model_dump() for turn in request.recent_messages
            ]
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
            "domain_answer": "",
            "domain_notes": [],
            "domain_handoff_to": None,
            "pending_confirmation": None,
            "pending_tool_call": None,
            "confirmation_approved": False,
            "artifact": None,
            "domain_draft": None,
            "domain_tool_results": [],
        }

    async def decide(self, state: DomainTaskState) -> dict[str, Any]:
        iterations = state.get("domain_iterations", 0) + 1
        if iterations > self.MAX_ITERATIONS:
            raise RuntimeError("domain agent exceeded its model-call limit")
        request = self._request(state)
        executed = state.get("domain_tool_executed", False)
        response = await self._runtime(state).decide(
            request.task.objective,
            list(state.get("domain_messages", [])),
            task_id=request.task.id,
            answering=executed,
        )
        pending: dict[str, Any] | None = None
        registered = None
        validation_messages: list[ToolMessage] = []
        said = ""
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
            arguments: dict[str, Any] = {}
            if not isinstance(raw_arguments, Mapping):
                error = "工具参数必须是对象"
            else:
                try:
                    registered = self._runtime(state).tool(name)
                except (KeyError, ValueError):
                    error = f"工具 {name} 不在当前领域白名单中"
                else:
                    said, arguments = registered.split_message(raw_arguments)
                    # 参数在决策阶段就按契约校验。放到执行时才校验的话，写操作会带着
                    # 非法参数先弹确认卡，用户点了确认才失败；这里失败则交给模型自行更正。
                    error = registered.argument_error(arguments) or (
                        self._handoff_error(state, arguments)
                        if name == HANDOFF_TOOL
                        else None
                    )
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
                pending = {"name": name, "args": arguments, "id": str(raw_call["id"])}

        retry_required = bool(validation_messages) or (pending is None and not executed)
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

        notes = list(state.get("domain_notes", []))
        tools_so_far = [item.tool for item in state.get("domain_tool_results", [])]
        # 看过工具结果后，调工具时对用户说的话要进会话，否则刷新就没了，后面的轮次也不知道
        # 说过。这回合的原始消息（连同 message_to_user）留在 domain_messages 里，所以本任务
        # 后面写回答时模型也知道自己说过什么。第一次决策手里还没有任何工具结果，说什么都
        # 不是查来的，不对外。
        if executed and response.tool_calls:
            # 正文已经随 messages 流逐字流出去了；参数里的话要等工具调用生成完才拿得到，
            # 在这里整段推给前端。两处都写了的话多半是同一个意思，只认已经流出去的正文，
            # 否则用户会看到两遍。参数不合法时不推，模型更正后会重新说。
            text = self._text(response)
            if not text and pending is not None and said:
                text = said
                get_stream_writer()(
                    {"answer": said, "agent": request.task.domain.value, "task_id": request.task.id}
                )
            if text:
                notes.append(AgentNote(text=text, tools=tools_so_far))
        confirmation = None
        if pending:
            if registered is None:
                raise RuntimeError("validated tool registration is missing")
            if registered.risk == ToolRisk.WRITE:
                confirmation = PendingConfirmation(
                    task_id=request.task.id,
                    action=str(pending["name"]),
                    tool_call_id=str(pending["id"]),
                    title=registered.label,
                    fields=registered.confirmation_fields(pending["args"]),
                    payload=dict(pending["args"]),
                    notes=notes,
                )
                # 交给确认卡带出去了，之后由恢复命令写进会话，结果里不再重复。
                notes = []
        return {
            "domain_notes": notes,
            "domain_messages": domain_messages,
            "domain_iterations": iterations,
            "pending_tool_call": pending,
            "pending_confirmation": confirmation,
            "domain_retry_required": retry_required,
            "domain_answer": (
                "" if retry_required or pending else self._answer_text(response)
            ),
        }

    def _handoff_error(self, state: DomainTaskState, arguments: Mapping[str, Any]) -> str | None:
        request = self._request(state)
        if arguments.get("target_domain") == request.task.domain.value:
            return "不能转交给当前领域；任务属于本领域就用本领域的工具办理"
        runtime = self._runtime(state)
        # 已经提交、修改或撤销过单据的任务再转交，接手方不知道这些已经发生，
        # 用户看到的也会是一件办了一半又换人的事。
        if any(
            result.success and runtime.tool(result.tool).risk == ToolRisk.WRITE
            for result in state.get("domain_tool_results", [])
        ):
            return "本任务已经执行过写操作，不能再转交，请直接回答用户"
        return None

    def after_decide(
        self, state: DomainTaskState
    ) -> Literal["confirm_tool", "execute_tool", "finish", "respond", "decide"]:
        if state.get("domain_retry_required"):
            return "decide"
        call = state.get("pending_tool_call")
        if not call:
            return "finish" if state.get("domain_answer") else "respond"
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
        CONFIRMATIONS.labels(decision="approved" if approved else "rejected").inc()
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
        runtime = self._runtime(state)
        registered = runtime.tool(name)
        started = time.perf_counter()
        try:
            raw = await runtime.invoke_tool(name, dict(call["args"]))
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
        TOOL_DURATION.labels(tool=name).observe(time.perf_counter() - started)
        TOOL_INVOCATIONS.labels(
            tool=name, outcome="success" if outcome.success else "failure"
        ).inc()
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
        # 一个任务可能连续调用多个写工具（例如先查制度再提交单据）。按工具名归档，
        # 否则后一次产出会覆盖前一次，依赖该任务的下游任务将拿不到先前的业务单号。
        artifact = dict(state.get("artifact") or {})
        if outcome.success:
            artifact[name] = outcome.model_dump(mode="json")
        terminal = registered.terminal and outcome.success
        handoff_to = (
            str(outcome.data.get("target_domain"))
            if name == HANDOFF_TOOL and outcome.success
            else None
        )
        draft = state.get("domain_draft")
        answer = ""
        if terminal:
            draft = TaskDraft.model_validate(outcome.data)
            # 问题本来就是写给用户的，原样发出。再调一次模型只是把它转述一遍。
            answer = str((outcome.data or {}).get("question", "")).strip()
        return {
            "domain_messages": [*state.get("domain_messages", []), message],
            "domain_tool_results": [*state.get("domain_tool_results", []), audit],
            "artifact": artifact or None,
            "pending_confirmation": None,
            "pending_tool_call": None,
            "confirmation_approved": False,
            "domain_waiting_input": terminal,
            "domain_draft": draft,
            "domain_failed": not outcome.success,
            "domain_retry_required": False,
            "domain_tool_executed": True,
            "domain_answer": answer,
            "domain_handoff_to": handoff_to,
        }

    def after_execute(self, state: DomainTaskState) -> Literal["decide", "finish", "respond"]:
        if state.get("domain_failed"):
            return "respond"
        if state.get("domain_handoff_to"):
            return "finish"
        if state.get("domain_waiting_input"):
            return "finish" if state.get("domain_answer") else "respond"
        return "decide"

    async def finish(self, state: DomainTaskState) -> dict[str, Any]:
        """收口已经确定的回答，不调模型。"""
        handoff_to = state.get("domain_handoff_to")
        if handoff_to:
            # 分错的任务不对用户说话，由父图改派后接手的领域 Agent 回答。
            request = self._request(state)
            return {
                "domain_result": DomainTaskResult(
                    task_id=request.task.id,
                    status=TaskStatus.HANDED_OFF,
                    tool_results=list(state.get("domain_tool_results", [])),
                    handoff_to=AgentName(handoff_to),
                    notes=list(state.get("domain_notes", [])),
                )
            }
        answer = str(state.get("domain_answer", ""))
        if not answer:
            raise RuntimeError("domain task finished without an answer")
        if state.get("domain_waiting_input"):
            # 决策调用写出的回答已经逐字流出去了；追问的问题没经过模型，
            # SSE 从 messages 流里拿不到它，要单独推一次，否则前端在多任务的
            # 轮次里会漏掉这一段。
            request = self._request(state)
            get_stream_writer()(
                {
                    "answer": answer,
                    "agent": request.task.domain.value,
                    "task_id": request.task.id,
                }
            )
        return {"domain_result": self._result(state, answer)}

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
        return {
            "domain_messages": [*state.get("domain_messages", []), response],
            "domain_result": self._result(state, answer),
        }

    def _result(self, state: DomainTaskState, answer: str) -> DomainTaskResult:
        request = self._request(state)
        if state.get("domain_waiting_input"):
            status = TaskStatus.WAITING_INPUT
        elif state.get("domain_rejected"):
            status = TaskStatus.REJECTED
        elif state.get("domain_failed"):
            status = TaskStatus.FAILED
        else:
            status = TaskStatus.COMPLETED
        return DomainTaskResult(
            task_id=request.task.id,
            status=status,
            answer=answer,
            artifact=state.get("artifact"),
            tool_results=list(state.get("domain_tool_results", [])),
            draft=state.get("domain_draft") if status == TaskStatus.WAITING_INPUT else None,
            notes=list(state.get("domain_notes", [])),
        )


def build_domain_graph(workflow: DomainTaskWorkflow) -> Any:
    graph = StateGraph(DomainTaskState)
    graph.add_node("initialize", workflow.initialize)
    graph.add_node("decide", workflow.decide)
    graph.add_node("confirm_tool", workflow.confirm_tool)
    graph.add_node("execute_tool", workflow.execute_tool)
    graph.add_node("finish", workflow.finish)
    graph.add_node("respond", workflow.respond)

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "decide")
    graph.add_conditional_edges("decide", workflow.after_decide)
    graph.add_conditional_edges("confirm_tool", workflow.after_confirm)
    graph.add_conditional_edges("execute_tool", workflow.after_execute)
    graph.add_edge("finish", END)
    graph.add_edge("respond", END)
    return graph.compile()
