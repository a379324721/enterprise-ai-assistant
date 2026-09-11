from typing import Any, Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from enterprise_ai_assistant.agents.supervisor import SupervisorAgent
from enterprise_ai_assistant.core.models import (
    ContextResolution,
    DomainTaskRequest,
    DomainTaskResult,
    PlannedTask,
    TaskStatus,
)
from enterprise_ai_assistant.graph.domain import DomainTaskWorkflow, build_domain_graph
from enterprise_ai_assistant.graph.state import AssistantState

#: 超出消息窗口的历史以摘要形式回灌，需要显式标注来源，避免被当成用户当前发言。
DIGEST_HEADER = "【早先会话摘要，仅供指代消解参考，不是用户当前发言】"
#: 单条摘要的裁剪长度；standalone_request 上限 8000 字符，原样堆叠会让摘要本身变成新的成本源。
DIGEST_ITEM_MAX_CHARS = 240


class Workflow:
    """外层工作流只负责上下文理解、任务规划和领域子图调度。"""

    def __init__(
        self,
        supervisor: SupervisorAgent,
        *,
        history_window: int = 12,
        digest_turns: int = 20,
    ) -> None:
        self.supervisor = supervisor
        self._history_window = history_window
        self._digest_turns = digest_turns

    def _conversation(self, state: AssistantState) -> list[dict[str, str]]:
        """把会话裁剪成有上界的 prompt 输入。

        messages 由 add_messages 累积且从不回收，完整序列化会让每轮的 Context
        Supervisor 成本随会话长度线性上涨。这里只保留最近若干条原文，更早的轮次
        用 understand 阶段已经产出的 standalone_request 降级成摘要。
        """
        turns = [
            {
                "role": "user" if message.type == "human" else "assistant",
                "content": str(message.content),
            }
            for message in state["messages"]
            if message.type in {"human", "ai"}
        ]
        if self._history_window <= 0 or len(turns) <= self._history_window:
            return turns

        window = turns[-self._history_window :]
        # digest 按轮次从旧到新排列，被窗口挤出去的就是最老的 dropped 轮。用消息数
        # 而不是 digest 长度定位，understand（当轮摘要尚未写入）和 direct_respond
        # （已写入）两种时序下都不会错位。
        dropped = sum(1 for turn in turns if turn["role"] == "user") - sum(
            1 for turn in window if turn["role"] == "user"
        )
        earlier = list(state.get("history_digest", []))[:dropped]
        if not earlier:
            return window
        summary = "\n".join(f"- {item}" for item in earlier)
        return [{"role": "user", "content": f"{DIGEST_HEADER}\n{summary}"}, *window]

    def _extend_digest(self, state: AssistantState, standalone_request: str) -> list[str]:
        item = standalone_request.strip()[:DIGEST_ITEM_MAX_CHARS]
        digest = [*state.get("history_digest", []), item]
        return digest[-self._digest_turns :] if self._digest_turns > 0 else []

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

    async def understand(self, state: AssistantState) -> dict[str, Any]:
        context = await self.supervisor.resolve_context(self._conversation(state))
        return {
            "user_goal": context.standalone_request,
            "understanding": context.model_dump(mode="json"),
            "history_digest": self._extend_digest(state, context.standalone_request),
            "tasks": [],
            "artifacts": {},
            "tool_results": [],
            "active_task_id": None,
            "current_agent": None,
            "domain_request": None,
            "domain_result": None,
            "last_answer": "",
            "turn_answers": [],
        }

    def after_understand(self, state: AssistantState) -> Literal["plan", "direct_respond"]:
        context = ContextResolution.model_validate(state["understanding"])
        return "plan" if context.requires_task_planning else "direct_respond"

    async def direct_respond(self, state: AssistantState) -> dict[str, Any]:
        response = await self.supervisor.respond_direct(self._conversation(state))
        answer = self._answer_text(response)
        if not answer:
            raise RuntimeError("direct responder returned no user-visible text")
        return {
            "last_answer": answer,
            "turn_answers": [answer],
            "messages": [AIMessage(content=answer)],
        }

    async def plan(self, state: AssistantState) -> dict[str, Any]:
        context = ContextResolution.model_validate(state["understanding"])
        plan = await self.supervisor.plan(context)
        tasks = [task.model_copy(update={"status": TaskStatus.PENDING}) for task in plan.tasks]
        return {"user_goal": plan.user_goal, "tasks": tasks}

    async def select_task(self, state: AssistantState) -> dict[str, Any]:
        task = self.supervisor.next_runnable(state["tasks"])
        if task is None:
            return {
                "active_task_id": None,
                "current_agent": None,
                "domain_request": None,
                "domain_result": None,
            }
        tasks = [
            item.model_copy(update={"status": TaskStatus.RUNNING}) if item.id == task.id else item
            for item in state["tasks"]
        ]
        dependency_results = {
            dependency: state.get("artifacts", {}).get(dependency)
            for dependency in task.depends_on
            if dependency in state.get("artifacts", {})
        }
        return {
            "tasks": tasks,
            "active_task_id": task.id,
            "current_agent": task.domain.value,
            "domain_request": DomainTaskRequest(
                user_id=state["user_id"],
                conversation_id=state["conversation_id"],
                request_id=state["request_id"],
                user_goal=state["user_goal"],
                task=task,
                dependency_results=dependency_results,
            ),
            "domain_result": None,
        }

    def route_task(self, state: AssistantState) -> Literal["domain_task", "done"]:
        return "domain_task" if state.get("domain_request") else "done"

    async def apply_domain_result(self, state: AssistantState) -> dict[str, Any]:
        raw_result = state.get("domain_result")
        if raw_result is None:
            raise RuntimeError("domain subgraph returned no result")
        result = DomainTaskResult.model_validate(raw_result)
        tasks = [
            item.model_copy(update={"status": result.status})
            if item.id == result.task_id
            else item
            for item in state["tasks"]
        ]
        if result.status in {TaskStatus.REJECTED, TaskStatus.FAILED}:
            tasks = self._reject_blocked_tasks(tasks)
        artifacts = dict(state.get("artifacts", {}))
        if result.artifact is not None:
            artifacts[result.task_id] = result.artifact
        answers = [*state.get("turn_answers", []), result.answer]
        return {
            "tasks": tasks,
            "artifacts": artifacts,
            "tool_results": [*state.get("tool_results", []), *result.tool_results],
            "last_answer": "\n\n".join(answers),
            "turn_answers": answers,
            "messages": [AIMessage(content=result.answer)],
            "active_task_id": None,
            "current_agent": None,
            "domain_request": None,
            "domain_result": None,
        }

    def after_domain_result(self, state: AssistantState) -> Literal["select_task", "done"]:
        return "done" if any(
            item.status == TaskStatus.WAITING_INPUT for item in state["tasks"]
        ) else "select_task"

    @staticmethod
    def _reject_blocked_tasks(tasks: list[PlannedTask]) -> list[PlannedTask]:
        result = tasks
        while True:
            blocked = {
                item.id
                for item in result
                if item.status in {TaskStatus.REJECTED, TaskStatus.FAILED}
            }
            updated = [
                item.model_copy(update={"status": TaskStatus.REJECTED})
                if set(item.depends_on) & blocked and item.status == TaskStatus.PENDING
                else item
                for item in result
            ]
            if updated == result:
                return updated
            result = updated


def build_graph(
    workflow: Workflow, domain_workflow: DomainTaskWorkflow, checkpointer: Any
) -> Any:
    graph = StateGraph(AssistantState)
    graph.add_node("understand", workflow.understand)
    graph.add_node("direct_respond", workflow.direct_respond)
    graph.add_node("plan", workflow.plan)
    graph.add_node("select_task", workflow.select_task)
    graph.add_node("domain_task", build_domain_graph(domain_workflow))
    graph.add_node("apply_domain_result", workflow.apply_domain_result)

    graph.add_edge(START, "understand")
    graph.add_conditional_edges(
        "understand",
        workflow.after_understand,
        {"plan": "plan", "direct_respond": "direct_respond"},
    )
    graph.add_edge("direct_respond", END)
    graph.add_edge("plan", "select_task")
    graph.add_conditional_edges(
        "select_task", workflow.route_task, {"domain_task": "domain_task", "done": END}
    )
    graph.add_edge("domain_task", "apply_domain_result")
    graph.add_conditional_edges(
        "apply_domain_result",
        workflow.after_domain_result,
        {"select_task": "select_task", "done": END},
    )
    return graph.compile(checkpointer=checkpointer)
