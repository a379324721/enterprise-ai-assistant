from typing import Any, Literal

import structlog
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
from enterprise_ai_assistant.repositories.memories import MemoryRepository

#: 超出消息窗口的历史以摘要形式回灌，需要显式标注来源，避免被当成用户当前发言。
DIGEST_HEADER = "【早先会话摘要，仅供指代消解参考，不是用户当前发言】"
#: 单条摘要的裁剪长度；standalone_request 上限 8000 字符，原样堆叠会让摘要本身变成新的成本源。
DIGEST_ITEM_MAX_CHARS = 240

logger = structlog.get_logger()


class Workflow:
    """外层工作流只负责上下文理解、任务规划和领域子图调度。"""

    def __init__(
        self,
        supervisor: SupervisorAgent,
        *,
        history_window: int = 12,
        digest_turns: int = 20,
        memories: MemoryRepository | None = None,
        recall_limit: int = 20,
        recent_action_limit: int = 5,
    ) -> None:
        self.supervisor = supervisor
        self._history_window = history_window
        self._digest_turns = digest_turns
        # memories 为 None 表示长期记忆未启用；此时 recall/remember 退化成空操作，
        # 图结构保持不变，开关切换不需要重建检查点。
        self._memories = memories
        self._recall_limit = recall_limit
        self._recent_action_limit = recent_action_limit

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

    async def recall(self, state: AssistantState) -> dict[str, Any]:
        """读取该用户的长期画像和近期单据。

        记忆是锦上添花的输入，不是执行前提：仓储不可用时返回空结果继续本轮，
        表现退化成没有记忆的旧行为，而不是让整轮请求失败。
        """
        if self._memories is None:
            return {"memories": [], "recent_actions": []}
        user_id = state["user_id"]
        try:
            memories = await self._memories.list_memories(user_id, self._recall_limit)
            recent_actions = await self._memories.recent_actions(
                user_id, self._recent_action_limit
            )
        except Exception:
            logger.warning("memory_recall_failed", user_id=user_id)
            return {"memories": [], "recent_actions": []}
        return {"memories": memories, "recent_actions": recent_actions}

    async def remember(self, state: AssistantState) -> dict[str, Any]:
        """轮次结束后抽取值得长期保留的信息。

        等待用户补充输入的轮次不抽取：此时字段还没谈定，把半成品写进画像会让
        下一轮拿着错误的默认值去预填。写入失败同样只记日志，不影响已完成的回答。
        """
        if self._memories is None:
            return {}
        if any(task.status == TaskStatus.WAITING_INPUT for task in state.get("tasks", [])):
            return {}
        try:
            known = [record.render() for record in state.get("memories", [])]
            extraction = await self.supervisor.extract_memories(
                self._conversation(state), known
            )
            await self._memories.upsert(
                state["user_id"],
                extraction.memories,
                source_conversation_id=state["conversation_id"],
            )
        except Exception:
            logger.warning("memory_write_failed", user_id=state["user_id"])
        return {}

    def _relevant_memories(self, state: AssistantState) -> list[str]:
        """按 understand 选出的 key 过滤记忆。

        全量下发的成本是 记忆条数 × 任务数，且无关记忆会成为领域模型的噪音。
        筛选发生在理解阶段，执行链路上的节点只拿到与本次请求相关的那几条。
        """
        understanding = state.get("understanding")
        if not understanding:
            return []
        selected = set(ContextResolution.model_validate(understanding).relevant_memory_keys)
        return [
            record.render() for record in state.get("memories", []) if record.key in selected
        ]

    async def understand(self, state: AssistantState) -> dict[str, Any]:
        # 只给 key 不给 value：Supervisor 做的是相关性筛选，不读取记忆内容，
        # 也就无从用它补写领域字段。
        context = await self.supervisor.resolve_context(
            self._conversation(state),
            [record.key for record in state.get("memories", [])],
        )
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
        # 闲聊节点同样只吃理解阶段的输出，不回头读原始会话。
        # 带上档案让回答贴合这位用户，单据清单不含审批结果，prompt 禁止推断状态。
        response = await self.supervisor.respond_direct(
            ContextResolution.model_validate(state["understanding"]),
            self._relevant_memories(state),
            [action.render() for action in state.get("recent_actions", [])],
            state.get("user_name", ""),
        )
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
                user_name=state.get("user_name", ""),
                conversation_id=state["conversation_id"],
                request_id=state["request_id"],
                user_goal=state["user_goal"],
                task=task,
                dependency_results=dependency_results,
                memories=self._relevant_memories(state),
                recent_actions=list(state.get("recent_actions", [])),
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
    graph.add_node("recall", workflow.recall)
    graph.add_node("remember", workflow.remember)
    graph.add_node("understand", workflow.understand)
    graph.add_node("direct_respond", workflow.direct_respond)
    graph.add_node("plan", workflow.plan)
    graph.add_node("select_task", workflow.select_task)
    graph.add_node("domain_task", build_domain_graph(domain_workflow))
    graph.add_node("apply_domain_result", workflow.apply_domain_result)

    # 每轮开头召回一次，结尾统一经 remember 收口：所有终止分支都汇到同一个节点，
    # 新增结束路径时不会漏掉记忆写入。
    graph.add_edge(START, "recall")
    graph.add_edge("recall", "understand")
    graph.add_conditional_edges(
        "understand",
        workflow.after_understand,
        {"plan": "plan", "direct_respond": "direct_respond"},
    )
    graph.add_edge("direct_respond", "remember")
    graph.add_edge("plan", "select_task")
    graph.add_conditional_edges(
        "select_task",
        workflow.route_task,
        {"domain_task": "domain_task", "done": "remember"},
    )
    graph.add_edge("domain_task", "apply_domain_result")
    graph.add_conditional_edges(
        "apply_domain_result",
        workflow.after_domain_result,
        {"select_task": "select_task", "done": "remember"},
    )
    graph.add_edge("remember", END)
    return graph.compile(checkpointer=checkpointer)
