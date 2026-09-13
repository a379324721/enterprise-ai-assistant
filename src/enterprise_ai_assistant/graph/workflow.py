import asyncio
from typing import Any, Literal, cast
from uuid import uuid4

import structlog
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from enterprise_ai_assistant.agents.supervisor import SupervisorAgent
from enterprise_ai_assistant.core.models import (
    ContextResolution,
    DomainTaskRequest,
    DomainTaskResult,
    OpenTask,
    PlannedTask,
    ShelvedPlan,
    TaskDraft,
    TaskStatus,
    TurnRelation,
)
from enterprise_ai_assistant.graph.domain import DomainTaskWorkflow, build_domain_graph
from enterprise_ai_assistant.graph.state import AssistantState
from enterprise_ai_assistant.repositories.memories import MemoryRepository

#: 超出消息窗口的历史以摘要形式回灌，需要显式标注来源，避免被当成用户当前发言。
DIGEST_HEADER = "【早先会话摘要，仅供指代消解参考，不是用户当前发言】"
#: 单条摘要的裁剪长度；standalone_request 上限 8000 字符，原样堆叠会让摘要本身变成新的成本源。
DIGEST_ITEM_MAX_CHARS = 240
#: 用户要取消事项、运行时却指认不到任何未办完的事项时的回复。
NOTHING_TO_CANCEL_REPLY = "现在没有尚未办完的事项可以放弃。已经提交的单据如果需要撤销，告诉我是哪一张。"
#: 任务被转交到无处可去时的回复。不经模型：转交链上的每个 Agent 都认为不归自己，
#: 让其中哪一个来措辞都可能说成"办不到"，而实际是没听懂该找谁。
HANDOFF_EXHAUSTED_REPLY = "这件事我没能判断该交给哪项业务办理，能换个说法，或者说明是差旅、报销、请假还是会议室方面的事吗？"

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
        # 事件循环只持有任务的弱引用，不留强引用的话后台抽取可能跑到一半被回收。
        self._background: set[asyncio.Task[None]] = set()

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
        # 而不是 digest 长度定位，understand（当轮摘要尚未写入）和 select_task
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
        """轮次结束后抽取值得长期保留的信息，抽取本身放到后台跑。

        抽取是一次完整的模型调用，而 remember 是每轮的最后一个节点：同步等它，
        回答早已流完，任务面板和单据却要再晚几秒才到，这段时间里会话还占着执行锁，
        用户发下一句会被 409 挡回。记忆只影响以后的轮次，不值得让这一轮等。
        代价是抽取的 token 不再计入本轮的用量统计。

        等待用户补充输入的轮次不抽取：此时字段还没谈定，把半成品写进画像会让
        下一轮拿着错误的默认值去预填。写入失败同样只记日志，不影响已完成的回答。
        """
        if self._memories is None:
            return {}
        if any(task.status == TaskStatus.WAITING_INPUT for task in state.get("tasks", [])):
            return {}
        known = [record.render() for record in state.get("memories", [])]
        # 只把用户说过的话交给抽取器。助手的回答里会出现会议室名、目的地、
        # 称呼这些内容，模型很容易把它们当成用户的稳定属性写进画像——
        # 实测就出现过把出差地"上海分部"记成常驻办公地。prompt 里的
        # "不得推断"挡不住，这里从输入上断掉。
        spoken = [turn for turn in self._conversation(state) if turn["role"] == "user"]
        task = asyncio.create_task(
            self._extract_memories(
                state["user_id"], state["conversation_id"], spoken, known, self._memories
            )
        )
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return {}

    async def _extract_memories(
        self,
        user_id: str,
        conversation_id: Any,
        spoken: list[dict[str, str]],
        known: list[str],
        repository: MemoryRepository,
    ) -> None:
        try:
            extraction = await self.supervisor.extract_memories(spoken, known)
            await repository.upsert(
                user_id, extraction.memories, source_conversation_id=conversation_id
            )
        except Exception:
            logger.warning("memory_write_failed", user_id=user_id)

    async def drain_background(self) -> None:
        """等后台记忆抽取全部结束。进程关停前调用，否则正在写的记忆会丢。"""
        if self._background:
            await asyncio.gather(*self._background, return_exceptions=True)

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

    @staticmethod
    def _open_tasks_of(
        plan_id: str,
        tasks: list[PlannedTask],
        drafts: dict[str, TaskDraft],
        *,
        shelved: bool,
    ) -> list[OpenTask]:
        return [
            OpenTask(
                plan_id=plan_id,
                task_id=task.id,
                title=task.title,
                domain=task.domain,
                missing_fields=(
                    TaskDraft.model_validate(drafts[task.id]).missing_fields
                    if task.id in drafts
                    else []
                ),
                shelved=shelved,
            )
            for task in tasks
            if task.status == TaskStatus.WAITING_INPUT
        ]

    @staticmethod
    def _plan_id(state: AssistantState) -> str:
        # 早于 plan_id 的检查点没有这个字段，给当前计划补一个，搁置后才能被指认。
        return state.get("plan_id") or uuid4().hex[:8]

    def _shelve_current(self, state: AssistantState) -> ShelvedPlan:
        return ShelvedPlan(
            plan_id=self._plan_id(state),
            user_goal=state.get("user_goal", ""),
            tasks=list(state.get("tasks", [])),
            artifacts=dict(state.get("artifacts", {})),
            drafts=dict(state.get("drafts", {})),
        )

    @staticmethod
    def _requeue_waiting(tasks: list[PlannedTask]) -> list[PlannedTask]:
        return [
            task.model_copy(update={"status": TaskStatus.PENDING})
            if task.status == TaskStatus.WAITING_INPUT
            else task
            for task in tasks
        ]

    def _target(
        self,
        state: AssistantState,
        context: ContextResolution,
        current_open: list[OpenTask],
    ) -> ShelvedPlan | Literal["current"] | None:
        """解析 continue / cancel 指向的事项；指不到任何未办完的事项时返回 None。"""
        shelved = state.get("shelved_plans", [])
        target_id = context.target_plan_id
        if target_id:
            for plan in shelved:
                if plan.plan_id == target_id:
                    return plan
        if current_open:
            return "current"
        # 当前没有未办完的事，而被搁置的只有一件："继续刚才那个"指的只能是它。
        # 多于一件时不猜，宁可落回常规路径重新规划，也不把补充信息塞给错的事项。
        if len(shelved) == 1:
            return shelved[0]
        return None

    @staticmethod
    def _recover_interrupted(
        tasks: list[PlannedTask], drafts: dict[str, TaskDraft]
    ) -> list[PlannedTask]:
        """把上一轮中途失败、停在 RUNNING 的任务放回可续跑的状态。

        同一会话同时只有一次执行，understand 开始时不可能有任务真的在跑。停在 RUNNING
        说明上一轮在领域子图里抛了异常（例如模型服务 403），任务状态没来得及写回。
        不恢复的话它既不算待补充也不会被调度：用户重发补充信息时 Supervisor 看不到
        未办完的事项，只能重新规划，草稿随之丢失。有草稿说明它本来停在待补充上，
        放回 WAITING_INPUT；没有草稿说明第一次执行就失败了，放回 PENDING。
        """
        return [
            task.model_copy(
                update={
                    "status": TaskStatus.WAITING_INPUT
                    if task.id in drafts
                    else TaskStatus.PENDING
                }
            )
            if task.status == TaskStatus.RUNNING
            else task
            for task in tasks
        ]

    async def understand(self, state: AssistantState) -> dict[str, Any]:
        recovered = self._recover_interrupted(state.get("tasks", []), state.get("drafts", {}))
        state = cast(AssistantState, {**state, "tasks": recovered})
        shelved = list(state.get("shelved_plans", []))
        plan_id = self._plan_id(state)
        current_open = self._open_tasks_of(
            plan_id, state.get("tasks", []), state.get("drafts", {}), shelved=False
        )
        open_tasks = [
            *current_open,
            *(
                item
                for plan in shelved
                for item in self._open_tasks_of(
                    plan.plan_id, plan.tasks, plan.drafts, shelved=True
                )
            ),
        ]
        # 只给 key 不给 value：Supervisor 做的是相关性筛选，不读取记忆内容，
        # 也就无从用它补写领域字段。代价是直接回复不做基于档案的个性化。
        context = await self.supervisor.resolve_context(
            self._conversation(state),
            [record.key for record in state.get("memories", [])],
            open_tasks,
            [action.render() for action in state.get("recent_actions", [])],
            state.get("user_name", ""),
        )
        update: dict[str, Any] = {
            "understanding": context.model_dump(mode="json"),
            "history_digest": self._extend_digest(state, context.standalone_request),
            "plan_id": plan_id,
            # 恢复后的任务状态要写回检查点，否则闲聊轮之后它还停在 RUNNING。
            "tasks": recovered,
            "tool_results": [],
            "active_task_id": None,
            "current_agent": None,
            "domain_request": None,
            "domain_result": None,
            "last_answer": "",
            "turn_answers": [],
        }
        relation = context.turn_relation
        target = (
            self._target(state, context, current_open)
            if relation in {TurnRelation.CONTINUE, TurnRelation.CANCEL}
            else None
        )

        if relation == TurnRelation.CONTINUE and target is not None:
            # 补充信息不重新规划：重新拆出来的任务 id、标题和粒度都可能变，前置任务的
            # 产物也会随 artifacts 一起被清掉。原计划保持不动，只把待补充的任务放回队列。
            # user_goal 同样不动：它是整件事的目标，界面据此展示；本轮的补充经
            # understanding 交给领域 Agent。
            if target == "current":
                update["tasks"] = self._requeue_waiting(state["tasks"])
                return update
            # 恢复被搁置的事项：它整体换回当前计划，当前计划若还没办完就换下去搁置。
            remaining = [plan for plan in shelved if plan.plan_id != target.plan_id]
            if current_open:
                remaining.append(self._shelve_current(state))
            update.update(
                {
                    "plan_id": target.plan_id,
                    "user_goal": target.user_goal,
                    "tasks": self._requeue_waiting(target.tasks),
                    "artifacts": target.artifacts,
                    "drafts": target.drafts,
                    "shelved_plans": remaining,
                }
            )
            return update

        if relation == TurnRelation.CANCEL:
            # 取消的回复由运行时按实际处理结果写，不用模型写的：模型在同一次输出里判断
            # 要取消哪件事，却无从知道运行时能不能指认到它，照它的话说就可能"取消了"
            # 一件其实还在的事。
            if target is None:
                return self._reply(update, NOTHING_TO_CANCEL_REPLY)
            # 只放弃还没提交的部分。已经提交的单据是业务事实，撤销要走领域的撤销工具
            # 和人工确认，这里不替用户处理它们。
            unfinished = {TaskStatus.WAITING_INPUT, TaskStatus.PENDING}
            plan_tasks = state["tasks"] if target == "current" else target.tasks
            if target == "current":
                update["tasks"] = [
                    task.model_copy(update={"status": TaskStatus.REJECTED})
                    if task.status in unfinished
                    else task
                    for task in state["tasks"]
                ]
                update["drafts"] = {}
            else:
                update["shelved_plans"] = [
                    plan for plan in shelved if plan.plan_id != target.plan_id
                ]
            return self._reply(update, self._cancelled_text(plan_tasks, unfinished))

        if not context.requires_task_planning:
            # 闲聊、道谢、清单外的诉求都不动计划：待补充期间插一句"好的稍等"，
            # 用户回过头补充时任务还在。
            return self._reply(update, context.reply)

        # 新的业务请求。当前计划没办完就静默搁置，用户说"继续刚才那个"时还能换回来。
        if current_open:
            shelved.append(self._shelve_current(state))
        update.update(
            {
                "plan_id": uuid4().hex[:8],
                "user_goal": context.standalone_request,
                "tasks": [],
                "artifacts": {},
                "drafts": {},
                "shelved_plans": shelved,
            }
        )
        return update

    @staticmethod
    def _reply(update: dict[str, Any], text: str) -> dict[str, Any]:
        answer = text.strip()
        return {
            **update,
            "last_answer": answer,
            "turn_answers": [answer],
            "messages": [AIMessage(content=answer)],
        }

    @staticmethod
    def _cancelled_text(tasks: list[PlannedTask], unfinished: set[TaskStatus]) -> str:
        dropped = "、".join(task.title for task in tasks if task.status in unfinished)
        text = f"好的，这件事不办了，已放弃：{dropped}。"
        if any(task.status == TaskStatus.COMPLETED for task in tasks):
            text += "其中已经办完的部分不受影响，需要撤销的话告诉我是哪一张单据。"
        return text

    def after_understand(self, state: AssistantState) -> Literal["plan", "select_task", "done"]:
        # understand 已经回复了用户（直接回复或取消），本轮不再执行任何任务。
        if state.get("turn_answers"):
            return "done"
        context = ContextResolution.model_validate(state["understanding"])
        # understand 只在认定为续跑时才会把任务放回 PENDING；Supervisor 误报 continue
        # 却指不到任何未办完的事项时，这里自然落回常规路径。
        if context.turn_relation == TurnRelation.CONTINUE and any(
            task.status == TaskStatus.PENDING for task in state["tasks"]
        ):
            return "select_task"
        return "plan"

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
        # 领域 Agent 要的是本轮改写后的请求，不是计划的总目标：续跑轮里用户的补充
        # （"当天往返""选第一间"）只在本轮的 standalone_request 里。
        context = ContextResolution.model_validate(state["understanding"])
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
                user_goal=context.standalone_request,
                task=task,
                dependency_results=dependency_results,
                memories=self._relevant_memories(state),
                recent_actions=list(state.get("recent_actions", [])),
                draft=state.get("drafts", {}).get(task.id),
                assistant_replies=[
                    turn["content"]
                    for turn in self._conversation(state)
                    if turn["role"] == "assistant"
                ],
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
        if result.status == TaskStatus.HANDED_OFF:
            return self._reroute(state, result)
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
        drafts = dict(state.get("drafts", {}))
        if result.draft is not None:
            drafts[result.task_id] = result.draft
        else:
            drafts.pop(result.task_id, None)
        answers = [*state.get("turn_answers", []), result.answer]
        return {
            "tasks": tasks,
            "artifacts": artifacts,
            "drafts": drafts,
            "tool_results": [*state.get("tool_results", []), *result.tool_results],
            "last_answer": "\n\n".join(answers),
            "turn_answers": answers,
            "messages": [AIMessage(content=result.answer)],
            "active_task_id": None,
            "current_agent": None,
            "domain_request": None,
            "domain_result": None,
        }

    MAX_HANDOFFS = 2

    def _reroute(self, state: AssistantState, result: DomainTaskResult) -> dict[str, Any]:
        """把领域 Agent 交还的任务改派给它指出的领域。

        不回到 Supervisor 重新理解：它读的还是同一段会话，大概率再分错一次，还要多等一次
        模型调用。领域 Agent 看过自己的工具清单和其他领域的能力，它指出的去处比重新猜更准。
        去过的领域不再去、次数有上限，超出就判失败请用户换个说法，不让任务来回踢。
        """
        task = next(item for item in state["tasks"] if item.id == result.task_id)
        target = result.handoff_to
        visited = {*task.handed_off_from, task.domain}
        update: dict[str, Any] = {
            "tool_results": [*state.get("tool_results", []), *result.tool_results],
            "active_task_id": None,
            "current_agent": None,
            "domain_request": None,
            "domain_result": None,
        }
        drafts = dict(state.get("drafts", {}))
        drafts.pop(task.id, None)
        update["drafts"] = drafts
        if (
            target is None
            or target in visited
            or len(task.handed_off_from) >= self.MAX_HANDOFFS
        ):
            logger.warning(
                "task_handoff_refused",
                task_id=task.id,
                domain=task.domain.value,
                target=target.value if target else None,
            )
            failed = [
                item.model_copy(update={"status": TaskStatus.FAILED})
                if item.id == task.id
                else item
                for item in state["tasks"]
            ]
            answers = [*state.get("turn_answers", []), HANDOFF_EXHAUSTED_REPLY]
            return {
                **update,
                "tasks": self._reject_blocked_tasks(failed),
                "last_answer": "\n\n".join(answers),
                "turn_answers": answers,
                "messages": [AIMessage(content=HANDOFF_EXHAUSTED_REPLY)],
            }
        logger.info(
            "task_handed_off", task_id=task.id, source=task.domain.value, target=target.value
        )
        rerouted = task.model_copy(
            update={
                "domain": target,
                "status": TaskStatus.PENDING,
                "handed_off_from": [*task.handed_off_from, task.domain],
            }
        )
        return {
            **update,
            "tasks": [rerouted if item.id == task.id else item for item in state["tasks"]],
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
        {"plan": "plan", "select_task": "select_task", "done": "remember"},
    )
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
