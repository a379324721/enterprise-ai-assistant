import re
from collections.abc import Mapping
from datetime import date, timedelta
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
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
    ExtractedSlots,
    GoalUnderstanding,
    PlannedTask,
    TaskRun,
    TaskStatus,
)
from enterprise_ai_assistant.graph.state import AssistantState


def _relative_date(user_text: str) -> str | None:
    weekday_match = re.search(r"下(?:周|星期)([一二三四五六日天])", user_text)
    if weekday_match:
        weekday = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
        next_monday = date.today() + timedelta(days=7 - date.today().weekday())
        return (next_monday + timedelta(days=weekday[weekday_match.group(1)])).isoformat()
    offsets = (("后天", 2), ("明天", 1), ("今天", 0))
    offset = next((days for phrase, days in offsets if phrase in user_text), None)
    return (date.today() + timedelta(days=offset)).isoformat() if offset is not None else None


def _explicit_travel_date_slots(user_text: str) -> dict[str, str]:
    slots: dict[str, str] = {}
    for phrase, offset in (("今天", 0), ("明天", 1), ("后天", 2)):
        value = (date.today() + timedelta(days=offset)).isoformat()
        if any(f"{phrase}{verb}" in user_text for verb in ("出发", "启程", "开始")):
            slots["start_date"] = value
        if any(f"{phrase}{verb}" in user_text for verb in ("回来", "返回", "返程", "结束")):
            slots["end_date"] = value
    return slots


def _single_day_leave_slots(
    user_text: str,
    understanding: GoalUnderstanding,
    tasks: list[PlannedTask],
) -> dict[str, str]:
    if not any(phrase in user_text for phrase in ("一天", "全天", "一整天")):
        return {}
    leave_context = any(
        phrase in f"{user_text}{understanding.normalized_goal}"
        for phrase in ("请假", "年假", "事假", "病假", "调休")
    ) or any("hr.leave.write" in task.required_capabilities for task in tasks)
    if not leave_context:
        return {}
    leave_date = _relative_date(user_text)
    if leave_date is None:
        return {}
    return {"leave_start": leave_date, "leave_end": leave_date}


def _travel_mode_slots(user_text: str) -> dict[str, bool]:
    if any(phrase in user_text for phrase in ("单程", "不返回", "不返程", "不回程")):
        return {"is_one_way": True}
    if any(phrase in user_text for phrase in ("往返", "会返回", "有返程")):
        return {"is_one_way": False}
    return {}


def _travel_follow_up_date_slots(
    user_text: str,
    slots: dict[str, Any],
    tasks: list[PlannedTask],
) -> dict[str, str]:
    supplied_date = _relative_date(user_text)
    if supplied_date is None:
        return {}
    is_travel_application = any(
        "travel.application.write" in task.required_capabilities for task in tasks
    )
    if not is_travel_application:
        return {}
    start_date = slots.get("start_date")
    end_date = slots.get("end_date")
    if start_date and not end_date:
        return {"end_date": supplied_date}
    if not start_date and end_date:
        return {"start_date": supplied_date}
    return {}


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
        # 只有这一边界会读取聊天历史；领域 Agent 接收的是状态投影。
        messages = state["messages"]
        user_text = str(messages[-1].content)
        history_lines = []
        for message in messages[-13:-1]:
            role = "用户" if isinstance(message, HumanMessage) else "助手"
            history_lines.append(f"{role}：{message.content}")
        understanding = await self.supervisor.understand(
            user_text, "\n".join(history_lines)
        )
        model_slots = understanding.inferred_slots.model_dump(exclude_none=True)
        travel_date_slots = _travel_follow_up_date_slots(
            user_text, state.get("slots", {}), state.get("tasks", [])
        )
        if "end_date" in travel_date_slots and not any(
            phrase in user_text for phrase in ("出发", "开始", "启程")
        ):
            model_slots.pop("start_date", None)
        inferred_slots = {
            **model_slots,
            **_explicit_travel_date_slots(user_text),
            **_single_day_leave_slots(user_text, understanding, state.get("tasks", [])),
            **_travel_mode_slots(user_text),
            **travel_date_slots,
        }
        understanding = understanding.model_copy(
            update={"inferred_slots": ExtractedSlots.model_validate(inferred_slots)}
        )
        return {
            "user_goal": understanding.normalized_goal,
            "understanding": understanding.model_dump(mode="json"),
            "slots": {**state.get("slots", {}), **inferred_slots},
        }

    async def plan(self, state: AssistantState) -> dict[str, Any]:
        understanding = GoalUnderstanding.model_validate(state["understanding"])
        plan = await self.supervisor.plan(understanding)
        update: dict[str, Any] = {
            "user_goal": plan.user_goal,
            "tasks": plan.tasks,
            "slots": {
                **state.get("slots", {}),
                **plan.extracted_slots.model_dump(exclude_none=True),
            },
        }
        previous_tasks = state.get("tasks", [])
        if previous_tasks:
            update["task_history"] = [
                *state.get("task_history", []),
                TaskRun(user_goal=state.get("user_goal", ""), tasks=previous_tasks),
            ]
        if not plan.tasks:
            answer = plan.direct_answer.strip() or (
                "我暂时没有识别到需要办理的企业事务。你可以告诉我需要处理的差旅、"
                "报销、请假或制度查询。"
            )
            update["last_answer"] = answer
            update["messages"] = [AIMessage(content=answer)]
        return update

    async def select_task(self, state: AssistantState) -> dict[str, Any]:
        task = self.supervisor.next_runnable(state["tasks"])
        if task is None:
            return {"active_task_id": None, "current_agent": None}
        try:
            agent = self.supervisor.route(task)
        except ValueError:
            capabilities = "、".join(task.required_capabilities)
            answer = (
                f"当前系统暂不支持“{task.title}”（{capabilities}）。"
                "如果你是在查询刚提交的申请，提交结果和编号可在当前对话的助手回复中查看。"
            )
            tasks = [
                item.model_copy(update={"status": TaskStatus.FAILED})
                if item.id == task.id
                else item
                for item in state["tasks"]
            ]
            return {
                "tasks": tasks,
                "active_task_id": None,
                "current_agent": None,
                "last_answer": answer,
                "messages": [AIMessage(content=answer)],
            }
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
            tasks = [
                item.model_copy(update={"status": TaskStatus.RUNNING})
                if item.id == pending.task_id
                else item
                for item in state["tasks"]
            ]
            return {"tasks": tasks, "confirmation_approved": True}
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
            self._active_task(state),
            state["user_id"],
            pending.payload,
            idempotency_key=f"{pending.action}:{pending.confirmation_id}",
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
        task_status = outcome.task_status
        if task_status is None and outcome.confirmation is not None:
            task_status = TaskStatus.WAITING_CONFIRMATION
        if task_status is not None:
            update["tasks"] = [
                item.model_copy(update={"status": task_status})
                if item.id == state["active_task_id"]
                else item
                for item in state["tasks"]
            ]
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
