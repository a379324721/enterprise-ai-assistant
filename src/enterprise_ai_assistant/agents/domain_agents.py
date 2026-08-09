from typing import Any

from langsmith import traceable

from enterprise_ai_assistant.agents.base import AgentOutcome
from enterprise_ai_assistant.core.models import (
    PendingConfirmation,
    PlannedTask,
    TaskStatus,
    ToolResult,
)
from enterprise_ai_assistant.repositories.actions import ActionRepository
from enterprise_ai_assistant.repositories.policies import PolicyRepository

_SLOT_LABELS = {
    "destination": "出差地点",
    "start_date": "出发日期",
    "end_date": "行程结束日期",
    "purpose": "出差事由",
    "leave_type": "请假类型",
    "leave_start": "请假开始日期",
    "leave_end": "请假结束日期",
}


def _has_slot(slots: dict[str, Any], field: str) -> bool:
    value = slots.get(field)
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "none", "null", "unknown", "未知", "未提供"}
    return True


def _missing_slot_message(prefix: str, fields: tuple[str, ...], slots: dict[str, Any]) -> str:
    missing = [_SLOT_LABELS[field] for field in fields if not _has_slot(slots, field)]
    return f"{prefix}还需要补充：{'、'.join(missing)}。"


class TravelAgent:
    def __init__(self, policies: PolicyRepository, actions: ActionRepository) -> None:
        self._policies = policies
        self._actions = actions

    @traceable(name="travel-agent", run_type="chain")
    async def prepare(self, task: PlannedTask, slots: dict[str, Any]) -> AgentOutcome:
        if "travel.policy.read" in task.required_capabilities:
            policies = await self._policies.search(task.title, "travel")
            return AgentOutcome(
                answer=policies[0]["content"],
                tool_result=ToolResult(
                    task_id=task.id, tool="policy_search", success=True, data={"items": policies}
                ),
            )
        required = ("destination", "start_date", "end_date", "purpose")
        if any(not _has_slot(slots, name) for name in required):
            if slots.get("is_one_way") and all(
                _has_slot(slots, name) for name in ("destination", "start_date", "purpose")
            ):
                return AgentOutcome(
                    answer=(
                        "单程不需要填写返程交通，但差旅申请仍需填写行程结束日期，"
                        "也就是本次出差预计结束的日期。请补充预计结束日期。"
                    ),
                    task_status=TaskStatus.WAITING_INPUT,
                )
            return AgentOutcome(
                answer=_missing_slot_message("创建差旅申请前", required, slots),
                task_status=TaskStatus.WAITING_INPUT,
            )
        payload = {name: slots[name] for name in required}
        return AgentOutcome(
            answer="差旅申请已准备，等待确认。",
            confirmation=PendingConfirmation(
                task_id=task.id,
                action="travel.application.submit",
                summary=f"提交 {payload['start_date']} 至 {payload['end_date']} 前往 {payload['destination']} 的差旅申请",
                payload=payload,
            ),
        )

    @traceable(name="travel-agent-execute", run_type="tool")
    async def execute(
        self, task: PlannedTask, user_id: str, payload: dict[str, Any]
    ) -> AgentOutcome:
        result = await self._actions.execute_once(
            idempotency_key=f"{user_id}:{task.id}",
            action_type="travel",
            user_id=user_id,
            payload=payload,
        )
        return AgentOutcome(
            answer=f"差旅申请已提交：{result['reference_id']}",
            slot_updates={"travel_application": result},
            tool_result=ToolResult(
                task_id=task.id, tool="submit_travel", success=True, data=result
            ),
        )


class ExpenseAgent:
    def __init__(self, policies: PolicyRepository, actions: ActionRepository) -> None:
        self._policies = policies
        self._actions = actions

    @traceable(name="expense-agent", run_type="chain")
    async def prepare(self, task: PlannedTask, slots: dict[str, Any]) -> AgentOutcome:
        if "expense.policy.read" in task.required_capabilities:
            policies = await self._policies.search(task.title, "expense")
            return AgentOutcome(
                answer=policies[0]["content"],
                tool_result=ToolResult(
                    task_id=task.id, tool="policy_search", success=True, data={"items": policies}
                ),
            )
        if "expense.reminder.write" in task.required_capabilities:
            travel = slots.get("travel_application")
            if not travel:
                return AgentOutcome(
                    answer="差旅尚未创建，暂时无法安排报销提醒。",
                    task_status=TaskStatus.WAITING_INPUT,
                )
            reminder = {
                "trigger_date": travel["end_date"],
                "travel_reference": travel["reference_id"],
            }
            return AgentOutcome(
                answer=f"已安排在行程结束日 {reminder['trigger_date']} 提醒报销。",
                slot_updates={"expense_reminder": reminder},
                tool_result=ToolResult(
                    task_id=task.id, tool="schedule_expense_reminder", success=True, data=reminder
                ),
            )
        amount = slots.get("expense_amount")
        if amount is None:
            return AgentOutcome(
                answer="提交报销前还需要报销金额和票据信息。",
                task_status=TaskStatus.WAITING_INPUT,
            )
        payload = {"amount": amount, "travel_application": slots.get("travel_application")}
        return AgentOutcome(
            answer="报销单已准备，等待确认。",
            confirmation=PendingConfirmation(
                task_id=task.id,
                action="expense.claim.submit",
                summary=f"提交金额为 {amount} 的报销单",
                payload=payload,
            ),
        )

    @traceable(name="expense-agent-execute", run_type="tool")
    async def execute(
        self, task: PlannedTask, user_id: str, payload: dict[str, Any]
    ) -> AgentOutcome:
        result = await self._actions.execute_once(
            idempotency_key=f"{user_id}:{task.id}",
            action_type="expense",
            user_id=user_id,
            payload=payload,
        )
        return AgentOutcome(
            answer=f"报销单已提交：{result['reference_id']}",
            tool_result=ToolResult(
                task_id=task.id, tool="submit_expense", success=True, data=result
            ),
        )


class HRAgent:
    def __init__(self, policies: PolicyRepository, actions: ActionRepository) -> None:
        self._policies = policies
        self._actions = actions

    @traceable(name="hr-agent", run_type="chain")
    async def prepare(self, task: PlannedTask, slots: dict[str, Any]) -> AgentOutcome:
        if "hr.leave.read" in task.required_capabilities:
            policies = await self._policies.search(task.title, "hr")
            return AgentOutcome(
                answer=policies[0]["content"],
                tool_result=ToolResult(
                    task_id=task.id,
                    tool="leave_query",
                    success=True,
                    data={"balance_days": 8, "policies": policies},
                ),
            )
        required = ("leave_type", "leave_start", "leave_end")
        if any(not _has_slot(slots, key) for key in required):
            return AgentOutcome(
                answer=_missing_slot_message("提交请假前", required, slots),
                task_status=TaskStatus.WAITING_INPUT,
            )
        payload = {key: slots[key] for key in required}
        return AgentOutcome(
            answer="请假申请已准备，等待确认。",
            confirmation=PendingConfirmation(
                task_id=task.id,
                action="hr.leave.submit",
                summary=f"提交 {payload['leave_start']} 至 {payload['leave_end']} 的{payload['leave_type']}申请",
                payload=payload,
            ),
        )

    @traceable(name="hr-agent-execute", run_type="tool")
    async def execute(
        self, task: PlannedTask, user_id: str, payload: dict[str, Any]
    ) -> AgentOutcome:
        result = await self._actions.execute_once(
            idempotency_key=f"{user_id}:{task.id}",
            action_type="leave",
            user_id=user_id,
            payload=payload,
        )
        return AgentOutcome(
            answer=f"请假申请已提交：{result['reference_id']}",
            tool_result=ToolResult(task_id=task.id, tool="submit_leave", success=True, data=result),
        )


class PolicyAgent:
    def __init__(self, policies: PolicyRepository) -> None:
        self._policies = policies

    @traceable(name="policy-agent", run_type="chain")
    async def prepare(self, task: PlannedTask, slots: dict[str, Any]) -> AgentOutcome:
        del slots
        policies = await self._policies.search(task.title, "general")
        return AgentOutcome(
            answer="\n".join(item["content"] for item in policies),
            tool_result=ToolResult(
                task_id=task.id, tool="policy_search", success=True, data={"items": policies}
            ),
        )
