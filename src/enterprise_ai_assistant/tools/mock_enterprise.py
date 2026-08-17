from typing import Any

from enterprise_ai_assistant.repositories.actions import ActionRepository
from enterprise_ai_assistant.repositories.policies import PolicyRepository
from enterprise_ai_assistant.tools.contracts import (
    BusinessToolOutcome,
    ExpenseClaimInput,
    ExpenseReminderInput,
    LeaveBalanceInput,
    LeaveRequestInput,
    PolicySearchInput,
    ToolContext,
    ToolEnvironment,
    TravelApplicationInput,
)


class MockEnterpriseToolProvider:
    """可替换的企业系统模拟器；所有结果均显式标记为 mock。"""

    environment = ToolEnvironment.MOCK

    def __init__(
        self,
        actions: ActionRepository,
        policies: PolicyRepository,
        *,
        annual_leave_balance: float = 8,
    ) -> None:
        self._actions = actions
        self._policies = policies
        self._annual_leave_balance = annual_leave_balance

    async def search_policy(self, payload: PolicySearchInput) -> BusinessToolOutcome:
        items = await self._policies.search(payload.query, payload.domain, payload.limit)
        return BusinessToolOutcome(
            tool="search_policy",
            success=True,
            environment=self.environment,
            status="simulated_completed",
            data={"items": items},
        )

    async def _record_write(
        self,
        *,
        tool: str,
        action_type: str,
        context: ToolContext,
        payload: dict[str, Any],
    ) -> BusinessToolOutcome:
        recorded = await self._actions.execute_once(
            idempotency_key=context.idempotency_key,
            action_type=action_type,
            user_id=context.user_id,
            payload={"environment": self.environment.value, **payload},
        )
        return BusinessToolOutcome(
            tool=tool,
            success=True,
            environment=self.environment,
            status="simulated_submitted",
            reference_id=str(recorded["reference_id"]),
            data=recorded,
        )

    async def create_travel_application(
        self, context: ToolContext, payload: TravelApplicationInput
    ) -> BusinessToolOutcome:
        return await self._record_write(
            tool="create_travel_application",
            action_type="travel_application",
            context=context,
            payload=payload.model_dump(mode="json"),
        )

    async def create_expense_claim(
        self, context: ToolContext, payload: ExpenseClaimInput
    ) -> BusinessToolOutcome:
        return await self._record_write(
            tool="create_expense_claim",
            action_type="expense_claim",
            context=context,
            payload=payload.model_dump(mode="json"),
        )

    async def schedule_expense_reminder(
        self, context: ToolContext, payload: ExpenseReminderInput
    ) -> BusinessToolOutcome:
        return await self._record_write(
            tool="schedule_expense_reminder",
            action_type="expense_reminder",
            context=context,
            payload=payload.model_dump(mode="json"),
        )

    async def get_leave_balance(
        self, context: ToolContext, payload: LeaveBalanceInput
    ) -> BusinessToolOutcome:
        del context
        balance = self._annual_leave_balance if payload.leave_type == "annual" else 0
        return BusinessToolOutcome(
            tool="get_leave_balance",
            success=True,
            environment=self.environment,
            status="simulated_completed",
            data={"leave_type": payload.leave_type, "balance_days": balance},
        )

    async def submit_leave_request(
        self, context: ToolContext, payload: LeaveRequestInput
    ) -> BusinessToolOutcome:
        return await self._record_write(
            tool="submit_leave_request",
            action_type="leave_request",
            context=context,
            payload=payload.model_dump(mode="json"),
        )
