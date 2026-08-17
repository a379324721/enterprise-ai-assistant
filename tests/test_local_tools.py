from datetime import date

import pytest

from enterprise_ai_assistant.repositories.actions import InMemoryActionRepository
from enterprise_ai_assistant.repositories.policies import InMemoryPolicyRepository
from enterprise_ai_assistant.tools import (
    ExpenseReminderInput,
    LeaveBalanceInput,
    LocalEnterpriseToolProvider,
    ToolContext,
)


def context(key: str = "u-1:task-1:call-1") -> ToolContext:
    return ToolContext(user_id="u-1", task_id="task-1", idempotency_key=key)


@pytest.mark.asyncio
async def test_local_write_is_idempotent() -> None:
    actions = InMemoryActionRepository()
    provider = LocalEnterpriseToolProvider(actions, InMemoryPolicyRepository())
    payload = ExpenseReminderInput(trigger_date=date(2026, 8, 20), note="提醒报销打车费")

    first = await provider.schedule_expense_reminder(context(), payload)
    repeated = await provider.schedule_expense_reminder(context(), payload)

    assert first == repeated
    assert first.status == "submitted"
    assert first.data["status"] == "recorded"
    assert len(actions.records) == 1


@pytest.mark.asyncio
async def test_local_leave_balance_uses_configured_backend_value() -> None:
    provider = LocalEnterpriseToolProvider(
        InMemoryActionRepository(), InMemoryPolicyRepository(), annual_leave_balance=6.5
    )

    result = await provider.get_leave_balance(context(), LeaveBalanceInput())

    assert result.status == "completed"
    assert result.data["balance_days"] == 6.5
