from datetime import date
from uuid import UUID, uuid4

import pytest

from enterprise_ai_assistant.core.models import AgentName
from enterprise_ai_assistant.repositories.actions import InMemoryActionRepository
from enterprise_ai_assistant.repositories.policies import InMemoryPolicyRepository
from enterprise_ai_assistant.tools import (
    ExpenseReminderInput,
    LeaveBalanceInput,
    LocalEnterpriseToolProvider,
    ToolContext,
)
from enterprise_ai_assistant.tools.registry import (
    CAPABILITY_SUMMARY,
    DomainToolRegistry,
)

DEFAULT_REQUEST_ID = UUID("00000000-0000-0000-0000-000000000002")


def context(request_id: UUID = DEFAULT_REQUEST_ID) -> ToolContext:
    return ToolContext(
        user_id="u-1",
        conversation_id=UUID("00000000-0000-0000-0000-000000000001"),
        request_id=request_id,
        task_id="task-1",
    )


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


@pytest.mark.asyncio
async def test_idempotency_is_scoped_to_request() -> None:
    actions = InMemoryActionRepository()
    provider = LocalEnterpriseToolProvider(actions, InMemoryPolicyRepository())
    payload = ExpenseReminderInput(trigger_date=date(2026, 8, 20), note="提醒报销")

    first = await provider.schedule_expense_reminder(context(), payload)
    second = await provider.schedule_expense_reminder(
        context(UUID("00000000-0000-0000-0000-000000000003")), payload
    )

    assert first.reference_id != second.reference_id
    assert len(actions.records) == 2


def test_capability_summary_covers_every_domain_with_tools() -> None:
    """模型没有能力清单就会自行编造，所以清单必须跟着领域一起长。"""
    provider = LocalEnterpriseToolProvider(InMemoryActionRepository(), InMemoryPolicyRepository())
    registry = DomainToolRegistry(provider)
    context = ToolContext(
        user_id="u-1", conversation_id=uuid4(), request_id=uuid4(), task_id="task-1"
    )
    domains = [agent for agent in AgentName if agent is not AgentName.SUPERVISOR]

    for agent in domains:
        # request_information 是通用兜底，不算领域能力。
        business = [
            item for item in registry.for_agent(agent, context)
            if item.tool.name != "request_information"
        ]
        assert business, f"{agent.value} 没有任何业务工具"
        assert agent in CAPABILITY_SUMMARY, f"{agent.value} 缺少面向用户的能力说明"

    assert set(CAPABILITY_SUMMARY) == set(domains)


def test_capability_summary_does_not_promise_status_lookups() -> None:
    """系统没有查审批进度的工具，能力自述里不能出现这类说法。"""
    text = "".join(CAPABILITY_SUMMARY.values())

    for forbidden in ("进度", "审批状态", "撤销", "修改申请"):
        assert forbidden not in text
