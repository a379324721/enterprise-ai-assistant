from datetime import date
from uuid import UUID

import pytest

from enterprise_ai_assistant.core.models import AgentName
from enterprise_ai_assistant.repositories.actions import InMemoryActionRepository
from enterprise_ai_assistant.repositories.policies import InMemoryPolicyRepository
from enterprise_ai_assistant.tools import LocalEnterpriseToolProvider, ToolContext, ToolRisk
from enterprise_ai_assistant.tools.registry import DomainToolRegistry


def registry() -> DomainToolRegistry:
    provider = LocalEnterpriseToolProvider(
        InMemoryActionRepository(), InMemoryPolicyRepository()
    )
    return DomainToolRegistry(provider)


def context() -> ToolContext:
    return ToolContext(
        user_id="u-1",
        conversation_id=UUID("00000000-0000-0000-0000-000000000001"),
        request_id=UUID("00000000-0000-0000-0000-000000000002"),
        task_id="task-1",
    )


def test_each_domain_only_receives_its_allowlisted_tools() -> None:
    expense = registry().for_agent(AgentName.EXPENSE, context())
    names = {item.tool.name for item in expense}

    assert names == {
        "search_expense_policy",
        "create_expense_claim",
        "schedule_expense_reminder",
        "request_information",
    }
    assert "create_travel_application" not in names
    assert next(item for item in expense if item.tool.name == "create_expense_claim").risk == ToolRisk.WRITE


@pytest.mark.asyncio
async def test_expense_reminder_does_not_require_travel() -> None:
    reminder = next(
        item
        for item in registry().for_agent(AgentName.EXPENSE, context())
        if item.tool.name == "schedule_expense_reminder"
    )

    result = await reminder.tool.ainvoke(
        {
            "trigger_date": date(2026, 8, 20),
            "note": "提醒报销餐费和打车费",
        }
    )

    assert result["success"] is True
    assert result["status"] == "submitted"
    assert result["data"]["travel_reference"] is None


@pytest.mark.asyncio
async def test_request_information_is_a_terminal_control_tool() -> None:
    information = next(
        item
        for item in registry().for_agent(AgentName.TRAVEL, context())
        if item.tool.name == "request_information"
    )

    result = await information.tool.ainvoke(
        {"missing_fields": ["purpose"], "question": "这次出差的事由是什么？"}
    )

    assert information.terminal is True
    assert result["status"] == "needs_user_input"
