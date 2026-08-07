import pytest

from enterprise_ai_assistant.core.models import AgentName, PlannedTask
from enterprise_ai_assistant.services.capabilities import CapabilityRegistry


def test_routes_by_complete_capability_set() -> None:
    task = PlannedTask(
        title="提交差旅",
        operation="submit",
        required_capabilities={"travel.application.write"},
        risk="high",
    )
    assert CapabilityRegistry().select(task) == AgentName.TRAVEL


def test_rejects_unknown_or_cross_agent_capabilities() -> None:
    task = PlannedTask(
        title="非法组合",
        operation="submit",
        required_capabilities={"travel.application.write", "expense.claim.write"},
    )
    with pytest.raises(ValueError, match="No agent"):
        CapabilityRegistry().select(task)
