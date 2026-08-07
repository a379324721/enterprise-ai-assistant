from dataclasses import dataclass

from enterprise_ai_assistant.core.models import AgentName, PlannedTask


@dataclass(frozen=True)
class AgentCapability:
    agent: AgentName
    capabilities: frozenset[str]


class CapabilityRegistry:
    """Routes planner requirements to the best capability provider.

    This deliberately knows nothing about raw user text or intent keywords.
    """

    def __init__(self) -> None:
        self._providers = (
            AgentCapability(
                AgentName.TRAVEL, frozenset({"travel.policy.read", "travel.application.write"})
            ),
            AgentCapability(
                AgentName.EXPENSE,
                frozenset({"expense.policy.read", "expense.claim.write", "expense.reminder.write"}),
            ),
            AgentCapability(AgentName.HR, frozenset({"hr.leave.read", "hr.leave.write"})),
            AgentCapability(AgentName.POLICY, frozenset({"policy.search"})),
        )

    def select(self, task: PlannedTask) -> AgentName:
        # The API schema uses a list for broad provider compatibility; routing
        # converts it to a set to preserve exact capability-set semantics.
        required = set(task.required_capabilities)
        ranked = sorted(
            ((len(required & item.capabilities), item.agent) for item in self._providers),
            key=lambda pair: pair[0],
            reverse=True,
        )
        score, agent = ranked[0]
        if score == 0 or not required <= next(
            item.capabilities for item in self._providers if item.agent == agent
        ):
            raise ValueError(f"No agent provides all capabilities: {sorted(required)}")
        return agent
