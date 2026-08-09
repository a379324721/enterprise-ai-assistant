from dataclasses import dataclass

from enterprise_ai_assistant.core.models import AgentName, PlannedTask


@dataclass(frozen=True)
class AgentCapability:
    agent: AgentName
    capabilities: frozenset[str]


class CapabilityRegistry:
    """将规划器的能力需求路由给最合适的能力提供方。

    此处刻意不感知用户原始文本或意图关键词。
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
        # API Schema 使用列表以兼容更多模型服务商；路由时将其转成集合，
        # 以保留精确的能力集合语义。
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
