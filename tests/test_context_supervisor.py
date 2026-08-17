import pytest

from enterprise_ai_assistant.agents.supervisor import SupervisorAgent
from enterprise_ai_assistant.core.models import ContextResolution, TaskPlan


class CapturingPlanningService:
    def __init__(self) -> None:
        self.conversation: list[dict[str, str]] = []

    async def resolve_context(
        self, conversation: list[dict[str, str]]
    ) -> ContextResolution:
        self.conversation = conversation
        return ContextResolution(
            standalone_request="将上一条差旅申请的开始日期改为下周三",
            intent_summary="修改差旅日期",
            referenced_task_ids=["task-1"],
        )

    async def plan(self, context: ContextResolution) -> TaskPlan:
        raise AssertionError(f"not used: {context}")


@pytest.mark.asyncio
async def test_supervisor_receives_complete_conversation() -> None:
    planning = CapturingPlanningService()
    supervisor = SupervisorAgent(planning)
    conversation: list[dict[str, str]] = [
        {"role": "user", "content": "帮我申请去上海出差"},
        {"role": "assistant", "content": "还需要开始日期"},
        {"role": "user", "content": "改成下周三"},
    ]

    result = await supervisor.resolve_context(conversation)

    assert planning.conversation == conversation
    assert result.referenced_task_ids == ["task-1"]
    assert not hasattr(result, "inferred_slots")
