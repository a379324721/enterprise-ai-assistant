from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from enterprise_ai_assistant.agents.supervisor import SupervisorAgent
from enterprise_ai_assistant.core.models import (
    ContextResolution,
    OpenTask,
    TaskPlan,
    TurnRelation,
)


class CapturingPlanningService:
    def __init__(self) -> None:
        self.conversation: list[dict[str, str]] = []

    async def resolve_context(
        self,
        conversation: list[dict[str, str]],
        memory_keys: Sequence[str] = (),
        open_tasks: Sequence[OpenTask] = (),
        recent_actions: Sequence[str] = (),
        user_name: str = "",
    ) -> ContextResolution:
        self.conversation = conversation
        return ContextResolution(
            standalone_request="将上一条差旅申请的开始日期改为下周三",
            intent_summary="修改差旅日期",
            requires_task_planning=True,
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


def test_a_turn_that_runs_nothing_must_carry_a_reply() -> None:
    """不执行任务、也不取消事项的轮次没有别的节点会开口，缺回复必须当成输出无效。"""
    with pytest.raises(ValidationError, match="必须写 reply"):
        ContextResolution(
            standalone_request="你好", intent_summary="问候", requires_task_planning=False
        )


def test_cancel_and_task_turns_need_no_reply() -> None:
    # 取消由运行时按实际结果回复，执行任务由领域 Agent 回复。
    ContextResolution(
        standalone_request="不出差了",
        intent_summary="取消",
        requires_task_planning=False,
        turn_relation=TurnRelation.CANCEL,
    )
    ContextResolution(standalone_request="查年假", intent_summary="查询", requires_task_planning=True)
