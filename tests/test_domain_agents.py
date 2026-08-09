import pytest

from enterprise_ai_assistant.agents.domain_agents import HRAgent, TravelAgent
from enterprise_ai_assistant.core.models import PlannedTask, TaskStatus
from enterprise_ai_assistant.repositories.actions import InMemoryActionRepository
from enterprise_ai_assistant.repositories.policies import InMemoryPolicyRepository


@pytest.mark.asyncio
async def test_travel_agent_uses_chinese_labels_for_missing_slots() -> None:
    agent = TravelAgent(InMemoryPolicyRepository(), InMemoryActionRepository())
    task = PlannedTask(
        title="创建差旅申请",
        operation="submit",
        required_capabilities=["travel.application.write"],
    )

    outcome = await agent.prepare(task, {"start_date": "2026-08-10"})

    assert outcome.answer == "创建差旅申请前还需要补充：出差地点、返回日期、出差事由。"
    assert outcome.task_status == TaskStatus.WAITING_INPUT


@pytest.mark.asyncio
async def test_hr_agent_uses_chinese_labels_for_missing_slots() -> None:
    agent = HRAgent(InMemoryPolicyRepository(), InMemoryActionRepository())
    task = PlannedTask(
        title="提交请假申请",
        operation="submit",
        required_capabilities=["hr.leave.write"],
    )

    outcome = await agent.prepare(task, {})

    assert outcome.answer == "提交请假前还需要补充：请假类型、请假开始日期、请假结束日期。"
    assert outcome.task_status == TaskStatus.WAITING_INPUT
