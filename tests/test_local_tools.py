from datetime import date, time
from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from enterprise_ai_assistant.core.models import AgentName
from enterprise_ai_assistant.repositories.actions import InMemoryActionRepository
from enterprise_ai_assistant.repositories.policies import InMemoryPolicyRepository
from enterprise_ai_assistant.tools import (
    ExpenseClaimInput,
    LeaveBalanceInput,
    LocalEnterpriseToolProvider,
    MeetingRoom,
    MeetingRoomBookingInput,
    MeetingRoomSearchInput,
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
    payload = ExpenseClaimInput(
        expense_type="交通", amount=Decimal("480.00"), receipt_refs=["INV-001"]
    )

    first = await provider.create_expense_claim(context(), payload)
    repeated = await provider.create_expense_claim(context(), payload)

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
    payload = ExpenseClaimInput(
        expense_type="交通", amount=Decimal("480.00"), receipt_refs=["INV-001"]
    )

    first = await provider.create_expense_claim(context(), payload)
    second = await provider.create_expense_claim(
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


def _meeting_provider() -> LocalEnterpriseToolProvider:
    return LocalEnterpriseToolProvider(
        InMemoryActionRepository(),
        InMemoryPolicyRepository(),
        rooms=(
            MeetingRoom("SH-301", "上海 301", "上海分部", 8),
            MeetingRoom("SH-302", "上海 302", "上海分部", 20),
            MeetingRoom("BJ-101", "北京 101", "北京总部", 6),
        ),
        bookings=(("SH-301", date(2026, 9, 22), time(9, 0), time(12, 0)),),
    )


def _search(**overrides: object) -> MeetingRoomSearchInput:
    payload: dict[str, object] = {
        "location": "上海分部",
        "date": date(2026, 9, 22),
        "start_time": time(10, 0),
        "end_time": time(11, 0),
        "capacity": 1,
    }
    payload.update(overrides)
    return MeetingRoomSearchInput.model_validate(payload)


@pytest.mark.asyncio
async def test_occupied_room_is_filtered_out() -> None:
    result = await _meeting_provider().find_available_rooms(context(), _search())

    assert [item["room_id"] for item in result.data["rooms"]] == ["SH-302"]


@pytest.mark.asyncio
async def test_back_to_back_slot_is_not_a_conflict() -> None:
    """前一场 12:00 结束、这一场 12:00 开始，不该算冲突。"""
    result = await _meeting_provider().find_available_rooms(
        context(), _search(start_time=time(12, 0), end_time=time(13, 0))
    )

    assert {item["room_id"] for item in result.data["rooms"]} == {"SH-301", "SH-302"}


@pytest.mark.asyncio
async def test_capacity_and_location_both_filter() -> None:
    too_small = await _meeting_provider().find_available_rooms(
        context(), _search(capacity=12)
    )
    elsewhere = await _meeting_provider().find_available_rooms(
        context(), _search(location="北京总部")
    )

    assert [item["room_id"] for item in too_small.data["rooms"]] == ["SH-302"]
    assert [item["room_id"] for item in elsewhere.data["rooms"]] == ["BJ-101"]


@pytest.mark.asyncio
async def test_no_room_available_is_a_success_with_an_empty_list() -> None:
    """查不到是确定的业务事实，不是工具故障——否则 Agent 会当成失败去重试。"""
    result = await _meeting_provider().find_available_rooms(
        context(), _search(capacity=100)
    )

    assert result.success is True
    assert result.data["rooms"] == []


@pytest.mark.asyncio
async def test_booking_an_occupied_room_fails() -> None:
    """查询与预订之间可能已被占用，执行前必须再判一次。"""
    result = await _meeting_provider().book_meeting_room(
        context(),
        MeetingRoomBookingInput(
            room_id="SH-301",
            date=date(2026, 9, 22),
            start_time=time(10, 0),
            end_time=time(11, 0),
            subject="项目评审",
        ),
    )

    assert result.success is False
    assert "已被占用" in str(result.error)


@pytest.mark.asyncio
async def test_booking_marks_the_room_busy_for_later_searches() -> None:
    provider = _meeting_provider()
    booking = MeetingRoomBookingInput(
        room_id="SH-302",
        date=date(2026, 9, 22),
        start_time=time(10, 0),
        end_time=time(11, 0),
        subject="项目评审",
    )

    booked = await provider.book_meeting_room(context(), booking)
    after = await provider.find_available_rooms(context(), _search())

    assert booked.success is True
    assert booked.data["room_name"] == "上海 302"
    assert after.data["rooms"] == []


@pytest.mark.asyncio
async def test_booking_an_unknown_room_fails() -> None:
    result = await _meeting_provider().book_meeting_room(
        context(),
        MeetingRoomBookingInput(
            room_id="NOPE-1",
            date=date(2026, 9, 22),
            start_time=time(10, 0),
            end_time=time(11, 0),
            subject="项目评审",
        ),
    )

    assert result.success is False
    assert "不存在" in str(result.error)


@pytest.mark.asyncio
async def test_location_is_matched_by_containment() -> None:
    """差旅产出的目的地是"上海"，会议室登记的是"上海分部"；
    要求完全相等会让依赖白白断掉，用户就得多说一遍地点。"""
    result = await _meeting_provider().find_available_rooms(
        context(), _search(location="上海")
    )

    assert {item["room_id"] for item in result.data["rooms"]} == {"SH-302"}


@pytest.mark.asyncio
async def test_empty_result_distinguishes_full_from_unknown_location() -> None:
    """全满和查无此地点要能分辨，否则 Agent 只会笼统地说"没有"。"""
    provider = _meeting_provider()

    full = await provider.find_available_rooms(context(), _search(capacity=100))
    nowhere = await provider.find_available_rooms(context(), _search(location="火星"))

    assert full.data["rooms"] == [] and full.data["location_exists"] is True
    assert nowhere.data["rooms"] == [] and nowhere.data["location_exists"] is False
    assert "上海分部" in nowhere.data["known_locations"]
