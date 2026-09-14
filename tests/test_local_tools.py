from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from enterprise_ai_assistant.core.models import AgentName, RecentAction
from enterprise_ai_assistant.repositories.actions import InMemoryActionRepository
from enterprise_ai_assistant.repositories.policies import InMemoryPolicyRepository
from enterprise_ai_assistant.tools import (
    ExpenseClaimInput,
    LeaveBalanceInput,
    LeaveRequestInput,
    LeaveRequestUpdateInput,
    LocalEnterpriseToolProvider,
    MeetingBookingUpdateInput,
    MeetingRoom,
    MeetingRoomBookingInput,
    MeetingRoomSearchInput,
    SubmissionQueryInput,
    SubmissionRevokeInput,
    ToolContext,
    TravelApplicationInput,
    TravelApplicationUpdateInput,
)
from enterprise_ai_assistant.tools.local_enterprise import RoomBooking, mock_submission_status
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


def test_capability_summary_does_not_promise_approvals() -> None:
    """系统没有代审批的工具，能力自述里不能出现这类说法。"""
    text = "".join(CAPABILITY_SUMMARY.values())

    for forbidden in ("审批申请", "代审批", "代替审批"):
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
        bookings=(RoomBooking("SH-301", 0, time(9, 0), time(12, 0)),),
    )


def _search(**overrides: object) -> MeetingRoomSearchInput:
    payload: dict[str, object] = {
        "location": "上海分部",
        "date": date.today(),
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
            date=date.today(),
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
        date=date.today(),
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
            date=date.today(),
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


@pytest.mark.asyncio
async def test_reference_id_is_short_and_hides_the_idempotency_key() -> None:
    """幂等键是 会话:请求:任务:工具 拼出来的，直接当单号会把内部结构给到用户。"""
    actions = InMemoryActionRepository()
    provider = LocalEnterpriseToolProvider(actions, InMemoryPolicyRepository())

    result = await provider.create_travel_application(
        context(),
        TravelApplicationInput(
            origin="杭州",
            destination="上海",
            start_date=date(2026, 9, 16),
            end_date=date(2026, 9, 17),
            purpose="客户拜访",
        ),
    )

    reference = str(result.reference_id)
    assert reference.startswith("TRV-")
    assert ":" not in reference
    assert "task-1" not in reference
    assert len(reference) <= 24


@pytest.mark.asyncio
async def test_replaying_a_write_returns_the_same_reference_id() -> None:
    actions = InMemoryActionRepository()
    provider = LocalEnterpriseToolProvider(actions, InMemoryPolicyRepository())
    payload = ExpenseClaimInput(
        expense_type="交通", amount=Decimal("480.00"), receipt_refs=["INV-001"]
    )

    first = await provider.create_expense_claim(context(), payload)
    replay = await provider.create_expense_claim(context(), payload)

    assert first.reference_id == replay.reference_id
    assert str(first.reference_id).startswith("EXP-")


def _leave(reason: str = "家里有事") -> LeaveRequestInput:
    return LeaveRequestInput(
        leave_type="annual",
        start_date=date(2026, 9, 21),
        end_date=date(2026, 9, 22),
        reason=reason,
    )


@pytest.mark.asyncio
async def test_submission_query_returns_the_users_documents_with_a_stable_status() -> None:
    provider = LocalEnterpriseToolProvider(InMemoryActionRepository(), InMemoryPolicyRepository())
    submitted = await provider.submit_leave_request(context(), _leave())
    reference = str(submitted.reference_id)

    first = await provider.query_submissions(
        context(), "leave_request", SubmissionQueryInput(reference_id=reference)
    )
    again = await provider.query_submissions(
        context(), "leave_request", SubmissionQueryInput(reference_id=reference)
    )

    [item] = first.data["items"]
    assert first.data["found"] is True
    assert item["reference_id"] == reference
    assert item["status"] in {"pending_approval", "approved"}
    # 同一张单据两次查询状态不能变，否则用户会以为系统在乱说。
    assert again.data["items"][0]["status"] == item["status"]


@pytest.mark.asyncio
async def test_submission_query_returns_every_field_but_not_the_idempotency_key() -> None:
    """本人主动查询要拿到原值才能核对和修改，但幂等键的内部结构仍不外露。"""
    provider = LocalEnterpriseToolProvider(InMemoryActionRepository(), InMemoryPolicyRepository())
    await provider.submit_leave_request(context(), _leave(reason="看病"))

    outcome = await provider.query_submissions(context(), "leave_request", SubmissionQueryInput())

    [item] = outcome.data["items"]
    assert item["fields"]["reason"] == "看病"
    assert "task-1" not in str(outcome.model_dump())


@pytest.mark.asyncio
async def test_blank_reference_id_lists_recent_documents_instead_of_matching_nothing() -> None:
    """实测模型用空字符串表示不指定单号；当成单号去查，有单据的用户会被告知一张都没有。"""
    provider = LocalEnterpriseToolProvider(InMemoryActionRepository(), InMemoryPolicyRepository())
    submitted = await provider.submit_leave_request(context(), _leave())

    outcome = await provider.query_submissions(
        context(), "leave_request", SubmissionQueryInput(reference_id="  ")
    )

    assert [item["reference_id"] for item in outcome.data["items"]] == [submitted.reference_id]
    # 没指定单号就不该出现 found=false 这种"单号没命中"的信号。
    assert "found" not in outcome.data


@pytest.mark.asyncio
async def test_submission_query_does_not_reveal_other_users_documents() -> None:
    """单号是用户随口能报出来的，不能靠它鉴权。"""
    provider = LocalEnterpriseToolProvider(InMemoryActionRepository(), InMemoryPolicyRepository())
    submitted = await provider.submit_leave_request(context(), _leave())
    stranger = context().model_copy(update={"user_id": "u-2"})

    outcome = await provider.query_submissions(
        stranger,
        "leave_request",
        SubmissionQueryInput(reference_id=str(submitted.reference_id)),
    )

    assert outcome.data == {
        "items": [],
        "reference_id": submitted.reference_id,
        "found": False,
    }


def test_meeting_bookings_have_no_approval_step() -> None:
    assert mock_submission_status("meeting_booking", "MTG-20260913-BD48AD") == "confirmed"


def test_every_business_domain_can_query_its_own_submissions() -> None:
    registry = DomainToolRegistry(
        LocalEnterpriseToolProvider(InMemoryActionRepository(), InMemoryPolicyRepository())
    )
    expected = {
        AgentName.TRAVEL: "query_travel_applications",
        AgentName.EXPENSE: "query_expense_claims",
        AgentName.HR: "query_leave_requests",
        AgentName.MEETING: "query_meeting_bookings",
    }

    for agent, tool_name in expected.items():
        tools = {item.tool.name: item for item in registry.for_agent(agent, context())}
        assert tools[tool_name].risk.value == "read"


def _round_trip() -> TravelApplicationInput:
    return TravelApplicationInput(
        origin="杭州",
        destination="上海",
        start_date=date(2026, 9, 16),
        end_date=date(2026, 9, 17),
        purpose="客户拜访",
    )


@pytest.mark.asyncio
async def test_update_changes_only_the_given_fields_and_resets_approval() -> None:
    provider = LocalEnterpriseToolProvider(InMemoryActionRepository(), InMemoryPolicyRepository())
    created = await provider.create_travel_application(context(), _round_trip())
    reference = str(created.reference_id)

    updated = await provider.update_travel_application(
        context(uuid4()),
        TravelApplicationUpdateInput(reference_id=reference, end_date=date(2026, 9, 18)),
    )
    [item] = (
        await provider.query_submissions(
            context(), "travel_application", SubmissionQueryInput(reference_id=reference)
        )
    ).data["items"]

    assert updated.success is True
    assert item["fields"]["end_date"] == "2026-09-18"
    assert item["fields"]["purpose"] == "客户拜访"
    assert item["revised_at"] is not None
    # 批过的申请改了日期，原来那次审批不再作数。
    assert item["status"] == "pending_approval"


@pytest.mark.asyncio
async def test_switching_to_one_way_drops_the_end_date() -> None:
    """补丁里的 None 表示不改，没法表达删掉；不自动清掉的话往返改单程永远过不了校验。"""
    provider = LocalEnterpriseToolProvider(InMemoryActionRepository(), InMemoryPolicyRepository())
    created = await provider.create_travel_application(context(), _round_trip())

    updated = await provider.update_travel_application(
        context(uuid4()),
        TravelApplicationUpdateInput(reference_id=str(created.reference_id), trip_type="one_way"),
    )

    assert updated.success is True
    assert updated.data["trip_type"] == "one_way"
    assert updated.data["end_date"] is None


@pytest.mark.asyncio
async def test_update_is_revalidated_against_the_create_contract() -> None:
    provider = LocalEnterpriseToolProvider(InMemoryActionRepository(), InMemoryPolicyRepository())
    created = await provider.submit_leave_request(context(), _leave())

    updated = await provider.update_leave_request(
        context(uuid4()),
        LeaveRequestUpdateInput(reference_id=str(created.reference_id), end_date=date(2026, 9, 1)),
    )

    assert updated.success is False
    assert "不合法" in str(updated.error)


@pytest.mark.asyncio
async def test_update_cannot_touch_other_users_documents() -> None:
    provider = LocalEnterpriseToolProvider(InMemoryActionRepository(), InMemoryPolicyRepository())
    created = await provider.submit_leave_request(context(), _leave())
    stranger = context(uuid4()).model_copy(update={"user_id": "u-2"})

    updated = await provider.update_leave_request(
        stranger, LeaveRequestUpdateInput(reference_id=str(created.reference_id), reason="改掉")
    )

    assert updated.success is False
    [item] = (
        await provider.query_submissions(context(), "leave_request", SubmissionQueryInput())
    ).data["items"]
    assert item["fields"]["reason"] == "家里有事"


@pytest.mark.asyncio
async def test_replaying_an_update_applies_it_once() -> None:
    actions = InMemoryActionRepository()
    provider = LocalEnterpriseToolProvider(actions, InMemoryPolicyRepository())
    created = await provider.submit_leave_request(context(), _leave())
    change = LeaveRequestUpdateInput(reference_id=str(created.reference_id), reason="改期")
    same_turn = context(uuid4())

    first = await provider.update_leave_request(same_turn, change)
    replay = await provider.update_leave_request(same_turn, change)

    assert first.data == replay.data
    assert len(actions.updates) == 1


@pytest.mark.asyncio
async def test_moving_a_booking_does_not_conflict_with_itself() -> None:
    provider = _meeting_provider()
    tomorrow = date.today() + timedelta(days=1)
    booked = await provider.book_meeting_room(
        context(),
        MeetingRoomBookingInput(
            room_id="SH-302",
            date=tomorrow,
            start_time=time(10, 0),
            end_time=time(11, 0),
            subject="周会",
        ),
    )

    # 往后挪半小时，和自己原来的时段重叠。
    moved = await provider.update_meeting_booking(
        context(uuid4()),
        MeetingBookingUpdateInput(
            reference_id=str(booked.reference_id), start_time=time(10, 30), end_time=time(11, 30)
        ),
    )
    taken = await provider.update_meeting_booking(
        context(uuid4()),
        MeetingBookingUpdateInput(
            reference_id=str(booked.reference_id),
            room_id="SH-301",
            date=date.today(),
            start_time=time(9, 0),
            end_time=time(10, 0),
        ),
    )

    assert moved.success is True
    assert moved.data["start_time"] == "10:30:00"
    assert taken.success is False
    assert "已被占用" in str(taken.error)


@pytest.mark.asyncio
async def test_revoked_document_stays_visible_and_cannot_be_changed() -> None:
    provider = LocalEnterpriseToolProvider(InMemoryActionRepository(), InMemoryPolicyRepository())
    created = await provider.submit_leave_request(context(), _leave())
    reference = str(created.reference_id)

    revoked = await provider.revoke_submission(
        context(uuid4()), "leave_request", SubmissionRevokeInput(reference_id=reference)
    )
    again = await provider.revoke_submission(
        context(uuid4()), "leave_request", SubmissionRevokeInput(reference_id=reference)
    )
    changed = await provider.update_leave_request(
        context(uuid4()), LeaveRequestUpdateInput(reference_id=reference, reason="复活")
    )
    [item] = (
        await provider.query_submissions(context(), "leave_request", SubmissionQueryInput())
    ).data["items"]

    assert revoked.success is True
    # 第二次撤销不能再报一次成功。
    assert again.success is False and "已经撤销" in str(again.error)
    assert changed.success is False and "已撤销" in str(changed.error)
    assert item["status"] == "revoked"


@pytest.mark.asyncio
async def test_revoke_cannot_touch_other_users_documents() -> None:
    provider = LocalEnterpriseToolProvider(InMemoryActionRepository(), InMemoryPolicyRepository())
    created = await provider.submit_leave_request(context(), _leave())
    stranger = context(uuid4()).model_copy(update={"user_id": "u-2"})

    outcome = await provider.revoke_submission(
        stranger, "leave_request", SubmissionRevokeInput(reference_id=str(created.reference_id))
    )

    assert outcome.success is False
    [item] = (
        await provider.query_submissions(context(), "leave_request", SubmissionQueryInput())
    ).data["items"]
    assert item["status"] != "revoked"


@pytest.mark.asyncio
async def test_cancelling_a_booking_frees_the_slot() -> None:
    provider = _meeting_provider()
    tomorrow = date.today() + timedelta(days=1)
    booking = MeetingRoomBookingInput(
        room_id="SH-302", date=tomorrow, start_time=time(10, 0), end_time=time(11, 0), subject="周会"
    )
    booked = await provider.book_meeting_room(context(), booking)

    await provider.revoke_submission(
        context(uuid4()),
        "meeting_booking",
        SubmissionRevokeInput(reference_id=str(booked.reference_id)),
    )
    rebooked = await provider.book_meeting_room(context(uuid4()), booking)

    assert rebooked.success is True


def test_revoked_action_is_marked_in_the_model_context() -> None:
    action = RecentAction(
        reference_id="LVE-1",
        action_type="leave_request",
        summary="leave_type=annual",
        created_at=datetime(2026, 9, 13, tzinfo=UTC),
        revoked_at=datetime(2026, 9, 13, 1, tzinfo=UTC),
    )

    assert action.render() == "[hr] leave_request LVE-1（2026-09-13）（已撤销）：leave_type=annual"
