import hashlib
from dataclasses import dataclass
from datetime import date as date_type
from datetime import time, timedelta
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from enterprise_ai_assistant.repositories.actions import ActionRepository
from enterprise_ai_assistant.repositories.policies import PolicyRepository
from enterprise_ai_assistant.tools.contracts import (
    BusinessToolOutcome,
    ExpenseClaimInput,
    ExpenseClaimUpdateInput,
    LeaveBalanceInput,
    LeaveRequestInput,
    LeaveRequestUpdateInput,
    MeetingBookingUpdateInput,
    MeetingRoomBookingInput,
    MeetingRoomSearchInput,
    PolicySearchInput,
    SubmissionQueryInput,
    SubmissionRevokeInput,
    SubmissionUpdateInput,
    ToolContext,
    TravelApplicationInput,
    TravelApplicationUpdateInput,
)


@dataclass(frozen=True)
class MeetingRoom:
    room_id: str
    name: str
    location: str
    capacity: int


#: 演示用的会议室清单。真实部署由远端适配器从会议室系统读取。
DEFAULT_MEETING_ROOMS: tuple[MeetingRoom, ...] = (
    MeetingRoom("SH-301", "上海分部 301 讨论室", "上海分部", 8),
    MeetingRoom("SH-302", "上海分部 302 会议室", "上海分部", 20),
    MeetingRoom("BJ-101", "北京总部 101 洽谈室", "北京总部", 6),
    MeetingRoom("BJ-201", "北京总部 201 大会议室", "北京总部", 30),
    MeetingRoom("HZ-501", "杭州研发中心 501", "杭州研发中心", 12),
)

@dataclass(frozen=True)
class RoomBooking:
    """一条预置占用。

    日期用相对"今天"的天数而不是绝对日期：写死的日期一旦过去，任何时段都查得到
    空闲，"没有可用会议室"这条路径就再也演示不出来了。
    """

    room_id: str
    day_offset: int
    start: time
    end: time


#: 预置占用。刻意让下周三上午的 SH-301、下周四上午的 SH-302 各自被占，于是常规
#: 演示能查到房间又能看出筛选生效；下周三下午两点则被排满，用来演示查不到的情形。
DEFAULT_ROOM_BOOKINGS: tuple[RoomBooking, ...] = (
    RoomBooking("SH-301", 4, time(9, 0), time(12, 0)),
    RoomBooking("SH-302", 5, time(9, 30), time(11, 0)),
    RoomBooking("BJ-201", 5, time(14, 0), time(17, 0)),
    RoomBooking("SH-301", 4, time(14, 0), time(16, 0)),
    RoomBooking("SH-302", 4, time(14, 0), time(16, 0)),
)


ModelT = TypeVar("ModelT", bound=BaseModel)

#: 演示用的审批状态。会议室预订即时生效，没有审批环节。
_MOCK_APPROVAL_STATUSES = ("pending_approval", "approved")


def mock_submission_status(
    action_type: str, reference_id: str, *, revised: bool = False, revoked: bool = False
) -> str:
    """给单据一个确定的演示状态。

    本地没有 OA 和 HR 系统可问，这里是替身，真实部署由远端适配器回查。按单号哈希取值
    而不是随机：同一张单据查两次得到两个状态，用户会以为系统在乱说。
    改过的单据一律回到审批中——已经批过的申请改了日期，原来那次审批就不再作数。
    """
    if revoked:
        return "revoked"
    if action_type == "meeting_booking":
        return "confirmed"
    if revised:
        return "pending_approval"
    digest = int(hashlib.sha256(reference_id.encode()).hexdigest(), 16)
    return _MOCK_APPROVAL_STATUSES[digest % len(_MOCK_APPROVAL_STATUSES)]


class LocalEnterpriseToolProvider:
    """当前企业工具实现；后续可在不改变 Agent 的情况下替换为远端适配器。"""

    def __init__(
        self,
        actions: ActionRepository,
        policies: PolicyRepository,
        *,
        annual_leave_balance: float = 8,
        rooms: tuple[MeetingRoom, ...] = DEFAULT_MEETING_ROOMS,
        bookings: tuple[RoomBooking, ...] = DEFAULT_ROOM_BOOKINGS,
    ) -> None:
        self._actions = actions
        self._policies = policies
        self._annual_leave_balance = annual_leave_balance
        self._rooms = rooms
        self._preset = bookings
        # 本次进程里新订的房间用绝对日期记录，后续查询能看到它们已被占用。按单号索引，
        # 改期时才能先把自己原来占的时段让出来，否则挪一个小时也会和自己冲突。
        # 真实部署由会议室系统持有这份状态，这里只是替身。
        self._booked: dict[str, tuple[str, date_type, time, time]] = {}

    async def search_policy(self, payload: PolicySearchInput) -> BusinessToolOutcome:
        items = await self._policies.search(payload.query, payload.domain, payload.limit)
        return BusinessToolOutcome(
            tool="search_policy",
            success=True,
            status="completed",
            data={"items": items},
        )

    async def _record_write(
        self,
        *,
        tool: str,
        action_type: str,
        context: ToolContext,
        payload: dict[str, Any],
    ) -> BusinessToolOutcome:
        recorded = await self._actions.execute_once(
            idempotency_key=context.idempotency_key(tool),
            action_type=action_type,
            user_id=context.user_id,
            payload=payload,
        )
        return BusinessToolOutcome(
            tool=tool,
            success=True,
            status="submitted",
            reference_id=str(recorded["reference_id"]),
            data=recorded,
        )

    async def create_travel_application(
        self, context: ToolContext, payload: TravelApplicationInput
    ) -> BusinessToolOutcome:
        return await self._record_write(
            tool="create_travel_application",
            action_type="travel_application",
            context=context,
            payload=payload.model_dump(mode="json"),
        )

    async def create_expense_claim(
        self, context: ToolContext, payload: ExpenseClaimInput
    ) -> BusinessToolOutcome:
        return await self._record_write(
            tool="create_expense_claim",
            action_type="expense_claim",
            context=context,
            payload=payload.model_dump(mode="json"),
        )

    def _occupied(self) -> list[tuple[str, date_type, time, time]]:
        """把预置占用按今天展开，再并上本次进程内的预订。

        每次查询都重新展开，进程跨过午夜后预置时段仍然跟着日期走。
        """
        today = date_type.today()
        preset = [
            (item.room_id, today + timedelta(days=item.day_offset), item.start, item.end)
            for item in self._preset
        ]
        return preset + list(self._booked.values())

    def _is_free(
        self, room_id: str, day: date_type, start: time, end: time, *, ignore: str | None = None
    ) -> bool:
        occupied = self._occupied()
        if ignore is not None and ignore in self._booked:
            occupied.remove(self._booked[ignore])
        # 半开区间比较：紧挨着的两场会议（10:00 结束、10:00 开始）不算冲突。
        return not any(
            booked_room == room_id and booked_day == day and start < booked_end and booked_start < end
            for booked_room, booked_day, booked_start, booked_end in occupied
        )

    @staticmethod
    def _location_matches(room_location: str, requested: str) -> bool:
        """按包含关系匹配地点。

        地点常常是从上游任务推断来的：差旅任务产出的目的地是"上海"，而会议室登记的
        是"上海分部"。要求完全相等会让这条依赖白白断掉，用户则要多说一遍地点。
        """
        return requested in room_location or room_location in requested

    async def find_available_rooms(
        self, context: ToolContext, payload: MeetingRoomSearchInput
    ) -> BusinessToolOutcome:
        del context
        at_location = [
            room for room in self._rooms
            if self._location_matches(room.location, payload.location)
        ]
        rooms = [
            {
                "room_id": room.room_id,
                "name": room.name,
                "location": room.location,
                "capacity": room.capacity,
            }
            for room in at_location
            if room.capacity >= payload.capacity
            and self._is_free(room.room_id, payload.date, payload.start_time, payload.end_time)
        ]
        # 查不到不是失败：这是一个确定的业务事实，Agent 应当如实转述并建议改期，
        # 而不是把它当成工具故障重试，更不能编一个房间出来。空结果时附上全部地点，
        # 否则 Agent 无从判断是这个时段满了，还是压根没有这个地点。
        data: dict[str, Any] = {"rooms": rooms, "location": payload.location}
        if not rooms:
            data["location_exists"] = bool(at_location)
            data["known_locations"] = sorted({room.location for room in self._rooms})
        return BusinessToolOutcome(
            tool="find_available_rooms",
            success=True,
            status="completed",
            data=data,
        )

    async def book_meeting_room(
        self, context: ToolContext, payload: MeetingRoomBookingInput
    ) -> BusinessToolOutcome:
        room = next((item for item in self._rooms if item.room_id == payload.room_id), None)
        if room is None:
            return BusinessToolOutcome(
                tool="book_meeting_room",
                success=False,
                status="failed",
                error=f"会议室 {payload.room_id} 不存在",
            )
        if not self._is_free(room.room_id, payload.date, payload.start_time, payload.end_time):
            # 查询与预订之间可能已被他人占用，执行前必须再判一次。
            return BusinessToolOutcome(
                tool="book_meeting_room",
                success=False,
                status="failed",
                error=f"{room.name} 在该时段已被占用",
            )
        outcome = await self._record_write(
            tool="book_meeting_room",
            action_type="meeting_booking",
            context=context,
            payload={**payload.model_dump(mode="json"), "room_name": room.name},
        )
        self._booked[str(outcome.reference_id)] = (
            room.room_id, payload.date, payload.start_time, payload.end_time
        )
        return outcome

    async def get_leave_balance(
        self, context: ToolContext, payload: LeaveBalanceInput
    ) -> BusinessToolOutcome:
        del context
        balance = self._annual_leave_balance if payload.leave_type == "annual" else 0
        return BusinessToolOutcome(
            tool="get_leave_balance",
            success=True,
            status="completed",
            data={"leave_type": payload.leave_type, "balance_days": balance},
        )

    async def submit_leave_request(
        self, context: ToolContext, payload: LeaveRequestInput
    ) -> BusinessToolOutcome:
        return await self._record_write(
            tool="submit_leave_request",
            action_type="leave_request",
            context=context,
            payload=payload.model_dump(mode="json"),
        )

    async def query_submissions(
        self, context: ToolContext, action_type: str, payload: SubmissionQueryInput
    ) -> BusinessToolOutcome:
        # 只按运行时注入的 user_id 查。单号是用户可以随口说出的，不能靠它鉴权，
        # 否则报一个别人的单号就能看到别人的请假日期。
        submitted = await self._actions.list_submitted(
            user_id=context.user_id,
            action_type=action_type,
            reference_id=payload.reference_id,
            limit=payload.limit,
        )
        items = [
            {
                "reference_id": item.reference_id,
                "submitted_at": item.created_at.isoformat(),
                "revised_at": item.revised_at.isoformat() if item.revised_at else None,
                "revoked_at": item.revoked_at.isoformat() if item.revoked_at else None,
                "status": mock_submission_status(
                    action_type,
                    item.reference_id,
                    revised=item.revised_at is not None,
                    revoked=item.revoked_at is not None,
                ),
                # 这里给全量字段，不走"我的单据"的白名单。白名单防的是每轮被动注入：
                # 用户没问，请假原因也会跟着档案进每一轮的上下文。查询是本人对自己单据的
                # 主动请求，用户要改原因、核对票据号时，Agent 手里必须有原值。
                "fields": item.payload,
            }
            for item in submitted
        ]
        # 查不到同样是确定的业务事实，不是故障。单号给了却没命中时要说清楚，
        # 否则 Agent 会把"这张不是你的或单号错了"说成"你还没有提交过"。
        data: dict[str, Any] = {"items": items}
        if payload.reference_id is not None:
            data["reference_id"] = payload.reference_id
            data["found"] = bool(items)
        return BusinessToolOutcome(
            tool="query_submissions", success=True, status="completed", data=data
        )

    async def revoke_submission(
        self, context: ToolContext, action_type: str, payload: SubmissionRevokeInput
    ) -> BusinessToolOutcome:
        tool = f"revoke_{action_type}"
        found = await self._actions.list_submitted(
            user_id=context.user_id,
            action_type=action_type,
            reference_id=payload.reference_id,
            limit=1,
        )
        if not found:
            return self._failed(tool, f"单据 {payload.reference_id} 不存在或不属于当前用户")
        if found[0].revoked_at is not None:
            # 如实告诉用户它早就撤销了，而不是再报一次"撤销成功"。
            return self._failed(tool, f"单据 {payload.reference_id} 此前已经撤销")
        result = await self._actions.revoke_submitted(
            idempotency_key=context.idempotency_key(tool),
            user_id=context.user_id,
            action_type=action_type,
            reference_id=payload.reference_id,
        )
        if result is None:
            return self._failed(tool, f"单据 {payload.reference_id} 不存在或不属于当前用户")
        # 会议室撤销后时段要让出来，否则撤掉的预订还挡着别人（和自己）再订。
        self._booked.pop(payload.reference_id, None)
        return BusinessToolOutcome(
            tool=tool,
            success=True,
            status="revoked",
            reference_id=payload.reference_id,
            data={**result, "status": "revoked"},
        )

    @staticmethod
    def _failed(tool: str, error: str) -> BusinessToolOutcome:
        return BusinessToolOutcome(tool=tool, success=False, status="failed", error=error)

    async def _merged(
        self,
        context: ToolContext,
        *,
        tool: str,
        action_type: str,
        payload: SubmissionUpdateInput,
        schema: type[ModelT],
    ) -> tuple[dict[str, Any], ModelT] | BusinessToolOutcome:
        """取出原单据，合并改动，再按新建时的同一份契约重新校验。"""
        found = await self._actions.list_submitted(
            user_id=context.user_id,
            action_type=action_type,
            reference_id=payload.reference_id,
            limit=1,
        )
        if not found:
            return self._failed(tool, f"单据 {payload.reference_id} 不存在或不属于当前用户")
        if found[0].revoked_at is not None:
            # 撤销是终态。允许改一张已撤销的单，状态又会回到审批中，等于绕过撤销复活了它。
            return self._failed(tool, f"单据 {payload.reference_id} 已撤销，不能再修改")
        changes = payload.changes()
        merged = {**found[0].payload, **changes}
        # 改成单程时结束日期必须清掉。补丁里的 None 表示"不改"，没法表达"删掉"，
        # 不在这里处理的话，往返改单程永远过不了校验。
        if merged.get("trip_type") == "one_way" and "end_date" not in changes:
            merged.pop("end_date", None)
        try:
            validated = schema.model_validate(
                {name: value for name, value in merged.items() if name in schema.model_fields}
            )
        except ValidationError as exc:
            return self._failed(tool, f"修改后的单据不合法：{exc.errors()[0]['msg']}")
        return {**merged, **validated.model_dump(mode="json")}, validated

    async def _apply_update(
        self,
        context: ToolContext,
        *,
        tool: str,
        action_type: str,
        reference_id: str,
        new_payload: dict[str, Any],
    ) -> BusinessToolOutcome:
        result = await self._actions.update_submitted(
            idempotency_key=context.idempotency_key(tool),
            user_id=context.user_id,
            action_type=action_type,
            reference_id=reference_id,
            payload=new_payload,
        )
        if result is None:
            return self._failed(tool, f"单据 {reference_id} 不存在或不属于当前用户")
        return BusinessToolOutcome(
            tool=tool,
            success=True,
            status="updated",
            reference_id=reference_id,
            data={
                **result,
                "status": mock_submission_status(action_type, reference_id, revised=True),
            },
        )

    async def _update(
        self,
        context: ToolContext,
        *,
        tool: str,
        action_type: str,
        payload: SubmissionUpdateInput,
        schema: type[BaseModel],
    ) -> BusinessToolOutcome:
        merged = await self._merged(
            context, tool=tool, action_type=action_type, payload=payload, schema=schema
        )
        if isinstance(merged, BusinessToolOutcome):
            return merged
        return await self._apply_update(
            context,
            tool=tool,
            action_type=action_type,
            reference_id=payload.reference_id,
            new_payload=merged[0],
        )

    async def update_travel_application(
        self, context: ToolContext, payload: TravelApplicationUpdateInput
    ) -> BusinessToolOutcome:
        return await self._update(
            context,
            tool="update_travel_application",
            action_type="travel_application",
            payload=payload,
            schema=TravelApplicationInput,
        )

    async def update_expense_claim(
        self, context: ToolContext, payload: ExpenseClaimUpdateInput
    ) -> BusinessToolOutcome:
        return await self._update(
            context,
            tool="update_expense_claim",
            action_type="expense_claim",
            payload=payload,
            schema=ExpenseClaimInput,
        )

    async def update_leave_request(
        self, context: ToolContext, payload: LeaveRequestUpdateInput
    ) -> BusinessToolOutcome:
        return await self._update(
            context,
            tool="update_leave_request",
            action_type="leave_request",
            payload=payload,
            schema=LeaveRequestInput,
        )

    async def update_meeting_booking(
        self, context: ToolContext, payload: MeetingBookingUpdateInput
    ) -> BusinessToolOutcome:
        tool = "update_meeting_booking"
        merged = await self._merged(
            context,
            tool=tool,
            action_type="meeting_booking",
            payload=payload,
            schema=MeetingRoomBookingInput,
        )
        if isinstance(merged, BusinessToolOutcome):
            return merged
        new_payload, booking = merged
        room = next((item for item in self._rooms if item.room_id == booking.room_id), None)
        if room is None:
            return self._failed(tool, f"会议室 {booking.room_id} 不存在")
        # 改期或换房和新订一样要判空闲，但要先把这张单自己原来的时段让出来。
        if not self._is_free(
            room.room_id,
            booking.date,
            booking.start_time,
            booking.end_time,
            ignore=payload.reference_id,
        ):
            return self._failed(tool, f"{room.name} 在该时段已被占用")
        outcome = await self._apply_update(
            context,
            tool=tool,
            action_type="meeting_booking",
            reference_id=payload.reference_id,
            new_payload={**new_payload, "room_name": room.name},
        )
        if outcome.success:
            self._booked[payload.reference_id] = (
                room.room_id, booking.date, booking.start_time, booking.end_time
            )
        return outcome
