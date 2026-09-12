from dataclasses import dataclass
from datetime import date as date_type
from datetime import time, timedelta
from typing import Any

from enterprise_ai_assistant.repositories.actions import ActionRepository
from enterprise_ai_assistant.repositories.policies import PolicyRepository
from enterprise_ai_assistant.tools.contracts import (
    BusinessToolOutcome,
    ExpenseClaimInput,
    LeaveBalanceInput,
    LeaveRequestInput,
    MeetingRoomBookingInput,
    MeetingRoomSearchInput,
    PolicySearchInput,
    ToolContext,
    TravelApplicationInput,
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
        # 本次进程里新订的房间用绝对日期记录，后续查询能看到它们已被占用。
        # 真实部署由会议室系统持有这份状态，这里只是替身。
        self._booked: list[tuple[str, date_type, time, time]] = []

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
        return preset + self._booked

    def _is_free(
        self, room_id: str, day: date_type, start: time, end: time
    ) -> bool:
        # 半开区间比较：紧挨着的两场会议（10:00 结束、10:00 开始）不算冲突。
        return not any(
            booked_room == room_id and booked_day == day and start < booked_end and booked_start < end
            for booked_room, booked_day, booked_start, booked_end in self._occupied()
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
        self._booked.append(
            (room.room_id, payload.date, payload.start_time, payload.end_time)
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
