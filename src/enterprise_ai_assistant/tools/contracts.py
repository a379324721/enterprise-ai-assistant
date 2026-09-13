# 会议室入参有个字段就叫 date，会遮住同名类型，这些类里的注解统一走模块路径。
import datetime as dt
from datetime import date, time
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from enterprise_ai_assistant.core.models import DraftField


class ToolRisk(StrEnum):
    """风险由服务端工具注册表声明，不能由模型自行决定。"""

    READ = "read"
    WRITE = "write"


class StrictToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolContext(BaseModel):
    """可信运行时上下文，不作为模型可填写的工具参数暴露。"""

    user_id: str = Field(min_length=1, max_length=128)
    conversation_id: UUID
    request_id: UUID
    task_id: str = Field(min_length=1, max_length=128)

    def idempotency_key(self, tool_name: str) -> str:
        return f"{self.conversation_id}:{self.request_id}:{self.task_id}:{tool_name}"


class PolicySearchInput(StrictToolInput):
    query: str = Field(min_length=1, max_length=2000)
    domain: str = Field(min_length=1, max_length=32)
    limit: int = Field(default=3, ge=1, le=10)


class PolicyQueryInput(StrictToolInput):
    """暴露给领域模型的查询参数；领域由工具注册表固定。"""

    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=3, ge=1, le=10)


class InformationRequestInput(StrictToolInput):
    # 会原样显示在界面的事项卡上，所以要求面向用户的中文字段名。
    missing_fields: list[str] = Field(
        min_length=1, description="缺失字段的中文名称，例如“结束日期”“出差事由”"
    )
    question: str = Field(min_length=1, max_length=1000)
    # 已经谈定或有建议值的字段。任务停在待补充时存为草稿，下一轮续跑时交还给领域 Agent。
    known_fields: list[DraftField] = Field(default_factory=list, max_length=20)


#: 行程类型的取值标签，确认卡片按它翻译。
_TRIP_TYPE_LABELS: dict[str, JsonValue] = {"round_trip": "往返", "one_way": "单程"}


class TravelApplicationInput(StrictToolInput):
    # 写操作入参的 title 是确认卡片上的字段名，测试会检查每个字段都声明了。
    # 出发地决定交通方式和差旅标准，没有它的申请单审批人无从判断；
    # "我的单据"里只写目的地时，也分不清是去上海还是从上海出发。
    origin: str = Field(title="出发地", min_length=1, max_length=200)
    destination: str = Field(title="目的地", min_length=1, max_length=200)
    start_date: date = Field(title="开始日期")
    # 单程（调动、外派、返程另行申请）没有结束日期。行程类型单独成字段而不是只把
    # end_date 放开：否则模型漏问返程日期时也能直接提交，看起来和单程一模一样。
    trip_type: Literal["round_trip", "one_way"] = Field(
        default="round_trip",
        title="行程类型",
        description="round_trip 往返，必须给出 end_date；one_way 单程，不填 end_date",
        json_schema_extra={"value_labels": _TRIP_TYPE_LABELS},
    )
    end_date: date | None = Field(default=None, title="结束日期")
    purpose: str = Field(title="出差事由", min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_dates(self) -> "TravelApplicationInput":
        if self.trip_type == "one_way":
            if self.end_date is not None:
                raise ValueError("end_date must be omitted for a one_way trip")
            return self
        if self.end_date is None:
            raise ValueError("end_date is required for a round_trip")
        if self.end_date < self.start_date:
            raise ValueError("end_date must not be earlier than start_date")
        return self


class ExpenseClaimInput(StrictToolInput):
    expense_type: str = Field(title="费用类型", min_length=1, max_length=100)
    amount: Decimal = Field(title="金额", gt=0)
    currency: str = Field(default="CNY", title="币种", pattern=r"^[A-Z]{3}$")
    receipt_refs: list[str] = Field(title="票据号", min_length=1)
    travel_reference: str | None = Field(default=None, title="关联差旅单号", max_length=256)


class LeaveBalanceInput(StrictToolInput):
    leave_type: str = Field(default="annual", min_length=1, max_length=100)


class LeaveRequestInput(StrictToolInput):
    leave_type: str = Field(title="假期类型", min_length=1, max_length=100)
    start_date: date = Field(title="开始日期")
    end_date: date = Field(title="结束日期")
    reason: str | None = Field(default=None, title="请假原因", max_length=1000)

    @model_validator(mode="after")
    def validate_dates(self) -> "LeaveRequestInput":
        if self.end_date < self.start_date:
            raise ValueError("end_date must not be earlier than start_date")
        return self


class MeetingRoomSearchInput(StrictToolInput):
    location: str = Field(min_length=1, max_length=100)
    date: date
    start_time: time
    end_time: time
    capacity: int = Field(default=1, ge=1, le=500)

    @model_validator(mode="after")
    def validate_window(self) -> "MeetingRoomSearchInput":
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be later than start_time")
        return self


class MeetingRoomBookingInput(StrictToolInput):
    room_id: str = Field(title="会议室", min_length=1, max_length=64)
    date: dt.date = Field(title="日期")
    start_time: time = Field(title="开始时间")
    end_time: time = Field(title="结束时间")
    subject: str = Field(title="会议主题", min_length=1, max_length=200)

    @model_validator(mode="after")
    def validate_window(self) -> "MeetingRoomBookingInput":
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be later than start_time")
        return self


class SubmissionQueryInput(StrictToolInput):
    """查询当前用户提交过的单据。单据类型由工具注册表固定，用户身份由运行时注入。"""

    reference_id: str | None = Field(
        default=None,
        max_length=64,
        description="单号，例如 MTG-20260913-BD48AD；用户没有指明具体哪张时留空，返回最近几张",
    )
    limit: int = Field(default=5, ge=1, le=20)


class SubmissionRevokeInput(StrictToolInput):
    """撤销一张已提交的单据。单据类型由工具注册表固定。"""

    reference_id: str = Field(title="单号", min_length=1, max_length=64, description="要撤销的单号")


class SubmissionUpdateInput(StrictToolInput):
    """修改已提交单据的公共部分：只传要改的字段，没传的沿用原值。"""

    reference_id: str = Field(title="单号", min_length=1, max_length=64, description="要修改的单号")

    def changes(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"reference_id"}, exclude_none=True)

    @model_validator(mode="after")
    def validate_has_changes(self) -> "SubmissionUpdateInput":
        # 空修改在确认卡上看起来和真修改一样，用户点了确认却什么也没发生。
        if not self.changes():
            raise ValueError("at least one field must be changed")
        return self


class TravelApplicationUpdateInput(SubmissionUpdateInput):
    origin: str | None = Field(default=None, title="出发地", min_length=1, max_length=200)
    destination: str | None = Field(default=None, title="目的地", min_length=1, max_length=200)
    start_date: date | None = Field(default=None, title="开始日期")
    trip_type: Literal["round_trip", "one_way"] | None = Field(
        default=None,
        title="行程类型",
        description="改成 one_way 时原来的 end_date 会被清掉；改成 round_trip 必须给 end_date",
        json_schema_extra={"value_labels": _TRIP_TYPE_LABELS},
    )
    end_date: date | None = Field(default=None, title="结束日期")
    purpose: str | None = Field(default=None, title="出差事由", min_length=1, max_length=1000)


class ExpenseClaimUpdateInput(SubmissionUpdateInput):
    expense_type: str | None = Field(default=None, title="费用类型", min_length=1, max_length=100)
    amount: Decimal | None = Field(default=None, title="金额", gt=0)
    currency: str | None = Field(default=None, title="币种", pattern=r"^[A-Z]{3}$")
    receipt_refs: list[str] | None = Field(default=None, title="票据号", min_length=1)
    travel_reference: str | None = Field(default=None, title="关联差旅单号", max_length=256)


class LeaveRequestUpdateInput(SubmissionUpdateInput):
    leave_type: str | None = Field(default=None, title="假期类型", min_length=1, max_length=100)
    start_date: date | None = Field(default=None, title="开始日期")
    end_date: date | None = Field(default=None, title="结束日期")
    reason: str | None = Field(default=None, title="请假原因", max_length=1000)


class MeetingBookingUpdateInput(SubmissionUpdateInput):
    room_id: str | None = Field(
        default=None,
        title="会议室",
        min_length=1,
        max_length=64,
        description="换房间时必须是空闲查询结果里的 room_id",
    )
    date: dt.date | None = Field(default=None, title="日期")
    start_time: time | None = Field(default=None, title="开始时间")
    end_time: time | None = Field(default=None, title="结束时间")
    subject: str | None = Field(default=None, title="会议主题", min_length=1, max_length=200)


class BusinessToolOutcome(BaseModel):
    tool: str
    success: bool
    status: str
    reference_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class EnterpriseToolProvider(Protocol):
    """企业系统边界；所有后端适配器必须遵循相同契约。"""

    async def search_policy(self, payload: PolicySearchInput) -> BusinessToolOutcome: ...

    async def create_travel_application(
        self, context: ToolContext, payload: TravelApplicationInput
    ) -> BusinessToolOutcome: ...

    async def create_expense_claim(
        self, context: ToolContext, payload: ExpenseClaimInput
    ) -> BusinessToolOutcome: ...

    async def find_available_rooms(
        self, context: ToolContext, payload: MeetingRoomSearchInput
    ) -> BusinessToolOutcome: ...

    async def book_meeting_room(
        self, context: ToolContext, payload: MeetingRoomBookingInput
    ) -> BusinessToolOutcome: ...

    async def get_leave_balance(
        self, context: ToolContext, payload: LeaveBalanceInput
    ) -> BusinessToolOutcome: ...

    async def submit_leave_request(
        self, context: ToolContext, payload: LeaveRequestInput
    ) -> BusinessToolOutcome: ...

    async def query_submissions(
        self, context: ToolContext, action_type: str, payload: SubmissionQueryInput
    ) -> BusinessToolOutcome: ...

    async def revoke_submission(
        self, context: ToolContext, action_type: str, payload: SubmissionRevokeInput
    ) -> BusinessToolOutcome: ...

    async def update_travel_application(
        self, context: ToolContext, payload: TravelApplicationUpdateInput
    ) -> BusinessToolOutcome: ...

    async def update_expense_claim(
        self, context: ToolContext, payload: ExpenseClaimUpdateInput
    ) -> BusinessToolOutcome: ...

    async def update_leave_request(
        self, context: ToolContext, payload: LeaveRequestUpdateInput
    ) -> BusinessToolOutcome: ...

    async def update_meeting_booking(
        self, context: ToolContext, payload: MeetingBookingUpdateInput
    ) -> BusinessToolOutcome: ...
