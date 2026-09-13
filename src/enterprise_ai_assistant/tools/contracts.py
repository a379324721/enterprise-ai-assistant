from datetime import date, time
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class TravelApplicationInput(StrictToolInput):
    destination: str = Field(min_length=1, max_length=200)
    start_date: date
    # 单程（调动、外派、返程另行申请）没有结束日期。行程类型单独成字段而不是只把
    # end_date 放开：否则模型漏问返程日期时也能直接提交，看起来和单程一模一样。
    trip_type: Literal["round_trip", "one_way"] = Field(
        default="round_trip",
        description="round_trip 往返，必须给出 end_date；one_way 单程，不填 end_date",
    )
    end_date: date | None = None
    purpose: str = Field(min_length=1, max_length=1000)

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
    expense_type: str = Field(min_length=1, max_length=100)
    amount: Decimal = Field(gt=0)
    currency: str = Field(default="CNY", pattern=r"^[A-Z]{3}$")
    receipt_refs: list[str] = Field(min_length=1)
    travel_reference: str | None = Field(default=None, max_length=256)


class LeaveBalanceInput(StrictToolInput):
    leave_type: str = Field(default="annual", min_length=1, max_length=100)


class LeaveRequestInput(StrictToolInput):
    leave_type: str = Field(min_length=1, max_length=100)
    start_date: date
    end_date: date
    reason: str | None = Field(default=None, max_length=1000)

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
    room_id: str = Field(min_length=1, max_length=64)
    date: date
    start_time: time
    end_time: time
    subject: str = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def validate_window(self) -> "MeetingRoomBookingInput":
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be later than start_time")
        return self


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
