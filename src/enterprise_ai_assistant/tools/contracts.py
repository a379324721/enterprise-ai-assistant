from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class ToolRisk(StrEnum):
    """风险由服务端工具注册表声明，不能由模型自行决定。"""

    READ = "read"
    WRITE = "write"


class StrictToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolContext(BaseModel):
    """可信运行时上下文，不作为模型可填写的工具参数暴露。"""

    user_id: str = Field(min_length=1, max_length=128)
    task_id: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=1, max_length=256)


class PolicySearchInput(StrictToolInput):
    query: str = Field(min_length=1, max_length=2000)
    domain: str = Field(min_length=1, max_length=32)
    limit: int = Field(default=3, ge=1, le=10)


class PolicyQueryInput(StrictToolInput):
    """暴露给领域模型的查询参数；领域由工具注册表固定。"""

    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=3, ge=1, le=10)


class InformationRequestInput(StrictToolInput):
    missing_fields: list[str] = Field(min_length=1)
    question: str = Field(min_length=1, max_length=1000)


class TravelApplicationInput(StrictToolInput):
    destination: str = Field(min_length=1, max_length=200)
    start_date: date
    end_date: date
    purpose: str = Field(min_length=1, max_length=1000)


class ExpenseClaimInput(StrictToolInput):
    expense_type: str = Field(min_length=1, max_length=100)
    amount: Decimal = Field(gt=0)
    currency: str = Field(default="CNY", pattern=r"^[A-Z]{3}$")
    receipt_refs: list[str] = Field(min_length=1)
    travel_reference: str | None = Field(default=None, max_length=256)


class ExpenseReminderInput(StrictToolInput):
    trigger_date: date
    note: str = Field(min_length=1, max_length=1000)
    travel_reference: str | None = Field(default=None, max_length=256)


class LeaveBalanceInput(StrictToolInput):
    leave_type: str = Field(default="annual", min_length=1, max_length=100)


class LeaveRequestInput(StrictToolInput):
    leave_type: str = Field(min_length=1, max_length=100)
    start_date: date
    end_date: date
    reason: str | None = Field(default=None, max_length=1000)


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

    async def schedule_expense_reminder(
        self, context: ToolContext, payload: ExpenseReminderInput
    ) -> BusinessToolOutcome: ...

    async def get_leave_balance(
        self, context: ToolContext, payload: LeaveBalanceInput
    ) -> BusinessToolOutcome: ...

    async def submit_leave_request(
        self, context: ToolContext, payload: LeaveRequestInput
    ) -> BusinessToolOutcome: ...
