from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_INPUT = "waiting_input"
    WAITING_CONFIRMATION = "waiting_confirmation"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"


class AgentName(StrEnum):
    SUPERVISOR = "supervisor"
    TRAVEL = "travel"
    EXPENSE = "expense"
    HR = "hr"
    POLICY = "policy"


class PlannedTask(BaseModel):
    """规划器输出；路由依据能力，而不是意图标签。"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    operation: str
    # 保持面向模型服务商的 JSON Schema 简单。部分 OpenAI 兼容 API
    # 不接受 Pydantic 为 Python 集合生成的 `uniqueItems` 关键字。
    required_capabilities: list[str] = Field(
        description=(
            "只能使用以下能力：travel.policy.read、travel.application.write、"
            "expense.policy.read、expense.claim.write、expense.reminder.write、"
            "hr.leave.read、hr.leave.write、policy.search"
        )
    )
    depends_on: list[str] = Field(default_factory=list)
    risk: str = Field(default="low", pattern="^(low|medium|high)$")
    status: TaskStatus = TaskStatus.PENDING


class ExtractedSlots(BaseModel):
    model_config = ConfigDict(extra="forbid")

    destination: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    purpose: str | None = None
    is_one_way: bool | None = None
    leave_type: str | None = None
    leave_start: str | None = None
    leave_end: str | None = None
    expense_amount: float | None = None


class TaskPlan(BaseModel):
    user_goal: str
    tasks: list[PlannedTask]
    extracted_slots: ExtractedSlots = Field(default_factory=ExtractedSlots)
    direct_answer: str = Field(
        default="",
        description="无需创建业务任务时，直接回复用户的自然语言内容",
    )

    @model_validator(mode="after")
    def validate_dependencies(self) -> "TaskPlan":
        ids = {task.id for task in self.tasks}
        if len(ids) != len(self.tasks):
            raise ValueError("task ids must be unique")
        for task in self.tasks:
            if task.id in task.depends_on or not set(task.depends_on) <= ids:
                raise ValueError(f"invalid dependencies for task {task.id}")
        return self


class TaskRun(BaseModel):
    user_goal: str
    tasks: list[PlannedTask]


class GoalUnderstanding(BaseModel):
    normalized_goal: str
    explicit_constraints: list[str] = Field(default_factory=list)
    inferred_slots: ExtractedSlots = Field(
        default_factory=ExtractedSlots,
        description=(
            "使用规范槽位名：差旅使用 destination、start_date、end_date、purpose；"
            "单程标记使用 is_one_way；请假使用 leave_type、leave_start、leave_end；"
            "报销金额使用 expense_amount"
        ),
    )
    ambiguities: list[str] = Field(default_factory=list)


class PendingConfirmation(BaseModel):
    confirmation_id: UUID = Field(default_factory=uuid4)
    task_id: str
    action: str
    summary: str
    payload: dict[str, Any]
    requested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ToolResult(BaseModel):
    task_id: str
    tool: str
    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TravelApplication(BaseModel):
    application_id: UUID = Field(default_factory=uuid4)
    user_id: str
    destination: str
    start_date: date
    end_date: date
    purpose: str
    status: str = "submitted"
