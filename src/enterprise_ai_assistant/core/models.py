from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
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
    required_capabilities: list[str]
    depends_on: list[str] = Field(default_factory=list)
    risk: str = Field(default="low", pattern="^(low|medium|high)$")
    status: TaskStatus = TaskStatus.PENDING


class TaskPlan(BaseModel):
    user_goal: str
    tasks: list[PlannedTask]
    extracted_slots: dict[str, Any] = Field(default_factory=dict)
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


class GoalUnderstanding(BaseModel):
    normalized_goal: str
    explicit_constraints: list[str] = Field(default_factory=list)
    inferred_slots: dict[str, Any] = Field(default_factory=dict)
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
