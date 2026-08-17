from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_CONFIRMATION = "waiting_confirmation"
    WAITING_INPUT = "waiting_input"
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
    """Planner 只描述领域目标和依赖，不决定字段、工具或风险。"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(min_length=1, max_length=200)
    domain: AgentName
    objective: str = Field(min_length=1, max_length=2000)
    depends_on: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list, max_length=20)
    status: TaskStatus = TaskStatus.PENDING

    @model_validator(mode="after")
    def reject_supervisor_domain(self) -> "PlannedTask":
        if self.domain == AgentName.SUPERVISOR:
            raise ValueError("supervisor cannot execute a domain task")
        return self


class TaskPlan(BaseModel):
    user_goal: str = Field(min_length=1, max_length=8000)
    tasks: list[PlannedTask] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_dependencies(self) -> "TaskPlan":
        ids = {task.id for task in self.tasks}
        if len(ids) != len(self.tasks):
            raise ValueError("task ids must be unique")
        for task in self.tasks:
            if task.id in task.depends_on or not set(task.depends_on) <= ids:
                raise ValueError(f"invalid dependencies for task {task.id}")
        dependencies = {task.id: set(task.depends_on) for task in self.tasks}
        ready = [task_id for task_id, required in dependencies.items() if not required]
        visited: set[str] = set()
        while ready:
            completed = ready.pop()
            if completed in visited:
                continue
            visited.add(completed)
            for task_id, required in dependencies.items():
                if task_id not in visited and required <= visited:
                    ready.append(task_id)
        if visited != ids:
            raise ValueError("task dependencies must form an acyclic graph")
        return self


class ContextResolution(BaseModel):
    """Supervisor 对完整会话的解析结果，不包含任何领域业务字段。"""

    standalone_request: str = Field(min_length=1, max_length=8000)
    intent_summary: str = Field(min_length=1, max_length=1000)
    requires_task_planning: bool
    explicit_constraints: list[str] = Field(default_factory=list)
    referenced_task_ids: list[str] = Field(default_factory=list)
    unresolved_references: list[str] = Field(default_factory=list)


class PendingConfirmation(BaseModel):
    confirmation_id: UUID = Field(default_factory=uuid4)
    task_id: str
    action: str
    tool_call_id: str
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
