"""评测数据集的结构定义与加载。

数据集与执行逻辑分离：`cases.yaml` 只描述"给什么输入、期望什么行为"，
任何 prompt 调整后都用同一份数据集回归对比，避免凭感觉判断效果。
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from enterprise_ai_assistant.core.models import AgentName

DATASET_PATH = Path(__file__).with_name("cases.yaml")


class ConversationTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1)


class ContextCase(BaseModel):
    """考察 Context Supervisor 能否区分闲聊与业务请求，并消解指代。"""

    id: str = Field(min_length=1)
    conversation: list[ConversationTurn] = Field(min_length=1)
    expect_task_planning: bool
    expect_keywords: list[str] = Field(default_factory=list)
    note: str = ""


class PlanningCase(BaseModel):
    """考察 Planner 的领域路由、拆分粒度和依赖识别。"""

    id: str = Field(min_length=1)
    request: str = Field(min_length=1)
    expect_domains: list[AgentName] = Field(min_length=1)
    expect_task_count: int = Field(ge=1, le=20)
    expect_dependency: bool = False
    note: str = ""


class ToolChoiceCase(BaseModel):
    """考察领域 Agent 在信息充分时是否选中正确工具。"""

    id: str = Field(min_length=1)
    domain: AgentName
    objective: str = Field(min_length=1)
    user_goal: str = Field(min_length=1)
    expect_tool: str = Field(min_length=1)
    note: str = ""


class GuardrailCase(BaseModel):
    """考察护栏：诱导性输入不得触发写操作，字段缺失必须反问而非编造。"""

    id: str = Field(min_length=1)
    domain: AgentName
    objective: str = Field(min_length=1)
    user_goal: str = Field(min_length=1)
    forbid_write: bool = False
    expect_information_request: bool = False
    note: str = ""


class EvalDataset(BaseModel):
    context_cases: list[ContextCase] = Field(default_factory=list)
    planning_cases: list[PlanningCase] = Field(default_factory=list)
    tool_choice_cases: list[ToolChoiceCase] = Field(default_factory=list)
    guardrail_cases: list[GuardrailCase] = Field(default_factory=list)

    def case_ids(self) -> list[str]:
        return [
            case.id
            for group in (
                self.context_cases,
                self.planning_cases,
                self.tool_choice_cases,
                self.guardrail_cases,
            )
            for case in group
        ]


def load_dataset(path: Path = DATASET_PATH) -> EvalDataset:
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    return EvalDataset.model_validate(raw)
