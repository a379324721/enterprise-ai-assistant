"""评测数据集的结构定义与加载。

数据集与执行逻辑分离：`cases.yaml` 只描述"给什么输入、期望什么行为"，
任何 prompt 调整后都用同一份数据集回归对比，避免凭感觉判断效果。
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from enterprise_ai_assistant.core.models import AgentName, OpenTask, TurnRelation

DATASET_PATH = Path(__file__).with_name("cases.yaml")


class ConversationTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1)


class ContextCase(BaseModel):
    """考察 Context Supervisor 能否区分闲聊与业务请求，并消解指代。"""

    id: str = Field(min_length=1)
    conversation: list[ConversationTurn] = Field(min_length=1)
    expect_task_planning: bool
    # 全部必须出现在改写后的请求里。
    expect_keywords: list[str] = Field(default_factory=list)
    # 至少命中一个即可。用于指代消解有多种正确表达的场景，例如既可以点名
    # 实体（"杭州"），也可以引用更精确的业务单号（"TR-001"）。
    expect_any_keywords: list[str] = Field(default_factory=list)
    # 上一轮停在待补充的任务。补充信息的短回复只有放在它们下面才能认出来。
    open_tasks: list[OpenTask] = Field(default_factory=list)
    # 留空表示不断言。
    expect_turn_relation: TurnRelation | None = None
    expect_target_plan_id: str | None = None
    # 指向这些事项即判失败，用于"可以留空、但绝不能指错"的场景。
    forbid_target_plan_ids: list[str] = Field(default_factory=list)
    # 留空表示不断言。只在 requires_task_planning 为 true 时有意义：领域归错了，
    # 单领域快路径会把请求交给没有对应工具的 Agent。
    expect_domains: list[AgentName] = Field(default_factory=list)
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
    # 同样正确的第一步。只用于确实存在多个合理起手的场景，例如修改前先查原单。
    also_accept: list[str] = Field(default_factory=list)
    # 注入该用户的历史画像，形如 "preferred_transport=高铁"。
    memories: list[str] = Field(default_factory=list)
    # 交给领域 Agent 的最近会话原文，形如 {role: user|assistant, content: ...}。
    recent_messages: list[ConversationTurn] = Field(default_factory=list)
    note: str = ""


class GuardrailCase(BaseModel):
    """考察护栏：诱导性输入不得触发写操作，字段缺失必须反问而非编造。"""

    id: str = Field(min_length=1)
    domain: AgentName
    objective: str = Field(min_length=1)
    user_goal: str = Field(min_length=1)
    forbid_write: bool = False
    expect_information_request: bool = False
    # 用于诉求超出本领域工具范围的场景：反问字段等于承诺办理，和编造字段一样是越界。
    forbid_information_request: bool = False
    # 用于诉求不属于任何领域、或只是夹带了别的任务的场景：转交只会让任务在领域间空转。
    forbid_handoff: bool = False
    memories: list[str] = Field(default_factory=list)
    # 交给领域 Agent 的最近会话原文，形如 {role: user|assistant, content: ...}。
    recent_messages: list[ConversationTurn] = Field(default_factory=list)
    note: str = ""


class DomainAnswerCase(BaseModel):
    """考察领域 Agent 的最终回答不越出系统真实能力。

    写操作只表示单据已提交，状态只能来自单据查询工具；系统没有代审批的工具，
    回答里出现这类承诺就是幻觉。
    """

    id: str = Field(min_length=1)
    domain: AgentName
    objective: str = Field(min_length=1)
    user_goal: str = Field(min_length=1)
    tool_results: list[str] = Field(default_factory=list)
    forbid_phrases: list[str] = Field(default_factory=list)
    note: str = ""


class SmallTalkCase(BaseModel):
    """考察 Supervisor 直接回复时不越出事实：不编审批状态、不许诺、不重复自己说过的话。

    workflow_actions 只记录写操作被调用过，不含审批结果；单据清单进入上下文后，
    最大的风险就是模型顺口编出一个状态。回复要求本轮确实不执行任务。
    """

    id: str = Field(min_length=1)
    conversation: list[ConversationTurn] = Field(min_length=1)
    user_name: str = ""
    recent_actions: list[str] = Field(default_factory=list)
    # 回答中一旦出现这些说法即判失败。
    forbid_phrases: list[str] = Field(default_factory=list)
    note: str = ""


class EvalDataset(BaseModel):
    context_cases: list[ContextCase] = Field(default_factory=list)
    planning_cases: list[PlanningCase] = Field(default_factory=list)
    tool_choice_cases: list[ToolChoiceCase] = Field(default_factory=list)
    guardrail_cases: list[GuardrailCase] = Field(default_factory=list)
    small_talk_cases: list[SmallTalkCase] = Field(default_factory=list)
    domain_answer_cases: list[DomainAnswerCase] = Field(default_factory=list)

    def case_ids(self) -> list[str]:
        return [
            case.id
            for group in (
                self.context_cases,
                self.planning_cases,
                self.tool_choice_cases,
                self.guardrail_cases,
                self.small_talk_cases,
                self.domain_answer_cases,
            )
            for case in group
        ]


def load_dataset(path: Path = DATASET_PATH) -> EvalDataset:
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    return EvalDataset.model_validate(raw)
