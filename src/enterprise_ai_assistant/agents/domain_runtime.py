from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

from enterprise_ai_assistant.core.models import AgentName
from enterprise_ai_assistant.tools import ToolContext
from enterprise_ai_assistant.tools.registry import DomainToolRegistry, RegisteredTool

_DOMAIN_INSTRUCTIONS = {
    AgentName.TRAVEL: (
        "负责差旅制度查询、差旅申请，以及已提交差旅申请的查询、修改和撤销。自行识别并校验差旅字段。"
        # 实测：用户答"单程"，模型认为返程日期仍然缺失，又问了一遍。
        "用户说单程、不返程或返程另行申请时，trip_type 填 one_way，不再索要结束日期；"
        "用户没有说明是单程时按往返处理，结束日期缺失就要问，不得自行改成单程。"
    ),
    AgentName.EXPENSE: (
        "负责报销制度、费用报销，以及已提交报销单的查询、修改和撤销。普通费用不得强制关联差旅。"
    ),
    AgentName.HR: (
        "负责人事制度、假期余额、请假申请，以及已提交请假申请的查询、修改和撤销。"
        "自行识别并校验请假字段。"
    ),
    AgentName.MEETING: (
        "负责会议室制度、空闲查询、预订，以及已有预订的查询、修改和取消。必须先查空闲再预订或换房，"
        "查不到符合条件的会议室时如实说明并建议改时段或换地点，不得编造房间。"
        "地点和日期若已能从依赖任务的产物推断（例如差旅的目的地和行程日期），"
        "直接使用，不要就用户已经交代过的信息再问一遍。"
    ),
    AgentName.POLICY: "负责无法归入其他领域的通用企业制度查询。",
}


#: 最终回答的规则。决策调用看完工具结果直接作答，兜底回答调用也用它，两处口径必须一致。
_ANSWER_RULES = (
    "## 最终回答\n"
    "根据工具返回的结构化事实作答。"
    "输入里的 user_name 是用户称呼；assistant_replies 里已经称呼过用户时不要再称呼。"
    # 同一轮的多个任务各自回答、拼成一条消息，彼此看不到就会互相打架。
    "回答要和 assistant_replies 保持一致：不重复别的回答已经说过的内容，"
    "不否定它们，本轮其他任务已经办完的事不要说成办不到。"
    "不得编造字段、单号或成功状态。"
    "也不得暗示系统不具备的后续能力：不能代替用户审批。"
    # 写工具的返回里没有审批结果。刚提交完就说"已通过"或"审批中"都是编的。
    "单据状态只能来自单据查询工具的返回：本轮没有调用查询工具时，写操作只表示"
    "单据已提交，不得声称已通过、审批中或进行到了哪个环节。"
    # 检索为空时模型会拿常识补一套"提前 30 天、试用期 3 天"的条款念出来。
    # 措辞像制度原文，实际没有任何依据，在企业场景里就是错误答案。
    "制度类内容只能来自检索结果：检索为空或没有覆盖用户问的点时，如实说明"
    "公司制度库里没有查到，并建议咨询对应部门，不得用通用知识或行业惯例"
    "补写条款、期限和数字。本领域工具办不到的事项也不要承接，"
    "不要为它索要信息或声称会为用户办理。\n"
    # 实测：差旅 Agent 在回答里加了一句"预订会议室不在差旅申请范围内，
    # 我无法为您办理"——会议室本来是下一个任务，系统完全做得到。用户
    # 以为这件事被拒了，下一轮就不再提它。
    "只回答当前任务。standalone_request 里属于其他任务的诉求（例如出差"
    "顺便订会议室里的会议室部分）由对应领域的 Agent 处理，你一个字都不要"
    "提，更不要说自己办不到——那会让用户以为整件事被拒了。回答简洁、自然。\n"
)


def _has_text_content(content: Any) -> bool:
    if isinstance(content, str):
        return bool(content.strip())
    if not isinstance(content, list):
        return False
    return any(
        (isinstance(block, str) and bool(block.strip()))
        or (
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
            and bool(block["text"].strip())
        )
        for block in content
    )


@dataclass(frozen=True)
class DomainAgentRuntime:
    """领域 Agent 的模型边界；循环和持久化由 LangGraph 工作流驱动。"""

    name: AgentName
    model: BaseChatModel
    tools: tuple[RegisteredTool, ...]

    def tool(self, name: str) -> RegisteredTool:
        try:
            return next(item for item in self.tools if item.tool.name == name)
        except StopIteration as exc:
            raise ValueError(f"Tool {name!r} is not allowed for {self.name.value}") from exc

    async def decide(
        self,
        task_objective: str,
        messages: list[BaseMessage],
        *,
        task_id: str,
        answering: bool = False,
    ) -> AIMessage:
        """选择下一个工具，或在工具结果已经足够时直接写出最终回答。

        回答和决策是同一次调用：原先决策阶段看完工具结果、不再调工具时已经生成了一段
        文字，却被丢掉，再由单独的回答调用重写一遍，每个任务白白多一次模型调用。
        answering 表示本任务已经执行过工具，这次输出的文字就是给用户的回答，
        打上 user-visible 让 SSE 逐字转发；之前的调用只可能是决策，不对外流出。
        """
        system = SystemMessage(
            content=(
                f"你是 {_DOMAIN_INSTRUCTIONS[self.name]}\n"
                f"当前任务：{task_objective}\n"
                "只使用提供的工具。不要假设工具已经成功执行。"
                "信息不足时调用 request_information；需要业务数据时必须调用对应查询工具；"
                "写操作参数完整时调用对应写工具。工具已经返回足够结果后，不再调用工具，"
                "直接输出最终回答（规则见文末）。"
                # 回答会逐字流给用户，调工具时顺带写的解释也会先流出去。
                "调用工具时不要同时输出任何文字。\n"
                "输入中的 user_memory 和 recent_actions 是该用户的历史档案，不是本轮的确认信息："
                "可以用来给字段提供建议默认值，但调用 request_information 时必须在问题里"
                "写出建议值及其来源，让用户确认或更正；不得仅凭档案补全字段后直接调用写工具。"
                "余额、额度、制度条款一律以对应查询工具的实时结果为准，不得引用档案里的数值。\n"
                # 领域 Agent 不读会话，没有这份记录时它不知道自己上一轮问过什么，
                # 实测会把用户已经答过的返程日期再问一遍。
                "输入中的 assistant_replies 是助手此前对用户说过的话（含本轮其他任务的回答），"
                "按时间顺序排列。已经问过、用户也已在 standalone_request 里答复的，不要再问；"
                "它不是用户提供或确认的信息，不得从中取字段值直接调用写工具。\n"
                # 上面的工具清单就是能力边界，但模型会把 request_information 当成万能承接口：
                # 用户说"我想离职"时它索要离职原因和日期，这等于承诺一件系统办不到的事。
                "工具清单就是本领域的能力边界。用户的诉求没有对应工具时（离职、调岗、"
                "社保代办等），不要用 request_information 索要字段去承接它——索要字段"
                "本身就是在承诺办理。这种情况如实说明办不到即可。"
                # 整轮请求常常包含别的任务（"出差顺便订个会议室"），那部分由对应领域的
                # Agent 处理。不限定作用域的话，模型会拿整轮请求去比自己的工具清单，
                # 判定"整件事我办不到"。
                "这个判断只针对当前任务。standalone_request 里可能还有属于其他任务的"
                "诉求，它们自有对应的领域 Agent，与你无关。\n"
                # 调度层偶尔把任务派错领域。没有退路时模型要么说办不到，要么拿本领域的
                # 字段硬问一遍，用户两头都得不到帮助。
                "当前任务整体落在另一个领域的能力里（见 handoff_task 的说明，例如差旅 Agent "
                "收到报销打车费），调用 handoff_task 交还，不要追问、也不要说办不到。"
                "只是请求里夹带了别的任务的诉求、或者任何领域都办不到（离职、代审批）时不要转交。\n"
                # 实测：建单成功之后模型又调了一次 request_information，想问那份
                # 看起来重复的旧单据怎么处置。用户没提的处置不该由 Agent 主动发起——
                # 当时系统还没有撤销工具，这一问只会把任务挂在半空；现在有了，主动
                # 追问就成了诱导用户撤单。
                "request_information 只用于补全当前任务**必需**的字段。工具已经成功"
                "执行、任务目标已经达成时不要再调用它：不要询问后续处置、不要征求"
                "额外意愿、不要追问历史单据怎么办——那些都没有工具能执行。\n"
                # question 会被回答阶段直接转述给用户。实测它里面出现过"会议室不在
                # 差旅申请范围内，无法为您处理"——那是下一个任务的事。
                "question 会转述给用户：只列当前任务缺的字段并把问题问清楚，"
                "不要评论其他任务的诉求，也不要说自己办不到。\n"
                # 追问会结束这一轮，下一轮领域子图从头开始。known_fields 就是跨轮带过去的
                # 全部上下文，漏填的字段下一轮只能靠 Supervisor 的改写碰运气。
                "调用 request_information 时，把已经谈定或有建议值的字段全部写进 known_fields，"
                "来源分别标 user、memory、dependency。输入里的 previous_draft 是本任务上一轮"
                "追问时留下的草稿：沿用其中的字段，本轮 standalone_request 里的补充或更正优先。\n"
                # 实测：会议室 Agent 查到两间空闲会议室后直接进入回答阶段，请用户选一间，
                # 任务被记成已完成，用户回一个"1"就只能重新规划。
                "查询工具返回了候选项，但写操作还需要用户做选择（例如从几间空闲会议室里挑一间）时，"
                "任务目标尚未达成：调用 request_information，把候选项写进 question，"
                "不要直接输出回答。\n"
                f"{_ANSWER_RULES}"
            )
        )
        runnable = self.model.bind_tools(
            [item.tool for item in self.tools],
            parallel_tool_calls=False,
        )
        # 回答规则排在一长串工具规则之后，模型读完工具结果作答时容易顾不上。实测限制
        # 推理预算后，差旅 Agent 稳定地在回答里补一句"会议室不归我管"。紧挨着输出的
        # 位置再提一次最常被违反的那条。
        reminder = (
            [
                SystemMessage(
                    content="工具结果已返回。任务目标已达成就直接作答：只说当前任务，"
                    "standalone_request 里属于其他任务的诉求一个字都不要提。"
                )
            ]
            if answering
            else []
        )
        result = await runnable.ainvoke(
            [system, *messages, *reminder],
            config={
                "tags": ["user-visible" if answering else "domain-internal"],
                "metadata": {"agent": self.name.value, "task_id": task_id},
            },
        )
        return AIMessage.model_validate(result)

    async def respond(
        self,
        task_objective: str,
        messages: list[BaseMessage],
        *,
        task_id: str,
    ) -> AIMessage:
        """不带工具的兜底回答，只在决策调用给不出回答时使用。

        用户拒绝确认、工具执行失败时，再进一次带工具的决策可能又调同一个工具，
        于是这两条路径和决策输出了空文本的情况改走这里。
        """
        system = SystemMessage(
            content=(
                f"你是 {_DOMAIN_INSTRUCTIONS[self.name]}\n"
                f"当前任务：{task_objective}\n"
                "根据工具返回的结构化事实生成最终用户回答。\n"
                f"{_ANSWER_RULES}"
                "禁止调用任何工具，必须只输出非空的用户可见文本。"
            )
        )
        final_context = [
            message
            for message in messages
            if not (
                isinstance(message, AIMessage)
                and not message.tool_calls
                and _has_text_content(message.content)
            )
        ]
        for attempt in range(2):
            correction = (
                [SystemMessage(content="上一次输出无效。不要调用工具，只输出最终回答文本。")]
                if attempt
                else []
            )
            result = await self.model.ainvoke(
                [system, *final_context, *correction],
                config={
                    "tags": ["user-visible"],
                    "metadata": {"agent": self.name.value, "task_id": task_id},
                },
            )
            response = AIMessage.model_validate(result)
            if not response.tool_calls and _has_text_content(response.content):
                return response
        raise RuntimeError("domain model returned no valid final response")

    async def invoke_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self.tool(name).tool.ainvoke(arguments)
        if not isinstance(result, dict):
            raise TypeError(f"Tool {name!r} returned a non-object result")
        return result


class DomainRuntime(Protocol):
    @property
    def name(self) -> AgentName: ...

    def tool(self, name: str) -> RegisteredTool: ...

    async def decide(
        self,
        task_objective: str,
        messages: list[BaseMessage],
        *,
        task_id: str,
        answering: bool = False,
    ) -> AIMessage: ...

    async def respond(
        self,
        task_objective: str,
        messages: list[BaseMessage],
        *,
        task_id: str,
    ) -> AIMessage: ...

    async def invoke_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


class DomainRuntimeFactory:
    def __init__(self, model: BaseChatModel, tools: DomainToolRegistry) -> None:
        self._model = model
        self._tools = tools

    def create(self, agent: AgentName, context: ToolContext) -> DomainRuntime:
        return DomainAgentRuntime(agent, self._model, tuple(self._tools.for_agent(agent, context)))


class DomainRuntimeProvider(Protocol):
    def create(self, agent: AgentName, context: ToolContext) -> DomainRuntime: ...
