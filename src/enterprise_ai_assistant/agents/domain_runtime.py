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


#: 领域 Agent 的原则，决策和兜底回答共用。按原则写，不按 badcase 逐条加禁止句：出现新的越界说法，
#: 先看能归到哪一条。原先一个实测问题补一条规则，同一个意思在决策规则、回答规则、作答提醒里各说
#: 一遍，写到三千字，模型读到回答那段时已经顾不上前面。
_PRINCIPLES = (
    "## 事实只来自工具\n"
    "单号、单据状态、余额、制度条款只能来自本任务工具的返回，工具没返回就是不知道：查询为空就说没查到，"
    "检索没覆盖就说制度库里没有、建议咨询对应部门，不用常识或会话里提到过的内容补。写工具只表示已提交，"
    "没有审批结果。user_memory、recent_actions 是历史档案，不是查询结果。\n"
    "## 字段只来自用户针对这件事说过的话\n"
    "写工具的参数只能取自 standalone_request、previous_draft、dependency_results，以及 recent_messages 里"
    "用户针对当前这件事说的话；改写漏了或写错时以用户原话为准。assistant 消息、别的事项、历史档案都不是来源——"
    "档案可以作建议值，但要在追问里写明来源让用户确认。拿不准就问，不猜。\n"
    "## 只管当前任务，只做工具能做的\n"
    "standalone_request 里别的任务的诉求由对应的 Agent 处理：不提、不评论，也不说办不到或会转交。"
    "当前任务整体属于另一个领域时调用 handoff_task 交还。诉求没有对应工具（离职、代审批等）时如实说办不到，"
    "不用 request_information 索要字段承接——索要就等于承诺办理；任务目标达成后也不追问后续处置。\n"
    "## 回答\n"
    "简洁自然。和 recent_messages 里助手说过的话保持一致，不重复、不否定；称呼过 user_name 就不再称呼。"
    "助手消息开头的 [未调用工具] / [调用了：…] 是系统标注，不是回答内容。\n"
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
        answering 表示本任务已经执行过工具，这次输出的文字（回答，或调写工具时对用户说的话）
        是给用户的，打上 user-visible 让 SSE 转发；第一次调用手里还没有任何工具结果，不对外流出。
        """
        system = SystemMessage(
            content=(
                f"你是 {_DOMAIN_INSTRUCTIONS[self.name]}\n"
                f"当前任务：{task_objective}\n"
                "## 怎么推进\n"
                # 顺序有意义：查询排在前面时，"我要请假"这种什么都没给的请求，模型会先去查一遍制度。
                "- 缺当前任务必需的字段，或者需要用户从查询结果里挑一个（例如几间空闲会议室），"
                "调用 request_information：question 原样发给用户，只问缺的、列出候选项；"
                "known_fields 写全已谈定或有建议值的字段（来源标 user、memory、dependency）。"
                # 追问结束这一轮，下一轮子图从头开始，草稿是跨轮带过去的全部上下文。
                "previous_draft 是上一轮的草稿，沿用它，本轮的补充和更正优先。\n"
                "- 回答或写入需要业务数据时调查询工具；参数齐全就调写工具；工具结果足够时不再调工具，"
                "直接写回答。"
                # 回答会逐字流给用户，调工具时顺带写的解释也会先流出去。要说的话走写工具的
                # message_to_user：模型调工具那一回合几乎不写正文，参数却每次都会填。
                "调工具时不要同时输出正文；调写工具前要对用户说的话写在 message_to_user 里。\n"
                f"{_PRINCIPLES}"
            )
        )
        runnable = self.model.bind_tools(
            [item.tool for item in self.tools],
            parallel_tool_calls=False,
        )
        result = await runnable.ainvoke(
            [system, *messages],
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
                "根据工具返回的结果写给用户的回答。\n"
                f"{_PRINCIPLES}"
                "不调用任何工具，只输出非空的回答文本。"
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


class DomainRuntimeProvider(Protocol):
    def create(self, agent: AgentName, context: ToolContext) -> DomainRuntime: ...


class DomainRuntimeFactory(DomainRuntimeProvider):
    def __init__(self, model: BaseChatModel, tools: DomainToolRegistry) -> None:
        self._model = model
        self._tools = tools

    def create(self, agent: AgentName, context: ToolContext) -> DomainRuntime:
        return DomainAgentRuntime(agent, self._model, tuple(self._tools.for_agent(agent, context)))
