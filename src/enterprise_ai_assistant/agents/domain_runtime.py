from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

from enterprise_ai_assistant.core.models import AgentName
from enterprise_ai_assistant.tools import ToolContext
from enterprise_ai_assistant.tools.registry import DomainToolRegistry, RegisteredTool

_DOMAIN_INSTRUCTIONS = {
    AgentName.TRAVEL: "负责差旅制度查询和差旅申请。自行识别并校验差旅字段。",
    AgentName.EXPENSE: "负责报销制度、费用报销和报销提醒。普通费用不得强制关联差旅。",
    AgentName.HR: "负责人事制度、假期余额和请假申请。自行识别并校验请假字段。",
    AgentName.POLICY: "负责无法归入其他领域的通用企业制度查询。",
}


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
    ) -> AIMessage:
        system = SystemMessage(
            content=(
                f"你是 {_DOMAIN_INSTRUCTIONS[self.name]}\n"
                f"当前任务：{task_objective}\n"
                "只使用提供的工具。不要假设工具已经成功执行。"
                "信息不足时调用 request_information；需要业务数据时必须调用对应查询工具；"
                "写操作参数完整时调用对应写工具。工具已经返回足够结果后，不再调用工具。"
                "工具决策阶段不要输出面向用户的解释。"
            )
        )
        runnable = self.model.bind_tools(
            [item.tool for item in self.tools],
            parallel_tool_calls=False,
        )
        result = await runnable.ainvoke(
            [system, *messages],
            config={
                "tags": ["domain-internal"],
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
        system = SystemMessage(
            content=(
                f"你是 {_DOMAIN_INSTRUCTIONS[self.name]}\n"
                f"当前任务：{task_objective}\n"
                "根据工具返回的结构化事实生成最终用户回答。"
                "不得编造字段、单号或成功状态。"
                "如果缺少信息，直接提出工具结果中给出的问题。回答简洁、自然。"
                "这是最终回答阶段：禁止调用任何工具，必须只输出非空的用户可见文本。"
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
