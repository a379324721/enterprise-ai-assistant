from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from enterprise_ai_assistant.core.models import AgentName
from enterprise_ai_assistant.tools.contracts import (
    BusinessToolOutcome,
    EnterpriseToolProvider,
    ExpenseClaimInput,
    ExpenseReminderInput,
    InformationRequestInput,
    LeaveBalanceInput,
    LeaveRequestInput,
    PolicyQueryInput,
    PolicySearchInput,
    ToolContext,
    ToolRisk,
    TravelApplicationInput,
)


@dataclass(frozen=True)
class RegisteredTool:
    tool: BaseTool
    risk: ToolRisk
    terminal: bool = False


class DomainToolRegistry:
    """为每个领域构造最小工具集，并在服务端固定风险等级。"""

    def __init__(self, provider: EnterpriseToolProvider) -> None:
        self._provider = provider

    @staticmethod
    def _tool(
        *,
        name: str,
        description: str,
        args_schema: type[Any],
        coroutine: Callable[..., Awaitable[dict[str, Any]]],
        risk: ToolRisk,
        terminal: bool = False,
    ) -> RegisteredTool:
        return RegisteredTool(
            tool=StructuredTool.from_function(
                coroutine=coroutine,
                name=name,
                description=description,
                args_schema=args_schema,
            ),
            risk=risk,
            terminal=terminal,
        )

    def for_agent(self, agent: AgentName, context: ToolContext) -> list[RegisteredTool]:
        async def request_information(**kwargs: Any) -> dict[str, Any]:
            payload = InformationRequestInput.model_validate(kwargs)
            return BusinessToolOutcome(
                tool="request_information",
                success=True,
                status="needs_user_input",
                data=payload.model_dump(mode="json"),
            ).model_dump(mode="json")

        information_tool = self._tool(
            name="request_information",
            description=(
                "当完成当前领域任务所需的信息缺失或存在歧义时调用。"
                "列出缺失字段，并给出一条面向用户的明确问题。"
            ),
            args_schema=InformationRequestInput,
            coroutine=request_information,
            risk=ToolRisk.READ,
            terminal=True,
        )

        async def policy_search(domain: str, **kwargs: Any) -> dict[str, Any]:
            query = PolicyQueryInput.model_validate(kwargs)
            outcome = await self._provider.search_policy(
                PolicySearchInput(domain=domain, **query.model_dump())
            )
            return outcome.model_dump(mode="json")

        def policy_tool(domain: str) -> RegisteredTool:
            async def invoke_policy(**kwargs: Any) -> dict[str, Any]:
                return await policy_search(domain, **kwargs)

            return self._tool(
                name=f"search_{domain}_policy",
                description=f"查询{domain}领域的企业制度；回答制度问题前必须先调用。",
                args_schema=PolicyQueryInput,
                coroutine=invoke_policy,
                risk=ToolRisk.READ,
            )

        if agent == AgentName.TRAVEL:
            async def create_travel(**kwargs: Any) -> dict[str, Any]:
                outcome = await self._provider.create_travel_application(
                    context,
                    TravelApplicationInput.model_validate(kwargs),
                )
                return outcome.model_dump(mode="json")

            return [
                policy_tool("travel"),
                self._tool(
                    name="create_travel_application",
                    description="创建差旅申请。只有字段完整且用户明确要求创建时才能调用。",
                    args_schema=TravelApplicationInput,
                    coroutine=create_travel,
                    risk=ToolRisk.WRITE,
                ),
                information_tool,
            ]

        if agent == AgentName.EXPENSE:
            async def create_claim(**kwargs: Any) -> dict[str, Any]:
                outcome = await self._provider.create_expense_claim(
                    context,
                    ExpenseClaimInput.model_validate(kwargs),
                )
                return outcome.model_dump(mode="json")

            async def schedule_reminder(**kwargs: Any) -> dict[str, Any]:
                outcome = await self._provider.schedule_expense_reminder(
                    context,
                    ExpenseReminderInput.model_validate(kwargs),
                )
                return outcome.model_dump(mode="json")

            return [
                policy_tool("expense"),
                self._tool(
                    name="create_expense_claim",
                    description="创建费用报销单；普通费用不要求必须关联差旅。",
                    args_schema=ExpenseClaimInput,
                    coroutine=create_claim,
                    risk=ToolRisk.WRITE,
                ),
                self._tool(
                    name="schedule_expense_reminder",
                    description="设置报销提醒；可以关联差旅，也可以设置普通费用提醒。",
                    args_schema=ExpenseReminderInput,
                    coroutine=schedule_reminder,
                    risk=ToolRisk.WRITE,
                ),
                information_tool,
            ]

        if agent == AgentName.HR:
            async def leave_balance(**kwargs: Any) -> dict[str, Any]:
                outcome = await self._provider.get_leave_balance(
                    context,
                    LeaveBalanceInput.model_validate(kwargs),
                )
                return outcome.model_dump(mode="json")

            async def submit_leave(**kwargs: Any) -> dict[str, Any]:
                outcome = await self._provider.submit_leave_request(
                    context,
                    LeaveRequestInput.model_validate(kwargs),
                )
                return outcome.model_dump(mode="json")

            return [
                policy_tool("hr"),
                self._tool(
                    name="get_leave_balance",
                    description="查询当前登录员工的假期余额。用户身份由运行时注入。",
                    args_schema=LeaveBalanceInput,
                    coroutine=leave_balance,
                    risk=ToolRisk.READ,
                ),
                self._tool(
                    name="submit_leave_request",
                    description="提交请假申请。只有字段完整且用户明确要求提交时才能调用。",
                    args_schema=LeaveRequestInput,
                    coroutine=submit_leave,
                    risk=ToolRisk.WRITE,
                ),
                information_tool,
            ]

        if agent == AgentName.POLICY:
            return [policy_tool("general"), information_tool]

        raise ValueError(f"Unsupported domain agent: {agent}")
