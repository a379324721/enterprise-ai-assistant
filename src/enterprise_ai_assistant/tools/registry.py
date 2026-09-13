from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ValidationError

from enterprise_ai_assistant.core.models import AgentName, ConfirmationField
from enterprise_ai_assistant.tools.contracts import (
    BusinessToolOutcome,
    EnterpriseToolProvider,
    ExpenseClaimInput,
    ExpenseClaimUpdateInput,
    InformationRequestInput,
    LeaveBalanceInput,
    LeaveRequestInput,
    LeaveRequestUpdateInput,
    MeetingBookingUpdateInput,
    MeetingRoomBookingInput,
    MeetingRoomSearchInput,
    PolicyQueryInput,
    PolicySearchInput,
    SubmissionQueryInput,
    SubmissionRevokeInput,
    ToolContext,
    ToolRisk,
    TravelApplicationInput,
    TravelApplicationUpdateInput,
)

#: 面向用户的能力自述，Supervisor 直接回答"你能干什么"和 Planner 判断能力边界时共用。
#:
#: 工具的 description 是写给模型看的调用条件，不适合直接念给用户，所以这里单独维护
#: 一份用户视角的说明。它必须只描述下面 for_agent 真的装配了工具的能力——模型没有
#: 清单时会自行编造（例如"代替审批"，本系统并无此工具）。
#: 新增领域时这里必须同步，测试会检查覆盖完整。
CAPABILITY_SUMMARY: dict[AgentName, str] = {
    AgentName.TRAVEL: "查询差旅制度、创建差旅申请、查询/修改/撤销已提交的差旅申请",
    AgentName.EXPENSE: "查询报销制度、提交费用报销、查询/修改/撤销已提交的报销单",
    AgentName.HR: "查询人事制度、查询假期余额、提交请假申请、查询/修改/撤销已提交的请假申请",
    AgentName.MEETING: "查询会议室制度、查询空闲会议室、预订会议室、查询/修改/取消已有的会议室预订",
    AgentName.POLICY: "查询其他企业通用制度",
}

#: 工具在界面执行步骤里的中文名。和风险等级一样由服务端声明，前端不维护工具名映射。
TOOL_LABELS: dict[str, str] = {
    "request_information": "向你确认缺失信息",
    "search_travel_policy": "检索差旅制度",
    "search_expense_policy": "检索报销制度",
    "search_hr_policy": "检索人事制度",
    "search_meeting_policy": "检索会议室制度",
    "search_general_policy": "检索通用制度",
    "create_travel_application": "提交差旅申请",
    "create_expense_claim": "提交费用报销",
    "find_available_rooms": "查询空闲会议室",
    "book_meeting_room": "预订会议室",
    "get_leave_balance": "查询假期余额",
    "submit_leave_request": "提交请假申请",
    "query_travel_applications": "查询差旅申请",
    "query_expense_claims": "查询报销单",
    "query_leave_requests": "查询请假申请",
    "query_meeting_bookings": "查询会议室预订",
    "update_travel_application": "修改差旅申请",
    "update_expense_claim": "修改报销单",
    "update_leave_request": "修改请假申请",
    "update_meeting_booking": "修改会议室预订",
    "revoke_travel_application": "撤销差旅申请",
    "revoke_expense_claim": "撤销报销单",
    "revoke_leave_request": "撤销请假申请",
    "revoke_meeting_booking": "取消会议室预订",
}


@dataclass(frozen=True)
class RegisteredTool:
    tool: BaseTool
    risk: ToolRisk
    terminal: bool = False

    @property
    def label(self) -> str:
        return TOOL_LABELS[self.tool.name]

    @property
    def schema(self) -> type[BaseModel]:
        schema = self.tool.args_schema
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise TypeError(f"tool {self.tool.name!r} must declare a pydantic args_schema")
        return schema

    def argument_error(self, arguments: Mapping[str, Any]) -> str | None:
        """按入参契约校验模型给出的参数；合法时返回 None，否则返回交给模型更正的说明。"""
        try:
            self.schema.model_validate(dict(arguments))
        except ValidationError as exc:
            problems = "；".join(
                f"{'.'.join(str(part) for part in item['loc']) or '参数'}：{item['msg']}"
                for item in exc.errors()
            )
            return f"工具 {self.tool.name} 的参数不合法：{problems}"
        return None

    def confirmation_fields(self, arguments: Mapping[str, Any]) -> list[ConfirmationField]:
        """把已通过校验的参数渲染成确认卡片的逐项字段。

        标签和取值标签都读入参契约里的声明（Field 的 title 与 json_schema_extra），
        界面和后端不各自维护一份字段中文名。只列出模型实际给出的字段：修改类工具
        只传要改的字段，卡片上就只该出现这几项。
        """
        schema = self.schema
        values = schema.model_validate(dict(arguments)).model_dump(mode="json")
        fields: list[ConfirmationField] = []
        for name, info in schema.model_fields.items():
            if name not in arguments or values[name] is None:
                continue
            extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
            value_labels = extra.get("value_labels", {})
            value = values[name]
            if isinstance(value, list):
                shown = "、".join(str(item) for item in value)
            elif isinstance(value_labels, dict):
                shown = str(value_labels.get(value, value))
            else:
                shown = str(value)
            fields.append(ConfirmationField(name=name, label=info.title or name, value=shown))
        return fields


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

        def submission_tool(name: str, action_type: str, noun: str) -> RegisteredTool:
            async def query(**kwargs: Any) -> dict[str, Any]:
                outcome = await self._provider.query_submissions(
                    context,
                    action_type,
                    SubmissionQueryInput.model_validate(kwargs),
                )
                return outcome.model_dump(mode="json")

            return self._tool(
                name=name,
                description=(
                    f"查询当前登录员工提交过的{noun}及其状态。用户问某张单据批了没有、"
                    "进行到哪一步、或想看自己提交过哪些时调用；回答状态前必须先调用。"
                    "status 取值：pending_approval 审批中，approved 已通过，"
                    "confirmed 已生效，revoked 已撤销。fields 是单据的完整字段，修改前据此核对原值。"
                    "found 为 false 表示该单号不存在或不属于当前用户。"
                ),
                args_schema=SubmissionQueryInput,
                coroutine=query,
                risk=ToolRisk.READ,
            )

        def revoke_tool(name: str, action_type: str, noun: str) -> RegisteredTool:
            async def revoke(**kwargs: Any) -> dict[str, Any]:
                outcome = await self._provider.revoke_submission(
                    context, action_type, SubmissionRevokeInput.model_validate(kwargs)
                )
                return outcome.model_dump(mode="json")

            return self._tool(
                name=name,
                description=(
                    f"撤销当前登录员工已提交的{noun}，撤销后不可恢复、也不能再修改。"
                    "只有用户明确要求撤销且指明了是哪一张时才能调用；不确定是哪张时先调查询工具，"
                    "把候选单据列给用户确认。"
                ),
                args_schema=SubmissionRevokeInput,
                coroutine=revoke,
                risk=ToolRisk.WRITE,
            )

        def update_tool(
            name: str,
            noun: str,
            schema: type[Any],
            method: Callable[[ToolContext, Any], Awaitable[BusinessToolOutcome]],
            extra: str = "",
        ) -> RegisteredTool:
            async def update(**kwargs: Any) -> dict[str, Any]:
                outcome = await method(context, schema.model_validate(kwargs))
                return outcome.model_dump(mode="json")

            return self._tool(
                name=name,
                description=(
                    f"修改当前登录员工已提交的{noun}。reference_id 必填，只传要改的字段，"
                    "没传的沿用原值；不知道单号或原值时先调查询工具。"
                    "只有用户明确要求修改、且改成什么已经说清楚时才能调用。"
                    f"修改后需要重新审批。{extra}"
                ),
                args_schema=schema,
                coroutine=update,
                risk=ToolRisk.WRITE,
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
                submission_tool("query_travel_applications", "travel_application", "差旅申请"),
                update_tool(
                    "update_travel_application",
                    "差旅申请",
                    TravelApplicationUpdateInput,
                    self._provider.update_travel_application,
                ),
                revoke_tool("revoke_travel_application", "travel_application", "差旅申请"),
                information_tool,
            ]

        if agent == AgentName.EXPENSE:
            async def create_claim(**kwargs: Any) -> dict[str, Any]:
                outcome = await self._provider.create_expense_claim(
                    context,
                    ExpenseClaimInput.model_validate(kwargs),
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
                submission_tool("query_expense_claims", "expense_claim", "报销单"),
                update_tool(
                    "update_expense_claim",
                    "报销单",
                    ExpenseClaimUpdateInput,
                    self._provider.update_expense_claim,
                ),
                revoke_tool("revoke_expense_claim", "expense_claim", "报销单"),
                information_tool,
            ]

        if agent == AgentName.MEETING:
            async def find_rooms(**kwargs: Any) -> dict[str, Any]:
                outcome = await self._provider.find_available_rooms(
                    context,
                    MeetingRoomSearchInput.model_validate(kwargs),
                )
                return outcome.model_dump(mode="json")

            async def book_room(**kwargs: Any) -> dict[str, Any]:
                outcome = await self._provider.book_meeting_room(
                    context,
                    MeetingRoomBookingInput.model_validate(kwargs),
                )
                return outcome.model_dump(mode="json")

            return [
                policy_tool("meeting"),
                self._tool(
                    name="find_available_rooms",
                    description=(
                        "按地点、日期和时段查询空闲会议室；预订前必须先查。"
                        "返回空列表表示该时段确实没有符合条件的会议室，不是调用失败。"
                    ),
                    args_schema=MeetingRoomSearchInput,
                    coroutine=find_rooms,
                    risk=ToolRisk.READ,
                ),
                self._tool(
                    name="book_meeting_room",
                    description="预订会议室。只能预订上一步查询结果中出现过的 room_id。",
                    args_schema=MeetingRoomBookingInput,
                    coroutine=book_room,
                    risk=ToolRisk.WRITE,
                ),
                submission_tool("query_meeting_bookings", "meeting_booking", "会议室预订"),
                update_tool(
                    "update_meeting_booking",
                    "会议室预订",
                    MeetingBookingUpdateInput,
                    self._provider.update_meeting_booking,
                    "换房间或改时段前先查空闲。",
                ),
                revoke_tool("revoke_meeting_booking", "meeting_booking", "会议室预订"),
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
                submission_tool("query_leave_requests", "leave_request", "请假申请"),
                update_tool(
                    "update_leave_request",
                    "请假申请",
                    LeaveRequestUpdateInput,
                    self._provider.update_leave_request,
                ),
                revoke_tool("revoke_leave_request", "leave_request", "请假申请"),
                information_tool,
            ]

        if agent == AgentName.POLICY:
            return [policy_tool("general"), information_tool]

        raise ValueError(f"Unsupported domain agent: {agent}")
