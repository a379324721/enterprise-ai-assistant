from uuid import UUID

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool

from enterprise_ai_assistant.core.models import AgentName
from enterprise_ai_assistant.repositories.actions import InMemoryActionRepository
from enterprise_ai_assistant.repositories.policies import InMemoryPolicyRepository
from enterprise_ai_assistant.tools import LocalEnterpriseToolProvider, ToolContext, ToolRisk
from enterprise_ai_assistant.tools.registry import MESSAGE_ARG, DomainToolRegistry, RegisteredTool


def registry() -> DomainToolRegistry:
    provider = LocalEnterpriseToolProvider(
        InMemoryActionRepository(), InMemoryPolicyRepository()
    )
    return DomainToolRegistry(provider)


def context() -> ToolContext:
    return ToolContext(
        user_id="u-1",
        conversation_id=UUID("00000000-0000-0000-0000-000000000001"),
        request_id=UUID("00000000-0000-0000-0000-000000000002"),
        task_id="task-1",
    )


def test_each_domain_only_receives_its_allowlisted_tools() -> None:
    expense = registry().for_agent(AgentName.EXPENSE, context())
    names = {item.tool.name for item in expense}

    assert names == {
        "search_expense_policy",
        "create_expense_claim",
        "query_expense_claims",
        "update_expense_claim",
        "revoke_expense_claim",
        "request_information",
        "handoff_task",
    }
    assert "create_travel_application" not in names
    assert next(item for item in expense if item.tool.name == "create_expense_claim").risk == ToolRisk.WRITE
    assert next(item for item in expense if item.tool.name == "update_expense_claim").risk == ToolRisk.WRITE


@pytest.mark.asyncio
async def test_request_information_is_a_terminal_control_tool() -> None:
    information = next(
        item
        for item in registry().for_agent(AgentName.TRAVEL, context())
        if item.tool.name == "request_information"
    )

    result = await information.tool.ainvoke(
        {"missing_fields": ["purpose"], "question": "这次出差的事由是什么？"}
    )

    assert information.terminal is True
    assert result["status"] == "needs_user_input"


@pytest.mark.asyncio
async def test_request_information_restores_newlines_the_model_escaped_twice() -> None:
    """追问原样发给用户，字面的 \\n 会让候选列表挤成一行。"""
    information = next(
        item
        for item in registry().for_agent(AgentName.MEETING, context())
        if item.tool.name == "request_information"
    )

    result = await information.tool.ainvoke(
        {
            "missing_fields": ["会议室"],
            "question": "请问您想预订哪一间？\\n1. SH-301\\n2. SH-302",
        }
    )

    assert result["data"]["question"] == "请问您想预订哪一间？\n1. SH-301\n2. SH-302"


def _write_tools() -> list[RegisteredTool]:
    return [
        item
        for agent in AgentName
        if agent is not AgentName.SUPERVISOR
        for item in registry().for_agent(agent, context())
        if item.risk == ToolRisk.WRITE
    ]


def test_write_tools_give_the_model_a_place_to_speak_outside_the_contract() -> None:
    """模型调工具那回合几乎不写正文，要说的话只能放进参数。它排第一、可以不填，不属于契约。"""
    for item in _write_tools():
        parameters = convert_to_openai_tool(item.tool)["function"]["parameters"]
        assert next(iter(parameters["properties"])) == MESSAGE_ARG, item.tool.name
        assert MESSAGE_ARG not in parameters.get("required", []), item.tool.name
        assert MESSAGE_ARG not in item.schema.model_fields, item.tool.name
    # 契约里只给确认卡用的取值标签不能漏进模型看到的 schema。
    travel = next(item for item in _write_tools() if item.tool.name == "create_travel_application")
    assert "value_labels" not in str(convert_to_openai_tool(travel.tool))


def test_split_message_leaves_business_arguments_for_the_contract() -> None:
    leave = next(item for item in _write_tools() if item.tool.name == "submit_leave_request")
    balance = next(
        item
        for item in registry().for_agent(AgentName.HR, context())
        if item.tool.name == "get_leave_balance"
    )
    arguments = {MESSAGE_ARG: " 年假还剩 8 天。 ", "leave_type": "annual",
                 "start_date": "2026-09-18", "end_date": "2026-09-18"}

    said, business = leave.split_message(arguments)

    assert said == "年假还剩 8 天。"
    assert leave.argument_error(business) is None
    # 模型漏填或乱填不算错，当它没说。
    assert leave.split_message({**business, MESSAGE_ARG: 3}) == ("", business)
    assert leave.split_message(business) == ("", business)
    # 查询工具没有这个参数，原样交给契约校验。
    assert balance.split_message({MESSAGE_ARG: "x"}) == ("", {MESSAGE_ARG: "x"})


def test_every_write_argument_has_a_label_for_the_confirmation_card() -> None:
    """卡片上的字段名只从入参契约读。漏写 title 时卡片会露出英文字段名。"""
    for item in _write_tools():
        for name, info in item.schema.model_fields.items():
            assert info.title, f"{item.tool.name}.{name} 缺少 title"


def test_confirmation_fields_use_contract_labels_and_value_labels() -> None:
    travel = next(item for item in _write_tools() if item.tool.name == "create_travel_application")

    fields = travel.confirmation_fields(
        {
            "origin": "杭州",
            "destination": "上海",
            "start_date": "2026-09-16",
            "trip_type": "one_way",
            "purpose": "培训",
        }
    )

    assert travel.label == "提交差旅申请"
    assert [(field.label, field.value) for field in fields] == [
        ("出发地", "杭州"),
        ("目的地", "上海"),
        ("开始日期", "2026-09-16"),
        ("行程类型", "单程"),
        ("出差事由", "培训"),
    ]


def test_update_card_lists_only_the_fields_being_changed() -> None:
    update = next(item for item in _write_tools() if item.tool.name == "update_leave_request")

    fields = update.confirmation_fields({"reference_id": "LVE-1", "end_date": "2026-09-23"})

    assert [(field.label, field.value) for field in fields] == [
        ("单号", "LVE-1"),
        ("结束日期", "2026-09-23"),
    ]


def test_invalid_arguments_are_reported_before_confirmation() -> None:
    """非法参数要在弹确认卡之前交还模型更正，而不是等用户点了确认才失败。"""
    travel = next(item for item in _write_tools() if item.tool.name == "create_travel_application")

    error = travel.argument_error({"destination": "上海", "start_date": "2026-09-16"})

    assert error is not None and "origin" in error and "purpose" in error
    assert travel.argument_error(
        {"origin": "杭州", "destination": "上海", "start_date": "2026-09-16",
         "trip_type": "one_way", "purpose": "培训"}
    ) is None


def test_room_name_in_place_of_room_id_is_reported_before_confirmation() -> None:
    """模型把草稿里"名称（编号）"的展示写法当 room_id 填进来时，要在确认卡之前拦下。"""
    tools = {item.tool.name: item for item in _write_tools()}
    arguments = {
        "date": "2026-09-17",
        "start_time": "09:00",
        "end_time": "12:00",
        "subject": "开会",
    }

    error = tools["book_meeting_room"].argument_error(
        {**arguments, "room_id": "北京总部201大会议室（BJ-201）"}
    )

    assert error is not None and "BJ-201" in error
    assert tools["book_meeting_room"].argument_error({**arguments, "room_id": "BJ-201"}) is None
    assert tools["update_meeting_booking"].argument_error(
        {"reference_id": "MTG-1", "room_id": "201 大会议室"}
    ) is not None
