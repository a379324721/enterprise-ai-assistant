"""结构化输出：以强制调用的工具下发 schema，不走流式，校验不通过时把错误说明交还模型修正。

DashScope 上的 deepseek-v4-flash 对 response_format 不按 schema 生成，工具参数才按；
流式生成 JSON 时一跑偏就整段中断，整轮对话失败。图执行本身是流式的，模型调用会跟着走
astream，所以这三个节点必须显式关掉流式。直接回复写在理解结果里，也随之一次性给出。
"""

import json
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import ValidationError

from enterprise_ai_assistant.services.planning import LLMPlanningService


class SpyModel(GenericFakeChatModel):
    """记录每次结构化阶段绑定工具时，该实例是否已经关掉流式。"""

    def bind(self, **kwargs: Any) -> Any:
        SEEN.append((kwargs["tools"][0]["function"]["name"], self.disable_streaming))
        BOUND.append(kwargs)
        return super().bind(**kwargs)


#: (schema 名, 该次绑定时的 disable_streaming)
SEEN: list[tuple[str, Any]] = []
#: 每次绑定时传入的参数
BOUND: list[dict[str, Any]] = []


def test_every_structured_stage_disables_streaming() -> None:
    SEEN.clear()
    BOUND.clear()
    LLMPlanningService(SpyModel(messages=iter([])))  # type: ignore[arg-type]

    assert [name for name, _ in SEEN] == [
        "ContextResolution",
        "TaskPlan",
        "MemoryExtraction",
    ]
    assert all(disabled is True for _, disabled in SEEN)


def test_schema_is_a_forced_tool_without_length_limits_and_all_fields_required() -> None:
    """强制调用时参数 schema 带字符串长度限制，deepseek-v4-flash 会一直不返回；
    不在 required 里的字段它一律不写。"""
    BOUND.clear()
    LLMPlanningService(SpyModel(messages=iter([])))  # type: ignore[arg-type]

    context = BOUND[0]
    assert context["tool_choice"] == {
        "type": "function",
        "function": {"name": "ContextResolution"},
    }
    sent = json.dumps(context["tools"][0]["function"]["parameters"])
    assert "maxLength" not in sent and "minLength" not in sent
    # 只删关键字：字段本身和数量限制都还在。
    assert "standalone_request" in sent and "maxItems" in sent
    parameters = context["tools"][0]["function"]["parameters"]
    assert set(parameters["required"]) == set(parameters["properties"])
    outline = parameters["properties"]["tasks"]["items"]
    assert set(outline["required"]) == set(outline["properties"])


#: 每次模型调用收到的消息，按调用顺序。
CALLS: list[list[BaseMessage]] = []


class RecordingModel(GenericFakeChatModel):
    async def _agenerate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        CALLS.append(list(messages))
        return await super()._agenerate(messages, *args, **kwargs)


def _resolution(depends_on: list[str]) -> str:
    return json.dumps(
        {
            "standalone_request": "预订上海分部 301 讨论室，会议主题为培训",
            "intent_summary": "选定会议室并补充主题",
            "requires_task_planning": True,
            "turn_relation": "continue",
            "tasks": [
                {
                    "title": "预订上海会议室",
                    "domain": "meeting",
                    "objective": "预订会议室",
                    "depends_on": depends_on,
                }
            ],
        },
        ensure_ascii=False,
    )


def _tool_reply(output: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "ContextResolution", "args": json.loads(output), "id": "call_1"}],
    )


async def _resolve(outputs: list[AIMessage]) -> Any:
    CALLS.clear()
    model = RecordingModel(messages=iter(outputs))
    service = LLMPlanningService(model)  # type: ignore[arg-type]
    return await service.resolve_context([{"role": "user", "content": "1 嗯"}])


@pytest.mark.asyncio
async def test_invalid_output_is_sent_back_with_the_validation_error() -> None:
    """实测"1 嗯"这一轮：原样重试时模型两次都写了不存在的前置任务，整轮失败。"""
    result = await _resolve([_tool_reply(_resolution(["hr"])), _tool_reply(_resolution([]))])

    assert [task.depends_on for task in result.tasks] == [[]]
    assert len(CALLS) == 2
    # 第二次调用带着模型自己的错误输出和校验说明，它才知道改哪里。
    *_, wrong, feedback = CALLS[1]
    assert wrong.content == _resolution(["hr"])
    assert "depends_on" in str(feedback.content)
    assert "排在它前面的任务没有这些领域" in str(feedback.content)
    assert "Value error" not in str(feedback.content)


@pytest.mark.asyncio
async def test_gives_up_after_the_last_attempt() -> None:
    with pytest.raises(ValidationError):
        await _resolve([_tool_reply(_resolution(["hr"]))] * 3)

    assert len(CALLS) == 3


@pytest.mark.asyncio
async def test_a_text_reply_without_the_tool_call_is_sent_back_too() -> None:
    result = await _resolve([AIMessage(content="好的"), _tool_reply(_resolution([]))])

    assert result.turn_relation == "continue"
    *_, wrong, feedback = CALLS[1]
    assert wrong.content == "好的"
    assert "没有通过校验" in str(feedback.content)
