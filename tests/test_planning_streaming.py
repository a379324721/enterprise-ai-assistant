"""结构化输出：不走流式，校验不通过时把错误说明交还模型修正。

DashScope 在 response_format 下边流边生成 JSON，模型一跑偏就整段中断，返回
InternalError.Algo.InvalidParameter。400 不在 SDK 的重试范围内，于是 understand
节点抛异常、整轮对话失败。图执行本身是流式的，模型调用会跟着走 astream，所以这
三个节点必须显式关掉流式。直接回复写在理解结果里，也随之一次性给出。
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
    """记录每次结构化阶段绑定 response_format 时，该实例是否已经关掉流式。"""

    def bind(self, **kwargs: Any) -> Any:
        response_format = kwargs["response_format"]
        SEEN.append((response_format["json_schema"]["name"], self.disable_streaming))
        return super().bind(**kwargs)


#: (schema 名, 该次绑定时的 disable_streaming)
SEEN: list[tuple[str, Any]] = []


def test_every_structured_stage_disables_streaming() -> None:
    SEEN.clear()
    LLMPlanningService(SpyModel(messages=iter([])))  # type: ignore[arg-type]

    assert [name for name, _ in SEEN] == [
        "ContextResolution",
        "TaskPlan",
        "MemoryExtraction",
    ]
    assert all(disabled is True for _, disabled in SEEN)


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


async def _resolve(outputs: list[str]) -> Any:
    CALLS.clear()
    model = RecordingModel(messages=iter(AIMessage(content=item) for item in outputs))
    service = LLMPlanningService(model)  # type: ignore[arg-type]
    return await service.resolve_context([{"role": "user", "content": "1 嗯"}])


@pytest.mark.asyncio
async def test_invalid_output_is_sent_back_with_the_validation_error() -> None:
    """实测"1 嗯"这一轮：原样重试时模型两次都写了不存在的前置任务，整轮失败。"""
    result = await _resolve([_resolution(["hr"]), _resolution([])])

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
        await _resolve([_resolution(["hr"])] * 3)

    assert len(CALLS) == 3
