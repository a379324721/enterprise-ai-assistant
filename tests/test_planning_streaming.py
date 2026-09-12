"""结构化输出不得走流式。

DashScope 在 response_format 下边流边生成 JSON，模型一跑偏就整段中断，返回
InternalError.Algo.InvalidParameter。400 不在 SDK 的重试范围内，于是 understand
节点抛异常、整轮对话失败。图执行本身是流式的，模型调用会跟着走 astream，所以这
三个节点必须显式关掉流式——它们的结果都不面向用户，流式本来也没有收益。
"""

from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.runnables import RunnableLambda

from enterprise_ai_assistant.services.planning import LLMPlanningService


class SpyModel(GenericFakeChatModel):
    """记录每次结构化输出调用时，该实例是否已经关掉流式。"""

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        del kwargs
        SEEN.append((getattr(schema, "__name__", str(schema)), self.disable_streaming))
        # __init__ 只拼装 chain，不会真的调用，返回个占位 Runnable 就够。
        return RunnableLambda(lambda _: None)


#: (schema 名, 该次调用时的 disable_streaming)
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


def test_the_user_visible_responder_keeps_streaming() -> None:
    """闲聊回答是逐字送到界面上的，关掉它的流式会让整段答案一次性蹦出来。"""
    model = SpyModel(messages=iter([]))
    service = LLMPlanningService(model)  # type: ignore[arg-type]

    responder_model = service._direct_responder.steps[-1]  # type: ignore[attr-defined]
    assert responder_model is model
    assert model.disable_streaming is False
