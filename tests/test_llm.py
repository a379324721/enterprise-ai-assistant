"""请求参数按模型支持什么下发：下发模型不认的参数是 400，整轮对话失败。"""

from typing import Any

from enterprise_ai_assistant.core.config import Settings
from enterprise_ai_assistant.services.llm import _chat_request, can_force_tool_choice


def _settings(model: str, **overrides: Any) -> Settings:
    # 显式写死思考相关的每一项，未传的字段会回落到 .env。
    defaults: dict[str, Any] = {
        "openai_api_key": "test-key",
        "openai_model": model,
        "openai_embedding_model": "test-embedding",
        "supervisor_enable_thinking": False,
        "domain_enable_thinking": True,
        "supervisor_thinking_budget": 1000,
        "domain_thinking_budget": 2000,
    }
    return Settings(**{**defaults, **overrides})


def test_switchable_model_follows_the_role_settings() -> None:
    settings = _settings("deepseek-v4-pro-0813")

    assert _chat_request(settings, "supervisor").extra_body == {"enable_thinking": False}
    assert _chat_request(settings, "domain").extra_body == {
        "enable_thinking": True,
        "thinking_budget": 2000,
    }
    assert can_force_tool_choice(settings) is True


def test_always_thinking_model_cannot_turn_it_off() -> None:
    """qwen3.8-2.4t-a95b：enable_thinking=false 报 400，开思考时强制 tool_choice 也报 400。"""
    settings = _settings("qwen3.8-2.4t-a95b")

    assert _chat_request(settings, "supervisor").extra_body == {
        "enable_thinking": True,
        "thinking_budget": 1000,
    }
    assert can_force_tool_choice(settings) is False


def test_unregistered_model_gets_no_thinking_parameters_and_no_forced_tool() -> None:
    settings = _settings("some-new-model")

    assert _chat_request(settings, "supervisor").extra_body is None
    assert _chat_request(settings, "domain").extra_body is None
    assert can_force_tool_choice(settings) is False


def test_empty_switch_sends_nothing_and_does_not_force() -> None:
    settings = _settings("qwen3.7-max", supervisor_enable_thinking=None)

    assert _chat_request(settings, "supervisor").extra_body is None
    assert can_force_tool_choice(settings) is False
