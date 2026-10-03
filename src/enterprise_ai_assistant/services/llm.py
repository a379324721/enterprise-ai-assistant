from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Literal, TypeVar

import structlog
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from enterprise_ai_assistant.core.config import Settings, get_settings

T = TypeVar("T")

logger = structlog.get_logger()

Role = Literal["supervisor", "domain"]


class ThinkingControl(StrEnum):
    """模型对思考开关（DashScope 的 enable_thinking / thinking_budget 请求参数）的支持。"""

    #: 可开可关，混合思考模型。
    SWITCHABLE = "switchable"
    #: 只能开着：传 enable_thinking=false 直接 400。
    ALWAYS = "always"
    #: 不认或不知道是否认这两个参数，一律不下发，按模型自己的默认行为。
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ModelProfile:
    """一个模型接受哪些请求参数。按模型而不是按配置开关做兼容：同一组配置换个模型，
    能下发的参数就不一样，下发了模型不认的参数是 400，整轮对话失败。"""

    thinking: ThinkingControl


#: 已登记的模型，按模型名前缀匹配，先列的优先，所以具体的写在前面。新接一个模型时在这里
#: 登记，先用 /chat/completions 实测它对 enable_thinking=false 和强制 tool_choice 的反应。
_PROFILES: tuple[tuple[str, ModelProfile], ...] = (
    # 实测 enable_thinking=false 报 "restricted to True"；开着思考时强制 tool_choice 也报 400。
    ("qwen3.8-2.4t-a95b", ModelProfile(ThinkingControl.ALWAYS)),
    ("qwen3", ModelProfile(ThinkingControl.SWITCHABLE)),
    ("deepseek-v4", ModelProfile(ThinkingControl.SWITCHABLE)),
)

_UNREGISTERED = ModelProfile(ThinkingControl.UNKNOWN)


def model_profile(model: str) -> ModelProfile:
    for prefix, profile in _PROFILES:
        if model.startswith(prefix):
            return profile
    return _UNREGISTERED


@dataclass(frozen=True)
class _ChatRequest:
    extra_body: dict[str, object] | None
    #: 是否确定没开思考。DashScope 在思考模式下不支持强制 tool_choice，只有确定关着才能强制。
    thinking_off: bool


def _chat_request(settings: Settings, role: Role) -> _ChatRequest:
    """角色配置是期望，模型说明是上限，两者合起来才是实际下发的参数。"""
    if role == "supervisor":
        wanted, budget = settings.supervisor_enable_thinking, settings.supervisor_thinking_budget
    else:
        wanted, budget = settings.domain_enable_thinking, settings.domain_thinking_budget
    control = model_profile(settings.openai_model).thinking
    if control is ThinkingControl.UNKNOWN:
        return _ChatRequest(extra_body=None, thinking_off=False)
    if control is ThinkingControl.ALWAYS:
        wanted = True
    elif wanted is None:
        return _ChatRequest(extra_body=None, thinking_off=False)
    extra_body: dict[str, object] = {"enable_thinking": wanted}
    if wanted and budget:
        extra_body["thinking_budget"] = budget
    return _ChatRequest(extra_body=extra_body, thinking_off=not wanted)


def can_force_tool_choice(settings: Settings | None = None, role: Role = "supervisor") -> bool:
    """结构化输出能否强制模型调用指定工具；不能时退回 tool_choice=auto，没调工具靠校验反馈重来。"""
    return _chat_request(settings or get_settings(), role).thinking_off


@lru_cache
def build_chat_model(role: Role = "domain") -> ChatOpenAI:
    """按调用方构造模型客户端；两类调用对推理的需要不同，见 Settings 里思考开关的说明。"""
    settings = get_settings()
    if model_profile(settings.openai_model) is _UNREGISTERED:
        logger.warning("chat_model_unregistered", model=settings.openai_model)
    temperature = (
        settings.supervisor_temperature if role == "supervisor" else settings.domain_temperature
    )
    # ChatOpenAI 可通过 base_url 连接兼容 OpenAI 的 /chat/completions 接口。
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        temperature=temperature,
        max_retries=3,
        timeout=60,
        extra_body=_chat_request(settings, role).extra_body,
        # 流式响应默认不带用量，领域 Agent 的调用在 LangSmith 和 LLMUsageTracker 里都记成 0，
        # 会话 token 预算也就只算得到 Supervisor 那一小部分。
        stream_usage=True,
    )


def build_embeddings(settings: Settings | None = None) -> OpenAIEmbeddings:
    config = settings or get_settings()
    return OpenAIEmbeddings(
        api_key=config.openai_api_key,
        base_url=config.openai_base_url,
        model=config.openai_embedding_model,
        # 许多兼容 OpenAI 的服务商（包括 DashScope）仅接受原始字符串作为嵌入输入，
        # 而 LangChain 默认可能发送词元 ID。
        check_embedding_ctx_length=False,
        # 明确请求 JSON 浮点向量，因为并非所有服务都支持 base64。
        model_kwargs={"encoding_format": "float"},
        # DashScope 一次最多接受 10 条，超过直接 400；LangChain 默认 1000 条一批，
        # 制度语料一过 10 条，启动时的语料初始化就整批失败，检索一直落空。
        chunk_size=10,
        max_retries=3,
    )
