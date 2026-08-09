from functools import lru_cache
from typing import TypeVar

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from enterprise_ai_assistant.core.config import Settings, get_settings

T = TypeVar("T")


@lru_cache
def build_chat_model() -> ChatOpenAI:
    settings = get_settings()
    # ChatOpenAI 可通过 base_url 连接兼容 OpenAI 的 /chat/completions 接口。
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        temperature=0,
        max_retries=3,
        timeout=60,
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
        max_retries=3,
    )
