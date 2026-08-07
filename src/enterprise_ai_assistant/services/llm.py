from functools import lru_cache
from typing import TypeVar

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from enterprise_ai_assistant.core.config import Settings, get_settings

T = TypeVar("T")


@lru_cache
def build_chat_model() -> ChatOpenAI:
    settings = get_settings()
    # ChatOpenAI supports OpenAI-compatible /chat/completions endpoints via base_url.
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
        # Many OpenAI-compatible providers (including DashScope) accept only
        # raw strings for embeddings, while LangChain otherwise sends token IDs.
        check_embedding_ctx_length=False,
        # Request JSON float vectors explicitly; base64 is not universally supported.
        model_kwargs={"encoding_format": "float"},
        max_retries=3,
    )
