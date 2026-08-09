from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """经过校验的运行时配置；敏感信息绝不设置默认值。"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Enterprise AI Assistant"
    app_env: str = "development"
    openai_api_key: SecretStr
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str
    openai_embedding_model: str
    langsmith_tracing: bool = True
    langsmith_api_key: SecretStr | None = None
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_project: str = "enterprise-ai-assistant"
    postgres_dsn: str = "postgresql://enterprise:enterprise@localhost:5432/enterprise_ai"
    redis_url: str = "redis://localhost:6379/0"
    milvus_uri: str = "http://localhost:19530"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
