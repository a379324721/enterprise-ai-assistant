import secrets
from functools import lru_cache

from pydantic import Field, SecretStr, model_validator
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

    jwt_secret: SecretStr | None = None
    jwt_algorithm: str = "HS256"
    jwt_issuer: str = "enterprise-ai-assistant"
    jwt_audience: str = "enterprise-ai-assistant"
    access_token_ttl_minutes: int = Field(default=60, ge=1, le=1440)
    # 本地联调用的签发接口；生产环境的令牌应由企业 SSO 颁发。
    dev_login_enabled: bool = False

    @model_validator(mode="after")
    def require_jwt_secret_outside_development(self) -> "Settings":
        if self.jwt_secret is not None:
            return self
        if self.app_env != "development":
            raise ValueError("JWT_SECRET must be configured outside the development environment")
        # 开发环境允许开箱即用：随进程生成一次性密钥，重启后旧令牌自然失效。
        object.__setattr__(self, "jwt_secret", SecretStr(secrets.token_urlsafe(48)))
        return self

    @model_validator(mode="after")
    def require_strong_hmac_secret(self) -> "Settings":
        # RFC 7518 3.2：HMAC 密钥长度不得短于哈希输出长度，否则签名强度会被削弱。
        if self.jwt_algorithm.startswith("HS") and len(self.signing_key.encode()) < 32:
            raise ValueError("JWT_SECRET must be at least 32 bytes for HS* algorithms")
        return self

    @model_validator(mode="after")
    def dev_login_requires_development(self) -> "Settings":
        if self.dev_login_enabled and self.app_env != "development":
            raise ValueError("DEV_LOGIN_ENABLED is only allowed in the development environment")
        return self

    @property
    def signing_key(self) -> str:
        assert self.jwt_secret is not None  # 由 require_jwt_secret_outside_development 保证
        return self.jwt_secret.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
