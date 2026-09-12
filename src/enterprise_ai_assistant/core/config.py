import secrets
from functools import lru_cache
from typing import Literal

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

    # 模型单价用于把 token 折算成成本指标；默认 0 表示不统计金额。
    llm_input_cost_per_1k_usd: float = Field(default=0.0, ge=0)
    llm_output_cost_per_1k_usd: float = Field(default=0.0, ge=0)
    # 单个会话累计 token 上限，0 表示不限制。防止异常会话无上限消耗额度。
    conversation_token_budget: int = Field(default=0, ge=0)
    conversation_budget_ttl_hours: int = Field(default=168, ge=1)
    # Context Supervisor 每轮都要读完整会话，历史无上限增长会让单轮 token 线性上涨。
    # 只把最近 N 条消息原样送进 prompt，更早的轮次降级成一句话摘要，0 表示不截断。
    context_window_messages: int = Field(default=12, ge=0)
    # 保留的历史摘要条数（每轮一条），决定超出消息窗口后还能回溯多少轮指代。
    context_digest_turns: int = Field(default=20, ge=0)

    # 跨会话长期记忆。默认关闭：记错一条画像会污染该用户后续所有会话，
    # 需要先有删除入口和灰度范围再打开。
    memory_enabled: bool = False
    # 单轮注入领域子图的画像条数上限，防止记忆增长把每轮 prompt 撑大。
    memory_recall_limit: int = Field(default=20, ge=0)
    # 从 workflow_actions 派生的近期单据条数；只用于关联 travel_reference 这类字段。
    memory_recent_action_limit: int = Field(default=5, ge=0)

    # SSE 订阅者断开时后台执行的默认处置：continue 表示继续跑完并落检查点，
    # 客户端重连后仍能拿到结果。
    run_on_disconnect: Literal["cancel", "continue"] = "continue"
    # 每个运行保留的事件条数，供断线重连回放；超出窗口的游标会收到 gap 事件。
    run_event_buffer_size: int = Field(default=512, ge=16)
    # 运行结束后事件缓冲的保留时长，给断线客户端留出回来取终态的时间。
    run_retention_seconds: float = Field(default=300.0, gt=0)
    # SSE 空闲心跳间隔，防止反向代理掐掉长时间没有输出的连接。
    sse_heartbeat_seconds: float = Field(default=15.0, gt=0)

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
