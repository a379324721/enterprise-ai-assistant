import secrets
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import BeforeValidator, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: 生产环境允许的访问令牌有效期上限（分钟）。
_PRODUCTION_TOKEN_TTL_CEILING_MINUTES = 1440


class Settings(BaseSettings):
    """经过校验的运行时配置；敏感信息绝不设置默认值。"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Enterprise AI Assistant"
    app_env: str = "development"
    openai_api_key: SecretStr
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str
    openai_embedding_model: str
    # 混合思考模型（DashScope 的 qwen3 系列）默认先推理再作答，不设上限时一次调用常要
    # 上千个推理 token、十几秒。以 enable_thinking / thinking_budget 请求参数下发，
    # OpenAI 官方接口不认这两个参数，接它时把开关留空（不下发）。
    # 两类调用分开配，依据是 qwen3.7-flash 上的评测：
    # - Context Supervisor（理解、规划、记忆抽取）开关思考结果一样，关掉。
    # - 领域 Agent 关掉思考后 guardrail 从 12/12 掉到 9/12，缺结束日期也直接提交差旅
    #   申请；预算 200 仍是 9/12，500 恢复 12/12。
    supervisor_enable_thinking: Annotated[
        bool | None, BeforeValidator(lambda value: None if value == "" else value)
    ] = False
    domain_enable_thinking: Annotated[
        bool | None, BeforeValidator(lambda value: None if value == "" else value)
    ] = True
    # 领域 Agent 推理 token 的上限，0 表示不限。
    domain_thinking_budget: int = Field(default=500, ge=0)
    # 采样温度同样分角色配。原先全局写死 0，但 Qwen3 的模型说明不建议思考模式用贪心解码
    # （温度 0），容易在推理里陷入重复；实测不限预算时领域 Agent 一次决策推理了 81,920
    # token、622 秒才停。Supervisor 关思考、只做结构化输出，保持 0 让理解结果稳定。
    supervisor_temperature: float = Field(default=0.0, ge=0, le=2)
    domain_temperature: float = Field(default=0.0, ge=0, le=2)
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
    # 领域 Agent 读的最近会话条数。比 Supervisor 的窗口小：它只需要把手头的任务放回
    # 语境里，读得越多越容易从不相干的旧事项里捡字段。0 表示不给会话原文。
    domain_context_messages: int = Field(default=10, ge=0)

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
    # 开发环境不设上限，演示时可以签发长到形同永久的令牌；生产由下面的校验
    # 收敛在 24 小时以内。
    access_token_ttl_minutes: int = Field(default=60, ge=1)
    # 本地联调用的签发接口；生产环境的令牌应由企业 SSO 颁发。
    dev_login_enabled: bool = False
    # 演示用的名字注册/登录。没有任何凭据，输入他人的名字即可读到对方的会话与记忆，
    # 因此与 dev_login 一样只允许在开发环境开启。
    demo_login_enabled: bool = False

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

    @model_validator(mode="after")
    def long_lived_tokens_require_development(self) -> "Settings":
        # 访问令牌没有吊销机制，签出去就一直有效到过期；长效令牌泄漏后的暴露窗口
        # 等于它的有效期，因此只在开发环境放行。
        if self.access_token_ttl_minutes > _PRODUCTION_TOKEN_TTL_CEILING_MINUTES:
            if self.app_env != "development":
                raise ValueError(
                    "ACCESS_TOKEN_TTL_MINUTES above "
                    f"{_PRODUCTION_TOKEN_TTL_CEILING_MINUTES} is only allowed "
                    "in the development environment"
                )
        return self

    @model_validator(mode="after")
    def demo_login_requires_development(self) -> "Settings":
        if self.demo_login_enabled and self.app_env != "development":
            raise ValueError("DEMO_LOGIN_ENABLED is only allowed in the development environment")
        return self

    @property
    def signing_key(self) -> str:
        assert self.jwt_secret is not None  # 由 require_jwt_secret_outside_development 保证
        return self.jwt_secret.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
