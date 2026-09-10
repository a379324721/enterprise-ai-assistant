"""运行指标定义。

LangSmith 记录的是单次调用的轨迹，便于排查个案；这里的指标回答的是另一类
问题：整体吞吐、延迟分布、token 消耗和成本是否在预期内，以及工具成功率
和人工确认通过率的长期走势。
"""

from prometheus_client import CollectorRegistry, Counter, Histogram

# 使用独立注册表，避免多进程/多次导入时与默认注册表上的全局指标互相干扰。
REGISTRY = CollectorRegistry()

HTTP_REQUESTS = Counter(
    "assistant_http_requests_total",
    "HTTP 请求总数",
    labelnames=("method", "endpoint", "status"),
    registry=REGISTRY,
)

HTTP_DURATION = Histogram(
    "assistant_http_request_duration_seconds",
    "HTTP 请求耗时",
    labelnames=("method", "endpoint"),
    # 助手请求包含多次模型调用，尾部分桶需要覆盖到分钟级。
    buckets=(0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 60, 120),
    registry=REGISTRY,
)

LLM_CALLS = Counter(
    "assistant_llm_calls_total",
    "模型调用次数",
    labelnames=("agent", "outcome"),
    registry=REGISTRY,
)

LLM_DURATION = Histogram(
    "assistant_llm_call_duration_seconds",
    "单次模型调用耗时",
    labelnames=("agent",),
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60),
    registry=REGISTRY,
)

LLM_TOKENS = Counter(
    "assistant_llm_tokens_total",
    "模型 token 消耗",
    labelnames=("agent", "kind"),
    registry=REGISTRY,
)

LLM_COST = Counter(
    "assistant_llm_cost_usd_total",
    "按配置单价折算的模型调用成本（美元）",
    labelnames=("model",),
    registry=REGISTRY,
)

TOOL_INVOCATIONS = Counter(
    "assistant_tool_invocations_total",
    "企业工具调用次数",
    labelnames=("tool", "outcome"),
    registry=REGISTRY,
)

TOOL_DURATION = Histogram(
    "assistant_tool_duration_seconds",
    "企业工具调用耗时",
    labelnames=("tool",),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
    registry=REGISTRY,
)

CONFIRMATIONS = Counter(
    "assistant_confirmations_total",
    "高风险操作的人工确认结果",
    labelnames=("decision",),
    registry=REGISTRY,
)

BUDGET_REJECTIONS = Counter(
    "assistant_budget_rejections_total",
    "因会话 token 预算耗尽而被拒绝的请求数",
    registry=REGISTRY,
)
