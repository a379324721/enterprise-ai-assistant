"""测试环境的基础配置。

单元测试不依赖真实的模型服务和中间件，这里为必填配置项提供占位值，
使测试在没有 `.env` 的 CI 环境中也能运行。
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_MODEL", "test-model")
os.environ.setdefault("OPENAI_EMBEDDING_MODEL", "test-embedding")
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("APP_ENV", "development")
