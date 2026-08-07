FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock* README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev || uv sync --no-dev

FROM python:3.12-slim-bookworm
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
COPY --from=builder /app/.venv /app/.venv
COPY src ./src
EXPOSE 8000
CMD ["uvicorn", "enterprise_ai_assistant.main:app", "--host", "0.0.0.0", "--port", "8000"]

