.PHONY: install dev test lint typecheck eval compose-up
install:
	uv sync
dev:
	uv run uvicorn enterprise_ai_assistant.main:app --reload
test:
	uv run pytest
lint:
	uv run ruff check .
typecheck:
	uv run mypy
eval:
	uv run python -m evals.runner
compose-up:
	docker compose up --build
