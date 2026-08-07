.PHONY: install dev test lint compose-up
install:
	uv sync
dev:
	uv run uvicorn enterprise_ai_assistant.main:app --reload
test:
	uv run pytest
lint:
	uv run ruff check .
compose-up:
	docker compose up --build
