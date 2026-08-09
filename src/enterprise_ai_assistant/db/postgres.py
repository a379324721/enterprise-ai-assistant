import asyncpg

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS workflow_actions (
    idempotency_key TEXT PRIMARY KEY,
    action_type TEXT NOT NULL,
    user_id TEXT NOT NULL,
    payload JSONB NOT NULL,
    result JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_workflow_actions_user_created
    ON workflow_actions(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS assistant_conversations (
    conversation_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_assistant_conversations_user_updated
    ON assistant_conversations(user_id, updated_at DESC);
"""


async def create_pool(dsn: str) -> asyncpg.Pool:
    pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10, command_timeout=30)
    async with pool.acquire() as connection:
        await connection.execute(SCHEMA_SQL)
    return pool
