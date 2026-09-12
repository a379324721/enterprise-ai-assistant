import asyncpg

# COMMENT ON 是幂等的，重复执行只覆盖同名对象的注释，可以和建表语句一起每次启动执行。
# 注释写进数据库而不是只留在这个文件里：DBA 和排障的人通常先看 \d+，不会先读代码。
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

COMMENT ON TABLE workflow_actions IS
    '企业写操作的幂等与审计记录。只表示适配器被调用过，不代表外部企业系统已成功受理；'
    '同时是长期记忆中"近期业务事实"的唯一真相来源，单号不另行复制到 user_memories。';
COMMENT ON COLUMN workflow_actions.idempotency_key IS
    '幂等键，由可信运行时按 用户/会话/请求/任务/工具 派生，同时用作对外可见的业务单号 reference_id。'
    '重复提交同一个键不会产生第二次写入。';
COMMENT ON COLUMN workflow_actions.action_type IS
    '写操作类型：travel_application、expense_claim、expense_reminder、leave_request。'
    '记忆派生时按此值选择摘要字段白名单，未登记的类型不会进入模型上下文。';
COMMENT ON COLUMN workflow_actions.user_id IS
    '发起该操作的员工标识，取自访问令牌的 sub 声明，不接受模型或客户端指定。';
COMMENT ON COLUMN workflow_actions.payload IS
    '经工具契约校验后的业务参数原文。含票据号、请假原因等敏感字段，'
    '派生记忆摘要时必须走白名单，不得整体读出。';
COMMENT ON COLUMN workflow_actions.result IS
    '返回给领域 Agent 的执行结果，含 reference_id 与 status。';
COMMENT ON COLUMN workflow_actions.created_at IS
    '写入时间。与 user_id 组成倒序索引，用于召回该用户最近的单据。';

CREATE TABLE IF NOT EXISTS user_memories (
    id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    source_conversation_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- 同一用户的同类事实只保留一条：换部门、改偏好是覆盖而不是新增，
    -- 否则召回时会同时命中新旧两个值，模型无从判断哪个还有效。
    UNIQUE (user_id, kind, key)
);
CREATE INDEX IF NOT EXISTS idx_user_memories_user_updated
    ON user_memories(user_id, updated_at DESC);

COMMENT ON TABLE user_memories IS
    '跨会话的用户长期画像。只存会话里说出来、而业务系统中没有的稳定属性和偏好；'
    '余额额度等易变数值、金额票据等一次性凭证、请假原因等敏感信息一律不得写入。'
    '按 MEMORY_ENABLED 开关启用，用户可通过 DELETE /api/v1/memories/{id} 自行删除。';
COMMENT ON COLUMN user_memories.id IS
    '记忆标识，也是对外的删除句柄。覆盖写时保持不变，用户先前拿到的删除链接不会失效。';
COMMENT ON COLUMN user_memories.user_id IS
    '记忆归属的员工标识，取自访问令牌的 sub 声明。所有读写路径都必须带此条件，'
    '否则等于开放跨用户读取。';
COMMENT ON COLUMN user_memories.kind IS
    '记忆类别：profile 为稳定身份属性（常驻城市、成本中心、职级），'
    'preference 为可复用的办事偏好（交通方式、默认币种、提醒习惯）。';
COMMENT ON COLUMN user_memories.key IS
    '同类事实的稳定标识，小写下划线形式，如 home_city、cost_center、preferred_transport。'
    '与 user_id、kind 构成唯一键，是覆盖写而非追加的依据。';
COMMENT ON COLUMN user_memories.value IS
    '记忆内容，简短中文陈述，上限 200 字。只记录用户明确说过的内容，不得为推断结果。';
COMMENT ON COLUMN user_memories.source_conversation_id IS
    '写入该值的会话 ID，用于向用户说明记忆来源和事后追溯误写。可为空以兼容历史数据。';
COMMENT ON COLUMN user_memories.created_at IS
    '该 key 首次写入的时间；覆盖写不更新此列。';
COMMENT ON COLUMN user_memories.updated_at IS
    '最后一次覆盖写的时间。召回按此列倒序取前 N 条，越新的画像越可能仍然有效。';
"""


async def create_pool(dsn: str) -> asyncpg.Pool:
    pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10, command_timeout=30)
    async with pool.acquire() as connection:
        await connection.execute(SCHEMA_SQL)
    return pool
