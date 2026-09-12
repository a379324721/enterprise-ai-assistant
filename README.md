# Enterprise AI Assistant

面向企业内部事务的生产级 Multi-Agent。后端使用 FastAPI + LangGraph，模型通过 OpenAI-compatible Chat Completions API 接入；PostgreSQL 保存工作流 checkpoint 与业务写操作，Redis 缓存制度查询，Milvus 存储制度向量，LangSmith 记录 Agent、规划器和工具调用 trace。前端提供任务进度与 Human-in-the-loop 确认界面。

## 架构

系统把会话理解、任务规划、领域执行和企业工具调用分成独立边界：

```mermaid
flowchart TB
    U[用户输入 + 会话历史] --> S[Context Supervisor\n指代消解/输入改写]
    S --> P[Task Planner\n领域任务 DAG]
    P --> D[Task Scheduler]
    D --> SG
    subgraph SG[Domain Task Subgraph]
        A[领域 Agent\nLLM 决策] --> M{缺少信息?}
        M -- 是 --> Q[生成澄清问题]
        M -- 否 --> T[选择白名单工具]
        T --> R{写工具?}
        R -- 是 --> I[LangGraph Interrupt\n逐工具确认]
        I --> X[执行工具]
        R -- 否 --> X
        X --> A
        A --> O[LLM 原生流式回答]
    end
    SG --> G[Result Aggregator]
    G --> D
```

- Context Supervisor 阅读完整对话，只负责消解指代、分析整体意图并生成独立请求；不抽取领域字段。
- 读原始 `messages` 的只有从对话中提取信息的节点：`understand`（提意图）和 `remember`（提记忆）。
  执行链路上的 `plan`、`select_task`、领域子图和闲聊节点一律只消费 `ContextResolution`。
- Planner 只生成任务领域、目标、成功标准和依赖；不选择工具、不生成参数、不判断风险。
- Travel、Expense、HR、Policy Agent 分别拥有独立 Prompt 和最小工具白名单。
- 每项计划任务调用一次通用领域子图；子图根据任务领域装配 Prompt 和最小工具集。
- 领域子图在受限循环中自行判断字段、请求补充信息、选择工具并解释工具结果。
- 父图与领域子图只通过 `DomainTaskRequest` 和 `DomainTaskResult` 通信；模型消息、重试计数和待执行工具保留在子图内部。
- 工具风险由服务端注册表声明。所有写工具在执行前保存 checkpoint 并要求用户确认。
- 用户身份、会话 ID、请求 ID 和幂等键来自可信运行时，不作为模型参数暴露。

## State 设计

核心定义位于 `src/enterprise_ai_assistant/graph/state.py`：

| 字段 | 用途 |
|---|---|
| `messages` | 持久化对话记录，供 Context Supervisor 消解上下文 |
| `user_goal` | 当前轮次已改写的独立请求 |
| `tasks` | 带领域、目标、依赖和状态的任务 DAG |
| `artifacts` | 按 task ID 保存的结构化工具产物 |
| `tool_results` | 可审计的工具执行结果 |
| `current_agent` | 当前能力提供者 |
| `active_task_id` | 当前执行任务 |
| `domain_request` | 父图发给领域子图的任务、依赖产物与可信上下文 |
| `domain_result` | 领域子图返回的状态、回答、产物和工具审计结果 |
| `memories` | 本轮召回的全量用户画像，经 `relevant_memory_keys` 过滤后才下发 |
| `recent_actions` | 从 `workflow_actions` 派生的近期单据摘要 |

`domain_messages`、工具决策和确认状态属于子图私有状态。领域 Agent 不共享内部消息，
后续任务只接收依赖任务在 `artifacts` 中留下的结构化结果。人工确认通过 LangGraph
interrupt payload 暴露，API 不依赖子图内部节点名。

## 长期记忆

会话内的连续性由检查点和 `messages` 保证；跨会话的连续性由长期记忆提供，
默认关闭（`MEMORY_ENABLED=false`）。记忆分两层，来源不同：

| 层 | 存储 | 内容 | 可写 |
|---|---|---|---|
| `profile` / `preference` | `user_memories` 表 | 常驻城市、成本中心、职级、交通与币种偏好 | 抽取写入，用户可删 |
| 近期业务事实 | 从 `workflow_actions` 派生 | 最近的差旅、报销、请假单号与关键字段 | 只读投影 |

单号不复制进记忆表：`workflow_actions` 已经是写操作的幂等与审计记录，
复制一份会在单据作废或改期后留下无法失效的旧值。派生摘要走字段白名单
（见 `repositories/memories.py` 的 `_ACTION_SUMMARY_FIELDS`），请假原因、票据号、
备注正文这类一次性或敏感内容不进摘要。

图在每轮开头经 `recall` 节点召回，结尾统一经 `remember` 节点收口——所有终止分支
都汇到同一个节点，新增结束路径不会漏掉写入。等待用户补充输入的轮次不抽取，
避免把没谈定的半成品写进画像。仓储故障时两个节点都降级为空操作，本轮行为
退化成没有记忆的旧路径而不是失败。

记忆经理解阶段筛选后才下发。`recall` 把**全部记忆的 key 清单**（只有 key、没有值）交给
Context Supervisor，后者在改写请求的同时把相关的 key 写进 `ContextResolution.relevant_memory_keys`，
`select_task` 和闲聊节点按这个列表过滤后才使用。这样做有两个原因：全量下发的成本是
`记忆条数 × 任务数`，且无关档案会成为领域模型的噪音。Supervisor 只看 key 不看 value，
做的是相关性筛选而非字段判断，`不得抽取或补写领域字段` 的边界仍然成立。

领域 Agent 收到的 `user_memory` 只是建议默认值，调用 `request_information` 时必须
写出建议值及其来源交用户确认，不得仅凭档案补全字段后直接调用写工具；余额、额度和
制度条款一律以实时查询工具的结果为准。所有写工具照旧逐个走人工确认。

闲聊节点（`direct_respond`）也会收到筛选后的档案和最近提交过的单据，用于让回答贴合
这位用户、提示待办。但 `workflow_actions` 只记录写操作被调用过，**不含审批结果**，
所以 prompt 明确禁止声称任何单据已受理、已通过或进行到哪个环节。系统也没有查询
审批状态的工具，用户问状态时只能如实说明查不到，而不是编一个进度。评测集的
`small_talk` 套件用短语黑名单守这条线。

管理接口：

```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/memories
curl -X DELETE -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/memories/<memory_id>
# 提交过的单据；界面右栏的"我的单据"用的就是它
curl -H "Authorization: Bearer $TOKEN" "http://localhost:8000/api/v1/actions?limit=10"
```

`/actions` 刻意不受 `MEMORY_ENABLED` 门控：单据来自 `workflow_actions`，是用户自己
办过的事，不因为关掉长期记忆就该从界面上消失。它返回的是 `result` 里存下的对外短
单号，不是内部幂等键。

身份一律取自令牌，不接受调用方指定 `user_id`。删除入口是必需的：
一条记错的画像会持续影响该用户之后的每一轮对话。

## 企业工具

工具统一通过 `EnterpriseToolProvider` 接口接入，当前实现和后续 OA、财务、HR API 适配器遵循同一契约。替换后端实现不需要修改 Agent 或工作流。

| Agent | 可用工具 |
|---|---|
| Travel | 差旅制度查询、创建差旅申请 |
| Expense | 报销制度查询、创建报销单 |
| HR | 人事制度查询、假期余额查询、提交请假申请 |
| Meeting | 会议室制度查询、空闲会议室查询、预订会议室 |
| Policy | 通用制度查询 |

所有领域都可调用 `request_information` 暂停当前任务并向用户询问缺失字段。工具输入使用 Pydantic 严格校验，未知字段会被拒绝。

## 示例流程

输入：`下周三到周五去上海出差，帮我申请，顺便订个周四上午的会议室`

1. Supervisor 结合历史会话把输入改写为独立请求，不抽取差旅字段。
2. Planner 生成 Travel 任务和依赖它的 Meeting 任务。
3. Travel Agent 自行识别字段；缺失时调用 `request_information`，完整时提出创建差旅工具调用。
4. 子图冻结精确工具参数并触发 interrupt；确认请求必须携带对应的 `confirmation_id`，通过后再使用“会话 + 请求 + 任务 + 工具”幂等键执行。
5. 差旅结果写入 `artifacts`，其中的目的地与日期作为 `dependency_results` 交给 Meeting Agent——
   用户没有再说一遍「上海」，地点是从上游任务的产物推断出来的。
6. Meeting Agent 先查空闲会议室；查不到符合条件的就如实说明并建议改期，查到才提出预订调用并再次请求确认。

所有写工具都需要确认；制度和余额读取不需要确认。

## 目录

```text
src/enterprise_ai_assistant/
├── agents/          # Supervisor 与领域 Agent
├── api/             # FastAPI 路由和 DTO
├── core/            # 配置、日志、领域类型
├── db/              # PostgreSQL schema/bootstrap
├── graph/           # 父调度图、领域任务子图与各自 State
├── repositories/    # PostgreSQL、Redis、Milvus 边界
└── services/        # LLM、理解/规划、能力注册表
frontend/            # React + TypeScript + Vite
tests/               # 无真实外部依赖的工作流测试
```

## 运行

需要 Docker、Docker Compose、[uv](https://docs.astral.sh/uv/) 和 Node.js 18+。

```bash
cp .env.example .env
# 编辑 .env，至少设置 OPENAI_API_KEY 与 LANGSMITH_API_KEY
docker compose up --build
```

- 前端：<http://localhost:5173>
- API 文档：<http://localhost:8000/docs>
- 健康检查：<http://localhost:8000/api/v1/health>

### 本地开发（推荐）

本地开发时只在 Docker 中运行基础设施

终端 1——启动 PostgreSQL、Redis、etcd、MinIO 和 Milvus：

```bash
docker compose up -d postgres redis etcd minio milvus
```

终端 2——启动 FastAPI 后端：

```bash
uv sync
uv run uvicorn enterprise_ai_assistant.main:app --reload
```

终端 3——启动前端：

```bash
cd frontend
npm install    # 首次运行或依赖变化时执行
npm run dev
```

开发地址：

- 前端：<http://localhost:5173>
- API 文档：<http://localhost:8000/docs>
- 健康检查：<http://localhost:8000/api/v1/health>

Vite 支持前端热更新；Uvicorn 使用 `--reload` 后支持后端代码自动重载。日常启动时，如果依赖没有变化，可以跳过 `uv sync` 和 `npm install`。

所有配置均来自环境变量，完整示例见 `.env.example`。`OPENAI_BASE_URL` 可指向任何实现兼容 `/chat/completions` 与 `/embeddings` 的服务。官方 OpenAI 文档说明 Chat Completions 使用 `model` 与 `messages`，API Key 应从服务端环境变量安全加载；本项目遵循这一边界。

## API

所有会话接口要求 `Authorization: Bearer <token>` 请求头。用户身份取自令牌的 `sub`
声明，客户端无法自行声明身份。生产部署由企业 SSO 颁发令牌；本地联调可在
`APP_ENV=development` 且 `DEV_LOGIN_ENABLED=true` 时换取测试令牌：

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/dev-token \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"u-1001"}' | python -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')
```

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"message":"查询差旅住宿标准"}'
```

### 演示登录

给人演示时可以让对方用自己的名字进来，各自拥有独立的会话和记忆。需要
`APP_ENV=development` 且 `DEMO_LOGIN_ENABLED=true`：

```bash
curl -X POST http://localhost:8000/api/v1/auth/login -H 'Content-Type: application/json' -d '{"name":"王宁"}'
```

只有这一个入口，没有单独的注册动作：名字没见过就顺手建一个，响应的 `created`
字段区分新建与回归。在没有凭据的前提下注册和登录本来就是同一件事，把名字占用做成
冲突错误挡不住任何冒用，只会让演示现场打错字的人多点一次按钮。

响应给出令牌和 `conversation_id`。会话 ID 由 `user_id` 经 uuid5 确定性派生，
所以**一个演示用户固定一个会话**：换设备、清了浏览器存储都会回到同一个 thread，
服务端不需要额外存映射，前端也不需要会话列表。

令牌里带 `name` 声明，助手据此称呼用户。姓名不进长期记忆——用户不会对助手自报
姓名，抽取阶段看不到它；而称呼是"永远相关"的信息，交给按相关性筛选的记忆链路
会在问候这类输入上被筛掉。会话归属和幂等键一律只认 `user_id`。

**这不是身份系统**：没有凭据，任何人输入他人的名字即可接管该身份。配置校验会拒绝
在非开发环境开启，接口关闭时返回 404 而不是 403，不暴露其存在。正式部署由企业
SSO 取代。

### 会话历史

```bash
curl "http://localhost:8000/api/v1/conversations/$CID/messages?limit=20" -H "Authorization: Bearer $TOKEN"
curl "http://localhost:8000/api/v1/conversations/$CID/messages?limit=20&before=12" -H "Authorization: Bearer $TOKEN"
```

演示用户长期停在同一个会话里，首屏铺开整段历史会越用越慢，因此默认只返回最近一页，
用 `before` 游标向前翻，`has_more` 指示是否还有更早的消息。序号在过滤掉工具消息和
空占位之后分配，游标不会指向界面上不存在的位置。注意分页只减少传输量：检查点仍然
整体反序列化，要压这部分成本得在状态层面回收历史。

### SSE 流式响应

前端默认调用 `POST /api/v1/chat/stream`，通过 streaming fetch 消费 SSE。服务端会推送以下事件：

| 事件 | 内容 |
|---|---|
| `metadata` | 会话 ID 与本次运行 ID |
| `progress` | Supervisor、Planner、领域循环和工具执行进度 |
| `answer_start` | 一个领域回答开始，包含 message/agent/task ID |
| `token` | 带 `user-visible` 标签的模型原生内容增量 |
| `done` | 完整任务、artifacts、工具结果和确认状态 |
| `error` | 流建立后的执行错误 |
| `gap` | 重连游标已超出事件缓冲窗口，需改用会话快照恢复 |

每条事件都带 `id:` 序号，另有 `: heartbeat` 注释帧在空闲时保活。

命令行验证：

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"message":"查询差旅住宿标准"}'
```

高风险操作确认后的剩余任务通过
`POST /api/v1/conversations/{conversation_id}/confirm/stream` 继续流式执行。Nginx 已关闭该路径的代理缓冲。

领域任务缺字段时的追问走同一条恢复路径：
`POST /api/v1/conversations/{conversation_id}/input/stream`，请求体是 `{input_id, text}`，
`input_id` 来自 `pending_input`。它**不是**新的一轮——把回答当成新消息发出去会重新规划，
同一请求里尚未执行的任务就被新计划覆盖掉了，所以挂在追问上的会话拒绝普通 chat 请求。
补充内容会作为一条用户发言进入会话历史。

### 断线恢复

图执行跑在后台运行里，SSE 连接只是订阅者：**关闭标签页或网络抖动不会中断执行**，
工作流会继续跑完并落检查点。客户端重连访问：

```bash
curl -N http://localhost:8000/api/v1/conversations/<conversation-id>/stream \
  -H "Authorization: Bearer $TOKEN" -H 'Last-Event-ID: 42'
```

- 带 `Last-Event-ID` 只补发缺失的增量；游标超出缓冲窗口则收到 `gap` 事件，改用
  `GET /api/v1/conversations/{conversation_id}` 重建状态。
- 运行结束后事件仍保留 `RUN_RETENTION_SECONDS`，断线的客户端回来仍能取到 `done`。
- 重连是只读旁观：断开不会影响后台执行。
- 执行期间读会话返回 `status=running` 与 `run_id`，不会把半成品检查点误报为失败。
- 同一会话同时只允许一次执行，并发提交返回 409；重复提交同一个 `request_id`
  视为客户端重试，复用同一次运行而不会重跑。
- 创建流时可传 `on_disconnect: "cancel"` 要求断线即终止执行，默认由
  `RUN_ON_DISCONNECT` 决定（默认 `continue`）。

单进程部署使用内存事件桥。多副本部署需要把 `StreamBridge` 换成跨进程实现
（如 Redis Streams），或按 `conversation_id` 做粘性路由，否则重连可能落到没有该
运行的实例上。

确认高风险操作：

```bash
curl -X POST http://localhost:8000/api/v1/conversations/<conversation-id>/confirm \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"confirmation_id":"<pending-confirmation-id>","approved":true}'
```

## 可观测性

`GET /api/v1/metrics` 暴露 Prometheus 指标，与健康检查一样供基础设施抓取，不要求业务令牌：

| 指标 | 用途 |
|---|---|
| `assistant_http_requests_total` / `assistant_http_request_duration_seconds` | 请求量与延迟分布，标签使用路由模板避免会话 ID 造成标签基数爆炸 |
| `assistant_llm_calls_total` / `assistant_llm_call_duration_seconds` | 按 agent 维度的模型调用次数与耗时 |
| `assistant_llm_tokens_total` / `assistant_llm_cost_usd_total` | token 消耗与按配置单价折算的成本 |
| `assistant_tool_invocations_total` / `assistant_tool_duration_seconds` | 企业工具成功率与耗时 |
| `assistant_confirmations_total` | 高风险操作的人工确认通过率 |
| `assistant_budget_rejections_total` | 因会话用量超限被拒绝的请求数 |

每次请求结束会输出一条 `llm_usage` 结构化日志，包含调用次数、输入/输出 token 和折算成本。

设置 `CONVERSATION_TOKEN_BUDGET` 后，单个会话累计 token 达到上限时新请求返回 429。
计数存放在 Redis，Redis 不可用时放行而不是阻断业务。

## 测试与质量

```bash
uv run pytest
uv run ruff check .
uv run mypy
uv run python -m evals.runner   # 回归评测，需要可用的模型服务
cd frontend && npm install && npm run build
```

测试使用可注入的 PlanningService、领域 Runtime、内存 checkpointer 和企业工具实现，不消耗模型额度；覆盖 DAG 校验、子图状态隔离、领域工具白名单、非法工具自恢复、复合任务、逐工具确认、拒绝传播、原生流过滤和请求级幂等。

## 生产扩展点

- 将 policy bootstrap 替换为带版本、权限标签和生效区间的离线摄取流水线；检索时增加 ABAC filter 与引用返回。
- 为 `EnterpriseToolProvider` 增加 OA、财务和 HR 远端适配器，并配套 outbox、状态回查和补偿任务。
- 在 API Gateway 接入 OIDC/JWT、租户隔离、速率限制、审计日志与 PII 脱敏。
- 增加会话级并发租约、模型/工具熔断、分布式限流和请求级成本预算。
- 后台运行注册表落库并加租约与孤儿回收，使进程崩溃后的半途运行可被识别和恢复；
  事件流改用 Redis Streams 承载，支持多副本下的断线重连。
- PostgreSQL checkpoint 支持多实例恢复；大规模部署需设置连接池、checkpoint 清理策略和 Redis/Milvus 高可用。
- 需要到期主动通知这类能力时，先接入专用调度服务（如 Temporal/Celery）再开放对应工具。
  当前刻意不提供只写一条记录、到期什么都不会发生的提醒工具：承诺未来动作却不兑现，比不提供这个能力更糟。

## 已知边界

这是可运行、可替换后端的生产级架构基线。具体企业的差旅额度、发票校验、组织审批链、身份权限和业务 API 仍需通过 `EnterpriseToolProvider` 适配，并在上线前完成安全评审、容量测试和故障演练。
