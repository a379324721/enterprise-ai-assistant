# AGENTS.md

本文件为编码 agent 提供本仓库的上下文，是各工具共用的唯一真相源。
Claude Code 不原生读取 AGENTS.md，它通过 CLAUDE.md 里的 `@AGENTS.md` 导入读到这里的内容。

## 常用命令

```bash
make install     # uv sync
make dev         # uvicorn enterprise_ai_assistant.main:app --reload
make test        # uv run pytest
make lint        # uv run ruff check .
make typecheck   # uv run mypy（strict，覆盖 enterprise_ai_assistant 和 evals）
make compose-up  # docker compose up --build（Postgres + Redis + Milvus + api）
```

跑单个测试：

```bash
uv run pytest tests/test_memory.py -q
uv run pytest tests/test_memory.py::test_leave_summary_drops_the_reason_text -q
```

`pytest` 配置了 `asyncio_mode = "auto"`，异步测试不需要 `@pytest.mark.asyncio` 也能跑（现有测试仍显式标注）。CI（`.github/workflows/ci.yml`）跑的就是 lint + typecheck + pytest 三件套，都不需要模型服务或中间件。

## 评测

```bash
make eval                                        # 全部评测集
uv run python -m evals.runner --suite guardrail  # 单个评测集
uv run python -m evals.runner --json report.json --min-accuracy 0.85
```

**评测会真实调用模型服务，产生费用**，所以不在 CI 每次推送时跑（`eval.yml` 是手动触发 + 每周定时）。改动 prompt 后应当手动跑一次。

`guardrail` 和 `small_talk` 是硬指标：前者不通过意味着模型可能在信息不全或被诱导时执行企业写操作，后者不通过意味着模型会凭最近单据编造审批状态。新增评测用例写在 `evals/cases.yaml`，新增 suite 需要同步 `evals/dataset.py`、`evals/runner.py` 的 `SUITES` 和 `eval.yml` 的 choices。

## 前端

```bash
cd frontend && npm run dev    # vite
cd frontend && npm run build  # tsc -b && vite build
```

## 架构

后端是 FastAPI + LangGraph 的多 Agent 系统。真正需要跨文件理解的是下面几条边界，它们是这个代码库的设计核心，改动时最容易被无意破坏。

### 职责分层不可越界

三层各自的禁区写在各自的 prompt 里（`services/planning.py`、`agents/domain_runtime.py`）：

- **Context Supervisor**（`resolve_context`）只消解指代、判断意图、生成独立请求。**不得抽取或补写差旅、报销、请假等领域字段。**
- **Planner**（`plan`）只产出任务 DAG：领域、目标、成功标准、依赖。**不选工具、不生成参数、不判断风险。**
- **领域 Agent**（`DomainAgentRuntime`）才判断字段是否齐全、选择工具、解释结果。

领域有 travel、expense、hr、meeting 四个业务域加一个兜底的 policy。新增领域时，`CAPABILITY_SUMMARY`、`_DOMAIN_INSTRUCTIONS`、planner prompt 的领域清单和评测集都要跟上——评测里有一条断言会检查数据集是否覆盖了每个领域。

往上层塞领域逻辑是最常见的错误改法。需要字段级能力时，应该落在领域子图里。

### 父图与领域子图只通过两个契约通信

`graph/workflow.py` 是调度图，`graph/domain.py` 是领域任务子图。两者之间只有 `DomainTaskRequest` 和 `DomainTaskResult`（`core/models.py`）。

`domain_messages`、工具决策、确认状态都是子图私有状态，不进 `AssistantState`。后续任务只能通过 `artifacts[task_id]` 拿到前置任务的结构化产物，拿不到它的对话过程。

子图有两种中断，都通过 LangGraph interrupt payload 暴露，API 层不依赖子图内部节点名：写操作前的人工确认（`PendingConfirmation`）和缺字段时的追问（`PendingInput`，payload 里的 `kind` 是判别位）。两者的标识都必须在进入中断**之前**存进子图状态——中断节点会在 resume 时重新执行，当场生成的 id 和客户端回带的那个对不上。

**追问不结束本轮。** 早先 `request_information` 之后整轮就收尾，用户的回答成为新的一轮、重新走 `understand` → `plan`，于是同一请求里尚未执行的任务被新计划覆盖掉——"出差顺便订个会议室"补完事由之后，会议室任务凭空消失，而差旅那张单还有被重复创建的风险（幂等键按 `request_id` 派生，新一轮就是新的键）。现在它和确认走同一条恢复路径（`POST /conversations/{id}/input/stream`），任务 DAG 留在检查点里原地继续。挂在追问上的会话不接受普通 `chat` 请求（409）：那会被当成新的一次 invoke，把中断点连同 DAG 一起丢掉。

追问轮**不走 `respond`**：面向用户的问题就是 `request_information` 的 `question` 参数，让模型再复述一遍既多花一次调用，又会同时存在两份措辞不同的问题（界面上就成了对话里问一遍、提示条里再问一遍）。因此这一轮没有流式输出，`question` 就是助手那条消息的正文，它**不在** `messages` 里。恢复时 `_resume_input_command` 把提问（`AIMessage`）和回答（`HumanMessage`）成对补进历史——不补的话刷新之后对话中间凭空少一条，看起来像用户无缘无故答了"培训"。用户尚未回答就刷新的那段时间里，界面从 `pending_input.question` 临时补显示。

### 谁能读原始 messages

只有从对话中提取信息的两个节点：`understand`（提意图）和 `remember`（提记忆）。执行链路上的 `plan`、`select_task`、领域子图、`direct_respond` 一律只消费 `ContextResolution` 等结构化输出。

因此闲聊节点看不到用户原话，回复语言只能靠 `ContextResolution.user_language` 传下去。

人工确认的决定会随恢复命令追加进 `messages`（`_decision_message`），刻意用 `SystemMessage`：
`_conversation()` 只挑 human/ai，于是这条记录进得了会话历史和界面（`ConversationMessage.role`
的第三种取值 `decision`），进不了模型上下文。改成 `HumanMessage` 会让下一轮的 Context
Supervisor 把它当成用户的新输入。

### 可信上下文不经过模型

`user_id` 取自访问令牌的 `sub`（`core/security.py` 的 `CurrentUser`），`conversation_id`、`request_id`、幂等键都来自运行时，通过 `ToolContext` 注入工具，**不作为模型可见的工具参数**。让模型或客户端指定身份等于开放越权。

### 工具风险与人工确认

工具风险由服务端注册表声明（`tools/registry.py` 的 `ToolRisk`），不由模型判断。所有 `WRITE` 工具在执行前保存检查点并逐个要求用户确认。写操作经 `PostgresActionRepository.execute_once` 落 `workflow_actions` 表做幂等，`idempotency_key` 同时就是对外的 `reference_id`。

注意 `result.status` 是 `"recorded"`，只表示适配器被调用过，**不代表外部企业系统已受理或审批通过**。任何"已通过""审批中"的说法都是幻觉。

### 执行与 SSE 连接解耦

图执行跑在 `RunManager` 的后台任务里（`core/runs.py`），SSE 只是订阅者。客户端断开默认不中断执行（`RUN_ON_DISCONNECT=continue`），重连带 `Last-Event-ID` 只补发缺失增量，游标滚出缓冲窗口会收到 `gap` 事件。`StreamBridge` 是可替换抽象，单进程用 `MemoryStreamBridge`，多副本部署需要跨进程实现或粘性路由。

### 长期记忆

默认关闭（`MEMORY_ENABLED=false`）。分两层，来源不同：

- `profile` / `preference` 存 `user_memories` 表，`UNIQUE(user_id, kind, key)` 覆盖写。
- 近期业务事实**不复制**，从 `workflow_actions` 派生。单号的真相只有那一处，复制一份在单据作废后无法失效。派生摘要走 `_ACTION_SUMMARY_FIELDS` 白名单，请假原因、票据号、备注正文不进模型上下文。

界面右栏的"我的单据"（`GET /actions`）走的是同一个 `recent_actions`，因此**不受 `MEMORY_ENABLED` 门控**——单据是用户自己办过的事，不是画像。露出的单号只能取 `result` 里存下的短单号（`build_reference_id` 的产物）；`idempotency_key` 是 会话:请求:任务:工具 拼成的，退回它就等于把内部结构同时泄漏进模型上下文和界面。

`recall` 在轮首、`remember` 在轮尾，所有终止分支都汇到 `remember`。记忆只作为字段的建议默认值，不构成用户已确认的事实——余额、额度、制度条款一律以实时查询工具为准。

### 演示登录与会话模型

`/auth/login` 用名字换令牌，**没有凭据**：输入他人的名字即可接管该身份。没有独立的
注册接口——没有凭据时注册和登录是同一件事，名字没见过就直接建号（`get_or_create`）。它受 `APP_ENV=development` 和 `DEMO_LOGIN_ENABLED` 双重保护，配置校验拒绝在
非开发环境开启，关闭时返回 404 而非 403。改动这一带时不要放宽任何一层。

演示用户固定一个会话：`conversation_id` 由 `user_id` 经 uuid5 派生（`repositories/users.py`），
不存映射表，所以没有会话列表这个概念。名字先经 `normalize_name` 折叠空白再作主键，
否则"张三"和"张 三 "会成为两个互不可见的身份。

展示名走令牌的 `name` 声明进入 `AssistantState.user_name`，只用于 prompt 里的称呼；
会话归属、幂等键和一切鉴权仍然只认 `user_id`。它刻意不进长期记忆，理由见上一节的
筛选机制——姓名永远相关，会被相关性筛选丢掉。

### 降级原则

Redis、Milvus、记忆仓储不可用时记日志并继续，不阻断业务：制度检索失败让工具返回明确的失败结果而不是编造内容，token 预算查不到时放行，记忆查不到时退化成无记忆行为。启动阶段用 `AsyncExitStack` 登记资源，任一步失败都会按逆序释放。

## 配置

所有配置经 `core/config.py` 的 `Settings` 校验，敏感项绝不设默认值。`.env.example` 是权威列表。生产环境必须显式配置 `JWT_SECRET`（HS* 要求至少 32 字节），开发环境留空会生成一次性密钥。

注意：`get_settings()` 会读取项目根目录的 `.env`。如果本地 `.env` 里 `DEV_LOGIN_ENABLED=true`，`tests/test_auth.py::test_dev_token_endpoint_is_hidden_by_default` 会失败——该测试假设默认关闭，CI 无 `.env` 时通过。这是环境差异，不是代码缺陷。

## 代码风格

- 注释、文档、README 用中文；commit message 用英文（主题行小写祈使句，正文说明"原来怎样、为什么有问题、现在怎样"）。
- 注释解释**为什么**这样做、以及不这样做会出什么问题，不复述代码在做什么。现有注释密度和这个取向是刻意的，新代码应当匹配。
- ruff line-length 100，`select = ["E", "F", "I", "UP", "B", "ASYNC"]`；mypy `strict`。
- 仓储类一律提供 `InMemory*` 实现供测试和 evals 使用，不要在测试里 mock 数据库。
