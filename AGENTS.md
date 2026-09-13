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

改动思考开关或 `DOMAIN_THINKING_BUDGET` 后同样要跑，至少跑 `guardrail`：领域 Agent 关掉思考会在缺字段时直接调写工具（说明见 `core/config.py`）。

`guardrail` 和 `small_talk` 是硬指标：前者不通过意味着模型可能在信息不全或被诱导时执行企业写操作，后者不通过意味着模型会凭最近单据编造审批状态。新增评测用例写在 `evals/cases.yaml`，新增 suite 需要同步 `evals/dataset.py`、`evals/runner.py` 的 `SUITES` 和 `eval.yml` 的 choices。

## 排查线上行为：LangSmith trace

实测出现异常回答时，先看 trace 再下结论，不要凭回答文本猜是哪个节点出的错。trace 由运行中的服务上报到 `.env` 里的 `LANGSMITH_PROJECT`。

**key 要从 `Settings` 取出来显式传给 `Client`。** 它只在项目 `.env` 里，shell 环境变量里没有，直接 `Client()` 会拿不到 key、返回 401。

```python
from langsmith import Client
from enterprise_ai_assistant.core.config import get_settings

s = get_settings()
client = Client(api_key=s.langsmith_api_key.get_secret_value(), api_url=s.langsmith_endpoint)
```

- **找某一轮**：根 run 的 `extra.metadata` 带 `user_id` 和 `conversation_id`（`routes.py` 的 `_config` 写入）。用 `client.list_runs(project_name=s.langsmith_project, is_root=True, start_time=...)` 取回后按这两个字段过滤；根 run 的 `inputs.messages` 就是这一轮的用户输入，确认恢复的轮次没有 messages。
- **看节点输入输出**：`client.list_runs(trace_id=<根 run id>)` 取整棵树，按 `dotted_order` 排序。`context-supervisor` 的 outputs 是 `ContextResolution`，`run_type == "llm"` 的子 run 的 inputs 是完整 prompt。
- **本机网络**：shell 里有 SOCKS 代理变量，而 httpx 没装 socksio。访问 LangSmith 和模型服务的脚本要用 `env -u all_proxy -u ALL_PROXY` 启动。
- **看不到 trace 时**：会话停在确认卡上，父图 checkpoint 里还留着本轮的 `understanding` 和 `domain_request`（其中有 `recent_messages`），可以用 `AsyncPostgresSaver.aget_tuple({"configurable": {"thread_id": <conversation_id>, "checkpoint_ns": ""}})` 读出来。

## 前端

```bash
cd frontend && npm run dev    # vite
cd frontend && npm run build  # tsc -b && vite build
```

## 架构

后端是 FastAPI + LangGraph 的多 Agent 系统。真正需要跨文件理解的是下面几条边界，它们是这个代码库的设计核心，改动时最容易被无意破坏。

### 职责分层不可越界

三层各自的禁区写在各自的 prompt 里（`services/planning.py`、`agents/domain_runtime.py`）：

- **Context Supervisor**（`resolve_context`）消解指代、判断意图、生成独立请求，并在同一次结构化输出里给出任务 DAG（`ContextResolution.tasks`：领域、标题、目标、依赖）。依赖写前置任务的**领域**而不是 id：不开思考时模型写字符串 id 数组会生成残缺值，写整数序号会 0 起 1 起混用，而拆分粒度本来就是一个领域一个目标。`continue` 轮次也要给出任务，误报 `continue`、指认不到事项时照它执行。**不得抽取或补写差旅、报销、请假等领域字段，不选工具、不生成参数、不判断风险。** 依赖指向的领域不在排在前面的任务里时，`ContextResolution` 校验抛错（只能依赖前面的任务，也就不会成环），交给结构化输出重试。
- **Planner**（`plan`）只是兜底：理解结果需要执行、模型却漏写了任务时才调用，`SupervisorAgent.plan` 负责二选一。原先它是每个多领域轮次的必经调用，但它的输入只有理解结果，领域归类也已在理解阶段做完，并进去每轮省一次模型调用。两处的领域归类共用 `services/planning.py` 的 `_DOMAIN_ROUTING`，改归类规则只改那一处。
- **领域 Agent**（`DomainAgentRuntime`）才判断字段是否齐全、选择工具、解释结果。

领域有 travel、expense、hr、meeting 四个业务域加一个兜底的 policy。新增领域时，`CAPABILITY_SUMMARY`、`_DOMAIN_INSTRUCTIONS`、`_DOMAIN_ROUTING` 和评测集都要跟上——评测里有一条断言会检查数据集是否覆盖了每个领域。

往上层塞领域逻辑是最常见的错误改法。需要字段级能力时，应该落在领域子图里。

### 父图与领域子图只通过两个契约通信

`graph/workflow.py` 是调度图，`graph/domain.py` 是领域任务子图。两者之间只有 `DomainTaskRequest` 和 `DomainTaskResult`（`core/models.py`）。

`domain_messages`、工具决策、确认状态都是子图私有状态，不进 `AssistantState`。后续任务只能通过 `artifacts[task_id]` 拿到前置任务的结构化产物，拿不到它的对话过程。人工确认通过 LangGraph interrupt payload（`PendingConfirmation`）暴露，API 层不依赖子图内部节点名。

任务派错领域时，领域 Agent 调 `handoff_task`（每个领域都有，描述里列出其他领域的能力）交还，子图返回 `status=HANDED_OFF` 且不带回答。父图的 `Workflow._reroute` 用代码把任务改派给 `handoff_to`，任务 id 不变、回到 `PENDING`，下游依赖照常；**不回到 Supervisor 重新理解**——它读的是同一段会话，大概率再分错一次，还多一次模型调用。防空转的规则：去过的领域（`PlannedTask.handed_off_from`）不再去、最多转交 `MAX_HANDOFFS` 次，超出就判 `FAILED` 并回固定文案 `HANDOFF_EXHAUSTED_REPLY`；执行过写操作的任务不能转交（`DomainTaskWorkflow._handoff_error`），转交给本领域也会被打回给模型。`HANDED_OFF` 只出现在子图结果上，不会存进计划。

### 未办完的事项：续跑、搁置、取消

领域 Agent 缺字段时调用 `request_information`，任务停在 `WAITING_INPUT`，这一轮结束。当前计划（`plan_id`、`user_goal`、`tasks`、`artifacts`，以及追问时报告的 `drafts`）跨轮保留；用户换话题时，没办完的计划整体移进 `shelved_plans`，不自动过期。

每轮 `understand` 把当前计划和搁置计划里待补充任务的摘要（`OpenTask`：`plan_id`、标题、缺失字段**名**、是否搁置，没有字段值）交给 Context Supervisor，由它填 `turn_relation` 和 `target_plan_id`：

- `continue`：不重新规划，待补充的任务放回 `PENDING` 续跑，草稿经 `DomainTaskRequest.draft` 交还领域 Agent。指向搁置计划时整体换回来，当前计划没办完就换下去搁置。`user_goal` 不变（界面展示的是整件事的目标），本轮的补充经 `standalone_request` 进 `DomainTaskRequest.user_goal`。
- `cancel`：目标计划里 `WAITING_INPUT` / `PENDING` 的任务改为 `REJECTED`（搁置计划直接移除），回复由运行时按实际放弃的事项用固定文案写出（`Workflow._cancelled_text`），指认不到事项时回 `NOTHING_TO_CANCEL_REPLY`——不用模型写的 `reply`，模型在同一次输出里无从知道运行时能不能指认到它。已提交的单据不在清单里，也不受影响——撤销已提交单据是 `new` 的业务请求，走领域的 `revoke_*` 工具。
- `new` 且需要规划：当前计划没办完就搁置，然后照常规划。
- `new` 且不需要规划（闲聊、道谢、清单外诉求）：什么都不动，直接发出 `ContextResolution.reply`，用户回头还能补充。

指向规则（`Workflow._target`）：`target_plan_id` 命中搁置计划就用它；否则当前计划有待补充任务就是当前计划；否则搁置计划只有一件时就是它；都不满足则忽略 `continue` / `cancel`，按常规路径处理。多件搁置时不猜，宁可重新规划也不把补充信息塞给错的事项。

界面右栏的"进行中"就是这些计划的投影：`routes.py` 的 `_matters` 把卡在待补充 / 待确认上的当前计划和全部搁置计划转成 `AssistantResponse.matters`，字段来自 `drafts`，所以 `request_information` 的 `missing_fields` 要求中文字段名。办完的计划不成为事项。对话流里的执行步骤是 `AssistantResponse.steps`，由本轮 `tool_results` 加 `tools/registry.py` 的 `TOOL_LABELS` 生成；新增工具要同步起中文名，有测试检查。

不要改回"每轮清空再规划"：重新拆出来的任务 id、标题、粒度都可能变，前置任务的产物也跟着丢，实测会议室任务就是这样在差旅追问之后消失的。也不要给 `OpenTask` 加字段值——Supervisor 拿到值就有了补写领域字段的材料。

### 谁对用户说话、各自读得到什么

对用户输出文字的模型只有两类，而且都必须知道助手之前说过什么，否则会重复称呼、重复追问、同一条消息里互相否定：

- **Context Supervisor**（`understand`）：读完整会话窗口（含全部助手回复），本轮不执行任务时在同一次结构化输出里写 `ContextResolution.reply`。没有单独的闲聊节点——拆出去的节点看不到会话，重复称呼、许诺办不到的事都出在这里。`reply` 由校验器强制：不执行任务、也不是取消的轮次缺 `reply` 就算输出无效，交给结构化输出的重试。代价是这类回复一次性给出，不逐字流出（结构化输出必须关流式）。
- **领域 Agent**（`decide`）：经 `DomainTaskRequest.recent_messages` 读最近 `DOMAIN_CONTEXT_MESSAGES`（默认 10）条会话原文，用户和助手的都有，包括本轮排在前面的任务刚写进 `messages` 的回答；不带 Supervisor 那份早先会话摘要。回答不另起一次调用：执行过工具之后的 `decide` 不再调工具时，它的文字就是回答（`answering=True`，打 `user-visible` 标签流出）；`request_information` 的 `question` 原样发出，不经模型，经 LangGraph custom 流推给 SSE。SSE 侧的 `_AnswerRelay` 先压住回答开头，见到工具调用增量就整条丢弃——前端不会用 `done` 覆盖已经画出去的文字。不带工具的 `respond` 只兜底用户拒绝确认、工具失败和 `decide` 给不出文字这几种情况。

领域 Agent 读原文而不是只拿改写后的 `standalone_request`：改写会丢信息或解析错（评测 `tool-original-words-recover-what-the-rewrite-dropped` 里漏掉了出发地），只有改写时领域 Agent 发现不了，也不知道用户追问的"为什么"指什么。代价是可能从不相干的旧事项里串字段，由两层守住：prompt 规定写工具参数只能取自改写请求、草稿、依赖产物和用户针对**当前这件事**说过的话，旧事项的值不得沿用、assistant 消息不作取值来源（评测 `guard-old-trip-values-are-not-reused`）；确认卡把写工具的每个参数逐项给用户过目。窗口比 Supervisor 的小，是为了少给串字段的材料；不要为了"信息更全"把它调成完整会话。

其余用户可见文字都是模板，不经模型：确认卡片（`PendingConfirmation.title` 取 `TOOL_LABELS`，`fields` 的中文名取工具入参契约里的 `Field(title=...)`，取值标签取 `json_schema_extra["value_labels"]`，有测试检查每个写工具字段都声明了 title）、执行步骤、取消事项的回复、错误提示。写工具的参数在 `decide` 阶段就按契约校验（`RegisteredTool.argument_error`），非法参数交还模型更正，不会先弹出确认卡。

Supervisor 只拿记忆的 key（理由见"长期记忆"），所以直接回复不做基于档案的个性化；它拿单据清单（`RecentAction.render()`）和称呼，用于指认"第一条"是哪张单、称呼用户。

人工确认的决定会随恢复命令追加进 `messages`（`_decision_message`），刻意用 `SystemMessage`：
`_conversation()` 只挑 human/ai，于是这条记录进得了会话历史和界面（`ConversationMessage.role`
的第三种取值 `decision`），进不了模型上下文。改成 `HumanMessage` 会让下一轮的 Context
Supervisor 把它当成用户的新输入。

### 可信上下文不经过模型

`user_id` 取自访问令牌的 `sub`（`core/security.py` 的 `CurrentUser`），`conversation_id`、`request_id`、幂等键都来自运行时，通过 `ToolContext` 注入工具，**不作为模型可见的工具参数**。让模型或客户端指定身份等于开放越权。

### 工具风险与人工确认

工具风险由服务端注册表声明（`tools/registry.py` 的 `ToolRisk`），不由模型判断。所有 `WRITE` 工具在执行前保存检查点并逐个要求用户确认。写操作经 `PostgresActionRepository.execute_once` 落 `workflow_actions` 表做幂等，`idempotency_key` 同时就是对外的 `reference_id`。

注意 `result.status` 是 `"recorded"`，只表示适配器被调用过，**不代表外部企业系统已受理或审批通过**。审批状态只能来自各领域的单据查询工具（`query_*`，READ 风险，只按 `ToolContext.user_id` 查，单号不作鉴权依据），不是来自查询结果的"已通过""审批中"都是幻觉。本地适配器的状态是 `mock_submission_status` 按单号哈希给出的替身，改过的单据回到审批中。

查询工具返回单据的**全部**字段，不走 `_ACTION_SUMMARY_FIELDS` 白名单：白名单防的是每轮被动注入（用户没问，请假原因也跟着档案进上下文），查询是本人对自己单据的主动请求，修改前也必须拿到原值。修改工具（`update_*`）是 WRITE，照常逐个确认；它只传要改的字段，合并后按新建时的同一份契约重新校验，原单据行原地更新，另记一行 `<action_type>_update` 做幂等和审计。撤销工具（`revoke_*`，WRITE）只给原单据打 `revoked_at` 标记不删行，另记 `<action_type>_revoke`；撤销是终态，撤销后不能再修改，"我的单据"标为已撤销，会议室撤销后时段让出。系统没有代审批的工具。

查询、修改、撤销单据的请求由 Context Supervisor 归入单据所属领域（规则在 `_DOMAIN_ROUTING`）。Supervisor 直接回复时手里没有查询结果，不得断言状态，也不得许诺"帮你查一下"——不执行任务的轮次下一步什么也不会发生。

### 执行与 SSE 连接解耦

图执行跑在 `RunManager` 的后台任务里（`core/runs.py`），SSE 只是订阅者。客户端断开默认不中断执行（`RUN_ON_DISCONNECT=continue`），重连带 `Last-Event-ID` 只补发缺失增量，游标滚出缓冲窗口会收到 `gap` 事件。`StreamBridge` 是可替换抽象，单进程用 `MemoryStreamBridge`，多副本部署需要跨进程实现或粘性路由。

### 没有依赖的任务并行执行

`select_task` 一次挑出所有依赖已完成的任务（`SupervisorAgent.runnable_tasks`），`route_task` 用 `Send` 并行派发，`apply_domain_result` 等本批全部结束后按**计划顺序**归并（回答写进会话的顺序、失败连带取消都和计划一致，不取决于谁先跑完）。有依赖的任务等下一批。

- 领域子图包在 `run_domain_task` 函数里调用，不直接挂成节点：并行分支会同时写父图的 `domain_result`，改写进带归并规则的 `domain_results`（写 `None` 清空）。子图的检查点和 interrupt 照常继承。
- 两个分支可能同时停在确认卡上。恢复必须按中断 id 指明（`_resume_command`），给单个值 LangGraph 直接报错；只恢复一个时另一个分支不会重跑它前面的决策（`tests/test_parallel.py` 断言了调用次数）。已恢复跑完的分支在本批结束前仍挂着原中断记录，`_pending_interrupt` 按 `snapshot.tasks` 过滤掉有结果的，界面一次出一张卡。本批里先确认的任务，回答要等整批结束才归并进会话。
- 执行并行、展示串行：`_AnswerRelay` 同一时刻只转发一段回答，先开口的先流，其余攒着依次放出——前端把增量追加到最后一个气泡，交错转发会把两段话搅在一起。回答按节点执行的 `langgraph_checkpoint_ns` 区分，靠 `chunk_position="last"` 判断一次调用结束；不要改回按节点事件放行，并行时别的分支的节点事件会把另一个分支的工具调用前缀提前放出去。
- 同一批并行的任务彼此看不到对方的回答（`recent_messages` 在派发时就定了），更容易出现"另一件事不归我管"这类越界说法，见 `docs/known-issues.md`。

### 长期记忆

默认关闭（`MEMORY_ENABLED=false`）。分两层，来源不同：

- `profile` / `preference` 存 `user_memories` 表，`UNIQUE(user_id, kind, key)` 覆盖写。
- 近期业务事实**不复制**，从 `workflow_actions` 派生。单号的真相只有那一处，复制一份在单据作废后无法失效。派生摘要走 `_ACTION_SUMMARY_FIELDS` 白名单，请假原因、票据号、备注正文不进模型上下文。

界面右栏的"我的单据"（`GET /actions`）走的是同一个 `recent_actions`，因此**不受 `MEMORY_ENABLED` 门控**——单据是用户自己办过的事，不是画像。露出的单号只能取 `result` 里存下的短单号（`build_reference_id` 的产物）；`idempotency_key` 是 会话:请求:任务:工具 拼成的，退回它就等于把内部结构同时泄漏进模型上下文和界面。

`recall` 在轮首、`remember` 在轮尾，所有终止分支都汇到 `remember`。抽取在 `remember` 里用后台任务跑，不阻塞本轮的 `done` 和执行锁；进程关停时 `Workflow.drain_background` 等它写完再关连接池，测试里要显式调用它才能断言写入结果。记忆只作为字段的建议默认值，不构成用户已确认的事实——余额、额度、制度条款一律以实时查询工具为准。

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
