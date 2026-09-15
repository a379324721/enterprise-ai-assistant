# AGENTS.md

本文件为编码 agent 提供本仓库的上下文，是各工具共用的唯一真相源。Claude Code 通过 CLAUDE.md 里的 `@AGENTS.md` 导入读到这里。

只写改代码时容易被无意破坏的边界和约定，每条附一句为什么。不写事故经过和历史演变——那些在 commit message 里。

## 常用命令

```bash
make install     # uv sync
make dev         # uvicorn enterprise_ai_assistant.main:app --reload
make test        # uv run pytest
make lint        # uv run ruff check .
make typecheck   # uv run mypy（strict，覆盖 enterprise_ai_assistant 和 evals）
make compose-up  # docker compose up --build（Postgres + Redis + Milvus + api）
cd frontend && npm run dev    # vite；npm run build 是 tsc -b && vite build
uv run pytest tests/test_memory.py::test_leave_summary_drops_the_reason_text -q
```

CI 跑 lint + typecheck + pytest，都不需要模型服务或中间件。`asyncio_mode = "auto"`。

## 评测

```bash
uv run python -m evals.runner --suite guardrail  # 单个评测集；不带 --suite 跑全部
```

- **评测真实调用模型、按 token 计费**，不在 CI 里跑。改了 prompt、模型、思考开关、推理预算或温度后，跑受影响的评测集，连跑两轮，两轮不一致再加跑——单轮结果波动大。
- `guardrail`（信息不全或被诱导时不得执行写操作）和 `small_talk`（直接回复不得编造状态、许诺、谎称查过）是硬指标。
- 用例写在 `evals/cases.yaml`；新增 suite 要同步 `evals/dataset.py`、`evals/runner.py` 的 `SUITES` 和 `eval.yml` 的 choices。

## 排查线上行为

出现异常回答先看 LangSmith trace，不要凭回答文本猜是哪个节点出的错。

- key 只在 `.env` 里，要从 `Settings` 取出来显式传：`Client(api_key=s.langsmith_api_key.get_secret_value(), api_url=s.langsmith_endpoint)`。
- 根 run 的 `extra.metadata` 带 `user_id`、`conversation_id`；`client.list_runs(trace_id=...)` 按 `dotted_order` 排序得到整棵树，`run_type == "llm"` 的 inputs 是完整 prompt，`context-supervisor` 的 inputs 是理解阶段的入参，可以拿来重放。
- shell 里有 SOCKS 代理而 httpx 没装 socksio，访问 LangSmith 和模型服务的脚本用 `env -u all_proxy -u ALL_PROXY` 启动。
- 看不到 trace 时，停在确认卡上的会话可以用 `AsyncPostgresSaver.aget_tuple({"configurable": {"thread_id": <conversation_id>, "checkpoint_ns": ""}})` 读检查点。

## 架构

FastAPI + LangGraph 的多 Agent 系统。`graph/workflow.py` 是调度父图，`graph/domain.py` 是领域任务子图。

### 职责分层

- **Context Supervisor**（`services/planning.py` 的 `resolve_context`）：读完整会话，改写请求、判断 `turn_relation`、在同一次结构化输出里给出任务 DAG（领域、标题、目标，`depends_on` 写前置任务的领域）；本轮不执行时写 `reply`。**不抽取领域字段、不选工具、不判断风险。**
- **Planner**（`plan`）只是兜底：需要执行、模型却没给任务时才调用。领域归类规则只在 `_DOMAIN_ROUTING` 一处。
- **领域 Agent**（`agents/domain_runtime.py`）判断字段是否齐全、选工具、写回答。需要字段级能力时落在这一层，往上层塞领域逻辑是最常见的错误改法。
- 领域：travel、expense、hr、meeting，加兜底的 policy。新增领域要同步 `CAPABILITY_SUMMARY`、`_DOMAIN_INSTRUCTIONS`、`_DOMAIN_ROUTING` 和评测集（有断言检查覆盖）。

### prompt 按原则写

两份 prompt 都按原则写，不按 badcase 逐条加禁止句；出现新的越界说法，先看能不能归到已有原则，改措辞或补例子，而不是再加一条。

- Supervisor：不调用工具；单据清单是每轮实时读的，可以直接用，审批状态、余额、制度只有执行环节查得到；说到之前做过什么只依据会话里的来源标注，和之前的说法冲突时以标注为准。
- 领域 Agent（`_PRINCIPLES`，决策和兜底回答共用）：事实只来自工具；字段只来自用户针对这件事说过的话；只管当前任务、只做工具能做的事。

### 父图与子图只通过两个契约通信

- 两者之间只有 `DomainTaskRequest` 和 `DomainTaskResult`。`domain_messages`、工具决策、确认状态是子图私有状态；后续任务只能经 `artifacts[task_id]` 拿前置任务的产物。人工确认经 interrupt payload（`PendingConfirmation`）暴露，API 层不依赖子图节点名。
- 派错领域时领域 Agent 调 `handoff_task`，父图 `_reroute` 用代码改派（任务 id 不变、回到 `PENDING`），**不回 Supervisor 重新理解**——同一段会话大概率再分错。去过的领域不再去，最多 `MAX_HANDOFFS` 次，执行过写操作的任务不能转交。

### 未办完的事项

- 缺字段时 `request_information` 让任务停在 `WAITING_INPUT`。当前计划（`plan_id`、`tasks`、`artifacts`、`drafts`）跨轮保留，换话题时整体移进 `shelved_plans`，不过期。**不要改回每轮清空再规划**：重拆的任务 id 和产物会丢。
- Supervisor 拿到的 `OpenTask` 只有标题和缺失字段**名**，不要加字段值——有值它就能补写领域字段。
- `continue` 续跑待补充的任务，草稿经 `DomainTaskRequest.draft` 交还；`cancel` 由运行时放弃事项并用固定文案回复（不用模型的 `reply`）；撤销已提交单据是 `new`，走 `revoke_*`。指向规则在 `Workflow._target`，多件搁置时不猜。
- 右栏"进行中"是这些计划的投影，规则只在 `api/matters.py` 的 `project_matters`：执行中随根图 `checkpoints` 流推 `matters` 事件，`done` 和页面加载从快照投影，前端整体替换。"是否在执行"看 `RunManager`，没有运行时停在 RUNNING 的任务按 `recover_interrupted` 解读。**不要为某个时机另写投影路径**，两套算法的差异就是 bug。

### 谁对用户说话

- 只有两类模型输出文字：Supervisor 的 `reply`（结构化输出，一次性给出，校验器强制不执行的轮次必须有 `reply`），和领域 Agent 的回答（执行过工具后 `decide` 的文字，打 `user-visible` 流出；`respond` 只兜底工具失败、没给文字）。执行过工具后，调写工具时对用户说的话（比如提交请假前先报余额）写在写工具的 `message_to_user` 参数里：模型调工具那回合几乎不写正文，prompt 怎么要求都一样，参数却每次都填。它只加在模型可见的 schema 上，`split_message` 在校验前取出，不进契约、确认卡和幂等记录；子图整段推给前端，记为 `AgentNote`：停在确认卡上时随 `PendingConfirmation.notes` 带出，历史接口补上、恢复命令写进会话；否则随 `DomainTaskResult.notes` 在归并时排在回答前面。两条路径只走一条，不然会话里会记两遍。`request_information` 的 `question` 原样发出。其余都是模板：确认卡、执行步骤、取消回复、错误提示；确认卡上点取消不回复。
- 领域 Agent 读最近 `DOMAIN_CONTEXT_MESSAGES` 条会话原文，而不是只读改写：改写会丢信息。窗口刻意比 Supervisor 小，少给串字段的材料；串字段由 prompt 的字段来源原则和确认卡兜底。
- 每条助手消息在 `additional_kwargs["tools_called"]` 记着那一轮调用过的工具（`reply_message` 写入），`_conversation()` 渲染成 `[未调用工具]` / `[调用了：…]` 交给模型，界面只取正文。没有它，Supervisor 分不清哪条回复查过，会谎称调用了系统接口。领域任务的回答另记 `task`（id 和当时的标题）和 `steps`（工具名和成败，不带 data），只给历史接口还原刷新前的标题和步骤气泡，不交给模型；步骤和 `task_done` 推的是同一份，改展示规则要两边一起改。
- 人工确认的决定用 `SystemMessage`（`decision_message`）追加进 `messages`，不用 `HumanMessage`，否则会被当成用户输入。界面历史显示正文；`_conversation()` 把它渲染成助手一侧的 `[用户在确认卡上选择了…]` 标注交给模型。取消不生成回答（`DomainTaskResult.answer` 为空），这条标注是模型知道"用户取消、没有执行"的唯一依据。

### 结构化输出

理解、兜底规划、记忆抽取共用 `_StructuredStage`：schema 以强制调用的工具下发，**不要改回 `response_format`**（DashScope 上的 DeepSeek 不按它生成，字段全靠猜）；下发的 schema 去掉 `maxLength`/`minLength`（强制调用时 DeepSeek 会一直不返回），长度由 pydantic 本地校验；关流式；校验失败时把原输出和错误说明交还模型修正，最多三次。校验器的报错会进 prompt，要写成模型能照着改的中文。

### 可信上下文与工具风险

- `user_id` 取自令牌 `sub`，`conversation_id`、`request_id`、幂等键来自运行时，经 `ToolContext` 注入，**不作为模型可见的工具参数**。
- 工具风险由注册表声明（`ToolRisk`），不由模型判断。`WRITE` 工具执行前逐个确认，参数在 `decide` 阶段就按契约校验，确认卡字段名取契约里的 `Field(title=...)`。写操作经 `execute_once` 落 `workflow_actions` 做幂等。
- `result.status == "recorded"` 只表示适配器被调用过，**不代表已受理或审批通过**；审批状态只能来自 `query_*`（只按 `user_id` 查）。查询返回全部字段；`update_*` 合并后按新建契约重新校验；`revoke_*` 只打 `revoked_at` 标记，是终态。系统没有代审批的工具。

### 执行、并行与流式

- 图执行跑在 `RunManager` 的后台任务里，SSE 只是订阅者；断开默认不中断（`RUN_ON_DISCONNECT=continue`），重连带 `Last-Event-ID` 补发。多副本需要跨进程的 `StreamBridge`。
- 依赖都完成的任务由 `select_task` 一次挑出、`Send` 并行派发，`apply_domain_result` 等整批结束后按**计划顺序**归并，每归并一个推一条 `task_done`（带执行步骤）。领域子图包在函数里调用，并行分支写带归并规则的 `domain_results`。
- 两个分支可能同时停在确认卡上，恢复必须按中断 id 指明（`_resume_command`）。
- 执行并行、展示串行：`_AnswerRelay` 同一时刻只转发一段回答，按 `langgraph_checkpoint_ns` 区分、`chunk_position="last"` 判断结束，压住回答开头，见到工具调用增量就当这段话结束、此后不再转发（前端不会覆盖已画出的文字）。
- 检查点序列化器（`graph/serde.py`）自动登记 `core.models` 里的类型；进状态的新类型放在那个模块里。

### 长期记忆

- 默认关闭（`MEMORY_ENABLED`），只管画像的读取和抽取；近期单据不受它门控——清单被连带清空时 Supervisor 看到"（暂无）"，会对有单据的用户说没有。画像存 `user_memories`；近期单据**不复制**，每轮从 `workflow_actions` 实时派生，摘要走 `_ACTION_SUMMARY_FIELDS` 白名单（请假原因、票据号不进上下文）。
- 右栏"我的单据"（`GET /actions`）用同一份数据，不受 `MEMORY_ENABLED` 门控。露出的单号只取短单号，`idempotency_key` 会泄漏内部结构。
- `recall` 在轮首，`remember` 在轮尾后台抽取；测试要调 `Workflow.drain_background` 才能断言写入。记忆只是建议默认值。Supervisor 只拿记忆的 key。

### 演示登录与降级

- `/auth/login` 用名字换令牌、没有凭据，受 `APP_ENV=development` 和 `DEMO_LOGIN_ENABLED` 双重保护，关闭时返回 404。不要放宽任何一层。`conversation_id` 由 `user_id` 经 uuid5 派生，名字先 `normalize_name`。展示名只用于称呼。
- Redis、Milvus、记忆仓储不可用时记日志继续：检索失败返回明确的失败结果，预算查不到放行，记忆查不到退化成无记忆。启动资源用 `AsyncExitStack` 登记。

## 配置

- 所有配置经 `core/config.py` 的 `Settings` 校验，敏感项不设默认值，`.env.example` 是权威列表。生产必须配 `JWT_SECRET`（HS* 至少 32 字节）。
- 模型按角色配思考和温度：Supervisor 关思考、温度 0（分类要稳定）；领域 Agent 开思考、预算 2000、温度 0.6（关思考或预算太小会缺字段直接提交，不限预算会长时间推理）。
- 部署前先读 `docs/deployment.md`。
- 本地 `.env` 里 `DEV_LOGIN_ENABLED=true` 时 `test_dev_token_endpoint_is_hidden_by_default` 会失败，是环境差异。

## 代码风格

- 注释、文档用中文；commit message 用英文（主题行小写祈使句，正文说明原来怎样、为什么有问题、现在怎样）。
- 注释解释**为什么**以及不这样做会怎样，不复述代码。
- ruff line-length 100，`select = ["E", "F", "I", "UP", "B", "ASYNC"]`；mypy strict。
- 仓储类提供 `InMemory*` 实现供测试和 evals 使用，不在测试里 mock 数据库。
