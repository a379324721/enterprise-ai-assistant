# 回归评测

用固定数据集衡量模型行为，让 prompt 和工具描述的调整有可比较的依据，
而不是靠人工试几句话判断"感觉变好了"。

## 评测什么

| 评测集 | 被测组件 | 指标含义 |
| --- | --- | --- |
| `context` | `LLMPlanningService.resolve_context` | 闲聊/业务请求判别准确率、指代消解是否保留关键实体 |
| `planning` | `LLMPlanningService.plan` | 领域路由准确率、拆分粒度、依赖识别 |
| `tool_choice` | `DomainAgentRuntime.decide` | 信息充分时的工具选择准确率 |
| `guardrail` | `DomainAgentRuntime.decide` | 越权写入拦截率、字段缺失时的反问率 |
| `small_talk` | `LLMPlanningService.respond_direct` | 闲聊带档案时不把"已提交"说成"已通过" |
| `domain_answer` | `DomainAgentRuntime.respond` | 领域回答不编造审批状态，不承诺代审批等系统没有的能力 |

`guardrail` 和 `small_talk` 的通过率是硬指标，任何一条不通过都应优先修复：

- `guardrail` 不通过意味着模型可能在信息不全或被诱导的情况下执行企业写操作。
- `small_talk` 不通过意味着模型会凭最近单据编造审批状态。`workflow_actions`
  只记录写操作被调用过，不含任何审批结果；状态只能来自领域 Agent 的单据查询工具，
  闲聊节点把"已提交"说成"已通过"、或许诺"帮你查一下"都是越界。
- `domain_answer` 用短语黑名单抽查领域回答。它挡得住明显的退化，但**不能证明**
  幻觉已根除：越界措辞往往在更长的真实上下文里才出现，构造的单工具场景复现不了。

## 运行

评测复用生产代码路径，但把 Postgres/Milvus 换成内存实现，因此**不需要**
启动 docker-compose，只需要配置好模型服务：

```bash
uv run python -m evals.runner                          # 全部
uv run python -m evals.runner --suite guardrail        # 单个评测集
uv run python -m evals.runner --json report.json       # 导出明细
uv run python -m evals.runner --min-accuracy 0.9       # 低于阈值以非零码退出
make eval
```

输出示例：

```
suite             passed   total   accuracy
-------------------------------------------
context                8       8     100.0%
planning               6       7      85.7%
tool_choice            7       7     100.0%
guardrail              6       6     100.0%
-------------------------------------------
overall               27      28      96.4%

失败用例（1）：
  [planning] plan-leave-then-travel: 依赖关系存在=False，期望 True
```

## 维护约定

- 用例只断言可稳定复现的行为，不断言模型的具体措辞。
- 存在多个合理答案的场景不要放进数据集，否则准确率会失去意义。
- 修 badcase 的顺序是：**先在 `cases.yaml` 补一条复现用例，再改 prompt**，
  这样这次修复在后续迭代中不会被悄悄改坏。
- `tests/test_evals.py` 在 CI 中离线校验数据集完整性和判定逻辑，不消耗模型额度。
