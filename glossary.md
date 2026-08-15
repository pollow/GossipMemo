# GossipMemo Glossary

这份词汇表规定领域文档中的最小术语边界。除非特别说明，泛指长期语义记录时使用 `Memory`，不要用 `fact` 代替它。

## Domain terms

- **Message**：持久化的原始 evidence，记录输入中实际说了什么。它是不可变、durable 的依据，不是语义归纳结果。
- **Memory**：一条 durable semantic claim/record（系统决定长期保留的语义认识）。它可以被检索、修正、retract、supersede 和追溯，但不保证客观真实；`reported`、`inferred` 等 `basis` 必须保留。
- **fact**：`Memory.kind` 的一种静态陈述分类。`fact ⊂ Memory`；它不是 Memory 的同义词，也不表示该记录已被客观验证。
- **Hypothesis**：一个有依据、但尚不足以作为稳定认识使用的 tentative interpretation。它属于 `user`、`person` 或 `relationship` owner，支持 support/counter evidence 和显式 lifecycle；它不是 Memory，也不能作为后续 reasoning 的 evidence。
- **CoverageMap**：每个 Space 一份、从非 inferred Memory 增量审计得到的可重建认知覆盖 projection。它记录 20 个稳定 criteria 的 `unknown / fragmentary / grounded / rich` 状态，以及 edge、blind spot、conflict；它不是用户画像，也不直接暴露给 chat agent。
- **LearningGoal**：由 UserLearningGoalReasoner 从最新 CoverageMap、open Hypotheses 和既有 goals 规划的、user-owned 可选了解方向。它描述“值得进一步了解什么”，不表示用户有义务回答，也不会因创建而提高 coverage。
- **projection**：从 canonical data（主要是 active Memories）生成的、可删除并重建的 card 或 context，例如 Person、Relationship、UserModel card 和 rolling continuity。projection 本身不是新的 evidence。

## Processing terms

- **extraction**：从 Message 产生 Memory 的处理阶段，即 `Message → Memory`。批量 LLM extraction 可以先产生 candidate，但对外流程仍按这一语义理解。
- **reasoning**：针对一个 owner 的一组 LLM computation。Person、Relationship 和 UserModel 先生成 projection，再用相同 context/prefix 做 epistemic review；continuity reasoning 读取 prior continuity 和一个有界的 Message window。
- **induction**：启动时及每日本地午夜运行的 orchestration。它扫描 stale Person、Relationship、UserModel 和 CoverageMap projections，并调度对应 reasoning；它不是一次 reasoning，也不负责按消息阈值触发 continuity。
- **epistemic review**：owner reasoning 的第二个 call。它读取 evidence Memories、现有 inferred Memories 和 open Hypotheses，显式返回 upsert/retract/transition；遗漏永远是 no-op。User owner 只维护 Hypotheses，不产生 inferred Memories。
- **coverage audit**：UserLearningGoalReasoner 对 bounded Memory chunks 做的增量审计。Hypothesis 只能帮助发现 boundary/conflict，不能提高 coverage level。
- **goal planning**：仅在 CoverageMap catch up 后运行的独立 LLM call；它维护 LearningGoal lifecycle，并从已知边界中选择少量、可拒绝的了解方向。
- **projection refresh**：一次完整的 projection 更新动作，包括 reasoning、stale/optimistic freshness check，以及成功后的 optimistic writeback。只有写回成功才算 refresh 完成。

profile projection 的标准处理流程：

```text
Message → extraction → Memory → owner reasoning → projection + epistemic actions
                             ↘ coverage audit → CoverageMap → goal planning
```

## Design terms

以下词汇沿用 `codebase-design` 的含义：

- **Module**：有一个 Interface 和一个 Implementation 的单元。
- **Interface**：调用方正确使用 Module 所需知道的全部约束，不只是类型签名。
- **Implementation**：Module 内部的实现细节。
- **Seam**：Interface 所在、可以替换行为的位置。
- **Adapter**：在 Seam 处满足 Interface 的具体实现。

这里不使用 `API` 泛指 Interface；HTTP API 仅指 transport adapter 暴露的具体 HTTP surface。
