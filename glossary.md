# GossipMemo Glossary

这份词汇表规定领域文档中的最小术语边界。除非特别说明，泛指长期语义记录时使用 `Memory`，不要用 `fact` 代替它。

## Domain terms

- **Message**：持久化的原始 evidence，记录输入中实际说了什么。它是不可变、durable 的依据，不是语义归纳结果。
- **Memory**：一条 durable semantic claim/record（系统决定长期保留的语义认识）。它可以被检索、修正、retract、supersede 和追溯，但不保证客观真实；`reported`、`inferred` 等 `basis` 必须保留。
- **fact**：`Memory.kind` 的一种静态陈述分类。`fact ⊂ Memory`；它不是 Memory 的同义词，也不表示该记录已被客观验证。
- **Hypothesis**：一个有依据、但尚不足以作为稳定认识使用的 tentative interpretation。它属于 `user`、`person` 或 `relationship` owner，支持 support/counter evidence 和显式 lifecycle；它不是 Memory，也不能作为后续 reasoning 的 evidence。
- **CoverageEntry**：一条「我们在某个 path 上知道什么」的总结（数十条 Memory → 一小段话），由 coverage audit 从非 inferred Memory 增量产生。entry 只写「知道什么」，不写「还不知道什么」；它不是 Memory 的复述，不是用户画像，也不是回忆录草稿。`root` 级 overview entry 就是 path 为空的那条 entry——两者只有粒度差异，没有类型差异。
- **coverage root**：20 个稳定 memoir/persona 视角之一，是 coverage entries 唯一的结构锚点。它由 audit 的调用结构决定（哪次请求产生的就属于哪个 root），模型不填。每个 root 各自持有增量 cursor 与 CAS revision。
- **coverage**：一个 Space 的全部 coverage entries，即「我们对这个用户知道什么」的累积状态。它是**可重放的累积状态而不是可重建 projection**：entries 允许自由改写，删库重跑会得到不同但同样有效的结果。它只被 goal planning 消费，不直接暴露给 chat agent。
- **LearningGoal**：由 UserLearningGoalReasoner 按 root 扇出读 coverage entries 规划出的、user-owned 可选了解方向。它描述“值得进一步了解什么”，不表示用户有义务回答，也不会因创建而提高 coverage。消费端不做相关性排序，随机取 3–5 条随 turn 返回，是否使用由消费 agent 判断。
- **projection**：从 canonical data（主要是 Memories）生成的、可删除并重建的 card 或 context，例如 Person、Relationship、UserModel card 和 rolling continuity。projection 本身不是新的 evidence。owner card 日常由 fold 维护，「可重建」说的是它不含 canonical data 之外的信息，不是说每次刷新都重算。

## Processing terms

- **extraction**：从 Message 产生 Memory 的处理阶段，即 `Message → Memory`。批量 LLM extraction 可以先产生 candidate，但对外流程仍按这一语义理解。
- **reasoning**：针对一个 owner 的一组 LLM computation。Person、Relationship 和 UserModel 先生成 projection，再用相同 context/prefix 做 epistemic review；continuity reasoning 读取 prior continuity 和一个有界的 Message window。
- **fold**：owner reasoning 维护 card 的唯一动作，即 `当前 card + 一批 Memories → 新 card`。稳态下那批是 delta——`updated_at` 晚于 card 自身水位的 Memories，含已 retract/supersede 的行；card 没有水位时 delta 就是全部历史，按 budget 分批依次 fold，每批读上一批产出的 card。它不是重建的优化版本，重建是它的边界情形。
- **induction**：启动时及每日本地午夜运行的 orchestration。它扫描 stale Person、Relationship、UserModel projections 与落后的 coverage roots，并调度对应 reasoning；它不是一次 reasoning，也不负责按消息阈值触发 continuity。
- **epistemic review**：owner reasoning 的第二个 call。它读取 evidence Memories、现有 inferred Memories 和 open Hypotheses，显式返回 upsert/retract/transition；遗漏永远是 no-op。User owner 只维护 Hypotheses，不产生 inferred Memories。
- **coverage audit**：UserLearningGoalReasoner 对新增 Memory 做的增量审计，按 root 扇出：每个 root 一次请求，读该 root 的全部 active entries 加这一批新证据。操作面只有 add 与 modify；归属由是哪次请求决定，模型不填分类字段。audit 只忠实总结已知，不负责找盲点。
- **goal planning**：仅在 coverage catch up 后运行的独立 LLM call，同样按 root 扇出。它只读 coverage entries（不读 Memory、不读 Hypothesis），沿纵向、横向、时间纵深和 entry 中出现的人四个方向扩展出候选，再由一次 reconciliation 收口，并维护 LearningGoal lifecycle。找缺口是这里的创造性工作。
- **projection refresh**：一次完整的 projection 更新动作，包括 reasoning、stale/optimistic freshness check，以及成功后的 optimistic writeback。只有写回成功才算 refresh 完成。

profile projection 的标准处理流程：

```text
Message → extraction → Memory → owner reasoning → projection + epistemic actions
                             ↘ coverage audit → CoverageEntries → goal planning
```

## Design terms

以下词汇沿用 `codebase-design` 的含义：

- **Module**：有一个 Interface 和一个 Implementation 的单元。
- **Interface**：调用方正确使用 Module 所需知道的全部约束，不只是类型签名。
- **Implementation**：Module 内部的实现细节。
- **Seam**：Interface 所在、可以替换行为的位置。
- **Adapter**：在 Seam 处满足 Interface 的具体实现。

这里不使用 `API` 泛指 Interface；HTTP API 仅指 transport adapter 暴露的具体 HTTP surface。
