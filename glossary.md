# GossipMemo Glossary

这份词汇表规定领域文档中的最小术语边界。除非特别说明，泛指长期语义记录时使用 `Memory`，不要用 `fact` 代替它。

## Domain terms

- **Message**：持久化的原始 evidence，记录输入中实际说了什么。它是不可变、durable 的依据，不是语义归纳结果。
- **Memory**：一条 durable semantic claim/record（系统决定长期保留的语义认识）。它可以被检索、修正、retract、supersede 和追溯，但不保证客观真实；`reported`、`inferred` 等 `basis` 必须保留。
- **fact**：`Memory.kind` 的一种静态陈述分类。`fact ⊂ Memory`；它不是 Memory 的同义词，也不表示该记录已被客观验证。
- **projection**：从 canonical data（主要是 active Memories）生成的、可删除并重建的 card 或 context，例如 Person、Relationship、UserModel card 和 rolling continuity。projection 本身不是新的 evidence。

## Processing terms

- **extraction**：从 Message 产生 Memory 的处理阶段，即 `Message → Memory`。批量 LLM extraction 可以先产生 candidate，但对外流程仍按这一语义理解。
- **reasoning**：针对一个对象的一次 LLM computation。Person、Relationship 和 UserModel reasoning 读取相关 Memories；continuity reasoning 读取 prior continuity 和尚未覆盖的 Messages。
- **induction**：启动时及每日本地午夜运行的 profile orchestration。它扫描 stale Person、Relationship 和 UserModel projections，并调度对应的 reasoning；它不是一次 reasoning，也不负责按消息阈值触发 continuity。
- **projection refresh**：一次完整的 projection 更新动作，包括 reasoning、stale/optimistic freshness check，以及成功后的 optimistic writeback。只有写回成功才算 refresh 完成。

profile projection 的标准处理流程：

```text
Message → extraction → Memory → induction scan → reasoning → refreshed projection
```

## Design terms

以下词汇沿用 `codebase-design` 的含义：

- **Module**：有一个 Interface 和一个 Implementation 的单元。
- **Interface**：调用方正确使用 Module 所需知道的全部约束，不只是类型签名。
- **Implementation**：Module 内部的实现细节。
- **Seam**：Interface 所在、可以替换行为的位置。
- **Adapter**：在 Seam 处满足 Interface 的具体实现。

这里不使用 `API` 泛指 Interface；HTTP API 仅指 transport adapter 暴露的具体 HTTP surface。
