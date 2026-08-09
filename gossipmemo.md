---
schema: v1
id: gossipmemo
type: project
status: proposed
title: "GossipMemo"
domain: agent-memory
created: 2026-08-09
updated: 2026-08-09
related: []
sources:
  - https://github.com/monicahq/monica
  - https://github.com/getzep/graphiti
  - https://github.com/vectorize-io/hindsight
  - https://arxiv.org/abs/2605.17789
---

# GossipMemo

> A provenance-aware social memory server for agents.

## Goal

参考 Monica 的 Personal CRM，构建一个独立的 Agent Memory server，使 Agent 不只记住当前用户，还能建立可检索、可校正的个人社交世界模型。系统需要同时支持多人群聊分析和第三方人物/gossip（转述、耳闻、观察、推测）识别。

核心问题是区分：

```text
author      谁说了这句话
subject     这件事关于谁
reporter    谁转述、观察或知道这件事
source      通过什么渠道得知
audience    谁应该能看到
confidence  这是已确认事实、转述、推测还是不确定
```

候选 tagline：*Remember who said what about whom.*

## Current status

这是一个已命名、尚未开始实现的项目提案。独立 server 比单一 Agent 插件更适合后续接入 Hermes、群聊机器人、CLI 和 Web UI，也便于用合成对话和真实脱敏样本测试多人物归因。

`Gossip` 是一种需要保留来源和可信度的输入类型，不代表系统把传闻当作事实传播。

## Decisions

- 参考 Monica 的个人关系管理领域模型，但不复制其 UI 或产品边界。
- GossipMemo 作为独立 server；Agent 通过 adapter 接入。
- 原始消息/evidence 与抽取后的人物、事实、事件、关系分开保存。
- `author`、`subject`、`reporter` 不得合并为单一 speaker/person 字段。
- 任何第三方转述都必须保留 `reported`、`confirmed`、`inferred`、`uncertain` 等认知状态。
- 不因为普通消息里出现名字就自动创建人物或社交关系。
- Honcho 可继续作为原始对话和语义证据层；Ada 可作为低频人物/关系抽取 adapter。

## Initial scope

1. 接收单人和多人 session/episode，并保留原始消息来源。
2. 解析人物、别名、事实、事件、关系和 follow-up。
3. 支持“事情关于谁”与“谁说/谁转述”的独立归因。
4. 保存有效时间、记录时间、来源消息、可信度、可见性和历史版本。
5. 提供人物 dossier、事件时间线、关系查询和证据回溯。
6. 先使用全文检索和结构化过滤，再按需要加入 embedding/图检索。
7. 支持纠正、合并、supersede、失效、删除和人工确认。

## Non-goals for the first version

- 不做社交网络或公开 gossip 分享平台。
- 不把 LLM 推断直接当成客观事实。
- 不复刻 Honcho 的完整 peer/observer 语义、dreamer 或通用 SDK。
- 不以“向量相似度最高”替代人物身份、来源和事实状态判断。

## Next actions

1. Review 并收敛 [数据结构草案](data_schema.md)，明确 Person、Relationship、Session、Message 和 Memory 的边界。
2. 设计 `author/subject/reporter/audience` 归因和 `reported/confirmed/inferred/uncertain` 状态的测试样例。
3. 建立多人群聊、同名人物、关系变化、成员离开和 gossip 转述的合成 fixture。
4. 比较 SQLite/Postgres 作为 canonical store，明确 FTS、embedding 和图索引的可重建边界。
5. 实现最小 HTTP ingest/query/reconcile API，再接入 Ada/Hermes adapter。
6. 参考 SocialMemBench 的问题类型建立可重复的回归评测。

## Open questions

- 人物实体的创建阈值和同名人物的人工确认流程是什么？
- `reported` 信息是否默认进入 Agent 上下文，还是先进入 review queue？
- 不同 Agent/profile 是否拥有不同的 perspective 和 visibility？
- Honcho 原始 evidence 与 GossipMemo claim 如何通过稳定 source ID 关联？
- follow-up、提醒和关系维护是否属于核心 server，还是单独的任务层？

## Related knowledge

- [Data schema draft](data_schema.md)：当前最小数据结构、参考系统取舍和待确认问题。
- [Monica](https://github.com/monicahq/monica)：个人关系管理领域模型的参考。
- [Graphiti](https://github.com/getzep/graphiti)：时间化实体关系图和 episode provenance 的参考。
- [Hindsight](https://github.com/vectorize-io/hindsight)：world/experience/mental-model 记忆层和混合检索的参考。
- [SocialMemBench](https://arxiv.org/abs/2605.17789)：多人社交记忆的 benchmark 及失败模式参考。
