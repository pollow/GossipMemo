---
schema: v1
id: gossipmemo
type: project
status: active
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
asserter    这个说法、判断或立场归属于谁
reporter    谁转述了别人的说法
source      通过什么渠道得知
basis       这是直接表达、观察、转述还是系统归纳
```

候选 tagline：*Remember who said what about whom.*

## Current status

第一版 server、SQLite Adapter、FIFO LLM queue、Python SDK 和 Hermes provider 已实现。下一阶段用合成对话和真实脱敏样本校准 Extract/Reason prompt，并补人物消歧与 merge 工作流。

`Gossip` 是一种需要保留来源和可信度的输入类型，不代表系统把传闻当作事实传播。

## Decisions

- 参考 Monica 的个人关系管理领域模型，但不复制其 UI 或产品边界。
- GossipMemo 作为独立 server；Agent 通过 adapter 接入。
- 原始 Message/evidence 与整理后的 Memory、Person 和 Relationship 分开保存。
- `author`、`subject`、`asserter`、`reporter` 不得合并为单一 speaker/person 字段。
- 任何第三方转述都必须保留 `reported` basis 和原文中的不确定性。
- 不因为普通消息里出现名字就自动创建人物或社交关系。
- Space 是全局长期记忆边界；conversation/thread 只是 Message 来源，不建立 Session 领域实体。
- 不主动建立 Group；多人模式通过 query 指定 Person 集合临时归纳。
- Honcho 可继续作为原始对话和语义证据层；Ada 可作为低频人物/关系抽取 adapter。

## Initial scope

1. 接收 Agent 对话、私聊转述和多人消息，并保留原始来源坐标。
2. 解析人物、别名、Memory 和 Relationship。
3. 支持“事情关于谁”与“谁说/谁转述”的独立归因。
4. 保存有效时间、记录时间、来源 Message、basis 和历史版本。
5. 提供人物 dossier、关系查询、多人临时归纳和证据回溯。
6. 先使用全文检索和结构化过滤，再按需要加入 embedding/图检索。
7. 首个可运行版本支持 manual memory、supersede 和 retract；人物 merge、定时失效与隐私硬删除在真实样例验证后加入。

## Non-goals for the first version

- 不做社交网络或公开 gossip 分享平台。
- 不把 LLM 推断直接当成客观事实。
- 不复刻 Honcho 的完整 peer/observer 语义、dreamer 或通用 SDK。
- 不以“向量相似度最高”替代人物身份、来源和事实状态判断。
- 不在第一版建立 Session、Group、提醒、follow-up 或任务层。

## Next actions

1. 建立 Agent gossip 转述、直接多人消息、同名人物和关系变化的真实脱敏 fixture。
2. 用 fixture 调整 Extract/Reason prompt 的粒度与人物创建阈值。
3. 设计同名人物的人工消歧和 merge 工作流。
4. 增加 SQLite schema migration 与可重建 FTS projection 管理。
5. 从 SocialMemBench 中选择与人物归因、关系和时间变化有关的问题建立回归评测，不以完整 benchmark 为唯一目标。

## Open questions

- 人物实体的创建阈值和同名人物的人工确认流程是什么？
- `reported` 信息是否默认进入 Agent 上下文，还是先进入 review queue？
- 单条 reported Memory 可以进入 profile card 的哪些部分？
- 什么程度的 implicit signal 值得在 Extract 阶段形成 Memory？
- 只有 inferred evidence 时，何时自动创建 Relationship？

## Related knowledge

- [Data schema draft](data_schema.md)：第一版持久数据结构、关联和约束。
- [First-version design](design.md)：处理阶段、reasoning 时机、HTTP endpoints 和功能范围。
- [Monica](https://github.com/monicahq/monica)：个人关系管理领域模型的参考。
- [Graphiti](https://github.com/getzep/graphiti)：时间化实体关系图和 episode provenance 的参考。
- [Hindsight](https://github.com/vectorize-io/hindsight)：world/experience/mental-model 记忆层和混合检索的参考。
- [SocialMemBench](https://arxiv.org/abs/2605.17789)：多人社交记忆的 benchmark 及失败模式参考。
