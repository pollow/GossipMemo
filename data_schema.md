# GossipMemo 数据结构草案

> Status: draft for review

## 1. 设计目标

GossipMemo 把零散消息整理成两类持续更新的认识：

- 对一个人的认识（`Person`）
- 对两个人之间关系的认识（`Relationship`）

原始消息本身不是长期记忆。系统从消息中挑选、整理出值得长期保留的 `Memory`，再用这些 Memory 更新 Person 和 Relationship 的当前画像。

```text
Session ──contains──> Message ──extract/reconcile──> Memory
                                                        │
                                                        ├──> Person profile
                                                        └──> Relationship profile
```

第一版只保留五个核心领域实体：

```text
Person
Relationship
Session
Message
Memory
```

`Space` 是数据隔离和视角边界，不属于社交记忆领域模型。不同用户、Agent 或互相不可见的记忆空间使用不同的 Space。

## 2. 核心边界

### Message

Message 是不可变的原始输入，回答“当时具体说了什么”。它主要用于回溯和重新抽取，不直接充当人物画像。

### Memory

Memory 是从一条或多条 Message 中整理出来、值得长期保留、可以独立引用和修正的一条记忆卡片。

它回答“系统决定把什么认识带到未来”。Memory 可以是：

- 直接表达的事实或偏好
- 一次值得记住的事件
- 当前计划或处境
- 第三方转述
- 系统根据多次互动形成的印象

Memory 不是每条 Message 的机械摘要，也不是完整的人物或关系画像。

### Person

Person 表示一个经过身份解析的人物，并保存系统对这个人的当前归纳。仅仅在消息中出现一个名字，不自动创建 Person。

### Relationship

Relationship 是两个 Person 之间独立、持续、会演化的关系对象。它不是某条 Memory 的附属物。

Relationship 拥有自己的类型、状态和当前摘要。Memory 可以与 Relationship 关联，成为更新关系画像的材料；Relationship 即使暂时没有关联 Memory，也可以由人工创建并独立存在。

### Session

Session 是一段具有明确参与者和时间边界的共同上下文，例如一次群聊、私聊线程、会议或导入 episode。它回答“哪些人在什么时候共同看到了哪些 Message”。

Session 不能只退化成 Message 上的来源字符串。参与者的加入和离开会影响人物归因、可见范围，以及成员离开后能否正确保留其历史。

## 3. 逻辑数据结构

以下结构表达领域语义，不限定最终使用 SQLite 还是 PostgreSQL。

### 3.1 spaces

```yaml
id: space_01
name: personal
created_at: 2026-08-09T12:00:00Z
```

规则：

- 所有 Person、Relationship、Session、Message 和 Memory 都属于一个 Space。
- 第一版在 Space 内共享可见性，不实现逐条 Memory 的复杂 audience ACL。
- 如果两个 Agent 不应共享记忆，应使用不同 Space。

### 3.2 people

```yaml
id: person_wang
space_id: space_01
display_name: 小王
aliases:
  - 王明
  - 产品小王
profile:
  summary: 小王是产品负责人，做事强调确定性和提前规划。
  traits:
    - 对临时变化比较敏感
  preferences:
    - 重要安排最好提前通知
  current_situation:
    - 正在推动新版产品上线
status: active
merged_into_person_id: null
created_at: 2026-08-09T12:00:00Z
updated_at: 2026-08-09T12:00:00Z
```

建议字段：

| 字段 | 含义 |
| --- | --- |
| `id` | 稳定人物 ID |
| `space_id` | 所属记忆空间 |
| `display_name` | 当前显示名称 |
| `profile` | 当前人物画像，可先用 JSON 保存 |
| `status` | `active \| merged \| deleted` |
| `merged_into_person_id` | 人物合并后的目标 ID |
| `created_at` / `updated_at` | 创建和更新时间 |

别名建议实际存入 `person_aliases` 表，以支持同名、上下文别名和后续身份合并：

```yaml
id: alias_01
person_id: person_wang
value: 产品小王
context_key: group_product
valid_from: null
valid_to: null
```

外部系统 ID 与别名分开保存。平台 ID 是身份解析依据，显示名只是线索：

```yaml
id: identity_01
person_id: person_wang
provider: honcho
external_id: peer_wang_9988
```

建议对 `(space_id, provider, external_id)` 建唯一约束。这样同名人物不会因为名称相似被自动合并，成员离开 Session 后也不会丢失原 Person。

### 3.3 relationships

一个 Relationship 表示 Space 内一对人物之间的整体关系档案。两个人可以同时是朋友、同事和上下级，这些身份作为同一个 Relationship 的多个 facet 保存。

```yaml
id: rel_alice_bob
space_id: space_01
person_a_id: person_alice
person_b_id: person_bob
facets:
  - kind: coworker
    direction: symmetric
    status: active
    since: 2025-03
    until: null
  - kind: friend
    direction: symmetric
    status: active
    since: 2025-11
    until: null
closeness: close
tone: mixed
summary: >
  Alice 与 Bob 合作频繁。近期主要摩擦来自排期和临时需求变更，
  但双方仍愿意配合推进项目。
status: active
started_at: 2025-03
ended_at: null
created_at: 2026-08-09T12:00:00Z
updated_at: 2026-08-09T12:00:00Z
```

建议字段：

| 字段 | 含义 |
| --- | --- |
| `id` | 稳定关系 ID |
| `space_id` | 所属记忆空间 |
| `person_a_id` / `person_b_id` | 关系双方 |
| `facets` | 朋友、同事、家人等可并存的关系类型 |
| `closeness` | 可选的关系接近程度，使用分类而非伪精确分数 |
| `tone` | 可选的当前关系氛围，例如 `positive \| mixed \| tense` |
| `summary` | 对这段关系的当前归纳 |
| `status` | `active \| ended \| unknown` |
| `started_at` / `ended_at` | 已知的关系有效时间 |
| `created_at` / `updated_at` | 创建和更新时间 |

约束：

- 同一 Space 内，同一对人物默认只有一个 Relationship。
- `person_a_id` 和 `person_b_id` 使用稳定排序，避免 A-B 与 B-A 重复。
- facet 可以是对称的，也可以包含 `from_person_id` / `to_person_id` 表达“Alice 是 Bob 的经理”这类方向性身份；整个 Relationship 仍然没有方向。
- 群体关系不纳入第一版；需要时再引入 Group 或多参与者关系。

### 3.4 sessions

```yaml
id: session_product_chat
space_id: space_01
source_system: honcho
source_session_id: external_session_77
kind: group_chat
title: 产品群聊
started_at: 2026-08-01T09:00:00Z
ended_at: null
status: active
metadata: {}
created_at: 2026-08-01T09:00:01Z
```

建议字段：

| 字段 | 含义 |
| --- | --- |
| `id` | 内部稳定 Session ID |
| `space_id` | 所属记忆空间 |
| `source_system` / `source_session_id` | 外部系统中的稳定来源引用 |
| `kind` | `direct_chat \| group_chat \| meeting \| episode \| import` |
| `title` | 可选的人类可读名称 |
| `started_at` / `ended_at` | Session 的时间边界 |
| `status` | `active \| ended \| archived` |
| `metadata` | 来源特有信息 |

参与者及其成员生命周期存入 `session_people`：

```yaml
session_id: session_product_chat
person_id: person_wang
role: member
joined_at: 2026-08-01T09:00:00Z
left_at: 2026-08-07T18:00:00Z
```

规则：

- Session 与 Person 是多对多关系。
- 参与者离开只结束 `session_people` 的成员区间，不删除 Person、历史 Message 或 Memory。
- 谁能观察到一条 Message，第一版由 Message 时间与参与者成员区间推导，不物化完整的 observer-observed 表示。
- 对于无法提供成员变动历史的来源，可把参与者视为覆盖整个 Session。

### 3.5 messages

```yaml
id: msg_456
space_id: space_01
session_id: session_product_chat
source_message_id: external_msg_9988
author_person_id: person_wang
author_raw: 小王
content: 下个月发布前，千万不要再临时改需求了。
sent_at: 2026-08-09T11:30:00Z
ingested_at: 2026-08-09T11:30:05Z
metadata: {}
```

建议字段：

| 字段 | 含义 |
| --- | --- |
| `id` | 内部稳定 ID |
| `space_id` | 所属记忆空间 |
| `session_id` | 所属 Session；来源系统由 Session 提供 |
| `source_message_id` | 外部系统中的消息 ID |
| `author_person_id` | 已解析的作者；无法确认时为空 |
| `author_raw` | 原始作者名称或标识 |
| `content` | 原始消息内容 |
| `sent_at` / `ingested_at` | 消息时间和入库时间 |
| `metadata` | 来源特有信息 |

唯一约束建议使用：

```text
(session_id, source_message_id)
```

### 3.6 memories

```yaml
id: mem_123
space_id: space_01
content: 小王不喜欢在发布前临时修改需求
kind: preference
basis: direct
valid_from: null
valid_to: null
status: active
supersedes_memory_id: null
invalidated_at: null
created_by: extractor
created_at: 2026-08-09T12:00:00Z
updated_at: 2026-08-09T12:00:00Z
```

建议字段：

| 字段 | 含义 |
| --- | --- |
| `id` | 稳定 Memory ID |
| `space_id` | 所属记忆空间 |
| `content` | 可独立理解的规范化自然语言内容 |
| `kind` | 内容类别 |
| `basis` | 信息如何得出 |
| `valid_from` / `valid_to` | 内容描述的现实有效时间 |
| `status` | 当前生命周期状态 |
| `supersedes_memory_id` | 此 Memory 替代的旧 Memory |
| `invalidated_at` | 系统何时停止采用此 Memory；与现实世界的 `valid_to` 分开 |
| `created_by` | `extractor \| consolidator \| human` |
| `created_at` / `updated_at` | 创建和更新时间 |

`kind` 第一版采用小型开放词表，不做严格领域建模：

```text
fact
event
preference
plan
situation
impression
```

`basis` 只描述信息的来源方式：

```text
direct      当事人直接表达或系统直接观察
reported    第三方转述
inferred    系统从一条或多条材料中归纳
manual      人工直接录入
```

`status`：

```text
active
superseded
retracted
expired
```

第一版不加入数值 confidence。诸如“可能”“据说”“尚未确认”等不确定性应保留在 `content` 中，避免一个分数同时混合说话者确定性、来源可信度和抽取准确度。

## 4. 关联表

Memory 与 Message、Person、Relationship、Session 都可以是多对多关系。

### memory_sources

记录产生或支持一条 Memory 的原始消息。

```yaml
memory_id: mem_123
message_id: msg_456
evidence_text: 下个月发布前，千万不要再临时改需求了。
source_role: support
```

一条直接抽取的 Memory 通常只有一个 source；跨多次互动形成的 impression 可以有多个 source。`evidence_text` 保存最小必要原文片段，方便审计抽取结果；`source_role` 第一版使用 `support \| contradict`。

### memory_people

```yaml
memory_id: mem_123
person_id: person_wang
role: subject
```

初始 role 词表：

```text
subject       事情主要关于谁
author        谁说出了来源消息
reporter      谁转述、观察或提供了信息
participant   事件中的其他参与者
```

同一个人可以在一条 Memory 中拥有多个角色。第一版不构建任意深度的转述链。

### memory_relationships

```yaml
memory_id: mem_123
relationship_id: rel_alice_bob
```

该关联表示 Memory 与这段关系有关。它不表示 Relationship 依附于 Memory，也不要求 Relationship 必须拥有 Memory。

### memory_sessions

```yaml
memory_id: mem_group_norm_01
session_id: session_product_chat
```

该关联用于表达群体或 Session 层面的认识，例如“这个群通常先讨论再投票”。个人例外仍创建单独 Memory，并同时关联该 Person 与 Session，避免把群体规范直接套用到每个人身上。

## 5. Memory 的产生与整理

```text
1. 创建或解析 Session，并更新参与者成员区间
2. 保存原始 Message
3. 判断消息中是否有未来仍有价值的内容
4. 生成一个或多个 Memory candidate
5. 解析涉及的 Person 和已有 Relationship
6. 与现有 Memory 对比
7. 执行 create / merge / ignore / supersede
8. 必要时重写 Person profile 或 Relationship summary
```

写入可分为两条节奏：

- 同步 retain：可靠保存 Session、Message 和来源 ID。
- 异步 consolidate：抽取 Memory、跨消息归纳 impression，并刷新人物或关系画像。

异步处理失败不能影响原始消息写入；任务应按 Session 顺序处理，并可根据 Message 重新执行。

### 产生阈值

通常值得形成 Memory：

- 稳定偏好或习惯
- 重要经历和事件
- 当前处境、计划或承诺
- 关系建立、变化或冲突
- 对未来互动有帮助的印象

通常不形成 Memory：

- 寒暄和低信息量对话
- 没有未来价值的即时状态
- 已经存在的重复信息
- 仅仅出现了某个人名

### reconcile 行为

| 行为 | 使用场景 |
| --- | --- |
| `create` | 没有对应的已有记忆 |
| `merge` | 多条信息共同支持同一个认识；补充 source 或改写内容 |
| `ignore` | 重复、无价值或无法可靠解析 |
| `supersede` | 新认识替代旧认识，旧 Memory 保留并标记为 superseded |
| `retract` | 原信息被明确否认或录入错误 |

例子：

```text
旧 Memory：小王对临时变化比较敏感

新 Memory：小王过去排斥临时变化，但最近适应性有所改善

旧.status = superseded
新.supersedes_memory_id = 旧.id
```

## 6. 画像更新

Person.profile 和 Relationship.summary 是面向检索的当前归纳，不是不可变事实。

更新画像时：

- 只使用当前有效且调用者可见的 Memory。
- 可以做概括和判断，不要求逐句复述原消息。
- 应保留重要的不确定性和时间变化。
- 新信息与旧画像冲突时，应重新归纳，而不是机械追加。
- 需要解释时，通过关联表回溯相关 Memory 和 Message。

默认查询顺序：

```text
Person profile / Relationship summary
    ↓ 需要更多细节
相关 active Memories
    ↓ 需要验证或查看原话
source Messages
```

## 7. 参考系统与取舍

### Monica

[Monica](https://github.com/monicahq/monica) 的价值在于 Personal CRM 的产品边界：Contact、联系人之间的 Relationship、Note 和 Activity 分开存在，并强调私有、可控和简单。

GossipMemo 借鉴：

- Relationship 是独立档案，不退化成 Person 属性或 Memory 类型。
- 同一对人物允许多种关系身份。
- 人物和关系都应该有便于人直接阅读和修改的页面级摘要。

暂不借鉴提醒、任务、地址、联系方式等完整 PRM 功能。

### Honcho

[Honcho](https://github.com/plastic-labs/honcho) 使用 Workspace → Peer ↔ Session → Message 作为稳定存储层，再由后台 deriver 产生 peer representation。它还支持基于 `(observer, observed)` 的方向性人物表示。

GossipMemo 借鉴：

- Space、Person、Session、Message 使用相同层次的稳定 ingest 边界。
- Session 与 Person 多对多，保留加入和离开时间。
- 原始写入同步完成，Memory 和画像异步推导。
- Honcho adapter 的直接映射为 `Workspace → Space`、`Peer → person_external_id`、`Session → Session`、`Message → Message`。

第一版不复制 Honcho 的 observer-observed collection、dreaming 或通用 peer 语义。GossipMemo 的 Person 专指社交世界中的人物；Agent、项目或抽象概念不自动建成 Person。

### Graphiti

[Graphiti](https://github.com/getzep/graphiti) 把原始 episode 与推导后的实体/关系分开，并为事实保存现实有效区间和失效历史。

GossipMemo 借鉴：

- Message 是可回溯的原始 evidence。
- Memory 使用 `valid_from / valid_to` 表达现实时间，使用 `created_at / invalidated_at` 表达系统记录时间。
- 新状态替代旧状态时保留 supersede 链，而不是覆盖或删除旧值。
- Person 和 Relationship 的摘要随增量输入持续演化。

第一版不采用图数据库，也不把每条人物认识建成 entity-edge triplet。

### Hindsight

[Hindsight](https://github.com/vectorize-io/hindsight) 区分 raw facts、自动归纳的 observations 和持续刷新的 mental models，并让写入、检索和反思使用不同操作节奏。

GossipMemo 中的对应关系：

```text
Hindsight raw fact     ≈ direct/reported Memory
Hindsight observation  ≈ inferred Memory
Hindsight mental model ≈ Person.profile / Relationship.summary
Hindsight bank         ≈ Space
```

不为这三层分别建表；Memory 的 `basis` 已足够区分直接材料和系统归纳，Profile/summary 作为更高层的持久画像。

### SocialMemBench

[SocialMemBench](https://arxiv.org/abs/2605.17789) 暴露了多人社交记忆的五类结构性失败：说话者与 subject 混淆、时间状态覆盖、人物错误合并、缺失跨人物知识、群体规范覆盖个人例外。

对当前 schema 的直接影响：

- `memory_people.role` 明确保留 subject、author、reporter 和 participant。
- 外部身份 ID 与显示名分离，Person 合并必须显式发生。
- Session 成员离开不改变 Person 生命周期。
- supersede 历史保留旧状态、变化后的状态和变化时间。
- 群体规范使用 session-scoped Memory；个人例外使用 person + session-scoped Memory，不把两者合并成一个摘要事实。
- `memory_sources.evidence_text` 保存最小证据片段，便于构造 attribution、temporal shift 和 relationship 的回归样例。

第一版不物化 `KNOWS_ABOUT` 等 theory-of-mind 边。Session 成员区间和 Message 来源先保留足够信息，等评测证明有必要后再增加派生索引。

## 8. 暂不进入第一版

- 任意深度的 provenance graph
- 数值 confidence 和来源信誉评分
- embedding 或图数据库作为 canonical store
- Event、Claim、Follow-up 的独立实体
- 群体 Relationship
- 逐条 Memory 的复杂 audience ACL
- 持久化的 observer-observed / theory-of-mind 表示
- 自动人格诊断或固定人格标签体系

这些能力后续可以从现有对象和关联中扩展，不需要提前进入核心 schema。

## 9. 待 Review 的问题

1. Person 的 `profile` 应保存为一段文本，还是保存为 `summary / traits / preferences / current_situation` 等结构化 JSON？
2. 同一对人物是否始终只有一个 Relationship，并通过 facets 表达同事、朋友等多重身份？
3. `kind` 是否需要固定枚举，还是完全由应用层使用开放标签？
4. 跨多条消息形成 inferred Memory 时，应该新增 Memory，还是直接更新 Person/Relationship 画像？
5. reported Memory 是否默认参与画像归纳，还是必须先经过人工确认？
6. 第一版是否需要保存 Person profile 和 Relationship summary 的历史版本？
7. Session 成员区间是否足以表达“谁看到过什么”，还是第一版就需要显式 observer 关联？
