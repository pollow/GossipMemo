# GossipMemo Data Schema

> Status: first-version draft

本文只定义 GossipMemo 的持久数据结构、关联和约束。处理阶段、HTTP endpoints 和产品功能见 [design.md](design.md)。

## 1. 模型范围

GossipMemo 保存一个以个人为观察原点、沿人物和关系向外扩展的社交世界模型。

```text
Space
├── Person
├── Relationship (Person ↔ Person)
├── UserModel (one compact projection per Space)
├── Message
└── Memory
    ├── Person links
    ├── Relationship links
    └── Message evidence
```

第一版的持久领域实体只有：

```text
Person
Relationship
Message
Memory
```

`Space` 是数据隔离和观察视角，不是社交实体。当前 User 不是 Person，
也不建立 ego/author-person binding；关于当前 User 的 Memory 使用
`about_user: true` 标记。

核心约束：

- Space 是全局长期记忆边界；对话线程不是 Memory namespace。
- Conversation、thread、channel 只作为 Message 的外部来源坐标。
- Group 不作为第一版实体；多人查询通过一组 Person ID 表达。
- Message author、Memory subject、asserter 和 reporter 不得混为一个字段。
- Person 和 Relationship 的画像是可重建 projection，Memory 和 Message 才是其依据。
- 人物合并必须显式发生，不能仅凭同名或语义相似自动合并。

## 2. spaces

一个 Space 表示一份隔离的全局社交世界。消息作者不会因此成为 Person。

```yaml
id: space_personal
name: My social world
created_at: 2026-08-09T12:00:00Z
updated_at: 2026-08-09T12:00:00Z
```

| 字段 | 含义 |
| --- | --- |
| `id` | 稳定 Space ID |
| `name` | 人类可读名称 |
| `created_at` / `updated_at` | 创建和更新时间 |

第一版在 Space 内共享可见性。需要完全隔离的用户、Agent 或记忆世界使用不同 Space，不实现逐条 Memory ACL。

## 3. people

Person 表示消息内容中被识别出的一个人；`user` 和 `assistant` 消息作者不会自动成为 Person。

```yaml
id: person_bob
space_id: space_personal
display_name: Bob
status: active
merged_into_person_id: null

profile_card:
  summary: Bob 是 Alice 的同事，近期可能在考虑换工作。
  traits:
    - 做决定前倾向先收集信息
  preferences:
    - 重要安排更喜欢提前确认
  current_state:
    - 最近工作状态不太稳定
  interaction_notes:
    - 询问工作变化时不宜把传闻当作已确认事实

profile_source_updated_at: 2026-08-09T12:20:00Z
profile_updated_at: 2026-08-09T12:30:00Z
created_at: 2026-08-01T09:00:00Z
updated_at: 2026-08-09T12:30:00Z
```

| 字段 | 含义 |
| --- | --- |
| `id` | 稳定 Person ID |
| `space_id` | 所属 Space |
| `display_name` | 当前显示名称 |
| `status` | `active \| merged \| deleted` |
| `merged_into_person_id` | 合并后的目标 Person |
| `profile_card` | 当前人物画像 JSON；属于可重建 projection |
| `profile_source_updated_at` | 生成当前 profile 时，相关 Memories 的最新 `updated_at` |
| `profile_updated_at` | 当前 profile card 的生成时间 |
| `created_at` / `updated_at` | 创建和更新时间 |

当任何相关 Memory 的 `updated_at` 与 `profile_source_updated_at` 不一致时，profile 处于 stale 状态。这同时覆盖新增、retract 和 supersede。

### person_aliases

别名是身份解析线索，不是稳定身份。

```yaml
id: alias_bob_01
space_id: space_personal
person_id: person_bob
value: 产品 Bob
normalized_value: 产品 bob
```

`person_aliases` 是按 normalized value 查询的独立 reverse index；Person 的 `display_name` 也应写入其中。相同 normalized alias 可以指向多个 Person，解析时必须报告 ambiguous，不能自动合并。

“妈妈”“老板”“小王”等相对称呼不能被当作全局唯一别名；若未来需要来源上下文，应作为额外解析条件，而不是替代 alias index。

约束：`UNIQUE(person_id, normalized_value)`，并建立 `(space_id, normalized_value)` 索引。

### person_external_identities

外部平台 ID 与别名分开保存。

```yaml
id: identity_bob_telegram
person_id: person_bob
provider: telegram
external_id: user_9988
created_at: 2026-08-01T09:00:00Z
```

约束：

```text
UNIQUE(space_id, provider, external_id)
```

如果来源只提供显示名而没有稳定 ID，应保留 unresolved author/mention，不自动合并到同名 Person。

## 4. relationships

Relationship 是两个 Person 之间独立、持续、会演化的关系档案。它不是 Person 属性，也不依附于某条 Memory。

同一对人物默认只有一个 Relationship；朋友、同事、上下级等多重身份作为 facets 共存。

```yaml
id: relationship_alice_bob
space_id: space_personal
person_a_id: person_alice
person_b_id: person_bob

facets:
  - kind: coworker
    direction: symmetric
    status: active
    valid_from: 2025-03
    valid_to: null
  - kind: manager
    from_person_id: person_alice
    to_person_id: person_bob
    status: active
    valid_from: 2026-01
    valid_to: null

closeness: regular
tone: mixed
summary: >
  Alice 与 Bob 合作频繁。近期在排期和需求变更上有摩擦，
  但双方仍愿意继续配合。
status: active

profile_source_updated_at: 2026-08-09T12:20:00Z
profile_updated_at: 2026-08-09T12:30:00Z
created_at: 2026-08-01T09:00:00Z
updated_at: 2026-08-09T12:30:00Z
```

| 字段 | 含义 |
| --- | --- |
| `id` | 稳定 Relationship ID |
| `space_id` | 所属 Space |
| `person_a_id` / `person_b_id` | 关系双方 |
| `facets` | 可并存、可带方向和有效时间的当前关系身份 projection |
| `closeness` | 可选的粗粒度接近程度，如 `distant \| acquaintance \| regular \| close` |
| `tone` | 可选的当前氛围，如 `positive \| mixed \| tense` |
| `summary` | 当前关系画像；属于可重建 projection |
| `status` | `active \| ended \| unknown` |
| `profile_source_updated_at` | 生成当前关系画像时，相关 Memories 的最新 `updated_at` |
| `profile_updated_at` | 当前关系画像的生成时间 |
| `created_at` / `updated_at` | 创建和更新时间 |

约束：

- `person_a_id` 和 `person_b_id` 使用稳定排序，防止 A-B 与 B-A 重复。
- 同一 Space 内对 `(person_a_id, person_b_id)` 建唯一约束。
- 共同出现在一条 Message 中不会自动创建 Relationship。
- Relationship 可以人工创建，即使暂时没有关联 Memory。

## 5. messages

Message 是 durable、不可变的原始 evidence，回答“输入中实际说了什么”。Agent 对话、私聊转述和批量导入的群聊消息都归一成同一种结构。

```yaml
id: message_123
space_id: space_personal
author: user
content: Alice 跟我说，Bob 最近可能准备离职。
occurred_at: 2026-08-09T12:00:00Z
ingested_at: 2026-08-09T12:00:05Z

source_provider: agent_chat
source_conversation_key: conversation_456
source_item_id: turn_789
source_metadata: {}

extraction_state: pending
extraction_attempts: 0
extracted_at: null
last_extraction_error: null
```

| 字段 | 含义 |
| --- | --- |
| `id` | 内部稳定 Message ID |
| `space_id` | 所属 Space |
| `author` | `user \| assistant`；仅表示消息角色，不关联 Person |
| `content` | 原始消息内容 |
| `occurred_at` | 消息实际发生时间 |
| `ingested_at` | 系统接收时间 |
| `source_provider` | Agent chat、Telegram、Slack、CLI 等来源 |
| `source_conversation_key` | 可选的 thread/channel/conversation 坐标，不是 Memory scope |
| `source_item_id` | 外部系统中的稳定消息 ID |
| `source_metadata` | 来源特有信息 |
| `extraction_state` | `pending \| completed \| failed`；用于恢复本地非持久 queue |
| `extraction_attempts` | 已尝试 Extract 的次数 |
| `extracted_at` | Extract 成功完成时间；即使产生零条 Memory 也必须记录 |
| `last_extraction_error` | 最近一次失败摘要；不保存 secret 或完整 provider payload |

推荐幂等约束：

```text
UNIQUE(space_id, source_provider, COALESCE(source_conversation_key, ''), source_item_id)
```

无法提供稳定外部 ID 的来源应由 ingest 调用方提供 idempotency key。

本地 LLM queue 不持久化。进程启动时重新安排 `pending/failed` Message；因此 extraction state 是处理完成标记，不是数据库任务队列，也没有 lease、owner 或 lock 字段。

## 6. memories

Memory 是系统决定长期保留的一条 durable semantic claim/record。它可以独立检索、修正、失效和追溯，但不保证客观真实；它不是每条 Message 的机械摘要，也不是完整画像。

```yaml
id: memory_bob_job_01
space_id: space_personal
content: Bob 最近可能在考虑离职
kind: situation
basis: reported
status: active

valid_from: 2026-08
valid_to: null
supersedes_memory_id: null
invalidated_at: null
invalidation_reason: null

created_by: extractor
created_at: 2026-08-09T12:00:08Z
updated_at: 2026-08-09T12:00:08Z
```

| 字段 | 含义 |
| --- | --- |
| `id` | 稳定 Memory ID |
| `space_id` | 所属 Space |
| `content` | 可独立理解的规范化自然语言内容 |
| `kind` | 内容类别 |
| `basis` | 该认识如何获得 |
| `about_user` | 是否是关于当前 User 的依据；不建立 Person 关联 |
| `status` | 当前生命周期状态 |
| `valid_from` / `valid_to` | 内容在现实世界中的有效时间 |
| `supersedes_memory_id` | 此 Memory 替代的旧 Memory |
| `invalidated_at` | 系统何时停止采用此 Memory |
| `invalidation_reason` | 人工 retract、supersede 或系统失效的原因 |
| `created_by` | `extractor \| reasoner \| human` |
| `created_at` / `updated_at` | 创建和更新时间 |

`kind` 使用小型开放词表：

```text
fact
event
preference
plan
situation
impression
```

`basis`：

```text
stated      某人直接表达的事实、偏好或立场
observed    某人直接观察到的事情
reported    第三方转述、耳闻或 gossip
inferred    系统综合一条或多条 Memory 形成的认识
manual      人工直接录入
```

`status`：

```text
active
superseded
retracted
expired
```

第一版不加入数值 confidence。诸如“可能”“据说”“尚未确认”等重要限定必须保留在 `content` 中，不能在整理时被改写成确定事实。

## 7. Memory 关联

### memory_people

Memory 与 Person 是多对多关系，只负责索引这条 Memory 涉及哪些人物，不保存人物在事实中的角色。

```yaml
memory_id: memory_bob_job_01
person_id: person_bob
```

Message author 不放入这里；`user` 和 `assistant` 只是消息角色，不是 Person。

Memory 中的 subject、asserter、reporter、witness、participant 等语义直接保留在自然语言 `Memory.content` 和 Message evidence 中；不再定义 `PersonRole` 或 `PersonLink` 概念。

### memory_relationships

```yaml
memory_id: memory_alice_bob_01
relationship_id: relationship_alice_bob
```

该关联表示 Memory 与这段关系有关。Relationship 有独立生命周期，不要求必须由某条 Memory 创建。

### extraction_batches

```yaml
id: batch_123
space_id: space_personal
messages: [message_1, message_2, message_3]
completed_at: 2026-08-09T12:00:10Z
```

一次 LLM Extract request 对应一个 batch。队列默认等待 6 条 Message；未满时，
从最老一条消息入库开始等待最多 30 分钟，然后提交当前 partial batch。
进程重启后根据持久化的 `ingested_at` 恢复剩余等待时间。
Message 通过 `extraction_batch_id` 记录处理批次，Extract 产生的 Memory 通过
`source_batch_id` 指向同一批次。查询证据时沿 Batch 展开其中的原始 Message。

Relationship 由 Memory 创建或更新，因此通过 `memory_relationships` 间接继承同一来源，暂不重复保存 batch ID。

### memory_derivations

inferred Memory 与其直接依据的既有 Memory 通过单独关联保存。

```yaml
derived_memory_id: memory_bob_planning_impression
source_memory_id: memory_bob_delay_01
derivation_role: support
```

`derivation_role` 第一版使用 `support \| contradict`。这不是通用 provenance graph，只保存一层直接依据，防止 reasoning 结论失去可解释性。

## 8. Projection 与 canonical data

数据分为两类：

### Canonical

- Person 身份、aliases 和 external identities
- Relationship 身份；人工确认的关系认识先保存为 manual Memory
- Message
- Memory 及其状态、时间和关联

### Regenerable projection

- Person.profile_card
- Relationship.facets、summary、closeness、tone 和 status projection
- `user_models` 中每个 Space 一条 `profile_card` projection；仅从 active `about_user` Memory 低频重建，保持 compact、有界且可删除重建
- 全文、embedding 或图索引

Projection 可以删除并从 active Memory 重建。人工编辑 projection 时，系统应先创建一条 `created_by: human` 的 Memory，再重新生成 projection，避免人工判断只存在于可重建字段中。

## 9. 第一版数据库映射

第一版 canonical store 使用 SQLite：

- JSON projection 和来源 metadata 以 JSON text 保存；
- 时间统一保存为带时区的 ISO 8601 text；
- 全文召回使用可重建的 FTS5 projection；
- 每个写入方法只做一次短原子 apply，schema 不包含数据库任务 queue；
- 一个数据库文件只由一个 GossipMemo server 进程使用。

`WorldStore` 是 implementation 内部 seam。未来 PostgreSQL Adapter 可以使用不同 SQL、JSONB 或索引实现，但必须满足相同的幂等、projection freshness check 和 provenance 行为；具体 freshness watermark 的持久化形式仍待决定。

## 10. 示例：转述与直接导入归一

### 通过 Agent 转述

```text
我 → Agent：Alice 跟我说，Bob 最近可能准备离职。
```

```yaml
message.author: user

memory.content: Bob 最近可能在考虑离职
memory.basis: reported
memory.people:
  - person_bob
  - person_alice
```

### 直接导入群聊

```text
Alice：Bob 最近可能准备离职。
```

```yaml
message.author: user

memory.content: Bob 最近可能在考虑离职
memory.basis: stated
memory.people:
  - person_bob
  - person_alice
```

两种输入得到相同的 Memory；转述来源等语义保留在自然语言 `content` 和 Message evidence 中。Conversation 不是长期记忆边界。

## 11. 第一版不包含

- Session 或 Episode 领域实体
- Group、群体画像或群体规范实体
- 任意深度的转述/provenance graph
- 持久化 observer-observed 或 theory-of-mind 模型
- 数值 confidence 和来源信誉评分
- Event、Claim、Follow-up 的独立实体
- 逐条 Memory audience ACL
- 图数据库作为 canonical store
