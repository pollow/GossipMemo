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
├── CoverageEntry (recursive per-root coverage summaries per Space)
├── Hypothesis (owned by user / Person / Relationship)
├── LearningGoal (user-owned; may focus Person / Relationship)
├── Message
└── Memory
    ├── Person links
    ├── Relationship links
    └── Message evidence
```

第一版的持久领域记录包括：

```text
Person
Relationship
Message
Memory
Hypothesis
LearningGoal
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
- Hypothesis 是待确认 interpretation，不是 Memory/evidence；LearningGoal 是可选了解方向，不是用户事实。
- Coverage entries 是可重放的 user-learning 累积状态，只由 UserLearningGoalReasoner 消费。
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

## 8. Epistemic state 与 user learning

### hypotheses

Hypothesis 保存“有 evidence、值得讨论、但尚不稳定”的 interpretation。它独立于
inferred Memory，不能被 reasoning 当作 evidence。

| 字段 | 含义 |
| --- | --- |
| `id` / `space_id` | 稳定 ID 与所属 Space |
| `owner_kind` | `user \| person \| relationship` |
| `owner_id` | Person/Relationship owner ID；user owner 为 null |
| `content` / `kind` | tentative interpretation 及语义类型 |
| `confidence` | `low \| medium \| high`，只表达当前校准，不等于事实概率 |
| `status` | `open \| promoted \| rejected \| superseded \| retired` |
| `promoted_memory_id` | promote 时必须指向同 owner 的 active Memory |

`hypothesis_evidence` 保存 support/counter Memory IDs。Upsert/transition 只能引用
reasoner 本次 context 中 server 提供的 owner、Memory 和 Hypothesis IDs；遗漏是 no-op。

### coverage_roots 与 coverage_entries

Coverage 是一张递归的 entry 表。一条 entry 是「我们在这个 path 上了解到什么程度」的
总结（数十条 Memory → 一小段话），不是 Memory 的复述，也不写「还不知道什么」——找缺口
是 goal planning 的工作。

| 字段 | 含义 |
| --- | --- |
| `root` | 20 个稳定 memoir/persona 视角之一；由 audit 的调用结构决定，模型不填 |
| `path` | 自由文本、不规范化；`root` 级 overview entry 的 path 为空 |
| `content` | 该 path 上的认识总结，上限约一两百字，超了应拆 |
| `status` | `active \| superseded` |

Audit 的操作面只有 add 与 modify：合并 = 改写一条 + 把另一条标 `superseded`，
拆分 = 缩小一条 + 新增一条，两步都不需要原子性保证。

`coverage_roots` 每个 Space 每个 root 一行，保存该 root 自己的
`(source_watermark, source_cursor_id)` 增量 cursor 与 CAS `revision`。cursor 按所有
non-inferred Memory 的 `(updated_at, id)` 前进，因此同 timestamp 以及 retract/supersede
都不会漏审；某个 root 的 audit 失败也不会回退其它 root 已推进的进度。

### learning_goals

LearningGoal 属于 Space/current user，即使 `focus_kind` 指向 Person 或 Relationship，也不是
第三方 dossier 任务。

| 字段 | 含义 |
| --- | --- |
| `prompt` | chat agent 可选择如何转化为对话的一个建议措辞 |
| `rationale` | 这是哪个方向、为什么值得了解 |
| `entry_ids` | 该方向所出自的 coverage entry；best-effort，解析不了的 id 被丢弃而 goal 保留 |
| `focus_kind` / `focus_id` | `user`，或同 Space 的 Person；由 store 用确定性别名匹配从 goal 文本推导，模型不填 |
| `status` | `open \| partial \| answered \| deferred \| retired` |

Goal planning 按 root 扇出读 coverage entries（不读 Memory、不读 hypothesis），并用所有
coverage root revision 之和做 optimistic compare-and-swap。私密方向不是天然禁区：planner
忠实维护包括敏感内容在内的未知方向，**现在是否询问、怎样询问由消费 agent 决定**。
deferred 不进入 chat guidance。创建 Goal 本身不提高 coverage。

消费端不对 goal 做相关性排序：在 focus 过滤（`user`，或指向已激活 Person / 相关 Relationship）
之后，随机取 3–5 条随 turn 返回。「现在在聊什么」只有消费 agent 知道，服务端手上只有一个
query 字符串。样本的价值因此完全来自池子的多样性——这正是 planner 按 root 扇出的理由。

## 9. Projection 与 canonical data

数据分为两类：

### Canonical

- Person 身份、aliases 和 external identities
- Relationship 身份；人工确认的关系认识先保存为 manual Memory
- Message
- Memory 及其状态、时间和关联
- Hypothesis 及其显式 lifecycle/evidence links
- LearningGoal 及其显式 lifecycle

### Regenerable projection

- Person.profile_card
- Relationship.facets、summary、closeness、tone 和 status projection
- `user_models` 中每个 Space 一条 `profile_card` projection；仅从 active `about_user` Memory 低频重建，保持 compact、有界且可删除重建
- `coverage_entries` 与 `coverage_roots`：从 non-inferred Memory 增量累积的 coverage 状态（可重放，但重放结果依赖 audit 顺序）
- rolling continuity；每次只读取有界 Message window 并推进 through-message watermark
- 全文、embedding 或图索引

Projection 可以删除并从 active Memory 重建。人工编辑 projection 时，系统应先创建一条 `created_by: human` 的 Memory，再重新生成 projection，避免人工判断只存在于可重建字段中。

## 10. 第一版数据库映射

第一版 canonical store 使用 SQLite：

- JSON projection 和来源 metadata 以 JSON text 保存；
- 时间统一保存为带时区的 ISO 8601 text；
- 全文召回使用可重建的 FTS5 projection；
- 每个写入方法只做一次短原子 apply，schema 不包含数据库任务 queue；
- 一个数据库文件只由一个 GossipMemo server 进程使用。

`WorldStore` 是 implementation 内部 seam。未来 PostgreSQL Adapter 可以使用不同 SQL、JSONB 或索引实现，但必须满足相同的幂等、projection freshness check 和 provenance 行为；具体 freshness watermark 的持久化形式仍待决定。

### schema_migrations：迁移历史

```yaml
version: 2
applied_at: 2026-08-17T00:00:00Z
description: "coverage_maps -> coverage_roots/coverage_entries; learning_goals.criteria_refs/boundary_ids -> entry_ids"
checksum: "<sha256 of \"version:description\">"
```

`schema.sql` 的每次结构变更都对应一行历史记录，由 `src/gossipmemo/migrations.py`
在 `SqliteWorldStore.initialize()` 启动时写入，行只增不改：

| 字段 | 含义 |
| --- | --- |
| `version` | 该行把数据库升级到的 schema 版本；主键，必须与已存在行连续递增 |
| `applied_at` | 迁移执行时间 |
| `description` | 迁移内容的固定文字说明；与 `version` 一起决定 `checksum` |
| `checksum` | 校验值；篡改或伪造的历史行在下次启动时会被拒绝而不是被信任 |

已部署的第一版 schema（`coverage_maps`、`learning_goals.criteria_refs`/
`boundary_ids`）被视为 version 1，即使它从未在自己的数据库里写过
`schema_migrations` 行——迁移器识别出这个遗留结构后回填一条 version 1
基线行，再应用到 version 2。全新创建的空数据库直接标记为当前版本，不回放历史。
迁移永远不自动删除、降级或静默采用无法识别的数据库结构；应用每个升级前先在
挂载数据目录内创建一次性 SQLite 备份，升级在一个写事务内完成，失败即回滚。
完整规则见 README.md 的 "Schema migrations" 一节。

## 11. 示例：转述与直接导入归一

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

## 12. 第一版不包含

- Session 或 Episode 领域实体
- Group、群体画像或群体规范实体
- 任意深度的转述/provenance graph
- 持久化 observer-observed 或 theory-of-mind 模型
- 数值 confidence 和来源信誉评分
- Event、Claim、Follow-up 的独立实体
- 逐条 Memory audience ACL
- 图数据库作为 canonical store
