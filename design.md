# GossipMemo First-Version Design

> Status: first-version draft

本文定义第一版的处理流程、reasoning 时机、模块 interface、HTTP endpoints 和功能范围。持久数据结构见 [data_schema.md](data_schema.md)；术语边界见 [glossary.md](glossary.md)。

## 1. 产品中心

GossipMemo 是一个以使用者为观察原点、沿人物和关系向外扩展的社交世界模型。

主要输入场景：

```text
我和 Agent 对话
  → 告诉它我听到的 gossip
  → 复述我与其他人的讨论
  → 让它持续整理人物和关系
```

批量导入 WhatsApp、Telegram 或 Slack 多人对话是另一种输入方式，不是另一套记忆模型。

核心原则：

```text
Conversations are sources;
people and relationships are the model.
```

- Space 内的长期记忆是全局的，不按 conversation/thread 分割。
- Group 不主动建模；多人模式通过 query 指定 Person 集合临时归纳。
- 系统可以形成对人的判断和定义，但这些判断必须可修正、带时间语境，并能回溯依据。

## 2. 深模块与 seam

第一版由一个对调用方保持简单的 `SocialMemoryWorld` module 承担主要行为。

它的外部 interface 只有三类动作：

```text
ingest(messages)  保存输入并更新长期认识
query(request)    从人物和关系出发检索、扩展和综合
apply(change)     纠正 Memory、合并 Person 或人工补充认识
```

调用方不需要理解 extraction、resolve、owner reasoning、coverage audit、goal planning、projection refresh 或索引维护。这些都是 module implementation。

外部 LLM 是 module implementation 的依赖，放在内部 seam 后：

- production 使用实际模型 adapter；
- 测试使用确定性 fake adapter；
- module 的测试仍通过 `ingest/query/apply` interface 验证最终行为。
- 进程初始化时从环境一次性构造 immutable global `Settings`，再显式分发给各 Module；`GOSSIPMEMO_LLM_BASE_URL`、`GOSSIPMEMO_LLM_MODEL` 或 `GOSSIPMEMO_LLM_API_KEY` 任一缺失都拒绝启动，不提供默认 provider URL 或无模型 fallback。

Canonical store 通过内部 `WorldStore` seam 隔离数据库差异，第一版只实现 SQLite Adapter。这个 interface 表达 `record/read/apply` 等领域行为，不为每张表建立 CRUD repository。全文或 embedding 索引属于独立、可重建的 retrieval projection。

第一版所有 LLM 调用进入进程内的严格 FIFO sequential queue：

- 只有一个 worker，不设置 priority 或并发参数；
- queue 不持久化，不使用数据库任务表、lease 或显式 lock；
- Message 的 extraction state 和 projection timestamp gap 用于进程重启后恢复工作；
- 一个 SQLite 文件只由一个 GossipMemo server 进程打开。

## 3. 处理阶段

```text
Message
  ↓ retain
Message persisted
  ↓ extraction (Message → Memory; ingest returns queued)
Memory
  ├─ owner reasoning: projection call → epistemic review call
  │    └─ refreshed card + explicit inferred/Hypothesis actions
  └─ coverage audit (per root) → CoverageEntries → goal planning (per root)

Context/turn reads select at most one open Hypothesis deterministically and
sample three to five open/partial LearningGoals at random; the request path
does not call an LLM.
```

### 3.1 Retain

职责：

- 幂等保存原始 Message。
- 保留 author、时间和外部 source reference。
- 不把 conversation key 当作 Memory scope。

### 3.2 Extraction

职责：

- 判断输入中是否存在值得长期保留的内容。
- 从 Message 产生零个或多个 Memory candidate。
- 区分 subject、asserter、reporter、witness 和 participant。
- 保留“可能”“据说”“我猜”等不确定性。
- 识别明确的人物身份和关系表达。

Extraction 可以理解语言中的隐含语义，但不做跨历史的长期归纳。例如：

```text
“下个月发布前千万别再临时改需求了”
```

可以抽取为：

```text
说话者不喜欢发布前临时修改需求
```

但“这个人总体上抗拒变化”需要结合历史，应留给 reasoning。

### 3.3 Resolve

职责：

- 用稳定 external identity、alias 和上下文解析 Person。
- 无法可靠解析时保留 unresolved reference。
- 不因同名或 embedding 相似自动合并 Person。
- 只有明确关系表达或具有可说明依据的关系认识才创建 Relationship；共同出现不创建关系。

### 3.4 Persistence reconciliation

职责：

- 将 extraction candidate 与 caller-supplied comparison Memories 比较。
- 执行 `create / ignore / explicit supersede`；不因为一次随机 LLM 输出遗漏旧结果而 retract。
- 维护 source Message 和人物关联；人物在 Memory 中的语义角色保留在 content 和 evidence 中，不单独结构化。
- 计算受影响的 Person 与 Relationship。
- Memory 自身的 `updated_at` 作为对应 Person/Relationship projection 的 freshness 水位。

第一版不设置独立的通用 Reconciliation LLM stage。Person/Relationship owner
reasoner 的第二个 call 负责自己的 inferred Memory 与 Hypothesis lifecycle；所有
retract/transition 都必须显式返回、带 reason，并且只能引用本次 context 中由
server 提供的 ID。User owner 同样维护 Hypothesis，但不产生 inferred Memory。

`ingest` 在 Message 幂等落库并安排 extraction 后返回 `202 queued`。Memory 在后台处理完成后可查询；调用方可以轮询 Message processing state，但不需要理解 queue implementation。

### 3.5 Induction and projection refresh

Induction 扫描 stale projections，并为每个受影响 owner 调度 reasoning；coverage 的
每个 root 各自持有 composite `(updated_at, id)` cursor 并独立增量 catch up，因此某个
root 的 audit 失败不会回退其它 root 已推进的进度。一次 projection refresh 包含
reasoning、optimistic freshness check 和成功后的写回，详见下一节。

Coverage catch up 后，UserLearningGoalReasoner 才执行 goal planning。Coverage entries
只服务这一 reasoner；它们不进入 extraction、Person/Relationship reasoning，也不直接
进入 Hermes prompt——外泄给 agent 的只有它们支持的少量 LearningGoal。

Coverage 不是可重建 projection，而是**可重放的累积状态**。entry 允许被反复改写扩充，
所以重建结果依赖 audit 的执行顺序：删库重跑会得到不同但同样有效的 entries。这是接受的
代价，因为纯证据层（Message 与 Memory）仍然完整、仍可重放，没有不可恢复的东西。
Person/Relationship/UserModel card 和 rolling continuity 则仍是严格意义上的可重建
projection。

### 3.6 Query synthesis

Query synthesis 是只读、按需发生的推理：

- 从一个或多个人物出发读取 profile、Memory 和 Relationship。
- 按需要沿 Relationship 扩展。
- 回答跨人物、跨时间或关系问题。
- 多人共同模式在这里临时归纳，不自动创建 Group。

Query 默认不写入 Memory，也不修改 profile card。需要把 query 结论长期保存时，调用方必须显式通过 `apply(change)` 保存。

## 4. Infer 与 reasoning

### 4.1 两个不同问题

“Infer”容易混淆两类行为：

1. 从一句话理解它表达了什么。
2. 从长期历史归纳这个人或关系是什么样的。

第一类属于 extraction；第二类属于 reasoning。

```text
extraction：Message → Memory
owner reasoning call 1：active evidence Memories → current projection
owner reasoning call 2：same context + projection result → explicit epistemic actions
coverage audit（每 root 一次）：该 root 的 active entries + 新增 evidence → entry add/modify
goal planning（每 root 一次 + 一次 reconciliation）：该 root 的 entries → LearningGoals
```

### 4.2 Reasoning 什么时候触发

Reasoning 不在每条 Message 写入时同步执行。应用每天在本地午夜运行一次 induction，扫描 stale projection 后进入同一个本地 sequential queue；启动时先执行一次 stale catch-up。

触发条件：

- 新增、合并、supersede、retract 或 expire Memory。
- Person 合并或拆分。
- Relationship 被创建、修正或结束。
- 人工要求重新生成指定 Person/Relationship。
- reasoning prompt 或模型版本升级后进行重建。

进程内调度以 Person/Relationship 为 key 避免同时安排重复任务。例如连续导入 20 条关于 Bob 的消息，会在当前 reasoning 完成后检查相关 Memory 的最新 `updated_at` 决定是否重算。LLM queue 本身只保证 FIFO，不实现 priority。

Reasoning 在调用模型前记录相关 Memories 的最新 `updated_at`，模型返回后用同一水位做 optimistic check。如果期间 Memory 已变化，旧结果不写入，直接读取新状态重算；整个 LLM 调用期间不持有数据库 transaction 或 lock。optimistic check 和成功写回共同构成 projection refresh。

### 4.3 Reasoning 做什么

对受影响的 Person：

1. 构建 evidence Memories、当前 card、active inferred Memories 和 open Hypotheses 的同一 context。
2. call 1 仅重建 profile card。
3. call 2 复用 call 1 的完整 transcript/prefix，显式 upsert/retract inferred Memory，或 upsert/transition Hypothesis。
4. omission 是 no-op；只有 optimistic watermark 仍匹配时一起写回。

对受影响的 Relationship：

1. 使用相同的两阶段结构读取 relationship evidence、当前画像、inferred Memories 和 Hypotheses。
2. call 1 更新 facets、closeness、tone、status 和 summary。
3. call 2 显式维护可独立引用的 inferred Memory 与 Hypothesis。
4. 更新 Relationship 的 projection freshness 水位。

UserModel 使用相同 cached-prefix 两阶段结构，但第二阶段只维护 Hypothesis。稳定的
user information 继续由 active `about_user` Memories 支撑 UserModel；不为 user 复制一套
inferred Memory。

### 4.4 什么应该成为 inferred Memory

判断标准：

> 这个结论是否值得拥有独立生命周期，并在 profile card 之外被检索、纠正或引用？

应该保存为 inferred Memory：

```text
Bob 在多个项目中都低估了交付时间，时间估计可能不够稳定。
Alice 和 Bob 最近的主要摩擦来自临时需求变更。
```

只更新 projection、不创建 Memory：

```text
把已有事实重新组织成更流畅的 profile summary。
调整 profile card 的章节顺序和措辞。
```

每条 inferred Memory 必须通过 `memory_derivations` 指向直接依据，不能只存在一个没有来源的模型判断。尚不稳定但值得讨论的 interpretation 应进入 Hypothesis，而不是降格成一个“低置信度 inferred Memory”。

### 4.5 Reported 信息如何参与 reasoning

Reported Memory 可以参与 reasoning，但不能在归纳时丢失其性质：

- 单条 gossip 可以进入 `current_state`，但应保留“据说”“可能”等限定。
- 单条负面 gossip 不应直接固化成稳定 trait。
- 多个独立来源或后续直接 evidence 可以使描述逐渐更确定。
- 人工纠正和 retraction 必须触发画像重建。

第一版不引入数值信誉分；用来源数量、basis、时间和内容一致性提供模型上下文。

### 4.6 防止 reasoning 自我强化

- inferred Memory 必须有非 inferred 的直接依据，或明确引用人工 Memory。
- 一次 reasoning run 新建的 inferred Memory 不在同一次 run 中继续产生更高层 inference。
- Profile card、existing inferred Memory 和 Hypothesis 可以作为 comparison context，但不能当作新 evidence。
- LLM 随机 omission 不触发 soft-retract；retraction 与 Hypothesis transition 必须显式返回 context ID 和 reason。
- Query synthesis 不自动回写。

## 5. HTTP endpoints

HTTP 只是 `SocialMemoryWorld` interface 的 transport adapter。

### 5.1 Turns (write path)

```http
POST /v1/spaces/{space_id}/turns
```

这是唯一的写入端点，合并了原先分开的 `/ingest` 和 `/turns`：两者的写入路径完全相同
（`record_messages` + 触发后台 intake），过去唯一的差别只是 `/turns` 限定单条 user
消息并附带读富化。现在 `messages` 接受 1–100 条、任意 author 顺序的批次：

```json
{
  "messages": [
    {
      "author": "user",
      "content": "Alice 跟我说，Bob 最近可能准备离职。",
      "occurred_at": "2026-08-09T12:00:00Z",
      "source": {
        "provider": "agent_chat",
        "conversation_key": "conversation_456",
        "item_id": "turn_789"
      }
    }
  ],
  "context_version": null,
  "memory_limit": 5
}
```

返回（202）：

```json
{
  "status": "accepted",
  "message_ids": ["message_123"],
  "known_people": [],
  "memory_recall": [],
  "guidance": {"items": []},
  "context_update": null,
  "context_status": "available"
}
```

读富化（alias matching、guidance、FTS recall、context version 比对，即 5.3 描述的
turn facade 行为）当且仅当这批消息的**最后一条**消息 `author == "user"` 时才发生，
用的是那条消息的 content；否则以上字段返回空值，`context_status` 保持
`"available"`。这条规则由数据本身推导（批次最后一条消息的 author），而不是一个
`enrich` 之类的布尔开关——加开关等价于让两个端点换皮共存，违背合并的初衷。

### 5.2 Query

```http
POST /v1/spaces/{space_id}/query
```

单人查询：

```json
{
  "question": "Bob 最近的工作状态怎么样？",
  "people": ["person_bob"],
  "include_evidence": true
}
```

多人临时归纳：

```json
{
  "question": "这些人在做产品决策时分别是什么风格？",
  "people": ["person_alice", "person_bob", "person_carol"],
  "include_relationships": true,
  "expand_relationships": 0,
  "include_evidence": true
}
```

`expand_relationships` 表示从显式指定人物向外扩展的跳数。第一版建议只允许 `0` 或 `1`，防止无界图遍历和上下文污染。

返回包含：

- query-time synthesis；
- 使用的 Person profile cards；
- 使用的 Relationship summaries；
- 相关 active Memories；
- 可选的 Message evidence；
- stale projection 标记。

### 5.3 Context and turn facade

```http
GET  /v1/spaces/{space_id}/context
POST /v1/spaces/{space_id}/turns
```

Context 返回带稳定 version 的 compact UserModel、rolling continuity、continuity-related
Person cards，以及最多一条 user-owned open Hypothesis 和随机 3–5 条 open/partial
LearningGoal。Turn（见 5.1）先持久化整批 Message，当批次最后一条消息来自 user 时，
再用 deterministic alias matching、SQLite FTS 和该条消息文字选择相关 context；
Person/Relationship guidance 只有 owner 被本轮激活时才返回。两条路径都不调用 LLM。

两类 guidance 的选取方式不同，是刻意的。Hypothesis 是关于某个 owner 的断言，激活的
Person 已经把它收窄到当轮主体，取最匹配的一条去确认是有意义的。LearningGoal 是长期
方向而不是断言，「现在在聊什么」只有消费 agent 知道，服务端手上只有一个 query 字符串——
硬做相关性就是用坏输入做难的事。因此服务端不排序，只取样，由 agent 决定用不用。
取样意味着池子的多样性就是样本的全部价值，这正是 goal planning 必须按 root 扇出的理由。
因为 goal 每次读取都重新取样，guidance 不参与 context version 的计算：version 只跟踪
持久 context 状态，否则 warm caller 每轮都会被判定为 stale。

Hermes 必须将两类 guidance 分别标成 tentative understanding 和 optional learning
invitation，不能伪装成 Memory/fact；并且必须明确告诉 agent 这几条 goal 是可默认忽略的
方向、不是要逐条询问的问题，否则并排列出的 3–5 条会把对话变成采访或问卷。

`conversation_key` 只保留为 Message source coordinate。Context version、coverage 和长期
Memory scope 都是 Space 级，不要求 Hermes session key 与历史 import key 同步。

### 5.4 Dossier reads

```http
GET /v1/spaces/{space_id}/people/{person_id}
GET /v1/spaces/{space_id}/relationships/{relationship_id}
```

返回当前 projection、相关 active Memories、关系邻接和 evidence 摘要，不执行开放式 query synthesis。

### 5.5 Corrections

```http
POST /v1/spaces/{space_id}/memories/{memory_id}/supersede
POST /v1/spaces/{space_id}/memories/{memory_id}/retract
POST /v1/spaces/{space_id}/memories
```

- `supersede` 创建新 Memory 并保留旧版本。
- `retract` 标记原认识不再采用。
- 直接创建 Memory 用于人工补充或确认认识。

这些操作完成后只会使相关实体变为 stale；下一次每日 induction 会安排其 reasoning。

### 5.6 后续管理接口

```http
POST /v1/spaces/{space_id}/reason
```

```json
{
  "people": ["person_bob"],
  "relationships": ["relationship_alice_bob"],
  "force": false
}
```

该 endpoint 用于人工刷新、模型升级后的重建和调试。首个可运行版本不暴露它：正常 ingest、manual memory、supersede 和 retract 都会使 projection 变 stale，由每日 induction 处理；进程启动也会扫描 stale projection。Person merge 同样留到有真实同名样例后实现，避免过早固定合并语义。

## 6. 第一版功能

### Ingest 与归因

- 接收单条或一批 Message。
- 支持 Agent 对话中的转述和直接导入的多人消息。
- 区分 author；Memory 涉及的人物及其语义角色保留在自然语言内容和 evidence 中。
- 保留原文 evidence 和外部 source reference。

### 人物与身份

- 创建和查询 Person；同名不自动合并。
- 管理 aliases 和 external identities；aliases 使用独立的 indexed reverse lookup。
- 不因名字相同自动合并。
- 从 ego 沿 Relationship 查询一跳社交网络。

### Memory

- 创建、supersede 和 retract，并保留历史版本。
- 保存现实有效时间和系统记录时间。
- 保存 reported 和 inferred 信息而不把它们伪装成客观事实。
- 支持从 Memory 回溯 Message evidence。

### Person 与 Relationship modeling

- Person profile card 持续更新。
- Relationship 作为独立档案持续更新。
- 允许多重、方向性 relationship facets。
- 关系画像包括 summary、closeness、tone 和当前状态。

### Epistemic guidance 与 user learning

- Person/Relationship reasoner 显式维护 inferred Memory 与 Hypothesis；User reasoner 只维护 Hypothesis。
- Coverage 是一张递归的 entry 表：一条 entry 是某个 path 上「知道什么」的总结，`root` 级 overview 就是 path 为空的那条。20 个稳定 memoir/persona 视角降级为 root 与 prompt 视角提示，不再是 `unknown/fragmentary/grounded/rich` 这样的刻度，也不再有独立的 boundary 记录。
- Coverage audit 按 root 扇出增量更新，操作面只有 add 与 modify；entry 只写「知道什么」，找缺口是 goal planning 的创造性工作。
- Coverage catch up 后单独规划 LearningGoals，同样按 root 扇出，只读 entries，沿纵向、横向、时间纵深和 entry 中出现的人四个方向扩展；Goal 可以 focus Person/Relationship，但始终属于 user/Space。
- Planner 忠实维护包括私密与敏感内容在内的未知方向；现在是否问、怎样问由消费 agent 决定。goal 是可拒绝的邀请，禁止直接病理诊断。
- Hermes 每轮最多消费一条 Hypothesis 和随机 3–5 条 LearningGoal，并被明确告知默认忽略这些方向，不在请求路径增加 LLM latency。

### Query

- 单人 dossier。
- 两人 Relationship 查询。
- 指定多人集合进行临时比较或共同模式归纳。
- 可选的一跳关系扩展。
- 返回结论及 evidence trace。

## 7. 一致性与失败行为

- Message 成功落库并进入本地调度后，`ingest` 返回 `202 queued`。
- 进程崩溃可能丢失内存 queue，但启动时会重新安排 `pending/failed` Message 和 stale projection。
- 每个 persistence 方法只进行短原子写入；调用方看不到 transaction interface，LLM 调用期间不持有 transaction。
- Reasoning 失败不回滚 Message 或 Memory；projection 保持 stale 并可重试。
- owner reasoning 的第二个 call 失败时，projection 和 epistemic actions 都不写入；两个阶段作为一次 optimistic refresh 提交。
- 每个 coverage root 用自己的 composite cursor 防止相同 timestamp 的 Memory 被跳过；retract/supersede 会显式进入 audit，让下一次审阅据此改写受影响的 entry。单个 root 的 audit 失败只让该 root 停在原地重试。
- Query 必须识别 stale projection，并补充读取比 projection 更新的 active Memories。
- 重复 ingest 由 source identity 或 idempotency key 去重。
- Memory retract 和 supersede 由 SQLite Adapter 保证单次 apply 的原子性，但领域流程不依赖跨阶段 transaction。
- 删除原始 Message 时，引用它的 Memory 需要重新评估；隐私硬删除不能仅用 tombstone 代替。

## 8. 第一版不做

- Session、Group 或群体画像。
- 自动扫描所有聊天平台。
- 任意深度 provenance 或 theory-of-mind graph。
- 图数据库作为 canonical store。
- 数值化人物信誉、人格评分或关系分数。
- 自动提醒、任务管理或未经 chat agent 选择就主动追问 LearningGoal。
- query 结果默认回写长期记忆。
- 自动 Person merge、Memory 内容合并和定时 expire job。
- 手工触发 reasoning 的管理 endpoint。

## 9. 仍需通过 fixture 决定

以下问题先保留为可测试决策，不提前扩张 schema：

1. 什么程度的 implicit signal 值得在 extraction 阶段形成 Memory？
2. 只有 inferred evidence 时，何时自动创建 Relationship？
3. 单条 reported Memory 可以进入 profile card 的哪些章节？
4. Profile card 的固定 JSON sections 是否需要允许应用自定义扩展字段？
5. Query 的一跳扩展怎样排序，才能避免带入无关人物？
