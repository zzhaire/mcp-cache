# M1–M4 缓存调度算法设计稿（v0 草案）

> **2026-11-21 更新：默认预设已按设计稿《MCP-Cache Design》对齐。** 默认不设容量上限、不淘汰、不做闲置遗忘（`capacity.total_bytes`、`max_entries`、`max_entry_bytes`、`forget.idle_steps` 均为 0，`forget.on_turn_end` 为 keep）；`publish.mode` 默认为 `key_values`；`publish.large_repeat_min_bytes` 默认为 0。下文里与此不同的默认值以 `configs/scheduler-presets.json` 为准。对应关系见 [summary.md](summary.md) 的"与设计稿的对应"一节。

更新：2026-11-19。状态：**v0 已实现并通过机制测试，尚未做真实模型评测**（实现说明见 §10）。本文只给出机制和可调参数，不含任何性能数字。按 Supervisor-Skills 的 Vibe Research 准则，研究设计与技术路线由作者决定。文中标 **【待定】** 的位置需要作者拍板，草案只给出推荐值和理由。

配套文件：
- 参数 schema：[ours/scheduler.schema.json](../ours/scheduler.schema.json)
- 默认参数与场景预设：[configs/scheduler-presets.json](../configs/scheduler-presets.json)
- 实现：[exp_core/scheduler.py](../exp_core/scheduler.py)、[exp_core/scheduler_policy.py](../exp_core/scheduler_policy.py)、契约补充 [exp_core/contracts/scheduler.json](../exp_core/contracts/scheduler.json)
- 测试：[tests/test_scheduler.py](../tests/test_scheduler.py)

---

## 0. 与论文挑战的对应（tech-paper-template 自洽检查）

| 论文挑战（paper.md §1） | 本设计中的模块 | 小节 |
|---|---|---|
| C1：四类状态有效性条件不同且互相影响 | 统一条目模型 + 依赖标签失效图 + 跨模块联动规则 | §2、§3 |
| C2：AI 提出的策略修改必须可执行、保持约束，并用执行反馈评价 | 分层参数空间：安全锁定项不可调，可调项有范围；schema 校验后才执行 | §6、§7 |
| C3：在未参与优化的场景中保留收益，开销不抵消节省 | 场景预设 + 场景特征 + 开销计入目标 + 留出场景评测 | §7、§8 |

自洽检查结果：每个挑战对应一个模块；每个模块对应 §8 中的消融项。

---

## 1. 当前原型的问题（来自代码，不是实验结论）

读 [exp_core/runtime.py](../exp_core/runtime.py)、[exp_core/access.py](../exp_core/access.py) 和 [ours/publication.py](../ours/publication.py) 得到的结构性限制：

| # | 现状 | 后果 |
|---|---|---|
| P1 | M2/M4 共用一个全局 `ttl` | 静态数据（目录、文档）和易变数据（余额、队列）只能取同一个期限 |
| P2 | 任何写操作都会让 `data_epoch += 1` 并调用 `results.clear()` | 写 A 应用也会清掉 B 应用的读取结果。这是 M2 命中极低（首轮 0/1/1/0）的可能原因之一，需要实验确认 |
| P3 | 授权变化会同时清空 M1、M2、M3 | 换一个账号登录也会丢掉与授权无关的公共目录 |
| P4 | 没有容量上限和淘汰策略 | 不能研究"开多大空间"；长任务内存无界 |
| P5 | 每个任务都新建 `Runtime` | M1 不能跨任务复用，`rw_openai` 的目录缓存因此没有真实命中（paper §4.1.4） |
| P6 | 状态提供按 identity 字典序截断预算 | 预算用在哪些条目上与价值无关 |
| P7 | 时间都用墙钟秒 | 智能体的大部分时间花在模型推理上。"60 s"对应的步数随模型速度变化，参数难以跨模型迁移 |
| P8 | M3 只做精确 scope 匹配，不会提前刷新 | 无法表达"权限超集可复用"和"快过期时预刷新" |
| P9 | 重试规则写死为 `max_attempts=2` | 不同错误类别（限流、授权失效、业务错误、未知写结果）没有各自的恢复动作 |

---

## 2. 统一条目模型

M1–M4 共用一个条目结构，由调度器统一管理有效性、容量和淘汰。

```text
Entry {
  module        : M1 | M2 | M3 | M4
  key           : 规范化键（见下表）
  value_ref     : 内容寻址引用 sha256(value)，值存放在共享 ValueStore（相同结果只存一份）
  size_bytes    : 序列化大小
  principal     : 授权主体（service, subject, scope_set），与授权无关时为 null
  tags          : 依赖标签集合，例如 {"srv:appworld", "app:spotify", "res:spotify/playlist/42"}
  fresh_class   : static | slow | volatile | strict          （M2 使用）
  validity      : { expires_at, expires_at_step, soft_expires_at, catalog_hash, version }
  cost          : { fetch_latency_ms, response_bytes, est_tokens }   （回源一次的代价，用于淘汰）
  stats         : { hits, last_access_step, created_step, freq }
  pinned        : bool  （M4 的 running/uncertain 条目、M3 的有效凭据不可被淘汰）
  tier          : task | session | host
}
```

| 模块 | 键 | 主要有效性依据 | 默认存储层级 |
|---|---|---|---|
| M1 目录 | `(server, catalog_hash或version, principal_scope?)` | 目录版本 / `notifications/tools/list_changed` / 授权范围 | session |
| M2 读取 | `(scope, tool, canonical_args, principal)` | 标签未被失效、未过期、目录 schema 未变 | task |
| M3 凭据 | `(service, subject, scope_set)` | `valid_until`、撤销事件 | session |
| M4 操作 | `operation_id`（同一逻辑操作的内部重试沿用同一个 id） | `identity_sha256`、授权/数据条件、状态机 | task |

**层级说明**：`task` 表示任务结束即清空；`session` 表示同一 Host 会话内的多个任务共享；`host` 表示跨会话持久化。**M2 默认不跨任务**：每个评测任务都从同一个初态开始，跨任务复用读取结果会把上一个任务的数据带进来，破坏评测独立性。

---

## 3. 有效性与失效（解决 C1）

### 3.1 新鲜度分级（替代全局 TTL，解决 P1、P7）

| 级别 | 含义 | 默认期限 | 例子 |
|---|---|---|---|
| `static` | 任务内不会变化 | 到任务结束 / 版本变化为止 | API 文档、`tools/list`、歌曲元数据 |
| `slow` | 很少变化，只因本任务的写操作而变 | `ttl.slow` | 个人资料、联系人、播放列表内容 |
| `volatile` | 可能被外部因素改变 | `ttl.volatile`（较短） | 余额、通知、当前时间、播放队列 |
| `strict` | 每次都必须回源 | 0（不缓存） | 支付确认前的余额校验、`get_current_date_and_time` |

级别来源依次为：工具契约中的声明 → `fresh_class_overrides` 参数 → MCP `readOnlyHint` 推断为 `slow` → 都没有则不缓存（与现在"未声明不缓存"的保守规则一致）。

**期限单位【待定】**：`ttl.unit ∈ {seconds, steps, turns}`。推荐 **steps**（决策步数），理由见 P7：步数不受模型速度影响，换模型、换链路时参数更好迁移。同时保留秒级的 `ttl.wall_clock_cap_s` 作为上限，防止很长的步骤导致数据过旧。

### 3.2 依赖标签失效（替代全局 data_epoch，解决 P2）

- 每个读取条目带标签集合，由契约声明，例如 `show_playlist(42)` 带 `{app:spotify, res:spotify/playlist/42}`。
- 每个写操作声明它影响的标签，例如 `add_song_to_playlist(42)` 影响 `{res:spotify/playlist/42}`。
- 写成功后，失效所有与影响标签相交的 M2 条目。
- **失效粒度**参数 `invalidation.granularity ∈ {global, server, app, resource}`：
  - `global` 等价于当前行为，用作消融的基线；
  - 如果写操作没有声明影响标签，自动退回到 `max(当前粒度, app)`。宁可多失效，也不能漏失效。
- **未知结果的写**（`outcome=unknown`）：开启 `quarantine_on_unknown_write` 时，失效范围至少扩大到 app 粒度（未知的写可能产生部分副作用）；同一操作在对账前禁止重放（见 §5）。

### 3.3 跨模块联动规则（体现"联合管理"，消融时可逐条关闭）

所有联动都通过同一套版本化标签实现：授权变化提升 `auth:<主体>`（或全局 `auth*`）标签，目录变化提升 `tool:<server>/<工具>` 或 `srv:<server>` 标签。因此 M1、M2、M3、M4 中依赖同一标签的条目会一起失效，不需要逐对写规则。

| 触发事件 | M1 | M2 | M3 | M4 |
|---|---|---|---|---|
| 写成功 | 不变 | 按标签失效 | 不变 | 记为 completed；依赖被写标签的失败记录不再视为"相同条件" |
| 写结果未知 | 不变 | 按标签失效（隔离时至少 app 粒度） | 不变 | 记为 uncertain，固定（pinned），对账前禁止重放 |
| 登录 / 凭据更新 | 不变 | 规则 `auth_upgrade_keeps_m2` 开启时保留；关闭时失效该主体的条目 | 写入新凭据 | 不变 |
| 登出 / 撤销 / 改密 | `partition_by_principal` 开启时只失效该主体的目录，关闭时全部失效；公开目录是否失效由 `catalog_auth_scoped` 决定 | 同左 | 同左 | 该主体下的写操作因授权变化被拒绝重放；失败记录不再视为"相同条件" |
| 目录版本变化 | 替换 | 规则 `catalog_change_scoped` 开启时只失效 schema 改变的工具，关闭时失效整个 server | 不变 | 同左 |
| 凭据快过期 | 不变 | 不变 | 规则 `credential_refresh_ahead` 开启且注册了刷新器时提前刷新 | 不变 |
| 请求的权限是已有凭据的子集 | — | — | 规则 `credential_scope_superset` 开启时复用 | — |

这张表就是论文 C1 中"凭据变化既可能影响目录，也可能改变失败操作的恢复条件"的具体化。**每条规则关闭时都退回更保守的行为**（多失效、少复用），不会变得不安全；因此可以逐条关闭做消融，回答"联合比各自独立配置多出多少收益"。

---

## 4. 空间与淘汰（解决 P4，回答"开多大"）

### 4.1 预算

```text
capacity:
  total_bytes          全局字节上限
  share: {M1, M2, M3, M4}   各模块份额（和为 1）；某个模块用不完的份额可借给其他模块（borrow=true）
  max_entries: {M1..M4}     条目数上限
  max_entry_bytes           单条上限，超过则不缓存（防止一个大结果挤掉所有条目）
```

### 4.2 准入（admission）

满足以下全部条件才写入：
1. 契约允许缓存，且 `fresh_class ≠ strict`；
2. `size ≤ max_entry_bytes`；
3. 如果 `admission.min_refetch_cost_ms > 0`，回源代价必须高于这个阈值，避免为很便宜的读取占用空间。

### 4.3 淘汰（eviction）

`eviction.policy ∈ {lru, lfu, gdsf}`，推荐 **gdsf**（GreedyDual-Size-Frequency 的代价感知变体）：

```text
refetch_cost(e) = w_lat · fetch_latency_ms + w_bytes · response_bytes + w_tok · est_tokens
priority(e)     = L + freq(e)^φ · refetch_cost(e) / size_bytes(e)
淘汰 priority 最小且未固定的条目；L 更新为被淘汰条目的 priority（老化）
```

- `w_tok` 把"模型重新读一遍这个结果的 Token 代价"计入回源代价，对应论文中的"模型决策成本"。
- 固定条目不参与淘汰：M4 中 `running/uncertain` 的条目、已完成的写操作记录（防止同一 op_id 被重复提交），以及 M3 中仍有效的凭据。
- 份额与借用：`borrow=false` 时每个模块不得超过 `total_bytes × share`；`borrow=true` 时只在总量超限时淘汰，并优先从超出自身份额的模块中淘汰。
- 上限参数为 0 表示不限（`ablation_current` 使用）。
- 推荐值：`lru` 用于消融基线；`gdsf` 用于 Ours。

### 4.4 遗忘（主动过期，独立于容量）

| 参数 | 作用 |
|---|---|
| `ttl.{static, slow, volatile}` | 硬过期；`static` 为 null 时到任务结束。M1 目录不使用 TTL，只按版本和授权判断 |
| `forget.idle_steps` | 连续 N 步未被访问就丢弃，即使还没过期，也不再提供给模型 |
| `forget.on_turn_end ∈ {keep, drop_volatile, drop_all_M2}` | 用户 turn 边界的处理。BFCL 多轮任务中，一个用户 turn 可能改变外部状态 |
| `ttl.m4_completed` | 已完成的只读操作按 op_id 复用结果的期限 |

---

## 5. M4 恢复策略（解决 P9）

按错误类别给出动作。**重试上限 `retry.max_attempts` 在所有方法间共用**（paper §4.1.2 的公平性要求），锁定，不交给 RSI 调。RSI 只能调"怎么用这几次机会"。

| 错误类别 | 判定依据 | 动作（参数） |
|---|---|---|
| `not_executed` + 可重试（限流、超时、连接失败） | 契约中的 `outcome` | 指数退避：`retry.backoff_base_ms × factor^k`，上限 `retry.backoff_cap_ms` |
| 授权失败 | `error_code ∈ auth_codes` | `recovery.auth = refresh_then_retry_once \| surface_only`。刷新依赖 M3 |
| 业务错误（参数非法等） | `business_error` | 不重试相同参数；只把失败记录提供给模型（这就是"失败状态复用"） |
| 写结果未知 | `outcome=unknown` | `recovery.unknown_write = reconcile_read \| surface_only`。前者调用契约声明的对账读取，确认已执行则记为 completed，确认未执行才允许重试 |
| 相同条件的重复失败 | M4 命中 | 直接返回记录的失败（不回源）。`recovery.failure_index=identity` 时按"工具 + 参数"跨调用索引，只复用不可重试、确认未执行的失败。**这条很关键**：runner 每次模型调用都会生成新的 op_id，旧实现只按 op_id 索引，M4 失败复用因此几乎不可能跨调用命中 |

---

## 6. 调度主流程（伪代码）

```text
on_tool_call(tool, args, op_id):
    entry_kind = classify(tool)               # M1 目录 / M3 登录 / M2 读 / 写
    if entry_kind == M3: return credential_flow()
    if op_id in M4: r = m4_check(op_id)       # 完成→复用；uncertain→阻断；同条件失败→复用失败
                    if r: return r
    if readonly and cacheable(tool):
        e = lookup(key)
        if e and valid(e):                    # 标签未失效 ∧ 未过期 ∧ 主体匹配 ∧ schema 未变 ∧ 不在隔离中
            touch(e); return e.value
        if e and soft_expired(e) and allow_revalidate:
            return revalidate(e)              # 调用契约声明的轻量校验读取；没有则回源
    v = fetch_with_recovery(tool, args, op_id)    # §5
    if readonly: admit(v) → maybe_evict()        # §4
    else:        propagate_invalidation(write_tags, outcome)   # §3.2 / §3.3
    record_m4(op_id, outcome)
    return v

before_each_decision_step():
    forget_idle(); maybe_refresh_ahead_M3()
    provision = select_for_publication(budget)   # §6.1
    return provision
```

### 6.1 决策前状态提供（解决 P6）

这一步是一个 0-1 背包：在 `publish.budget_bytes` 内选出要提供给模型的条目。

- 候选：当前有效、且**已在对话中出现过**的 M2 结果（沿用现在"只提供引用、不注入未见过的结果"的原则），以及 M4 中 failed/uncertain 的记录。
- 顺序：读取引用沿用 runner 的"最近读取优先"；`publish.ops_first` 决定失败/未知操作记录是否先占用预算。v0 没有实现按价值打分的背包，留待有数据后再定。
- 形式 `publish.mode ∈ {ref_only, full, hybrid}`：`hybrid` 对大于 `publish.full_max_bytes` 的结果只给引用。
- `publish.max_items` 限制条数，防止长任务中提供的内容本身变成 Token 负担。首轮 smoke 中 Ours 的平均每轮输入增加了 12.30%，这里要特别注意。
- `context_dedup`：同一结果再次出现时用引用替换。保留为开关，因为选参发现它影响正确性。
- 2026-11-20：易变类的默认有效期从 4 步改为 0，即易变数据默认不缓存。原因是复用几步会漏掉同一轮内的外部变化；在 40 条开发 case 上改成 0 只多 1 次请求。需要缓存易变数据的场景可以用预设或自动定参调回去。
- 2026-11-20 新增的三处行为（由 lab 服务器的新 case 查出）：
  - `invalidation.on_refused_write`（默认开）：写操作被业务原因拒绝时，它声明会影响的资源上的缓存读取一并失效。
  - `recovery.failure_ages`（默认开）和 `recovery.failure_ttl`（默认 4 步）：记住的业务失败过了这段时间就不再复用，同样的请求会重新发给服务器。
  - 不可关闭的两条：一次写只要有过结果未知的尝试，后续尝试被拒绝也不改变结论，交给代理的是"结果未知"；凭据按自己的有效期到期时，该账号名下的缓存读取失效。
  - 新预设 `fixed_rules`：能力对等的强基线用的固定规则，见 `docs/lab-server.md`。
- `large_repeat_min_bytes`（默认 4096，0 为关闭）：只针对大结果的去重。结果不小于这个字节数、且与本次对话里更早的某个结果逐字节相同时，只给模型一个指向那次调用的引用。小结果仍然原样给出。加这一项的原因是真实模型评测里发现：缓存命中不花请求，但结果仍会整份写进上下文，大结果被反复命中时 token 反而比无缓存多。复刻旧原型的预设里它是 0。

---

## 7. 参数空间与 RSI（解决 C2、C3）

### 7.1 三层参数

| 层 | 内容 | 谁能改 |
|---|---|---|
| **L0 安全锁定** | `retry.max_attempts`、`strict` 类永不缓存、uncertain 写永不重放、未声明标签的写退回到 ≥app 粒度、M2 不跨任务 | 只能由作者修改；RSI 修改会被 schema 拒绝 |
| **L1 结构开关** | `invalidation.granularity`、`eviction.policy`、`coupling.enabled_rules`、`publish.mode`、各层级 tier、`recovery.*` 模式 | RSI 可调，但主要用于消融 |
| **L2 数值** | 各级 TTL、容量与份额、`idle_steps`、gdsf 权重、发布预算与条数、退避参数 | RSI 的主要搜索对象 |

完整定义见 [ours/scheduler.schema.json](../ours/scheduler.schema.json)。每个参数都带 `x-tier` 标注（L0/L1/L2），RSI 只能生成 L1/L2 的覆盖值，并且必须通过 schema 校验后才执行。

### 7.2 场景预设（起点，不是结论）

见 [configs/scheduler-presets.json](../configs/scheduler-presets.json)：

| 预设 | 面向的场景特征 | 主要倾向 |
|---|---|---|
| `default` | 不知道场景时使用 | 中等 TTL、resource 粒度、gdsf |
| `read_heavy` | 读写比高、重复读取多 | 更长的 slow TTL、更大的 M2 份额、hybrid 提供 |
| `write_heavy` | 写多、读写交错 | 短 TTL、resource 粒度、更多 M4 份额、开启对账 |
| `auth_churn` | 频繁登录、切换账号 | 按主体分区、开启 refresh_ahead、目录按授权范围区分 |
| `catalog_heavy` | 多个 Server、目录大 | M1 使用 session 层级、更大的 M1 份额、`max_entry_bytes` 放宽 |
| `high_latency` | RTT 高、回源贵 | gdsf 中 `w_lat` 调高、允许 revalidate、降低准入门槛 |
| `ablation_current` | 复刻当前原型的行为 | global 粒度、所有级别 TTL 都是 120 s、无容量限制、lru、按 op_id 记录失败、全部为任务级。已与旧 runtime 做随机序列对拍（§10） |

### 7.3 RSI 适配流程【待定：搜索器选择】

1. **场景特征提取**（从开发 trace 中统计）：读写比、重复读取率、写影响标签的分布、授权事件频率、目录大小、平均回源延迟、平均结果大小。
2. **选起点**：按特征选预设（规则映射，或者最近邻）。
3. **搜索**：在 L2 上做坐标下降或小规模贝叶斯优化；L1 只在预设之间切换。候选必须先通过 schema 校验和机制单测（§8 的机制核验），才进入真实执行。
4. **目标**：保持当前"正确完成数优先"的规则，然后比较归一化成本。成本项在现有三项（Active time、输入 Token、MCP 请求）之外，**增加调度器自身开销**（状态提供字节、调度计算时间）；搜索成本单独报告。
5. **防过拟合**：每个场景的开发 case 和留出 case 分开。在场景 A 上调好的参数要在场景 B 上测试，回答论文 C3。

后续如果要"模型生成策略"，就让模型输出符合同一 schema 的覆盖 JSON。可执行性和约束由 schema 与 L0 锁保证，这一点与离线参数搜索共用同一套执行和评分。

---

## 8. 对 MCP 实验服务器与 case 构造的要求（下一阶段）

调度器需要服务器提供以下"可控因素"，才能逐项验证机制、复现相关工作：

| 服务器能力 | 用来验证 | 对应相关工作 |
|---|---|---|
| 工具注解：`readOnlyHint`、`idempotentHint`、自定义 `x-fresh-class`、`x-tags`、`x-affects` | §3.1、§3.2 | OpenAI Hint |
| `tools/list` 版本号与 `notifications/tools/list_changed`，可以在任务中途改变目录 | M1 失效、联动规则 | OpenAI `cache_tools_list`/`invalidate_tools_cache` |
| 按主体返回不同目录 | `catalog_auth_scoped` | OpenAI `tool_filter`、FastMCP OAuth |
| 有过期时间的 token、刷新、撤销 | M3、refresh_ahead | FastMCP 客户端 OAuth |
| 可开关的服务端响应缓存 | 与 Host 侧缓存对照 | FastMCP ResponseCachingMiddleware |
| 注入延迟分布、限流（429 + retry_after）、超时、"写成功但响应丢失" | §4 gdsf、§5 恢复 | Mehan 失败可操作性 |
| 外部变更注入（后台修改 volatile 数据） | `volatile` TTL 与正确性的权衡 | Merchant 时间感知缓存 |
| 多个 Server 和跨服务器依赖 | 多个 Server 时 M1、并行 | Merchant 跨服务器并行、Gateway 路由 |
| 对账读取接口（按 op_id 查询写状态） | `reconcile_read` | — |

case 分族建议：每族包含**机制核验 case**（确定性、无模型，检查命中、失效是否正确）和**端到端 case**（真实 Agent）。论文 §4.3 已要求"机制行为正确与端到端收益分别验证"。

| case 族 | 关键变量 |
|---|---|
| 重复读取 | 重复间隔（步数）、结果大小 |
| 读写交错 | 写影响的标签与后续读取的重叠率（0 / 部分 / 全部） |
| 授权变化 | 中途过期、切换账号、撤销 |
| 目录变化 | 中途新增或删除工具、schema 改变 |
| 故障恢复 | 限流、超时、未知写结果 |
| 外部变化 | volatile 数据被后台修改的频率 |
| 容量压力 | 结果总量与容量之比 |

---

## 9. 待作者决定的事项

| # | 问题 | 草案推荐 | 理由 |
|---|---|---|---|
| D1 | TTL 单位 | steps，加墙钟上限 | 跨模型可迁移（P7） |
| D2 | 失效粒度依赖契约中的标签。AppWorld/BFCL/τ² 的标签由谁标注？ | 先做 app 级标注（低成本），再给新建的 MCP 服务器做 resource 级标注 | 现有数据集逐工具标 resource 的工作量大 |
| D3 | M1/M3 是否跨任务（session 层级）？ | 是，但在报告中单列"跨任务复用"的收益 | 不跨任务的话，OpenAI 目录缓存基线本身就没有收益可测 |
| D4 | 是否允许"软过期后校验"（revalidate）？ | 只对 slow 类、且有声明的校验读取时允许 | MCP 没有 ETag 机制，需要服务器提供 |
| D5 | RSI 搜索器 | 先用坐标下降（可解释、便于写入论文），之后再考虑模型生成 | 与现有的两轮离线搜索衔接 |
| D6 | 调度器开销是否进入 RSI 目标 | 是 | C3 要求开销不能抵消节省 |

---

## 10. v0 实现说明（2026-11-19）

**使用方式**：调度器只对 `ours` 方法生效，不传参数时仍走旧 runtime，已有结果不受影响。

```bash
./run smoke --methods ours --scheduler default
```

`--scheduler` 接受预设名或 JSON 文件。文件可以是完整策略，也可以写成 `{"base": "<预设>", "overrides": {...}}`。matrix 先把预设解析成完整策略再计算 `policy_sha256`，所以改动预设文件内容会得到新的结果身份，不会和旧结果混在一起。

**接线**：
- runner 在每个决策步前调用 `advance_step()`，每个用户 turn 结束时调用 `end_turn()`；
- 状态提供的预算、条数、形式、是否先提供操作记录、是否去重，全部来自调度器策略的 `publish` 段；
- 结果行新增 `scheduler_preset`、`scheduler_policy_sha256`、`scheduler_usage`（各模块的字节数和条目数）；
- `runtime` 统计新增淘汰、拒绝准入、失效、遗忘、对账、退避等计数。

**验证**（scratchpad 中的 Python 3.11 环境）：
- 新增 34 个机制测试，全部通过：每个预设和 L0 锁、三种失效粒度、未知写的隔离、strict 不缓存、覆盖只能调严、按步数的 TTL 与墙钟上限、LRU/GDSF 淘汰、单条大小上限、固定条目不被淘汰、空闲遗忘与 turn 边界、主体分区、公开目录、凭据超集与提前刷新、按"工具 + 参数"复用失败、对账、退避与共享重试上限、授权刷新重试、目录变化只失效改变的工具、session 层共享、publication 与 Access 集成、runner 与 matrix 接线。
- **对拍**：40 条随机操作序列（读、写、未知写、业务错误、可重试失败、授权变化、目录、凭据、时钟推进）上，`ablation_current` 与旧 `Runtime` 的每一步返回值、回源次数、命中统计和 sleep 序列全部一致；换成 `default` 或 `write_heavy` 后，80 条中有 76 条不同，说明对拍能发现差异。
- **端到端**：用脚本化的假模型在真实 BFCL（`bfcl-discovery-dev-1`）和本机 MCP 服务器上运行。`ablation_current` 与旧 Ours 的回源次数、M2 命中、状态提供次数完全一致（7 / 11 / 11）。`default` 少回源 1 次，来源是 M4 按"工具 + 参数"复用了重复 `mkdir` 的业务失败。这是 M4 第一次在真实执行路径上自然命中。
- 全部测试 204 个，只有 `test_scientist_adapter` 失败；这是改动前就存在的导入错误（缺 `ai_scientist` 外部包），与本次改动无关。

**与设计稿的偏离**：
- `allow_revalidate` 未实现。MCP 没有 ETag，需要服务器提供轻量校验读取后再做；schema 中保留该字段，加载器拒绝 true。
- 联动规则改为四条，关闭任意一条都退回保守行为（见 §3.3）。
- `session` 层只在调用方传入共享存储时生效，目前 runner 没有传，等 D3 决定后再接。只共享公开目录和凭据。
- 授权刷新重试和对账需要注册刷新器和对账器。现有三个数据集都没有提供，所以这两项目前只在单元测试和后续的 MCP 服务器上生效。
- 旧 runtime 会把非只读的"中性"调用（login、complete_task）结果也写进读取缓存，但从不用于复用；调度器不写。对拍只比较可观察行为，这一点不影响结果。

**实现过程中发现的问题（需要实验确认）**：
1. **M4 几乎不可能命中**：runner 每次调用都生成新的 op_id，旧实现的失败复用只按 op_id 索引。新增的 `failure_index=identity` 修正了这一点。
2. **BFCL 的只读契约只覆盖文件系统工具**：`ls`、`cat` 等在声明内，而 TradingBot、Vehicle、Travel、Message 等类的查询工具都未声明为只读。所以 12 个 BFCL case 中有 10 个根本没有 M2 可缓存的读取，这也可能是首轮 M2 命中很低的原因之一。是否补充声明需要作者核对工具语义后决定（D7）。
3. **AppWorld 的 app 粒度失效有一个前提**：一个 app 的写不会改变另一个 app 的只读结果。目前的只读声明只覆盖 supervisor、spotify、venmo、phone、simple_note，这个前提看起来成立，但还没有逐个 API 核实（D8）。

| # | 新增待定事项 | 草案推荐 |
|---|---|---|
| D7 | 是否给 BFCL 非文件系统类的查询工具补只读声明 | 补充，但逐个核对工具实现，并作为新版本契约单独记录 |
| D8 | 是否核实 AppWorld 不同 app 之间没有写→读的影响 | 用 AppWorld 数据库快照做一次确定性检查 |

---

## 11. v1 扩充：从相关工作吸收的机制（2026-11-19；第一档已实现，见 §12）

依据：[综述报告](survey-agent-tool-caching.md)。方括号里的编号是综述的参考文献编号。原则是把已有的好方法直接吸收进来，注明出处；我们自己的工作放在把它们统一到一套 M1–M4 状态和一套可调参数里。

下面每一项都给出：吸收什么、来自哪里、落在哪个参数上、为什么值得做。优先级按"收益大、改动小、证据强"排序，分三档。

### 11.1 第一档：建议先做（改动小，直接补上当前的短板）

| # | 机制 | 来源 | 做法 | 新增或修改的参数 |
|---|---|---|---|---|
| A1 | **抽样校验并自适应调整 TTL** | vCache 的错误率上界 [45]、Adaptive TTL 的随机逼近 [42] | 命中时以一定概率同时回源，比对缓存值与真实值。不一致就缩短该工具的 TTL，一致就逐步放宽，使陈旧命中率收敛到设定的目标值。校验的回源计入成本 | `ttl.adaptive.{enabled, verify_sample_rate, target_stale_rate, step, min_scale, max_scale}` |
| A2 | **陈旧命中率指标** | 综述的开放问题 2；FreshCache 是唯一报告过期错误率的工作 [14] | 评测模式下对每次命中都做影子回源（不改变返回值，不计入方法成本），统计命中中有多少与真实值不一致。A1 的抽样校验是它的在线版本 | 命令行 `--shadow-verify`（评测开关，不属于调度策略） |
| A3 | **幂等键透传** | IETF Idempotency-Key 草案 [71]、Stripe 的设计 [72]、沙箱实验中重复率 28% 降到 4% [22] | 契约声明支持幂等键的写操作，Host 把操作标识作为幂等键发给 Server。此时"结果未知的写"可以带同一个键安全重试，不必先对账 | `recovery.unknown_write` 增加 `idempotent_retry`；契约增加 `idempotency_key: true` |
| A4 | **失败记录用描述而不是原样调用** | Feedback That Backfires [80] | 提供给模型的失败记录从"工具名 + 参数"改为一句运行时生成的描述，例如"向 X 转账的操作在当前余额下被拒绝，条件未变"。保留原样形式作为对照 | `publish.failure_form ∈ {verbatim, described}` |
| A5 | **重试加抖动与断路器** | Exponential Backoff and Jitter [81]、重试放大是亚稳态失败的维持机制 [82] | 退避加全抖动；同一工具或 Server 连续失败达到阈值后，在冷却期内直接返回失败而不回源 | `retry.jitter ∈ {none, full}`；`recovery.breaker.{threshold, cooldown_steps}` |
| A6 | **对齐 MCP 协议的目录缓存语义** | SEP-2549 [6] | Server 返回的 `ttlMs` 作为 M1 的新鲜期，`cacheScope` 决定目录是否随授权失效，`list_changed` 通知立即失效。Server 不提供时退回现有的版本号判断 | `coupling.catalog_auth_scoped` 改为默认读取 `cacheScope`；新增 `ttl.catalog_default` |
| A7 | **目录内容哈希** | 工具描述在批准后被修改的攻击 [12] | 目录条目记录内容哈希。版本号没变而哈希变了，按目录变化处理并在结果中标记 | 无新参数，属于安全锁 |

A1 和 A2 值得特别说明。现在的 TTL 是人为设定的固定值，对不对只能靠猜。A1 让 TTL 根据实际观察到的陈旧情况自己调整，A2 让我们能报告一个现有工作都没有的数字。两者共用同一套回源比对代码。

### 11.2 第二档：值得做，但需要新的 MCP 服务器或更多标注

| # | 机制 | 来源 | 做法 | 依赖 |
|---|---|---|---|---|
| B1 | **由模型生成工具契约，再用执行校验** | ToolCacheAgent 为每个工具生成缓存计划 [4] | 让模型根据工具的 schema、描述和运行 trace，为每个工具提出新鲜度级别、读取的资源标签、写影响的标签。提出的契约先经过 A2 的影子校验，陈旧命中率超标的自动收紧。这同时解决了待定事项 D2（标签由谁标注）和 D7（BFCL 的只读声明） | A2 |
| B2 | **按过期概率决定是否复用** | FreshCache [14] | 对易变类数据，用"距上次获取的时间 + 该工具的历史变化率"估计已过期的概率，低于错误预算才复用，代替固定 TTL | 需要有外部变化的数据，即新服务器 |
| B3 | **版本校验代替完整回源** | Invalidation Contracts 的版本戳 [21] | Server 为资源提供版本号。缓存过期后先做一次轻量的版本查询，版本没变就续期。这就是 v0 里预留但未实现的 `allow_revalidate` | 新服务器提供版本接口 |
| B4 | **先校验后置条件再重试** | Verified Tool Calls [74] | 写操作失败且结果不明时，调用契约声明的后置条件检查，确认未生效才重试。对不支持幂等键的工具，这是 A3 的替代 | 契约声明后置条件 |
| B5 | **用任务结构预测下次访问** | KVFlow 用工作流图指导淘汰与预取 [30] | 淘汰时优先保留"接下来还会用到"的条目。预测信号来自当前 turn 提到的资源和最近读取的序列 | 只在出现容量压力时有意义，见 11.4 |
| B6 | **准入过滤** | TinyLFU [58]、AdaptSize [59] | 结果第二次出现才写入缓存，或按结果大小做概率准入，避免一次性的大结果占用空间 | 同 B5 |
| B7 | **把剩余有效期写进淘汰优先级** | Hyperbolic Caching [57] | 快过期的条目优先淘汰，统一处理淘汰和过期 | 同 B5 |
| B8 | **屏蔽已失效的旧结果** | 简单的观测屏蔽即可把成本减半 [91] | 对话里某条读取结果已被后续的写操作作废时，把它替换成一句"此结果已失效"的短说明，既省 token 又防止模型引用旧值 | 无 |

### 11.3 第三档：扩展方向，先不排期

| # | 机制 | 来源 | 说明 |
|---|---|---|---|
| C1 | 推测式预取只读结果 | PASTE [95]、LLMCompiler [94] | 在模型生成期间提前取回很可能要读的结果。能降延迟，但会增加请求数，与论文"减少远程交互"的目标有冲突，需要单独核算 |
| C2 | 重复调用检测 | AgentLoop [78]、DeLoop [77] | 相同调用得到相同观测时提示模型"已观察过且未变化"。现有的有效性引用已经部分起到这个作用 |
| C3 | 历史前缀匹配作为最严格的有效性模式 | TVCACHE [2] | 只在完整调用历史一致时复用。适合作为正确性上界的对照，不适合作为默认 |
| C4 | 执行前先落盘操作意图 | LogAct [25] | 现在操作记录是在调用返回后才写入。改成发出前先写入"进行中"状态，崩溃恢复时能知道哪些写处于未知状态 |

### 11.4 淘汰相关机制的前提

B5、B6、B7 都只在缓存真的装不下时才起作用。综述没有找到任务内缓存遇到容量压力的证据，v0 的端到端检查里 M2 也只占了一百多字节。建议先用现有任务测一次缓存的实际占用：

- 如果远低于上限，这三项就只在"跨任务共享"或"人为压低容量"的实验里讨论，不进入默认策略；
- 如果长任务确实会撑满，再按 B6 → B7 → B5 的顺序实现。

### 11.5 RSI 可以吸收的做法

| 做法 | 来源 | 怎么用 |
|---|---|---|
| bandit 在线调准入 | ToolCaching [1] | 作为 RSI 的一个轻量搜索器，适合在线、单参数的调整 |
| 带安全约束的贝叶斯优化 | HARBOR [7] | 作为 L2 数值参数的离线搜索器，替代现在的坐标下降；它的"机会约束"对应我们"正确率不得下降"的门槛 |
| 把模型生成限制在受限的策略空间里 | Vulcan [97] | 我们的 schema 加 L0 安全锁是同一思路。后续如果让模型生成策略，沿用这套约束 |
| 根据失败分析反思并改写策略 | GEPA [100] | 让模型读失败 trace，提出对参数或契约的修改（与 B1 结合） |

### 11.6 吸收之后，方案的整体样子

一句话概括：**用版本化标签统一判断四类状态是否有效，用抽样校验让有效期自己收敛，用幂等键和后置条件让失败后的重试有依据，再用搜索为不同场景选参数。**

对应到模块：

- M1：协议原生的 TTL 和通知（A6）+ 内容哈希（A7）
- M2：标签失效（v0 已有）+ 自适应 TTL（A1）+ 过期概率和版本校验（B2、B3）+ 屏蔽失效结果（B8）
- M3：保持 v0 的保守做法
- M4：幂等键（A3）+ 后置条件（B4）+ 描述式失败记录（A4）+ 断路器和抖动（A5）
- 契约：模型生成、执行校验（B1）
- 调参：贝叶斯优化或 bandit（11.5）
- 评测：陈旧命中率（A2）

### 11.7 需要你们决定的

| # | 问题 | 草案推荐 |
|---|---|---|
| D9 | 第一档七项是否全部实现 | 是。其中 A3 需要 Server 支持幂等键，现有三个数据集的环境不支持，先实现 Host 侧逻辑和单元测试，等新服务器接入 |
| D10 | A1 的校验回源算不算方法成本 | 算。这样自适应机制的开销和收益才能如实比较 |
| D11 | B1（模型生成契约）是否作为论文的一个贡献点 | 值得考虑。它把"谁来标注"这个工程问题变成了方法的一部分，并且与 RSI 的叙事一致 |

---

## 12. 第一档实现说明（2026-11-19）

A1–A7 已全部实现。默认预设（`default`）开启了这些机制，`ablation_current` 全部关闭，仍与旧 runtime 逐步一致。

**使用方式**

```bash
./run smoke --methods ours --scheduler default --shadow-verify
```

`--shadow-verify` 只用于评测：每次缓存命中时通过不计量的控制通道回源比对，结果行里的 `stale_hits.stale_hit_rate` 就是陈旧命中率。它不改变 agent 看到的内容，不计入 MCP 请求数，耗时单独记录在 `shadow_time_s` 并从 Active time 中扣除。

**各项的实现要点**

| # | 实现 | 默认值 |
|---|---|---|
| A1 自适应 TTL | 命中时按 `verify_sample_rate` 抽样回源。每次校验后，该工具的 TTL 倍率乘以 `exp(step × (target_stale_rate − 是否陈旧))`，因此陈旧命中率会向目标值收敛。校验的回源计入请求数和后端调用（对应 D10）；被校验的命中不算作 M2 命中 | 抽样率 0.1，目标 0.02，step 0.5，倍率范围 0.1–4 |
| A2 陈旧命中率 | 见上 | 关闭，由命令行打开 |
| A3 幂等键 | 契约声明支持幂等键的写操作，Host 把操作标识作为键随请求发出。结果未知时带同一个键重试，占用共享的重试次数。没有键的写仍然先对账，对不了账就阻断 | `unknown_write = idempotent_retry` |
| A4 描述式失败记录 | 提供给模型的失败记录只含一句描述：工具名、发生在第几个决策步、错误类别、相关状态是否变化。不重复参数 | `failure_form = described` |
| A5 抖动与断路器 | 退避值在 0 到上限之间均匀取值，Server 给的 `retry_after` 始终作为下限。同一工具连续出现可重试的失败达到阈值后，冷却期内直接返回 `circuit_open`，不回源。业务错误不触发断路器 | 全抖动；阈值 3，冷却 4 步 |
| A6 协议目录语义 | 目录结果里的 `ttlMs` 作为新鲜期，`cacheScope` 决定是否随授权失效；`notify_list_changed()` 立即丢弃目录。Server 不提供时用 `ttl.catalog_default`，再没有就只按版本、内容和授权判断 | `catalog_default = null` |
| A7 内容哈希 | 每次读到目录都记录内容哈希。版本号没变而内容变了，计入 `catalog_silent_changes`，并失效依赖它的读取结果。收到变更通知但内容其实没变时，不失效任何读取结果 | 始终开启 |

**验证**

- 调度器测试从 34 个增加到 50 个，全部通过。新增的 16 个覆盖：陈旧时缩短、新鲜时放宽、收敛不发散、关闭时不校验、影子比对不改变行为、幂等键重试与关闭、抖动范围与 `retry_after` 下限、断路器的打开与恢复、业务错误不触发断路器、描述式记录不含参数、协议 TTL 与作用域、静默变更检测、变更通知、Access 层的幂等键与影子读取。
- 全量测试 220 个，仍然只有改动前就存在的 `test_scientist_adapter` 导入错误。
- 对拍：`ablation_current` 与旧 runtime 在 40 条随机序列上仍然完全一致。
- 端到端（脚本化假模型，真实 BFCL 环境与本机 MCP 服务器）：
  - `ablation_current` 与旧版 Ours 仍然一致（回源 7 次、M2 命中 11 次、MCP 请求 8 次）。
  - `default` 做了 2 次抽样校验，MCP 请求从 8 次变为 9 次，这是自适应机制的真实开销。
  - 打开 `--shadow-verify` 后做了 11 次影子比对，MCP 请求数不变（仍为 9 次），陈旧命中率为 0。这符合预期：这个任务里的数据只会被 agent 自己的写操作改变，而写后失效已经处理了这种情况。

**目前用不上的部分**

- A3：现有三个数据集的环境都不支持幂等键，契约里的列表是空的。逻辑只在单元测试里生效，要等新的 MCP 服务器。
- A6：现有的适配服务器不返回 `ttlMs` 和 `cacheScope`，runner 也没有订阅 `list_changed` 通知。同样要等新服务器。
- A1：现有任务环境是隔离的，没有外部变化，所以抽样校验几乎不会发现陈旧，TTL 倍率只会缓慢变大。它的作用要在有外部变化的服务器上才能体现。在现有数据集上它只增加开销，跑对比时可以考虑把抽样率设为 0。
- A2：陈旧命中率目前只对调度器有效，旧 runtime 和其他对照方法没有接入。要比较各方法的陈旧命中率，需要给它们也接上影子比对。

**一处需要注意的默认值**

A4 的默认值设成了描述式。它的依据只有一项针对小模型的研究，在你们使用的模型上是否有效还不知道。建议第一次真实实验就把 `described` 和 `verbatim` 各跑一遍。

---

## 13. RSI 接入调度器参数（2026-11-19）

RSI 搜索程序现在有两种模式。原来的五参数模式不变，仍是默认；新增的调度器模式搜索 M1–M4 调度策略。

**使用方式**

```bash
./run rsi --space scheduler --rounds 3
```

加上 `--dry-run` 只打印计划、不执行。加上 `--shadow-verify --max-stale-rate 0.02` 会测量每个候选的陈旧命中率，超过上限的候选不会被选中。

结果默认写到 `results/rsi-scheduler/`，其中 `selected-policy.json` 可以直接交给评测：

```bash
./run smoke --methods ours --ours-policy-file results/rsi-scheduler/selected-policy.json
```

**搜索空间从 schema 自动得到**

[ours/rsi_space.py](../ours/rsi_space.py) 读取参数 schema，把所有标为 L1 或 L2 的参数作为可调参数，取值范围就是 schema 里的范围。目前共 62 个（26 个结构开关、36 个数值）。安全锁（L0）不在空间里。以后在 schema 里新增参数并标上层级，它会自动进入搜索空间。

每个候选只比它的父策略改动一处：

| 参数类型 | 怎么生成邻居 |
|---|---|
| 数值 | 乘以 0.5 和 2，再裁剪到 schema 范围；整数取整；当前值为 0 时试 1；"不限"（null）保持不动 |
| 开关 | 取反 |
| 枚举 | 换成其他每个取值 |
| 规则集合 | 逐条开或关 |
| 容量份额 | 从份额最大的其他模块挪 0.1 给目标模块，总和保持为 1 |

每个邻居都要通过完整的策略校验，通不过的直接丢弃（例如尚未实现的 `allow_revalidate`）。

**搜索过程**

搜索计划预先登记在 [configs/rsi-scheduler-space.json](../configs/rsi-scheduler-space.json)：

- 第 1 轮比较四个预设：`default`、`read_heavy`、`write_heavy`、`ablation_current`；
- 之后每轮从**到目前为止最好的策略**出发，沿计划里列出的 16 个参数依次生成邻居，每轮最多评估 6 个新候选；下一轮从上一轮停下的参数接着往后走；
- 选择规则沿用原来的：先比正确完成数，再比 Active time、输入 Token、MCP 请求三项归一化后的等权成本；
- 已经评估过的策略不会重复跑，中断后可以续跑。

调度器自身的开销已经包含在这三项成本里：抽样校验的回源计入 MCP 请求，状态提供的内容计入输入 Token。所以待定事项 D6（开销是否进入目标）不需要额外处理。

**成本**

按默认的 9 个开发 case 计算，3 轮是 (4 + 6 + 6) × 9 = 144 次真实任务执行。之前五参数的两轮搜索是 81 次。可以用 `--rounds`、计划里的 `round_limit` 和 `--cases` 控制规模。

**验证**

- 新增 16 个测试，全部通过。覆盖：安全锁不在空间内；数值的缩放与裁剪；开关、枚举、规则集合和份额的邻居；每个参数的每个邻居都是合法策略且安全锁不变；计划里出现安全锁参数会被拒绝；每轮上限、跳过已评估、按参数顺序续走；候选编号可以安全地用作目录名。
- 用假的任务执行函数跑完整的三轮搜索：在一个"有效期越长成本越低"的假设环境里，搜索先选中 `read_heavy`，随后继续把有效期调大。
- 陈旧门槛：在同一个环境里再假设"有效期长就会返回陈旧值"，加上门槛后被选中的是不陈旧的策略；去掉门槛则选中陈旧的那个。
- 搜索产出的策略文件能被评测入口读取，两边算出的策略哈希一致。
- 全量测试 236 个，仍然只有改动前就存在的 `test_scientist_adapter` 导入错误。

**没有做的**

- 搜索器仍然是坐标下降。综述里提到的贝叶斯优化和 bandit 没有实现（待定事项 D5）。
- 没有"按场景特征自动选起点预设"这一步（设计稿 §7.3 第 1、2 步）。现在第 1 轮是把几个预设都跑一遍。
- 没有分场景搜索。现在是在 9 个开发 case 上选一组全局最优的参数。要做"针对某个场景适配"，目前的办法是用 `--cases` 只传该场景的 case，分别跑几次。
- 以上全部没有用真实模型跑过。
