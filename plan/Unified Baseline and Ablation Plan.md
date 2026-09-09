# Unified Baseline and Ablation 实验计划

## 1. 统一规则

所有变体复用：

- Dataset.md 的数据版本、6775 维特征、split 和 Target 标签隔离。
- Shared GCN Encoder Plan.md 的两层 GCN、128 维接口和 classifier。
- P=128、metis_seed=0 的 partition。
- 相同 Source validation 配置选择规则和正式 seeds。
- Integration and Verification Plan.md 的 PASS report。

除变体指定因素外，不得改变结构、预算、数据或评估入口。

## 2. 核心矩阵

| ID | 名称 | DA | Attribute | Structure | Memory | Source PL | Target PL | final checkpoint |
|---|---|---:|---:|---:|---:|---:|---:|---|
| B0 | source_only | 否 | 否 | 否 | 否 | 否 | 否 | encoder_best |
| B1 | dann | 是 | 否 | 否 | 否 | 否 | 否 | da_best |
| B2 | attribute_only | 是 | 是 | 否 | 否 | 否 | 否 | da_best |
| B3 | structure_only | 是 | 否 | 是 | 否 | 否 | 否 | da_best |
| B4 | as_no_memory | 是 | 是 | 是 | 否 | 否 | 否 | da_best |
| B5 | as_memory_source_pl | 是 | 是 | 是 | 是 | 是 | 否 | source_pl_best |
| B6 | sgda_full | 是 | 是 | 是 | 是 | 是 | 是 | full_best |

实现要求：

- B0 训练期间不读取 Target；final evaluation 用同一 GCN 对 Target 前向。
- B1 不建 A/S，使用 Linear(128,128)+LayerNorm 作为表示适配层。
- B2/B3 只建一个分支，分支输出经 Linear(128,128)+LayerNorm 接 classifier/discriminator。
- B4 停在完整 A/S DA。
- B5 增加 Memory warm-up 和 Source PL，不生成 Target PL。
- B6 执行完整五阶段链。

## 3. 机制消融

所有消融继承 B6 resolved config，仅改变一项：

| ID | 名称 | 唯一变化 |
|---|---|---|
| A1 | mlp_scorer | self-attention scorer 改为 MLP |
| A2 | no_soft_threshold | 移除软阈值 |
| A3 | local_structure_only | 移除 inter-cluster attention |
| A4 | symmetric_selection | Source/Target 都执行 selection |
| A5 | shared_theta | $\theta_A=\theta_S$ |
| M1 | uniform_anchors | 均匀无放回采样 Source anchors |
| M2 | degree_quantile_anchors | degree-quantile 平衡采样 |
| M3 | has_similarity | 使用 $H^{AS}$ 而非 $H^S$ 构造 Query/Key |
| M4 | no_query_residual | 移除 Memory Query residual |
| M5 | no_memory_warmup | Memory 不进行 Source 监督 warm-up |
| M6 | no_consistency | 移除 teacher/student 和跨轮一致性 |

M5 保存 untrained_memory 标记，不得伪装为正常 memory_best。

## 4. 配置公平性

- B0–B6 使用相同最大 development 调参预算。
- 每个核心模型优先锁定一套全局配置；满足 End-to-End Training Plan.md 的条件时，最多按 1%/3%/5% 拆成三套。
- 禁止方向级配置和 Target 指标驱动调参。
- A1–M6 不重新调参。
- 相同 direction/rate/seed 必须复用 split、METIS、anchor candidate pool 和 Target node order。
- M1/M2 只改变 anchor 抽样策略，K 不变。

## 5. 运行层级

每个注册变体执行：

$$
6\text{ directions}\times3\text{ rates}\times5\text{ seeds}=90.
$$

- 核心确认性矩阵：7×90=630 runs。
- 机制消融矩阵：11×90=990 runs。
- 二者合计 1620 次确认性运行。

Sensitivity analysis 独立执行，不进入 1620 次：

    model = B6
    label_rate = 3%
    directions = all 6
    seeds = 0,1,2

单因素候选：

| 因素 | 值 | 默认 |
|---|---|---|
| K | 64,128,256 | 128 |
| T | 0.05,0.1,0.2 | 0.1 |
| gamma | 0.85,0.90,0.95 | 0.90 |
| q | 0.10,0.20,0.30 | 0.20 |
| METIS P | 64,128,256 | 128 |

复用默认结果后新增 180 runs。主实验始终使用 P=128。

## 6. 输出与验收

每个 run 至少记录：

    variant/parent/changed_component
    direction/rate/seed
    config/split/METIS/anchor hashes
    final checkpoint name/hash
    Source validation metrics
    Target final Accuracy/Macro-F1
    time/VRAM/parameter count

验收：

- B0–B6 开关与表格一致。
- B5 无 Target PL；B6 的 PL 来自冻结 Memory teacher。
- 十一项消融均为单因素变化。
- 630、990、180 三层清单和结果目录分离。
- Target 指标只由相应 final checkpoint 锁定后的独立评估产生。

