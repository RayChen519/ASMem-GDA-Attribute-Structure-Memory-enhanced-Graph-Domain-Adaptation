# Full Experiment Execution and Analysis 实验计划

## 1. 前置条件

执行顺序固定为：

    Dataset
    → Shared GCN Encoder
    → Attribute/Structure
    → Memory Network
    → End-to-End Training
    → Integration PASS
    → Baseline/Ablation registry
    → Full Experiment

启动正式运行前必须锁定：

    dataset/split/attribute union/METIS manifests
    development_seed=2026 resolved configs
    Integration PASS report
    B0–B6 and A1–M6 registry
    immutable run manifests

上游代码、配置或 hash 变化后，相关 PASS report 和 run manifest 必须失效并重建。

## 2. 运行清单

任务：

- ACMv9 → Citationv1、ACMv9 → DBLPv7
- Citationv1 → ACMv9、Citationv1 → DBLPv7
- DBLPv7 → ACMv9、DBLPv7 → Citationv1

正式配置：

$$
r_s\in\{1\%,3\%,5\%\},\qquad
seed\in\{0,1,2,3,4\}.
$$

每个变体 90 runs；核心 630 runs，机制消融 990 runs。Sensitivity 180 runs 使用独立清单。

run_id：

    {tier}__{variant}__{source}-to-{target}__rate-{rate}__seed-{seed}

run_id 必须由字段确定性生成，不使用时间戳作为身份。

## 3. 固定数据资产

主实验统一：

    feature_dim = 6775
    GCN = 6775→256→128
    METIS P = 128
    metis_seed = 0

同一 dataset 的特征映射和 partition 在所有方向、标签率、seeds 和变体间复用。P 只在 sensitivity 中取 64/128/256。

## 4. 配置与元数据

配置覆盖层级：

    base
    → variant
    → locked development config
    → optional rate-level config
    → direction/rate/seed
    → explicit smoke or sensitivity override

禁止方向级超参数配置。启动前输出完整 resolved config 并计算稳定 hash。

每个 run 保存：

    run_id, tier, variant, direction, rate, seed
    resolved_config and hash
    dataset/attribute_union/split/METIS/anchor hashes
    code snapshot hash or git commit
    dependency/CUDA/device information
    timestamps, host, process ID

## 5. Checkpoint 与恢复

完整模型只允许以下继承链：

$$
\text{encoder_best}
\rightarrow\text{da_best}
\rightarrow\text{memory_best}
\rightarrow\text{source_pl_best}
\rightarrow\text{full_best}.
$$

B0–B5 在各自 final checkpoint 结束，不伪造后续阶段。

每阶段保存：

- latest：同阶段断点恢复。
- best：只由 Source validation 更新。
- 直接父 checkpoint 路径和内容 hash。

恢复内容：

    model and requires_grad state
    optimizer/scheduler/early-stop state
    epoch and patience
    all RNG states
    GRL and lambda schedules
    anchor IDs/hash
    pseudo-label/confidence/mask/refresh history
    parent checkpoint hash

配置、数据、split、METIS、anchor 或父 hash 不一致时拒绝恢复。checkpoint 采用临时文件、完整性校验和原子替换。

## 6. 调度与失败处理

执行层级：

1. B0/B1 sanity。
2. B0–B6 核心矩阵。
3. A1–M6 机制消融。
4. sensitivity。
5. final evaluation。
6. 汇总与统计。

基础设施故障可从 latest 恢复。代码/config 缺陷修复后，使所有受影响 manifests 失效并统一重跑。禁止因 Target 表现不佳而重跑、换 seed 或改变配置。

只有 checkpoint/evaluation hash 完整的 COMPLETED run 进入汇总。

## 7. Target final evaluation

训练只使用 TargetTrainView。满足以下条件后，独立 final-evaluation 才能创建 TargetEvaluationView：

1. 规定 final checkpoint 完整写入。
2. checkpoint hash 写入锁定 run manifest。
3. config、best epoch 和选择决策已锁定。
4. 训练已结束或处于只读 evaluation 模式。
5. 输出绑定 checkpoint 与 evaluation code hash。

| 变体 | final checkpoint |
|---|---|
| B0 | encoder_best |
| B1–B4 | da_best |
| B5 | source_pl_best |
| B6 | full_best |

Target 标签不得影响训练、伪标签、早停、checkpoint、调参、重试或人工选择。离线 pseudo-label accuracy 同样在该 gate 后计算。

## 8. 汇总与统计

每个 run 输出 Target Accuracy 和 Macro-F1。对每个 variant/direction/rate 的五个 seeds 报告：

$$
\bar M=\frac15\sum_{i=1}^5M_i,\qquad
s=\sqrt{\frac14\sum_{i=1}^5(M_i-\bar M)^2}.
$$

使用样本标准差；缺失 seed 时不生成正式均值。

显著性：

- 先形成 18 个 direction×rate 条件的五-seed均值。
- B6 对 B0–B5：6 个配对双侧 Wilcoxon。
- B6 对 A1–M6：11 个配对双侧 Wilcoxon。
- 两个 family 分别做 Holm 校正，$\alpha=0.05$。
- 报告 raw/adjusted p、rank-biserial effect size、配对中位差和分层 bootstrap 95% CI。
- Macro-F1 为主要终点，Accuracy 单独作为次要结果。

## 9. 负迁移与效率

对每个匹配 run：

$$
\Delta_{NT}=M_{variant}-M_{B0}.
$$

报告每个 direction×rate 的均值±标准差、90 个匹配 runs 中 $\Delta_{NT}<0$ 的比例、18 条件分布及 bootstrap 95% CI。

统一记录：

    stage/total training time
    final inference and PL refresh time
    peak GPU memory
    trainable/total parameters
    completed epochs and checkpoint size

相同硬件、缓存和计时边界下比较 B0、B1、B4、B5、B6。Sensitivity 单独报告 K/P 的性能与开销。

## 10. 产物与验收

输出：

    locked manifests and resolved configs
    checkpoint lineage
    per-run raw metrics
    five-seed summaries
    significance tables
    negative-transfer analysis
    efficiency and sensitivity tables
    failure/retry and Target-access audit logs

验收：

- 核心、消融和 sensitivity 清单数量分别为 630、990、180。
- 完整模型五级 checkpoint 父 hash 连续。
- 每个 Target 指标可追溯到锁定 checkpoint。
- 统计口径在查看结果前锁定。

