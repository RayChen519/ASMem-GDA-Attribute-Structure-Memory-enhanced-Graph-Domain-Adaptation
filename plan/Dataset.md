# Dataset 实验计划

## 1. 输出与任务

处理 ACMv9、Citationv1、DBLPv7，并生成六个有向任务：

- ACMv9 → Citationv1、ACMv9 → DBLPv7
- Citationv1 → ACMv9、Citationv1 → DBLPv7
- DBLPv7 → ACMv9、DBLPv7 → Citationv1

每个处理后图至少包含：

    x, edge_index, node_id, degree, domain_id
    feature_dim, attribute_union_hash, dataset_version
    source_train_mask, source_val_mask, source_unlabeled_mask
    target_unlabeled_mask, target_test_mask

生成 dataset_manifest，记录节点/边/类别/特征统计、异常值、原始文件哈希、类别映射和 attribute vocabulary 映射。

## 2. 图与类别处理

1. 节点 ID 重映射为 0 到 N_d-1。
2. 删除重复边，保留孤立节点，将 citation 边对称化。
3. Anchor degree 使用对称化后、加自环前的度。
4. GCN 邻接加入自环并使用

$$
\widehat A_d=\widetilde D_d^{-1/2}(A_d+I)\widetilde D_d^{-1/2}.
$$

5. 三个数据集只保留共同类别，统一映射到 0 到 C-1；任一 Source 类别少于两个节点则停止该任务。

## 3. 统一特征空间

三个 domain 固定使用同一 6775 维 attribute union vocabulary：

$$
X_d\in\mathbb R^{N_d\times6775}.
$$

- 已对齐的 6775 维文件在验证列映射和 vocabulary hash 后直接使用。
- 否则按 attribute 名称/ID 建立并集与固定列顺序；缺失 attribute 列置 0。
- 禁止按原始列号拼接、截断、无依据补零或学习 domain-specific 输入映射。
- 每个节点做 L1 行归一化。
- 保存统一列清单、原始列映射及 attribute_union_hash。
- 归一化后的 X_d 直接输入第一层 GCNConv(6775,256)。

## 4. Source/Target 划分

Source 标签率：

$$
r_s\in\{1\%,3\%,5\%\},\qquad
n_{s,c}^{L}=\max(2,\operatorname{round}(r_sN_{s,c})).
$$

每类标记预算按 80%/20% 分成：

- $V_s^{L,tr}$：唯一可用于真实标签梯度更新的集合。
- $V_s^{L,val}$：只用于早停、checkpoint 选择和配置锁定。
- $V_s^U$：隐藏真实标签，用于 anchor 和 Source 伪标签。

每类在 train/validation 中至少各有一个节点，三个 Source mask 两两不重叠并覆盖全部 Source 节点。

Target 在训练中完全无标签：

$$
V_t^L=\varnothing,\qquad V_t^U=V_t^{test}=V_t.
$$

训练可读取 Target 特征、图和域标签，但不得读取 Target 类别标签。

## 5. Seeds 与配置选择

正式统计 seeds 为 $\{0,1,2,3,4\}$。同一 Source、标签率和 seed 的 split 在不同迁移方向与变体间复用。

独立开发划分固定：

    development_seed = 2026
    label_rate = 1%, 3%, 5%
    train/validation = 80%/20%

development 只使用 Source validation 锁定超参数，不进入最终五-seed统计；Target 仍保持无标签。

## 6. 标签隔离接口

训练入口只构造：

- SourceTrainView：图数据、$V_s^{L,tr}$ 标签和 Source masks。
- TargetTrainView：x、edge_index、node_id、target_unlabeled_mask、domain_id，不含 y。
- TargetEvaluationView：只由独立 final-evaluation 入口创建。

只有 run manifest 和该变体的 final checkpoint 锁定后才能创建 TargetEvaluationView：

| 变体 | final checkpoint |
|---|---|
| B0 | encoder_best |
| B1–B4 | da_best |
| B5 | source_pl_best |
| B6 | full_best |

Target 真实标签不得影响训练、伪标签、筛选、早停、checkpoint、调参、重跑或人工选择。

## 7. 缓存键

缓存必须覆盖所有决定性输入：

- 统一特征：dataset_version + attribute_union_hash + normalization_version。
- Source split：dataset_version + source_domain + label_rate + split_seed。
- METIS：dataset_version + domain + P + metis_seed。
- Anchor：dataset_version + direction + label_rate + split_seed + K + sampler + sampler_hparams。

相同键必须确定性复现；任一字段变化不得命中旧缓存。

## 8. 下游接口

执行顺序固定为：

    Dataset
    → Shared GCN Encoder
    → Attribute/Structure DA
    → Memory warm-up 与伪标签
    → End-to-End 五阶段训练
    → Integration gate
    → Baseline/Ablation
    → Full Experiment

Dataset 输出 $X_s,X_t,\widehat A_s,\widehat A_t$、split/hash 和无标签 Target view；后续模块不得重新划分数据或改变 feature vocabulary。

## 9. 验收

- 三个 X_d 均为有限的 [N_d,6775] 张量且 attribute_union_hash 相同。
- edge index 合法，邻接和 mask 可复现。
- Source train/validation 每类非空。
- TargetTrainView 不含真实标签。
- 切换 split、P、K 或 sampler 后缓存正确失效。

