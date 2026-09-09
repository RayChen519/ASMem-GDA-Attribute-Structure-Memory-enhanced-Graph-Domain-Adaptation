# Shared GCN Encoder 实现计划

## 1. 接口

输入来自 Dataset.md：

$$
X_d\in\mathbb R^{N_d\times6775},\qquad
\widehat A_d\in\mathbb R^{N_d\times N_d},\qquad d\in\{s,t\}.
$$

不设置 GCN 前置输入映射层。Source/Target 直接复用同一组 GCN 参数。

输出：

$$
H_d^{(1)}\in\mathbb R^{N_d\times256},\qquad
Z_d\in\mathbb R^{N_d\times128}.
$$

$Z_s,Z_t$ 交给 Attribute and Structure level.md。

## 2. 主实验默认结构

以下为主实验默认工程实现，不是 Idea 强制结构：

| 层 | 输入→输出 | 后处理 |
|---|---|---|
| GCNConv-1 | 6775→256 | LayerNorm→ReLU→Dropout(0.5) |
| GCNConv-2 | 256→128 | LayerNorm |

$$
H_d^{(1)}=
\operatorname{Dropout}_{0.5}
\left(\operatorname{ReLU}\left(
\operatorname{LN}(\operatorname{GCNConv}_1(X_d,\widehat A_d))
\right)\right),
$$

$$
Z_d=\operatorname{LN}
\left(\operatorname{GCNConv}_2(H_d^{(1)},\widehat A_d)\right).
$$

第二层不使用 ReLU。GCN 权重使用 Xavier uniform，bias 为 0；不添加残差 GCN、Jumping Knowledge 或额外传播层。

分类头固定为：

    Linear(128,64) → ReLU → Dropout(0.5) → Linear(64,C)

A/S fusion 输出同为 128 维，因此后续阶段继承同一分类头。

## 3. Encoder warm-up

仅使用 Source training 标签：

$$
\mathcal L_{enc}=
\operatorname{CE}
\left(C_\psi(Z_{s,V_s^{L,tr}}),y_{s,V_s^{L,tr}}\right).
$$

训练 shared GCN 和 classifier；A/S、Memory 和 discriminator 尚未建立。

    max_epochs = 100
    min_epochs = 20
    patience = 15
    selection = Source validation Macro-F1
    tie_break = Source validation loss
    output = encoder_best

Target 只允许以 eval + no_grad 前向检查形状和有限值，不提供监督或模型选择信号。

## 4. encoder_best

至少保存：

    stage, source, target, label_rate, seed
    raw_feature_dim=6775, attribute_union_hash
    shared_gcn_state, classifier_state
    optimizer_state, scheduler_state, early_stop_state
    epoch, Source train/validation mask hashes
    dataset_manifest_hash, resolved_config_hash
    Python/NumPy/Torch/CUDA RNG states
    requires_grad state

DA 加载时校验第一层为 6775→256、最终维度为 128、类别数和全部 hash 一致。

## 5. 验收

- Source/Target 使用相同 GCN 参数对象。
- 第一层输出 [N_d,256]，Z_d 为 [N_d,128]，所有值有限。
- GCN 和 classifier 在 Source 训练步中获得有限梯度。
- loss 只索引 $V_s^{L,tr}$。
- encoder_best 可直接初始化 DA，分类头不被重建。

