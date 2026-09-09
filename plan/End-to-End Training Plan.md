# End-to-End Training 实验计划

## 1. 执行链与模块

完整方法顺序：

    数据集处理
    → 两层共享 GCN
    → Attribute selection 与 Structure local/global + selection
    → 两分支拼接 + Linear(256,128)
    → GRL 域对抗
    → Memory Network 生成 Source/Target 伪标签
    → Source 真实标签与筛选后伪标签监督分类器
    → 双域联合微调

完整模型按顺序执行：

$$
\boxed{
\text{encoder_best}
\rightarrow\text{da_best}
\rightarrow\text{memory_best}
\rightarrow\text{source_pl_best}
\rightarrow\text{full_best}
}.
$$

模块来源：

- Dataset.md：6775 维特征、split 和标签隔离。
- Shared GCN Encoder Plan.md：两层 GCN 与 classifier。
- Attribute and Structure level.md：A/S selection、fusion 和 discriminator。
- Memory Network.md：Source anchors、teacher 和伪标签。
- 本文：阶段、优化器、冻结状态和 checkpoint 继承。

## 2. 默认配置

模型：

    shared GCN = 6775→256→128
    classifier = 128→64→C
    discriminator = 128→64→1
    A/S and Memory hidden_dim = 128

所有阶段使用 AdamW：

    weight_decay = 5e-4
    gradient_clip_norm = 5.0
    warm-up = 5 epochs
    scheduler = cosine decay
    minimum_lr_ratio = 0.1

默认学习率：

| 参数组 | 常规阶段 | Stage 5 解冻后 |
|---|---:|---:|
| shared GCN | 5e-4 | 1e-4 |
| Attribute/Structure/fusion | 1e-3 | 1e-4 |
| classifier | 1e-3 | 5e-4 |
| discriminator | 1e-3 | 5e-4 |
| Memory | 1e-3 | 冻结 |

默认初始超参数：

    lambda_adv_max=0.1, lambda_sp=1e-4
    lambda_s=0.5, lambda_t_max=0.5
    K=128, T=0.1, gamma=0.90, q=0.20
    R=10, alpha=2, beta=1

这些值须由 development_seed=2026 的 Source validation 最终锁定。

## 3. 五阶段训练

### Stage 1：Encoder warm-up

- 初始化：随机初始化两层 GCN 和 classifier。
- 训练：GCN、classifier。
- 数据：只用 $V_s^{L,tr}$ 真实标签；Target 仅 no_grad 形状检查。
- 损失：$\mathcal L_{enc}$。
- 调度：最多 100 epoch，至少 20，patience 15。
- 选择：Source validation Macro-F1，loss 作 tie-break。
- 输出：encoder_best。

### Stage 2：Attribute/Structure DA

- 继承：encoder_best。
- 新建：Attribute、Structure、fusion、discriminator。
- 训练：GCN、A/S、fusion、classifier、discriminator。
- Memory：不存在。
- Target：只提供特征、图和域标签。

$$
\mathcal L_2=
\mathcal L_{cls}^s+
\lambda_{adv}(p)\mathcal L_{dom}+
\lambda_{sp}\mathcal L_{sp},
$$

$$
\lambda_{adv}(p)=\lambda_{adv,max}
\left(\frac{2}{1+e^{-10p}}-1\right).
$$

域特征只经过一个 GRL；总损失中 $\mathcal L_{dom}$ 为正号。

- 调度：最多 150 epoch，至少 30，patience 20。
- 选择：Source validation。
- 输出：da_best。

### Stage 3：Memory warm-up

- 继承：da_best。
- 冻结：GCN、A/S、fusion、classifier、discriminator。
- 训练：Memory Network。
- Anchor：只从 $V_s^U$ 构建并固定 ID。
- 监督：仅 $V_s^{L,tr}$。
- 损失：$\mathcal L_{mem}$。
- 调度：最多 100 epoch，至少 20，patience 15。
- 选择：Source validation。
- 输出：memory_best，并记录 da_best 父 hash 与 anchor hash。

### Stage 4：Source pseudo-label self-training

- 继承：da_best + memory_best。
- 冻结：Memory teacher、GCN、A/S、fusion、discriminator。
- 训练：classifier。
- 伪标签：只生成 Source PL；teacher 输出全部 detach。

$$
\mathcal L_4=
\mathcal L_{sup}^s+\lambda_s\mathcal L_{pl}^s.
$$

- 每 R epoch 刷新；空接收集合对应损失为 0。
- 调度：最多 50 epoch，至少 10，patience 10。
- 选择：Source validation。
- 输出：source_pl_best。

### Stage 5：Dual-domain self-training 与联合微调

- 继承：source_pl_best + memory_best。
- Memory teacher：全阶段 eval、requires_grad=False、不进入 optimizer。
- Target：使用同一 Source anchor bank 生成 Target PL；全部 teacher 输出 detach。
- epoch 1–20：冻结 GCN、A/S、fusion，只训练 classifier 和 discriminator。
- epoch 21 起：解冻 GCN、A/S、fusion，同时训练 classifier 和 discriminator；Memory 保持冻结。

解冻后的目标固定为

$$
\mathcal L_{full}=
\mathcal L_{sup}^s+
\lambda_s\mathcal L_{pl}^s+
\lambda_t\mathcal L_{pl}^t+
\lambda_{adv}\mathcal L_{dom}+
\lambda_{sp}\mathcal L_{sp}.
$$

$\lambda_t$ 在本阶段前 30 epoch 从 0 线性增长至 $\lambda_{t,max}$；$\lambda_{adv}$ 按本阶段进度重新执行 logistic 调度。

每 R epoch：

1. 用当前 encoder 重算 Source/Target 表示。
2. 按固定 anchor IDs 重算 Key/Value。
3. 用冻结 teacher 生成候选。
4. detach prediction、confidence 和 mask。
5. 应用阈值、逐类上限、teacher/student 与跨轮一致性。
6. 保存刷新状态。

- 调度：最多 100 epoch，至少 30，patience 20。
- 选择：Source validation。
- 输出：full_best。

## 4. 配置锁定

只使用 development_seed=2026，Target 类别标签不可见。

优先为全部方向和标签率锁定一套全局配置。只有全局配置在某标签率发生预注册数值/coverage失败，或 rate-specific 配置使六方向 Source validation Macro-F1 平均提高至少 1 个百分点且任一方向下降不超过 1 个百分点时，才允许拆成 1%/3%/5% 三套配置。

禁止维护 18 套方向×标签率配置。机制消融继承 B6 配置，仅改变目标因素。

## 5. Checkpoint 与恢复

每阶段重新建立 optimizer/scheduler，只加入 requires_grad=True 的参数。checkpoint 至少保存：

    stage/epoch/best epoch/patience
    model state by module, requires_grad state
    optimizer and scheduler state
    parent checkpoint path/hash
    resolved config/hash
    dataset/split/METIS/anchor hashes
    pseudo-labels/confidences/masks/refresh history
    GRL progress, lambda_adv, lambda_t
    Python/NumPy/Torch/CUDA RNG states
    best Source validation metrics

恢复时必须校验直接父 checkpoint、配置、数据、split、METIS 和 anchor hash；不一致则拒绝加载。

## 6. 全局不变量

- Target 真实标签不参与训练、伪标签、早停、checkpoint 或调参。
- Memory teacher 在 Stage 4/5 始终冻结。
- pseudo-label、confidence、teacher probability 和 mask 全部 detach。
- GRL 只反转一次，禁止给域损失再加负号。
- 所有 best checkpoint 只依据 Source validation 选择。
