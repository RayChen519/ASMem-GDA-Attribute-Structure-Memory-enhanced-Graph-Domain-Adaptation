# Integration and Verification 实验计划

## 1. Gate

只有本计划全部通过，才能执行 Unified Baseline and Ablation Plan.md 与 Full Experiment Execution and Analysis Plan.md。

验证顺序：

    static config
    → dimensions
    → stage gradients
    → GRL
    → Target sentinel
    → pseudo-label refresh
    → synthetic smoke
    → full-chain smoke
    → PASS

## 2. 维度契约

| 张量 | 形状 |
|---|---|
| $X_d$ | $[N_d,6775]$ |
| $H_d^{(1)}$ | $[N_d,256]$ |
| $Z_d,H_d^A,H_d^{LG},H_d^S,H_d^{AS}$ | $[N_d,128]$ |
| $[H_d^A\Vert H_d^S]$ | $[N_d,256]$ |
| class/domain logits | $[N_d,C]$ / $[N_d,1]$ |
| $Q,M,G$ | $[N,128]$ |
| $K_{mem},V_{mem}$ | $[K,128]$ |
| $E,A$ | $[N,K]$ |

静态断言：

- 第一层 GCN 输入宽度为 6775，GCN 前无额外输入映射。
- Source/Target attribute_union_hash 相同。
- Attribute 为 1 layer/1 head/full-domain full-batch。
- Structure 为 1 BGA block、1 head、P=128、metis_seed=0。
- Attention/FFN 使用 residual、dropout 和 post-norm。
- RestoreOrder 后 node_id 与原图逐节点一致。

## 3. 阶段梯度白名单

| 阶段 | 必须可训练 | 必须冻结或不存在 |
|---|---|---|
| Encoder warm-up | GCN、classifier | A/S、Memory、discriminator |
| A/S DA | GCN、A/S、fusion、classifier、discriminator | Memory |
| Memory warm-up | Memory | GCN、A/S、fusion、classifier、discriminator |
| Source PL | classifier | Memory、GCN、A/S、fusion、discriminator |
| Dual PL epoch 1–20 | classifier、discriminator | Memory、GCN、A/S、fusion |
| Dual PL epoch 21+ | GCN、A/S、fusion、classifier、discriminator | Memory |

每阶段执行一次 forward/backward：

- 可训练模块的相关非空损失必须产生有限梯度。
- 冻结参数必须 requires_grad=False、grad=None 且不在 optimizer。
- 阶段切换后重新检查 optimizer parameter IDs，无重复或冻结参数。

## 4. GRL 测试

对同一 feature、discriminator 和 domain labels 比较普通 backward 与 GRL backward：

- feature-side：$g_{grl}\approx-g_{plain}$。
- discriminator-side：$d_{grl}\approx d_{plain}$。
- 代码与损失配置中不存在第二个负号。

## 5. Target label sentinel

训练只接收 TargetTrainView，不含 y。分别用原标签、随机标签和 sentinel 标签替换隔离评估副本，再运行相同训练。

必须满足：

- deterministic 模式下训练状态与输出严格一致。
- 无法完全确定的 GPU FP32 运算用 atol=1e-6、rtol=1e-5。
- 混合精度用 atol=1e-4、rtol=1e-3。
- split、anchor IDs、接受 mask、checkpoint 选择和元数据严格一致。
- 训练、伪标签、早停、checkpoint、调参和重跑路径均不访问 TargetEvaluationView。

只有相应 final checkpoint 与 run manifest 锁定后才能创建 TargetEvaluationView：

| 变体 | gate |
|---|---|
| B0 | encoder_best |
| B1–B4 | da_best |
| B5 | source_pl_best |
| B6 | full_best |

## 6. Memory 与刷新测试

- pseudo-label、confidence、teacher probability、accepted mask 均无计算图。
- Memory teacher 在 Source/Target PL 生成和 self-training 中 grad=None。
- Target 不进入 anchor bank。
- 逐类上限独立执行；空集合 loss=0。
- 第二轮起检查 teacher/student 和连续两轮类别一致性。
- encoder 解冻后的下一次刷新重算 anchor 表示，anchor IDs 保持不变。
- resume 恢复 refresh index、上一轮预测、mask 和 anchor hash。

## 7. Smoke tests

### 7.1 合成图

构造同类别数、6775 维特征的 Source/Target 小图，设置 $N\ne K$，每阶段执行一个 optimizer step。覆盖：

- 全部维度与梯度白名单。
- 固定 METIS 与 RestoreOrder。
- 空/非空伪标签集合。
- checkpoint 保存、加载和 Target sentinel。

### 7.2 完整链

固定任务：

    ACMv9 → Citationv1
    source_label_rate = 5%
    seed = 0

smoke 配置将每阶段缩短到 1–2 epoch并减小 K，但不得跳过阶段、冻结切换、checkpoint 加载、伪标签刷新或 final-evaluation 入口。

必须依次产生并加载：

$$
\text{encoder_best}
\rightarrow\text{da_best}
\rightarrow\text{memory_best}
\rightarrow\text{source_pl_best}
\rightarrow\text{full_best}
\rightarrow\text{final evaluation}.
$$

每个继承点校验模型字段、optimizer/scheduler、父 hash、配置 hash、split、anchor 和维度。

## 8. PASS report

报告至少包含：

    dimension_contract
    gradient_whitelist
    grl_single_reversal
    target_label_sentinel
    memory_teacher_frozen
    pseudo_label_detach
    refresh_and_resume
    checkpoint_lineage
    smoke_final_evaluation

任一项失败则阻断正式实验。性能提升不是 integration gate。

