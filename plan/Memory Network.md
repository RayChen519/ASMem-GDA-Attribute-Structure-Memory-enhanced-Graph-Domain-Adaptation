# Memory Network 实现计划

## 1. 接口

输入来自 Attribute and Structure level.md：

$$
H_d^S,H_d^{AS}\in\mathbb R^{N_d\times128},\qquad d\in\{s,t\}.
$$

- $H^S$ 用于结构 Query/Key。
- $H^{AS}$ 用于 Memory Value、Query residual 和节点分类。
- Memory Key/Value 只来自 K 个 Source anchors。
- Target 节点不得进入 anchor bank。

## 2. Source anchor bank

只从 $V_s^U$ 无放回采样：

$$
w_v=\alpha^{\log(d_v+1)}+\beta,\qquad
p(v)=\frac{w_v}{\sum_{u\in V_s^U}w_u}.
$$

$d_v$ 使用对称化后、加自环前的 Source degree。默认初始配置：

    K = 128
    alpha = 2
    beta = 1

K、alpha、beta 最终只依据独立 development split 的 Source validation 锁定。保存 anchor IDs 和 hash；Target 永不作为 anchor。

Memory warm-up 时 encoder 冻结，可缓存 anchor 表示。联合阶段 encoder 解冻后，anchor IDs 不变，但每次伪标签刷新重新计算其表示、Key 和 Value。

## 3. Memory 前向

一次包含 N 个 Query 节点的前向必须满足：

$$
Q\in\mathbb R^{N\times128},\qquad
K_{mem},V_{mem}\in\mathbb R^{K\times128}.
$$

定义

$$
Q=\operatorname{norm}(H_d^SW_Q),
$$

$$
K_{mem}=\operatorname{norm}(H_{s,\mathcal A_s}^SW_K),
\qquad
V_{mem}=H_{s,\mathcal A_s}^{AS}W_V.
$$

相似度与读取：

$$
E=\frac{QK_{mem}^{\top}}{T}\in\mathbb R^{N\times K},
$$

$$
A=\operatorname{Softmax}(E)\in\mathbb R^{N\times K},
\qquad
M=AV_{mem}\in\mathbb R^{N\times128}.
$$

若 Source Query 本身是 anchor，将其自匹配 logit 置为 $-\infty$。

Query 与 Memory 融合：

$$
G=\operatorname{LayerNorm}\left(
H_d^{AS}+W_F[H_d^{AS}\Vert M]
\right)\in\mathbb R^{N\times128},
$$

$$
P_M(y\mid d)=\operatorname{Softmax}(GW_C+b_C)
\in\mathbb R^{N\times C}.
$$

默认 temperature $T=0.1$。

## 4. Memory warm-up

加载 da_best，冻结 GCN、Attribute、Structure、fusion、classifier 和 discriminator，只训练

$$
\Theta_M=\{W_Q,W_K,W_V,W_F,W_C,b_C\}.
$$

监督只来自 Source training Query：

$$
\mathcal L_{mem}=
\operatorname{CE}\left(P_M(y\mid V_s^{L,tr}),y_s^{L,tr}\right).
$$

Source validation 只用于早停和保存 memory_best。Target 真实标签不进入进程。

## 5. 伪标签生成与筛选

对 domain $d$：

$$
\widehat y_i^d=\arg\max_c P_M(y=c\mid i,d),\qquad
c_i^d=\max_cP_M(y=c\mid i,d).
$$

默认初始配置：

    gamma = 0.90
    q = 0.20
    refresh interval R = 10

筛选顺序：

1. 保留 $c_i^d\ge\gamma$ 的节点。
2. 按预测类别分别排序，并应用相同的逐类接收比例上限 q。
3. 第一轮只使用 Memory 置信度。
4. 第二轮起要求 Memory teacher 与当前 classifier 类别一致。
5. 后续刷新还要求连续两轮预测类别一致。
6. 空接收集合对应损失严格为 0。

Source 先对 $V_s^U$ 生成伪标签；Source self-training 完成后，再对全部 $V_t$ 生成 Target 伪标签。Target 使用同一个 Source anchor bank和由 Source validation 锁定的 gamma/q。

伪标签、置信度、teacher probability 和 accepted mask 在进入筛选、缓存或损失前全部 detach。

## 6. 伪标签监督

分类器概率：

$$
P_C(y\mid i,d)=\operatorname{Softmax}(C_\psi(H_{d,i}^{AS})).
$$

损失：

$$
\mathcal L_{sup}^s=
\operatorname{CE}(P_C(V_s^{L,tr}),y_s^{L,tr}),
$$

$$
\mathcal L_{pl}^s=
-\frac1{|V_s^{PL}|}\sum_{i\in V_s^{PL}}
c_i^s\log P_C(\widehat y_i^s\mid i,s),
$$

$$
\mathcal L_{pl}^t=
-\frac1{|V_t^{PL}|}\sum_{i\in V_t^{PL}}
c_i^t\log P_C(\widehat y_i^t\mid i,t).
$$

Source PL 阶段使用

$$
\mathcal L_{sup}^s+\lambda_s\mathcal L_{pl}^s.
$$

Dual-domain 阶段使用

$$
\mathcal L_{sup}^s+
\lambda_s\mathcal L_{pl}^s+
\lambda_t\mathcal L_{pl}^t
$$

并由 End-to-End Training Plan.md 加入域对抗与稀疏损失。

Memory teacher 在全部伪标签生成和 self-training 中保持 eval、requires_grad=False 且不进入 optimizer。自身生成的伪标签不得更新 Memory 参数。

## 7. 刷新状态

每次刷新保存：

    anchor IDs/hash
    refresh index
    pseudo-labels, confidences, accepted masks
    previous-round predictions
    per-class coverage
    anchor representation hash

encoder 解冻后先重算 Source/Target 与 anchor 表示，再由冻结 teacher 生成候选。恢复训练时必须恢复上一轮预测和 refresh index。

## 8. 验收

- Query/融合/读取为 [N,128]，Key/Value 为 [K,128]，相似度为 [N,K]。
- anchor 只来自 $V_s^U$ 且无重复。
- Target 标签不参与生成、筛选或调参。
- 所有 teacher 输出无计算图，Memory teacher 无梯度。
- 空集合、逐类上限、两轮一致性和恢复逻辑可运行。

