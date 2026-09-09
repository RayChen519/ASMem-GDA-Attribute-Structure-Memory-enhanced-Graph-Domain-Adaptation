# Attribute and Structure Level 实验计划

## 1. 接口与固定结构

输入为 Shared GCN Encoder Plan.md 的

$$
Z_d\in\mathbb R^{N_d\times128},\qquad d\in\{s,t\}.
$$

输出：

$$
H_d^A,H_d^{LG},H_d^S,H_d^{AS}\in\mathbb R^{N_d\times128}.
$$

主实验固定：

    hidden_dim = 128
    attention heads = 1
    BGA blocks = 1
    attention/FFN dropout = 0.1
    residual = true
    LayerNorm = true
    norm = post-norm
    MLP_S = 128→128→128, ReLU, dropout 0.1
    FFN = 128→512→128, ReLU, dropout 0.1
    METIS P = 128, seed = 0

## 2. Selection 算子

对 $Y\in\mathbb R^{n\times128}$，使用单层、单头 self-attention：

$$
\operatorname{Attn}(Y)=
\operatorname{Softmax}\left(\frac{YW_Q(YW_K)^\top}{\sqrt{128}}\right)YW_VW_O.
$$

Attention 和 FFN 均按 post-norm 执行：

$$
h=\operatorname{LN}\left(Y+\operatorname{Dropout}_{0.1}(\operatorname{Attn}(Y))\right),
$$

$$
U(Y)=\operatorname{LN}\left(
h+\operatorname{Dropout}_{0.1}(\operatorname{FFN}(h))
\right),
$$

其中 FFN 为 128→512→128、ReLU、dropout 0.1。

软阈值和 selection：

$$
\tau(x;\theta)=\operatorname{sgn}(x)\odot\max(0,|x|-\theta),
$$

$$
\operatorname{Sel}(Y;\theta)=\tau(U(Y);\theta)\odot Y,\qquad
\theta=\operatorname{softplus}(\rho).
$$

Attribute 与 Structure 分别使用独立的 $U_A,\theta_A$ 和 $U_S,\theta_S$。

## 3. Attribute Level

主实验只筛选 Source：

$$
H_s^A=\tau(U_A(Z_s);\theta_A)\odot Z_s,\qquad
H_t^A=Z_t.
$$

Attribute attention 对当前 domain 全部节点做 full-domain full-batch 计算，禁止随机节点 mini-batch。同一变体不得按 run 自动切换实现。

只有记录到明确 OOM 时才允许注册独立的 attribute_partitioned_attention_oom 变体；其节点划分固定，结果不得混入 full-batch 主实验。

## 4. Structure Level

### 4.1 METIS 与输入

分别将 Source/Target 划分为 128 个互不重叠簇。partition 按 dataset/domain 缓存，并在所有方向、标签率、seeds 和变体间复用。

$$
H_{d,p}^{0}=\operatorname{MLP}_S(Z_{d,p}),
$$

其中

    Linear(128,128) → ReLU → Dropout(0.1) → Linear(128,128)

Source/Target 共享 MLP_S、BGA 和融合参数。

### 4.2 单个 BGA block

对每个簇做一次单头 intra-cluster attention：

$$
L_{d,p}=\operatorname{LN}\left(
H_{d,p}^{0}+\operatorname{Dropout}_{0.1}(\operatorname{Attn}_{intra}(H_{d,p}^{0}))
\right),
$$

$$
\widehat H_{d,p}=\operatorname{LN}\left(
L_{d,p}+\operatorname{Dropout}_{0.1}(\operatorname{FFN}_{intra}(L_{d,p}))
\right).
$$

对每个簇均值池化：

$$
p_{d,p}=\operatorname{MEAN}(\widehat H_{d,p}),\qquad
P_d=[p_{d,1};\ldots;p_{d,128}]\in\mathbb R^{128\times128}.
$$

在 128 个簇表示之间做一次单头 inter-cluster attention：

$$
L_{P,d}=\operatorname{LN}\left(
P_d+\operatorname{Dropout}_{0.1}(\operatorname{Attn}_{inter}(P_d))
\right),
$$

$$
\widehat P_d=\operatorname{LN}\left(
L_{P,d}+\operatorname{Dropout}_{0.1}(\operatorname{FFN}_{inter}(L_{P,d}))
\right).
$$

将第 $p$ 个全局簇表示广播到该簇节点并融合：

$$
H_{d,p}^{1}=
[\widehat H_{d,p}\Vert\mathbf1_{N_{d,p}}\widehat p_{d,p}]W_f,
\qquad W_f\in\mathbb R^{256\times128}.
$$

按 node_id 恢复原顺序：

$$
H_d^{LG}=\operatorname{RestoreOrder}(\{H_{d,p}^{1}\})
\in\mathbb R^{N_d\times128}.
$$

主实验固定 P=128、metis_seed=0；P 只在 sensitivity analysis 中取 64/128/256。

### 4.3 Structure selection

$$
H_s^S=\tau(U_S(H_s^{LG});\theta_S)\odot H_s^{LG},
\qquad
H_t^S=H_t^{LG}.
$$

Target 不做 selection；对称 selection 仅作为消融。

## 5. 拼接、线性融合与域对抗

$$
H_d^{AS}=[H_d^A\Vert H_d^S]W_{AS}+b_{AS},
\qquad W_{AS}\in\mathbb R^{256\times128}.
$$

Source 分类损失只使用 $V_s^{L,tr}$：

$$
\mathcal L_{cls}^s=
\operatorname{CE}\left(
C_\psi(H_{s,V_s^{L,tr}}^{AS}),y_{s,V_s^{L,tr}}
\right).
$$

域判别器接收经一个 GRL 的 $H_s^{AS},H_t^{AS}$：

$$
\mathcal L_{DA}=
\mathcal L_{cls}^s+
\lambda_{adv}\mathcal L_{dom}+
\lambda_{sp}\mathcal L_{sp},
$$

$$
\mathcal L_{sp}=
\frac{\|\tau(U_A(Z_s);\theta_A)\|_1+
\|\tau(U_S(H_s^{LG});\theta_S)\|_1}{N_s\cdot128}.
$$

GRL 只反转一次 feature-side 域梯度；总损失中的域损失保持正号，禁止再手工取负。Target 只提供特征、图结构和域标签。

## 6. 阶段接口

1. 加载 encoder_best。
2. 计算 Source/Target 的 Z。
3. 执行 Attribute selection。
4. 执行 Structure local/global BGA 与 selection。
5. 拼接并线性变换为 $H^{AS}$。
6. 联合优化 GCN、A/S、fusion、classifier、discriminator。
7. 仅依据 Source validation 保存 da_best。
8. 将 $H_s^S,H_t^S,H_s^{AS},H_t^{AS}$ 交给 Memory Network.md。

epoch、学习率、GRL 调度和冻结边界由 End-to-End Training Plan.md 管理。

## 7. 验收

- 所有节点级输出均为 [N_d,128] 且按原 node_id 排列。
- Attribute 使用 full-domain attention；Structure 固定一个 BGA block和一个 head。
- METIS partition 固定为 P=128、seed=0。
- $\theta_A,\theta_S$ 非负且参数独立。
- Target 真实标签未被读取。
- 域损失只经过一个 GRL。

