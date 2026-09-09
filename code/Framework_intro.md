# 项目代码目录结构

```text
code/
├── contracts/                         # 统一跨模块接口与元数据契约
│   ├── data/                          # 图、数据视图、split/hash 契约 [P1]
│   ├── models/                        # 模型输入输出及维度契约 [P2–P4,P6]
│   ├── training/                      # 阶段、checkpoint、刷新状态契约 [P4,P5]
│   └── experiments/                   # 变体、run manifest、评估结果契约 [P7,P8]
│
├── data/                              # 数据处理逻辑 [P1]
│   ├── loaders/                       # 三个数据集的读取适配 [P1]
│   ├── preprocessing/                 # 节点重编号、类别对齐、边与邻接处理 [P1]
│   ├── features/                      # 6775 维 vocabulary 对齐与归一化 [P1]
│   ├── splits/                        # Source 划分、mask 与跨实验复用 [P1]
│   ├── views/                         # 训练数据视图与标签访问隔离 [P1,P6]
│   ├── partitions/                    # METIS 划分及节点顺序映射 [P1,P3]
│   ├── anchors/                       # Source anchor 抽样与固定 ID [P4,P7]
│   └── cache/                         # 数据缓存键、校验与失效逻辑 [P1]
│
├── models/                            # 模型组件及前向计算 [P2–P4,P7]
│   ├── encoders/                      # 两层共享 GCN [P2]
│   ├── classifiers/                   # 跨阶段继承的分类头 [P2,P5]
│   ├── selection/                     # scorer、软阈值与 selection 算子 [P3,P7]
│   ├── attribute/                     # Attribute 分支与全域注意力 [P3]
│   ├── structure/                     # BGA、簇内/簇间注意力和顺序恢复 [P3]
│   ├── fusion/                        # Attribute/Structure 拼接与线性融合 [P3]
│   ├── adversarial/                   # GRL 与域判别器 [P3,P5]
│   ├── memory/                        # Query/Key/Value、读取与 Memory 分类 [P4]
│   └── adapters/                      # B1–B3 表示适配层 [P7]
│
├── training/                          # 训练流程与状态管理 [P5]
│   ├── stages/                        # 五阶段执行与阶段切换 [P5]
│   │   ├── encoder_warmup/            # Stage 1 → encoder_best [P2,P5]
│   │   ├── domain_adaptation/         # Stage 2 → da_best [P3,P5]
│   │   ├── memory_warmup/             # Stage 3 → memory_best [P4,P5]
│   │   ├── source_self_training/      # Stage 4 → source_pl_best [P4,P5]
│   │   └── dual_domain_finetuning/    # Stage 5 → full_best [P4,P5]
│   ├── losses/                        # 监督、域对抗、稀疏及伪标签损失 [P2–P5]
│   ├── optimization/                  # 参数组、冻结/解冻和梯度裁剪 [P5]
│   ├── schedules/                     # 学习率、GRL 与损失权重调度 [P5]
│   ├── checkpoints/                   # best/latest、父链校验和恢复 [P5,P8]
│   └── pseudo_labels/                 # 冻结 teacher、筛选、一致性与刷新 [P4,P5]
│
├── evaluation/                        # 验证与独立最终评估 [P1,P5,P8]
│   ├── source_validation/             # Source 验证及模型选择信号 [P2–P5]
│   ├── final_target/                  # 锁定后独立创建评估视图并评估 Target [P1,P6,P8]
│   ├── metrics/                       # Accuracy、Macro-F1 等指标 [P5,P8]
│   └── diagnostics/                   # gate 后的伪标签准确率等离线诊断 [P8]
│
├── experiments/                       # 实验组织与结果分析 [P7,P8]
│   ├── registry/                      # B0–B6、A1–M6 及独立变体注册 [P3,P7]
│   ├── development/                   # 开发划分调参与配置锁定 [P1,P5,P7]
│   ├── manifests/                     # 确定性 run_id、清单锁定和失效管理 [P8]
│   ├── scheduling/                    # 矩阵调度、前置检查与失败恢复 [P7,P8]
│   └── analysis/                      # 五-seed 汇总、显著性、负迁移及效率分析 [P8]
│
├── configs/                           # 后续配置文件存放位置 [P5,P7,P8]
│   ├── base/                          # 数据、模型、训练的基础配置 [P1–P5]
│   ├── variants/                      # 核心变体和单因素消融覆盖项 [P7]
│   ├── locked/                        # development 验证后锁定的配置 [P5,P8]
│   ├── rates/                         # 满足 Plan 条件时的标签率配置 [P5,P8]
│   └── overrides/                     # 显式例外配置 [P6–P8]
│       ├── smoke/                     # 冒烟运行配置 [P6]
│       └── sensitivity/               # 独立敏感性实验配置 [P7,P8]
│
├── verification/                      # 集成验收与正式实验准入 [P6]
│   ├── static_config/                 # 固定结构及配置约束检查 [P6]
│   ├── dimensions/                    # 张量维度与节点顺序检查 [P6]
│   ├── gradients/                     # 阶段梯度白名单、optimizer 参数检查 [P6]
│   ├── grl/                           # 单次梯度反转验证 [P6]
│   ├── label_isolation/               # Target sentinel 与标签访问隔离 [P6]
│   ├── memory_refresh/                # teacher 冻结、detach 和刷新验证 [P6]
│   ├── checkpoint_resume/             # checkpoint 父链与恢复一致性 [P6]
│   ├── smoke/                         # 冒烟验收 [P6]
│   │   ├── synthetic/                 # 合成图与边界情况 [P6]
│   │   └── full_chain/                # 五阶段及最终评估完整链 [P6]
│   └── gate/                          # 汇总验收项、生成 PASS/FAIL 结果 [P6,P8]
│
├── utils/                             # 无业务流程的公共基础工具 [P1–P8]
│   ├── configuration/                 # 配置合并与解析 [P5,P8]
│   ├── hashing/                       # 数据、配置、产物的稳定 hash [P1,P5,P8]
│   ├── randomness/                    # seeds、确定性及 RNG 状态 [P1,P5,P8]
│   ├── io/                            # 序列化、完整性检查和原子写入 [P5,P8]
│   ├── logging/                       # 运行、失败重试及访问审计日志 [P8]
│   └── runtime/                       # 环境信息、计时、显存与参数量采集 [P8]
│
└── artifacts/                         # 后续生成的数据和运行产物 [P1,P6–P8]
    ├── datasets/                      # 数据资产 [P1]
    │   ├── raw/                       # 原始数据
    │   ├── processed/                 # 处理后的统一图与特征
    │   └── manifests/                 # 数据统计、映射和版本/hash 清单
    ├── cache/                         # 特征、split、METIS、anchor 等缓存 [P1,P4]
    ├── runs/                          # 各 run 的配置、checkpoint、日志与指标 [P8]
    │   ├── development/               # 独立开发运行 [P5,P7]
    │   ├── smoke/                     # 冒烟运行 [P6]
    │   ├── core/                      # 核心矩阵，630 runs [P7,P8]
    │   ├── ablations/                 # 机制消融，990 runs [P7,P8]
    │   └── sensitivity/               # 敏感性新增运行，180 runs [P7,P8]
    └── reports/                       # 汇总报告 [P6,P8]
        ├── integration/               # 集成验收报告 [P6]
        └── analysis/                  # 统计、负迁移、效率和敏感性报告 [P8]
```

# Plan 编号与文件对应关系

| 编号 | 文件名 |
| ---- | ------ |
| P1 | `Dataset.md` |
| P2 | `Shared GCN Encoder Plan.md` |
| P3 | `Attribute and Structure level.md` |
| P4 | `Memory Network.md` |
| P5 | `End-to-End Training Plan.md` |
| P6 | `Integration and Verification Plan.md` |
| P7 | `Unified Baseline and Ablation Plan.md` |
| P8 | `Full Experiment Execution and Analysis Plan.md` |