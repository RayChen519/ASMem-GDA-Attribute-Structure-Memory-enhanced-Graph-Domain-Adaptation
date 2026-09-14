# Memory / Stage 3–5 实现与验证记录（2026-09-14）

## 结果与范围

已实现 Memory、Source-only anchor、伪标签筛选/加权损失、Memory warm-up、Source self-training、Dual fine-tuning 和对应 checkpoint 保存/恢复。所有新增训练产物位于 `code/artifacts/runs/smoke/`，未更改已有正式配置/checkpoint，未运行完整正式训练或实验矩阵。

本次是功能与阶段衔接验证，**完整 Integration Gate 为 NOT_RUN**。未执行 final Target evaluation、development 配置锁定、GPU/CUDA 验证、正式训练收敛与性能比较。当前无阻止本次实现交付的未解决失败。

## 文件

- 模型/数据：`models/memory/network.py`、`data/anchors/source.py`。
- 契约/训练：`contracts/training/memory.py`、`training/pseudo_labels/selection.py`、`training/losses/pseudo_labels.py`、`training/optimization/memory.py`、`training/checkpoints/memory.py`。
- 阶段：`training/stages/memory_warmup/{trainer.py,__main__.py,README.md}`；Source/Dual 各自目录的 `__main__.py` 调用共享执行器。
- 配置：三个新 `configs/base/*` 阶段配置及 `configs/overrides/smoke/memory.json`。
- 验证：`verification/memory_refresh/test_memory.py`、`verification/checkpoint_resume/test_memory_resume.py`、`verification/smoke/synthetic/memory.py`。

具体假设（逐类上限分母/取整、LayerNorm affine、全屏蔽行、跨阶段轮次、解冻时 optimizer/scheduler 处理）见 `training/stages/memory_warmup/README.md`，同时保存在 resolved config 中。

## 环境与命令

工作目录 `D:\Codex\Projects\SGDA_0826\code`。解释器为 `D:\Codex\Projects\SGDA_0826\.venv\Scripts\python.exe`；实际设备 CPU，PyTorch 2.14.0+cpu，CUDA unavailable。NumPy 2.5.3、SciPy 1.18.1、pytest 9.1.1；`pip check` 无损坏依赖。

```powershell
..\.venv\Scripts\python.exe -m pip check
..\.venv\Scripts\python.exe -m pytest verification -q --basetemp=artifacts/runs/smoke/memory_regression_20260914_a --junitxml=artifacts/reports/integration/memory-regression-20260914.xml
..\.venv\Scripts\python.exe -m pytest verification/memory_refresh verification/checkpoint_resume/test_memory_resume.py -q --basetemp=artifacts/runs/smoke/memory_final_20260914 --junitxml=artifacts/reports/integration/memory-final-20260914.xml
..\.venv\Scripts\python.exe -m pytest verification/checkpoint_resume/test_memory_resume.py -q -k nonempty --basetemp=artifacts/runs/smoke/memory_nonempty_20260914 --junitxml=artifacts/reports/integration/memory-nonempty-20260914.xml
..\.venv\Scripts\python.exe -m verification.smoke.synthetic.memory --output artifacts/runs/smoke/memory_synthetic_20260914_verified
```

回归结果为 **126 passed**；随后针对最终 checkpoint 加固及新增测试执行 **24 passed**，再补充非空伪标签训练路径 **1 passed**。这些集合存在重叠，不应直接相加。最终新增 Memory 专项共 25 项通过。所有 XML 的 errors/failures/skipped 均为 0。`git diff --check` 无报错。

首次恢复测试遇到系统 pytest 临时目录访问限制，改用项目内独立 basetemp；解冻边界测试的 Stage 5 patience 从误用 15 修正为计划规定的 20 后通过。没有放宽正式配置来使测试通过。

## 衔接冒烟

合成图为 256/257 节点、6775 维、P=128，encoder/DA 各 1 epoch，Memory/Source PL/Dual PL 各 3 epoch，K=8、R=1。依次产生选优 checkpoint，并在各阶段读取 latest 恢复；当前版本输出在 `artifacts/runs/smoke/memory_synthetic_20260914_verified/`。三个新增阶段 best epoch 分别为 2、3、3。

真实任务为已有 `da_acm_citation_seed0/da_best.pt` 对应的 ACMv9→Citationv1、5%、seed=0，继承已有 encoder/DA 冒烟产物，本次没有重跑真实 Stage 1/2。新增阶段各 1 epoch：

```powershell
..\.venv\Scripts\python.exe -m training.stages.memory_warmup --da-best artifacts/runs/smoke/da_acm_citation_seed0/da_best.pt --output artifacts/runs/smoke/memory_real_20260914_verified/memory --smoke --smoke-epochs 1 --device cpu --threads 2
..\.venv\Scripts\python.exe -m training.stages.source_self_training --da-best artifacts/runs/smoke/da_acm_citation_seed0/da_best.pt --memory-best artifacts/runs/smoke/memory_real_20260914_verified/memory/memory_best.pt --output artifacts/runs/smoke/memory_real_20260914_verified/source --smoke --smoke-epochs 1 --device cpu --threads 2
..\.venv\Scripts\python.exe -m training.stages.dual_domain_finetuning --da-best artifacts/runs/smoke/da_acm_citation_seed0/da_best.pt --memory-best artifacts/runs/smoke/memory_real_20260914_verified/memory/memory_best.pt --source-pl-best artifacts/runs/smoke/memory_real_20260914_verified/source/source_pl_best.pt --output artifacts/runs/smoke/memory_real_20260914_verified/dual --smoke --smoke-epochs 1 --device cpu --threads 2
```

最后一条命令追加 `--resume artifacts/runs/smoke/memory_real_20260914_verified/dual/full_latest.pt` 也已成功。三个 best 都是 epoch 1，同一 anchor hash 为 `41117f8e38c12ffb2550689e2059b392b0b4bd08ccb8f6d8324438841b80b319`。Source PL 轮次 0，Dual 中 Source 轮次 1、Target 轮次 0。

真实短冒烟的 Source/Target 接收集合均为空，保持 gamma=0.90，没有降低门槛。这只能证明真实数据的空集合与衔接路径；非空伪标签路径由明确注入高置信候选的单元测试覆盖，不代表真实训练已获得有效伪标签或性能提升。

## 要求对应证据

| 要求 | 验证证据 |
| --- | --- |
| Memory 公式/维度、Source 自匹配、Target ID 重名不屏蔽 | `test_formula_mask_and_gradients`、`test_all_masked_and_target_id_collision` |
| anchor 来源、权重、degree、固定 IDs/cache、数量不足 | `test_anchor_cache_source_degree_shortage`、`test_weighted_sampler_matches_plan` |
| 阶段冻结/梯度与 checkpoint 继承 | `test_exact_resume_and_frozen_parameters` 三阶段参数化测试、合成/真实冒烟 |
| 筛选顺序、逐类 cap、跨轮一致性、detach、空/非空加权损失 | `test_cap_before_consistency_and_detach`、`test_nonempty_pseudo_labels_train_classifier_only` |
| Source 完成后才生成 Target、Target 标签隔离 | `test_dual_rejects_unfinished_source`、`test_target_label_sentinel_all_three_stages`；替换隔离标签副本为原/重排/sentinel，并禁止读取 evaluation_only |
| epoch 20→21 解冻与刷新、teacher 不变 | `test_epoch20_21_gradients_refresh_and_teacher`；仅两次 optimizer step，未训练 21 epoch |
| 恢复 RNG/optimizer/模型/PL/refresh 全状态一致 | 三阶段连续 3 epoch 与 1 epoch 后恢复至 3 epoch 的完整 payload 精确比较 |
| 拒绝损坏 checkpoint | anchor/representation/mask/round/optimizer/scheduler/mode/config/RNG/teacher 参数化拒载，且失败不改变 encoder |
| 正式早停与 tie-break | 三阶段 `test_formal_early_stop_loop_without_training`；替代训练步与 checkpoint 写入，仅验证控制流 |
| 既有 Dataset/Encoder/DA、维度/GRL 回归 | 126 项整套回归 XML；未把这些局部证据汇总为完整 Gate PASS |
