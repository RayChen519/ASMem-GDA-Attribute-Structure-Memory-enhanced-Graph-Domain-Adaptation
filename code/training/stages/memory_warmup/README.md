# Memory 与伪标签阶段

`models/memory/network.py` 实现 P4 的结构 Query/Key、Source H^AS Value、温度读取、残差融合和分类；训练直接使用 logits 交叉熵。`data/anchors/source.py` 复用 Dataset 的 SourceTrainView、原始 degree 和缓存契约。`training/pseudo_labels/selection.py` 负责筛选，`training/losses/pseudo_labels.py` 负责置信度加权监督。

`MemoryTraining` 是 Stage 3–5 的共享执行器，复用 `DomainAdaptation.representations`，不重新定义数据或 A/S 模型。三个阶段入口分别位于 `memory_warmup`、`source_self_training`、`dual_domain_finetuning`。每阶段重新建立 AdamW/scheduler；只有 Source validation 参与选优。Stage 4/5 的 Memory 始终 eval、冻结、无梯度且不进入 optimizer。

## 明确记录的解释

- LayerNorm 不带可训练 affine 参数，因为 P4 的参数白名单只有 W_Q/W_K/W_V/W_F/W_C/b_C。前三种投影和融合不加未列出的 bias。
- 逐类上限为 `floor(q * 该域候选集合中预测为该类的节点数)`；先筛置信度，再按置信度排序截取上限。相同置信度按 node ID 升序。之后才执行一致性，不补录被一致性淘汰的名额。计划未给出分母与取整，采用此显式解释。
- 依 P6 更明确的要求，第二轮起同时执行 teacher/classifier 和连续两轮类别一致性。
- 无可用 anchor 的全屏蔽行读取零向量；K 超过 Source 未标注数量直接拒绝，不静默缩小 K。
- 刷新发生在阶段 epoch 1、1+R、… 的训练更新之前。Stage 5 延续 Source best 保存的 Source 轮次，Target 从第零轮开始；只在 Source latest 证明阶段已结束、且 best 匹配后进入 Stage 5。
- Stage 5 前 20 epoch 也计算完整目标，但表示模块冻结，因此稀疏项不更新它们；域项更新 discriminator。lambda_t 从 epoch 1 的 0 到 epoch 30 的 0.5。
- epoch 21 重建 optimizer，保留 classifier/discriminator 的 Adam moments，为新解冻模块建立新状态；使用解冻后的基础学习率并延续阶段内 scheduler 时钟，不另起 warm-up。新阶段则完全重建 optimizer/scheduler。
- 正式配置采用计划默认值，尚未经过 development_seed=2026 锁定；当前不支持未经注册的超参数更改，也不声称完成正式实验准入。

## 配置、缓存与恢复

`configs/base/{memory_warmup,source_self_training,dual_domain_finetuning}.json` 给出正式 epoch/min/patience，其余默认值由 `MemoryConfig` 校验。`configs/overrides/smoke/memory.json` 仅用于独立的 1–3 epoch 冒烟（K=8、R=1）；不会覆盖正式配置。

checkpoint 保存模块/冻结状态、optimizer/scheduler、早停/历史、Python/NumPy/Torch/CUDA RNG、直接父路径/hash 与递归父链、配置、Dataset/split/METIS/anchor/data hash、anchor 表示/Key/Value/hash、每域轮次、上一轮预测、当前概率/标签/置信度/mask、逐类 coverage 和刷新历史。恢复只允许原输出目录，先验证 latest/best 与契约，再加载。下一次刷新从保存的固定 IDs 重算表示及 Key/Value。

所有训练入口只使用 Source train、Source validation 和无真实标签的 TargetTrainView；不创建 TargetEvaluationView。最终评估和完整 Gate 是独立验收，本模块不会自动宣布 PASS。

## 运行

PowerShell 工作目录为 `D:\Codex\Projects\SGDA_0826\code`，统一使用 `..\.venv\Scripts\python.exe`。

```powershell
..\.venv\Scripts\python.exe -m verification.smoke.synthetic.memory --output artifacts/runs/smoke/memory_new
..\.venv\Scripts\python.exe -m training.stages.memory_warmup --da-best <da_best.pt> --output <new-memory-dir> --smoke --smoke-epochs 1
..\.venv\Scripts\python.exe -m training.stages.source_self_training --da-best <da_best.pt> --memory-best <memory_best.pt> --output <new-source-dir> --smoke --smoke-epochs 1
..\.venv\Scripts\python.exe -m training.stages.dual_domain_finetuning --da-best <da_best.pt> --memory-best <memory_best.pt> --source-pl-best <source_pl_best.pt> --output <new-dual-dir> --smoke --smoke-epochs 1
..\.venv\Scripts\python.exe -m pytest verification -q --basetemp=artifacts/runs/smoke/<new-test-directory>
```

恢复使用同一命令追加 `--resume <stage_latest.pt>`。smoke checkpoint 无法初始化正式训练。所有新运行必须使用新的输出目录。
