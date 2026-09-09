# Shared GCN / Encoder warm-up

实现范围为 P2 和 Stage 1 必需的 P5 配套；不创建 A/S、discriminator、Memory、伪标签或最终评估视图，不产生完整 Integration Gate PASS。

## 模型接口

```python
from models.encoders.shared_gcn import SharedGCNEncoder
from models.classifiers.shared import SharedClassifier

encoder = SharedGCNEncoder()
classifier = SharedClassifier(num_classes)
source = encoder(source_view.graph.x, source_view.graph.normalized_adjacency)
# source.h1: [Ns, 256]，第一层 LN / ReLU / Dropout 后
# source.z:  [Ns, 128]，第二层 LN 后，无 ReLU，供 A/S 输入
logits = classifier(source.z)  # [Ns, C]
```

`encoder.forward_pair(xs, as_hat, xt, at_hat)` 是 DA 的共享参数入口；两个域调用的是同一个对象，无域专属映射或邻接缓存。Stage 1 的 Target 只能调用 `inspect_target`，其检查限定为 eval + no_grad 下的维度/有限值，并恢复每个子模块的原训练模式。

GCN 直接执行 `A_hat @ (X @ W) + b`；GCN 权重 Xavier uniform、bias 0，LayerNorm 使用 PyTorch 默认的 weight=1 / bias=0。分类头使用 PyTorch Linear 的默认初始化（计划未另行规定）。仅数据模块负责自环和对称归一化；不要把原始 `edge_index` 传入编码器。

## 训练与命令

从项目根目录运行测试：

```powershell
.venv/Scripts/python.exe -m pytest code/verification/dataset/test_dataset.py code/verification/checkpoint_resume/test_encoder.py -q
```

以下命令从 `code/` 运行。合成图和真实数据的冒烟均仅执行 Stage 1：

```powershell
../.venv/Scripts/python.exe -m verification.smoke.synthetic.encoder --output artifacts/runs/smoke/encoder_synthetic
../.venv/Scripts/python.exe -m training.stages.encoder_warmup --smoke --output artifacts/runs/smoke/encoder_acm_citation
```

不传 `--smoke` 即使用 100/20/15 的 max/min/patience；默认方向 ACMv9→Citationv1、5%、seed=0。`--manifest` 可显式固定版本化 manifest，`--source/--target/--label-rate/--seed/--device` 可选择现有任务和设备，不重新 split。输出目录要求为空；正式实验仍须先通过后续完整 Gate，不能用本阶段 smoke 代替准入。

配置唯一来源为 `contracts.training.encoder.WarmupConfig`，每个 CLI run 保存完整 `resolved_config.json` 及 hash。两个 AdamW 参数组分别为 GCN 5e-4、classifier 1e-3，weight_decay=5e-4，整体梯度 norm 裁剪到 5。前五轮使用基础 LR 的 0.2/0.4/0.6/0.8/1.0，随后余弦下降，在 max_epochs 达到 0.1；两轮 smoke 只执行线性预热的前两轮，不改变五轮预热定义。

只对 Source train ID 的 Z 计算交叉熵。Source validation 使用独立视图、eval/no_grad 和所有 C 类的 Macro-F1，精确相同 F1 时较小 validation loss 优先；两项都相同视为无改进。min_epochs 约束停止时间，不限制最佳模型来自早期 epoch。

## checkpoint 与阶段衔接

- `encoder_best.pt/.json`：Source validation 选优的完整状态与 SHA256 记录。
- `encoder_latest.pt/.json`：每轮验证及 scheduler.step 完成后的完整状态，用于续训。
- 保存模型、optimizer、scheduler、early-stop、历史指标、Python/NumPy/Torch/CUDA RNG、每个参数的 requires_grad、各子模块 train/eval 模式，以及方向/seed/label_rate、维度/类别数、vocabulary/dataset/config/split/mask hashes。
- Stage 1 的 parent/METIS/anchor 明确为 null，因为这些模块尚未建立。
- 模型和记录各自原子写入；若两次写入之间发生中断，hash 校验将拒绝不一致产物，不静默恢复。
- CLI 的 `run_manifest.json` 保持 unlocked；此入口不锁定 run、不调用 TargetEvaluationView。

完整同阶段续训必须回到原 run 目录，保留配套的 best 文件：

```powershell
../.venv/Scripts/python.exe -m training.stages.encoder_warmup --smoke --output artifacts/runs/smoke/encoder_acm_citation --resume artifacts/runs/smoke/encoder_acm_citation/encoder_latest.pt
```

`EncoderWarmup.resume` 恢复全部状态，从下一轮继续；已满足早停条件的 latest 不再训练。`fit` 后内存中的模型仍是 latest，应明确加载 best 才能进入 DA。

DA 在既有模型对象上加载，不替换分类头：

```python
from training.checkpoints.encoder import load_checkpoint

state = load_checkpoint(
    encoder_best_path, encoder, classifier, expected_stage1_metadata,
    expected_sha256=pinned_encoder_best_sha256,
)
# expected_stage1_metadata 应从当前数据视图、固定 manifest 和 Stage-1 配置重新生成。
# state['parent_checkpoint'] 包含绝对路径/hash/name，供 da_best 保存父链。
# 新建 A/S/fusion/discriminator 后，按 Stage 2 白名单重新建立 optimizer/scheduler。
# classifier(fused_as) 继续使用该对象，fused_as 为 [N, 128]。
```

加载前对全部 metadata 及模型 tensor 维度/类别数/hash 做检查；同阶段恢复还检查 optimizer 参数对象/顺序/矩张量和 scheduler 的 epoch/LR，并恢复 RNG。CUDA 数量不一致拒绝精确恢复；跨设备仅继承模型可使用默认的 `resume=False`，不恢复 Stage-1 optimizer/RNG。

## 验证边界

针对性测试包括参数对象共享、独立稠密邻接公式对照、孤立节点/零特征、初始化、H1/Z/分类维度、负 Z、有限梯度和裁剪、Source train-only 梯度、validation 标签变化不影响训练步、Target sentinel/隔离评估文件原标签-随机标签-sentinel 三次状态严格相等、真实 early-stop 循环、100 轮调度曲线、best/latest 精确恢复、DA 对象继承及不匹配拒绝。

CUDA、正式 100 epoch 真实数据训练、A/S、Memory、GRL、五阶段完整链及 final-evaluation 不由本阶段 smoke 声称通过。测试和 smoke 的实际结果见 `artifacts/reports/integration/encoder-stage-report.md`。
