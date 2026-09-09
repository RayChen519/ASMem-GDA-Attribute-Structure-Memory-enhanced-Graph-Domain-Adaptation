# P3 / Stage 2

本入口只实现 Attribute/Structure 域适应；不创建 Memory，不读取 Target 类别标签，不执行完整集成 Gate。

## 模型与接口

- `models/selection/operators.py`：完整域单层单头 attention、两次 residual + post-LayerNorm、128→512→128 FFN；`theta=softplus(rho)`，初始 `rho=0`。A/S 独立 scorer 和标量阈值。
- `models/attribute/branch.py`：Source 执行 selection；Target 返回原 Z。SDPA 采用全 N×N 非因果注意力，没有节点采样、分区 mask 或自动降级。
- `models/structure/bga.py`：共享 128→128→128 MLP_S；一个 BGA block，共享簇内 attention、均值池化、簇间 attention、无 bias 的 256→128 广播融合、RestoreOrder；Source 再执行全域 Structure selection，Target 返回 H^LG。
- `models/fusion/attribute_structure.py`：A/S 拼接后 Linear(256,128)，含 bias。
- `contracts/models/attribute_structure.py`：输出 `h_a`、`h_lg`、`h_s`、`h_as` 均为 `[N_d,128]`，行顺序与原 node_id 一致。后续 Memory 可使用 `h_s` 作为 Query/Key、`h_as` 作为 Value/residual。
- `models/adversarial/domain.py`：128→64→1 + ReLU 判别器。一个单位 GRL；logistic 权重仅在总损失正号域 BCE 前乘一次。域 BCE 对两个域全部节点取均值。
- `training/losses/domain_adaptation.py`：分类仅索引 Source train IDs；稀疏项为两个 **乘原表示之前** 的软阈值张量 L1 和除以 N_s×128。

## 数据、缓存与失败处理

复用既有 Dataset 训练视图、Source validation 视图和已归一化 GCN 邻接。Target 不含 y；训练器不访问 evaluation_only 文件。

`data/partitions/metis.py` 使用 P=128、seed=0。缓存键仅含 dataset_version/domain/P/metis_seed；跨方向、标签率和训练 seed 复用，检查文件 hash、图 hash、完整覆盖和节点顺序。小于 P 的图或空簇明确报错，不静默减少簇数。合成冒烟使用 256/257 节点以保持 P=128。

主实验遇到 OOM 会写 `oom_evidence.json` 后抛出原错误，保存设备、epoch、全域尺寸、配置和异常信息。没有自动替代实现；若未来需要分区 Attribute attention，必须另行注册 `attribute_partitioned_attention_oom`，使用独立配置/产物，不能混入主实验。本阶段未发生实际 OOM，测试仅注入 OOM 验证记录路径。

## 优化与 checkpoint

正式默认配置为 `configs/base/domain_adaptation.json`，150/30/20（max/min/patience）；冒烟单独读取 `configs/overrides/smoke/domain_adaptation.json`，仅允许 1–3 epoch。AdamW、参数组学习率、5 epoch warm-up、cosine floor=0.1、weight decay=5e-4、clip=5 遵循 P5。

训练开始加载并校验 `encoder_best` 的配置/数据/split/维度/hash，保留 GCN 与 classifier 对象及权重。所有 Stage 2 模块联合训练。smoke parent 不能初始化正式运行。选优仅比较 Source validation Macro-F1，再以 loss 打破平局；达到 min_epochs 且 bad_epochs≥patience 才早停，最多 max_epochs。

每个 epoch 原子写入 `da_latest.pt`；改善时写 `da_best.pt`。两者均包含模块权重、requires_grad、train/eval modes、optimizer、scheduler、early-stop、history、RNG、GRL progress/权重、配置和父 checkpoint 路径/hash，以及 dataset/split/METIS hashes；anchor/PL 状态明确为 None。

精确恢复要求原 run 目录、原父 checkpoint、同配置及数据/METIS 和 RNG 设备拓扑。加载前校验张量、optimizer moments/step/参数组、scheduler 和 best 状态；恢复 latest 或 best 后继续原调度。正式训练结束后应显式加载 `da_best` 供下一阶段使用。

## 运行（PowerShell，工作目录为 code）

所有命令均使用项目虚拟环境；不使用系统 Python。

```powershell
Set-Location D:\Codex\Projects\SGDA_0826\code
& ..\.venv\Scripts\python.exe -m verification.smoke.synthetic.domain_adaptation --output artifacts/runs/smoke/da_synthetic_new
& ..\.venv\Scripts\python.exe -m training.stages.domain_adaptation --encoder-best artifacts/runs/smoke/encoder_acm_citation_seed0_verified/encoder_best.pt --output artifacts/runs/smoke/da_real_new --smoke --smoke-epochs 2 --device cpu --threads 2
& ..\.venv\Scripts\python.exe -m training.stages.domain_adaptation --encoder-best artifacts/runs/smoke/encoder_acm_citation_seed0_verified/encoder_best.pt --output artifacts/runs/smoke/da_real_new --smoke --smoke-epochs 2 --resume artifacts/runs/smoke/da_real_new/da_latest.pt
& ..\.venv\Scripts\python.exe -m pytest verification -q
```

不带 `--smoke` 使用正式配置；本次不运行该模式。已有输出目录必须通过 `--resume` 显式恢复。恢复报告另存 `resume_report.json`，保留初始冒烟的 `stage_report.json`。

## 本机 Windows METIS 依赖

PyPI 的 pymetis 2025.2.2 源码在当前 MinGW 上编译失败（缺少 sys/resource.h）。使用 [PyMetis 官方 README 列出的 conda-forge 二进制来源](https://github.com/inducer/pymetis/blob/main/README.rst)，将 CPython 3.13 Windows pymetis 2025.2.2 和 METIS 5.1.0 安装至本项目 .venv。下载地址和 SHA256 固定在 `utils/runtime/install_metis_windows.py`，保留许可证；不创建额外 Python 环境。

```powershell
& ..\.venv\Scripts\python.exe -m pip install zstandard==0.25.0
& ..\.venv\Scripts\python.exe utils/runtime/install_metis_windows.py
```

其他兼容平台可使用 `requirements-da.txt`。CUDA、其他平台和完整五阶段链尚未验证。
