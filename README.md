# SGDA 服务器运行指南

从仓库根目录按顺序执行，使用 Linux、Python 3.13 和 CUDA GPU。命令遇错即停；所有输出位于 `code/artifacts/`。`server.sh` 的路径参数相对于 `code/`。

## 1. 安装环境

先 clone 本仓库并进入根目录。将下方 wheel 地址替换为服务器驱动兼容、提供 PyTorch 2.14.0 的官方 CUDA index。

```bash
set -euo pipefail
export TORCH_INDEX_URL=https://download.pytorch.org/whl/<CUDA版本>
export DEVICE=cuda:0
PYTHON=python3.13 bash scripts/install-linux.sh
```

## 2. 准备数据并验收

```bash
bash scripts/server.sh -m data.loaders.acquire
# 已有原始 mat 文件时，上条命令追加 --local-dir /path/to/raw
bash scripts/server.sh -m data.prepare
bash scripts/server.sh -m verification.dataset.reproduce
bash scripts/server.sh -m verification.dataset.check
export ACCEPT_ID=server-$(date +%Y%m%d-%H%M%S)
bash scripts/accept-linux.sh
export GATE=artifacts/reports/integration/$ACCEPT_ID/integration_report.json
```

必须完整 Gate PASS 才能继续。每次重验使用新 ACCEPT_ID；CPU 验收不能替代 CUDA 验收。

## 3. Development 与配置锁定

```bash
bash scripts/develop-defaults.sh
```

执行 B0–B6 各一个默认候选，共 126 个 development runs，仅按 Source validation 选优并生成配置锁。中断后重执行此命令即可恢复。该流程不包含额外候选搜索；A1–A5、M1–M6 继承 B6 配置。

## 4. 全部训练、消融与敏感性实验

先生成确定性清单，再顺序执行三个批次：

```bash
for tier in core ablation sensitivity; do
  bash scripts/server.sh -m experiments list --tier "$tier" \
    --manifest artifacts/datasets/manifests/dataset_manifest.json \
    --development artifacts/experiments/development-lock.json \
    --integration-report "$GATE" --output "artifacts/experiments/$tier.json"
done

for tier in core ablation sensitivity; do
  bash scripts/server.sh -m experiments batch \
    --list "artifacts/experiments/$tier.json" --device "$DEVICE" --resume
done
```

- core：B0–B6，共 630 runs。
- ablation：A1–A5、M1–M6，共 990 runs。
- sensitivity：180 个新增 runs，默认点复用核心 B6。

各模型自动执行其定义的训练阶段。恢复时只重执行第二个循环，保留原清单、配置、数据和运行路径；已完成任务核验后跳过。断电遗留 `.running` 时，确认对应进程已退出后再移除该任务的标记。

可选：单任务使用 `experiments run --list artifacts/experiments/core.json --index 0 --device "$DEVICE" --resume`（通过 `bash scripts/server.sh -m` 调用）。多 GPU 的 batch 分别追加 `--shards 2 --shard 0` / `--shards 2 --shard 1`，并指定不同 device。

## 5. 独立 Target 评估与统计

全部训练成功后执行：

```bash
for tier in core ablation sensitivity; do
  for run in code/artifacts/runs/formal/"$tier"/*; do
    test -f "$run/run_lock.json"
    bash scripts/server.sh -m experiments evaluate \
      --manifest artifacts/datasets/manifests/dataset_manifest.json \
      --run-dir "${run#code/}" --device "$DEVICE"
  done
done
bash scripts/server.sh -m experiments analyze --root artifacts/runs/formal \
  --output artifacts/reports/analysis/results.json
```

结果输出 JSON/CSV，包含均值、标准差、配对检验、Holm 校正、效应量与置信区间。训练不读取 Target 标签；禁止依据 Target 结果调参或重选 checkpoint。

代码、依赖、数据或配置变化后须重新验收并生成匹配清单。Linux/CUDA 安装、显存、确定性及完整实验矩阵仍须服务器实测。

详细协议见 [统一实验计划](plan/Unified%20Baseline%20and%20Ablation%20Plan.md)、[执行与分析计划](plan/Full%20Experiment%20Execution%20and%20Analysis%20Plan.md)；本机历史验收见 [ACCEPTANCE](code/verification/experiments/ACCEPTANCE.md)。
