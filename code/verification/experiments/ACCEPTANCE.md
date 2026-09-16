## 2026-09-16 Documentation and report cleanup

- Preserved latest full local Gate PASS: integration/local-20260916-004030 (snapshot 38b18c0e923fe3cb5816cda9064f659ae00afe2077b275022acdb587c952c3f6) and reports/atomic-fix-20260916.xml. Deleted nine superseded Integration report directories; references below to deleted reports are historical only.
- README simplified to sequential server commands for installation, data, Gate, development, all three matrices, resume and evaluation.
- git check-ignore verified source/test/scripts remain eligible and all six generated output categories are ignored. Existing tracked artifact .gitkeep placeholders remain tracked.
- README changes invalidate the previous Gate snapshot. A new server Gate PASS is required; Linux/CUDA remains unverified. No training code changed in this cleanup.

## 2026-09-16 Windows atomic write regression (current status)

- Retry Windows replacement errors 5/32/33 at most 6 times (total backoff 1.55 seconds). Serialize once; never delete the old checkpoint. Persistent errors still fail; cleanup errors cannot hide the original exception.
- Gate disables pytest cacheprovider. The registered_experiments group automatically collects test_atomic_write.py.
- Six new cases cover transient errors, persistent denial, other IO errors and serialization failure. Together with the previously failing Target sentinel and resume/independent evaluation cases: **8 passed in 57.72s**, Windows CPU.
- Evidence: code/artifacts/reports/atomic-fix-20260916.xml.
- Original external file owner/ACL cause remains unidentified. Persistent permission denial is not repaired by retries. Linux/CUDA not tested.
- **Historical Gate PASS snapshots below are stale after this code change. Only focused regression was rerun; a fresh full Gate PASS is required for formal admission.**

# P8 服务器框架验收记录（2026-09-15）

## 最终结果

**Integration Gate PASS，且 require_pass 已重新核验当前源码与全部 87 份证据文件。**

- 报告：`code/artifacts/reports/integration/p8_portable_release/integration_report.json`
- 源码 snapshot：`e71f2c09b93b1328709aaf0680eed5f2b9cb2f4356a1cc45c1c4b965de458d2c`
- 当前仓库：`D:/Agent-WorkSpace/Codex/Projects/SGDA_0826`
- 环境：Windows 11、Python 3.13.1、CPU；torch 2.14.0、numpy 2.5.3、scipy 1.18.1、pymetis 2025.2.2、pytest 9.1.1。

| Gate 组 | 通过的测试条目 |
|---|---:|
| static_config | 10 |
| dimensions | 6 |
| stage_gradients | 7 |
| grl | 2 |
| target_sentinel | 60 |
| pseudo_label_refresh | 77 |
| registered_experiments | 24 |

合计 186 个测试条目（各组含重复覆盖），无失败或跳过。另有合成与真实 ACMv9→Citationv1 五阶段短链通过；短链预算为 1/2/1/2/2 epochs，K=8，未执行正式调参或实验矩阵。

## 修改范围

- 复用 EncoderWarmup、DomainAdaptation、MemoryTraining 执行五阶段。
- 注册 B0–B6、A1–A5、M1–M6；单因素配置差异校验、全部变体短链与 Target 访问哨兵。
- B0 只加载 Source；B1/B2/B3 使用规定的适配层；B5 不生成 Target PL。
- M5 使用 epoch 0 的 untrained_memory，未伪造 memory_best；其余完整模型保留五级 checkpoint 继承。
- Development seed=2026 的 Source validation 证据、共同最大预算、全局及受限 rate 覆盖、稳定哈希锁。
- 630/990/180 独立确定性清单，锁定 split/METIS/anchor/Target 顺序，单任务及分片批量入口。
- 同阶段 latest 恢复、完成阶段的 RNG 延续；缺失 latest、配置/数据/代码/父 checkpoint 不匹配均拒绝继续。
- final checkpoint/run manifest 锁定，独立 Target 评估、COMPLETED 与证据绑定、访问和失败日志。
- 五-seed 样本标准差、配对 Wilcoxon、分 family Holm、rank-biserial、分层 bootstrap、负迁移、效率与独立 sensitivity 汇总。
- Linux 安装、预检、启动及默认 development 脚本；四份脚本统一 LF，`.gitattributes` 固定 shell 脚本换行，加入回归测试。

## 干净源码与附加检查

最终源码副本 `code/artifacts/clean-p8-portable/repo` 排除了历史 artifacts、缓存和 checkpoint，snapshot 与工作区完全相同。

该副本中 **3 tests passed**：清单身份、训练→中断→恢复→独立评估、Linux LF 规则。恢复测试比较最终模型全部张量完全相同，并验证缺失 latest 会被拒绝。JUnit：`code/artifacts/clean-p8-portable/repo/clean-check.xml`。

使用的是本机已有 .venv；该检查不是 Linux 全新安装证明。CPU 稀疏 GCN/全域 attention 前后向及 METIS 预检通过；CLI help、四份 Bash 脚本逐一语法检查、git diff --check 通过。CPU 预检历史报告位于 `code/artifacts/reports/integration/p8-preflight-cpu.json`。

先前报告 p8_20260915_01、p8_20260915_final、p8_20260915_release、p8_relocated_release 因后续代码加固、目录迁移或 LF 修复已被取代，不能用于当前正式准入。没有改写旧报告哈希以绕过 Gate。

## 剩余服务器前置条件与未实测项

1. 本机未执行 development 调参、630/990/180 矩阵，也未创建假 development lock。服务器必须先完成 Source validation development，才能锁配置并生成正式清单。
2. Linux 从零创建 .venv、PyTorch CUDA wheel 安装、驱动兼容性未实测。脚本要求显式提供匹配服务器的官方 wheel index，失败不自动降级。
3. CUDA 稀疏/SDPA 确定性、真实完整图峰值显存、CUDA 精确恢复、长任务稳定性、多 GPU 分片和共享文件系统并发未实测。服务器须重新跑 CUDA Gate，CPU PASS 不可准入 CUDA 训练。
4. checkpoint 父路径为绝对路径，恢复须保留同一挂载路径；目录迁移后应重新验收。断电遗留 `.running` 需要确认原进程退出后移除；checkpoint/sidecar 不完整时拒绝恢复并审计重跑。
5. 可选 rate 配置当前接受六方向实测改善条件；仅声称 numerical/coverage 失败不能解锁配置，尚未提供该例外路径的预注册失败策略。
6. 没有正式 Target 结果，不给出科学性能结论。缺 seeds/条件时不输出正式均值或不完整 family 的显著性结论。

安装→数据准备→验收→development 锁定→训练→恢复→评估的服务器命令见仓库根 README.md。
