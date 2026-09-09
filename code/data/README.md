# Dataset 阶段

实现依据：`plan/Dataset.md` 和 `code/Framework_intro.md`。只准备数据、数据契约与标签访问入口；不包含模型、训练、METIS 算法或 Anchor 抽样算法。

## 构建与检查

在项目根目录安装一次依赖，然后从 `code/` 运行：

```powershell
uv venv .venv
uv pip install --python .venv/Scripts/python.exe -r code/requirements-dataset.txt
cd code
../.venv/Scripts/python.exe -m data.loaders.acquire
../.venv/Scripts/python.exe -m data.prepare
../.venv/Scripts/python.exe -m verification.dataset.reproduce
../.venv/Scripts/python.exe -m verification.dataset.check
../.venv/Scripts/python.exe -m pytest verification/dataset -q
```

已有原始文件可使用 `data.loaders.acquire --local-dir <包含三个mat的目录>`。
下载与本地导入都核验固定 SHA-256，任何不匹配均停止。构建中遇到损坏缓存也停止，不会静默使用或覆盖损坏文件。

## 输入来源与映射依据

使用 [SGDA 官方仓库](https://github.com/joe817/SGDA/tree/731fd50ce65fc19b89db9d157448af92e2a01c81/data)
commit `731fd50ce65fc19b89db9d157448af92e2a01c81` 的 `acmv9.mat`、`citationv1.mat`、`dblpv7.mat`。
固定版本和原始 SHA-256 存于 `configs/base/dataset_release.json`。
[论文 §5.1](https://www.ijcai.org/proceedings/2023/0253.pdf#page=5) 明确说明这三个域使用统一的 6775 维 attribute union。

发布文件只有 `attrb/network/group`，没有词名或原始论文 ID。这里采用**固定官方发布版的列 ID**
`sgda-731fd50-attribute-0000` 到 `sgda-731fd50-attribute-6774`；这不是恢复出的词名词表，也不能独立验证每列的自然语言语义。
`attribute_metadata.json` 保存每域原始列的 ID 和原始文件 SHA-256。只有字节级匹配的文件才能使用该身份列映射，
不能仅凭 6775 维接纳其他文件。统一 ID 清单的规范 JSON SHA-256 是 `attribute_union_hash`。
替换其他数据源时必须提供绑定其文件哈希的真实 attribute ID/名称元数据；按 ID 映射并集，拒绝重复、未知或无依据补出的列。
MAT 未提供节点 ID，因此本发布版的 `raw_node_id` 是零起始 MAT 行号；过滤后另存 `raw_row -> node_id`。

类别以同一发布版的五个 `group` 列为共享类别 ID，映射 `sgda-group-0..4 -> 0..4`；
没有足够依据确定类别名称对应哪个列，因此不编造名称映射。对其他显式类别 ID 会先取共同类别，再固定排序并重映射。

原始文件存在 multi-hot 行：ACMv9 581、Citationv1 166、DBLPv7 15。
默认沿用 [上游 DomainData.py](https://github.com/joe817/SGDA/blob/731fd50ce65fc19b89db9d157448af92e2a01c81/gnn/dataset/DomainData.py)
的 `argmax`（并列取最小列）生成单类别标签，保留这些节点。Dataset 计划未规定多标签转换，故此选择显式保存为
`multilabel_policy: upstream_argmax`，并写入版本与异常统计；可改为 `reject` 或 `single_label_only`，变更会生成新版本。

## 处理规则

1. 验证维度、节点 ID、有限非负特征/邻接和合法类别成员关系。无标签或非法数值直接报错。
2. 转换类别并保留共同类别节点，建立连续节点 ID；诱导子图删除重复边和原自环，再对称化，保留孤立点。
3. `degree` 是对称化、去自环后的度。另存 `gcn_edge_index/gcn_edge_weight`，对应 `D^-1/2 (A+I) D^-1/2`。
4. 按已验证 ID 对齐到 6775 列，float64 计算 L1 行归一化，输出 dense float32 `x`。全零行保持全零并计数。
5. 为每 Source 生成 3 个标签率 × 6 个 seed = 18 份 split，共 54 份；同一 Source 的 split 跨方向共享。
   每类预算 `max(2, round(rate*Nc))`，采用 Python round 的 ties-to-even 规则。
   train 数 `clamp(round(0.8*budget), 1, budget-1)`，其余为 validation；随机数发生器固定 NumPy PCG64。
   小于两个节点的 Source 类别导致该 Source 的任务停止。三种 Source mask 互斥且覆盖全部节点。
6. 生成六个有向任务，正式 seeds 0–4 共 90 个方向配置；development seed 2026 共 18 个配置，独立标记。

## 输出目录

```text
artifacts/datasets/raw/<domain>/<原始小写文件名>.mat
artifacts/datasets/raw/<domain>/<原始小写文件名>.provenance.json
artifacts/datasets/processed/<dataset_version>/<domain>/graph.pt
artifacts/datasets/processed/<dataset_version>/<domain>/node_mapping.json
artifacts/datasets/processed/<dataset_version>/<domain>/evaluation_only/labels.pt
artifacts/datasets/processed/<dataset_version>/<domain>/source_splits/<key>/train_labels.pt
artifacts/datasets/processed/<dataset_version>/<domain>/source_splits/<key>/validation_only/labels.pt
artifacts/datasets/manifests/dataset_manifest.json
artifacts/datasets/manifests/dataset_manifest_<dataset_version>.json
artifacts/datasets/manifests/attribute_vocabulary_<attribute_union_hash>.json
artifacts/cache/features/<key>.pt + .json
artifacts/cache/split/<key>.pt + .json
artifacts/reports/integration/dataset_integrity.json
artifacts/reports/integration/dataset_reproducibility.json
artifacts/reports/integration/dataset_tests.xml
```

`graph.pt` 不含真实标签；包含 Dataset 要求的图、特征、元数据及五种 mask。它是可复用基础图：
Source 三种 mask 初始为全 False/False/True；Target 两种 mask 全 True。
加载具体任务时，Source view 使用该 split 覆盖 Source mask，并清空 Source 的 Target mask。
TargetTrainView 只暴露无标签 Target 字段；`target_test_mask` 留在基础图和独立评估视图。

`dataset_manifest.json` 是最新发现入口；后续 run 应锁定版本化 manifest 及其文件 SHA-256。
manifest 包括原始/产物哈希、实现/环境版本、类别映射、词表及原列映射、节点映射、统计、异常、任务和 split 引用。
`dataset_version` 由原始哈希、配置、词表元数据、处理代码和依赖版本计算，避免算法或标签策略变化仍命中旧缓存。
特征缓存额外包括 domain，避免不同域在共同 vocabulary 下碰撞。
METIS 与 Anchor 提供 `cache_key` 和通用完整性存储接口，覆盖 Dataset 列出的全部字段；P、K、sampler 等未指定，未生成虚构分区或 anchor。

## 下游使用与标签隔离

```python
from data.views.training import load_training_views

source, target = load_training_views(
    'artifacts/datasets/manifests/dataset_manifest.json',
    source='ACMv9', target='Citationv1', label_rate=0.01, seed=0)
Xs, Xt = source.graph.x, target.x
As, At = source.graph.normalized_adjacency, target.normalized_adjacency
train_nodes, train_y = source.train_node_id, source.train_y
```

梯度监督仅使用 `train_y`（只包含 train 节点标签），不存在填满真实标签的 `y` 再附 mask 的接口。
Source validation 入口为 `evaluation.source_validation.data.load_source_validation`，只返回 validation 节点标签，
供未来早停与配置锁定调用。训练加载器不读取该文件，也不读取任何完整域标签。
后续 GCN 可使用原始 `edge_index` 让 GCNConv 自行归一化，或传入已归一化边权并关闭重复归一化/自环添加；不得两者叠加。

`evaluation.final_target.data.load_target_evaluation(dataset_manifest, run_manifest, lock)` 是唯一创建 TargetEvaluationView 的入口。
Dataset 阶段不产生训练 checkpoint 或锁文件。未来 run-locking 阶段需提供：

```json
{
  "run_manifest": {
    "variant": "B0",
    "source": "ACMv9",
    "target": "Citationv1",
    "label_rate": 0.01,
    "split_seed": 0,
    "split_hash": "实际 split hash",
    "dataset_version": "实际 dataset version",
    "dataset_manifest_sha256": "实际版本化 dataset manifest 的文件 SHA-256"
  },
  "final_lock": {
    "status": "locked",
    "run_manifest_sha256": "整个 run manifest 的文件 SHA-256",
    "final_checkpoint": {"name": "encoder_best", "path": "encoder_best.pt", "sha256": "checkpoint 文件 SHA-256"},
    "lock_hash": "移除 lock_hash 后其余 lock 字段的规范 JSON SHA-256"
  }
}
```

上述是两个文件的内容示意。B0/B1–B4/B5/B6 分别校验 encoder_best/da_best/source_pl_best/full_best。
检查 run、数据版本、方向、Source split 和 checkpoint 字节锁定后才读取 Target 标签。
这是程序接口及完整性隔离，不是操作系统权限隔离；拥有磁盘访问权限的代码仍可自行打开原始 MAT 或 evaluation_only 文件。
训练、调参、伪标签、早停和人工模型选择不得使用这些文件。完整性检查属于离线 Dataset 审计，不能作为训练选择信号。

## 验证范围

单元测试覆盖图对称化、重复边、自环、孤立点、节点过滤重编号、ID 列对齐、零行、非法特征、类别转换、
全部标签率/seed 的分层预算、稀有类停止、所有缓存决定字段变化、损坏检测、序列化复现以及 B0–B6 锁定门禁。
真实数据审计检查全部图和 54 个 split，并用文件读取 sentinel 检查全部 108 个训练视图配置，
禁止训练加载器读取 evaluation_only 或 validation_only 标签。反复构建比较 manifest/产物哈希确认复现。
JUnit 测试报告可用 `pytest verification/dataset -q --junitxml=artifacts/reports/integration/dataset_tests.xml` 生成。
