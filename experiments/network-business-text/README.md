# network-business-text

年报业务章节文本竞争网络（M-A）。阶段：S3 M-A；脚本：`scripts/data_audit/s3_ma_build_edges.py`。

## 目的

用公司文本（业务章节 / MD&A）的相似度构造产品市场竞争网络，为 PeerMAX 提供 peer 定义。对应假设：文本识别出的产品市场同侪是 MAX 溢出的传导渠道。

## 参数

- 命令：`python -m peerlab run --exp network-business-text`
- 脚本参数：`--corpus teacher`（主口径，老师 MD&A，网络年 2015-2023）；`--corpus self`（自建语料，历史口径）
- 上游：`s3_extract_business_v7.py` → `s3_b3_ma_cluster_grid.py --corpus teacher --years 2019 2021 2023` → `s3_b3_ma_freeze.py --corpus teacher`

## Baseline

- 主口径目录 `data/processed/network_ma_teacher/`，不覆盖 `network_ma/`。
- 自建语料 `network_ma/` 为历史口径（业务描述文本自 2018 年起可得），与主口径窗口不同。

## 实现细节

- 文本向量化用自有 TF-IDF 实现，smooth IDF 与 M-B 侧 sklearn `TfidfVectorizer` 默认口径可比。
- 逐年 K 由聚类网格冻结，边由余弦相似度阈值筛选；每个年度产出 similarity / support / build_report 三件。

## 输入与前置

- `configs/version_manifest.yaml`；`data/processed/`（聚类标签 `{net_dir}/clusters/{year}_clusters.csv`）。
- 上游：`network-business-text` 的聚类冻结步骤（本实验的 steps 只做构边，聚类需先跑）。

## 产物与验收

- `data/processed/network_ma_teacher/{year}_similarity.csv`、`{year}_support.csv`、`{year}_build_report.json`
- 验收：build_report 无 fail；年度边数与节点覆盖符合聚类报告。

## 耗时与资源

分钟级（未实测）。

## 已知坑

- `s3_build_network.py` 是 SUPERSEDED pilot，产出 `network/`（Top-200 截断），不得作为分析输入。
- `network_ma_teacher_idf/` 是 IDF 变体的诊断产物，主口径不读它。
