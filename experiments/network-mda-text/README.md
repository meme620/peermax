# network-mda-text

老师 MD&A 语料的文本竞争网络（M-B）。阶段：S3 M-B；脚本：`s3_mb_full_matrix.py` → `s3_b1_calibrate_tau.py` → `s3_mb_build_edges.py`。

## 目的

用老师提供的 MD&A 语料构造公司间文本相似度矩阵，按校准阈值 τ 生成竞争网络，作为 PeerMAX 的第二条 peer 定义。对应假设：管理层讨论中的产品/竞争表述能识别同侪。

## 参数

- 命令：`python -m peerlab run --exp network-mda-text`
- 步骤 1：`s3_mb_full_matrix.py --years 2017-2025`（全期矩阵，sklearn TfidfVectorizer，min_df=1）
- 步骤 2：`s3_b1_calibrate_tau.py`（τ 以申万二级中位 peer 数为目标校准）
- 步骤 3：`s3_mb_build_edges.py --year 2021`（正式构边；不传 `--tau` 与 `--peer-target` 时落 `network_mb_stub/`）

## Baseline

- 主口径目录 `data/processed/network_mb_teacher/`；`network_mb_stub/` 是占位模式产物，不入分析。
- τ 校准值登记在 `reports/data_gates/G0/s3_b1_tau_calibration*.json`。

## 实现细节

- 全期 TF-IDF 余弦矩阵按年切片，阈值 τ 决定边的存在。
- τ 通过匹配申万二级行业的中位 peer 数量校准，避免网络密度与行业划分脱节。

## 输入与前置

- `configs/parameter_registry.yaml`；`data/raw/teacher/mda_data_2023.csv`（UTF-8，33,315 行）。
- 上游：老师语料解析（`s3_teacher_corpus_ingest.py`）。

## 产物与验收

- `data/processed/network_mb_teacher/{year}_{similarity,support}.csv`、`{year}_build_report.json`、`{year}_diagnostics.csv`
- `reports/data_gates/G0/s3_b1_tau_calibration*.json`
- 验收：τ 校准报告含目标与实测中位数；边密度落在登记区间。

## 耗时与资源

分钟级（未实测）。

## 已知坑

- 缺 `--tau`/`--peer-target` 时产物会落到 `network_mb_stub/`，容易被误读为主口径。
- 全期矩阵对语料版本敏感：换 MD&A 导出年份会改变 τ 校准结果。
