# estimation-sample

构建 M1/S5 的估计样本。阶段：M1/S5；脚本：`scripts/data_audit/s3_m1_1_build_sample.py`。

## 目的

把定价面板对齐成可供 Fama-MacBeth 回归使用的月度横截面样本：treatment-only 宇宙、IQR 口径、完整月与非零成交日门槛。供 `fm-main-table`、`portfolio-sorts`、`robustness-checks`、`gate-report` 共用。对应研究假设：PeerMAX 对下月收益的横截面解释力。

## 参数

- 命令：`python -m peerlab run --exp estimation-sample`
- 脚本参数：`--v2-report-only`（只产出 v2 报告）、`--force` / `--force-reason`（覆盖既有样本必须给理由）、`--rebuild-amount-cache`
- registry 常量：样本版本 v3（`M1_SAMPLE_VERSION_V3`）、窗口 `sample_start` / `sample_end`、完整月口径 `complete_month`、`n_valid_days >= 15`（volume>0 交易日）、microcap 筛除、IQR 缩放规则 `iqr_scale_rule`

## Baseline

- 主口径列 `ma_teacher`（中文法 × 老师 MD&A），FM 窗口 2015-05..2023-12（104 月）。
- v1（无版本后缀）未施加 microcap 筛除，是历史基线；v2/v3 为当前口径。
- 对照列：`mb_teacher`（方法稳健）、`ma` / `mb`（自建语料，窗口 2018-05..2023-12，68 月）。

## 实现细节

- 读 `PANELS`（v3 面板）→ 施加覆盖与 IQR 门槛 → 写 `m1_in_fm_sample` 标志 → 落样本 parquet 与样本报告。
- 样本路径由 `s3_m1_lib.sample_path(key, version)` 决定；v1 文件没有 `_v1` 后缀，v2/v3 才带后缀。
- 报告含覆盖统计、IQR 诊断、样本门明细。

## 输入与前置

- `configs/parameter_registry.yaml`；`data/processed/monthly_pricing_panel_*.csv`（或 v4 面板，取决于版本开关）。
- 上游：`peermax-panel`。缺 registry 时导入即 `SystemExit(2)`。

## 产物与验收

- `reports/data_gates/G0/s3_m1_sample_report*.json`
- `data/processed/m1_common_sample_{ma,mb}[_teacher]_{v2,v3}.parquet`
- 验收：样本报告无 `fail` 项；parquet 行数与报告一致；`m1_in_fm_sample` 覆盖率与报告一致。

## 耗时与资源

分钟级到小时级（未实测）。

## 已知坑

- 脚本 docstring 仍写产出 `_v2.parquet`，实际写 v3（`docs/pipeline.md` §3.1 已记）。以脚本行为为准。
- `--force` 会覆盖既有样本，重跑前确认下游没有引用旧样本。
