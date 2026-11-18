# 实验索引

每个实验一个目录，固定 8 节：目的、参数、Baseline、实现细节、输入与前置、产物与验收、耗时与资源、已知坑。执行统一走 `python -m peerlab run --exp <id> --seed <n>`，命令与产物定义在 `peerlab/registry.py`。

| id | 一句话 | 入口脚本 | 产物 | 种子敏感 |
|---|---|---|---|---|
| `estimation-sample` | M1/S5 估计样本构建 | `s3_m1_1_build_sample.py` | `G0/s3_m1_sample_report*.json`、`m1_common_sample_*.parquet` | 否 |
| `network-business-text` | 年报业务章节文本网络（M-A） | `s3_ma_build_edges.py --corpus teacher` | `data/processed/network_ma_teacher/**` | 否 |
| `network-mda-text` | 老师 MD&A 语料文本网络（M-B） | `s3_mb_full_matrix.py` → `s3_b1_calibrate_tau.py` → `s3_mb_build_edges.py` | `network_mb_teacher/**`、`G0/s3_b1_tau_calibration*.json` | 否 |
| `peermax-panel` | MAX / PeerMAX 定价面板 | `s4_step1_max.py` → `s4_step2_peermax_formal.py` | `monthly_pricing_panel_*_v4.csv`、`G0/s4_*_v4*` | 否 |
| `fm-main-table` | Fama-MacBeth 主表 + horse race | `s3_m1_2_fm_main.py` | `G0/s3_m1_pricing_table*`、`s3_m1_horse_race*` | 否 |
| `portfolio-sorts` | 五分位/十分位组合表 | `s3_m1_3_portfolio_shape.py` | `G0/s3_m1_portfolio_deciles*`、`s3_m1_diagnostics*` | 否 |
| `robustness-checks` | placebo、长度控制、行业吸收 | `s3_m1_4_robustness_placebo.py` + `s3_m1_5_length_industry.py` | `G0/s3_m1_robustness_placebo.json` 等三件 | 否 |
| `gate-report` | M1 数据门报告 | `s3_m1_6_gate_report.py` | `G0/s3_m1_gate_report*` | 否 |
| `shadow-inference` | Phase3 shadow 推断层 | `p3_4_inference_shadow.py` | `artifacts/phase3/3.4_*.csv` | 是 |
| `density-null` | 网络密度零模型 | `s3_network_density_null.py` | `G0/s3_network_density_null.json` | 是 |

依赖顺序：`network-*` → `peermax-panel` → `estimation-sample` → `fm-main-table` / `portfolio-sorts` / `robustness-checks` → `gate-report`。`shadow-inference` 与 `density-null` 依赖各自上游，不阻塞主表。

当前状态：10 个实验全部 blocked——`configs/` 六个 YAML 缺失，`data/processed/` 为空。`python -m peerlab status` 会逐条列出缺口。
