# 结果产出链路

## 从脚本到 result/

1. 脚本把产物写到三个固定位置：`reports/data_gates/G0/`（结果与审计 JSON/MD，`s3_m1_lib.OUT_DIR` 指向这里）、`data/processed/`（面板、网络、样本）、`artifacts/`（shadow 工作记录）。
2. `python -m peerlab run --exp X --seed N` 在子进程前后对这三个目录做 (路径, 大小, mtime) 快照，把差异里命中注册表 glob 的文件复制到 `result/runs/<run_id>/outputs/`，保留原相对路径。
3. `python -m peerlab report` 从各 run 的产物里按 dotted path 抽指标，写 `result/metrics.csv`、`result/tables/<exp>.csv`、`result/summary.md`。
4. 结果文件本身不被改名、不被搬动；`result/` 只是按运行归拢 + 跨种子聚合。产物名被 `s3_m1_6_gate_report.py` 与契约测试硬编码引用，改名会让门禁失效。

## 实验与产物

| id | 入口 | 关键产物 | 随机性 |
|---|---|---|---|
| `estimation-sample` | `s3_m1_1_build_sample.py` | `G0/s3_m1_sample_report*.json`、`data/processed/m1_common_sample_*.parquet` | 无 |
| `network-business-text` | `s3_ma_build_edges.py --corpus teacher` | `data/processed/network_ma_teacher/**` | 无 |
| `network-mda-text` | `s3_mb_full_matrix.py` → `s3_b1_calibrate_tau.py` → `s3_mb_build_edges.py` | `network_mb_teacher/**`、`G0/s3_b1_tau_calibration*.json` | 无 |
| `peermax-panel` | `s4_step1_max.py` → `s4_step2_peermax_formal.py --use-case ma` | `data/processed/monthly_pricing_panel_*_v4.csv` | 无 |
| `fm-main-table` | `s3_m1_2_fm_main.py --sample-version v3 --out-suffix _s{seed}` | `G0/s3_m1_pricing_table*`、`G0/s3_m1_horse_race*` | 无 |
| `portfolio-sorts` | `s3_m1_3_portfolio_shape.py --sample-version v2 --out-suffix _s{seed}` | `G0/s3_m1_portfolio_deciles*`、`G0/s3_m1_diagnostics*` | 无 |
| `robustness-checks` | `s3_m1_4_robustness_placebo.py` + `s3_m1_5_length_industry.py` | `G0/s3_m1_robustness_placebo.json`、`s3_m1_length_control.json`、`s3_m1_industry_absorb.json` | 无 |
| `gate-report` | `s3_m1_6_gate_report.py --out-suffix _s{seed}` | `G0/s3_m1_gate_report*` | 无 |
| `shadow-inference` | `p3_4_inference_shadow.py --seed {seed}` | `artifacts/phase3/3.4_*.csv` | 是（bootstrap） |
| `density-null` | `s3_network_density_null.py --seed {seed}` | `G0/s3_network_density_null.json` | 是（随机重连） |

样本版本不一致是既有口径：`s3_m1_2` 缺省 v3，`s3_m1_3/4/5` 缺省 v2。注册表按脚本缺省值固定，改版本属于口径变更，需要单独登记。

## 种子协议

- 唯一来源：`peerlab/registry.py` 的 `SEEDS = (0, 1, 2, 3, 4)`。
- 每次运行注入 `PYTHONHASHSEED=<seed>` 与 `PEERLAB_SEED=<seed>`；`peerlab/bootstrap.py` 在 `runpy` 执行脚本前 `random.seed` + `numpy.random.seed`。
- 声明 `accepts_seed` 的实验追加 `--seed <seed>`（目前 `shadow-inference`、`density-null`）。
- 支持 `--out-suffix` 的实验追加 `_s<seed>`，避免同名产物互相覆盖；不支持的实验靠运行前后差异收集。
- 边界：脚本内 9 处硬编码 `random_state` 与 registry 里固定的 `random_seed` 不受注入影响，因此 7 个实验的 5 个种子结果应当完全一致。聚合表用 `identical=True` 标出来，这是可复现性证据，不是缺陷。

## 冒烟

| 级 | 命令 | 检查内容 | 现状 |
|---|---|---|---|
| L0 | `python -m peerlab smoke` | 脚本存在、`requires` 里的输入是否到位、6 个 configs 是否存在 | 可跑，10 个实验全部 blocked（退出码 3） |
| L1 | `python -m peerlab run --exp X --smoke` | 最小切片实跑（像 `network-mda-text` 传 `--years 2021`），1 个种子，产物落 `result/smoke/` | 需 `configs/`，当前不可用 |

`make smoke` 走 L0；导入冒烟是 `python -m peerlab smoke --imports`（`make imports`）。

## 命名

- 目录不带 `0N_` 前缀；实验 id 用语义短名。
- `run_id` = `<exp>--s<seed>--<UTC 时间戳>`，时间戳精确到秒，同秒重复时后者覆盖前者（不会发生：串行执行）。
