# peermax-panel

MAX / PeerMAX 定价面板。阶段：S4；脚本：`s4_step1_max.py` → `s4_step2_peermax_formal.py`。

## 目的

把个股月度 MAX（当月最高若干日收益均值）聚合到 peer 层面，得到 PeerMAX、OwnMAX 与 PeerReturn，形成后续 FM 回归的定价面板。对应假设：同侪的彩票偏好会外溢到本公司下月收益。

## 参数

- 命令：`python -m peerlab run --exp peermax-panel`
- 步骤 1：`s4_step1_max.py`（source-month MAX / argmax / 完整月 gate；可选 `--min-valid-days`、`--registry`）
- 步骤 2：`s4_step2_peermax_formal.py --use-case ma --net-dir data/processed/network_ma_teacher --corpus-id ma_teacher`（可选 `--panel-version`、`--start`/`--end`、`--out`、`--report`）

## Baseline

- 主口径 `ma_teacher`（中文法网络 × 老师 MD&A 语料）；`mb_teacher` 为方法稳健列。
- 主列未归一化（W1 bundle），renorm 版本单独登记，不与主列混用。
- 后复权收益（`R_NEXT_POST_ADJ_v1`）是敏感性口径，主口径为原始收益。

## 实现细节

- 逐月取个股收益最大的 N 日平均作为 MAX，聚合到 peer 集合得到 PeerMAXraw；同月自身 MAX 为 OwnMAXraw；peer 等权月收益为 PeerReturn。
- 完整月 gate：当月有效交易日不足则不参与。

## 输入与前置

- `configs/parameter_registry.yaml`；`data/processed/network_ma_teacher/`；日/月面板。
- 上游：`network-business-text`、`network-mda-text`。

## 产物与验收

- `data/processed/monthly_pricing_panel_{ma,mb,ma_teacher,mb_teacher}_v4.csv`
- `reports/data_gates/G0/s4_step2_{corpus}_v4_report.json`、`s4_c1_acceptance_{corpus}_v4.json`
- 验收：C1 三轮验收（`s4_step2_acceptance.py`）全通过；面板行数与 report 一致。

## 耗时与资源

分钟级（未实测）。

## 已知坑

- `s4_step2_peermax.py` 是 SUPERSEDED pilot，读 `network/` 旧口径并缺源收益填 0，不得作为分析输入。
- `s3a_step10_panel_merge.py` 依赖 pilot 产出的 `monthly_pricing_panel_partial.csv`，属旧口径，不能与本实验的 v4 面板混用。
