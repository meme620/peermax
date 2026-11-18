# fm-main-table

Fama-MacBeth 主表与 horse race。阶段：M1/S5；脚本：`scripts/data_audit/s3_m1_2_fm_main.py`。

## 目的

估计 PeerMAX 对下月收益的横截面定价效应，并与 OwnMAX、PeerReturn 及若干替代解释做 horse race。这是论文主表。对应假设：控制自身 MAX 与 peer 收益后，PeerMAX 仍有独立解释力。

## 参数

- 命令：`python -m peerlab run --exp fm-main-table --seed 0`
- 脚本参数：`--sample-version v3`（缺省即 v3）、`--out-suffix _s{seed}`（按种子隔离产物）
- registry 常量：`CONTROLS`（控制变量集）、`NW_LAG = 6`、`MIN_N`（每月最小观测）、`MIN_MONTHS`（最小有效月数）、IQR 池化规则 `iqr_scale_rule`

## Baseline

- 四列并列：`ma_teacher`（主口径）、`mb_teacher`（方法稳健）、`ma`（文本稳健）、`mb`（语料对照）。
- **四列窗口不同**：ma_teacher / mb_teacher 为 2015-05..2023-12（104 月），ma / mb 为 2018-05..2023-12（68 月）。N 与系数不可跨列直接比较，重叠窗口仅 2018-05..2023-12。
- 主变量：`PeerMAXraw`；协变量 `OwnMAXraw`、`PeerReturn`、`CONTROLS`。

## 实现细节

- 逐月横截面 OLS（等权）/ WLS（市值加权）→ 系数序列 → 时间序列均值 + Newey-West(6) 标准误。
- 每月最少 `MIN_N` 个观测、至少 `MIN_MONTHS` 个有效月才计算表内读数。
- 输出同时包含 headline 规格（未归一化 PeerMAXraw）、post-adjusted 敏感性、horse race 规格。

## 输入与前置

- `configs/parameter_registry.yaml`；`data/processed/m1_common_sample_*_v3.parquet`（由 `estimation-sample` 产出）。
- 缺样本时抛 `FileNotFoundError`；缺 registry 时导入即 `SystemExit(2)`。

## 产物与验收

- `reports/data_gates/G0/s3_m1_pricing_table{sfx}.{json,md}`
- `reports/data_gates/G0/s3_m1_horse_race{sfx}.json`
- 验收：`n_months_used` 与样本报告一致；PeerMAXraw 系数符号与显著性落在登记区间。

## 耗时与资源

分钟级（未实测）。

## 已知坑

- horse race 的 `m1_horse_own_ind_l1/l2` 规格**不含 PeerReturn**，对 PeerMAXraw 的吸收幅度被高估，不能据此下「被行业解释掉」的结论；判读以 `s3_m1_industry_absorb.json` 含 PeerReturn 的 L1 规格为准。
- IQR_pool 用 treatment-only 宇宙算一次、全规格共用（登记值 0.02341287），各规格自身 IQR 仅作诊断。
