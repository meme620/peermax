# portfolio-sorts

组合排序表（五分位 / 十分位）。阶段：M1/S5；脚本：`scripts/data_audit/s3_m1_3_portfolio_shape.py`。

## 目的

用组合排序展示 PeerMAX 与下月收益的单调关系，作为连续变量主表的 supporting evidence。对应假设：PeerMAX 与预期收益单调递增，spread 显著。

## 参数

- 命令：`python -m peerlab run --exp portfolio-sorts --seed 0`
- 脚本参数：`--sample-version v2`（脚本缺省值）、`--out-suffix _s{seed}`
- registry 常量：`portfolio_tie_rule_id`（并列处理规则）、`K_SPREAD`、`IQR_POOL_REGISTERED`

## Baseline

- headline 是连续变量规格（`fm-main-table`），组合表只作 supporting。
- 主口径样本列 `ma_teacher`；窗口随 FM 样本版本变化。
- v1（未施加 microcap 筛除）历史版本有独立敏感性块，窗口与 N 与主口径不可混读。

## 实现细节

- 按 PeerMAXraw 分 5 组与 10 组，等权/市值加权计算组均收益，输出高减低 spread。
- 分组用 treatment-only 宇宙的池化中位/IQR（事前固定口径），不是逐月重算。

## 输入与前置

- `configs/parameter_registry.yaml`；`data/processed/m1_common_sample_*_v2.parquet`。
- 上游：`estimation-sample`。

## 产物与验收

- `reports/data_gates/G0/s3_m1_portfolio_deciles{sfx}.json`
- `reports/data_gates/G0/s3_m1_diagnostics{sfx}.json`
- 验收：quintile/decile 两组读数齐全；spread 与连续变量系数同号。

## 耗时与资源

分钟级（未实测）。

## 已知坑

- 样本版本缺省不一致：本脚本缺省 v2，而 `fm-main-table` 缺省 v3，两表不能默认同口径对比。
- 组合表不是因果证据，分组基于同期 PeerMAX，不能用来说明预测性。
