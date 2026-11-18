# density-null

网络密度零模型。阶段：S3 审计；脚本：`scripts/data_audit/s3_network_density_null.py`。

## 目的

用同 N 同 E 的随机简单图作零模型，检验观测网络的密度是否只是节点数与边数的机械结果。对应假设：文本网络的结构密度显著高于随机基线。

## 参数

- 命令：`python -m peerlab run --exp density-null --seed 0`
- 脚本参数：`--seed`（缺省读 registry `random_seed` = 20260727）、`--reps`（缺省读 registry `rewiring_reps` = 1000）、`--years`、`--net-dir`、`--graph-check-reps`

## Baseline

- 随机重连零模型分布；观测网络的密度与零分布分位数比较。

## 实现细节

- 对每年网络做 R 次「同 N 同 E 随机简单图」重连，统计密度分布；附带小规模图结构抽查。

## 输入与前置

- `configs/parameter_registry.yaml`；`data/processed/`（网络产物）。
- 上游：`network-business-text`（或指定 `--net-dir`）。

## 产物与验收

- `reports/data_gates/G0/s3_network_density_null.json`
- 验收：每年的观测密度与零分布同时落盘；R 与 seed 记录在产物里。

## 耗时与资源

分钟级到小时级（R=1000，未实测）。

## 已知坑

- 耗时随 `--reps` 线性增长，冒烟时用 `--reps 10` 起步。
- 随机重连消费 seed，是本仓库里少数真正随种子变化的实验。
