# robustness-checks

稳健性与 placebo 检验。阶段：M1/S5；脚本：`s3_m1_4_robustness_placebo.py` + `s3_m1_5_length_industry.py`。

## 目的

检验主表结论是否被样本长度、行业结构、安慰剂处理或换手/规模等机制解释掉。对应假设：PeerMAX 效应不是行业共振或机械相关。

## 参数

- 命令：`python -m peerlab run --exp robustness-checks`
- 脚本参数：`--sample-version v2`（脚本缺省）
- 前一组产出：`s3_m1_robustness_placebo.json`；后一组：`s3_m1_length_control.json`、`s3_m1_industry_absorb.json`

## Baseline

- 基准为主表 headline 规格（`ma_teacher`，未归一化 PeerMAXraw，原始收益）。
- 行业吸收规格含 PeerReturn（与 horse race 的行业规格不同），是判读「是否被行业解释」的依据。

## 实现细节

- placebo：随机重排 peer 映射或时间平移，检验效应是否消失。
- 长度控制：样本长度分箱后重估；行业吸收：加入行业固定效应/行业 peer 均值。

## 输入与前置

- `configs/parameter_registry.yaml`；`data/processed/m1_common_sample_*_v2.parquet`。
- 上游：`estimation-sample`。

## 产物与验收

- `reports/data_gates/G0/s3_m1_robustness_placebo.json`
- `reports/data_gates/G0/s3_m1_length_control.json`、`s3_m1_industry_absorb.json`
- 验收：placebo 效应不显著；行业吸收后 PeerMAXraw 显著性与符号方向与主表一致。

## 耗时与资源

分钟级（未实测）。

## 已知坑

- `gate-report` 引用的行业吸收值（v1 样本 γ=+0.172 / t=+1.75）标注「待 A12 复核轮重跑后确认」，引用前重读本实验最新产物。
- 样本版本缺省 v2 与主表 v3 不同口径。
