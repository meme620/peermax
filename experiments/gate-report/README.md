# gate-report

M1 数据门报告。阶段：M1/S5；脚本：`scripts/data_audit/s3_m1_6_gate_report.py`。

## 目的

把主表、组合表、稳健性三组的验收项汇总成一份门禁报告，决定 M1 结果是否可以进入下游（论文/汇报）。对应需求：任何数字引用前有可核查的登记证据。

## 参数

- 命令：`python -m peerlab run --exp gate-report --seed 0`
- 脚本参数：`--out-suffix _s{seed}`、`--input-suffix _s{seed}`（读取同种子的上游产物）

## Baseline

- 期望上游同名产物（`s3_m1_pricing_table`、`s3_m1_portfolio_deciles`、`s3_m1_robustness_placebo`、`s3_m1_length_control`、`s3_m1_industry_absorb`）齐全。

## 实现细节

- 读取上游 JSON 的登记字段，按 A1–A12 清单逐项判定 pass / fail / NOT_RUN。
- 输出 markdown 供人读，json 供程序读。

## 输入与前置

- `configs/parameter_registry.yaml`；上游三个实验的产物。
- 上游缺失时对应项记 NOT_RUN，不中断整份报告。

## 产物与验收

- `reports/data_gates/G0/s3_m1_gate_report{sfx}.{json,md}`
- 验收：报告无 fail 项，NOT_RUN 项逐条有原因。

## 耗时与资源

秒级到分钟级（未实测）。

## 已知坑

- 缺上游产物时报告仍会生成，容易误读为「通过」，引用前先看 NOT_RUN 列表。
