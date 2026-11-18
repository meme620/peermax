# shadow-inference

Phase3 shadow 推断层。阶段：Phase3；脚本：`scripts/data_audit/p3_4_inference_shadow.py`。

## 目的

对 N-1（分层）与 N-11（PC 三层）推断做 shadow 复核，检验主表显著性对推断口径的敏感性。对应需求：给主结论一个独立推断路径的对照。

## 参数

- 命令：`python -m peerlab run --exp shadow-inference --seed 0`
- 脚本参数：`--seed`（缺省 20261026；本仓库按 5 个全局种子运行）
- 只写 `artifacts/phase3/3.4_*`，不写 G0、不改 configs。

## Baseline

- 对照 Phase2/Phase3 其它 shadow 产物（`artifacts/phase2/`、`artifacts/phase3/`）。
- 主表推断（NW(6)）是 headline，本实验是推断层的替代口径。

## 实现细节

- 分层 FM 读数（两套层基 × EW/VW）、方差分解与组内 sd 分箱、PC 三层推断。
- 抽样/自助部分消费 `--seed`，是本仓库里少数真正随种子变化的实验。

## 输入与前置

- `configs/parameter_registry.yaml`；`artifacts/phase3/`（上游 shadow 产物）。

## 产物与验收

- `artifacts/phase3/3.4_inference.csv`、`3.4_struct_layers.csv`、`3.4_variance_decomposition.csv`、`3.4_pc_inference.csv`
- 验收：四个 CSV 齐全；分层读数与主表方向一致。

## 耗时与资源

分钟级（未实测）。

## 已知坑

- 产物不进 `reports/data_gates/G0`，聚合时靠 `result/runs/<run_id>/outputs/artifacts/phase3/` 取。
