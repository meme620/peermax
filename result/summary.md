# 结果摘要

种子集合：(0, 1, 2, 3, 4)；成功运行 64 次。

## estimation-sample — pricing_gate 样本构建（run-all 只跑一次）

- 种子：[0]

## network-business-text — M-A 产品市场竞争网络构边（老师 MD&A 语料）

- 种子：[0, 1, 2, 3, 4]

## network-mda-text — M-B 文本竞争网络：全期矩阵 → τ 校准 → 构边

- 种子：[0, 1, 2, 3, 4]

## peermax-panel — panel：MAX/PeerMAX 面板（主口径 ma_teacher）

- 种子：[0, 1, 2]

## fm-main-table — pricing_gate 主表：Fama-MacBeth + horse race

- 种子：[0, 1, 2, 3, 4]
- beta_peer：0.0488252 ± 0.000528
- t_peer：0.734542 ± 0.00794

## portfolio-sorts — pricing_gate 组合表：五分位/十分位 spread

- 种子：[0, 1, 2, 3, 4]

## robustness-checks — pricing_gate 稳健性：placebo 与长度/行业控制

- 种子：[0]

## gate-report — pricing_gate 数据门报告

- 种子：[0]

## shadow-inference — Phase3 shadow 推断层（bootstrap，随 seed 变化）

无成功运行。

## density-null — 网络密度零模型（随机重连，随 seed 变化）

- 种子：[0]
