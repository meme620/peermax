# configs/

六个冻结配置，2026-11-18 按「新基线」重建（原主仓库 registry 1.5.32 / manifest 1.0.110 不在本机）。

| 文件 | 作用 | 谁读 |
|---|---|---|
| `parameter_registry.yaml` | 参数登记表（26 条）：样本窗口、winsor、NW 滞后、M1 门禁阈值、S3A 冻结参数、面板版本指针 | `pricing_lib` 导入即校验；`peermax_panel_build`、`pricing_gate_report`、`analyst_gate1_consistency_check` 等 |
| `version_manifest.yaml` | 版本清单：网络版本白名单（10 个目录）、M1 主输入登记 | `pricing_lib.load_manifest_entries`、`peermax_panel_build.load_versions` |
| `s3a_final_spec.yaml` | S3A 规格：输入路径、网络版本与规则、样本偏差登记 | `analyst_*` 全套 |
| `m1_contracts.yaml` | M1 九个主输入的合同（路径 / sha256 / 行数） | `shadow_monthly_status` 等 Phase2 脚本 |
| `claim_registry.yaml` | claim 与七类连接登记（M1_headline、M1_portfolio） | `check_link_audit.py`、`tests/test_analyst_contracts.py` |
| `scope_card.yaml` | 最小口径卡（treatment/outcome/窗口/上限） | `check_link_audit.py` |

## 基线的来源

- **契约测试断言的值**：`sample_start=2015-01-01`、`sample_end=2023-12-31`、`winsorization_pct=[0.01,0.99]`、`min_focal_count_per_month=100`、`rewiring_reps=1000`。
- **源码注释/docstring 的值**：`nw_lag=6`、`min_valid_days=15`、`m_b_tau_fixed=0.25`、`df_threshold=0.25`、`min_product_terms=10`、`cluster_linkage=average`、`random_seed=20260727`、`sw2_median_peer_target=56`、`epsilon_weight=1e-4`。
- **基线取值（原值不可得，按项目口径取保守值）**：`alpha_m1=0.05`、`K_Econ_M1=0.05`、`K_Spread_M1=0.05`、`K_Consist_M1=0.05`、`min_effective_month_count=60`、`portfolio_breakpoints=[0.2,0.4,0.6,0.8]`、`panel_versions` 四个版本串、`sample_filter_thresholds` 五个口径串。

每个参数在文件里带 `note` 标明来源。这不是主仓库那份冻结 registry 的复制品，引用前先看文件头。

## 还没锚定的东西

`version_manifest.yaml` 的 `deliverables` 与 `m1_contracts.yaml` 的 `m1_inputs` 里，9 个 M1 主输入目前 `sha256: null`、`status: pending`——`data/processed/` 为空，算不出真实哈希。`s3a_final_spec.yaml` 的 `provenance.input_hashes` 同理。

补锚步骤（数据生成后）：

1. 对每个主输入算 `shasum -a 256`，写进 `version_manifest.yaml` 的 `deliverables[].sha256` 并把 `status` 改成 `active`；
2. 同一哈希与行数写进 `m1_contracts.yaml` 的 `m1_inputs[].sha256 / expected_rows`；
3. 把四份主输入的哈希写进 `s3a_final_spec.yaml` 的 `provenance.input_hashes`；
4. 重算 `s3a_final_spec.yaml` 的 `governance.manifest_sha256 / registry_sha256`（manifest/registry 改过就要重算）；
5. 跑 `python scripts/check_link_audit.py`、`python -m peerlab smoke --imports`、`python -m pytest tests -q`。

在锚补齐之前，任何读 sha 的脚本都会按 fail-closed 退出 2——这是设计行为，不要用猜测值填。
