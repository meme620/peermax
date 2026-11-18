# PeerMAX 计算链（2026-11-17 代码快照）

本文件说明代码快照。快照的 `scripts/`、`tests/` 铺在仓库根——与主仓库层级一致，`Path(__file__).resolve().parents[2]` 据此指向仓库根；外部输入文档在 `ref/`，教师数据在 `data/`，来源说明见 `docs/materials.md`。下文命令一律在仓库根执行。

## 1. 本快照是什么

PeerMAX 项目的**计算链代码只读快照**：原始数据抓取 → 清洗与面板构建 → 产品市场网络构造 → 月度定价面板 / MAX / PeerMAX → M1（S5）月度定价事实与 PC 对照回归 → 契约测试与数据门审计。

**只收代码。** 不含数据（`data/`）、配置（`configs/`）、报告与审计产物（`reports/`）、汇报成品（`deliverables/`）、审计工作记录（`artifacts/`）。这些都在主仓库原路径，本快照不收录。

**其中 2 个是 SUPERSEDED pilot 脚本**——`scripts/data_audit/s3_build_network.py`、`scripts/data_audit/s4_step2_peermax.py`，仅为 pilot 链完整性与历史留档而纳入，docstring 均明令「请勿使用」，**不得作为分析输入**；正式结果一律来自 `s4_step2_peermax_formal.py` 等 active 脚本（见 §10）。

规模：**233 个代码文件**（232 `.py` + 1 `.sh`）、**3,845,923 字节**、**72,444 行**。

| 目录（相对仓库根） | 文件数 | 字节 | 行数 | 职责 |
|---|---|---|---|---|
| `scripts/data/` | 35 | 237,750 | 5,487 | 抓数与基础 builder（baostock / 东财 / 腾讯 / CNINFO / Drive / 申万 / 词库） |
| `scripts/data_audit/` | 163（162 `.py` + 1 `.sh`） | 3,134,884 | 56,722 | 阶段管线与审计：S1、S2、S2.5、S3 网络、S3A、S4、M1/S5、Phase2/3/3B shadow、Gate1/2/5 |
| `scripts/external/` | 1 | 3,653 | 93 | 外部基准（HP TNIC-3 结构统计） |
| `scripts/deliverables/` | 1 | 23,016 | 420 | 仅 `make_s3a_brief_pptx.py`（被 Gate5 读源码，见 §8） |
| `tests/` | 33 | 446,620 | 9,722 | 契约测试（33 模块；主仓库当前收集 568 个 test） |

与仓库其它部分的分工：`ref/proposal/` 是立项/研究计划文档（PDF、TeX），本快照不含叙述性文档；老师提供的原始数据包（分析师预测、CFND、WSVI、老师 MD&A、申万分类、Google Drive 共享）在 `data/raw/{analyst,cfnd,wsvi,teacher,drive}` 与 `data/external/`，代码读的正是这些路径；落盘位置与快照日期见 `docs/materials.md`。

## 2. 目录结构

```text
仓库根/
├── docs/pipeline.md             本文件
├── ref/                         外部输入文档：立项文档、老师材料、组内论文
├── data/                        教师数据（raw/）与派生产物（processed/）
├── configs/                     冻结参数与版本锚（待主仓库提供）
├── scripts/
│   ├── data/                    抓数与基础 builder（35 .py）
│   ├── data_audit/              阶段管线、网络、面板、回归、审计（162 .py + 1 .sh）
│   ├── external/                外部基准对表（1 .py）
│   └── deliverables/            汇报脚本（1 .py）
└── tests/                       契约测试（33 .py）
```

1. `scripts/data/` — 从外部源取数落 `data/raw/`、`data/processed/`；`build_*` 系列整理 ID 映射、日/月收益、股本、市值、行业映射、产品词库。
2. `scripts/data_audit/` — 项目主计算链，按阶段前缀组织，每阶段有可独立执行的入口脚本与配套审计。
3. `scripts/external/` — 只做外部基准统计（HP TNIC-3），不参与 A 股任何计算。
4. `scripts/deliverables/` — 只产汇报物料；此处仅一个，因 Gate5 读其源码文本。
5. `tests/` — 契约与护栏测试；多数只读真实产物断言合同，少数用合成序列对拍独立实现。

## 3. 管线全景

所有命令均在**主仓库根**（含 `configs/`、`data/`、`reports/`、`deliverables/` 的那一层）执行。本仓库只有代码与材料，缺 `configs/`、`data/`、`reports/`，不能就地运行（§5 约束 ①）。

### 3.1 阶段总表

| 段 | 做什么 | 入口脚本 | 主要产物 |
|---|---|---|---|
| S1 | 可得性盘点：expected universe、取数 manifest、schema/主键抽检、覆盖三分类 + 停牌核验、数据门状态表 | `s1_step1_universe` → `s1_step2_manifest` → `s1_step3_schema` → `s1_step4_coverage` → `s1_step5_gates` | `reports/data_gates/G0/`：`s1_expected_universe.json`、`data_access_manifest.json`、`s1_schema_check.json`、`field_mapping_manifest.json`、`s1_coverage_report.json`、`data_gate_report.json` |
| S2 | 日面板（raw + qfq + `rraw` + 除权 flag）、日/月收益、质量门 8 项；配套 `scripts/data/build_*` | `s2_build_daily_panel`、`s2_step3_quality`、`s2_mv_crosscheck`、`scripts/data/build_id_map`… | `data/processed/daily_panel/{year}.csv`、`monthly_returns{,_clean,_clean_v2}.csv`、`corporate_action_events.csv`、`firm_id_map.csv`、`firm_chars_clean.csv`、`market_equity_panel.csv`、`share_capital_history_v2.csv`、`sw2_industry_{map,latest}.csv`；`G0/quality_report_s2.json`、`mv_crosscheck_report.json` |
| S2.5 | 自建 FF4 日度因子（MKT/SMB/HML/MOM，2×3 双排序，市值加权）、REV，7 项验收 | `s25_build_factors`、`s25_validate_factors`、`build_rev_factor`；全期扩展 `s25_full_period_factors` | `factor_returns_daily.csv`、`rev_factor_monthly.csv`；`G0/factor_construction_report.json`、`factor_validation_report.json` |
| S3 M-A | 年报业务章节提取（v7：去 900 字截断 + 审计列）→ 聚类参数网格 → 逐年 K 冻结 → 正式构边 | `s3_extract_business_v7` → `s3_b3_ma_cluster_grid` → `s3_b3_ma_freeze` → `s3_ma_build_edges` | `business_texts_v7.csv`；`{net_dir}/clusters/{year}_clusters.csv`；`{net_dir}/{year}_{similarity,support}.csv`、`{year}_build_report.json`；`G0/s3_b3_ma_cluster_grid*.json`、`s3_b3_ma_param_freeze*.json` |
| S3 M-B | 老师 MD&A 语料 TF-IDF 余弦全期矩阵 → τ 校准（申万二级中位 peer 目标）→ 正式构边 | `s3_extract_mda_v2`（自建语料侧）、`s3_mb_full_matrix`、`s3_b1_calibrate_tau`、`s3_mb_build_edges` | `G0/s3_mb_full_matrix.json`、`s3_b1_tau_calibration.json`；`network_mb/`、`network_mb_teacher/` 下 `{year}_{similarity,support}.csv`、`{year}_build_report.json`、`{year}_diagnostics.csv` |
| S3 审计 | 边精度抽样+审计、文本完整性、跨源归因、年级密度、下游一致性、权重交叉、主链审计、九项验收 | `s3_edge_precision_sample` → `s3_edge_precision_audit`、`s3_data_integrity_audit`、`s3_text_audit`、`s3_cross_audit_attribution`、`s3_network_cross_audit`、`s3_ma_acceptance_check`、`s3_ma_downstream_consistency`、`s3_ma_weighting_cross_audit`、`s3_b2_acceptance_check`、`s3_bp_mainline_audit`、`s3_code_design_audit` | `G0/s3_*.json` / `*.md` |
| S3A | CNRDS 分析师预测 → 覆盖快照 → 12 月窗口投影为公司网络 → 派生变量 → 面板合并 → CF/NPM 复刻 → 9A–9M → Gate1/2/5 | `run_s3a_pipeline.py`（13 步）；9A–9M 与 step11 在 runner 外；Gate 审计 7 个脚本 | `data/raw/analyst/`（`forecast_raw_all.csv`、`step1_acceptance.json`…）；`data/processed/analyst/`（`analyst_forecast_clean.csv`、`firm_analyst_coverage.csv`、`edges/`、`firm_analyst_derived.csv`、`cf_mom_replication*.json`、`step5_snapshot.json`、`step9_derived.json`）；`monthly_pricing_panel_with_analyst.csv`；`G0/s3a_*.json` |
| S4 | source-month MAX/argmax/完整月 gate → PeerMAXraw / OwnMAXraw / PeerReturn（W1 bundle：主列未归一化 + renorm 独立登记）→ C1 三轮验收 | `s4_step1_max` → `s4_step2_peermax_formal` → `s4_step2_acceptance` | `source_month_argmax_panel.csv`；`monthly_pricing_panel_{ma,mb,ma_teacher,mb_teacher}_v4.csv`；`G0/s4_step2_{corpus}_v4_report.json`、`s4_c1_acceptance_{corpus}_v4.json` |
| M1/S5 | A1–A5 输入合同 + 下月 outcome + 控制变量 + 样本冻结 → A6+A7 FM 主表与 horse race → A8 组合/形状 → A9 稳健性 → A11+A12 → A10 gate | `s3_m1_1_build_sample` → `s3_m1_2_fm_main` → `s3_m1_3_portfolio_shape` → `s3_m1_4_robustness_placebo` → `s3_m1_5_length_industry` → `s3_m1_6_gate_report` | `m1_common_sample_{ma,mb}_v3.parquet`、`m1_s5_v3_sample_summary.json`、`s3_m1_sample_report_v3.json`；`G0/s3_m1_{pricing_table,horse_race,portfolio_deciles,diagnostics,robustness_placebo,length_control,industry_absorb,gate_report}` |
| Phase2/3/3B | shadow 对照：月收益/状态、MAX、复权 MAX、换手率、REV、shadow M1；as-of 分类、校准、原文规则、推断层、行业吸收、shadow 网络 M1；断点分段、共同窗口、VW 权重矩阵、A9 族多重比较 | `p2_1…p2_6`、`p3_1…p3_6`、`p3b_*` | `phase2_shadow/`、`phase3_shadow/`、`network_ma_teacher_p3*`、`artifacts/phase2/`、`artifacts/phase3/`、**仅** `artifacts/phase3b/3b_*` |

注记（注释陈旧，非归档问题）：`s3_m1_1_build_sample.py:27` 的 docstring 仍写产出 `..._v2.parquet`，但脚本实际写 v3（`:494`、`:863`、`:879`）——样本路径由 `s3_m1_lib.sample_path(key, 'v3')` 决定。**以脚本行为为准，不要按该 docstring 反改文档。**这是上游注释未随 v3 升级更新。

### 3.2 分阶段命令

```bash
# S1（顺序不可换；step5 读前四步报告）
python scripts/data_audit/s1_step1_universe.py
python scripts/data_audit/s1_step2_manifest.py
python scripts/data_audit/s1_step3_schema.py
python scripts/data_audit/s1_step4_coverage.py
python scripts/data_audit/s1_step5_gates.py

# S2（日面板默认覆盖 2018-2025，且默认拒绝重跑已有年度文件：需 --append --append-reason）
python scripts/data_audit/s2_build_daily_panel.py --years 2018-2025
python scripts/data_audit/s2_step3_quality.py
python scripts/data_audit/s2_mv_crosscheck.py
python scripts/data/build_id_map.py
python scripts/data/build_daily_returns.py
python scripts/data/build_monthly_returns.py
python scripts/data/build_monthly_returns_status.py    # 重建 monthly_returns_clean_v2.csv
python scripts/data/build_firm_chars.py
python scripts/data/build_market_equity.py
python scripts/data/build_sw2_industry_map.py

# S2.5
python scripts/data_audit/s25_build_factors.py
python scripts/data_audit/s25_validate_factors.py
python scripts/data_audit/build_rev_factor.py
python scripts/data_audit/s25_full_period_factors.py   # 2006-2025 全期；子进程无 cwd，必须仓库根启动

# S3 M-A（--corpus teacher = 主口径，老师 MD&A，网络年 2015-2023；缺省 self 为自建语料历史口径）
python scripts/data_audit/s3_extract_business_v7.py
python scripts/data_audit/s3_b3_ma_cluster_grid.py --corpus teacher --years 2019 2021 2023
python scripts/data_audit/s3_b3_ma_freeze.py --corpus teacher
python scripts/data_audit/s3_ma_build_edges.py --corpus teacher

# S3 M-B（不给 --tau 也不给 --peer-target 时默认占位模式，产物落 network_mb_stub/）
python scripts/data_audit/s3_mb_full_matrix.py --years 2017-2025
python scripts/data_audit/s3_b1_calibrate_tau.py
python scripts/data_audit/s3_mb_build_edges.py --year 2021 --tau 0.25

# S3A（runner 内部 subprocess 固定 cwd=ROOT；全流程重跑 step1-4 约 10-20 分钟，日常从 5 起）
python scripts/data_audit/run_s3a_pipeline.py          # 1-13 步
python scripts/data_audit/run_s3a_pipeline.py 5        # 从 snapshot 起
python scripts/data_audit/s3a_step11_merge_chars.py    # runner 之外：AnalystCoverage 接入 firm_chars_clean
python scripts/data_audit/s3a_gate1_entry_audit.py
python scripts/data_audit/s3a_gate1_provenance_audit.py
python scripts/data_audit/s3a_gate1_metadata_audit.py
python scripts/data_audit/s3a_gate1_consistency_check.py
python scripts/data_audit/s3a_gate2_path_registry.py
python scripts/data_audit/s3a_gate2_pit_audit.py
python scripts/data_audit/s3a_gate5_closure_check.py

# S4
python scripts/data_audit/s4_step1_max.py
python scripts/data_audit/s4_step2_peermax_formal.py --use-case ma \
  --net-dir data/processed/network_ma_teacher --corpus-id ma_teacher \
  --panel-version s4_panel_ma_teacher_2015_01_2023_12_v1 \
  --out data/processed/monthly_pricing_panel_ma_teacher_v4.csv \
  --report reports/data_gates/G0/s4_step2_ma_teacher_v4_report.json --start 2015-01 --end 2023-12
python scripts/data_audit/s4_step2_acceptance.py \
  --panel data/processed/monthly_pricing_panel_ma_teacher_v4.csv --use-case ma \
  --net-dir data/processed/network_ma_teacher \
  --ref-panel data/processed/monthly_pricing_panel_ma_v4.csv \
  --out reports/data_gates/G0/s4_c1_acceptance_ma_teacher_v4.json

# M1/S5
python scripts/data_audit/s3_m1_1_build_sample.py
python scripts/data_audit/s3_m1_2_fm_main.py            # --sample-version {v1,v2,v3,v4}、--out-suffix
python scripts/data_audit/s3_m1_3_portfolio_shape.py
python scripts/data_audit/s3_m1_4_robustness_placebo.py
python scripts/data_audit/s3_m1_5_length_industry.py
python scripts/data_audit/s3_m1_6_gate_report.py        # --out-suffix / --input-suffix

# Phase 2 / 3 / 3B（只读 headline，只写 shadow）
python scripts/data_audit/p2_1_build_monthly_status.py
python scripts/data_audit/p3_1_build_asof_shadow.py
python scripts/data_audit/p3b_s2_weight_matrix.py
```

### 3.3 S3A runner 的 13 步与 9A–9M

runner 的 `STEPS`：1 ingest → 2 audit → 3 clean → 4 dedup → 5 snapshot → 6 edges（`s3a_step67_edges.py`）→ 7 diag（`s3a_step8_diag.py`）→ 8 derived（`s3a_step9_derived.py`）→ 9 panel（`s3a_step10_panel_merge.py`）→ 10 sens（`s3a_sens_specs.py`）→ 11 cf_mom（4 变体）→ 12 alpha → 13 identity。

9A–9M 在 runner 之外（`s3a_gate2_path_registry.py` 记为 `OUTSIDE_RUNNER`）；Gate1 元数据审计覆盖其中 9 个：

| 标签 | 脚本 | 主要产物 |
|---|---|---|
| 9A | `s3a_fundamental_link.py` | `data/processed/analyst/fundamental_link.json` |
| 9B | `s3a_industry_contrast.py` | `G0/s3a_9b_industry_contrast.json` |
| 9C | `s3a_9c_fm_regression.py` | `G0/s3a_9c_fm_regression.json`（另有 `_extreme` 变体） |
| 9D | `s3a_st_sensitivity.py` | `data/processed/analyst/npm_st_excluded_<weight>.json` |
| 9E | `s3a_9e_subsample.py` | `G0/s3a_9e_subsample.json` |
| 9F | 无独立脚本：`s3a_cf_mom_replication` / `s3a_npm_replication` 的**显式**收益口径参数 | 同主脚本，不同文件名 |
| 9G | `s3a_9g_cffr.py` | `G0/s3a_9g_cffr.json` |
| 9H | `s3a_9h_absorb.py` | `G0/s3a_9h_absorb.json` |
| 9I | `s3a_9i_additional_tests.py` | `G0/s3a_9i_bivariate_decile_cumret.json` |
| 9J | `s3a_9j_extra.py` | `G0/s3a_9j_extra.json` |
| 9K | `s3a_9k_nofy_build.py` → `s3a_9k_compare.py` | `G0/s3a_9k_nofy_compare.{json,md}` |
| 9L | **无实现** | 主仓库验收矩阵登记 `NOT_RUN`（`reports/execution/s3a/s3a_final_acceptance_matrix.md`） |
| 9M | `s3a_9m_cfret2.py` | `G0/s3a_9m_cfret2.json` |

9F 与 9L 需注意：9F 不是独立脚本，而是一组显式参数（源码注释：「收益口径为显式变体，禁止环境变量静默切换」）；9L 在本快照内无对应文件，任何引用都应保持 `NOT_RUN`，不得当作已实现。

M1 样本版本（`s3_m1_lib.py:95-101`）：`m1_s5_v1_2015_2023` / `m1_s5_v2_2015_2023` / `m1_s5_v3_20261115` / `m1_s5_v4_20261116`；`M1_SAMPLE_VERSION` 当前指向 **v3**，registry `m1_sample_and_spec_registry.headline_value.sample_version` 同为 `m1_s5_v3_20261115`。v1、v2 标 superseded 但文件保留、不得覆盖。

## 4. 契约测试

```bash
python -m pytest tests/ -q        # 必须显式写 tests/
```

裸 `pytest` 会同时收集 `archive/pre_reset/tests`（10 个历史模块，含 `test_st1…test_st7`、fixtures），主仓库的既有记录是收集阶段报错中断；本机实测 120s 内也未完成收集。全量 33 模块约 15–18 分钟；主仓库当前收集 **568** 个 test。长效或并行跑用外部 basetemp：`python -m pytest tests/ -q -p no:cacheprovider --basetemp=<repo 外的目录>`。

| 组 | 模块 | 主要断言对象 |
|---|---|---|
| M1/S4 主链（11） | `test_s3_m1_contracts`、`test_s3_m1_contract_gates`、`test_registry_param_binding`、`test_sample_version_computed`、`test_sample_window_actual`、`test_source_month_complete_gate`、`test_n_valid_days_semantics`、`test_peer_aggregation_contract`、`test_available_at_guard`、`test_network_versions_manifest`、`test_s4_net_dir_guard` | 样本合同、四条输入硬门、registry 参数绑定、版本/窗口计算、完整月 gate、共享聚合不变量、`available_at` 护栏、网络版本白名单、输出路径护栏 |
| 基础设施（6） | `test_manifest_input_guard`、`test_illiq_cache_contract`、`test_drive_monthly_wide`、`test_nw_fm_synthetic`、`test_tie_rule_registry`、`test_s2_daily_panel_atomic` | M1 主输入 sha 护栏、ILLIQ 缓存契约、Drive 月宽表 how 分支、NW/FM 合成对拍、tie_rule 读 registry、日面板拒重跑 + 原子写 |
| S3 网络（3） | `test_s3_business_contracts`、`test_s3_mda_contracts`、`test_s3_teacher_corpus_contracts` | 业务章节 / MD&A / 老师语料面板的数据契约（唯一键、列、状态、去截断回归） |
| S3A（1） | `test_s3a_contracts` | 管线合同 + claim/scope 一致性 |
| Phase 2（6） | `test_p2_shadow_contracts`、`test_p2_max_shadow_contracts`、`test_p2_adj_max_contracts`、`test_p2_turnover_shadow`、`test_p2_rev_shadow`、`test_p2_shadow_m1_compare` | shadow 面板、shadow MAX、复权 MAX、换手率、REV、shadow M1 对比 |
| Phase 3（6） | `test_p3_asof_shadow`、`test_p3_calibration_shadow`、`test_p3_pc_origrule`、`test_p3_inference_shadow`、`test_p3_industry_absorb`、`test_p3_shadow_network_m1` | as-of 口径、校准变体、原文规则、推断层、行业吸收、shadow 网络 M1 |

Phase 3B 无独立测试模块，其产物仅由脚本自身路径守卫与 `artifacts/phase3b/` 对账覆盖。测试数量不等于覆盖强度：S3A 只有 1 个模块但断言密度最高（7 个 test），Phase 2/3 的 6+6 个模块多数同时使用真实产物只读断言与合成序列断言。

## 5. 运行硬约束

以下六条是**结构约束**，不是风格建议。每条后附「违反会怎样」。

### ① 目录层级不可改

`scripts/` 下 **110** 个 `.py` 用 `parents[2]` 定位仓库根；33 个测试**全部**用 `parents[1]`。`ocr_tesseract_chunk.sh` 无 Python 路径逻辑，由 `s3_network_prospectus_anchor.py` 经 `wsl.exe` 调用（路径同样由该脚本的 `ROOT` 拼出）。本仓库根**没有** `configs/`、`data/`、`reports/`、`deliverables/`，直接就地运行会在 import 阶段就失败——`s3_m1_lib` 找不到 `configs/parameter_registry.yaml`，测试收集直接 `SystemExit: 2`（快照打包前实测复现：`[REGISTRY] registry 不存在: …\configs\parameter_registry.yaml`，随后 `INTERNALERROR`）。

**违反会怎样**：所有路径解析偏移一级，读不到 registry/spec/manifest，或写到错误目录；fail-closed 脚本退出 2，非 fail-closed 脚本产出空结果或写到错位置。**正确做法**：把 `scripts/` 与 `tests/` 铺回主仓库对应位置，保证 `Path(__file__).resolve().parents[2]` 指向含 `configs/` 的根。

### ② 文件名与目录名不可改

33 个测试模块中 **30** 个用硬编码路径定位被测脚本：**22** 个用 `importlib.util.spec_from_file_location` 直接加载 `scripts/data_audit/<name>.py`（或 `scripts/data/<name>.py`），**16** 个用 `sys.path.insert` 把 `scripts/data_audit` 或 `scripts/data` 加进 `sys.path`（两组有交集）。只有 3 个不依赖路径：`test_s3_business_contracts`、`test_s3_mda_contracts`、`test_s3_teacher_corpus_contracts`——它们纯读数据产物。

**违反会怎样**：改名或移动被加载脚本 → `spec_from_file_location` 返回 `None` 或抛 `FileNotFoundError`；改名模块 → `ImportError`。30/33 模块受影响，测试基线整体失效。

### ③ 跨目录 import

| 引用方 | 被引用 | 方式 |
|---|---|---|
| `scripts/data_audit/p2_1_build_monthly_status.py:66` | `scripts/data/build_monthly_returns_status.trade_days_long` | 依赖 `sys.path` 已含 `scripts/data` |
| `tests/test_sample_version_computed.py:24-26` | 同上 | `sys.path.insert(0, ROOT/'scripts'/'data')` → `import build_monthly_returns_status` |
| `scripts/data/audit_free_sources.py:55` | 同目录 `fetch_sogou_lexicon.parse_scel` | 同目录 `sys.path` |

**违反会怎样**：`ImportError: No module named 'build_monthly_returns_status'`（或 `fetch_sogou_lexicon`）。两个 import 都不在 try 块内，不会降级。

### ④ subprocess 必须在仓库根启动

| 脚本 | 启动方式 | 固定 cwd |
|---|---|---|
| `run_s3a_pipeline.py:45` | `subprocess.run(..., cwd=ROOT)` | 是 |
| `s3_ma_df_grid.py:80` | `subprocess.run(..., cwd=str(ROOT))` | 是 |
| `s3a_gate1_entry_audit.py:115` | `subprocess.run(..., cwd=str(ROOT))` | 是 |
| `s25_full_period_factors.py:55,57` | `subprocess.run([sys.executable, "scripts/data_audit/..."])` | **否**（相对路径） |
| `s3_network_prospectus_anchor.py:765` | `wsl.exe … bash <OCR 脚本>` | **否**（脚本内部 `cd "$dir"`） |

**违反会怎样**：`s25_full_period_factors.py` 在非仓库根启动 → 子进程 `can't open file`；父进程只把 `returncode` 与 stdout/stderr 尾部打印出来，**不按返回码退出**（`sys.exit` 缺失），容易误读为「跑完了」。其余脚本的 `subprocess` 调用多带 `capture_output=True`，子进程 stderr 被吞，失败信息只在 `returncode != 0` 分支出现。

### ⑤ OCR 通道依赖 WSL

`s3_network_prospectus_anchor.py` 的 OCR 回退经 `wsl.exe -u 27697 bash <脚本>` 调用 `scripts/data_audit/ocr_tesseract_chunk.sh`；实测引擎 = WSL Ubuntu **tesseract 5.3.4**、`chi_sim`、`--psm 3`、200 DPI、`OMP_THREAD_LIMIT=2`（并发压满 vCPU 时降到 0.12 页/s）。仅在文本层不足（`公司` < 20 或字符数 < max(5000, 50×页数)）且 CID 确定性还原不可用时启用。

**违反会怎样**：无 WSL / 无 tesseract 5 / 无 `chi_sim` 时该通道静默失败或等到 7200s 超时；招股书锚审计只能退到 pdfminer + CID→GID 通道，文本覆盖不足时该项审计无法完成。

### ⑥ fail-closed（禁止回退实测值）

`s3_m1_lib` **导入即校验** `configs/parameter_registry.yaml`：缺失、参数缺失/重复、值域非法 → `SystemExit(2)`。`configs/version_manifest.yaml` 的 sha 锚同理：缺条目、缺 sha、非 active、非 64-hex、与实际不符 → `SystemExit(2)`，源码注释明确「禁止回退实测值或自动刷新 manifest」。

两种 fail-closed 形态需区分：

| 模块 | 形态 | 缺 spec/registry 时的实际行为 |
|---|---|---|
| `s3_m1_lib` | 导入即校验，显式退出 | `raise SystemExit(2)`（`s3_m1_lib.py:112` 的 `_registry_fail`），带 `[REGISTRY] …` stderr 提示；`pytest` 收集以 `INTERNALERROR` 中断 |
| `s3a_meta` | **模块级读取** spec | `s3a_meta.py:14` `_SPEC_BYTES = SPEC.read_bytes()`，缺 `configs/s3a_final_spec_v1.yaml` 时抛**未捕获的 `FileNotFoundError`**（不是 `SystemExit`）；`pytest` 收集阶段同样报错，但不是以 `SystemExit` 中断 |

两者都属设计上的 fail-closed：不许在无 spec/registry 的情况下跑出数字。

**违反会怎样**：registry 被改坏或换成旧版本 → 依赖 `s3_m1_lib` 的 36 个文件全部在 import 阶段退出，M1/S4/shadow 全链不可运行；这是进程级退出而非可捕获异常，`pytest` 会以 `INTERNALERROR` 中断收集。

## 6. 外部依赖

无 `requirements.txt` / `pyproject.toml`；默认解释器 = 系统 **Python 3.13**。

| 包 | 用途 | 必需性（真实 import 的文件数） |
|---|---|---|
| `pandas`、`numpy` | 全链数据操作 | 必需（147 / 116） |
| `scipy` | 统计分布与检验 | 必需（10） |
| `scikit-learn` | TF-IDF 向量化（M-B 侧） | 必需（14） |
| `PyYAML` | registry / spec / manifest | 必需（41） |
| `pyarrow` | parquet 样本读写 | 必需（4） |
| `pytest` | 契约测试 | 必需（29） |
| `jieba` | M-B 分词 | 条件必需（6） |
| `pyahocorasick` | M-A 词库多模式匹配 | 条件必需（2：`s3_ma_lexicon`、`s3_edge_precision_audit`） |
| `pymupdf` | 年报/招股书 PDF 文本层 | 条件必需（4：三个 fetch + 招股书锚） |
| `baostock` | 仅 `fetch_baostock_*` | 仅抓 baostock 时（2 个文件真实 `import baostock as bs`） |
| `pdfminer.six` + `fontTools` | 仅招股书锚 CID→GID 通道 | 仅该审计（各 1） |
| `python-pptx` | 仅 `make_s3a_brief_pptx.py` | 仅重出 PPTX（1） |

```bash
# 未锁版本，与主仓库现状一致
python -m pip install pandas numpy scipy scikit-learn PyYAML pyarrow pytest jieba pyahocorasick pymupdf baostock pdfminer.six fonttools python-pptx

# 自检
python -c "import pandas,numpy,scipy,sklearn,yaml,pyarrow,pytest; print('core ok')"
python -c "import jieba,ahocorasick,pymupdf,pptx; print('optional ok')"
python -c "import baostock; print('baostock ok')"
```

区分「真实 import」与「只是字符串提到」：全仓 `baostock` 字样出现在 20+ 文件，多数是路径 `data/raw/baostock/`，真正 `import baostock as bs` 只有 `fetch_baostock_financial.py`、`fetch_baostock_stocklist.py`。装依赖按真实 import 计。

## 7. 代码读取的配置与数据（均不在本仓库）

### 7.1 `configs/`

| 文件 | 本快照内引用文件数 | 作用 |
|---|---|---|
| `parameter_registry.yaml` | **31** | 全部冻结参数、样本版本指针、阈值、控制集、`panel_versions`；`s3_m1_lib` 导入即校验 |
| `s3a_final_spec_v1.yaml` | 16 | S3A 统一规格：输入路径、网络主版本、horizon；`spec_hash` 写入全部 9A–9M 产物 |
| `version_manifest.yaml` | 15 | 版本与 sha256 锚；M1 输入、网络版本白名单、产出登记 |
| `m1_contracts.yaml` | 7 | M1 数据合同（Phase2 shadow 共用） |
| `claim_registry.yaml` | 2 | claim 状态与禁止表述 |
| `scope_card.yaml` | 1 | 最小口径卡、claim 上限、`blocked_claims` |

`parameter_registry.yaml` 的 31 个引用方全在本快照内；主仓库全量 32 个，多出的一个是未纳入的 `scripts/deliverables/pack_mda_bundle.py`。当前状态（主仓库，2026-11-17）：registry = `1.5.32` / `stage4_frozen` / 93 个参数（无重名）；manifest = `1.0.110`。这两个数字会随主仓库变更，引用前重读。

### 7.2 `data/raw/`（14 个源目录）

`analyst/`、`baostock/`、`cfnd/`、`cninfo/`、`cninfo_prospectus/`、`drive/`、`eastmoney/`、`institutional/`、`lexicon/`、`sina/`、`sw2/`、`teacher/`、`tencent/`、`wsvi/`。

### 7.3 `data/processed/`（按阶段）

| 组 | 代表路径 |
|---|---|
| 日/月收益 | `daily_panel/{year}.csv`、`daily_returns_clean/`、`monthly_returns_clean*.csv` |
| 市场与股本 | `market_equity_panel*.csv`、`share_capital_history_v2.csv`、`firm_id_map.csv`、`firm_chars_clean.csv` |
| 因子 | `factor_returns_daily.csv`、`rev_factor_monthly.csv` |
| 文本语料 | `business_texts_v7.csv`、`mda_texts_v2.csv`、`teacher_mda_panel.csv`、`annual_fulltext_panel.parquet` |
| 词库 | `product_lexicon_v2*.csv`、`lexicon_sogou_words.csv`、`em_business_terms.csv` |
| 网络 | `network_ma/`、`network_ma_teacher/`、`network_ma_fulltext/`、`network_mb/`、`network_mb_teacher/`、`network_mb_stub/`、`network/`（pilot） |
| 分析师 | `analyst/`（覆盖表、边表、派生变量、CF/NPM 复刻 JSON） |
| 定价面板 | `source_month_argmax_panel.csv`、`monthly_pricing_panel_*[_v4].csv`、`monthly_pricing_panel_with_analyst.csv`、`monthly_pricing_panel_partial.csv` |
| M1 样本 | v1 **无版本后缀**：`m1_common_sample_{ma,mb}.parquet`、`m1_common_sample_{ma,mb}_teacher.parquet`；v2/v3：`m1_common_sample_{ma,mb}[_teacher]_v2.parquet`、`..._v3.parquet`；`m1_s5_v*_sample_summary.json` |
| shadow | `phase2_shadow/`、`phase3_shadow/` |

M1 样本命名注意：**v1 文件没有 `_v1` 后缀**（实际为 `m1_common_sample_ma.parquet` / `m1_common_sample_mb.parquet`，teacher 侧同理），只有 v2/v3 才带 `_v2`/`_v3`（`s3_m1_lib.py:184-196` 的 `sample_path`）。`_v4_renorm` 仅是 `s3_m1_lib.py:189-191` 的命名逻辑分支，**当前无实体文件**；v4 亦未落盘（W2 才创建），不要按命名模式假想不存在文件。

### 7.4 `reports/data_gates/G0/`

数据门与阶段审计的登记证据（主仓库当前 154 个 `.json` + 37 个 `.md`）。脚本直接读写该目录，`s3_m1_lib.OUT_DIR` 即指向此处；这些 JSON/MD 属 registry/manifest 登记产物，不得手改。

老师原始数据的落盘位置与快照日期（analyst_forecast 2026-05-27 打包、cfnd/wsvi 年度分片、mda `mda_data_2023.csv` 1005.5 MiB / UTF-8 / 2014-2022、`all_stkcds.xlsx` 2026-09-15、drive 36 文件快照 20260630）记录在 `docs/materials.md`，文件在 `data/raw/` 与 `data/external/`，本快照不重复。

## 8. 共享模块最小闭包

以下 9 个模块被大量脚本 import，**缺任一即大面积 ImportError**：

| 模块（`scripts/data_audit/`） | import 语句数 | 引用文件数 |
|---|---|---|
| `s3_m1_lib.py` | **38** | 36 |
| `s3_ma_lexicon.py` | 18 | 15 |
| `s3a_meta.py` | 12 | 12 |
| `s3_b3_ma_cluster_grid.py` | 10 | 10 |
| `s3a_cf_mom_replication.py` | 10 | 9 |
| `s4_step2_peermax_formal.py` | 9 | 8 |
| `s3_m1_1_build_sample.py` | 7 | 4 |
| `s3_ma_chapter_cross_audit.py` | 6 | 6 |
| `s3_mb_build_edges.py` | 5 | 4 |

计数按 AST 统计 `import X` / `from X import …` 语句条数，同一文件多次 import 不合并；`s3_m1_lib` 的 38 条中 8 条是函数内延迟 import，调用时同样会失败。

闭包特例：**`make_s3a_brief_pptx.py` 只产出 PPTX，却是 `scripts/deliverables/` 下唯一被纳入的脚本**——因为 `s3a_gate5_closure_check.py` (`:25` 定义路径、`:77` 读取) 用 `PPTX_SCRIPT.read_text(encoding='utf-8')` 直接读它的源码文本做「陈旧数字扫描」（查 `β=0.131`、`t=6.06` 是否残留）。缺这个文件，Gate5 收口检查抛 `FileNotFoundError`，整个 Gate5 无法完成。这是唯一纳入理由，不是因为它属于计算链。

## 9. 未纳入文件与原因

主仓库同目录下另有 **12 个 `.py`** 未纳入本快照。三类，逐条一行证据：

| 类 | 文件（主仓库路径） | 证据 |
|---|---|---|
| A 与 PeerMAX 无关 | `scripts/deliverables/fix_v4_ppt_p0.py` | docstring：「v4 PPT P0 级修复：Slide 33 残留清除…」；硬编码基底 `长度与吸阻检测的反事实质量筛查技术讲座_v4_70pp_fixed_optimized.pptx` |
| A | `scripts/deliverables/fix_v4_ppt_p1.py` | 同系列，操作同一份「长度与吸阻检测」讲座 PPT |
| A | `scripts/deliverables/fix_v4_ppt_p2.py` | 同系列 |
| A | `word_report/make_word_report.py` | docstring：「生成 FinTech 三题短答的 Word 成品」 |
| A | `word_report/answers_content.py` | docstring：「FinTech 三题短答的唯一内容源」 |
| C 只产汇报成品 | `scripts/data_audit/build_network_tables.py` | docstring：「从现有结果 JSON 聚合生成论文可直接使用的表格（markdown）」——纯格式化，无计算 |
| C | `scripts/deliverables/pack_mda_bundle.py` | docstring：「打包 MD&A 交付物为两个 zip」；正文只做 zip 封装 |
| D 一次性文档手术/ tmp | `scripts/deliverables/closeout_full_sample.py` | docstring：「收口 2026-08-29 全期扩展：执行计划 Step 4 状态更新 + manifest 登记」——对 docs/manifest 做 in-place 改 |
| D | `scripts/deliverables/closeout_mainline_full.py` | docstring：「主线全期化收口：s3a_report §7.2/§7.3/§11 + 执行计划 + manifest + memory」 |
| D | `scripts/deliverables/update_s3a_report_s8_full.py` | docstring：「s3a_report §8 更新为全期数字」——文本替换 |
| D | `scripts/deliverables/update_s3a_report_v23.py` | docstring：「s3a_report v2.2 -> v2.3：§7 全期主样本重构…」——文本替换 |
| D | `scripts/data_audit/tmp_verify_p11.py` | docstring：「P1-1 验证：FY1/FY2 过滤顺序对覆盖的影响」——一次性验证工具；其结论已固化进 `s3a_step5_snapshot.py` 的 `report_level_audit` 输出块，并由 `tests/test_s3a_contracts.py:121-123` 对回归值（309341 / 1815139 / 31466）断言 |

各类别计数：**A 类 5 个**（与 PeerMAX 无关）、**C 类 2 个**（只产汇报成品）、**D 类 5 个**（一次性文档手术 / tmp），合计 12。原本的「B 已废弃 pilot」两类（`s3_build_network.py`、`s4_step2_peermax.py`）已移入 §10 纳入归档，理由见该节。

注：4 个 D 类中的 `closeout_*` / `update_*` 位于 `scripts/deliverables/`，不是 `scripts/data_audit/`；同名前缀易混。

**`s3_build_network.py` 的 manifest 冲突（待主仓库治理）**：`configs/version_manifest.yaml` 对它**有两条互相冲突的登记**——`:981` 为 `status: superseded` / `version: "0.9-pilot"`（note 明写「已标 SUPERSEDED，不得作为分析输入」），`:2678` 为 `status: active` / `version: "1.0"` / `role: generation_script`（2026-08-09 的旧登记）。归档的判读依据是「以 docstring 与无消费者证据为准 → 仅作历史留档」，不采信 `:2678` 的 active 状态：全树无任何脚本 import 或 subprocess 调用 `s3_build_network`，其产物目录 `data/processed/network/`（pilot）亦已被正式 `network_ma/`、`network_mb/` 取代。**这是待主仓库治理的 manifest 冲突，不由归档解决**；引用脚本状态前须先回主仓库确认该重复条目的处置结果。

**连带影响（必读）**：`scripts/data_audit/s3a_step10_panel_merge.py` 读取 `data/processed/monthly_pricing_panel_partial.csv`（`:13`、`:23`），该文件由 `s4_step2_peermax.py` 生成（`:21`、`:34`）。该生成脚本虽标 SUPERSEDED，**已连同其上游 `s3_build_network.py` 一并纳入本归档**（§10），因此 S3A step10→step11 在本归档内**代码层面可追溯**。但可追溯不等于可重跑：pilot 链读旧的 `data/processed/network/`（Top-200 截断口径），缺源收益填 0，产物不得作为分析输入。要得到正式口径面板，须改用 `s4_step2_peermax_formal.py` 的 `monthly_pricing_panel_*[_v4].csv`，但这会改变 step10 的输入口径，属口径变更，必须登记，不得静默替换。

**归档内重跑审计的局限**：以下三处审计/测试会扫 `scripts/` 全树，在归档内因文件集合不同（缺上述 12 个排除项）结果与主仓库登记证据**不可比**：

| 位置 | 做什么 | 归档内为何不可比 |
|---|---|---|
| `s3a_gate1_provenance_audit.py:101` | `for f in sorted((ROOT / 'scripts').rglob('*.py'))` — 产生方检测 | 少 12 个脚本，可能漏判/误判产出路径归属 |
| `s3a_gate1_entry_audit.py:36,48` | 同上全树扫描 — env 读取扫描、旧参数名扫描 | 命中集合变小，行号与条数与登记证据不符 |
| `tests/test_manifest_input_guard.py:194` | `for p in sorted((ROOT / 'scripts').rglob('*.py'))` — sha 刷新函数扫描 | 扫描面缩小，「未发现」结论不能等同于主仓库结论 |

**这些审计/测试应在主仓库完整树上跑，不应在 `code/` 内跑并以此对照登记证据。**归档内的运行结果只能说明「本子集内无命中」。

## 10. 标注 [非主线] 的纳入项

纳入了归档、但**不是当前主线计算链一环**的 6 个文件，引用时必须带标注：

| 文件 | 标注 | 理由 |
|---|---|---|
| `scripts/data/extract_ar_product_tables.py` | [非主线] 原型 | docstring 自称「原型」；从年报原文提「主营业务分产品情况」明细，属 manifest skeleton / prototype 阶段产物 |
| `scripts/data_audit/s3_extract_business.py` | [非主线] 历史 v1 | 被 `s3_extract_business_v7.py` 取代（v6 缺陷：88.4% 文本被 900 字硬截断）；产物 `business_texts.csv` 留档且被审计报告引用 |
| `scripts/data_audit/s3_extract_mda.py` | [非主线] 历史 v1 | 被 `s3_extract_mda_v2.py` 取代（v1 三处缺陷：STOP 半句误切、30k 硬截断、无来源身份列）；`mda_texts.csv` 留档并被 `mda_integrity_audit.json` 引用 |
| `scripts/deliverables/make_s3a_brief_pptx.py` | [非主线] 仅汇报 | 只输出 PPTX，但 `s3a_gate5_closure_check.py` 用 `read_text()` 读其源码做陈旧数字扫描（§8），故必须纳入 |
| `scripts/data_audit/s3_build_network.py` | [非主线][SUPERSEDED] pilot 构图 | pilot 构网：TF-IDF 字符 bigram + 余弦 + **Top-200 截断口径**、按整年取 Y-1 版本、权重分母截断。docstring 首行：「SUPERSEDED（2026-09-15 审计）：pilot 构网…**请勿使用**……**不得作为任何分析输入**」。纳入的唯一理由是**保证 pilot 链代码完整**：它是 `s4_step2_peermax.py` 的上游（产出 `data/processed/network/{year}_similarity.csv`）。版本状态另见 §9 的 manifest 冲突说明 |
| `scripts/data_audit/s4_step2_peermax.py` | [非主线][SUPERSEDED] pilot S4 面板 | pilot S4 面板：读旧 `data/processed/network/`、**缺源收益填 0**。docstring 首行：「SUPERSEDED（2026-09-15 审计）：pilot S4 面板构建（读旧 network/、缺失源收益填 0）」+ **请勿使用、不得作为任何分析输入**。纳入的唯一理由是**保证 pilot 链代码完整**：它是已保留脚本 `s3a_step10_panel_merge.py:23` 读取的 `data/processed/monthly_pricing_panel_partial.csv` 的**生成者**（`:21`、`:34`），纳入后 S3A step10→step11 在归档内**代码层面可追溯**（不代表可重跑，见 §9 连带影响） |

`s3_extract_mda_v2.py` 是**主线**（M-B 自建语料侧），不在此列。

上表两条 SUPERSEDED 项的共同判据：它们是**本项目自己的代码**（pilot 链），既非「与 PeerMAX 无关」，也不属于「只产其它成品」，因此按「不缺漏」原则纳入；但 docstring 明令不得使用，所以必须同时打 `[SUPERSEDED]` 标注，避免被误当作可用实现。

## 11. 校验与重建

```bash
find scripts tests -name '*.py' | xargs python -m py_compile   # 全部 .py 语法通过
bash -n scripts/data_audit/ocr_tesseract_chunk.sh              # shell 语法通过
```

本仓库不保存逐文件清单与校验和。快照的源文件仍在主仓库原路径，本仓库是只读副本，可随时删除重建；不要在此编辑脚本后当主线用。

行数口径差异：本文的 72,444 行按每个文件的行数分项相加，`wc -l` 合计 72,439 行，差 5 行来自 5 个文件末尾无换行符（`s3_ma_weighting_cross_audit.py`、`s3_pc_sign_diag_h1h8.py`、`s3a_identity_conflict_sensitivity.py`、`s3a_smoketest_c4_min_months.py`、`s3a_smoketest_no_top1_variants.py`），两个数都对，只是计数方式不同。

**关于复现能力**：本快照**不保证**能复现任何结果。能否跑起来、能否得到相同数字，取决于三件事同时成立：

1. 主仓库 `configs/` 与快照一致（registry `1.5.32`、manifest `1.0.110`、spec 的 sha 未被改动）；
2. `data/` 下对应阶段产物存在且 sha 锚校验通过（`s3_m1_lib` 会逐项校验 M1 主输入）；
3. `reports/data_gates/G0/` 的登记证据未被替换。

任一不满足，fail-closed 脚本 `SystemExit(2)` 退出，非 fail-closed 脚本读到与登记不一致的输入并产出不可比数字。**不要**把本快照的任何产物当作「可完整复现」的证据。

## 12. 同步纪律

快照日期是 **2026-11-17**。主仓库之后每次变更（新增脚本、修 bug、改路径、改参数实现）都会让它滞后。引用前做两件事：

1. 用 `shasum -a 256` 比对快照文件与主仓库当前文件的哈希；不一致即本快照对应文件已过期。
2. 若引用脚本涉及冻结参数或样本版本，重读主仓库 `configs/parameter_registry.yaml` 的 `registry_version` 与 `configs/version_manifest.yaml` 的 `manifest_version`，确认与 §7.1 记录的版本号是否仍一致。

刷新方式：回主仓库原路径重新生成副本，不要在副本上打补丁。归档的生命周期是「快照」，不是「工作副本」。
