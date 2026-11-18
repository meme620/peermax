# PeerMAX 计算链

本文件说明本仓的计算链代码。`scripts/` 与 `tests/` 平铺在仓库根：脚本用 `Path(__file__).resolve().parents[1]` 定位仓库根，测试同样用 `parents[1]`。外部输入文档在 `ref/`，教师数据在 `data/`，来源说明见 `docs/materials.md`。下文命令一律在仓库根执行。

## 1. 本仓是什么

PeerMAX 项目的计算链代码：原始数据抓取 → 清洗与面板构建 → 产品市场网络构造 → 月度定价面板 / MAX / PeerMAX → M1（S5）月度定价事实与 PC 对照回归 → 契约测试与数据门审计。

**只收代码。** 不含数据（`data/`）、配置（`configs/`）、报告与审计产物（`reports/`）、汇报成品（`deliverables/`）、审计工作记录（`artifacts/`）。这些都在主仓库原路径。

规模：**192 个 `.py` + 1 个 `.sh`**、**3,289,008 字节**、**61,891 行**（`.py` 部分），全部在 `scripts/` 一层。

| 目录 | 内容 |
|---|---|
| `scripts/` | 计算链全部脚本：阶段入口、链上分析、共享库、一次性审计、抓数与 builder，平铺 192 个 `.py` |
| `tests/` | 45 个测试模块：33 个契约测试 + 5 个 peerlab 单测 + 1 个共享模块单测 + 6 个数据门单测 |
| `peerlab/` | 执行 CLI 与注册表 |

与仓库其它部分的分工：`ref/proposals/` 是立项/研究计划文档（PDF、TeX）；老师提供的原始数据包（分析师预测、CFND、WSVI、老师 MD&A、申万分类、Google Drive 共享）在 `data/raw/{analyst,cfnd,wsvi,teacher,drive}` 与 `data/external/`，代码读的正是这些路径；落盘位置见 `docs/materials.md`。

## 2. 目录结构

```text
仓库根/
├── docs/pipeline.md             本文件
├── ref/                         外部输入文档：立项文档、老师材料、组内论文
├── data/                        教师数据（raw/）与派生产物（processed/）
├── configs/                     冻结参数与版本锚（新基线重建，见 configs/README.md）
├── scripts/                     192 个 .py + ocr_tesseract_chunk.sh（平铺）
├── tests/                       45 个测试模块 + conftest.py
├── peerlab/                     执行 CLI
├── docs/experiments.md          10 个实验的规范文档（单文件）
└── result/                      结果层
```

1. `scripts/fetch_*` — 从外部源取数落 `data/raw/`；`build_*` 整理 ID 映射、日/月收益、股本、市值、行业映射、产品词库。
2. `scripts/inventory_*`、`daily_panel_*`、`factor_*`、`network_*`、`pricing_*`、`peermax_*` — 主计算链，按语义前缀分家族；顺序由 §3.1 阶段总表固定。
3. `scripts/analyst_*` — S3A 分析师覆盖网络全链。
4. `scripts/shadow_*` — Phase2/3/3B shadow 对照，只读 headline、只写 shadow。
5. `scripts/audit_*`、`*_check`、`*_smoketest` — 一次性审计与复核，不当主线入口。
6. `scripts/fileio.py`、`fmt.py`、`dates.py`、`statsutil.py` — 四个共享模块。

## 3. 管线全景

所有命令均在**主仓库根**（含 `configs/`、`data/`、`reports/`、`deliverables/` 的那一层）执行。本仓库只有代码与材料，缺 `configs/`、`data/`、`reports/`，不能就地运行（§5 约束 ①）。

### 3.1 阶段总表

| 段 | 做什么 | 入口脚本 | 主要产物 |
|---|---|---|---|
| S1 | 可得性盘点：expected universe、取数 manifest、schema/主键抽检、覆盖三分类 + 停牌核验、数据门状态表 | `inventory_universe` → `inventory_source_manifest` → `inventory_schema_check` → `inventory_coverage` → `inventory_gates` | `reports/data_gates/G0/`：`s1_expected_universe.json`、`data_access_manifest.json`、`s1_schema_check.json`、`field_mapping_manifest.json`、`s1_coverage_report.json`、`data_gate_report.json` |
| S2 | 日面板（raw + qfq + `rraw` + 除权 flag）、日/月收益、质量门 8 项；配套 `build_*` | `daily_panel_build`、`daily_panel_quality_gate`、`daily_panel_market_value_check`、`build_id_map`… | `data/processed/daily_panel/{year}.csv`、`monthly_returns{,_clean,_clean_v2}.csv`、`corporate_action_events.csv`、`firm_id_map.csv`、`firm_chars_clean.csv`、`market_equity_panel.csv`、`share_capital_history_v2.csv`、`sw2_industry_{map,latest}.csv`；`G0/quality_report_s2.json`、`mv_crosscheck_report.json` |
| S2.5 | 自建 FF4 日度因子、REV，7 项验收 | `factor_build`、`factor_validate`、`build_rev_factor`；全期扩展 `factor_full_period_build` | `factor_returns_daily.csv`、`rev_factor_monthly.csv`；`G0/factor_construction_report.json`、`factor_validation_report.json` |
| S3 M-A | 年报业务章节提取 → 聚类参数网格 → 逐年 K 冻结 → 正式构边 | `text_business_extract` → `network_business_cluster_grid` → `network_business_cluster_freeze` → `network_business_edges_build` | `business_texts.csv`；`{net_dir}/clusters/{year}_clusters.csv`；`{net_dir}/{year}_{similarity,support}.csv`；`G0/s3_b3_ma_cluster_grid*.json`、`s3_b3_ma_param_freeze*.json` |
| S3 M-B | 老师 MD&A 语料 TF-IDF 余弦全期矩阵 → τ 校准 → 正式构边 | `text_mda_extract`、`network_mda_full_matrix`、`network_tau_calibrate`、`network_mda_edges_build` | `G0/s3_mb_full_matrix.json`、`s3_b1_tau_calibration.json`；`network_mb/`、`network_mb_teacher/` 下年度边表 |
| S3 审计 | 边精度抽样+审计、文本完整性、跨源归因、密度零模型、下游一致性、权重交叉、主链审计、九项验收 | `network_edge_precision_sample` → `network_edge_precision_audit`、`data_integrity_audit`、`text_layer_audit`、`network_cross_audit_attribution`、`network_cross_audit`、`network_business_acceptance_check`、`network_business_downstream_consistency`、`network_business_weighting_cross_audit`、`network_report_acceptance_check`、`network_mainline_audit`、`code_design_audit` | `G0/s3_*.json` / `*.md` |
| S3A | CNRDS 分析师预测 → 覆盖快照 → 12 月窗口投影为公司网络 → 派生变量 → CF/NPM 复刻 → 9A–9M → Gate1/2/5 | `run_analyst_pipeline.py`（12 步）；9A–9M 与 step11 在 runner 外 | `data/raw/analyst/`、`data/processed/analyst/`、`G0/s3a_*.json` |
| S4 | source-month MAX/argmax/完整月 gate → PeerMAXraw / OwnMAXraw / PeerReturn → C1 三轮验收 | `peermax_max_build` → `peermax_panel_build` → `peermax_panel_acceptance` | `source_month_argmax_panel.csv`；`monthly_pricing_panel_{ma,mb,ma_teacher,mb_teacher}_v4.csv`；`G0/s4_step2_{corpus}_v4_report.json`、`s4_c1_acceptance_{corpus}_v4.json` |
| M1/S5 | A1–A5 输入合同 + 下月 outcome + 控制变量 + 样本冻结 → FM 主表与 horse race → 组合/形状 → 稳健性 → 篇幅与行业 → gate | `pricing_sample_build` → `pricing_fm_main` → `pricing_portfolio_sorts` → `pricing_robustness_placebo` → `pricing_length_industry_control` → `pricing_gate_report` | `m1_common_sample_{ma,mb}_v3.parquet`、`m1_s5_v3_sample_summary.json`、`s3_m1_sample_report_v3.json`；`G0/s3_m1_{pricing_table,horse_race,portfolio_deciles,diagnostics,robustness_placebo,length_control,industry_absorb,gate_report}` |
| Phase2/3/3B | shadow 对照：月收益/状态、MAX、复权 MAX、换手率、REV、shadow M1；as-of 分类、校准、原文规则、推断层、行业吸收、shadow 网络 M1；断点分段、共同窗口、VW 权重矩阵、A9 族多重比较 | `shadow_monthly_status` … `shadow_pricing_compare`、`shadow_asof_industry` … `shadow_network_pricing`、`shadow_segment_break` 等 | `phase2_shadow/`、`phase3_shadow/`、`artifacts/phase2/`、`artifacts/phase3/`、**仅** `artifacts/phase3b/3b_*` |

注记：`pricing_sample_build.py` 的 docstring 仍写产出 `..._v2.parquet`，脚本实际写 v3——样本路径由 `pricing_lib.sample_path(key, 'v3')` 决定。**以脚本行为为准，不要按该 docstring 反改文档。**

### 3.2 分阶段命令

```bash
# S1（顺序不可换；step5 读前四步报告）
python scripts/inventory_universe.py
python scripts/inventory_source_manifest.py
python scripts/inventory_schema_check.py
python scripts/inventory_coverage.py
python scripts/inventory_gates.py

# S2（日面板默认覆盖 2018-2025，且默认拒绝重跑已有年度文件：需 --append --append-reason）
python scripts/daily_panel_build.py --years 2018-2025
python scripts/daily_panel_quality_gate.py
python scripts/daily_panel_market_value_check.py
python scripts/build_id_map.py
python scripts/build_daily_returns.py
python scripts/build_monthly_returns.py
python scripts/build_monthly_returns_status.py    # 重建 monthly_returns_clean_v2.csv
python scripts/build_firm_chars.py
python scripts/build_market_equity.py
python scripts/build_sw2_industry_map.py

# S2.5
python scripts/factor_build.py
python scripts/factor_validate.py
python scripts/build_rev_factor.py
python scripts/factor_full_period_build.py   # 2006-2025 全期；子进程无 cwd，必须仓库根启动

# S3 M-A（--corpus teacher = 主口径，网络年 2015-2023；缺省 self 为自建语料历史口径）
python scripts/text_business_extract.py
python scripts/network_business_cluster_grid.py --corpus teacher --years 2019 2021 2023
python scripts/network_business_cluster_freeze.py --corpus teacher
python scripts/network_business_edges_build.py --corpus teacher

# S3 M-B（不给 --tau 也不给 --peer-target 时默认占位模式，产物落 network_mb_stub/）
python scripts/network_mda_full_matrix.py --years 2017-2025
python scripts/network_tau_calibrate.py
python scripts/network_mda_edges_build.py --year 2021 --tau 0.25

# S3A（runner 内部 subprocess 固定 cwd=ROOT；全流程重跑 step1-4 约 10-20 分钟，日常从 5 起）
python scripts/run_analyst_pipeline.py          # 1-12 步
python scripts/run_analyst_pipeline.py 5        # 从 snapshot 起
python scripts/analyst_chars_merge.py           # runner 之外：AnalystCoverage 接入 firm_chars_clean
python scripts/analyst_gate1_entry_audit.py
python scripts/analyst_gate1_provenance_audit.py
python scripts/analyst_gate1_metadata_audit.py
python scripts/analyst_gate1_consistency_check.py
python scripts/analyst_gate2_path_registry.py
python scripts/analyst_gate2_pit_audit.py
python scripts/analyst_gate5_closure_check.py

# S4
python scripts/peermax_max_build.py
python scripts/peermax_panel_build.py --use-case ma \
  --net-dir data/processed/network_ma_teacher --corpus-id ma_teacher \
  --out data/processed/monthly_pricing_panel_ma_teacher_v4.csv \
  --report reports/data_gates/G0/s4_step2_ma_teacher_v4_report.json --start 2015-01 --end 2023-12
python scripts/peermax_panel_acceptance.py \
  --panel data/processed/monthly_pricing_panel_ma_teacher_v4.csv --use-case ma \
  --net-dir data/processed/network_ma_teacher \
  --ref-panel data/processed/monthly_pricing_panel_ma_v4.csv \
  --out reports/data_gates/G0/s4_c1_acceptance_ma_teacher_v4.json

# M1/S5
python scripts/pricing_sample_build.py
python scripts/pricing_fm_main.py            # --sample-version {v1,v2,v3,v4}、--out-suffix
python scripts/pricing_portfolio_sorts.py
python scripts/pricing_robustness_placebo.py
python scripts/pricing_length_industry_control.py
python scripts/pricing_gate_report.py        # --out-suffix / --input-suffix

# Phase 2 / 3 / 3B（只读 headline，只写 shadow）
python scripts/shadow_monthly_status.py
python scripts/shadow_asof_industry.py
python scripts/shadow_weight_matrix.py
```

### 3.3 S3A runner 的 12 步与 9A–9M

runner 的 `STEPS`：1 ingest → 2 audit → 3 clean → 4 dedup → 5 snapshot → 6 edges（`analyst_edges_build.py`）→ 7 diag（`analyst_network_diagnostic.py`）→ 8 derived（`analyst_derived_build.py`）→ 9 sens（`analyst_sensitivity_specs.py`）→ 10 cf_mom（4 变体）→ 11 alpha → 12 identity。

9A–9M 在 runner 之外（`analyst_gate2_path_registry.py` 记为 `OUTSIDE_RUNNER`）；Gate1 元数据审计覆盖其中 9 个：

| 标签 | 脚本 | 主要产物 |
|---|---|---|
| 9A | `analyst_fundamental_link.py` | `data/processed/analyst/fundamental_link.json` |
| 9B | `analyst_industry_contrast.py` | `G0/s3a_9b_industry_contrast.json` |
| 9C | `analyst_fm_regression.py` | `G0/s3a_9c_fm_regression.json`（另有 `_extreme` 变体） |
| 9D | `analyst_st_sensitivity.py` | `data/processed/analyst/npm_st_excluded_<weight>.json` |
| 9E | `analyst_subsample.py` | `G0/s3a_9e_subsample.json` |
| 9F | 无独立脚本：`analyst_cf_mom_replication` / `analyst_npm_replication` 的显式收益口径参数 | 同主脚本，不同文件名 |
| 9G | `analyst_cffr.py` | `G0/s3a_9g_cffr.json` |
| 9H | `analyst_industry_absorb.py` | `G0/s3a_9h_absorb.json` |
| 9I | `analyst_decile_cumret.py` | `G0/s3a_9i_bivariate_decile_cumret.json` |
| 9J | `analyst_extra_tests.py` | `G0/s3a_9j_extra.json` |
| 9K | `analyst_nofy_build.py` → `analyst_nofy_compare.py` | `G0/s3a_9k_nofy_compare.{json,md}` |
| 9L | **无实现** | 主仓库验收矩阵登记 `NOT_RUN`（`reports/execution/s3a/s3a_final_acceptance_matrix.md`） |
| 9M | `analyst_cfret_second_order.py` | `G0/s3a_9m_cfret2.json` |

9F 与 9L 需注意：9F 不是独立脚本，而是一组显式参数（源码注释：「收益口径为显式变体，禁止环境变量静默切换」）；9L 在本仓内无对应文件，任何引用都应保持 `NOT_RUN`，不得当作已实现。

M1 样本版本（`pricing_lib.py`）：`m1_s5_v1_2015_2023` / `m1_s5_v2_2015_2023` / `m1_s5_v3_20261115` / `m1_s5_v4_20261116`；`M1_SAMPLE_VERSION` 当前指向 **v3**，registry `m1_sample_and_spec_registry.headline_value.sample_version` 同为 `m1_s5_v3_20261115`。v1、v2 标 superseded 但文件保留、不得覆盖。

## 4. 契约测试

```bash
python -m pytest tests/ -q        # 必须显式写 tests/
```

裸 `pytest` 只收集 `tests/`（`pytest.ini` 的 `testpaths`），但全量收集仍会在 `pricing_lib` 导入时因缺 `configs/parameter_registry.yaml` 以 `SystemExit(2)` 中断。长效或并行跑用外部 basetemp：`python -m pytest tests/ -q -p no:cacheprovider --basetemp=<repo 外的目录>`。

33 个契约模块按主题分六组：

| 组 | 模块 | 主要断言对象 |
|---|---|---|
| M1/S4 主链（11） | `test_pricing_contracts`、`test_pricing_contract_gates`、`test_registry_param_binding`、`test_sample_version_computed`、`test_sample_window_actual`、`test_source_month_complete_gate`、`test_n_valid_days_semantics`、`test_peer_aggregation_contract`、`test_available_at_guard`、`test_network_versions_manifest`、`test_peermax_net_dir_guard` | 样本合同、四条输入硬门、registry 参数绑定、版本/窗口计算、完整月 gate、共享聚合不变量、`available_at` 护栏、网络版本白名单、输出路径护栏 |
| 基础设施（6） | `test_manifest_input_guard`、`test_illiq_cache_contract`、`test_drive_monthly_wide`、`test_nw_fm_synthetic`、`test_tie_rule_registry`、`test_daily_panel_atomic` | M1 主输入 sha 护栏、ILLIQ 缓存契约、Drive 月宽表 how 分支、NW/FM 合成对拍、tie_rule 读 registry、日面板拒重跑 + 原子写 |
| S3 网络（3） | `test_network_business_contracts`、`test_network_mda_contracts`、`test_corpus_teacher_contracts` | 业务章节 / MD&A / 老师语料面板的数据契约 |
| S3A（1） | `test_analyst_contracts` | 管线合同 + claim/scope 一致性 |
| Phase 2（6） | `test_shadow_monthly_status_contracts`、`test_shadow_max_contracts`、`test_shadow_adjusted_max_contracts`、`test_shadow_turnover`、`test_shadow_rev`、`test_shadow_pricing_compare` | shadow 面板、shadow MAX、复权 MAX、换手率、REV、shadow M1 对比 |
| Phase 3（6） | `test_shadow_asof_industry`、`test_shadow_calibration`、`test_shadow_origrule`、`test_shadow_inference`、`test_shadow_industry_absorb`、`test_shadow_network_pricing` | as-of 口径、校准变体、原文规则、推断层、行业吸收、shadow 网络 M1 |

另有 5 个 peerlab 单测与 1 个共享模块单测，`make selftest` 只跑这 6 个（不碰缺 configs 的契约测试）：`pytest tests/test_peerlab_*.py tests/test_script_utils.py -q`，当前 **34 passed**。

## 5. 运行硬约束

以下六条是**结构约束**，不是风格建议。每条后附「违反会怎样」。

### ① 目录层级

`scripts/` 平铺在仓库根：脚本用 `Path(__file__).resolve().parents[1]` 定位仓库根，`tests/` 同样用 `parents[1]`。`ocr_tesseract_chunk.sh` 无 Python 路径逻辑，由 `network_prospectus_anchor.py` 经 `wsl.exe` 调用（路径由该脚本的 `ROOT` 拼出）。`configs/` 已按新基线重建（`configs/README.md`），但 M1 主输入的 sha256 锚尚未登记；读锚的脚本会按 fail-closed 退出 2。`reports/`、`deliverables/` 仍不在本仓。

**违反会怎样**：把脚本再下移一层，`parents[1]` 会指向 `scripts/`，读不到 registry/spec/manifest，或写到错误目录。

### ② 文件名与注册表

`peerlab/registry.py` 的 10 个实验共 14 处脚本路径、25 个测试模块里的 `scripts/<name>.py` 都是硬编码字符串。改名必须同步三处：脚本文件、注册表、测试与文档。

**违反会怎样**：注册表指向不存在的文件 → `make smoke` 报缺脚本；测试 `spec_from_file_location` 返回 `None` 或抛 `FileNotFoundError`。

`make lint` 会拦住阶段代号、大写、裸数字与深层目录，但不会替你改注册表和测试。

### ③ 同目录 import

| 引用方 | 被引用 | 方式 |
|---|---|---|
| `scripts/shadow_monthly_status.py` | `build_monthly_returns_status.trade_days_long` | 同目录；脚本把 `scripts/` 入 `sys.path` |
| `tests/test_manifest_input_guard.py` 等 | 同名脚本 | 测试把 `ROOT/scripts` 入 `sys.path` |
| `scripts/audit_free_sources.py` | `fetch_sogou_lexicon.parse_scel` | 同目录 import |

**违反会怎样**：`ImportError: No module named 'build_monthly_returns_status'`（或 `fetch_sogou_lexicon`）。这些 import 都不在 try 块内，不会降级。

### ④ subprocess 必须在仓库根启动

| 脚本 | 启动方式 | 固定 cwd |
|---|---|---|
| `run_analyst_pipeline.py` | `subprocess.run(..., cwd=ROOT)` | 是 |
| `network_business_df_grid.py` | `subprocess.run(..., cwd=str(ROOT))` | 是 |
| `analyst_gate1_entry_audit.py` | `subprocess.run(..., cwd=str(ROOT))` | 是 |
| `factor_full_period_build.py` | `subprocess.run([sys.executable, "scripts/..."]) ` | **否**（相对路径） |
| `network_prospectus_anchor.py` | `wsl.exe … bash <OCR 脚本>` | **否**（脚本内部 `cd "$dir"`） |

**违反会怎样**：`factor_full_period_build.py` 在非仓库根启动 → 子进程 `can't open file`；父进程只把 returncode 与输出尾部打印出来，**不按返回码退出**，容易误读为「跑完了」。

### ⑤ OCR 通道依赖 WSL

`network_prospectus_anchor.py` 的 OCR 回退经 `wsl.exe -u 27697 bash <脚本>` 调用 `scripts/ocr_tesseract_chunk.sh`；实测引擎 = WSL Ubuntu **tesseract 5.3.4**、`chi_sim`、`--psm 3`、200 DPI、`OMP_THREAD_LIMIT=2`。仅在文本层不足且 CID 确定性还原不可用时启用。macOS 上这条通道不可用，只影响招股书锚审计。

### ⑥ fail-closed（禁止回退实测值）

`pricing_lib` **导入即校验** `configs/parameter_registry.yaml`：缺失、参数缺失/重复、值域非法 → `SystemExit(2)`。`configs/version_manifest.yaml` 的 sha 锚同理：缺条目、缺 sha、非 active、非 64-hex、与实际不符 → `SystemExit(2)`，源码注释明确「禁止回退实测值或自动刷新 manifest」。

`analyst_meta` 是模块级读取 spec：缺 `configs/s3a_final_spec.yaml` 时抛**未捕获的 `FileNotFoundError`**（不是 `SystemExit`）。两者都属设计上的 fail-closed：不许在无 spec/registry 的情况下跑出数字。

## 6. 外部依赖

| 包 | 用途 | 必需性（真实 import 的文件数） |
|---|---|---|
| `pandas`、`numpy` | 全链数据操作 | 必需 |
| `scipy` | 统计分布与检验 | 必需 |
| `scikit-learn` | TF-IDF 向量化（M-B 侧） | 必需 |
| `PyYAML` | registry / spec / manifest | 必需 |
| `pyarrow` | parquet 样本读写 | 必需 |
| `pytest` | 契约测试 | 必需 |
| `jieba` | M-B 分词 | 条件必需 |
| `pyahocorasick` | M-A 词库多模式匹配 | 条件必需（`network_business_lexicon`、`network_edge_precision_audit`） |
| `pymupdf` | 年报/招股书 PDF 文本层 | 条件必需 |
| `baostock` | 仅 `fetch_baostock_*` | 仅抓 baostock 时 |
| `pdfminer.six` + `fontTools` | 仅招股书锚 CID→GID 通道 | 仅该审计 |
| `python-pptx` | 仅 `make_analyst_network_brief_pptx.py` | 仅重出 PPTX |

## 7. 代码读取的配置与数据（均不在本仓库）

### 7.1 `configs/`

| 文件 | 作用 |
|---|---|
| `parameter_registry.yaml` | 全部冻结参数、样本版本指针、阈值、控制集、`panel_versions`；`pricing_lib` 导入即校验 |
| `s3a_final_spec.yaml` | S3A 统一规格：输入路径、网络主版本、horizon |
| `version_manifest.yaml` | 版本与 sha256 锚；M1 输入、网络版本白名单、产出登记 |
| `m1_contracts.yaml` | M1 数据合同（Phase2 shadow 共用） |
| `claim_registry.yaml` | claim 状态与禁止表述 |
| `scope_card.yaml` | 最小口径卡、claim 上限、`blocked_claims` |

### 7.2 `data/raw/`（14 个源目录）

`analyst/`、`baostock/`、`cfnd/`、`cninfo/`、`cninfo_prospectus/`、`drive/`、`eastmoney/`、`institutional/`、`lexicon/`、`sina/`、`sw2/`、`teacher/`、`tencent/`、`wsvi/`。

### 7.3 `data/processed/` 与 `reports/data_gates/G0/`

路径与产物名保持主仓库原样（见 `docs/script_naming.md` 的例外表）：网络 `network_ma/`、`network_ma_teacher/`、`network_mb/`、`network_mb_teacher/`；面板 `monthly_pricing_panel_*`；样本 `m1_common_sample_*`；门禁证据 `reports/data_gates/G0/`。

## 8. 共享模块闭包

以下模块被大量脚本 import，**缺任一即大面积 ImportError**（按 AST 统计文件内 import 语句出现次数）：

| 模块 | import 语句数 |
|---|---|
| `pricing_lib.py` | 46 |
| `network_business_lexicon.py` | 20 |
| `peermax_panel_build.py` | 14 |
| `analyst_meta.py` | 12 |
| `network_business_cluster_grid.py` | 11 |
| `analyst_cf_mom_replication.py` | 10 |
| `statsutil.py` | 8 |
| `network_business_chapter_cross_audit.py` | 6 |
| `dates.py` | 5 |
| `network_mda_edges_build.py` | 5 |
| `peermax_max_build.py` | 5 |

闭包特例：`make_analyst_network_brief_pptx.py` 只产出 PPTX，却是 `analyst_gate5_closure_check.py` 的必需输入——Gate5 用 `read_text()` 读它的源码文本做「陈旧数字扫描」。缺这个文件，Gate5 收口检查抛 `FileNotFoundError`。

## 9. 未纳入的文件与原因（主仓库原路径）

主仓库同目录下另有 **12 个 `.py`** 未纳入本仓。三类：

| 类 | 文件（主仓库路径） |
|---|---|
| A 与 PeerMAX 无关 | `scripts/deliverables/fix_v4_ppt_p0.py`、`fix_v4_ppt_p1.py`、`fix_v4_ppt_p2.py`、`word_report/make_word_report.py`、`word_report/answers_content.py` |
| C 只产汇报成品 | `scripts/data_audit/build_network_tables.py`、`scripts/deliverables/pack_mda_bundle.py` |
| D 一次性文档手术 / tmp | `scripts/deliverables/closeout_full_sample.py`、`closeout_mainline_full.py`、`update_s3a_report_s8_full.py`、`update_s3a_report_v23.py`、`scripts/data_audit/tmp_verify_p11.py` |

这些路径是主仓库的原路径，本仓已平铺，不按本仓的目录结构解释。

## 10. 标注 [非主线] 的纳入项

纳入了归档、但**不是当前主线计算链一环**的 1 个文件，引用时必须带标注：

| 文件 | 标注 | 理由 |
|---|---|---|
| `make_analyst_network_brief_pptx.py` | [非主线] 仅汇报 | Gate5 读其源码，故必须纳入 |

`text_mda_extract.py` 是**主线**（M-B 自建语料侧），不在此列。

## 11. 校验与重建

```bash
python3 -m py_compile $(git ls-files '*.py')     # 全部 .py 语法通过
bash -n scripts/ocr_tesseract_chunk.sh           # shell 语法通过
python -m peerlab smoke --imports --clean        # 192 个脚本逐个导入
python -m peerlab lint                           # 命名与结构规范
```

本仓库不保存逐文件清单与校验和；逐文件角色见 `docs/script_inventory.csv`。

**关于复现能力**：本仓**不保证**能复现任何结果。能否跑起来、能否得到相同数字，取决于三件事同时成立：

1. 主仓库 `configs/` 与本仓脚本一致（registry 1.5.32、manifest 1.0.110、spec 的 sha 未被改动）；
2. `data/` 下对应阶段产物存在且 sha 锚校验通过（`pricing_lib` 会逐项校验 M1 主输入）；
3. `reports/data_gates/G0/` 的登记证据未被替换。

任一不满足，fail-closed 脚本 `SystemExit(2)` 退出，非 fail-closed 脚本读到与登记不一致的输入并产出不可比数字。

## 12. 同步纪律

本仓 2026-11-18 起与主仓库分叉（见 `docs/fork_log.md`）：脚本已改名并全部平铺。同步主仓库时：

1. 用 `docs/script_inventory.csv` 的 `former_name` 列把主仓库文件名映射到本仓新名，逐文件比对函数体，不要按路径直接覆盖；
2. 主仓库 `version_manifest.yaml` / `parameter_registry.yaml` 里登记的旧脚本路径与产物名不会自动更新，需要在主仓库侧同步修改后重跑门禁；
3. 路径层的命名规则由 `peerlab lint` 强制，新增脚本先过 lint 再提交。
