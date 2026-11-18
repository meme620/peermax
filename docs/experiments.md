# 实验索引

每个实验一节，固定 8 个小节：目的、参数、Baseline、实现细节、输入与前置、产物与验收、耗时与资源、已知坑。执行统一走 `python -m peerlab run --exp <id> --seed <n>`，命令与产物定义在 `peerlab/registry.py`。

| id | 一句话 | 入口脚本 | 产物 | 种子敏感 |
|---|---|---|---|---|
| `estimation-sample` | M1/S5 估计样本构建 | `pricing_sample_build.py` | `G0/s3_m1_sample_report*.json`、`m1_common_sample_*.parquet` | 否 |
| `network-business-text` | 年报业务章节文本网络（M-A） | `network_business_edges_build.py --corpus teacher` | `data/processed/network_ma_teacher/**` | 否 |
| `network-mda-text` | 老师 MD&A 语料文本网络（M-B） | `network_mda_full_matrix.py` → `network_tau_calibrate.py` → `network_mda_edges_build.py` | `network_mb_teacher/**`、`G0/s3_b1_tau_calibration*.json` | 否 |
| `peermax-panel` | MAX / PeerMAX 定价面板 | `peermax_max_build.py` → `peermax_panel_build.py` | `monthly_pricing_panel_*_v4.csv`、`G0/s4_*_v4*` | 否 |
| `fm-main-table` | Fama-MacBeth 主表 + horse race | `pricing_fm_main.py` | `G0/s3_m1_pricing_table*`、`s3_m1_horse_race*` | 否 |
| `portfolio-sorts` | 五分位/十分位组合表 | `pricing_portfolio_sorts.py` | `G0/s3_m1_portfolio_deciles*`、`s3_m1_diagnostics*` | 否 |
| `robustness-checks` | placebo、长度控制、行业吸收 | `pricing_robustness_placebo.py` + `pricing_length_industry_control.py` | `G0/s3_m1_robustness_placebo.json` 等三件 | 否 |
| `gate-report` | M1 数据门报告 | `pricing_gate_report.py` | `G0/s3_m1_gate_report*` | 否 |
| `shadow-inference` | Phase3 shadow 推断层 | `shadow_inference.py` | `artifacts/phase3/3.4_*.csv` | 是 |
| `density-null` | 网络密度零模型 | `network_density_null.py` | `G0/network_density_null.json` | 是 |

依赖顺序：`network-*` → `peermax-panel` → `estimation-sample` → `fm-main-table` / `portfolio-sorts` / `robustness-checks` → `gate-report`。`shadow-inference` 与 `density-null` 依赖各自上游，不阻塞主表。

当前状态：10 个实验全部 blocked——`configs/` 六个 YAML 缺失，`data/processed/` 为空。`python -m peerlab status` 会逐条列出缺口。

## density-null

网络密度零模型。阶段：S3 审计；脚本：`scripts/network_density_null.py`。

### 目的

用同 N 同 E 的随机简单图作零模型，检验观测网络的密度是否只是节点数与边数的机械结果。对应假设：文本网络的结构密度显著高于随机基线。

### 参数

- 命令：`python -m peerlab run --exp density-null --seed 0`
- 脚本参数：`--seed`（缺省读 registry `random_seed` = 20260727）、`--reps`（缺省读 registry `rewiring_reps` = 1000）、`--years`、`--net-dir`、`--graph-check-reps`

### Baseline

- 随机重连零模型分布；观测网络的密度与零分布分位数比较。

### 实现细节

- 对每年网络做 R 次「同 N 同 E 随机简单图」重连，统计密度分布；附带小规模图结构抽查。

### 输入与前置

- `configs/parameter_registry.yaml`；`data/processed/`（网络产物）。
- 上游：`network-business-text`（或指定 `--net-dir`）。

### 产物与验收

- `reports/data_gates/G0/network_density_null.json`
- 验收：每年的观测密度与零分布同时落盘；R 与 seed 记录在产物里。

### 耗时与资源

分钟级到小时级（R=1000，未实测）。

### 已知坑

- 耗时随 `--reps` 线性增长，冒烟时用 `--reps 10` 起步。
- 随机重连消费 seed，是本仓库里少数真正随种子变化的实验。

## estimation-sample

构建 M1/S5 的估计样本。阶段：M1/S5；脚本：`scripts/pricing_sample_build.py`。

### 目的

把定价面板对齐成可供 Fama-MacBeth 回归使用的月度横截面样本：treatment-only 宇宙、IQR 口径、完整月与非零成交日门槛。供 `fm-main-table`、`portfolio-sorts`、`robustness-checks`、`gate-report` 共用。对应研究假设：PeerMAX 对下月收益的横截面解释力。

### 参数

- 命令：`python -m peerlab run --exp estimation-sample`
- 脚本参数：`--v2-report-only`（只产出 v2 报告）、`--force` / `--force-reason`（覆盖既有样本必须给理由）、`--rebuild-amount-cache`
- registry 常量：样本版本 v3（`M1_SAMPLE_VERSION_V3`）、窗口 `sample_start` / `sample_end`、完整月口径 `complete_month`、`n_valid_days >= 15`（volume>0 交易日）、microcap 筛除、IQR 缩放规则 `iqr_scale_rule`

### Baseline

- 主口径列 `ma_teacher`（中文法 × 老师 MD&A），FM 窗口 2015-05..2023-12（104 月）。
- v1（无版本后缀）未施加 microcap 筛除，是历史基线；v2/v3 为当前口径。
- 对照列：`mb_teacher`（方法稳健）、`ma` / `mb`（自建语料，窗口 2018-05..2023-12，68 月）。

### 实现细节

- 读 `PANELS`（v3 面板）→ 施加覆盖与 IQR 门槛 → 写 `m1_in_fm_sample` 标志 → 落样本 parquet 与样本报告。
- 样本路径由 `pricing_lib.sample_path(key, version)` 决定；v1 文件没有 `_v1` 后缀，v2/v3 才带后缀。
- 报告含覆盖统计、IQR 诊断、样本门明细。

### 输入与前置

- `configs/parameter_registry.yaml`；`data/processed/monthly_pricing_panel_*.csv`（或 v4 面板，取决于版本开关）。
- 上游：`peermax-panel`。缺 registry 时导入即 `SystemExit(2)`。

### 产物与验收

- `reports/data_gates/G0/s3_m1_sample_report*.json`
- `data/processed/m1_common_sample_{ma,mb}[_teacher]_{v2,v3}.parquet`
- 验收：样本报告无 `fail` 项；parquet 行数与报告一致；`m1_in_fm_sample` 覆盖率与报告一致。

### 耗时与资源

分钟级到小时级（未实测）。

### 已知坑

- 脚本 docstring 仍写产出 `_v2.parquet`，实际写 v3（`docs/pipeline.md` §3.1 已记）。以脚本行为为准。
- `--force` 会覆盖既有样本，重跑前确认下游没有引用旧样本。

## fm-main-table

Fama-MacBeth 主表与 horse race。阶段：M1/S5；脚本：`scripts/pricing_fm_main.py`。

### 目的

估计 PeerMAX 对下月收益的横截面定价效应，并与 OwnMAX、PeerReturn 及若干替代解释做 horse race。这是论文主表。对应假设：控制自身 MAX 与 peer 收益后，PeerMAX 仍有独立解释力。

### 参数

- 命令：`python -m peerlab run --exp fm-main-table --seed 0`
- 脚本参数：`--sample-version v3`（缺省即 v3）、`--out-suffix _s{seed}`（按种子隔离产物）
- registry 常量：`CONTROLS`（控制变量集）、`NW_LAG = 6`、`MIN_N`（每月最小观测）、`MIN_MONTHS`（最小有效月数）、IQR 池化规则 `iqr_scale_rule`

### Baseline

- 四列并列：`ma_teacher`（主口径）、`mb_teacher`（方法稳健）、`ma`（文本稳健）、`mb`（语料对照）。
- **四列窗口不同**：ma_teacher / mb_teacher 为 2015-05..2023-12（104 月），ma / mb 为 2018-05..2023-12（68 月）。N 与系数不可跨列直接比较，重叠窗口仅 2018-05..2023-12。
- 主变量：`PeerMAXraw`；协变量 `OwnMAXraw`、`PeerReturn`、`CONTROLS`。

### 实现细节

- 逐月横截面 OLS（等权）/ WLS（市值加权）→ 系数序列 → 时间序列均值 + Newey-West(6) 标准误。
- 每月最少 `MIN_N` 个观测、至少 `MIN_MONTHS` 个有效月才计算表内读数。
- 输出同时包含 headline 规格（未归一化 PeerMAXraw）、post-adjusted 敏感性、horse race 规格。

### 输入与前置

- `configs/parameter_registry.yaml`；`data/processed/m1_common_sample_*_v3.parquet`（由 `estimation-sample` 产出）。
- 缺样本时抛 `FileNotFoundError`；缺 registry 时导入即 `SystemExit(2)`。

### 产物与验收

- `reports/data_gates/G0/s3_m1_pricing_table{sfx}.{json,md}`
- `reports/data_gates/G0/s3_m1_horse_race{sfx}.json`
- 验收：`n_months_used` 与样本报告一致；PeerMAXraw 系数符号与显著性落在登记区间。

### 耗时与资源

分钟级（未实测）。

### 已知坑

- horse race 的 `m1_horse_own_ind_l1/l2` 规格**不含 PeerReturn**，对 PeerMAXraw 的吸收幅度被高估，不能据此下「被行业解释掉」的结论；判读以 `s3_m1_industry_absorb.json` 含 PeerReturn 的 L1 规格为准。
- IQR_pool 用 treatment-only 宇宙算一次、全规格共用（登记值 0.02341287），各规格自身 IQR 仅作诊断。

## gate-report

M1 数据门报告。阶段：M1/S5；脚本：`scripts/pricing_gate_report.py`。

### 目的

把主表、组合表、稳健性三组的验收项汇总成一份门禁报告，决定 M1 结果是否可以进入下游（论文/汇报）。对应需求：任何数字引用前有可核查的登记证据。

### 参数

- 命令：`python -m peerlab run --exp gate-report --seed 0`
- 脚本参数：`--out-suffix _s{seed}`、`--input-suffix _s{seed}`（读取同种子的上游产物）

### Baseline

- 期望上游同名产物（`s3_m1_pricing_table`、`s3_m1_portfolio_deciles`、`s3_m1_robustness_placebo`、`s3_m1_length_control`、`s3_m1_industry_absorb`）齐全。

### 实现细节

- 读取上游 JSON 的登记字段，按 A1–A12 清单逐项判定 pass / fail / NOT_RUN。
- 输出 markdown 供人读，json 供程序读。

### 输入与前置

- `configs/parameter_registry.yaml`；上游三个实验的产物。
- 上游缺失时对应项记 NOT_RUN，不中断整份报告。

### 产物与验收

- `reports/data_gates/G0/s3_m1_gate_report{sfx}.{json,md}`
- 验收：报告无 fail 项，NOT_RUN 项逐条有原因。

### 耗时与资源

秒级到分钟级（未实测）。

### 已知坑

- 缺上游产物时报告仍会生成，容易误读为「通过」，引用前先看 NOT_RUN 列表。

## network-business-text

年报业务章节文本竞争网络（M-A）。阶段：S3 M-A；脚本：`scripts/network_business_edges_build.py`。

### 目的

用公司文本（业务章节 / MD&A）的相似度构造产品市场竞争网络，为 PeerMAX 提供 peer 定义。对应假设：文本识别出的产品市场同侪是 MAX 溢出的传导渠道。

### 参数

- 命令：`python -m peerlab run --exp network-business-text`
- 脚本参数：`--corpus teacher`（主口径，老师 MD&A，网络年 2015-2023）；`--corpus self`（自建语料，历史口径）
- 上游：`text_business_extract_v7.py` → `network_business_cluster_grid.py --corpus teacher --years 2019 2021 2023` → `network_business_cluster_freeze.py --corpus teacher`

### Baseline

- 主口径目录 `data/processed/network_ma_teacher/`，不覆盖 `network_ma/`。
- 自建语料 `network_ma/` 为历史口径（业务描述文本自 2018 年起可得），与主口径窗口不同。

### 实现细节

- 文本向量化用自有 TF-IDF 实现，smooth IDF 与 M-B 侧 sklearn `TfidfVectorizer` 默认口径可比。
- 逐年 K 由聚类网格冻结，边由余弦相似度阈值筛选；每个年度产出 similarity / support / build_report 三件。

### 输入与前置

- `configs/version_manifest.yaml`；`data/processed/`（聚类标签 `{net_dir}/clusters/{year}_clusters.csv`）。
- 上游：`network-business-text` 的聚类冻结步骤（本实验的 steps 只做构边，聚类需先跑）。

### 产物与验收

- `data/processed/network_ma_teacher/{year}_similarity.csv`、`{year}_support.csv`、`{year}_build_report.json`
- 验收：build_report 无 fail；年度边数与节点覆盖符合聚类报告。

### 耗时与资源

分钟级（未实测）。

### 已知坑

- `network_ma_teacher_idf/` 是 IDF 变体的诊断产物，主口径不读它。

## network-mda-text

老师 MD&A 语料的文本竞争网络（M-B）。阶段：S3 M-B；脚本：`network_mda_full_matrix.py` → `network_tau_calibrate.py` → `network_mda_edges_build.py`。

### 目的

用老师提供的 MD&A 语料构造公司间文本相似度矩阵，按校准阈值 τ 生成竞争网络，作为 PeerMAX 的第二条 peer 定义。对应假设：管理层讨论中的产品/竞争表述能识别同侪。

### 参数

- 命令：`python -m peerlab run --exp network-mda-text`
- 步骤 1：`network_mda_full_matrix.py --years 2017-2025`（全期矩阵，sklearn TfidfVectorizer，min_df=1）
- 步骤 2：`network_tau_calibrate.py`（τ 以申万二级中位 peer 数为目标校准）
- 步骤 3：`network_mda_edges_build.py --year 2021`（正式构边；不传 `--tau` 与 `--peer-target` 时落 `network_mb_stub/`）

### Baseline

- 主口径目录 `data/processed/network_mb_teacher/`；`network_mb_stub/` 是占位模式产物，不入分析。
- τ 校准值登记在 `reports/data_gates/G0/s3_b1_tau_calibration*.json`。

### 实现细节

- 全期 TF-IDF 余弦矩阵按年切片，阈值 τ 决定边的存在。
- τ 通过匹配申万二级行业的中位 peer 数量校准，避免网络密度与行业划分脱节。

### 输入与前置

- `configs/parameter_registry.yaml`；`data/raw/teacher/mda_data_2023.csv`（UTF-8，33,315 行）。
- 上游：老师语料解析（`corpus_teacher_ingest.py`）。

### 产物与验收

- `data/processed/network_mb_teacher/{year}_{similarity,support}.csv`、`{year}_build_report.json`、`{year}_diagnostics.csv`
- `reports/data_gates/G0/s3_b1_tau_calibration*.json`
- 验收：τ 校准报告含目标与实测中位数；边密度落在登记区间。

### 耗时与资源

分钟级（未实测）。

### 已知坑

- 缺 `--tau`/`--peer-target` 时产物会落到 `network_mb_stub/`，容易被误读为主口径。
- 全期矩阵对语料版本敏感：换 MD&A 导出年份会改变 τ 校准结果。

## peermax-panel

MAX / PeerMAX 定价面板。阶段：S4；脚本：`peermax_max_build.py` → `peermax_panel_build.py`。

### 目的

把个股月度 MAX（当月最高若干日收益均值）聚合到 peer 层面，得到 PeerMAX、OwnMAX 与 PeerReturn，形成后续 FM 回归的定价面板。对应假设：同侪的彩票偏好会外溢到本公司下月收益。

### 参数

- 命令：`python -m peerlab run --exp peermax-panel`
- 步骤 1：`peermax_max_build.py`（source-month MAX / argmax / 完整月 gate；可选 `--min-valid-days`、`--registry`）
- 步骤 2：`peermax_panel_build.py --use-case ma --net-dir data/processed/network_ma_teacher --corpus-id ma_teacher`（可选 `--panel-version`、`--start`/`--end`、`--out`、`--report`）

### Baseline

- 主口径 `ma_teacher`（中文法网络 × 老师 MD&A 语料）；`mb_teacher` 为方法稳健列。
- 主列未归一化（W1 bundle），renorm 版本单独登记，不与主列混用。
- 后复权收益（`R_NEXT_POST_ADJ_v1`）是敏感性口径，主口径为原始收益。

### 实现细节

- 逐月取个股收益最大的 N 日平均作为 MAX，聚合到 peer 集合得到 PeerMAXraw；同月自身 MAX 为 OwnMAXraw；peer 等权月收益为 PeerReturn。
- 完整月 gate：当月有效交易日不足则不参与。

### 输入与前置

- `configs/parameter_registry.yaml`；`data/processed/network_ma_teacher/`；日/月面板。
- 上游：`network-business-text`、`network-mda-text`。

### 产物与验收

- `data/processed/monthly_pricing_panel_{ma,mb,ma_teacher,mb_teacher}_v4.csv`
- `reports/data_gates/G0/s4_step2_{corpus}_v4_report.json`、`s4_c1_acceptance_{corpus}_v4.json`
- 验收：C1 三轮验收（`peermax_panel_acceptance.py`）全通过；面板行数与 report 一致。

### 耗时与资源

分钟级（未实测）。

### 已知坑

- 缺 `configs/parameter_registry.yaml` 时导入即 `SystemExit(2)`；未登记的 `--net-dir` 会被输出路径护栏拦下（见 `tests/test_peermax_net_dir_guard.py`）。

## portfolio-sorts

组合排序表（五分位 / 十分位）。阶段：M1/S5；脚本：`scripts/pricing_portfolio_sorts.py`。

### 目的

用组合排序展示 PeerMAX 与下月收益的单调关系，作为连续变量主表的 supporting evidence。对应假设：PeerMAX 与预期收益单调递增，spread 显著。

### 参数

- 命令：`python -m peerlab run --exp portfolio-sorts --seed 0`
- 脚本参数：`--sample-version v2`（脚本缺省值）、`--out-suffix _s{seed}`
- registry 常量：`portfolio_tie_rule_id`（并列处理规则）、`K_SPREAD`、`IQR_POOL_REGISTERED`

### Baseline

- headline 是连续变量规格（`fm-main-table`），组合表只作 supporting。
- 主口径样本列 `ma_teacher`；窗口随 FM 样本版本变化。
- v1（未施加 microcap 筛除）历史版本有独立敏感性块，窗口与 N 与主口径不可混读。

### 实现细节

- 按 PeerMAXraw 分 5 组与 10 组，等权/市值加权计算组均收益，输出高减低 spread。
- 分组用 treatment-only 宇宙的池化中位/IQR（事前固定口径），不是逐月重算。

### 输入与前置

- `configs/parameter_registry.yaml`；`data/processed/m1_common_sample_*_v2.parquet`。
- 上游：`estimation-sample`。

### 产物与验收

- `reports/data_gates/G0/s3_m1_portfolio_deciles{sfx}.json`
- `reports/data_gates/G0/s3_m1_diagnostics{sfx}.json`
- 验收：quintile/decile 两组读数齐全；spread 与连续变量系数同号。

### 耗时与资源

分钟级（未实测）。

### 已知坑

- 样本版本缺省不一致：本脚本缺省 v2，而 `fm-main-table` 缺省 v3，两表不能默认同口径对比。
- 组合表不是因果证据，分组基于同期 PeerMAX，不能用来说明预测性。

## robustness-checks

稳健性与 placebo 检验。阶段：M1/S5；脚本：`pricing_robustness_placebo.py` + `pricing_length_industry_control.py`。

### 目的

检验主表结论是否被样本长度、行业结构、安慰剂处理或换手/规模等机制解释掉。对应假设：PeerMAX 效应不是行业共振或机械相关。

### 参数

- 命令：`python -m peerlab run --exp robustness-checks`
- 脚本参数：`--sample-version v2`（脚本缺省）
- 前一组产出：`s3_m1_robustness_placebo.json`；后一组：`s3_m1_length_control.json`、`s3_m1_industry_absorb.json`

### Baseline

- 基准为主表 headline 规格（`ma_teacher`，未归一化 PeerMAXraw，原始收益）。
- 行业吸收规格含 PeerReturn（与 horse race 的行业规格不同），是判读「是否被行业解释」的依据。

### 实现细节

- placebo：随机重排 peer 映射或时间平移，检验效应是否消失。
- 长度控制：样本长度分箱后重估；行业吸收：加入行业固定效应/行业 peer 均值。

### 输入与前置

- `configs/parameter_registry.yaml`；`data/processed/m1_common_sample_*_v2.parquet`。
- 上游：`estimation-sample`。

### 产物与验收

- `reports/data_gates/G0/s3_m1_robustness_placebo.json`
- `reports/data_gates/G0/s3_m1_length_control.json`、`s3_m1_industry_absorb.json`
- 验收：placebo 效应不显著；行业吸收后 PeerMAXraw 显著性与符号方向与主表一致。

### 耗时与资源

分钟级（未实测）。

### 已知坑

- `gate-report` 引用的行业吸收值（v1 样本 γ=+0.172 / t=+1.75）标注「待 A12 复核轮重跑后确认」，引用前重读本实验最新产物。
- 样本版本缺省 v2 与主表 v3 不同口径。

## shadow-inference

Phase3 shadow 推断层。阶段：Phase3；脚本：`scripts/shadow_inference.py`。

### 目的

对 N-1（分层）与 N-11（PC 三层）推断做 shadow 复核，检验主表显著性对推断口径的敏感性。对应需求：给主结论一个独立推断路径的对照。

### 参数

- 命令：`python -m peerlab run --exp shadow-inference --seed 0`
- 脚本参数：`--seed`（缺省 20261026；本仓库按 5 个全局种子运行）
- 只写 `artifacts/phase3/3.4_*`，不写 G0、不改 configs。

### Baseline

- 对照 Phase2/Phase3 其它 shadow 产物（`artifacts/phase2/`、`artifacts/phase3/`）。
- 主表推断（NW(6)）是 headline，本实验是推断层的替代口径。

### 实现细节

- 分层 FM 读数（两套层基 × EW/VW）、方差分解与组内 sd 分箱、PC 三层推断。
- 抽样/自助部分消费 `--seed`，是本仓库里少数真正随种子变化的实验。

### 输入与前置

- `configs/parameter_registry.yaml`；`artifacts/phase3/`（上游 shadow 产物）。

### 产物与验收

- `artifacts/phase3/3.4_inference.csv`、`3.4_struct_layers.csv`、`3.4_variance_decomposition.csv`、`3.4_pc_inference.csv`
- 验收：四个 CSV 齐全；分层读数与主表方向一致。

### 耗时与资源

分钟级（未实测）。

### 已知坑

- 产物不进 `reports/data_gates/G0`，聚合时靠 `result/runs/<run_id>/outputs/artifacts/phase3/` 取。
