# 脚本命名与目录约定

## 1. 规则

| 对象 | 规则 | 反例 |
|---|---|---|
| 目录 | 小写、语义全词；不嵌套超过 2 层；不出现只装一个文件的目录 | `scripts/lib/m1/`、`experiments/<id>/README.md` |
| 脚本文件 | `snake_case`，`<语义前缀><对象>_<动作>.py`；只用小写字母、数字、下划线 | `s3_m1_2_fm_main.py`、`s3_ma_K_explore2.py` |
| 阶段代号 | 任何路径里都不许出现 `s1/s2/s25/s3/s3a/s3b/s4/m1/m2/m3/p1/p2/p3/p3b/p4` | `s3a_9h_absorb.py`、`p2_6_shadow_m1_compare.py` |
| 序号 | 不用 `step3`、`9C`、`N4`、`_explore2` 这类序号/轮次 | `analyst_9c_fm_regression.py` |
| 版本 | `_v2`、`_v7` 后缀 | `_2`、裸 `2` |
| 批次 | `_batch1` 后缀 | `_b1` |
| 日期 | 不写进文件名 | `_20260903` |
| 实验 id | 全词名词短语，kebab-case | `m1-main`、`net-ma` |
| 实验文档 | 统一写在 `docs/experiments.md`：每个 id 一节、八小节 | `experiments/<id>/README.md` |

阶段编号（S1…S4、M1、Phase2/3）只作为**领域词汇**出现在 `docs/pipeline.md` 的叙述与表格里，不进入任何路径或模块名。

## 2. 语义前缀

| 前缀 | 含义 |
|---|---|
| `inventory_` | 可得性盘点 |
| `daily_panel_` / `factor_` | 日面板与收益 / FF4 与 REV 因子 |
| `network_business_` / `network_mda_` | M-A 业务章节网络 / M-B MD&A 网络 |
| `network_` | 网络审计、密度零模型、招股书锚、τ 校准 |
| `pricing_` | M1 定价事实与回归，共享库 `pricing_lib.py` |
| `competition_` | 竞争强度对表与反号诊断 |
| `corpus_` | 语料接入与文本质量 |
| `analyst_` | S3A 分析师覆盖网络 |
| `peermax_` | MAX / PeerMAX 面板 |
| `shadow_` | Phase2/3/3B shadow 对照 |
| `panel_` | 面板 metadata 重写 |
| `fetch_` / `build_` / `extract_` | 外部取数与基础构建 |
| `audit_` / `*_check` / `*_smoketest` | 一次性审计与复核 |

改名前的旧名逐文件登记在 `docs/script_inventory.csv` 的 `former_name` 列，这是与主仓库同步时唯一的映射依据。

## 3. lint 检查什么

`python -m peerlab lint`（`make lint`）逐条检查；有任一问题即返回 1：

1. `docs/experiments.md` 里每个注册 id 对应一个 `## <id>` 块、八个小节齐全，且不出现未登记的 `## ` 标题；
2. 实验 id 为 kebab-case，且不含阶段代号；
3. 每个被 Git 跟踪的路径：目录段与文件名不含阶段代号、目录不含数字前缀、`scripts` 与 `tests` 下的 `.py/.sh/.md` 是 snake_case（`__init__` 类文件除外）、不以裸数字结尾（`_v2`/`_batch1` 除外）；
4. 任何被跟踪路径距仓库根不超过 2 层，且没有只装一个文件的目录；
5. `scripts/`、`tests/`、`peerlab/` 里的 import：本地模块必须能在 `scripts/` 找到实体文件，且不得残留阶段代号模块名。

## 4. 外部契约例外（本轮不改名）

| 对象 | 例子 | 为什么保留 | 退役条件 |
|---|---|---|---|
| `configs/` 的 YAML 名 | `configs/m1_contracts.yaml`、`configs/s3a_final_spec_v1.yaml` | 主仓库提供，sha 锚写在 `version_manifest.yaml`；改名后 fail-closed 校验会退出 2 | 主仓库同步改名并迁移 manifest 锚 |
| 数据产物名 | `reports/data_gates/G0/s3_m1_pricing_table.json`、`data/processed/m1_common_sample_ma_teacher_v2.parquet`、`artifacts/phase3/3.4_*.csv` | 写它的脚本、读它的门禁、`version_manifest` 的 sha 与产物登记、主仓库 `reports/` 里的实体文件四处对齐；本仓没有 `configs/`，改不了登记表，改了整条链读不到自己的产物 | 主仓库同步改产物名 + 重跑门禁 + 更新 `version_manifest` |
| `data/raw/**`、`data/external/**` | `data/raw/teacher/mda_data_2023.csv` | 老师提供的原始输入，改名即断源 | 不退役 |
| 引用主仓库审计/执行报告的字符串 | `reports/audits/s3_pc_sign_diagnostics_20261021.md` | 本仓没有这些文件，改名会让引用指向不存在的路径 | 主仓库同步改名 |

除这四类之外，仓库树里不再出现任何阶段代号；路径层的检查由 lint 第 3、4 条固化。

## 5. 现状

- `scripts/` 平铺 203 个 `.py` + 1 个 `.sh`；4 个共享模块 `fileio.py`、`fmt.py`、`dates.py`、`statsutil.py` 与脚本同层。
- `tests/` 平铺 39 个测试模块 + `conftest.py`。
- 199 个原快照脚本全部有 `former_name`；4 个共享模块由 `scripts/lib/` 上移（`io.py`→`fileio.py`，`stats.py`→`statsutil.py`）。
- 原先按 `m1/`、`prospectus/`、`diag/`、`shadow/` 拆出去的 29 个模块已按 AST 逐节点核对后合回 15 个宿主脚本。
