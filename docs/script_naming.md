# 脚本命名与目录约定

## 1. 规则

| 对象 | 规则 | 反例 |
|---|---|---|
| 目录 | 小写、语义全词；不嵌套超过 2 层；不出现只装一个文件的目录 | `scripts/lib/m1/`、`experiments/<id>/README.md` |
| 脚本文件 | `snake_case`，`<语义前缀><对象>_<动作>.py`；只用小写字母、数字、下划线 | `s3_m1_2_fm_main.py`、`s3_ma_K_explore2.py` |
| 阶段代号 | 任何路径里都不许出现 `s1/s2/s25/s3/s3a/s3b/s4/m1/m2/m3/p1/p2/p3/p3b/p4` | `s3a_9h_absorb.py`、`p2_6_shadow_m1_compare.py` |
| 序号 | 不用 `step3`、`9C`、`N4`、`_explore2` 这类序号/轮次 | `analyst_9c_fm_regression.py` |
| 版本号 | 文件名不带版本号（`_v1`/`_v2`/`_v7` 一律不进路径）；口径版本只出现在产物名、`network_version` 与 `sample_version` 值里 | `text_mda_extract_v2.py`、`s3a_final_spec_v1.yaml` |
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
| `fetch_` / `build_` / `extract_` | 外部取数与基础构建 |
| `audit_` / `*_check` / `*_smoketest` | 一次性审计与复核 |

改名前的旧名逐文件登记在 `docs/script_inventory.csv` 的 `former_name` 列，这是与主仓库同步时唯一的映射依据。

## 3. lint 检查什么

`python -m peerlab lint`（`make lint`）逐条检查；有任一问题即返回 1：

1. `docs/experiments.md` 里每个注册 id 对应一个 `## <id>` 块、八个小节齐全，且不出现未登记的 `## ` 标题；
2. 实验 id 为 kebab-case，且不含阶段代号；
3. 每个被 Git 跟踪的路径：目录段与文件名不含阶段代号、目录不含数字前缀、`scripts` 与 `tests` 下的 `.py/.sh/.md` 是 snake_case（`__init__` 类文件除外）、不以裸数字结尾、文件名不带 `_v<数字>` 版本后缀；
4. 任何被跟踪路径距仓库根不超过 2 层，且没有只装一个文件的目录；
5. `scripts/`、`tests/`、`peerlab/` 里的 import：本地模块必须能在 `scripts/` 找到实体文件，且不得残留阶段代号模块名；
6. 代码与文档里的 `scripts/<name>.py` 字面量必须存在（删脚本、改名后的断链变成 lint 失败；`tests/` 跳过，那里会往 `tmp_path` 合成同名 fixture）；
7. `docs/script_inventory.csv` 与 `scripts/` 双向一致，`role` 在白名单内，`test_ref` 指向的测试文件存在；
8. 非 `oneoff` 脚本不得 import `oneoff` 脚本（证据脚本不能变成主链依赖）。

## 4. 外部契约例外（本轮不改名）

| 对象 | 例子 | 为什么保留 | 退役条件 |
|---|---|---|---|
| 数据产物名里的口径版本 | `data/processed/m1_common_sample_ma_teacher_v2.parquet` / `_v3`、`monthly_returns_clean_v2.csv`、`product_lexicon_v2.csv` | v2/v3 是不同的样本口径（v3 = headline，v2 = superseded），不是文件版本；两者并存，去后缀会撞名并改掉口径含义 | 只有口径合一后才改名，同时更新 `version_manifest` 的 sha 锚与门禁产物登记 |
| 网络版本号 | `pm_mb_tfidf_v1_2021`、`pm_ma_teacher_v1` | 写在 `version_manifest.yaml` 的白名单里，边表 / build_report / 下游选择都按它对齐；它是数据版本标识，不是文件名 | 换网构造方法时整链升版，旧版按 manifest 退役 |
| `data/raw/**`、`data/external/**` | `data/raw/teacher/mda_data_2023.csv` | 老师提供的原始输入，改名即断源 | 不退役 |
| 引用主仓库审计/执行报告的字符串 | `reports/audits/s3_pc_sign_diagnostics_20261021.md` | 本仓没有这些文件，改名会让引用指向不存在的路径 | 主仓库同步改名 |

仓库树里不再出现阶段代号与文件名版本号；路径层的检查由 lint 第 3、4 条固化。产物名与网络版本号里的口径版本留在数据侧，不进路径。

## 5. 现状

- `scripts/` 平铺 192 个 `.py` + 1 个 `.sh`；4 个共享模块 `fileio.py`、`fmt.py`、`dates.py`、`statsutil.py` 与脚本同层。
- `tests/` 平铺 45 个测试模块 + `conftest.py`（含 6 个数据门单测）。
- 188 个原快照脚本全部有 `former_name`；4 个共享模块由 `scripts/lib/` 上移（`io.py`→`fileio.py`，`stats.py`→`statsutil.py`）。
- 原先按 `m1/`、`prospectus/`、`diag/`、`shadow/` 拆出去的 29 个模块已按 AST 逐节点核对后合回 15 个宿主脚本。
- 执行不要手敲脚本路径：用 `python -m peerlab run --exp <id> --seed <n>`；逐文件的角色、调用方与测试引用见 `docs/script_inventory.csv`。
