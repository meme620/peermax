# 脚本命名与目录约定

## 1. 新文件的规则

| 对象 | 规则 | 反例 |
|---|---|---|
| 实验 id | 全词名词短语，kebab-case，对象在前、限定在后；不用阶段代号、缩写、序号 | `m1-main`、`net-ma`、`exp01` |
| 目录 | 小写；语义复数或单数统一；无数字前缀 | `01_proposal/`、`experiment_2/` |
| 文件 | `snake_case.py`，只用小写字母、数字、下划线 | `s3_ma_K_explore.py` |
| 版本 | `_v2`、`_v3` | `_2`、裸 `2` |
| 批次 | `_batch1` | `_b1` |
| 日期 | `_YYYYMMDD` | `_2026-09-03` |
| 产物文件名 | 沿用生成脚本的既定命名（被门禁与契约测试硬编码引用） | — |

## 2. 快照脚本的例外

`scripts/` 下 199 个脚本沿用主仓库原名，改名或换层级会立刻破坏三类依赖：

| 依赖 | 数量 | 改名后果 |
|---|---|---|
| 测试硬编码 `scripts/(data_audit|data)/<name>.py` | 21 | 30/33 个测试模块在收集阶段报 `FileNotFoundError` |
| 被其它脚本 `import` 的本地库 | 20 | 大量 `ImportError`，库里多数是函数内延迟 import，运行到才炸 |
| 用 `Path(__file__).resolve().parents[2]` 定位仓库根 | 110 | 只要层级不变就不受影响，因此搬移必须保持深度 2 |

已知偏差不改，只在此登记：`s3_ma_K_explore.py`、`s3_ma_K_explore2.py` 含大写字母 `K`。

## 3. 前缀族的含义

| 前缀 | 含义 |
|---|---|
| `s1_` `s2_` `s25_` `s3_` `s3a_` `s4_` | 阶段（可得性盘点、日面板、因子、网络、S3A 分析师覆盖、MAX/PeerMAX） |
| `p2_` `p3_` `p3b_` | Phase2 / Phase3 / Phase3B shadow 对照 |
| `s3_m1_*`、`m1` | M1/S5 定价事实与回归；`m1` 只在 `s3_m1_*` 里出现，不是独立族 |
| `fetch_*` | 外部取数 |
| `build_*` | 面板/映射/词库构建 |
| `audit_*`、`*_audit`、`*_check`、`*_crosscheck`、`*_smoketest`、`*_diag`、`*_probe` | 一次性检查与审计 |

## 4. 现状审计（199 个 `.py`）

| 角色 | 数量 | 位置 | 说明 |
|---|---|---|---|
| `stage` | 47 | `scripts/data_audit/` | `docs/pipeline.md` §3.2 的阶段入口 |
| `analysis` | 54 | `scripts/data_audit/` | 链上分析步骤，未列入阶段命令表 |
| `library` | 20 | `scripts/data_audit/`、`scripts/data/` | 被其它脚本 import |
| `oneoff` | 56 | `scripts/oneoff/` 43 个 + 原目录 13 个 | 一次性检查、抽样审计、历史复核 |
| `support` | 20 | `scripts/data/` | 抓数与基础 builder |
| `deliverable` / `external` | 1 / 1 | 同名目录 | 汇报物料、外部基准 |

逐文件归属、理由、搬迁目标见 `docs/script_inventory.csv`。

命名异常清单：

- 大写字母：`s3_ma_K_explore.py`、`s3_ma_K_explore2.py`（已在原位保留）。
- 尾部裸数字混用：`_explore2`（序号）、`_pass2`（轮次）、`_cfret2`（变体）、`_batch1`（批次）、`_h1h8`（规格名）、`_20260903`（日期）、`_v2`/`_v7`（版本）。新文件按 §1 区分。
- 前缀族不一致：`m1` 不是独立族，全部挂在 `s3_m1_*` 下。

## 5. 搬迁映射

共 43 个一次性脚本移入 `scripts/oneoff/`，文件名不变：

- 来自 `scripts/data/` 5 个：`audit_drive_batch1`、`audit_drive_crosscheck`、`audit_schema`、`audit_sw2_with_ths`、`check_link_audit`、`run_universe_audit` 一类（含 `drive_batch*_crosscheck`）。
- 来自 `scripts/data_audit/` 38 个：`s3a_smoketest_*`（13）、`s3_*_audit`（11）、`*_check`（5）、`*_crosscheck`（3）、其余诊断（6）。

13 个一次性脚本留在 `scripts/data_audit/` 原位，因为它们 `import` 同目录的库，搬走会 `ModuleNotFoundError`：

`s3_code_design_audit`、`s3_ma_K_explore`、`s3_ma_K_explore2`、`s3_ma_fulltext_cache_verify`、`s3_ma_lexicon_check`、`s3_ma_vector_diag`、`s3_ma_weighting_cross_audit`、`s3_pc_sign_diag_h1h8`、`s3_sign_reversal_probe_batch1`、`s3a_9k_compare`、`s3a_audit_delisting_20260903`、`s3a_rev_crosscheck`、`s3a_st_sensitivity`。

搬移前后 `python -m peerlab smoke --imports` 结果一致：146 ok、24 缺配置、29 缺数据目录、0 依赖失败、0 其它失败。

## 6. 检查

`python -m peerlab lint` 检查：

1. `experiments/<id>/README.md` 与注册表 id 一一对应，且含 8 个规定小节；
2. `scripts/` 之外的新增 `.py` 全部 snake_case、无大写、无裸尾部数字；
3. 目录名无 `0N_` 前缀；
4. 实验 id 不含 `s1|s2|s3|s4|m1|p2|p3` 之类阶段代号。

违规逐条打印并返回 1；`make lint` 是同一命令。
