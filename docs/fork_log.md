# 与主仓库的分叉记录

本仓库的 `scripts/` 最初是主仓库 2026-11-17 快照的只读副本。从下列改动开始，它与主仓库**不再逐字节一致**；同步主仓库时需要按文件逐条合并且重跑校验。

## 改动清单

| 日期 | 改动 | 影响 | 理由 |
|---|---|---|---|
| 2026-11-18 | 抽取重复工具到 `scripts/lib/`（`io.py`、`fmt.py`、`dates.py`、`stats.py`），17 个脚本改为导入，移除 30 个重复定义 | `_next_month`×5、`nw_t`×5、`nw_ols`×3、`sha256_file`×3、`_atomic_write_text`×3、`_json_default`×5、`write_json_atomic`×3、`fmt_float`×2、`fmt_int`×2 | 函数体逐 token 相同，样板重复无必要 |
| 2026-11-18 | 拆分 `s3_m1_lib`（1474 → 256 行）到 `scripts/lib/m1/{paths,contracts,stats,data}.py` | 24 个依赖方不受影响：`s3_m1_lib` 保留为再导出层，75 个被导入名字全部存在 | 枢纽库单文件过大 |
| 2026-11-18 | 拆分 `s3_network_prospectus_anchor`（3399 → 730 行）到 `scripts/lib/prospectus/` 六模块 | 47 个函数逐一搬迁 | 单文件 3399 行，文本/OCR/审计/报告混写 |
| 2026-11-18 | 拆分 `s3_pc_sign_diag_h1h8`（2123 → 187 行）到 `scripts/lib/diag/` 五模块 | 29 个函数逐一搬迁 | 单文件 2123 行 |
| 2026-11-18 | 12 个 shadow 脚本的辅助函数与常量搬到 `scripts/lib/shadow/<脚本名>.py`，原脚本保留 `main()` 与再导出 | `p2_*`/`p3_*` 单文件 714–1643 行降到 256–797 行 | `main()` 之外的实现与常量混在一个文件里 |

## 合并纪律

1. 函数体不做改写，只做位置搬迁；搬迁前后用 AST body 逐 token 哈希核对（本轮 514 个函数全部一致）。
2. 被测试硬编码引用的 21 个脚本路径不变（只改内容，不改名）。
3. 任何改动后 `python -m peerlab smoke --imports --clean` 必须 0 dep / 0 other。
4. 与主仓库合并冲突时，以主仓库逻辑为准，重新做一次抽取，不手工粘合。
5. `scripts/lib/` 下的模块自带 `sys.path` bootstrap（插入 `scripts/`、`scripts/data_audit`、`scripts/data`），单独导入也能解析兄弟模块。
