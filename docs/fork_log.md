# 与主仓库的分叉记录

本仓库的 `scripts/` 最初是主仓库 2026-11-17 快照的只读副本。从下列改动开始，它与主仓库**不再逐字节一致**；同步主仓库时需要按文件逐条合并且重跑校验。

## 2026-11-18 第一轮：抽取与拆分（本轮已回退）

| 日期 | 改动 | 结果 |
|---|---|---|
| 2026-11-18 | 抽取重复工具到 `scripts/lib/`（`io.py`、`fmt.py`、`dates.py`、`stats.py`），17 个脚本改为导入 | 四个工具保留，当晚随目录拍平上移到 `scripts/`（`io.py`→`fileio.py`、`stats.py`→`statsutil.py`） |
| 2026-11-18 | 把 `pricing_lib`、`network_prospectus_anchor`、`competition_sign_diagnostic` 与 12 个 shadow 脚本拆到 `scripts/lib/{m1,prospectus,diag,shadow}/` | **已回退**：29 个模块按 AST 逐节点核对后合回 15 个宿主脚本，`scripts/lib/` 目录消失 |
| 2026-11-18 | 拆分后宿主脚本保留为再导出层，`main()` 与常量混放 | **已回退**：合并按「依赖模块在前、宿主 `main()` 在后」重排，模块级常量先定义后使用 |

## 2026-11-18 第二轮：路径层去阶段代号 + 全平铺

| 改动 | 影响 |
|---|---|
| 199 个快照脚本去掉 `s1/s2/s25/s3/s3a/s3b/s4/m1/p2/p3/p3b/p4` 前缀，改为语义前缀（`inventory_`、`daily_panel_`、`factor_`、`network_business_`、`network_mda_`、`pricing_`、`competition_`、`corpus_`、`analyst_`、`peermax_`、`shadow_`、`panel_`） | 逐文件旧名→新名登记在 `docs/script_inventory.csv` 的 `former_name` 列 |
| `scripts/{data,data_audit,oneoff,external,deliverables}/` 与 `scripts/lib/` 全部拍平到 `scripts/` | 脚本定位根由 `parents[2]` 改 `parents[1]`；跨目录 `sys.path` 插入改成同目录 |
| 22 个测试模块改名（`test_p2_*`/`test_p3_*` → `test_shadow_*` 等），`tests_lib/` 与 `peerlab/tests/` 并入 `tests/` | 测试只剩一个根；`make selftest` 显式列 6 个快测文件 |
| `experiments/<id>/README.md` → `experiments/<id>.md` | 10 个单文件目录消失，lint 改读单文件 |
| `peerlab/lint.py` 新增路径层规则：阶段代号、大写、裸数字、目录深度、单文件目录、本地 import 可解析 | `make lint` 现在会拦住旧命名回流 |
| `configs/` 六个 YAML 按新基线重建（原主仓库 registry 1.5.32 / manifest 1.0.110 不可得） | 参数取契约测试、源码注释与保守基线值；M1 输入的 sha256 与 S3A 输入哈希留空，待数据生成后按 `configs/README.md` 补齐 |

**没有改的**：`configs/*.yaml` 文件名、`reports/data_gates/G0/` 与 `data/processed/` 的产物名、`data/raw/` 输入名。它们是主仓库 `version_manifest.yaml` 的 sha 锚与门禁登记契约，本仓没有 `configs/`，改不了登记表。退役条件写在 `docs/script_naming.md` 的例外表里。

## 合并纪律

1. 与主仓库同步时按 `docs/script_inventory.csv` 的 `former_name` 列做文件名映射，逐文件比对函数体，不要按路径直接覆盖。
2. 函数体不做改写；改名或搬移后跑 `python -m peerlab smoke --imports --clean`，要求 0 dep、0 other。
3. 注册表（`peerlab/registry.py`）、测试里的 `scripts/<name>.py`、文档里的命令三处必须同批更新。
4. 主仓库 `version_manifest.yaml` / `parameter_registry.yaml` 里登记的旧脚本路径不会自动跟随，需要在主仓库侧同步改并重跑门禁。
5. 新增脚本先过 `make lint`（阶段代号、大小写、目录深度、import 可解析）。
