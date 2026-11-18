# scripts/

计算链代码，**全部平铺**在这一层：203 个 `.py` + `ocr_tesseract_chunk.sh`。脚本用 `Path(__file__).resolve().parents[1]` 定位仓库根，同目录互相 import。

## 命名家族

| 前缀 | 含义 | 例子 |
|---|---|---|
| `inventory_` | 可得性盘点 | `inventory_universe.py` |
| `daily_panel_` / `factor_` | 日面板与收益 / FF4 与 REV 因子 | `daily_panel_build.py`、`factor_validate.py` |
| `network_business_` / `network_mda_` | M-A 业务章节网络 / M-B MD&A 网络 | `network_business_edges_build.py`、`network_mda_full_matrix.py` |
| `network_` | 网络审计、密度零模型、招股书锚 | `network_density_null.py` |
| `pricing_` | M1 定价事实与回归（共享库 `pricing_lib.py`） | `pricing_fm_main.py` |
| `competition_` | 竞争强度对表与反号诊断 | `competition_pricing_correspondence.py` |
| `corpus_` | 语料接入、章节扫描、完整性 | `corpus_teacher_ingest.py` |
| `analyst_` | S3A 分析师覆盖网络 | `analyst_edges_build.py` |
| `peermax_` | MAX / PeerMAX 面板 | `peermax_panel_build.py` |
| `shadow_` | Phase2/3/3B shadow 对照 | `shadow_inference.py` |
| `fetch_` `build_` `extract_` | 抓数与基础构建 | `fetch_tencent_daily.py` |
| `audit_` `*_check` `*_smoketest` | 一次性审计与复核（56 个，不当主线入口） | `network_business_acceptance_check.py` |

共享模块 4 个：`fileio.py`（原子写、JSON、sha）、`fmt.py`、`dates.py`、`statsutil.py`（NW / OLS）。

逐文件的角色、调用方与测试引用见 `docs/script_inventory.csv`（含 `former_name` 列，记录改名前的名字），命名规则与例外见 `docs/script_naming.md`。

执行不要手敲脚本路径：用 `python -m peerlab run --exp <id> --seed <n>`，实验清单与参数见 `docs/experiments.md`。
