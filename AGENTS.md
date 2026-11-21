# Repository Guidelines

PeerMAX（A 股产品市场同侪的彩票型收益溢出）的代码与材料仓：2026-11-18 起去阶段代号、`scripts/` 与 `tests/` 全部平铺。

## 文档分工

| 文件 | 回答什么 |
|---|---|
| `spec/design.yaml` | 设计口径的可执行版：变量→实现脚本、样本与 PIT、pricing 判据→registry 键、数据门→gate、结构约束→lint、十条识别红线 |
| `spec/stages.yaml` | 阶段 → 脚本 → 产物 |
| `spec/inputs.yaml` | 外部输入 → 消费者脚本 → 是否必需 |
| `peerlab/registry.py` | 实验 spec（id / steps / requires / outputs / notes）；`python -m peerlab experiments` 直接读 |
| `configs/*.yaml` | 冻结参数、版本锚、claim 与口径卡 |

文档只承担索引与历史。口径改动改 `spec/`，不写新文档；改完跑 `python -m peerlab spec` 与 `make lint`（lint 内含 spec 校验）。

## 目录

- `scripts/`：186 个 `.py` + `ocr_tesseract_chunk.sh`，平铺一层；脚本用 `Path(__file__).resolve().parents[1]` 定位仓库根。
- `tests/`：65 个测试模块 + `conftest.py`（含 5 peerlab 单测、7 数据门单测、1 共享模块单测）。
- `peerlab/`：执行 CLI 与注册表（10 个实验、`SEEDS = (0, 1, 2, 3, 4)`）。
- `configs/`：六个 YAML，按新基线重建（`configs/README.md` 记来源与未锚项）。
- `ref/`：权威外部输入（立项文档 4 份 PDF + 1 份真源 tex、老师团队材料 3、组内论文 2）。`data/raw/`：老师包与 Drive，二进制走 Release `data-raw-v1`。`data/processed/`：派生产物。`result/`：结果层。

## 命令

```bash
make env        # .venv（Python 3.13）+ requirements.txt
make doctor     # 解释器 / 依赖 / configs / 数据 四块自检
make imports    # 逐个导入 186 个脚本（会写出零星报告文件，跑完自动清理）
make smoke      # peerlab L0：10 个实验逐个判就绪
make lint       # 命名、清单、spec 引用、`scripts/<name>.py` 字面量断链、codegraph 索引新鲜度
make status     # 逐实验列出缺什么
make gate       # 数据不变量门
make selftest   # 快测：peerlab、共享模块、数据门
python -m pytest tests/ -q   # 契约测试，必须显式写 tests/；缺 data/processed 产物时会在收集期中断
```

## 数据与参数纪律

- `configs/parameter_registry.yaml` 是权威（逐参数登记取值与来源）。改登记值、headline 规则或样本定义要走 change order、升版本、同步 `configs/version_manifest.yaml`；不得为让结果过关热改登记值。
- 样本版本不可变（`pricing_sample_v1_2015_2023` / `v2` / `v3_20261115` / `v4_20261116`）。重建用新版本名，旧 parquet 保留不覆盖。
- PIT：join 用 `max{available_at < month}`（严格小于），不引入同月信息；契约测试与 Gate1 provenance/metadata 审计会抓。
- 缺失不是 0：`missing` / `not_covered` / `not_matched` / `unknown_timing` 分开编码；`actual_display` 缺失时 `V` 保持 missing，不得写 `V=D`。
- fail-closed 不许绕：`pricing_lib` 导入即校验 registry（参数完整性、值域、派生关系、`registry_version` 与 `analyst_spec.yaml` governance 对齐），输入合同按 `configs/pricing_inputs.yaml` 的 rows/columns 实测核对，缺条目或不符直接 `SystemExit(2)`；不要用实测值回填或自动刷新合同。
- 不得宣称 empirical pass：`pricing_gate_status = FAIL`（pricing 五项 gate 全未过），standalone MAX-pricing claim 停止、只保留描述性关联；契约测试、合成 fixture、shadow 运行都属非经验证据。

## 改动与校验

- 改脚本名要同步三处：文件、`peerlab/registry.py`、测试与文档里的 `scripts/<name>.py` 字面量。
- `make lint` 会检查 codegraph 索引是否落后于工作区，落后即失败并提示 `codegraph sync`。索引按设计不入库（`.gitignore`），没有 `.codegraph/` 的机器自动跳过。提交门（lefthook pre-commit）走 `--no-codegraph`：提交前工作区必然领先索引，否则每次提交恒红。
- 每次改动后跑 `make lint` 与相称的 `pytest tests/ -q`；全量慢，先跑相关模块再加全量。
- 改口径先改 `spec/*.yaml`（或 registry 的 `notes`），再跑 `python -m peerlab spec` 与 `make lint`；引用解析不到就是错，不要靠文档补充说明。

## 其他

- `data/raw/` 不要当垃圾清理（含下载断点与契约状态）；`artifacts/` 的生成物不手改；`ref/` 与 `data/` 里的输入按只读归档使用。
- 回复用简体中文（代码、路径、命令除外），结论带 `file:line` 或产物路径。
