# PeerMAX

A 股产品市场同侪的彩票型收益（MAX）溢出研究：计算链代码、外部输入材料与运行环境。

代码是 **2026-11-17** 的主仓库快照；2026-11-18 起按 `docs/fork_log.md` 分叉改造：文件名与目录层级已统一整改（去阶段代号、全部平铺），与主仓库不再逐字节一致。`scripts/` 与 `tests/` 平铺在仓库根，脚本用 `parents[1]` 定位仓库根。

## 目录

| 路径 | 内容 |
|---|---|
| `scripts/` | 计算链：192 个 `.py` + 1 个 `.sh`，平铺；命名家族见 `docs/script_naming.md`，逐文件角色见 `docs/script_inventory.csv` |
| `tests/` | 45 个测试模块（33 个契约、5 个 peerlab 单测、1 个共享模块单测、6 个数据门单测）+ `conftest.py` |
| `ref/` | 权威源头文档：立项文档 5、老师材料与参考论文 3、组内论文 2 |
| `docs/` | 项目文档 8 篇：计算链、材料来源、环境、数据布局、结果链路、脚本命名、分叉记录、脚本清单 |
| `docs/experiments.md` | 10 个实验的规范文档：每个 id 一节，八小节（目的/参数/Baseline/实现细节/输入与前置/产物与验收/耗时与资源/已知坑） |
| `peerlab/` | 执行 CLI：实验注册、种子注入、运行记录、结果聚合，`python -m peerlab --help` |
| `result/` | 结果层：`tables/` 跨种子聚合表、`summary.md` 摘要、`runs/` 运行记录 |
| `data/` | 数据根：`raw/` 原始输入（目录骨架入库，4.1 GB 二进制走 `data-raw-v1` Release 附件）、`external/` 外部参照、`processed/` 派生 |
| `Makefile` `requirements.txt` `.python-version` `pytest.ini` | 环境定义 |

## 快速开始

```bash
make env          # 建 .venv（Python 3.13）并装 requirements.txt
lefthook install  # 装 git hooks：pre-commit 跑 lint + selftest，pre-push 跑数据门
make doctor       # 解释器 / 依赖 / configs / 数据 四块自检
make imports      # 逐个导入 scripts/ 下 192 个脚本
make smoke        # L0 冒烟：10 个实验逐个判就绪
make lint         # 命名与结构规范检查
make status       # 每个实验缺什么
make run-all      # 全部实验 × SEEDS=(0,1,2,3,4)，当前全部 blocked
make report       # 重算 result/tables 与 summary.md
# 取原始数据（4.1 GB）：gh release download data-raw-v1 --repo meme620/peermax \
#   --pattern 'peermax-data-raw.tar.zst' --dir /tmp
#   tar --zstd -xf /tmp/peermax-data-raw.tar.zst -C data/raw
make gate         # 数据不变量门：pass / fail / blocked，缺产物记 blocked
make selftest     # peerlab、共享模块与数据门单测
```

## 当前状态

| 项 | 状态 |
|---|---|
| 环境 | 就绪：Python 3.13.12，14 个直接依赖与传递依赖全部固定版本 |
| 脚本导入 | 192 个：182 ok / 0 缺配置 / 10 缺数据目录 / 0 依赖问题 / 0 其它 |
| 教师数据 | 就绪：42 个文件在 `data/raw/` 与 `data/external/`；`raw/` 的 4.1 GB 二进制走 `data-raw-v1` Release 附件，不入 Git 历史 |
| `configs/` | 六个 YAML 已按新基线重建（`configs/README.md`）；M1 输入与 S3A 输入的 sha256 锚待数据生成后补 |
| `data/processed/` | 空：需要跑链生成，而跑链又依赖 `configs/` |
| 全量 `pytest tests` | 收集期中断：1 error（`data/processed/analyst/edges/` 没有边表版本），与改动前同形态 |

## 文档

- [`docs/pipeline.md`](docs/pipeline.md)：阶段总表、分阶段命令、结构约束、配置文件与数据目录清单。
- [`docs/materials.md`](docs/materials.md)：52 个外部输入文件的来源与批次、`data/` 目录布局与入库策略、raw 目录就绪状态、交付包到代码路径的映射。
- [`docs/environment.md`](docs/environment.md)：解释器与依赖版本、安装与检查命令、平台边界。
- [`docs/script_naming.md`](docs/script_naming.md)：命名规则、lint 检查项、外部契约例外表。
- [`docs/fork_log.md`](docs/fork_log.md)：与主仓库的分叉点与同步纪律。

## 使用纪律

- `data/raw/` 的二进制不入 Git，每个子目录留 `.gitkeep` 占位，原始文件从 `data-raw-v1` Release 取；`.venv/` 不入 Git。
- 补 `configs/` 时不要手改其中任何值：它们与 `version_manifest.yaml` 的 sha 锚绑定，fail-closed 脚本会逐项校验。
- 与主仓库同步时按 `docs/fork_log.md` 的改名表逐文件合并，不要直接覆盖。
