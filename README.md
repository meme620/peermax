# PeerMAX

A 股产品市场同侪的彩票型收益（MAX）溢出研究：计算链代码、外部输入材料与运行环境。

代码是 **2026-11-17** 的主仓库快照；`scripts/` 与 `tests/` 按主仓库层级铺在仓库根（`parents[2]` / `parents[1]` 依赖这一层级），**不要移动或改名**。

## 目录

| 路径 | 内容 |
|---|---|
| `scripts/` | 计算链：`data/` 抓数与 builder（35）、`data_audit/` 阶段管线与审计（163）、`external/`（1）、`deliverables/`（1） |
| `tests/` | 契约测试，33 个模块 |
| `ref/` | 权威源头文档：立项文档 5、老师材料与参考论文 3、组内论文 2 |
| `docs/` | 项目文档：`pipeline.md` 计算链、`materials.md` 材料来源、`environment.md` 环境、`data_layout.md` 数据布局 |
| `data/` | 数据根：`raw/` 原始输入、`external/` 外部参照、`processed/` 派生（不入 Git） |
| `configs/` | 冻结参数与版本锚（待主仓库提供） |
| `scripts/` | 计算链：`data/` 抓数、`data_audit/` 阶段与库、`oneoff/` 一次性检查、`lib/` 原子模块、`external/`、`deliverables/`（见 `scripts/README.md`） |
| `experiments/` | 10 个实验的规范文档：目的、参数、baseline、实现细节、产物与验收（见 `experiments/README.md`） |
| `peerlab/` | 执行 CLI：实验注册、种子注入、运行记录、结果聚合，`python -m peerlab --help` |
| `result/` | 结果层：`tables/` 跨种子聚合表、`summary.md` 摘要、`runs/` 运行记录（见 `result/README.md`） |
| `Makefile` `requirements.txt` `.python-version` `pytest.ini` | 环境定义 |

## 快速开始

```bash
make env          # 建 .venv（Python 3.13）并装 requirements.txt
make preflight    # 解释器 / 依赖 / configs / 数据 四块一次打印
make doctor       # 解释器 / 依赖 / configs / 数据 四块自检
make imports      # 逐个导入 scripts/ 下 199 个脚本
make smoke        # L0 冒烟：10 个实验逐个判就绪
make lint         # 命名与实验文档规范检查
make status       # 每个实验缺什么
make run-all      # 全部实验 × SEEDS=(0,1,2,3,4)，当前全部 blocked
make report       # 重算 result/tables 与 summary.md
make selftest     # peerlab 单测
```

## 当前状态

| 项 | 状态 |
|---|---|
| 环境 | 就绪：Python 3.13.12，14 个直接依赖与传递依赖全部固定版本 |
| 脚本导入 | 145 ok / 24 缺配置 / 30 缺数据目录 / 0 个依赖问题 |
| 教师数据 | 就绪：42 个文件在 `data/raw/` 与 `data/external/`（4.1 GB，不入 Git） |
| `configs/` | **缺 6 个 YAML**，需主仓库提供；缺它时 `s3_m1_lib` 导入即 `SystemExit(2)` |
| `data/processed/` | 空：需要跑链生成，而跑链又依赖 `configs/` |
| 全量 `pytest` | 收集阶段即中断（同上原因），`make test` 现在必然红 |

## 文档

- [`docs/pipeline.md`](docs/pipeline.md)：阶段总表、分阶段命令、六条运行硬约束、配置文件与数据目录清单。
- [`docs/materials.md`](docs/materials.md)：52 个外部输入文件的来源、批次、编码与时点快照说明。
- [`docs/environment.md`](docs/environment.md)：解释器与依赖版本、安装与三条检查命令、T1–T4 等级现状、平台边界。
- [`docs/data_layout.md`](docs/data_layout.md)：`data/` 结构、14 个 raw 目录就绪状态、交付包到代码路径的映射。

## 使用纪律

- `scripts/`、`tests/` 是只读快照，编辑后当主线用会导致与主仓库对不上。
- `data/` 与 `.venv/` 不入 Git；`data/` 有 4.1 GB，提交前先看 `git status --ignored`。
- 补 `configs/` 时不要手改其中任何值：它们与 `version_manifest.yaml` 的 sha 锚绑定，fail-closed 脚本会逐项校验。
