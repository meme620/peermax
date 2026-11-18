# PeerMAX

A 股产品市场同侪的彩票型收益（MAX）溢出研究：计算链代码与外部输入材料。

仓库内容来自 **2026-11-17** 的代码快照与同期材料归档。数据（`data/`）、配置（`configs/`）、报告与审计产物（`reports/`）、汇报成品（`deliverables/`）、审计工作记录（`artifacts/`）都不在本仓库。

## 目录

| 路径 | 内容 |
|---|---|
| `scripts/data/` | 抓数与基础 builder（baostock / 东财 / 腾讯 / CNINFO / Drive / 申万 / 词库），35 个脚本 |
| `scripts/data_audit/` | 项目主计算链：S1、S2、S2.5、S3 网络、S3A、S4、M1/S5、Phase2/3/3B shadow、Gate1/2/5，162 `.py` + 1 `.sh` |
| `scripts/external/` | 外部基准（HP TNIC-3）结构统计 |
| `scripts/deliverables/` | 汇报物料生成脚本 |
| `tests/` | 契约测试，33 个模块 |
| `docs/pipeline.md` | 计算链说明：阶段总表、分阶段命令、路径与运行约束、与主仓库的关系 |
| `materials/` | 外部输入：立项文档 5 份、老师材料与参考论文 3 份、组内论文 2 份，说明见 `materials/README.md` |
| `code_MANIFEST.csv` | 代码快照逐文件清单，233 条 |
| `code_SHA256SUMS.txt` | 代码快照校验和，233 行 |

## 来源与完整性

四个投递包解压、逐文件校验后合并，投递包本体已删除：

| 投递包 | sha256 | 解出到 |
|---|---|---|
| `code.zip` | `dc89f9ab10ab3e104b28fe178607c2bd7dfa807bdc490b5e422bb03ba8e2b1a9` | `scripts/`、`tests/`、`docs/pipeline.md`、`code_MANIFEST.csv`、`code_SHA256SUMS.txt` |
| `01_proposal(1).zip` | `af4d35c1c8f7ee8a83c04a4ec9e8bdbe47b898f6b2d8466fd29ac7c7e887b63d` | `materials/01_proposal/` |
| `02_teacher_note_and_papers(1).zip` | `31bc973bdd2443fb223a918e30997b2d7c438a08ff471ce11baeb06104d3dfda` | `materials/02_teacher_note_and_papers/` |
| `03_team_papers(1).zip` | `e48c4352af6f9a8ab7576e5b3c97186da11982e196d2ac72396ffcb7e41da72d` | `materials/03_team_papers/` |

校验：

```bash
shasum -a 256 -c code_SHA256SUMS.txt            # 233 条全部 OK
cd materials && shasum -a 256 -c SHA256SUMS.txt # 10 条 OK，42 条 missing（04_teacher_data 未随仓库分发）
```

材料归档的 `04_teacher_data/`（老师数据 42 个文件、4.11 GiB）不在本仓库。

## 运行前提

`scripts/` 与 `tests/` 按主仓库层级铺在仓库根：110 个脚本用 `Path(__file__).resolve().parents[2]` 定位仓库根，33 个测试用 `parents[1]`。仓库根当前缺 `configs/`、`data/`、`reports/`，直接运行会在 import 阶段失败（`s3_m1_lib` 找不到 `configs/parameter_registry.yaml`）。补齐这几层后按 `docs/pipeline.md` §3 的命令跑。

## 时间性

`scripts/`、`tests/` 是 2026-11-17 快照，主仓库之后的改动不会回流。引用前比对 `code_MANIFEST.csv` 的 `sha256` 列；刷新方式是回主仓库重新生成副本，不在副本上打补丁。
