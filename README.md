# PeerMAX

A 股产品市场同侪的彩票型收益（MAX）溢出研究：计算链代码、外部输入材料与运行环境。

代码起自 **2026-11-17** 的一份快照，2026-11-18 起在本仓库整改：文件名与目录层级已统一（去阶段代号、全部平铺）。`scripts/` 与 `tests/` 平铺在仓库根，脚本用 `parents[1]` 定位仓库根。

## 目录

| 路径 | 内容 |
|---|---|
| `scripts/` | 计算链：191 个 `.py` + 1 个 `.sh`，平铺；命名规则由 `peerlab lint` 强制 |
| `tests/` | 56 个测试模块（含 peerlab 单测、数据门单测、within 变换与 skip 判定自检）+ `conftest.py` |
| `ref/` | 权威输入：立项文档 5、老师材料与参考论文 3、组内论文 2 |
| `spec/` | 可执行 spec：`design.yaml`（口径、pricing 判据、数据门、约束、红线）、`stages.yaml`（阶段→脚本→产物）、`inputs.yaml`（外部输入与消费者） |
| `peerlab/` | 执行 CLI：实验注册、种子注入、运行记录、结果聚合，`python -m peerlab --help` |
| `result/` | 结果层：`tables/` 跨种子聚合表、`summary.md` 摘要、`runs/` 运行记录 |
| `data/` | 数据根：`raw/` 原始输入（目录骨架入库，4.1 GB 二进制走 `data-raw-v1` Release 附件）、`external/` 外部参照、`processed/` 派生 |
| `Makefile` `requirements.txt` `.python-version` `pytest.ini` | 环境定义 |

## 快速开始

```bash
make env          # 建 .venv（Python 3.13）并装 requirements.txt
lefthook install  # 装 git hooks：pre-commit 跑 lint + selftest，pre-push 跑数据门
make doctor       # 解释器 / 依赖 / configs / 数据 四块自检
make imports      # 逐个导入 scripts/ 下 191 个脚本
make smoke        # L0 冒烟：10 个实验逐个判就绪
make lint         # 命名与结构规范检查
make status       # 每个实验缺什么
make run-all      # 全部实验 × SEEDS=(0,1,2,3,4)，当前 0 就绪 / 10 受阻（上游产物未生成）
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
| 研究状态 | `pricing_gate_status = FAIL`：pricing 五项 gate 全未过，standalone MAX-pricing claim 停止、只保留描述性关联（判据见 `spec/design.yaml` 的 `pricing_gates`，claim 见 `configs/claim_registry.yaml`） |
| 环境 | 就绪：Python 3.13.12，14 个直接依赖与传递依赖全部固定版本 |
| 脚本导入 | 191 个：182 ok / 0 缺配置 / 10 缺数据目录 / 0 依赖问题 / 0 其它 |
| 教师数据 | 就绪：42 个文件在 `data/raw/` 与 `data/external/`；`raw/` 的 4.1 GB 二进制走 `data-raw-v1` Release 附件，不入 Git 历史 |
| `configs/` | 六个 YAML 已按新基线重建（`configs/README.md`）；pricing 九个主输入的 rows / columns 待 `data/processed/` 产出后登记 |
| `data/processed/` | 空：需要按 `python -m peerlab experiments` 列出的实验生成上游产物 |
| 全量 `pytest tests` | 423 passed / 278 skipped / 0 failed：缺产物的契约用例降级为 skip（`require_artifacts`），不再与真实缺陷混在一起 |

## spec 与校验

口径写成可执行 spec，不靠文档同步：

- [`spec/design.yaml`](spec/design.yaml)：研究问题、变量与实现脚本、样本与 PIT、pricing 五项判据（指向 registry 键）、G1–G8 数据门的机械检查、六条结构约束、十条识别红线。
- [`spec/stages.yaml`](spec/stages.yaml)：阶段 → 脚本 → 产物。
- [`spec/inputs.yaml`](spec/inputs.yaml)：外部输入路径、消费者脚本、是否必需。
- `peerlab/registry.py` 的 `EXPERIMENT_NOTES`：10 个实验的 purpose 与注意事项。

```bash
python -m peerlab spec          # 校验 spec 引用可解析：脚本 / 数据门 / lint 检查 / 测试 / 路径
python -m peerlab experiments   # 打印实验 spec（purpose、参数、坑、steps、产物）
make lint                       # 命名 + 清单 + spec，三者一起过才放行
```

## 使用纪律

- `data/raw/` 的二进制不入 Git，已建子目录留 `.gitkeep` 占位（当前 6 个），原始文件从 `data-raw-v1` Release 取；`.venv/` 不入 Git。
- 补 `configs/` 时按登记流程改值：`parameter_registry.yaml` 由 `pricing_lib` 导入即校验，改登记值要升 `registry_version` 并同步 `version_manifest.yaml`；定价主输入的行数与列名写进 `pricing_inputs.yaml`。
