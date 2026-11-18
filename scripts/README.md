# scripts/

计算链代码。**文件名与目录层级不可随意改**：110 个脚本用 `Path(__file__).resolve().parents[2]` 定位仓库根，21 个被测试硬编码引用，20 个被其它脚本 import。

| 目录 | 内容 | 规模 |
|---|---|---|
| `data/` | 抓数与基础 builder（baostock / 东财 / 腾讯 / CNINFO / Drive / 申万 / 词库） | 30 |
| `data_audit/` | 阶段入口、链上分析步骤、库、门禁 | 124 |
| `oneoff/` | 一次性检查、抽样审计、历史复核 | 43 |
| `external/` | 外部基准（HP TNIC-3）结构统计 | 1 |
| `deliverables/` | 汇报物料生成 | 1 |

逐文件角色与理由见 `docs/script_inventory.csv`，命名规则见 `docs/script_naming.md`。

执行不要手敲脚本路径：用 `python -m peerlab run --exp <id> --seed <n>`，实验清单与参数见 `experiments/`。

`scripts/oneoff/` 里是一次性脚本，保留原名（改名会与 `docs/script_inventory.csv` 的映射脱节），不要把它们当主线入口。
