# 结果层

这里是拿结论的唯一入口：`tables/` 是跨种子聚合表，`summary.md` 是人读摘要，`runs/` 是每次运行的原始记录。

## 结构

| 路径 | 内容 | 入库 |
|---|---|---|
| `tables/<exp>.csv` | 跨种子宽表：metric × s0…s4 + mean + sd + identical | 是 |
| `metrics.csv` | 长表：run_id, exp, seed, metric, value | 是 |
| `tables/_schema_probe.md` | 首轮运行实测到的产物 JSON 顶层键 | 是 |
| `summary.md` | 每个实验的读数与是否确定性 | 是 |
| `index.json` | 运行索引（run_id/exp/seed/状态/耗时/commit） | 是 |
| `runs/<run_id>/` | run.json、logs/、outputs/（从产物目录复制的结果文件） | 否 |
| `smoke/report.json` `.md` | L0 冒烟报告 | 否 |

`run_id` 规则：`<exp>--s<seed>--<UTC 时间戳>`。

## 怎么产出

```bash
python -m peerlab status              # 每个实验缺什么
python -m peerlab smoke               # L0 就绪检查，写 smoke/report.md
python -m peerlab run --exp fm-main-table --seed 0
python -m peerlab run-all             # 全部实验 × SEEDS=(0,1,2,3,4)
python -m peerlab report              # 重算 tables/ 与 summary.md
```

## 聚合规则

- 产物由 runner 按「运行前后差异」收集：只复制注册表里声明的 glob，单文件超过 20 MB 只记路径。
- 指标靠注册表的 dotted path 抽取；取不到就写 `null`，不猜。
- `identical=True` 表示 5 个种子数值完全一致。7 个确定性实验出现该标记是预期结果：它们的随机性来源被脚本写死，种子注入不会改变数字。
- 指标 spec 是从源码读出来的，尚未经过真实运行校准；首轮跑通后看 `tables/_schema_probe.md` 修正。

## 当前状态

没有任何有效运行。`configs/` 已按新基线重建；`data/processed/` 为空，上游网络与面板还没生成，10 个实验在就绪检查阶段全部受阻（0 就绪）。`smoke/report.md` 记录了逐实验的缺口。

推进顺序见 `spec/stages.yaml`：先网络与面板，再 pricing 样本与定价表。
