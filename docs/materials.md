# PeerMAX 外部输入材料（2026-11-17 批次）

收录范围：老师（及老师团队）提供的材料、组内/合作者提供的文献、本人撰写的立项文档，以及老师提供的数据包。项目自行下载的行情/财务数据与加工派生产物不在其中（见 §6）。

52 个文件分两处存放：

- **10 份文档在 `ref/`**，随仓库入库。
- **42 个数据文件在 `data/raw/` 与 `data/external/`**，合计 4.11 GiB，不入 Git。
- 数据文件由交付包 `04_teacher_data.zip` 解出，搬运时逐个计算 sha256 并与交付包清单比对，42/42 一致后才删除原目录。

## 1. 目录结构

```text
仓库根/
├── ref/
│   ├── 01_proposal/                     本人撰写的立项/研究计划（5 个）
│   ├── 02_teacher_note_and_papers/      老师团队提供的方法说明与参考论文（3 个）
│   └── 03_team_papers/                  组内/合作者提供的论文（2 个）
└── data/
    ├── raw/analyst/                     预测情况（CNRDS 分析师预测）
    ├── raw/cfnd/                        CFND 网络新闻量化统计
    ├── raw/wsvi/                        WSVI 上市公司网络搜索指数
    ├── raw/drive/                       Google Drive `data sharing` 全部 36 个文件
    ├── raw/teacher/mda_data_2023.csv    老师 MD&A 章节语料
    └── external/sw_industry_all_stkcds_20260915.xlsx   申万 2021 版分类快照
```

## 2. 老师提供的材料

### 2.1 方法说明与参考论文（`ref/02_teacher_note_and_papers/`）

| 文件 | 大小 | 说明 |
|---|---|---|
| `Methodology Description.pdf` | 110.2 KiB | 2 页英文，M-B「公司层面文本竞争网络」方法说明；项目台账记为 M-B 规格来源 |
| `Shared Analyst Coverage Unifying Momentum Spillover Effects.pdf` | 1.8 MiB | Ali & Hirshleifer 主参考论文，64 页（S3A 分析师共同覆盖网络） |
| `A股产品市场竞争.pdf` | 1.0 MiB | 《经济学（季刊）》2021 年第 21 卷第 6 期，中文产品市场竞争/TNIC 先例，22 页 |

来源判定：三份均为 2026-08-10 批次微信接收（发送者 `llangli`，文件名与字节和聊天截图一致）；项目记忆中记为「老师 2026-08-10 提供新文献/数据」。因此标注为「老师团队提供」而非「老师本人直接发送」。

### 2.2 数据包（`data/raw/` 与 `data/external/`）

| 文件 | 大小 | 说明 |
|---|---|---|
| `data/raw/analyst/预测情况(1).7z` | 159.5 MiB | CNRDS 分析师预测原始包，2026-05-27 打包；内含 6 个 xlsx + 字段说明 + 版权页，共 8 个文件 |
| `data/raw/analyst/预测情况_字段说明.txt` | 1.1 KiB | 上述压缩包内字段说明的解出副本（内容一致，便于检索；13 个字段：Scode/Anndats/Fpenddt/A_name/A_code/Star/B_name/B_code/Feps/Fpe/Fnetpro/Foperev/Anndats_act） |
| `data/raw/cfnd/CFND_网络新闻量化统计（按自然日）_ALL.zip` | 120.9 MiB | CFND 网络新闻量化统计，2001–2025 按自然日；77 个 entry（年度 xlsx + 每年字段说明 + 版权页） |
| `data/raw/wsvi/WSVI_上市公司网络搜索指数_全国_ALL.zip` | 488.8 MiB | WSVI 上市公司网络搜索指数（全国），2011–2025；76 个 entry（年度 xlsx 分片 + 字段说明 + 版权页） |
| `data/raw/teacher/mda_data_2023.csv` | 1005.5 MiB | 老师 MD&A 章节定向导出：4 列（`f_id,f_stock_id,f_year,f_chapter_content`），33,315 行 / 5,291 股 / 2014–2022；**UTF-8 编码** |
| `data/external/sw_industry_all_stkcds_20260915.xlsx` | 165.1 KiB | 申万 2021 版分类快照，5,800 只（31 一级 / 131 二级 / 337 三级）；2026-09-15 微信接收；原名 `all_stkcds.xlsx`，代码按 `data/external/sw_industry_all_stkcds_20260915.xlsx` 读取（`s3_sw2_third_source_recheck.py:42`） |

### 2.3 老师 Google Drive（`data/raw/drive/`）

36 个文件，共 2.37 GiB，均来自老师共享的 Google Drive `data sharing` 文件夹：

- **日频行情/交易**：未复权收盘、后复权收盘、市值、TTM BM、换手率、成交额、成交笔数、涨跌停 flag、停牌 flag、ST flag、新股 flag
- **月频基本面**：ROE/ROA/ROIC/EP/毛利/扣非净利/营收/总资产/销货成本等（快照 20260630，营收另有一份 20251231）
- **年度基本面**：营收/净利/总资产/权益/毛利
- **指数与基准**：沪深 300 / 中证 500 成分、中证 800 收盘与 mask、中证 500 月频 mask
- **其他**：`ind_code.pkl`（30 类行业代码）、`shibor_rates.pkl`（8 期限）、`monthly_returns_adj.csv` / `monthly_returns_adj_R.csv`（老师预计算的月频复权/原始收益）

来源证据（主仓库）：`reports/data_gates/G0/drive_batch1_report.md` 至 `drive_batch4_report.md` 均记录「数据源：Google Drive `data sharing`（老师提供）」。

## 3. 立项文档（`ref/01_proposal/`）

**注意：这几份 proposal 的作者是 Yangyi Liu（本人），不是老师撰写。** 收入本仓库是因为它们是项目立项/研究设计输入。

| 文件 | 页数 | 日期 | 说明 |
|---|---|---|---|
| `peer_lottery_proposal.pdf` | 13 | 2026-04-09 | 早期短版：Individual-Investor Demand Spillovers Across Product-Market Peers |
| `peer_Proposal_PeerMAX.pdf` | 14 | 2026-07-13 | 完整 written proposal：When Peer MAX Spills Over |
| `peer_Proposal_PeerMAX_slides.pdf` | 29 | 2026-07-13 | 上述 written 版的汇报 slides |
| `Proposal_PeerMAX_中文版.pdf` | 10 | 2026-08-05 编译 | 中文版研究计划 |
| `Proposal_PeerMAX_中文版.tex` | — | 2026-08-05 | 中文版 LaTeX 源文件 |

PDF 元数据已核：作者 `Yangyi Liu`，创建日期分别为 2026-04-09 / 2026-07-13 / 2026-08-05。

## 4. 组内/合作者提供（`ref/03_team_papers/`）

| 文件 | 大小 | 说明 |
|---|---|---|
| `Long_2025_net_peer_momentum_China.pdf` | 718.5 KiB | Long et al. (2025) Net Peer Momentum，14 页；2026-08-29 微信接收（发送者 Selina） |
| `Jiang_Danling_et_al_2023_A股关联公司动量溢出_分析师共同覆盖.pdf` | 495.7 KiB | 蒋丹凌等（2023）《系统工程理论与实践》，19 页；2026-08-29 微信接收（发送者 llangli） |

这两份不是老师数据包的一部分，单独放置以免与老师材料混淆。

## 5. 聊天截图里存在、但本机没有落盘的文件

| 文件 | 截图大小 | 发送者 | 状态 |
|---|---|---|---|
| `similarity_matrix.csv` | 1.3 KiB | 24 刘致远 金工 CUHKSZ | **本机未找到** |
| `extreme_analyst_days.csv` | 21.4 KiB | 24 刘致远 金工 CUHKSZ | **本机未找到** |

这两份是同学提供的补充文件，不是老师原始数据；如需纳入，需要重新从聊天中下载。

## 6. 不在本仓库的内容

以下均属「自行下载」或「自行加工」，不属于本批外部输入：

- `data/raw/{baostock,tencent,eastmoney,cninfo,cninfo_prospectus,sina,sw2,lexicon,institutional}/`：自行抓取的行情、公告、年报、申万分类、搜狗词库、ST/龙虎榜等；
- `data/raw/analyst/forecast_raw_all.csv`、`step1_acceptance.json`、`step2_audit.json`：对老师预测包的自行解析/验收产物；
- `data/processed/`：全部派生面板、网络、样本、收益文件；
- `deliverables/mda_20260912/`：用 CNINFO 年报自建的 MD&A 语料；
- `artifacts/external_benchmarks/tnic/`：HP TNIC-3 / FIC 公开数据；
- 从 `Shared Analyst Coverage…pdf` 抽出的纯文本：自行加工产物。

## 7. 使用注意

1. `预测情况(1).7z` 扩展名是 `.7z`，但 7-Zip 识别为 **ZIP 容器**（`Type = zip`），解压时按 zip 处理即可；`s3a_step1_ingest.py` 就是按 zip 直读的。
2. `mda_data_2023.csv` 是 **UTF-8** 编码；列名为 `f_id,f_stock_id,f_year,f_chapter_content`。
3. Drive 数据是**时点快照**：大部分为 20260630，`monthly_operating_revenue_mrq_0_20251231.pkl` 为 20251231；不是最新数据，使用时须记录快照日期。
4. `sw_industry_all_stkcds_20260915.xlsx` 是**无日期快照**（申万 2021 版分类）。项目已评估：第三源与现行分类在市一致率 98.68%，处置为「不替换现行分类」，仅作交叉核验证据。
5. `预测情况` 的字段说明与 `版权声明与数据来源说明.pdf` 保留在压缩包内，未单独解出（字段说明另有一份 `.txt` 副本）。
6. `ref/` 与 `data/` 内的这些文件都按**只读归档**使用，不要在上面跑加工脚本。

## 8. 完整性

本仓库不保存逐文件清单与校验和文件。搬运时的核对：42 个数据文件从交付包解出后逐个计算 sha256，与交付包自带的清单逐条比对，全部一致才删除原目录；10 份文档用 `diff -r` 与交付包对比，逐字节一致。

需要重新核对时以交付压缩包为准：`04_teacher_data.zip`（1.7 GiB，位于 QQ 下载目录）内含全部 52 个文件。