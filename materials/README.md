# PeerMAX 项目原始材料归档（2026-11-17）

本目录只收录**外部输入 / 立项输入**，即老师（及老师团队）提供的材料、组内/合作者提供的文献、本人撰写的立项文档。**不包含**项目自行下载的行情/财务数据，也不包含任何加工派生产物（面板、网络、MD&A 语料、回归结果等）。

本仓库随附 01/02/03 三个分卷共 **10 个文件**；`04_teacher_data/` 的 42 个文件（4.11 GiB）未随仓库分发。`MANIFEST.csv` 与 `SHA256SUMS.txt` 仍保留完整 52 条记录，校验时未收录项报 missing 属预期。

- 归档文件：**52 个**（另加本 README、`MANIFEST.csv`、`SHA256SUMS.txt`）
- 合计大小：**4.11 GiB**
- 完整性：52/52 源文件与副本 sha256 逐文件一致（`MANIFEST.csv` 的 `sha256_verified` 列全为 `YES`）
- 源文件仍保留在原路径，本目录是副本，可随时删除重建

## 1. 目录结构

```text
仓库根/materials/
├── 01_proposal/                     本人撰写的立项/研究计划（5 个）
├── 02_teacher_note_and_papers/      老师团队提供的方法说明与参考论文（3 个）
├── 03_team_papers/                  组内/合作者提供的论文（2 个）
├── 04_teacher_data/                 老师提供的数据（42 个，本仓库未收录）
│   ├── analyst_forecast/            预测情况（CNRDS 分析师预测）
│   ├── cfnd/                        CFND 网络新闻量化统计
│   ├── wsvi/                        WSVI 上市公司网络搜索指数
│   ├── mda/                         老师 MD&A 章节语料
│   ├── industry_classification/     申万 2021 版分类快照
│   └── drive/                       Google Drive `data sharing` 全部 36 个文件
├── MANIFEST.csv                     逐文件清单（路径/字节/sha256/来源/说明）
└── SHA256SUMS.txt                   校验和（可直接 sha256sum -c）
```

## 2. 老师提供的材料

### 2.1 方法说明与参考论文（`02_teacher_note_and_papers/`）

| 文件 | 大小 | 说明 |
|---|---|---|
| `Methodology Description.pdf` | 110.2 KiB | 2 页英文，M-B「公司层面文本竞争网络」方法说明；项目台账记为 M-B 规格来源 |
| `Shared Analyst Coverage Unifying Momentum Spillover Effects.pdf` | 1.8 MiB | Ali & Hirshleifer 主参考论文，64 页（S3A 分析师共同覆盖网络） |
| `A股产品市场竞争.pdf` | 1.0 MiB | 《经济学（季刊）》2021 年第 21 卷第 6 期，中文产品市场竞争/TNIC 先例，22 页 |

来源判定：三份均为 2026-08-10 批次微信接收（发送者 `llangli`，文件名与字节和聊天截图一致）；项目记忆中记为「老师 2026-08-10 提供新文献/数据」。因此标注为「老师团队提供」而非「老师本人直接发送」。

### 2.2 数据包（`04_teacher_data/`）

| 文件 | 大小 | 说明 |
|---|---|---|
| `analyst_forecast/预测情况(1).7z` | 159.5 MiB | CNRDS 分析师预测原始包，2026-05-27 打包；内含 6 个 xlsx + 字段说明 + 版权页，共 8 个文件 |
| `analyst_forecast/预测情况_字段说明.txt` | 1.1 KiB | 上述压缩包内字段说明的解出副本（内容一致，便于检索；13 个字段：Scode/Anndats/Fpenddt/A_name/A_code/Star/B_name/B_code/Feps/Fpe/Fnetpro/Foperev/Anndats_act） |
| `cfnd/CFND_网络新闻量化统计（按自然日）_ALL.zip` | 120.9 MiB | CFND 网络新闻量化统计，2001–2025 按自然日；77 个 entry（年度 xlsx + 每年字段说明 + 版权页） |
| `wsvi/WSVI_上市公司网络搜索指数_全国_ALL.zip` | 488.8 MiB | WSVI 上市公司网络搜索指数（全国），2011–2025；76 个 entry（年度 xlsx 分片 + 字段说明 + 版权页） |
| `mda/mda_data_2023.csv` | 1005.5 MiB | 老师 MD&A 章节定向导出：4 列（`f_id,f_stock_id,f_year,f_chapter_content`），33,315 行 / 5,291 股 / 2014–2022；**GBK 编码**，非 UTF-8 |
| `industry_classification/all_stkcds.xlsx` | 165.1 KiB | 申万 2021 版分类快照，5,800 只（31 一级 / 131 二级 / 337 三级）；2026-09-15 微信接收 |

### 2.3 老师 Google Drive（`04_teacher_data/drive/`）

36 个文件，共 2.37 GiB，均来自老师共享的 Google Drive `data sharing` 文件夹：

- **日频行情/交易**：未复权收盘、后复权收盘、市值、TTM BM、换手率、成交额、成交笔数、涨跌停 flag、停牌 flag、ST flag、新股 flag
- **月频基本面**：ROE/ROA/ROIC/EP/毛利/扣非净利/营收/总资产/销货成本等（快照 20260630，营收另有一份 20251231）
- **年度基本面**：营收/净利/总资产/权益/毛利
- **指数与基准**：沪深 300 / 中证 500 成分、中证 800 收盘与 mask、中证 500 月频 mask
- **其他**：`ind_code.pkl`（30 类行业代码）、`shibor_rates.pkl`（8 期限）、`monthly_returns_adj.csv` / `monthly_returns_adj_R.csv`（老师预计算的月频复权/原始收益）

来源证据：`reports/data_gates/G0/drive_batch1_report.md`、`drive_batch2_report.md`、`drive_batch3_report.md`、`drive_batch4_report.md` 均记录「数据源：Google Drive `data sharing`（老师提供）」。

## 3. 立项文档（`01_proposal/`）

**注意：这三份 proposal 的作者是 Yangyi Liu（本人），不是老师撰写。** 收入本目录是因为它们是项目立项/研究设计输入，而不是老师下发的文件。

| 文件 | 页数 | 日期 | 说明 |
|---|---|---|---|
| `peer_lottery_proposal.pdf` | 13 | 2026-04-09 | 早期短版：Individual-Investor Demand Spillovers Across Product-Market Peers |
| `peer_Proposal_PeerMAX.pdf` | 14 | 2026-07-13 | 完整 written proposal：When Peer MAX Spills Over |
| `peer_Proposal_PeerMAX_slides.pdf` | 29 | 2026-07-13 | 上述 written 版的汇报 slides |
| `Proposal_PeerMAX_中文版.pdf` | 10 | 2026-08-05 编译 | 中文版研究计划 |
| `Proposal_PeerMAX_中文版.tex` | — | 2026-08-05 | 中文版 LaTeX 源文件 |

PDF 元数据已核：作者 `Yangyi Liu`，创建日期分别为 2026-04-09 / 2026-07-13 / 2026-08-05。三份英文 proposal 的版本关系见 `docs/project/proposal_comparison.md`。

## 4. 组内/合作者提供（`03_team_papers/`）

| 文件 | 大小 | 说明 |
|---|---|---|
| `Long_2025_net_peer_momentum_China.pdf` | 718.5 KiB | Long et al. (2025) Net Peer Momentum，14 页；2026-08-29 微信接收（发送者 Selina） |
| `Jiang_Danling_et_al_2023_A股关联公司动量溢出_分析师共同覆盖.pdf` | 495.7 KiB | 蒋丹凌等（2023）《系统工程理论与实践》，19 页；2026-08-29 微信接收（发送者 llangli） |

这两份不是老师数据包的一部分，单独放置以免与老师材料混淆。

## 5. 聊天截图里存在、但本机没有落盘的文件

| 文件 | 截图大小 | 发送者 | 状态 |
|---|---|---|---|
| `similarity_matrix.csv` | 1.3 KiB | 24 刘致远 金工 CUHKSZ | **本机未找到**（项目目录、Desktop、Downloads、Documents 均无） |
| `extreme_analyst_days.csv` | 21.4 KiB | 24 刘致远 金工 CUHKSZ | **本机未找到**（同上） |

这两份是同学提供的补充文件，不是老师原始数据；如需纳入，需要重新从聊天中下载后补进 `03_team_papers/` 或单列目录。

## 6. 明确排除的内容

以下均属「自行下载」或「自行加工」，不属于本归档范围：

- `data/raw/baostock|tencent|eastmoney|cninfo|cninfo_prospectus|sina|sw2|lexicon|institutional/`：自行抓取的行情、公告、年报、申万分类、搜狗词库、ST/龙虎榜等；
- `data/raw/analyst/forecast_raw_all.csv`、`step1_acceptance.json`、`step2_audit.json`：对老师预测包的自行解析/验收产物；
- `data/processed/`：全部派生面板、网络、样本、收益文件（含 `monthly_returns_clean_v2.csv`、`monthly_returns_adj_full_2006_2025.csv`、`monthly_pricing_panel_*`、`network_*`、M1 parquet 等）；
- `deliverables/mda_20260912/`：用 CNINFO 年报自建的 MD&A 语料（1.04 GB zip）；
- `docs/literature/` 下 2026-10-05 批次：Hoberg-Phillips 等 7 篇方法文献与 3 份 readme，均为自行下载；
- `artifacts/external_benchmarks/tnic/`：HP TNIC-3 / FIC 公开数据，自行下载；
- 根目录 `Shared Analyst Coverage Unifying Momentum Spillover Effects.txt`：从 PDF 抽出的文本，自行加工；
- `data/external/sw_industry_all_stkcds_20260915.xlsx`：与 `all_stkcds.xlsx` 字节完全一致的副本（sha256 `11306982…1116f`），不重复收录。

## 7. 使用注意

1. `预测情况(1).7z` 扩展名是 `.7z`，但 7-Zip 识别为 **ZIP 容器**（`Type = zip`），解压时按 zip 处理即可。
2. `mda_data_2023.csv` 是 **GBK** 编码，直接按 UTF-8 读会乱码；列名为 `f_id,f_stock_id,f_year,f_chapter_content`。
3. Drive 数据是**时点快照**：大部分为 20260630，`monthly_operating_revenue_mrq_0_20251231.pkl` 为 20251231；不是当前最新数据，使用时须记录快照日期。
4. `all_stkcds.xlsx` 是**无日期快照**（申万 2021 版分类），项目已评估：第三源与现行分类在市一致率 98.68%，处置为「不替换现行分类」，仅作交叉核验证据（见 `reports/audits/sw2_third_source_recheck_20260915.md`）。
5. `预测情况` 的字段说明与 `版权声明与数据来源说明.pdf` 保留在压缩包内，未单独解出（字段说明另有一份 `.txt` 副本）。
6. 本目录内文件均为**只读归档用途**，不要在此目录上跑加工脚本；加工请在项目原路径进行。

## 8. 校验方法

```bash
cd materials
shasum -a 256 -c SHA256SUMS.txt     # 10 条 OK；未收录的 42 条报 missing，属预期
```

逐文件来源、字节数、sha256、源路径、接收批次见 `MANIFEST.csv`（UTF-8 with BOM，Excel 可直接打开）。
