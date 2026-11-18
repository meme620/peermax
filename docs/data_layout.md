# 数据布局

代码以仓库根为基准读写数据，根由 `Path(__file__).resolve().parents[2]` 定在 `scripts/data_audit/<file>.py` 的上两级。`data/` 不入 Git，体积 4.2 GB。

## 结构

```text
data/
├── raw/                     原始输入，代码只读
│   ├── analyst/             CNRDS 分析师预测包（2 个文件）
│   ├── cfnd/                CFND 网络新闻量化统计包（1 个）
│   ├── wsvi/                WSVI 网络搜索指数包（1 个）
│   ├── teacher/             老师 MD&A 语料（1 个，1.05 GB）
│   ├── drive/               老师 Google Drive 快照（36 个）
│   └── {baostock,cninfo,cninfo_prospectus,eastmoney,institutional,lexicon,sina,sw2,tencent}/   待重抓
├── external/
│   └── sw_industry_all_stkcds_20260915.xlsx   申万 2021 版分类（第三源核验用）
└── processed/               派生面板、网络、样本（待跑链生成）
```

## 14 个 raw 目录的就绪状态

| 目录 | 内容 | 状态 | 来源 |
|---|---|---|---|
| `analyst` | CNRDS 分析师预测（7z 实为 zip） | 就绪 | 老师包 |
| `cfnd` | CFND 网络新闻量化统计 | 就绪 | 老师包（当前代码 0 引用） |
| `wsvi` | WSVI 上市公司网络搜索指数 | 就绪 | 老师包（当前代码 0 引用） |
| `teacher` | 老师 MD&A 语料 | 就绪 | 老师包 |
| `drive` | 日/月基本面、指数、shibor | 就绪 | 老师 Drive |
| `baostock` | 股票列表、复权因子、财务 | 空 | `fetch_baostock_*` |
| `cninfo` | 公告、年报 | 空 | `fetch_cninfo_announcements/annual` |
| `cninfo_prospectus` | 招股说明书 | 空 | `fetch_cninfo_prospectus` |
| `eastmoney` | 公告、年报、龙虎榜、F10 | 空 | `fetch_eastmoney_*` / `fetch_em_f10_business` |
| `institutional` | 机构持仓等 | 空 | 无对应 fetch 脚本，来源待确认 |
| `lexicon` | 搜狗词库、产品词表 | 空 | `fetch_sogou_lexicon` |
| `sina` | 行情补充 | 空 | 无对应 fetch 脚本，来源待确认 |
| `sw2` | 申万二级行业历史 | 空 | `fetch_sw2_industry` |
| `tencent` | 日行情 | 空 | `fetch_tencent_daily` |

## 从交付包到代码路径的映射

| 交付包内路径 | 仓库内路径 | 代码消费方 |
|---|---|---|
| `04_teacher_data/analyst_forecast/预测情况(1).7z` | `data/raw/analyst/` | `s3a_step1_ingest.py:19` |
| `04_teacher_data/analyst_forecast/预测情况_字段说明.txt` | `data/raw/analyst/` | 字段参照 |
| `04_teacher_data/mda/mda_data_2023.csv` | `data/raw/teacher/mda_data_2023.csv` | `s3_teacher_corpus_ingest.py:28` |
| `04_teacher_data/drive/*`（36 个） | `data/raw/drive/*` | 44 处引用 |
| `04_teacher_data/cfnd/*.zip` | `data/raw/cfnd/` | 当前无代码引用 |
| `04_teacher_data/wsvi/*.zip` | `data/raw/wsvi/` | 当前无代码引用 |
| `04_teacher_data/industry_classification/all_stkcds.xlsx` | `data/external/sw_industry_all_stkcds_20260915.xlsx` | `s3_sw2_third_source_recheck.py:42` |

## 重抓的代价

9 个空目录全部是项目自行抓取的数据，可以用 `scripts/data/fetch_*` 重新下载，但拿到的是当前时点的数据，与 2026-11-17 快照不同口径。重抓后跑出来的数字不可与既有结果直接比较，需要同时登记为口径变更。

`data/processed/` 的产物同样只能靠重跑生成，而重跑又需要 `configs/`（见 `docs/environment.md` 的 T3）。`reports/data_gates/G0/` 是管线写入的登记证据，由各阶段脚本自行创建目录。
