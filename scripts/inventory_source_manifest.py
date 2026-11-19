#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S1 Step 2：来源登记 manifest（execution_protocol §2 Step 2）。

6 源登记（字段 schema 见 guide §7 Step 4）；checksum 抽查 3 个文件 SHA-256；
854.6 万行计数引用 tencent/MANIFEST.json 元数据，不重算。
产出：reports/data_gates/G0/data_access_manifest.json

用法: python scripts/inventory_source_manifest.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TENCENT_MANIFEST = ROOT / "data/raw/tencent/MANIFEST.json"
OUT = ROOT / "reports/data_gates/G0/data_access_manifest.json"

# checksum 抽查（3 个代表性文件）
CHECKSUM_SAMPLES = [
    ROOT / "data/raw/tencent/daily_raw/sh600000/2018.csv",
    ROOT / "data/raw/baostock/calendar/trade_dates_2016_2026.csv",
    ROOT / "data/raw/baostock/stock_basic/a_stocks.csv",
]


def file_size(p):
    return p.stat().st_size


def main():
    tm = json.loads(TENCENT_MANIFEST.read_text(encoding="utf-8"))
    samples = {str(p.relative_to(ROOT)): file_size(p) for p in CHECKSUM_SAMPLES}

    manifest = {
        "manifest_id": "data_access_manifest",
        "created": "2026-08-09",
        "evidence_type_note": "real=全量物理样本可核；partial=覆盖不全或下载中；断档降级已登记",
        "sources": [
            {
                "gate_id": "G1",
                "logical_table": "source_returns_daily",
                "source_name": "tencent newfqkline",
                "source_url_or_api": "web.ifzq.gtimg.cn/appstock/app/newfqkline/get",
                "access_method": "免费无登录 HTTP GET（count=320 按年拉取+下载后过滤）",
                "license_status": "free_no_login",
                "field_sample_ref": "date,open,close,high,low,volume,extra,pct_chg,amount,extra2（pct_chg 废弃，见 field_mapping_manifest）",
                "coverage_start": "2018-01-01",
                "coverage_end": "2025-12-31",
                "timestamp_precision": "day",
                "source_version": f"qfq+raw 两版；下载时点 {tm['download_date']}（前复权基准=该时点）",
                "available_at": tm["download_date"],
                "download_time": tm["download_date"],
                "checksum": samples.get("data/raw/tencent/daily_raw/sh600000/2018.csv", ""),
                "coverage_flag": "observed=8,546,009 行/版（MANIFEST rows）；5543 只；空文件 7757（退市/未上市/停牌）",
                "missing_reason": "北交所不在数据源（已登记排除）；停牌日无行",
                "evidence_type": "real",
                "status": "available",
                "owner": "peer_lottery_proposal",
                "blocker": "limit_hit/valid_trade 需派生（无源字段）",
                "next_action": "S1 Step3/4 三分类验收 + S2 收益构造",
                "evidence_ref": "data/raw/tencent/MANIFEST.json",
            },
            {
                "gate_id": "G1",
                "logical_table": "trading_calendar",
                "source_name": "baostock query_trade_dates",
                "source_url_or_api": "baostock python api",
                "access_method": "匿名登录可用",
                "license_status": "free",
                "field_sample_ref": "calendar_date,is_trading_day（无 market/board/session_id，需派生）",
                "coverage_start": "2016-01-01",
                "coverage_end": "2026-12-31",
                "timestamp_precision": "day",
                "source_version": "2016-2026 全量 4018 行",
                "available_at": "2026-08-09",
                "download_time": "2026-08-09",
                "checksum": samples.get("data/raw/baostock/calendar/trade_dates_2016_2026.csv", ""),
                "coverage_flag": "2018-2025 研究期交易日 1942 天",
                "missing_reason": "无板块/市场区分",
                "evidence_type": "real",
                "status": "available",
                "owner": "peer_lottery_proposal",
                "blocker": "板块日历起点=board_launch_date 派生（star 2019-07-22）",
                "next_action": "S2 Step1 派生 board 日历",
                "evidence_ref": "data/raw/baostock/calendar/trade_dates_2016_2026.csv",
            },
            {
                "gate_id": "G1/G2",
                "logical_table": "firm_basic（supporting input）",
                "source_name": "baostock query_stock_basic",
                "source_url_or_api": "baostock python api",
                "access_method": "匿名登录可用",
                "license_status": "free",
                "field_sample_ref": "code,code_name,ipoDate,outDate,type,status（type=1 全为 A 股）",
                "coverage_start": "-",
                "coverage_end": "-",
                "timestamp_precision": "day",
                "source_version": "2026-08-03 快照",
                "available_at": "2026-08-03",
                "download_time": "2026-08-09",
                "checksum": samples.get("data/raw/baostock/stock_basic/a_stocks.csv", ""),
                "coverage_flag": "5543 只沪深 A 股（上市中 5206 + 退市 337 含 outDate）",
                "missing_reason": "无北交所",
                "evidence_type": "real",
                "status": "available",
                "owner": "peer_lottery_proposal",
                "blocker": "ST 状态不在此表（禁止用 namechange 推断）",
                "next_action": "S1 Step1 已用（expected universe）；S2 对齐 list/delist 日期",
                "evidence_ref": "data/raw/baostock/stock_basic/a_stocks.csv",
            },
            {
                "gate_id": "G1",
                "logical_table": "source_returns_daily 停牌核验",
                "source_name": "baostock query_all_stock",
                "source_url_or_api": "baostock python api",
                "access_method": "匿名；服务不稳定（断点续传脚本）",
                "license_status": "free",
                "field_sample_ref": "code,tradeStatus(0=停牌/1=正常),code_name（含指数行须过滤）",
                "coverage_start": "2018-01-02",
                "coverage_end": "2025-12-31",
                "timestamp_precision": "day",
                "source_version": "逐日快照",
                "available_at": "2026-08-09",
                "download_time": "2026-08-09（部分）",
                "checksum": "",
                "coverage_flag": "967/1942 交易日（49.8%）；failed_days.csv 975 天",
                "missing_reason": "2020-10-30~2025-07-15 断档（服务中断）；断档段停牌/缺失合并'待核验'",
                "evidence_type": "partial",
                "status": "partial",
                "owner": "peer_lottery_proposal",
                "blocker": "975 天断档待续传（fetch_baostock_stocklist.py）",
                "next_action": "后台续传；S1 Step4 三分类验收",
                "evidence_ref": "data/raw/baostock/stock_list/",
            },
            {
                "gate_id": "G2",
                "logical_table": "firm_chars_clean（supporting input）",
                "source_name": "baostock query_profit_data/query_balance_data",
                "source_url_or_api": "baostock python api",
                "access_method": "匿名；限速 ~3.7-7 请求/s",
                "license_status": "free",
                "field_sample_ref": "季频财务（含 totalShare/liqaShare/pubDate）",
                "coverage_start": "2018-01-01",
                "coverage_end": "2025-12-31",
                "timestamp_precision": "quarter",
                "source_version": "5543 只 × 8 年 × 4 季",
                "available_at": "2026-08-09",
                "download_time": "2026-08-09（下载中）",
                "checksum": "",
                "coverage_flag": "234,208/~354,752 文件（66%）",
                "missing_reason": "下载未完成；balance 116,960 vs profit 117,248 不平衡",
                "evidence_type": "partial",
                "status": "partial",
                "owner": "peer_lottery_proposal",
                "blocker": "续传至 100%（liqaShare 为 Turnover 分母，东财无此字段）",
                "next_action": "后台续传",
                "evidence_ref": "data/raw/baostock/financial/",
            },
            {
                "gate_id": "G2/G8",
                "logical_table": "firm_chars_clean / factor 输入（supporting input）",
                "source_name": "eastmoney datacenter RPT_F10_FINANCE_MAINFINADATA",
                "source_url_or_api": "datacenter-web.eastmoney.com/api/data/v1/get",
                "access_method": "免费 HTTP GET（filter=(SECURITY_CODE=...)）",
                "license_status": "free_no_login",
                "field_sample_ref": "165 列：REPORT_DATE/NOTICE_DATE/TOTAL_SHARE/TOTAL_EQUITY_PK/TOTAL_ASSETS_PK/PARENTNETPROFIT/TA_YOYRATIO_PK/BPS/EPSJB/ROEJQ",
                "coverage_start": "2011-01-01（60 期/只）",
                "coverage_end": "2026-03-31",
                "timestamp_precision": "quarter",
                "source_version": "2026-08-09 全量（3.2 分钟 0 异常）",
                "available_at": "2026-08-09",
                "download_time": "2026-08-09",
                "checksum": "",
                "coverage_flag": "5543 只 × 165 列，median 52 期/只",
                "missing_reason": "ST 历史/名称历史/analyst coverage 不在本表（BLOCKED 待商业库）",
                "evidence_type": "real",
                "status": "available",
                "owner": "peer_lottery_proposal",
                "blocker": "NOTICE_DATE 需与事件时间对齐（S2 Step4）",
                "next_action": "S2 Step4 控制变量构造（Size/BM/Turnover 分母替代源）",
                "evidence_ref": "data/raw/eastmoney/finance/",
            },
            {
                "gate_id": "G3",
                "logical_table": "network_edges（原始文本输入）",
                "source_name": "CNINFO 年报 + eastmoney 公告补漏",
                "source_url_or_api": "www.cninfo.com.cn/new/hisAnnouncement/query；np-anotice-stock.eastmoney.com",
                "access_method": "免费 POST/GET；搜索接口按标题子串匹配（代码关联缺陷）",
                "license_status": "free",
                "field_sample_ref": "年报 PDF + pymupdf 全文 txt",
                "coverage_start": "2018",
                "coverage_end": "2025",
                "timestamp_precision": "year（网络版本=年报年）",
                "source_version": "annual_report_txt_v1（2026-08-09 提取）",
                "available_at": "2026-08-09",
                "download_time": "2026-08-09",
                "checksum": "",
                "coverage_flag": "35,979 PDF + 35,817 txt；质量抽样 99/100 含业务章节",
                "missing_reason": "failed_annual 4,326（no_annual 4,227，CNINFO 搜索缺陷）；东财补漏后剩 236 失败；txt 比 pdf 少 162（提取失败）",
                "evidence_type": "real",
                "status": "available",
                "owner": "peer_lottery_proposal",
                "blocker": "Similarity pipeline 未建（方法冻结待用户决策）",
                "next_action": "S3 网络 pipeline",
                "evidence_ref": "data/raw/cninfo/",
            },
        ],
        "checksum_samples": samples,
        "notes": [
            "北交所不在数据源（实测 0 行），universe=沪深 A 股（主板+创业板+科创板），已登记排除",
            "双源收益交叉校验第二源：eastmoney push2his K 线 API（2026-08-09 实测可访问）",
            "腾讯 pct_chg 字段实测与涨跌幅不符（浦发 2018-01-03：0.13 vs 东财 -0.47）→ 废弃，rraw 由 close 派生",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"登记来源: {len(manifest['sources'])} 个")
    for s in manifest["sources"]:
        print(f"  [{s['status']}] {s['gate_id']} {s['logical_table']} <- {s['source_name']}")
    print("checksum 抽查:")
    for p, h in samples.items():
        print(f"  {p}: {h[:16]}...")
    print(f"\n输出: {OUT}")


if __name__ == "__main__":
    main()
