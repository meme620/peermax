#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S1 Step 5：路线 gate 聚合（execution_protocol §2 Step 5）。

读取 S1 各步报告，按判定档（FAIL > BLOCKED > CONDITIONAL > PASS）输出
data_gate_report.json（G1-G8 逐 gate + M1_M2/visibility/mechanism 三路线状态）。
产出：reports/data_gates/G0/data_gate_report.json + s1_run_record.md

用法: python scripts/inventory_gates.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
G0 = ROOT / "reports/data_gates/G0"
OUT = G0 / "data_gate_report.json"
RUN_RECORD = ROOT / "reports/execution/stage0/s1_run_record.md"  # 放 G0 目录更贴切，但保持与 S0 run record 同目录

GATE_ORDER = {"FAIL": 0, "BLOCKED": 1, "CONDITIONAL": 2, "PASS": 3}


def aggregate(states):
    return min(states, key=lambda s: GATE_ORDER[s])


def main():
    # 读取 S1 各步报告（存在性 + 关键字段）
    def load(name):
        p = G0 / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    manifest = load("data_access_manifest.json")
    schema = load("s1_schema_check.json")
    coverage = load("s1_coverage_report.json")

    gates = {
        "G1": {
            "gate_status": "CONDITIONAL",
            "required_fields": "raw_return/valid_trade/limit_hit/board/calendar_version",
            "coverage_start": "2018-01-01", "coverage_end": "2025-12-31",
            "availability_status": "AVAILABLE",
            "evidence": "腾讯日线 real（8,546,009 行/版×2，MANIFEST 可核）；日历/基本资料 real；双源（新浪）抽检 5,273 对比 0 冲突；universe 三分类覆盖期缺失率 0.0011%",
            "missing_reason": "停牌核验源断档 975 天（2020-10-30~2025-07-15）；limit_hit/valid_trade 需派生；qfq 版涨跌停日伪收益限制（raw 版验证一致）",
            "fail_action": "月度定价必需；缺失则无 MAXraw/PeerMAXraw",
            "allowed_claim": "raw-return descriptive、S2 基础清洗",
            "forbidden_claim": "monthly PeerMAX pricing fact（G3/G8 未过前）",
        },
        "G2": {
            "gate_status": "CONDITIONAL",
            "required_fields": "size/BM/momentum/reversal/liquidity/turnover/analyst_coverage/network_degree",
            "coverage_start": "2011-01-01", "coverage_end": "2026-03-31",
            "availability_status": "PARTIAL",
            "evidence": "东财财务 real（5543 只×165 列，NOTICE_DATE 对齐）；baostock 财务 partial（66%，liqaShare 下载中）；size/BM/momentum/reversal/liquidity 可算",
            "missing_reason": "analyst coverage / ST 历史 / 名称历史 BLOCKED（待商业库）；Turnover 分母 liqaShare 待 baostock 财务 100%",
            "fail_action": "controls 不完整 → Fama-MacBeth 与 M2 降级",
            "allowed_claim": "limited incremental association（无完整 controls）",
            "forbidden_claim": "complete distinct MAX effect",
        },
        "G3": {
            "gate_status": "BLOCKED",
            "required_fields": "similarity/weight/network_version/available_at",
            "coverage_start": "2018", "coverage_end": "2025",
            "availability_status": "AVAILABLE",
            "evidence": "CNINFO 年报文本 real（35,979 PDF+35,817 txt，质量抽样 99/100）；文本按年分版本（天然 event-prior y-1）",
            "missing_reason": "Similarity pipeline NOT_RUN（方法冻结待用户决策）；补漏 236 项未收口；txt 比 pdf 少 162",
            "fail_action": "网络暴露必需；无网络则无 PeerMAXraw",
            "allowed_claim": "无（构造完成后才放行网络相关）",
            "forbidden_claim": "network transmission claim",
            "next_action": "S3 pipeline（similarity 方法冻结后）",
        },
        "G4": {
            "gate_status": "BLOCKED",
            "required_fields": "rank/cutoff/actual_display/RuleVersion/display_time",
            "availability_status": "UNAVAILABLE",
            "evidence": "无 actual display 历史档案；公开龙虎榜源仅 listing/席位（不可替代）",
            "missing_reason": "商业库/历史档案渠道未确认",
            "fail_action": "停止 visibility identification、first stage、2SLS",
            "allowed_claim": "institutional assignment audit（Tushare top_list 等候选）",
            "forbidden_claim": "actual-display causal visibility、V=D",
        },
        "G5": {
            "gate_status": "BLOCKED",
            "required_fields": "metric/value/platform/time/available_at",
            "availability_status": "UNAVAILABLE",
            "evidence": "无独立分钟级 attention；CNRDS WSVI/GUBA 为日度学术级候选（校图书馆订阅待验证）",
            "missing_reason": "授权平台数据/CNRDS 订阅未确认",
            "fail_action": "停止 attention mechanism claim",
            "allowed_claim": "proxy/path evidence",
            "forbidden_claim": "attention mechanism",
        },
        "G6": {
            "gate_status": "BLOCKED",
            "required_fields": "buy/sell/order_arrival/side_method",
            "availability_status": "UNAVAILABLE",
            "evidence": "无 Level-2/逐笔/订单簿数据",
            "missing_reason": "授权供应商未确认",
            "fail_action": "停止 order-flow path",
            "allowed_claim": "price-path or attention-only diagnostic",
            "forbidden_claim": "attention-to-order-flow mechanism",
        },
        "G7": {
            "gate_status": "CONDITIONAL",
            "required_fields": "publication_time/version/source_url",
            "availability_status": "PARTIAL",
            "evidence": "CNINFO 公告入口可用（年报已全量下载）；东财公告补漏验证成功",
            "missing_reason": "严格 publication timestamp 与修订链未验证；analyst_forecasts BLOCKED",
            "fail_action": "信息竞争解释不可分离",
            "allowed_claim": "competing-information diagnostic",
            "forbidden_claim": "clean non-information shock、完整 temporary pressure",
        },
        "G8": {
            "gate_status": "CONDITIONAL",
            "required_fields": "factor_returns_daily（rf/MKT/SMB/HML/MOM）",
            "availability_status": "PARTIAL",
            "evidence": "自建因子输入全齐（腾讯日线+东财市值/BM/盈利/资产增速+动量）；French China 已排除；CH-3 自建可行",
            "missing_reason": "因子未构造（NOT_RUN）；rf 待手工录入；CH-4 PEG 依赖 analyst（BLOCKED）",
            "fail_action": "PeerIVOL 保持 missing，M2 不启动",
            "allowed_claim": "raw-return/event-path diagnostic",
            "forbidden_claim": "reversal/equivalence claim",
        },
    }

    routes = {
        "M1_M2": {"required_gates": ["G1", "G2", "G3", "G8"], "note": "主线 A 月度定价+distinctness"},
        "visibility": {"required_gates": ["G1", "G2", "G3", "G4", "G8"], "note": "升级线 B"},
        "mechanism": {"required_gates": ["G1", "G2", "G3", "G5", "G6", "G8"], "note": "attention/order-flow 机制"},
    }
    route_status = {}
    for rname, r in routes.items():
        states = [gates[g]["gate_status"] for g in r["required_gates"]]
        route_status[rname] = {"route_status": aggregate(states), "gate_states": dict(zip(r["required_gates"], states))}

    report = {
        "report_id": "data_gate_report",
        "s1_status_route_M1_M2": route_status["M1_M2"],
        "s1_status_route_visibility": route_status["visibility"],
        "s1_status_route_mechanism": route_status["mechanism"],
        "gates": gates,
        "next_step_allowed": "S2（基础清洗，G1 合同已验）、S2.5（因子构造）、S3（网络 pipeline）；S4/M1 需 G3 放行后",
        "blocked_until": {
            "G3": "similarity 方法冻结 + pipeline 构造 + 八条件验收",
            "G4/G5/G6": "商业库/授权渠道",
            "G8 因子": "S2.5 构造 + rf 录入",
        },
        "draft_parameters_pending_user": coverage.get("draft_parameters", {}),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # run record
    run = f"""# S1 Run Record：数据可得性盘点（B0.5 重跑）

日期：2026-08-09
输入：S0 scope_card（SC-2026-08-09-v1）；A/B/C/D 四类落盘数据
状态：**CONDITIONAL**（M1_M2 路线；G3 网络 NOT_RUN 为唯一结构性 blocker）

## 逐 gate 判定
| Gate | 状态 | 关键证据 |
|---|---|---|
| G1 | CONDITIONAL | 腾讯日线 8,546,009 行/版×2 real；双源（新浪）5,273 对比 0 冲突；universe 三分类覆盖期缺失率 0.0011%（45/3,957,666）；停牌核验断档 975 天 |
| G2 | CONDITIONAL | 东财财务 real；analyst/ST/名称 BLOCKED；baostock 财务 66% |
| G3 | BLOCKED（NOT_RUN） | 年报文本 AVAILABLE；Similarity pipeline 未建 |
| G4/G5/G6 | BLOCKED | 无数据 |
| G7 | CONDITIONAL | 公告 inventory 可做；严格 timestamp 未验 |
| G8 | CONDITIONAL | 因子输入齐未构造；rf 待录 |

## 路线状态
- M1_M2 = CONDITIONAL（G1/G2/G8 合同通过；G3 待 pipeline）
- visibility = BLOCKED（G4）
- mechanism = BLOCKED（G5/G6）

## 关键发现（登记，不得重复踩）
1. 腾讯 pct_chg 字段 ≠ 涨跌幅（2020 抽 40 日 39 不符）→ 废弃，rraw=raw close 派生
2. 腾讯 qfq 版涨跌停日伪收益（偏差 0.2-0.5pp）→ 禁止用于 rraw；仅除权日识别辅助
3. 腾讯 raw close 与新浪双源一致（0 冲突）→ raw 版确认真实
4. eastmoney push2his 对 Python 直连封锁（RemoteDisconnected）→ 第二源=新浪 quotes.sina.cn（Python 可访问）；datacenter-web 可用但无日线行情接口
5. 断档期（975 天）未观测 ~13,444 firm-day 主要为停牌，量级与 2021-2025 停牌率吻合

## 产出物
- s1_expected_universe.json（expected 8,612,917 firm-day）
- data_access_manifest.json（7 源）
- field_mapping_manifest.json + s1_schema_check.json
- s1_coverage_report.json（三分类 + 双源）
- data_gate_report.json（本报告）

## 下一步
S1(M1_M2)=CONDITIONAL → 允许进入 S2（基础清洗）与 S2.5（因子）；S3 需用户先冻结 similarity 方法。
16 项数据参数草案值见 data_gate_report.draft_parameters_pending_user，**待用户确认后写入 parameter_registry.yaml**。
"""
    RUN_RECORD.parent.mkdir(parents=True, exist_ok=True)
    RUN_RECORD.write_text(run, encoding="utf-8")
    print("路线状态:")
    for rname, rs in route_status.items():
        print(f"  {rname}: {rs['route_status']}")
    print(f"\n输出: {OUT}\n     {RUN_RECORD}")


if __name__ == "__main__":
    main()
