#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S4 Step 1：source-month MAX / argmax / 完整月 gate（execution_plan §5.6 Step 1；P1.1-01）。

输入：data/processed/daily_panel/{year}.csv（2006-2025）
构造：
  n_valid_days_{j,t} = volume>0 的有效交易日数（P1-01 标准；完整月唯一依据）
  n_rraw_days_{j,t}  = rraw 非空日数（旧口径，仅审计）
  n_max_days_{j,t}   = valid_trade=1 & rraw 非空日数（日面板无 valid_trade → 按 rraw 非空派生）
  MAXraw_{j,t} = max rraw over rraw 非空日；d* = argmax（tie rule 由 registry `tie_rule` 解析）
  R_{j,t} = prod(1+rraw) - 1 over rraw 非空日
  complete_month = 1{n_valid_days >= min_valid_days}（min_valid_days 从 registry 顶层参数读取）
  limit_hit 派生（草案口径，登记）：|rraw| >= 0.095（主板）/0.195（创业板/科创板）→ 1；
    ST 5% 无数据 → limit_status_unknown 分层（ST 数据到位后精确）
输出：data/processed/source_month_argmax_panel.csv
      reports/data_gates/G0/s4_step1_report.json
用法: python scripts/peermax_max_build.py [--min-valid-days N]

P1.1-01（2026-11-17，W1 bundle；D-01=A / D-02=A / D-11=A）：
  - `min_valid_days` 唯一来源 = `configs/parameter_registry.yaml` 顶层参数（15）；缺失/重复/非正整数
    → SystemExit(2)，不回退 0。
  - D-02=A：`n_valid_days < 15` 但 `MAXraw` 已观测时**保留** `MAXraw/argmax_date`，
    写 `complete_month=0`、`max_status='incomplete'`，由下游按 `complete_month` 过滤；保留审计行。
  - D-11=A：主表行集保持 734,232（只含至少 1 个 rraw 非空日的 source-month），不新增零交易日行；
    2006–2013 日面板 `volume` 全空白 → 这些已有审计行 `n_valid_days=0`、`complete_month=0`。
  - 新列一律**追加表尾**；既有 11 列顺序不变。schema 见 07 P1.1-01 卡。
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/processed/daily_panel"
OUT = ROOT / "data/processed/source_month_argmax_panel.csv"
OUT_JSON = ROOT / "reports/data_gates/G0/s4_step1_report.json"
ST_FLAG = ROOT / "data/raw/drive/stock_st_flag_20260630.pkl"
REGISTRY = ROOT / "configs/parameter_registry.yaml"
# limit_hit 阈值唯一来源 = registry 顶层 limit_thresholds（原为模块常量，2026-11-19 审核收口）
# 供应商涨跌停 flag 可得时优先用 flag（data/raw/drive/daily_limit_up_flag_*.pkl），阈值口径降为回退路径
LIMIT_THRESHOLDS_PARAM = "limit_thresholds"
LIMIT_HIT_FLAG = ROOT / "data/raw/drive/daily_limit_up_flag_20260630.pkl"

MIN_VALID_DAYS_PARAM = "min_valid_days"
SOURCE_MONTH_VERSION = "src_month_v2_20261116"
SOURCE_VERSION = "tencent_daily_panel_20260630"
FORMULA_ID = "MAXRAW_v1"

# 新列（追加表尾；顺序即 schema，见 07 P1.1-01 卡）
NEW_COLS = [
    "source_month_version", "n_rraw_days", "n_max_days", "complete_month",
    "max_status", "return_status", "source_month_status", "limit_hit_status",
    "headline_max_eligible", "coverage_flag", "missing_reason", "available_at",
    "source_version", "MAXraw_incl_all", "n_excluded_limit", "n_excluded_resume",
]
OLD_COLS = [
    "source_id", "month", "MAXraw", "argmax_date", "n_valid_days", "R",
    "tie_rule_version", "formula_id", "limit_hit_at_argmax", "qualification_status",
    "st_flag_at_argmax",
]
COLS = OLD_COLS + NEW_COLS

# ---------------------------------------------------------------- tie rule（Phase 1 / I-15）
# registry `tie_rule.headline_value` 是中文说明（当前“选取月内最早达到该最大值的交易日作为 d*_{j,t}”）；
# 只按已知关键词/id 映射，命中数 != 1（缺参数/重名/多义/无法识别）一律 fail-closed。
TIE_RULE_EARLIEST = "earliest"
TIE_RULE_LATEST = "latest"
TIE_RULE_RANDOM = "random"
TIE_RULE_PICKERS = {TIE_RULE_EARLIEST: min, TIE_RULE_LATEST: max}
_TIE_RULE_CN_KEYWORDS = (
    ("最早", TIE_RULE_EARLIEST),
    ("最晚", TIE_RULE_LATEST),
    ("最新", TIE_RULE_LATEST),
    ("随机", TIE_RULE_RANDOM),
)
_TIE_RULE_EN_KEYWORDS = (
    ("earliest", TIE_RULE_EARLIEST),
    ("latest", TIE_RULE_LATEST),
    ("random", TIE_RULE_RANDOM),
)


def _tie_rule_fail(msg):
    """tie rule / registry 解析失败：stderr + SystemExit(2)，不回退默认值。"""
    print(f"[TIE-RULE] {msg}", file=sys.stderr, flush=True)
    raise SystemExit(2)


def _read_registry_doc(registry_path=None):
    """只读 registry YAML（不做任何写回）。"""
    import yaml
    p = Path(registry_path) if registry_path is not None else REGISTRY
    if not p.is_file():
        _tie_rule_fail(f"registry 不存在: {p}")
    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception as exc:  # YAML 解析失败也 fail-closed
        _tie_rule_fail(f"registry 解析失败: {p}: {exc}")
    if not isinstance(doc, dict) or not isinstance(doc.get("parameters"), list):
        _tie_rule_fail(f"registry 结构非法（缺 parameters 列表）: {p}")
    return doc


def resolve_min_valid_days(registry_path=None):
    """registry 顶层 `min_valid_days` -> int（唯一来源；fail-closed，不回退 0/15）。

    规则（P1.1-01 D-01=A）：参数必须恰好 1 条、`headline_value` 为 int（bool 不算）、>=1。
    缺失 / 重复 / 非正整数 → SystemExit(2)。
    """
    doc = _read_registry_doc(registry_path)
    hits = [p for p in doc["parameters"]
            if isinstance(p, dict) and p.get("name") == MIN_VALID_DAYS_PARAM]
    if len(hits) != 1:
        _tie_rule_fail(
            f"registry 参数 {MIN_VALID_DAYS_PARAM} 命中 {len(hits)} 条 != 1；拒绝猜测/回退")
    if "headline_value" not in hits[0]:
        _tie_rule_fail(f"registry.{MIN_VALID_DAYS_PARAM} 缺 headline_value")
    val = hits[0]["headline_value"]
    if isinstance(val, bool) or not isinstance(val, int) or val < 1:
        _tie_rule_fail(
            f"registry.{MIN_VALID_DAYS_PARAM}.headline_value 非法（须正整数）: {val!r}")
    return val


def resolve_limit_thresholds(registry_path=None):
    """registry 顶层 limit_thresholds -> {main, main_st, chinext_star}（fail-closed）。

    三条键必须齐全、为 float 且 ∈ (0,1)；缺失/重复/越界 → SystemExit(2)，不回退模块常量。
    """
    doc = _read_registry_doc(registry_path)
    hits = [p for p in doc["parameters"]
            if isinstance(p, dict) and p.get("name") == LIMIT_THRESHOLDS_PARAM]
    if len(hits) != 1:
        _tie_rule_fail(
            f"registry 参数 {LIMIT_THRESHOLDS_PARAM} 命中 {len(hits)} 条 != 1；拒绝回退默认阈值")
    hv = hits[0].get("headline_value")
    if not isinstance(hv, dict):
        _tie_rule_fail(f"registry.{LIMIT_THRESHOLDS_PARAM}.headline_value 不是映射: {hv!r}")
    out = {}
    for k in ("main", "main_st", "chinext_star"):
        if k not in hv:
            _tie_rule_fail(f"registry.{LIMIT_THRESHOLDS_PARAM} 缺键 {k!r}")
        v = hv[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not (0 < float(v) < 1):
            _tie_rule_fail(f"registry.{LIMIT_THRESHOLDS_PARAM}.{k} 非法（须 (0,1) 内实数）: {v!r}")
        out[k] = float(v)
    return out


def parse_tie_rule_id(text):
    """tie_rule 文本 -> tie_rule_id；0 个或多个不同 id → SystemExit(2)。"""
    raw = str(text)
    low = raw.lower()
    found = set()
    for kw, rid in _TIE_RULE_CN_KEYWORDS:
        if kw in raw:
            found.add(rid)
    for kw, rid in _TIE_RULE_EN_KEYWORDS:
        if re.search(rf"(?<![a-z]){re.escape(kw)}(?![a-z])", low):
            found.add(rid)
    if not found:
        _tie_rule_fail(f"无法从 tie_rule 文本解析 tie_rule_id: {text!r}")
    if len(found) > 1:
        _tie_rule_fail(f"tie_rule 文本含多个冲突 id {sorted(found)}: {text!r}")
    return found.pop()


def resolve_tie_rule(registry_path=None):
    """registry tie_rule -> {id, source, headline_value, sensitivity_grid, allowed...}。"""
    doc = _read_registry_doc(registry_path)
    hits = [p for p in doc["parameters"]
            if isinstance(p, dict) and p.get("name") == "tie_rule"]
    if len(hits) != 1:
        _tie_rule_fail(f"registry 参数 tie_rule 命中 {len(hits)} 条 != 1；拒绝猜测/回退")
    entry = hits[0]
    explicit = entry.get("tie_rule_id")
    if explicit is not None:
        rid = str(explicit).strip().lower()
        if rid not in (TIE_RULE_EARLIEST, TIE_RULE_LATEST, TIE_RULE_RANDOM):
            _tie_rule_fail(f"registry.tie_rule.tie_rule_id 非法: {explicit!r}")
        source = "registry.tie_rule.tie_rule_id"
        headline = entry.get("headline_value")
    else:
        if "headline_value" not in entry:
            _tie_rule_fail("registry.tie_rule 缺 headline_value（且无 tie_rule_id）")
        headline = entry["headline_value"]
        rid = parse_tie_rule_id(headline)
        source = "registry.tie_rule.headline_value"
    if rid == TIE_RULE_RANDOM:
        _tie_rule_fail("tie_rule=random 需要 seed；Phase 1 未实现，拒绝运行")
    grid = entry.get("sensitivity_grid")
    allowed = None
    if isinstance(grid, list):
        allowed = [str(x).strip().lower() for x in grid]
        if rid not in allowed:
            _tie_rule_fail(
                f"解析出的 tie_rule_id={rid!r} 不在 sensitivity_grid={allowed!r}")
    return {"id": rid, "source": source, "headline_value": headline,
            "tie_rule_id_field": explicit, "sensitivity_grid": grid,
            "allowed_in_sensitivity_grid": allowed}


def load_tie_rule(registry_path=None):
    """registry -> tie_rule_id（earliest / latest）。"""
    return resolve_tie_rule(registry_path)["id"]


def pick_argmax(days, rule):
    """并列 argmax 的日期取值：earliest→min、latest→max；只影响并列，不改 MAX。"""
    pick = TIE_RULE_PICKERS.get(rule)
    if pick is None:
        _tie_rule_fail(
            f"tie_rule={rule!r} 无对应并列取值规则（支持 {sorted(TIE_RULE_PICKERS)}）")
    maxv = max(r for _, r in days)
    return pick(d for d, r in days if r == maxv)


def load_st_flags(path=ST_FLAG):
    """返回 (df, colmap)：DataFrame（index=日期, 列='000001.SZ'）与 code6->列名映射。"""
    path = Path(path)
    if not path.exists():
        return None, {}
    import pandas as pd
    d = pd.read_pickle(path)
    colmap = {c.split('.')[0]: c for c in d.columns}
    d.index = [ts.strftime('%Y-%m-%d') if hasattr(ts, 'strftime') else str(ts)[:10] for ts in d.index]
    return d, colmap


def board_of(code):
    if code.startswith("sh"):
        return "star" if code[2:].startswith(("688", "689")) else "main"
    return "chinext" if code[2:].startswith(("300", "301", "302")) else "main"


def _num_or_none(x):
    """CSV 空串 -> None；否则 float。"""
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    return float(s)


def _count_rows(path) -> int:
    with open(path, "rb") as f:
        return max(0, sum(1 for _ in f) - 1)


def _file_size(path):
    return path.stat().st_size


def _rel(path):
    """相对仓库根 POSIX 路径。"""
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def _input_anchors(panel_dir, registry_path=None):
    entries = []
    for f in sorted(Path(panel_dir).glob("*.csv")):
        entries.append({"input": f"daily_panel/{f.name}", "file": _rel(f),
                        "rows": _count_rows(f), "bytes": f.stat().st_size})
    reg = Path(registry_path) if registry_path is not None else REGISTRY
    if reg.is_file():
        entries.append({"input": "parameter_registry", "file": _rel(reg),
                        "rows": None, "bytes": reg.stat().st_size})
    st = Path(ST_FLAG)
    if st.is_file():
        entries.append({"input": "stock_st_flag", "file": _rel(st), "rows": None,
                        "bytes": st.stat().st_size})
    return entries


def select_eligible_days(days, policy):
    out = []
    for d, r, resume, limit in days:
        if r is None:
            continue
        if policy.get("limit_hit") == "exclude" and limit:
            continue
        if policy.get("resume") == "exclude" and resume:
            continue
        out.append((d, r))
    return out


def resolve_max_day_policy(registry_path=None) -> dict:
    doc = _read_registry_doc(registry_path)
    hits = [p for p in doc["parameters"]
            if isinstance(p, dict) and p.get("name") == "max_day_policy"]
    if len(hits) != 1:
        _tie_rule_fail(f"registry 参数 max_day_policy 命中 {len(hits)} 条 != 1")
    hv = hits[0].get("headline_value")
    if not isinstance(hv, dict):
        _tie_rule_fail(f"registry.max_day_policy 不是映射: {hv!r}")
    out = {}
    for k in ("limit_hit", "resume"):
        v = str(hv.get(k, "")).strip()
        if v not in ("exclude", "include"):
            _tie_rule_fail(f"registry.max_day_policy.{k} 非法: {v!r}")
        out[k] = v
    return out


def build_monthly(panel_dir):
    """(code, month) -> [(date, rraw|None, volume|None)]（保留全部日频行）。

    同时返回全市场交易日历 month -> max(date)。
    行集（D-11=A）：只保留至少 1 个 rraw 非空日的 (code, month)；不新增零交易日 skeleton 行。
    """
    recs = defaultdict(list)
    dates = set()
    volume_months = set()
    for yf in sorted(Path(panel_dir).glob("*.csv")):
        with open(yf, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                vol = _num_or_none(r.get("volume"))
                recs[r["code"]].append((r["date"], _num_or_none(r.get("rraw")), vol))
                dates.add(r["date"])
                if vol is not None and vol > 0:
                    volume_months.add(r["date"][:7])
    all_dates = sorted(dates)
    prev_of = {d: (all_dates[i - 1] if i else None) for i, d in enumerate(all_dates)}
    monthly = defaultdict(list)
    for code, rows in recs.items():
        rows.sort(key=lambda x: x[0])
        last = None
        for d, rraw, vol in rows:
            resume = last is not None and prev_of[d] != last
            monthly[(code, d[:7])].append((d, rraw, vol, resume))
            last = d
    calendar = {}
    for d in all_dates:
        m = d[:7]
        if m not in calendar or d > calendar[m]:
            calendar[m] = d
    return monthly, calendar, volume_months


def _return_status(n_valid):
    """与 monthly_returns_clean_v2 同规则：PASS nv>=15；CONDITIONAL 5–14；BLOCKED 0–4。"""
    if n_valid >= 15:
        return "PASS"
    if n_valid >= 5:
        return "CONDITIONAL"
    return "BLOCKED"


def _source_month_row(*, code, month, n_valid, n_rraw, n_max, R, maxv, argmax_d,
                      tie_rule, limit_hit, limit_status, st_val, min_valid_days,
                      available_at, incl_all=None, n_excl_limit=0,
                      n_excl_resume=0):
    """固定 schema 的 source-month 行（旧 11 列 + 追加 13 列；见 07 P1.1-01 卡）。"""
    complete = int(n_valid >= min_valid_days)
    if complete == 1 and n_max > 0:
        max_status = "complete"
    elif n_max > 0:
        max_status = "incomplete"
    else:
        max_status = "missing"
    kls = "known" if limit_hit in (0, 1) else "unknown"
    eligible = int(complete == 1 and n_max > 0
                   and limit_status in ("qualified_limit_hit", "qualified_non_limit"))
    if complete == 1:
        coverage_flag = "covered"
    elif n_max > 0:
        coverage_flag = "partial"
    else:
        coverage_flag = "not_covered"
    reasons = []
    if complete == 0:
        reasons.append("incomplete_month")
    if kls == "unknown":
        reasons.append("limit_status_unknown")
    if n_max == 0:
        reasons.append("no_valid_trade_days")
    return {
        "source_id": code,
        "month": month,
        "MAXraw": round(maxv, 8) if maxv is not None else "",
        "argmax_date": argmax_d or "",
        "n_valid_days": int(n_valid),
        "R": round(R, 8) if R is not None else "",
        "tie_rule_version": tie_rule,
        "formula_id": FORMULA_ID,
        "limit_hit_at_argmax": "" if limit_hit is None else int(limit_hit),
        "qualification_status": "" if limit_status is None else limit_status,
        "st_flag_at_argmax": "" if st_val is None else int(st_val),
        "source_month_version": SOURCE_MONTH_VERSION,
        "n_rraw_days": int(n_rraw),
        "n_max_days": int(n_max),
        "complete_month": complete,
        "max_status": max_status,
        "return_status": _return_status(n_valid),
        "source_month_status": "ok" if complete == 1 else "incomplete_month",
        "limit_hit_status": kls,
        "headline_max_eligible": eligible,
        "coverage_flag": coverage_flag,
        "missing_reason": ";".join(reasons),
        "available_at": available_at,
        "source_version": SOURCE_VERSION,
        "MAXraw_incl_all": round(incl_all, 8) if incl_all is not None else "",
        "n_excluded_limit": int(n_excl_limit),
        "n_excluded_resume": int(n_excl_resume),
    }


def run(panel_dir=PANEL, out=OUT, out_json=OUT_JSON, st_flag=ST_FLAG,
        min_valid_days=None, registry_path=None):
    """构造 source-month 面板。min_valid_days=None → 从 registry 顶层参数读取。"""
    if min_valid_days is None:
        min_valid_days = resolve_min_valid_days(registry_path)
    if isinstance(min_valid_days, bool) or not isinstance(min_valid_days, int) \
            or min_valid_days < 1:
        raise SystemExit("min_valid_days 必须 >= 1（不提供关闭 gate 的路径）")
    limits = resolve_limit_thresholds(registry_path)
    policy = resolve_max_day_policy(registry_path)
    print(f"max_day_policy: {policy}", flush=True)
    tie = resolve_tie_rule(registry_path)
    tie_rule = tie["id"]
    print(f"tie_rule: {tie_rule} (source={tie['source']}); "
          f"min_valid_days={min_valid_days}", flush=True)

    monthly, calendar, volume_months = build_monthly(panel_dir)
    unobservable = sorted(m for m in calendar if m not in volume_months)
    pre2014 = [m for m in unobservable if m < "2014-01"]
    if pre2014:
        raise SystemExit(
            "[STOP] 面板在 " + ",".join(pre2014[:6]) + ("…" if len(pre2014) > 6 else "")
            + f" 共 {len(pre2014)} 个月无 volume 观测：n_valid_days/complete_month 结构性不可观测；"
            "不得静默产出 complete_month=0 的样本。可观测起点登记为 2014-01，"
            "如需提前请先补 volume 源（见 spec/design.yaml 的 sample 段）。")
    rows_keys = [k for k, days in monthly.items() if any(d[1] is not None for d in days)]
    excluded_no_rraw = len(monthly) - len(rows_keys)
    print(f"source-month cells={len(monthly)}; with rraw={len(rows_keys)}; "
          f"no-rraw cells excluded (D-11 A)={excluded_no_rraw}", flush=True)

    st_df, st_cols = load_st_flags(st_flag)
    if st_df is None:
        raise SystemExit(f"[STOP] ST 面板缺失: {st_flag}；限价判定不得在无 ST 状态下降级")
    print(f"ST 面板: {'已载入' if st_df is not None else '缺失'} "
          f"({0 if st_df is None else st_df.shape[0]} 日 × {0 if st_df is None else st_df.shape[1]} 股)", flush=True)
    lf_df, lf_cols = load_st_flags(LIMIT_HIT_FLAG)
    lf_dn, lf_dn_cols = load_st_flags(LIMIT_HIT_FLAG.parent / "daily_limit_down_flag_20260630.pkl")

    def _flag(df, colmap, code6, d):
        if df is None or code6 not in colmap or d not in df.index:
            return None
        v = df.at[d, colmap[code6]]
        return None if v != v else bool(v)

    def _day_limit(code, d):
        code6 = code[2:]
        up = _flag(lf_df, lf_cols, code6, d)
        dn = _flag(lf_dn, lf_dn_cols, code6, d)
        if up is not None or dn is not None:
            return bool(up) or bool(dn)
        st_val = None
        if st_df is not None and code6 in st_cols and d in st_df.index:
            v = st_df.at[d, st_cols[code6]]
            if v == v:
                st_val = bool(v)
        thr = limits["chinext_star"] if board_of(code) in ("chinext", "star") else (
            limits["main_st"] if st_val else limits["main"])
        return False
    print(f"涨跌停 flag 面板: {'已载入' if lf_df is not None else '缺失（回退阈值口径）'} "
          f"({0 if lf_df is None else lf_df.shape[0]} 日 × {0 if lf_df is None else lf_df.shape[1]} 股)", flush=True)
    rows = []
    stats = {"limit_hit": 0, "limit_unknown_zone": 0, "months": 0, "firms": 0,
             "st_at_argmax": 0, "st_flag_covered": 0,
             "limit_hit_source_vendor": 0, "limit_hit_source_derived": 0}
    firms = set()
    for code, month in rows_keys:
        days = sorted(monthly[(code, month)], key=lambda x: x[0])
        rraw_days = [(d, r) for d, r, _, _ in days if r is not None]
        n_valid = sum(1 for _, _, v, _ in days if v is not None and v > 0)
        flagged = [(d, r, resume, _day_limit(code, d)) for d, r, _, resume in days]
        eligible = select_eligible_days(flagged, policy)
        n_rraw = len(rraw_days)
        n_max = len(eligible)
        n_excl_limit = sum(1 for _, r, _, lim in flagged if r is not None and lim)
        n_excl_resume = sum(1 for _, r, res, _ in flagged if r is not None and res)
        prod = 1.0
        for _, r in rraw_days:
            prod *= 1 + r
        R = prod - 1
        maxv = max(r for _, r in eligible) if eligible else None
        argmax_d = pick_argmax(eligible, tie_rule) if eligible else ""
        incl_all = max(r for _, r in rraw_days) if rraw_days else None
        rows.append(_source_month_row(
            code=code, month=month, n_valid=n_valid, n_rraw=n_rraw, n_max=n_max,
            R=R, maxv=maxv, argmax_d=argmax_d, tie_rule=tie_rule,
            limit_hit=None, limit_status=None, st_val=None,
            min_valid_days=min_valid_days,
            available_at=calendar.get(month, max(d for d, _, _, _ in days)),
            incl_all=incl_all, n_excl_limit=n_excl_limit,
            n_excl_resume=n_excl_resume))
        firms.add(code)

    # 第二遍：按 argmax 日期批量取 ST 状态与涨跌停 flag（flag 优先，无覆盖才回退阈值口径）
    st_cache = {}
    lf_cache = {}
    for r in rows:
        if not r["argmax_date"]:
            continue
        code, d_ = r["source_id"], r["argmax_date"]
        code6 = code[2:]
        st_val = None
        if st_df is not None and code6 in st_cols:
            if d_ not in st_cache:
                try:
                    row_s = st_df.loc[d_]
                except KeyError:
                    row_s = None
                st_cache[d_] = row_s
            row_s = st_cache[d_]
            if row_s is not None and st_cols[code6] in row_s.index:
                v = row_s[st_cols[code6]]
                if v is not None and v == v:      # 非 NaN
                    st_val = bool(v)
                    stats["st_flag_covered"] += 1
        board = board_of(code)
        if r["MAXraw"] == "":
            continue
        maxv = float(r["MAXraw"])
        unknown = False
        if st_val is True:
            stats["st_at_argmax"] += 1
        flag_val = None
        if lf_df is not None and code6 in lf_cols:
            if d_ not in lf_cache:
                try:
                    lf_cache[d_] = lf_df.loc[d_]
                except KeyError:
                    lf_cache[d_] = None
            row_f = lf_cache[d_]
            if row_f is not None and lf_cols[code6] in row_f.index:
                fv = row_f[lf_cols[code6]]
                if fv is not None and fv == fv:      # 非 NaN
                    flag_val = bool(fv)
        if flag_val is not None:
            lh = int(flag_val)
            lim_status = ("qualified_limit_hit" if lh == 1 else "qualified_non_limit")
            stats["limit_hit_source_vendor"] += 1
        else:
            stats["limit_hit_source_derived"] += 1
            if board in ("chinext", "star"):
                thr = limits["chinext_star"]
            elif st_val is True:
                thr = limits["main_st"]
            elif st_val is False:
                thr = limits["main"]
            else:                                  # 无 ST 覆盖
                thr = limits["main"]
                if limits["main_st"] <= maxv < limits["main"]:
                    unknown = True
            lh = 1 if maxv >= thr else 0
            lim_status = ("limit_status_unknown" if unknown else
                          ("qualified_limit_hit" if lh == 1 else "qualified_non_limit"))
        if unknown:
            stats["limit_unknown_zone"] += 1
        st_out = "" if st_val is None else int(st_val)
        # 用最终 limit/ST 重建该行（保持列序与统一 schema）
        r.update(_source_month_row(
            code=code, month=r["month"], n_valid=r["n_valid_days"],
            n_rraw=r["n_rraw_days"], n_max=r["n_max_days"], R=float(r["R"]),
            maxv=maxv, argmax_d=d_, tie_rule=tie_rule, limit_hit=lh,
            limit_status=lim_status, st_val=None if st_val is None else bool(st_val),
            min_valid_days=min_valid_days, available_at=r["available_at"],
            incl_all=(float(r["MAXraw_incl_all"]) if r["MAXraw_incl_all"] != "" else None),
            n_excl_limit=int(r["n_excluded_limit"]),
            n_excl_resume=int(r["n_excluded_resume"])))
        r["st_flag_at_argmax"] = st_out
        stats["limit_hit"] += lh

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)

    def _count(pred):
        return int(sum(1 for r in rows if pred(r)))

    win = [r for r in rows if "2015-01" <= r["month"] <= "2023-12"]
    old_incomplete_with_max = int(sum(
        1 for r in win if r["n_rraw_days"] < 15 and r["MAXraw"] != ""))
    two_def_diff_win = int(sum(
        1 for r in win if (r["n_valid_days"] >= 15) != (r["n_rraw_days"] >= 15)))
    two_def_diff_all = int(sum(
        1 for r in rows if (r["n_valid_days"] >= 15) != (r["n_rraw_days"] >= 15)))
    nv0_rows = _count(lambda r: r["n_valid_days"] == 0)
    pre2014_nv0 = _count(lambda r: r["n_valid_days"] == 0 and r["month"] < "2014-01")
    report = {
        "report_id": "s4_step1_source_month_max",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "script": _rel(Path(__file__)),
        "script_bytes": _file_size(Path(__file__)),
        "input_anchors": _input_anchors(panel_dir, registry_path),
        "status": "PASS",
        "source_month_obs": len(rows),
        "firms": len(firms),
        "limit_hit_argmax": stats["limit_hit"],
        "limit_hit_source_counts": {
            "vendor_flag": stats["limit_hit_source_vendor"],
            "derived_threshold": stats["limit_hit_source_derived"],
        },
        "limit_thresholds": limits,
        "limit_thresholds_source": "configs/parameter_registry.yaml:parameters[name=limit_thresholds].headline_value",
        "unobservable_volume_months": unobservable[:12],
        "limit_status_unknown": stats["limit_unknown_zone"],
        "st_at_argmax_true": stats["st_at_argmax"],
        "note": "limit_hit 阈值按 (板块, ST 状态) 判定（D4，2026-09-14）：主板非 ST 10% / 主板 ST 5% / 创业板科创板 20%；"
                "ST 标记来自 Drive stock_st_flag；无标记（退市/未覆盖）且落在 5%-10% 区间时标 limit_status_unknown；"
                f"tie_rule={tie_rule}（{tie['source']}）",
        "tie_rule": tie,
        "min_valid_days": int(min_valid_days),
        "min_valid_days_source": "configs/parameter_registry.yaml:parameters[name=min_valid_days].headline_value",
        "source_month_version": SOURCE_MONTH_VERSION,
        "complete_month_rows": _count(lambda r: r["complete_month"] == 1),
        "incomplete_month_rows": _count(lambda r: r["complete_month"] == 0),
        "incomplete_with_max_observed": _count(
            lambda r: r["complete_month"] == 0 and r["n_max_days"] > 0),
        "old_rraw_incomplete_with_max_observed": old_incomplete_with_max,
        "two_definition_complete_diff_rows": two_def_diff_win,
        "two_definition_complete_diff_rows_all_period": two_def_diff_all,
        "n_max_days_zero_rows": _count(lambda r: r["n_max_days"] == 0),
        "n_valid_days_zero_rows": nv0_rows,
        "n_valid_days_zero_rows_pre_2014": pre2014_nv0,
        "unavailable_volume_years": "2006–2013 daily_panel volume 全空白；2026-11-19 起该段改为 SystemExit(2) 停机，不再静默产出 complete_month=0",
        "excluded_no_rraw_cells": excluded_no_rraw,
        "n_max_days_le_n_rraw_days_violations": _count(
            lambda r: r["n_max_days"] > r["n_rraw_days"]),
        "source_month_status_counts": {
            k: _count(lambda r, k=k: r["source_month_status"] == k)
            for k in ("ok", "incomplete_month")},
        "max_status_counts": {
            k: _count(lambda r, k=k: r["max_status"] == k)
            for k in ("complete", "incomplete", "missing")},
        "return_status_counts": {
            k: _count(lambda r, k=k: r["return_status"] == k)
            for k in ("PASS", "CONDITIONAL", "BLOCKED")},
        "coverage_flag_counts": {
            k: _count(lambda r, k=k: r["coverage_flag"] == k)
            for k in ("covered", "partial", "not_covered")},
        "headline_max_eligible_rows": _count(lambda r: r["headline_max_eligible"] == 1),
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"source-month 观测 {len(rows):,}，{len(firms)} 只；"
          f"complete_month={report['complete_month_rows']:,} / incomplete={report['incomplete_month_rows']:,}")
    print(f"limit_hit argmax {stats['limit_hit']:,}，limit_status_unknown {stats['limit_unknown_zone']:,}")
    print(f"输出: {out}\n     {out_json}")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-valid-days", type=int, default=None,
                    help="显式覆盖 registry min_valid_days（仅测试/审计注入；缺省从 registry 读取）")
    ap.add_argument("--registry", default=None,
                    help="registry YAML（默认 configs/parameter_registry.yaml；仅测试注入用）")
    args = ap.parse_args(argv)
    run(min_valid_days=args.min_valid_days, registry_path=args.registry)


if __name__ == "__main__":
    main()
