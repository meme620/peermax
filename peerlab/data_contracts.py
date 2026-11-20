from __future__ import annotations

import csv
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "spec" / "data_contracts.yaml"
EVIDENCE_DIR = Path("reports") / "data_contracts" / "G0"
TITLES = {
    "a1_schema": "原始文件形状、dtype、索引与时间跨度",
    "a2_universe": "股票池对齐",
    "a3_pit": "快照与 PIT 边界",
    "a4_units": "量纲与取值范围",
    "a5_cliffs": "除权断崖残留",
    "a6_limit_flags": "涨跌停派生口径与供应商 flag 一致",
    "a7_return_series": "自建月收益与老师交付序列对拍",
    "a8_flag_coverage": "flag 面板覆盖率与取值域",
}


def load_spec(path=None) -> dict:
    p = Path(path) if path is not None else SPEC_PATH
    if not p.is_file():
        raise SystemExit(f"[STOP] data contract spec 不存在: {p}")
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or not isinstance(doc.get("contracts"), list):
        raise SystemExit(f"[STOP] data contract spec 结构非法: {p}")
    ids = [c.get("id") for c in doc["contracts"] if isinstance(c, dict)]
    if len(ids) != len(set(ids)) or set(ids) != set(TITLES):
        raise SystemExit(f"[STOP] data contract id 集合不匹配: {sorted(ids)}")
    return doc


def _spec_for(root: Path, spec) -> dict:
    if spec is not None:
        return spec
    return load_spec(Path(root) / "spec" / "data_contracts.yaml")


def _contract(spec: dict, cid: str) -> dict:
    for c in spec["contracts"]:
        if c.get("id") == cid:
            return c
    raise SystemExit(f"[STOP] spec 缺 contract {cid}")


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _outcome(problems, checked, blocked=None, observed=None) -> dict:
    out = {"problems": list(problems), "checked": int(checked), "blocked": blocked}
    if observed is not None:
        out["observed"] = observed
    return out


def _read_pickle(path: Path) -> pd.DataFrame:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, pd.DataFrame):
        raise ValueError(f"not a DataFrame: {type(obj).__name__}")
    return obj


def _admit(root: Path, rel: str):
    p = Path(root) / rel
    return p if p.is_file() else None


def check_a1_schema(root, spec=None) -> dict:
    cfg = _contract(_spec_for(root, spec), "a1_schema")
    problems, checked, missing = [], 0, []
    tol = cfg.get("tolerance", {})
    rows_tol = int(tol.get("rows", 0))
    cols_tol = int(tol.get("cols", 0))
    for item in cfg.get("files", []):
        p = _admit(root, item["path"])
        if p is None:
            missing.append(item["path"])
            continue
        df = _read_pickle(p)
        checked += 1
        name = _rel(root, p)
        if abs(len(df) - int(item["rows"])) > rows_tol:
            problems.append(f"{name}: rows={len(df)} expected={item['rows']}")
        if abs(df.shape[1] - int(item["cols"])) > cols_tol:
            problems.append(f"{name}: cols={df.shape[1]} expected={item['cols']}")
        if not isinstance(df.index, pd.DatetimeIndex):
            problems.append(f"{name}: index 不是 DatetimeIndex")
            continue
        if not df.index.is_monotonic_increasing:
            problems.append(f"{name}: index 非升序")
        if not df.index.is_unique:
            problems.append(f"{name}: index 有重复")
        if not df.columns.is_unique:
            problems.append(f"{name}: 列名有重复")
        start = df.index.min().strftime("%Y-%m-%d") if len(df) else ""
        end = df.index.max().strftime("%Y-%m-%d") if len(df) else ""
        if item.get("start") and start != item["start"]:
            problems.append(f"{name}: start={start} expected={item['start']}")
        if item.get("end") and end != item["end"]:
            problems.append(f"{name}: end={end} expected={item['end']}")
        kinds = cfg.get("rules", {}).get("dtype_kinds")
        if kinds:
            bad = sorted({str(t) for t in df.dtypes.astype(str).unique() if str(t)[0] not in kinds})
            if bad:
                problems.append(f"{name}: dtype 类型不在白名单 {bad}")
    for item in cfg.get("csvs", []):
        p = _admit(root, item["path"])
        if p is None:
            missing.append(item["path"])
            continue
        with open(p, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            n = sum(1 for _ in reader)
        checked += 1
        name = _rel(root, p)
        if header != list(item["columns"]):
            problems.append(f"{name}: 表头={header} expected={item['columns']}")
        if abs(n - int(item["rows"])) > rows_tol:
            problems.append(f"{name}: rows={n} expected={item['rows']}")
    if missing:
        return _outcome(problems, checked, "缺原始文件: " + ", ".join(missing))
    return _outcome(problems, checked)


def check_a2_universe(root, spec=None) -> dict:
    cfg = _contract(_spec_for(root, spec), "a2_universe")
    base_path = _admit(root, cfg["base"]["path"])
    if base_path is None:
        return _outcome([], 0, "缺基准矩阵: " + cfg["base"]["path"])
    base = _read_pickle(base_path)
    base_codes = {str(c).split(".")[0] for c in base.columns}
    problems, checked, missing = [], 0, []
    for src in cfg.get("sources", []):
        p = _admit(root, src["path"])
        if p is None:
            missing.append(src["path"])
            continue
        if p.suffix == ".csv":
            d = pd.read_csv(p, dtype=str, usecols=[src["code_col"]], keep_default_na=False)
            codes = {str(x).zfill(6) for x in d[src["code_col"]] if str(x).strip()}
        else:
            codes = set(_read_pickle(p).index.astype(str))
        checked += 1
        ratio = len(codes & base_codes) / max(1, len(codes))
        if ratio < float(src["min_ratio"]):
            problems.append(
                f"{src['path']}: 覆盖率={ratio:.4f} < {src['min_ratio']}（{len(codes)} 只中 {len(codes & base_codes)} 在基准池）")
    if missing:
        return _outcome(problems, checked, "缺股票池来源: " + ", ".join(missing))
    return _outcome(problems, checked)


def check_a3_pit(root, spec=None) -> dict:
    cfg = _contract(_spec_for(root, spec), "a3_pit")
    problems, checked, missing = [], 0, []
    for rel, limit in cfg.get("max_index_date", {}).items():
        p = _admit(root, rel)
        if p is None:
            missing.append(rel)
            continue
        df = _read_pickle(p)
        checked += 1
        end = df.index.max().strftime("%Y-%m-%d")
        if end > str(limit):
            problems.append(f"{rel}: 末日期={end} 超出快照 {limit}")
    for rel, limit in cfg.get("max_month", {}).items():
        p = _admit(root, rel)
        if p is None:
            missing.append(rel)
            continue
        d = pd.read_csv(p, usecols=["month"], dtype=str)
        checked += 1
        end = d["month"].max()
        if end > str(limit):
            problems.append(f"{rel}: 末月={end} 超出快照 {limit}")
    for rel, limit in cfg.get("max_year", {}).items():
        p = _admit(root, rel)
        if p is None:
            missing.append(rel)
            continue
        d = pd.read_csv(p, usecols=["f_year"], dtype=str)
        checked += 1
        end = int(d["f_year"].astype(int).max())
        if end > int(limit):
            problems.append(f"{rel}: 末年={end} 超出快照 {limit}")
    if missing:
        return _outcome(problems, checked, "缺快照输入: " + ", ".join(missing))
    return _outcome(problems, checked)


def check_a4_units(root, spec=None) -> dict:
    cfg = _contract(_spec_for(root, spec), "a4_units")
    problems, checked, missing = [], 0, []
    for item in cfg.get("series", []):
        p = _admit(root, item["path"])
        if p is None:
            missing.append(item["path"])
            continue
        df = _read_pickle(p)
        arr = df.to_numpy(dtype="float64", na_value=np.nan)
        med = float(np.nanmedian(arr))
        checked += 1
        if "median_min" in item and med < float(item["median_min"]):
            problems.append(f"{item['path']}: median={med:.4g} < {item['median_min']}")
        if "median_max" in item and med > float(item["median_max"]):
            problems.append(f"{item['path']}: median={med:.4g} > {item['median_max']}")
        if "q999_max" in item:
            q = float(np.nanquantile(arr, 0.999))
            if q > float(item["q999_max"]):
                problems.append(f"{item['path']}: q999={q:.4g} > {item['q999_max']}")
    if missing:
        return _outcome(problems, checked, "缺量纲输入: " + ", ".join(missing))
    return _outcome(problems, checked)


_CLIFF_QUEUE_COLS = ["date", "order_book_id", "ret", "raw_ret", "category"]


def _cliff_rows(post_ret, raw_ret, thr) -> list:
    """后复权日收益 < thr 的格，逐格分类：genuine / adjusted_only / unclassified。

    全矩阵用 numpy 布尔索引一次取出，不逐格走 pandas 标签索引。旧实现用
    where(mask).stack(future_stack=True) 取格：pandas 2.3 的 stack 默认保留 NaN，
    漏 dropna 后会把 6178x5535 的整个网格（3419 万行）都堆出来写进队列，且分类把
    「未复权缺值」错记成「复权缺陷」。raw_ret 为 None 或该格缺值时类别是 unclassified。
    """
    P = post_ret.to_numpy()
    ii, jj = np.nonzero(P < thr)
    if not len(ii):
        return []
    dates = [d.strftime("%Y-%m-%d") for d in post_ret.index]
    names = post_ret.columns.to_numpy()
    R = None if raw_ret is None else raw_ret.to_numpy()
    rows = []
    for a, b in zip(ii.tolist(), jj.tolist()):
        v = float(P[a, b])
        if R is None:
            rv, cat = None, "unclassified"
        else:
            rv = float(R[a, b])
            if rv != rv:
                rv, cat = None, "unclassified"
            else:
                cat = "genuine" if rv < thr else "adjusted_only"
        rows.append({"date": dates[a], "order_book_id": names[b], "ret": v,
                     "raw_ret": rv, "category": cat})
    return rows


def check_a5_cliffs(root, spec=None) -> dict:
    cfg = _contract(_spec_for(root, spec), "a5_cliffs")
    thr = float(cfg["cliff_threshold"])
    post_path = _admit(root, cfg["post_adjusted"]["path"])
    if post_path is None:
        return _outcome([], 0, "缺复权价矩阵: " + cfg["post_adjusted"]["path"])
    post = _read_pickle(post_path)
    post_ret = post.pct_change(fill_method=None)
    cliff_mask = post_ret < thr
    n_post = int(cliff_mask.to_numpy().sum())
    queue_rel = cfg["post_adjusted"].get("review_queue")
    problems = []
    checked = 1
    raw_rel = cfg.get("raw", {}).get("path")
    raw = None
    if raw_rel:
        raw_path = _admit(root, raw_rel)
        if raw_path is not None:
            checked += 1
            raw = _read_pickle(raw_path)

    # 断崖判定的两个类别必须分开，否则真实行情会被当成复权缺陷：
    #   1) adjusted_only —— 复权**造成**的断崖：后复权日收益 < thr 而未复权同格 >= thr。
    #      这是复权链缺陷，任何一条都不放行。
    #   2) genuine —— 后复权与未复权**同为** < thr：真实行情（退市整理期、注册制新股无涨跌幅
    #      限制等），不是缺陷；只登记计数与清单，不参与 pass/fail。
    adjusted_only = []
    genuine = 0
    if raw is not None:
        raw_ret = raw.pct_change(fill_method=None)
        common = post_ret.columns.intersection(raw_ret.columns)
        common_idx = post_ret.index.intersection(raw_ret.index)
        pr = post_ret.loc[common_idx, common]
        rr = raw_ret.loc[common_idx, common]
        # NaN 既不算越界也不算未越界：缺值格单独排除，不能被判成"复权引入的断崖"
        both_present = pr.notna() & rr.notna()
        pm = (pr < thr) & both_present
        rm = (rr < thr) & both_present
        adjusted_only_mask = pm & ~rm & both_present
        genuine = int((pm & rm).to_numpy().sum())
        if int(adjusted_only_mask.to_numpy().sum()):
            stacked = pr.where(adjusted_only_mask).stack(future_stack=True).dropna()
            adjusted_only = [f"{i[0]:%Y-%m-%d}/{i[1]}={v:.4f}" for i, v in stacked.items()]
    else:
        # 无未复权序列可比时无法分类，退回总计数（保守：按缺陷处理）
        adjusted_only = [f"{i[0]:%Y-%m-%d}/{i[1]}={v:.4f}"
                         for i, v in post_ret.where(cliff_mask).stack(future_stack=True).dropna().items()]
        genuine = 0
    n_adjusted_only = len(adjusted_only)
    max_adjusted_only = int(cfg["post_adjusted"].get("max_adjusted_only", 0))
    if n_adjusted_only > max_adjusted_only:
        problems.append(
            f"{cfg['post_adjusted']['path']}: 复权引入的断崖 {n_adjusted_only} 条 > 容差 "
            f"{max_adjusted_only}（未复权同格不低于 {thr}）；样本：{adjusted_only[:5]}")
    observed = {"cliff_cells": n_post, "adjusted_only_cliffs": n_adjusted_only,
                "genuine_cliffs": genuine}
    if queue_rel:
        # 审查队列按类别落盘：adjusted_only 是要修的，genuine 是真实行情留痕
        queued = (_cliff_rows(pr, rr, thr) if raw is not None
                  else _cliff_rows(post_ret, None, thr))
        out_q = Path(root) / queue_rel
        out_q.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(queued, columns=_CLIFF_QUEUE_COLS).to_csv(out_q, index=False)
    if raw is None and n_post:
        problems.append(f"{cfg['post_adjusted']['path']}: 无法区分复权缺陷与真实行情（缺未复权序列），"
                        f"按缺陷处理 {n_post} 条")
    monthly_cfg = cfg.get("monthly", {})
    monthly_rel = monthly_cfg.get("path")
    if monthly_rel:
        m_path = _admit(root, monthly_rel)
        if m_path is None:
            return _outcome(problems, checked, "缺派生月收益: " + monthly_rel, observed)
        tol = float(monthly_cfg.get("tolerance", 1e-9))
        m = pd.read_csv(m_path, dtype={"firm_id": str, "month": str})
        close = post.resample("ME").last()
        close.index = [d.strftime("%Y-%m") for d in close.index]
        ret = close.pct_change(fill_method=None)
        long = ret.stack(future_stack=True).dropna().rename("R_ref").reset_index()
        long.columns = ["month", "col", "R_ref"]
        long["firm_id"] = long["col"].map(lambda c: c.split(".")[1].lower() + c.split(".")[0])
        j = m.merge(long[["firm_id", "month", "R_ref"]], on=["firm_id", "month"], how="inner")
        checked += 1
        if len(j) != len(m):
            problems.append(f"{monthly_rel}: 与后复权复算只有 {len(j)}/{len(m)} 行可对齐")
        if len(j):
            diff = (j["R"] - j["R_ref"]).abs()
            if float(diff.max()) > tol:
                problems.append(f"{monthly_rel}: 与后复权复算 max|diff|={float(diff.max()):.3e} > {tol}")
    return _outcome(problems, checked, observed=observed)


def check_a6_limit_flags(root, spec=None) -> dict:
    cfg = _contract(_spec_for(root, spec), "a6_limit_flags")
    paths = {k: _admit(root, cfg[k]) for k in ("price", "flag", "st")}
    missing = [cfg[k] for k, v in paths.items() if v is None]
    if missing:
        return _outcome([], 0, "缺涨跌停核验输入: " + ", ".join(missing))
    px = _read_pickle(paths["price"])
    flag = _read_pickle(paths["flag"])
    st = _read_pickle(paths["st"])
    idx = px.index.intersection(flag.index).intersection(st.index)
    idx = idx[(idx >= pd.Timestamp(cfg["start"])) & (idx <= pd.Timestamp(cfg["end"]))]
    cols = [c for c in px.columns.intersection(flag.columns).intersection(st.columns)
            if c.split(".")[0][:3] not in ("300", "301", "688", "689")]
    if not len(idx) or not cols:
        return _outcome([], 0, "涨跌停核验区间内无对齐的日频列")
    r = px.loc[idx, cols].pct_change(fill_method=None)
    up = flag.loc[idx, cols].astype("float64")
    stv = st.loc[idx, cols].astype("float64").fillna(0.0)
    thr = np.where(stv.to_numpy() > 0.5, float(cfg["thresholds"]["main_st"]), float(cfg["thresholds"]["main"]))
    mask = r.notna().to_numpy()
    derived = (r.to_numpy() >= thr) & mask
    vendor = up.to_numpy() > 0.5
    n = int(mask.sum())
    agree = float((derived == vendor)[mask].mean()) if n else 0.0
    problems = []
    if agree < float(cfg["min_agreement"]):
        problems.append(
            f"涨跌停派生与供应商 flag 一致率={agree:.5f} < {cfg['min_agreement']}"
            f"（假阳 {int((derived & ~vendor).sum())}、漏报 {int((~derived & vendor & mask).sum())}）")
    return _outcome(problems, n)


def _norm_codes(s: pd.Series) -> pd.Series:
    c = s.astype(str).str.extract(r"(\d{6})")[0].fillna("000000")
    return c.map(lambda x: ("sh" if x[0] in "69" else "sz") + x)


def check_a7_return_series(root, spec=None) -> dict:
    cfg = _contract(_spec_for(root, spec), "a7_return_series")
    left_path = _admit(root, cfg["left"]["path"])
    right_path = _admit(root, cfg["right"]["path"])
    missing = [cfg[k]["path"] for k, v in (("left", left_path), ("right", right_path)) if v is None]
    if missing:
        return _outcome([], 0, "缺对拍序列: " + ", ".join(missing))
    left = pd.read_csv(left_path, dtype={cfg["left"]["code_col"]: str, "month": str})
    right = pd.read_csv(right_path, dtype={cfg["right"]["code_col"]: str, "month": str})
    left["_code"] = _norm_codes(left[cfg["left"]["code_col"]])
    right["_code"] = _norm_codes(right[cfg["right"]["code_col"]])
    j = left.rename(columns={cfg["left"]["value_col"]: "L"}).merge(
        right.rename(columns={cfg["right"]["value_col"]: "R_"}),
        on=["_code", "month"], how="inner", suffixes=("", "_r"))
    problems = []
    if not len(j):
        return _outcome([f"{cfg['left']['path']} 与 {cfg['right']['path']} 无可对齐行"], 0)
    corr = float(j[["L", "R_"]].corr().iloc[0, 1])
    diff = (j["L"] - j["R_"]).abs()
    if corr < float(cfg["min_corr"]):
        problems.append(f"两序列相关系数={corr:.5f} < {cfg['min_corr']}")
    if float(diff.max()) > float(cfg["max_abs_diff"]):
        problems.append(f"两序列 max|diff|={float(diff.max()):.3e} > {cfg['max_abs_diff']}")
    return _outcome(problems, len(j))


def check_a8_flag_coverage(root, spec=None) -> dict:
    cfg = _contract(_spec_for(root, spec), "a8_flag_coverage")
    problems, checked, missing = [], 0, []
    for item in cfg.get("flags", []):
        p = _admit(root, item["path"])
        if p is None:
            missing.append(item["path"])
            continue
        df = _read_pickle(p)
        checked += 1
        share = float(df.notna().to_numpy().mean())
        if share < float(item["min_nonnull"]):
            problems.append(f"{item['path']}: 非空率={share:.4f} < {item['min_nonnull']}")
    if missing:
        return _outcome(problems, checked, "缺 flag 面板: " + ", ".join(missing))
    return _outcome(problems, checked)


CHECKS = {
    "a1_schema": check_a1_schema,
    "a2_universe": check_a2_universe,
    "a3_pit": check_a3_pit,
    "a4_units": check_a4_units,
    "a5_cliffs": check_a5_cliffs,
    "a6_limit_flags": check_a6_limit_flags,
    "a7_return_series": check_a7_return_series,
    "a8_flag_coverage": check_a8_flag_coverage,
}


def run_contracts(root=ROOT, only=None, spec=None) -> list:
    root = Path(root)
    results = []
    for cid, fn in CHECKS.items():
        if only and cid not in only:
            continue
        outcome = fn(root, spec)
        problems = outcome["problems"]
        blocked = outcome.get("blocked")
        status = "blocked" if blocked else ("fail" if problems else "pass")
        evidence = {
            "contract": cid,
            "title": TITLES[cid],
            "status": status,
            "checked": outcome["checked"],
            "blocked": blocked,
            "problems": problems,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        out = root / EVIDENCE_DIR / f"contract_{cid}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        results.append({**evidence, "evidence": _rel(root, out)})
    return results
