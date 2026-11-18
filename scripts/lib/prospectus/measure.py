"""s3_network_prospectus_anchor 的 measure 模块（逐字搬移，逻辑未改）。"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data_audit'))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data'))

from datetime import datetime, timezone
from lib.prospectus.consts import CACHE
from lib.prospectus.consts import CACHE_INFO
from lib.prospectus.consts import GEN_SUB_MAXLEN
from lib.prospectus.consts import KWS
from lib.prospectus.consts import REJECT_LOG
from lib.prospectus.consts import REPAIR_LOG
from lib.prospectus.consts import RESIDUAL_CLS
from lib.prospectus.consts import ROOT
from lib.prospectus.consts import WIN_POST
from lib.prospectus.consts import WIN_PRE
from lib.prospectus.consts import _MEAS
from lib.prospectus.ingest import resolve_network_version
from lib.prospectus.names import build_self_index
from lib.prospectus.names import rel
from lib.prospectus.names import self_names_for
from lib.prospectus.report import write_anchor_md
from lib.prospectus.text import _insufficient
from lib.prospectus.text import classify_unmapped
from lib.prospectus.text import extract_candidates
from lib.prospectus.text import locate
from lib.prospectus.text import map_name
from lib.prospectus.universe import build_full_name_map
from lib.prospectus.universe import load_firm_universe
from lib.prospectus.universe import load_universe_name_keys
import collections
import csv
import json
import math
import os
import time

def _init_measure_worker():
    """worker 初始化：加载 firm universe + 名称注册表（缓存）+ universe 名称键（每进程一次）。

    修 5（M2-i / registry 1.5.15）：同时构建**名称注册表按 code 倒排**（`build_self_index`），
    使发行人自名集合含「本发行人自身的代码锚定历史简称」（不再只用 universe 当前名 + 全称字典）。
    """
    firm, short = load_firm_universe()
    full, short_hist, full_info = build_full_name_map(firm)
    if not os.path.exists(CACHE) or full_info.get("cache_version") != 4:
        raise SystemExit("名称注册表缓存构建失败（cache_version != 4）")
    for nm, cs in short_hist.items():
        short.setdefault(nm, set()).update(cs)
    build_self_index(full, short_hist)
    _MEAS.update(firm=firm, short=short, full=full, uni_keys=load_universe_name_keys(),
                 full_info=full_info)

def _measure_one(job):
    """逐份测量：locate（三通道回退，与 ingest 同链）→ 抽名单 → 映射 → 分类；返回紧凑记录。"""
    code, year, ver, text_year, boundary, pdfd, txtd = job
    firm, short, full, uni_keys = _MEAS["firm"], _MEAS["short"], _MEAS["full"], _MEAS["uni_keys"]
    pdfp, txtp = os.path.join(pdfd, f"{code}.pdf"), os.path.join(txtd, f"{code}.txt")
    rec = {"code": code, "ipo_year": year, "network_version_used": ver,
           "network_text_year": text_year, "boundary_year_subset": boundary}
    empty = dict(map_modes={}, peers=[], mapped_peers_codes=[], unmapped_names=[],
                 unmapped_classified=[], rejected=[])
    if not (os.path.exists(pdfp) and os.path.exists(txtp)):
        rec.update(status="no_file", channel="no_file", located=False, page_count=0, char_count=0,
                   n_kw_hits=0, first_hit_page="", n_spans=0, n_unique_names=0, n_mapped=0,
                   n_unmapped=0, reason="pdf 或 txt 缺失（下载未成功，详见 download_log.csv）", **empty)
        return rec
    try:
        (located, n_hits, counts, first_page, n_pages, spans, text,
         offsets, channel) = locate(txtp, pdfp, code=code)
    except Exception as e:
        rec.update(status="parse_error", channel=f"error:{type(e).__name__}", located=False,
                   page_count=0, char_count=0, n_kw_hits=0, first_hit_page="", n_spans=0,
                   n_unique_names=0, n_mapped=0, n_unmapped=0,
                   reason=f"locate 异常：{str(e)[:120]}", **empty)
        return rec
    insuf = _insufficient(text, n_pages)
    rec.update(status="parsed" if located else "no_section", channel=channel, located=located,
               page_count=n_pages, char_count=len(text), n_kw_hits=n_hits, kw_counts=counts,
               first_hit_page=first_page if first_page else "", n_spans=len(spans),
               text_insufficient=insuf)
    if not located:
        rec.update(n_unique_names=0, n_mapped=0, n_unmapped=0, **empty)
        rec["reason"] = (f"文本层缺失/不足（CID/OCR 回退后仍不足；company_count={text.count('公司')}, "
                         f"chars={len(text)}, pages={n_pages}）" if insuf else
                         f"章节未定位（4 关键词全 0；chars={len(text)}, pages={n_pages}）")
        return rec
    self_exact, self_core = self_names_for(firm, code)     # 修 5（M2-i）：含代码锚定历史简称
    uni_keys = _MEAS["uni_keys"]
    REJECT_LOG.clear()
    REPAIR_LOG.clear()                                     # 修 5（M1②）：R2 左边界修复证据（逐份）
    cands = extract_candidates(text, spans, self_exact, self_core, uni_keys)
    rej = [dict(base=x["base"], peer=x.get("code"), why=x["why"]) for x in REJECT_LOG]
    REJECT_LOG.clear()
    repaired = list(REPAIR_LOG)                            # 证据：raw → fixed（不进方向性检查）
    REPAIR_LOG.clear()
    uniq = {}
    for c in cands:
        key = c["base"]
        prev = uniq.get(key)
        if prev is None or (prev["adjacent_code"] is None and c["adjacent_code"]):
            uniq[key] = c
    mapped, modes, unmapped = {}, {}, []
    for b, c in sorted(uniq.items()):
        code6, mode = map_name(c, short, full, firm, at_year=year)
        modes[mode] = modes.get(mode, 0) + 1
        if code6 == code:      # 映射到发行人自身 → 不计入自报同行（释义/正文自指，试点口径）
            continue
        if code6:
            mapped[b] = dict(code6=code6, mode=mode, raw=c["raw"])
        else:
            unmapped.append(dict(base=b, raw=c["raw"], rule=c["rule"]))
    cls_list = []
    for u in unmapped:
        cls, ev = classify_unmapped(u["base"], u["raw"], text, uni_keys, self_exact)
        cls_list.append(dict(name=u["base"], cls=cls, evidence=ev))
    rec.update(n_unique_names=len(uniq), n_mapped=len(mapped), n_unmapped=len(unmapped),
               rule_counts=dict(collections.Counter(c["rule"] for c in cands)),
               map_modes=modes, peers=sorted(mapped.keys()),
               mapped_peers_codes=sorted({v["code6"] for v in mapped.values()}),
               unmapped_names=[u["base"] for u in unmapped],
               unmapped_classified=cls_list, rejected=rej, r2_left_repairs=repaired)
    return rec

def _wilson(k, n, z=1.959963984540054):
    """Wilson score 区间（95%）；在 n = 0 时返回 (None, None)。"""
    if not n:
        return None, None
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)

def _load_network(text_year, net_dir):
    """读某网络版本的节点集/无向边集/无向对密度（与 lift_pair_density_readout 同口径）。"""
    with open(os.path.join(net_dir, f"{text_year}_support.csv"), encoding="utf-8") as f:
        sup = list(csv.DictReader(f))
    edges = set()
    with open(os.path.join(net_dir, f"{text_year}_similarity.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            a, b = r["focal_id"], r["source_id"]
            if a != b:
                edges.add((a, b) if a < b else (b, a))
    N, E = len(sup), len(edges)
    return {"text_year": text_year, "version": f"pm_ma_ewcos_teacher_v1_{text_year}",
            "N": N, "E": E, "nodes": {r["focal_id"] for r in sup}, "edges": edges,
            "d": E / (N * (N - 1) / 2) if N > 1 else float("nan")}

def _dedupe_readings(recs, label):
    """年内/池化去重口径下的名称统计：唯一名 / 未映射 / 四分类 + 分列失败率。

    修 5 ④（registry 1.5.15）：分类增第 4 类「抽取噪声」（additive、描述性）；**「剔碎片后」列口径不改**，
    另**增列「剔碎片与噪声后」**；主口径失败率（上界·含碎片）与停机线判定口径不变。
    """
    names, unmapped, cls_of = {}, {}, {}
    for r in recs:
        for nm in (r.get("peers") or []):
            names[nm] = names.get(nm, 0) + 1
        for nm in (r.get("unmapped_names") or []):
            names[nm] = names.get(nm, 0) + 1
            unmapped[nm] = unmapped.get(nm, 0) + 1
        for u in (r.get("unmapped_classified") or []):
            cls_of.setdefault(u["name"], u)
    by = {"碎片": [], "名称漂移": [], "抽取噪声": [], RESIDUAL_CLS: []}
    for n in sorted(cls_of):
        by.setdefault(cls_of[n]["cls"], []).append(cls_of[n])
    n_names, n_unm = len(names), len(unmapped)
    n_frag = len(by["碎片"])
    n_noise = len(by["抽取噪声"])
    return {
        "口径": label, "n_unique_names": n_names, "n_unmapped": n_unm,
        "上界（含碎片，主口径）": (n_unm / n_names) if n_names else None,
        "上界分子分母": f"{n_unm}/{n_names}",
        "剔碎片后": ((n_unm - n_frag) / (n_names - n_frag)) if n_names > n_frag else None,
        "剔碎片后分子分母": f"{n_unm - n_frag}/{n_names - n_frag}",
        "剔碎片与噪声后": ((n_unm - n_frag - n_noise) / (n_names - n_frag - n_noise))
                          if n_names > n_frag + n_noise else None,
        "剔碎片与噪声后分子分母": f"{n_unm - n_frag - n_noise}/{n_names - n_frag - n_noise}",
        "碎片": n_frag, "名称漂移": len(by["名称漂移"]), "抽取噪声": n_noise,
        RESIDUAL_CLS: len(by[RESIDUAL_CLS]),
        "明细": {k: [{"name": u["name"], "evidence": u["evidence"]} for u in v]
                 for k, v in by.items()},
    }

def _year_rates(rs, cohort_n=None):
    """单年四项中的前三项（解析/映射失败/覆盖）+ 通道/状态计数。"""
    n_dl = sum(1 for r in rs if r["status"] != "no_file")
    n_parsed = sum(1 for r in rs if r["status"] == "parsed")
    covered = [r for r in rs if r.get("n_mapped", 0) >= 1]
    net_nodes = [r for r in rs if r.get("network_node")]
    cov_net = [r for r in net_nodes if r.get("n_mapped", 0) >= 1]
    return {
        "cohort": len(rs), "downloaded": n_dl, "parsed": n_parsed,
        "no_section": sum(1 for r in rs if r["status"] == "no_section"),
        "no_file": sum(1 for r in rs if r["status"] == "no_file"),
        "parse_error": sum(1 for r in rs if r["status"] == "parse_error"),
        "parse_success_rate": (n_parsed / n_dl) if n_dl else None,
        "n_covered": len(covered), "coverage_rate": (len(covered) / len(rs)) if rs else None,
        "n_network_nodes": len(net_nodes), "n_covered_in_network": len(cov_net),
        # 修 6（registry 1.5.17）：网络节点**两口径**并报——located 口径 = 已定位件的在网家数；
        # 预登记会员口径 = cohort 中 firm_id ∈ 该网络版本节点集的家数（含未下载/未定位件）。
        "n_network_nodes_member_scope": sum(1 for r in rs if r.get("network_member")),
        "coverage_rate_in_network": (len(cov_net) / len(net_nodes)) if net_nodes else None,
        "peers_per_firm_mean": ((sum(r["n_mapped"] for r in rs if r.get("located"))
                                 / max(1, sum(1 for r in rs if r.get("located"))))),
        "channels": {k: sum(1 for r in rs if r.get("channel") == k)
                     for k in sorted({r.get("channel", "?") for r in rs})},
    }

def measure_main(args):
    """全量测量（汇总轮）：逐年 + 池化四项率与 lift，边界年子集单列。"""
    years = []
    for part in args.years.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            years += list(range(int(a), int(b) + 1))
        elif part:
            years.append(int(part))
    years = sorted(set(years))
    firm, short = load_firm_universe()
    full, short_hist, full_info = build_full_name_map(firm)
    CACHE_INFO.update(full_info or {})
    for nm, cs in short_hist.items():
        short.setdefault(nm, set()).update(cs)
    print(f"[M-1] 名称字典：firm universe {len(firm)}；简称键 {len(short)}；全称键 {len(full)}"
          f"（覆盖 {full_info.get('n_codes')} 只；cache_version={full_info.get('cache_version')}）", flush=True)

    verb = {y: resolve_network_version(y, args.net_dir) for y in years}
    nets = {}
    for ty in sorted({verb[y][0] for y in years}):
        nets[ty] = _load_network(ty, args.net_dir)
        print(f"[M-1] 网络 {ty}：N={nets[ty]['N']} E={nets[ty]['E']} d={nets[ty]['d']:.6f}", flush=True)

    jobs = []
    for y in years:
        ty, ver, bd = verb[y]
        pdfd = os.path.join(args.base, str(y), "pdf")
        txtd = os.path.join(args.base, str(y), "txt")
        for c, m in sorted(firm.items()):
            if m["ipo_date"][:4] == str(y):
                jobs.append((c, y, ver, ty, bd, pdfd, txtd))
    print(f"[M-1] 计划 {len(jobs)} 份（{years[0]}–{years[-1]}）；workers={args.workers}", flush=True)

    recs, t0 = [], time.time()
    if args.workers <= 1:
        _init_measure_worker()
        for i, j in enumerate(jobs, 1):
            recs.append(_measure_one(j))
            if i % args.progress_every == 0 or i == len(jobs):
                print(f"[M-2] {i}/{len(jobs)}（parsed="
                      f"{sum(1 for x in recs if x['status'] == 'parsed')} elapsed={time.time() - t0:.0f}s）",
                      flush=True)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers, initializer=_init_measure_worker) as pool:
            for i, r in enumerate(pool.imap_unordered(_measure_one, jobs, chunksize=1), 1):
                recs.append(r)
                if i % args.progress_every == 0 or i == len(jobs):
                    print(f"[M-2] {i}/{len(jobs)}（parsed="
                          f"{sum(1 for x in recs if x['status'] == 'parsed')} elapsed={time.time() - t0:.0f}s）",
                          flush=True)
    recs.sort(key=lambda x: (x["ipo_year"], x["code"]))
    print(f"[M-2] 完成 {len(recs)} 份（{time.time() - t0:.0f}s）", flush=True)

    # ---- 逐份：网络对计数（focal 在网才有对；边/节点均为 firm_id 口径）----
    for r in recs:
        net = nets[verb[r["ipo_year"]][0]]
        fid = firm[r["code"]]["firm_id"]
        r["network_member"] = fid in net["nodes"]     # 修 6：预登记会员口径（不以 located 为条件）
        if not r.get("located") or fid not in net["nodes"]:
            r.update(network_node=False, n_pairs=0, n_pairs_edge=0, overlap_share=None)
            continue
        pset = sorted({firm[j]["firm_id"] for j in r.get("mapped_peers_codes") or []} - {fid})
        tot = len(pset)
        hit = sum(1 for j in pset if (min(fid, j), max(fid, j)) in net["edges"])
        pset_in = {j for j in pset if j in net["nodes"]}
        hit_in = sum(1 for j in pset_in if (min(fid, j), max(fid, j)) in net["edges"])
        r.update(network_node=True, n_pairs=tot, n_pairs_edge=hit,
                 overlap_share=(hit / tot) if tot else None,
                 n_pairs_in_network=len(pset_in), n_pairs_edge_in_network=hit_in)

    cohort_all = len(recs)
    overall = _year_rates(recs, cohort_all)
    overall["cohort"] = cohort_all
    overall["coverage_rate"] = sum(1 for r in recs if r.get("n_mapped", 0) >= 1) / cohort_all

    per_year, per_year_lift = {}, {}
    for y in years:
        rs = [r for r in recs if r["ipo_year"] == y]
        ty, ver, bd = verb[y]
        net = nets[ty]
        yr = _year_rates(rs, len(rs))
        yr["network_version_used"] = ver
        yr["network_text_year"] = ty
        yr["boundary_year_subset"] = bd
        yr["network"] = {"N": net["N"], "E": net["E"], "d": net["d"]}
        pt = sum(r.get("n_pairs", 0) for r in rs)
        ph = sum(r.get("n_pairs_edge", 0) for r in rs)
        pt_in = sum(r.get("n_pairs_in_network", 0) for r in rs)
        ph_in = sum(r.get("n_pairs_edge_in_network", 0) for r in rs)
        share = (ph / pt) if pt else None
        share_in = (ph_in / pt_in) if pt_in else None
        lo, hi = _wilson(ph, pt)
        yr.update({
            "pair_total": pt, "pair_edge": ph, "overlap_share": share,
            "lift": (share / net["d"]) if (share is not None and net["d"] == net["d"]) else None,
            "overlap_share_ci95": [lo, hi],
            "lift_ci95": [(lo / net["d"]) if lo is not None else None,
                          (hi / net["d"]) if hi is not None else None],
            "diag_pair_total": pt_in, "diag_pair_edge": ph_in, "diag_overlap_share": share_in,
            "diag_lift": (share_in / net["d"]) if (share_in is not None and net["d"] == net["d"]) else None,
        })
        per_year[str(y)] = {k: v for k, v in yr.items() if k not in ("channels",)}
        per_year_lift[str(y)] = yr
        per_year[str(y)]["channel_dist"] = yr["channels"]

    # ---- 池化：主读数年（非边界）与全期（含边界，作对照）----
    def _pooled(sel_years):
        pt = sum(per_year_lift[str(y)]["pair_total"] for y in sel_years)
        ph = sum(per_year_lift[str(y)]["pair_edge"] for y in sel_years)
        pt_in = sum(per_year_lift[str(y)]["diag_pair_total"] for y in sel_years)
        ph_in = sum(per_year_lift[str(y)]["diag_pair_edge"] for y in sel_years)
        exp = sum(per_year_lift[str(y)]["pair_total"] * per_year_lift[str(y)]["network"]["d"]
                  for y in sel_years)
        exp_in = sum(per_year_lift[str(y)]["diag_pair_total"] * per_year_lift[str(y)]["network"]["d"]
                     for y in sel_years)
        share = (ph / pt) if pt else None
        share_in = (ph_in / pt_in) if pt_in else None
        lo, hi = _wilson(ph, pt)
        d_w = (exp / pt) if pt else None
        return {
            "years": sel_years, "pair_total": pt, "pair_edge": ph,
            "overlap_share": share, "pooled_density_dw": d_w,
            "lift": (ph / exp) if exp else None,
            "overlap_share_ci95": [lo, hi],
            "lift_ci95": [(lo / d_w) if (lo is not None and d_w) else None,
                          (hi / d_w) if (hi is not None and d_w) else None],
            "diag_pair_total": pt_in, "diag_pair_edge": ph_in, "diag_overlap_share": share_in,
            "diag_lift": (ph_in / exp_in) if exp_in else None,
            # 修 7（registry 1.5.17）：原名 `weighted_mean_of_year_lifts` 与池化 lift **代数恒等**
            #（Σ(pt_y·d_y·lift_y)/Σ(pt_y·d_y) ≡ Σph/Σexp ≡ 池化 lift）→ 改名 + 标注**非独立证据**（数值保留）。
            "identity_check_pooled_lift": ((sum(per_year_lift[str(y)]["pair_total"]
                                                * per_year_lift[str(y)]["network"]["d"]
                                                * (per_year_lift[str(y)]["lift"] or 0)
                                                for y in sel_years) / exp) if exp else None),
            "identity_check_pooled_lift_note": "恒等式校验（非独立证据）：Σ(pt_y·d_y·lift_y)/Σ(pt_y·d_y) "
                                               "≡ Σph/Σexp ≡ 池化 lift；仅作代数一致性校验，不得当独立读数引用。",
        }

    main_years = [y for y in years if not verb[y][2]]
    boundary_years = [y for y in years if verb[y][2]]
    pooled_main = _pooled(main_years)
    pooled_all = _pooled([y for y in years if per_year_lift[str(y)]["pair_total"]])
    boundary = {}
    for y in boundary_years:
        py = per_year_lift[str(y)]
        boundary[str(y)] = {
            "cohort": py["cohort"], "downloaded": py["downloaded"], "parsed": py["parsed"],
            "n_network_nodes": py["n_network_nodes"], "pair_total": py["pair_total"],
            "overlap_share": py["overlap_share"], "lift": py["lift"],
            "network_version_used": py["network_version_used"],
            "note": "边界年（y_target = min(Y+2, 2023) 不可得 → fallback v1_2022）：cohort 在该网络版本中"
                    "节点数 = 0（冻结样本筛选 ipo<2y）→ 锚读数退化（无对）；只报下载/解析/覆盖等计数，"
                    "不参与池化与主读数。",
        }

    # ---- 映射失败率分列（池化全局 / 主读数年 / 边界年 / 逐年）----
    fail_all = _dedupe_readings(recs, "全量池化（2015–2023 跨年去重，主口径）")
    fail_main = _dedupe_readings([r for r in recs if not verb[r["ipo_year"]][2]],
                                 "主读数年池化（2015–2020 跨年去重，诊断）")
    fail_bnd = _dedupe_readings([r for r in recs if verb[r["ipo_year"]][2]],
                                "边界年子集（2021–2023 跨年去重，单列）")
    fail_year = {str(y): _dedupe_readings([r for r in recs if r["ipo_year"] == y],
                                          f"{y} 年内去重") for y in years}

    # ---- 通道对照（与 ingest 明细逐份核对；不一致即口径漂移证据）----
    cross = {"ingest_detail": "", "n_compared": 0, "status_mismatch": [], "channel_mismatch": []}
    if args.ingest_detail and os.path.exists(args.ingest_detail):
        cross["ingest_detail"] = rel(args.ingest_detail)
        ing = {r["code"]: r for r in csv.DictReader(open(args.ingest_detail, encoding="utf-8"))}
        for r in recs:
            g = ing.get(r["code"])
            if not g:
                cross["status_mismatch"].append({"code": r["code"], "ingest": "missing"})
                cross["n_compared"] += 1
                continue
            cross["n_compared"] += 1
            if g["status"] != r["status"]:
                cross["status_mismatch"].append({"code": r["code"], "ingest": g["status"],
                                                 "measure": r["status"]})
            gi = g["channel"]
            if gi != r["channel"]:
                cross["channel_mismatch"].append({"code": r["code"], "ingest": g["channel"],
                                                  "measure": r["channel"]})

    # ---- 下载失败概览（19 件；逐条 reason 来自各年 download_log.csv）----
    dl = {"n_failed": 0, "by_status": {}, "items": []}
    for y in years:
        lp = os.path.join(args.base, str(y), "download_log.csv")
        if not os.path.exists(lp):
            continue
        last = {}
        for row in csv.DictReader(open(lp, encoding="utf-8")):
            last[row.get("code", "")] = row
        for r in recs:
            if r["ipo_year"] != y or r["status"] != "no_file":
                continue
            row = last.get(r["code"], {})
            st = row.get("status", "unknown")
            dl["by_status"][st] = dl["by_status"].get(st, 0) + 1
            dl["items"].append({"year": y, "code": r["code"], "status": st,
                                "tried": (row.get("tried") or "")[:120]})
            dl["n_failed"] += 1

    # ---- 非 pymupdf 通道逐件披露（OCR / CID）----
    chan = [{"year": r["ipo_year"], "code": r["code"], "channel": r["channel"],
             "pages": r["page_count"], "chars": r["char_count"], "kw_hits": r["n_kw_hits"],
             "status": r["status"]} for r in recs if r.get("channel") not in ("pymupdf", "no_file")]

    # ---- 方向性检查（全量）：被剔名-码对（双方均在网）落网率 vs 保留集 ----
    rej = {}
    for r in recs:
        for x in (r.get("rejected") or []):
            p = x.get("peer")
            if p and p in firm and r["code"] in firm:
                fid = firm[r["code"]]["firm_id"]
                j = firm[p]["firm_id"]
                ty = verb[r["ipo_year"]][0]
                if fid != j and fid in nets[ty]["nodes"] and j in nets[ty]["nodes"]:
                    rej.setdefault((ty, fid, j), x["base"])
    rej_hit = sum(1 for (ty, i, j) in rej if (min(i, j), max(i, j)) in nets[ty]["edges"])

    # ---- 停机线与外部锚条件（机械判定，不得放宽）----
    th = {"parse_min": 0.80, "map_fail_max": 0.20, "coverage_min": 0.50}
    stop = []
    if overall["parse_success_rate"] < th["parse_min"]:
        stop.append(f"解析成功率 {overall['parse_success_rate']:.4f} < {th['parse_min']}")
    if fail_all["上界（含碎片，主口径）"] > th["map_fail_max"]:
        stop.append(f"映射失败率 {fail_all['上界（含碎片，主口径）']:.4f} > {th['map_fail_max']}")
    if overall["coverage_rate"] < th["coverage_min"]:
        stop.append(f"覆盖率 {overall['coverage_rate']:.4f} < {th['coverage_min']}")
    lift_complete = bool(pooled_main.get("lift") is not None
                         and all(per_year[str(y)]["lift"] is not None for y in main_years)
                         and boundary)
    external = {
        "criteria": {
            "解析成功率 ≥0.80": {"value": overall["parse_success_rate"],
                                 "pass": overall["parse_success_rate"] >= th["parse_min"]},
            "映射失败率 ≤0.20": {"value": fail_all["上界（含碎片，主口径）"], "scope": "全量池化上界口径",
                                 "pass": fail_all["上界（含碎片，主口径）"] <= th["map_fail_max"]},
            "覆盖率 ≥0.50": {"value": overall["coverage_rate"], "scope": "分母 = 2015–2023 cohort 2,656",
                              "pass": overall["coverage_rate"] >= th["coverage_min"]},
        },
        "lift_reports_complete": lift_complete,
        "stop_triggered": bool(stop), "stop_reasons": stop,
        "external_anchor_satisfied": (not stop) and lift_complete,
    }

    # ---- 失败构成诊断（描述性，不改任何主口径）----
    ucnt, ucls, uyr = collections.Counter(), {}, {}
    for r in recs:
        for nm in (r.get("unmapped_names") or []):
            ucnt[nm] += 1
            uyr.setdefault(nm, set()).add(r["ipo_year"])
        for u in (r.get("unmapped_classified") or []):
            ucls.setdefault(u["name"], u)
    split = {"疑似抽取噪声（整名 ≈ 通用词/科目/句片段词）": 0, RESIDUAL_CLS: 0}
    for nm in ucnt:
        c = (ucls.get(nm) or {}).get("cls")
        if c == "抽取噪声":
            split["疑似抽取噪声（整名 ≈ 通用词/科目/句片段词）"] += 1
        elif c == RESIDUAL_CLS:
            split[RESIDUAL_CLS] += 1
    n_resid = split[RESIDUAL_CLS]
    n_noise_u = split["疑似抽取噪声（整名 ≈ 通用词/科目/句片段词）"]
    failure_diag = {
        "note": "描述性诊断（不改主口径、不参与任何率/门槛判定）：① unmapped 按文档频次排序（识别高频噪声 token）；"
                "② 分类第 4 类「抽取噪声」（registry 1.5.17 修 3：判据 = **整名 ≈ 通用词**（整名 ∈ 词表 或 ≤"
                + str(GEN_SUB_MAXLEN) + " 字且含词表词）+ **公司后缀豁免**（公司/集团/股份/有限/企业 收尾或含 股份/集团））单列；"
                "③ **残差桶（未过前三类判据）**（registry 1.5.17 修 2 改名）——**兜底 return**，其来源**需独立判读**，"
                "**不得**以本桶断言「非 A 股实体」：复核抽样审计（n=160，从该桶随机抽）显示 真非上市/境外 ≈62.5%、"
                "抽取伪影 ≈30.0%、A 股名漂移 ≈7.5%；复核量化分解（外推）见 residual_bucket_audit。",
        "top_unmapped_by_docfreq": [{"name": nm, "n_docs": c, "cls": (ucls.get(nm) or {}).get("cls"),
                                     "years": sorted(uyr.get(nm, [])),
                                     "evidence": (ucls.get(nm) or {}).get("evidence")}
                                    for nm, c in ucnt.most_common(40)],
        "residual_split": split,
        "rename_note": "原字段 `non_a_share_split` / 原类名「非 A 股实体」= 兜底断言；registry 1.5.17 修 2 改为中性命名"
                       "「残差桶（未过前三类判据）」，字段名同步为 `residual_split`（数值口径未改，仅命名与文案）。",
        "residual_bucket_audit": {
            "bucket_n": n_resid,
            "sampling_audit": {"n_sampled": 160, "真非上市/境外": 100, "抽取伪影": 48, "A股名漂移": 12,
                               "share": {"真非上市/境外": 0.625, "抽取伪影": 0.300, "A股名漂移": 0.075},
                               "source": "独立复核（2026-10-14 第二轮）随机抽样 160 名人工/AI 独立判读"},
            "quantified_decomposition": {
                "note": "复核量化分解（**基于上一轮/修复前四分类（碎片 282 / 名称漂移 146 / 抽取噪声 117 / "
                        "残差桶（旧名「非 A 股实体」）1482）+ n=160 抽样外推**，口径 = 修复前 2027 未映射 / 5818 唯一名）："
                        "2027 未映射唯一名 ≈ 抽取伪影 844（= 桶内伪影外推 445（30.0%×1482）+ 上一轮抽取噪声 117 + "
                        "上一轮碎片 282）/ 真非 A 股 926（= 桶内 62.5%×1482）/ A 股名漂移 257（= 桶内漂移外推 111（7.5%×1482）+ "
                        "上一轮名称漂移 146）。**本轮修 3 重分类后四分类计数另计**（碎片 "
                        + str(fail_all["碎片"] if isinstance(fail_all, dict) else "—") + " / 名称漂移 "
                        + str(fail_all["名称漂移"] if isinstance(fail_all, dict) else "—") + " / 抽取噪声 " + str(n_noise_u)
                        + " / 残差桶 " + str(n_resid) + "），故最宽松修正读数沿用**复核口径**（= 修复前分母 5818）并注记。",
                "抽取伪影": 844, "真非A股": 926, "A股名漂移": 257, "合计": 2027,
                "loosest_correction": {"rate": (2027 - 844) / (5818 - 844), "numerator_denominator": "1183/4974",
                                       "review_value": 0.2379,
                                       "note": "最宽松修正（同剔全部抽取伪影 844）后仍 >0.20 → **停机不解除**"},
            },
        },
        "n_unmapped_unique": len(ucnt),
        "outcome_reading": (
            f"本轮实测（唯一名口径，修复后）：抽取噪声 {n_noise_u} 个（判据 = 整名 ≈ 通用词 + 公司后缀豁免）/ "
            f"残差桶（未过前三类判据）{n_resid} 个；该桶**需独立判读**——复核抽样（n=160）显示其中 "
            "真非上市/境外 ≈62.5%、抽取伪影 ≈30.0%、A 股名漂移 ≈7.5%，即**不是单一来源**；"
            "最宽松修正（同剔全部抽取伪影 844）后失败率 = 1183/4974 ≈ 0.2378（复核载 0.2379）**仍 > 0.20** → "
            "**停机不解除**、外部锚条件仍不满足（机械判定，未放宽任何门槛）。"),
    }

    # ---- 修复轮证据（H1/M1/M2；描述性，不改任何口径）----
    n_r2_rep = sum(len(r.get("r2_left_repairs") or []) for r in recs)
    r2_ex = collections.Counter()
    for r in recs:
        for x in (r.get("r2_left_repairs") or []):
            r2_ex[f"{x['raw']}→{x['fixed']}"] += 1
    n_self_rej = sum(1 for r in recs for x in (r.get("rejected") or [])
                     if str(x.get("why", "")).startswith("发行人自指"))
    rule_tot = collections.Counter()
    for r in recs:
        for k, v in (r.get("rule_counts") or {}).items():
            rule_tot[k] += v
    fix_evidence = {
        "note": "registry 1.5.15「先修解析」修复轮证据（解析层规则化修复；**无人工补录、未放宽匹配口径、未剔失败件**）："
                "H1 = 年报头部名称注册表切分（cache_version 3→4 强制重算）；M1 = 招股书抽取规则（R2 标签集 + 左边界剥离 + "
                "R5 整名噪声判据）；M2 = 发行人自名过滤容错（含代码锚定历史简称）。"
                "**修复轮二（registry 1.5.17）**：修 1 抽取层触发条件（R4 复合触发 + 和/与/及 并列切分 + R2 括号内空白容错 + "
                "R2 左边界语境后缀片段/尾随「有」（名录守卫）剥离 + 新增 R4_tbl_head）——逐份规则计数见 `extraction_rule_counts`。",
        "extraction_rule_counts": dict(sorted(rule_tot.items())),
        "name_registry_cache_version": CACHE_INFO.get("cache_version"),
        "name_registry_info": CACHE_INFO,
        "n_r2_left_repairs": n_r2_rep,
        "r2_left_repair_examples": [{"raw→fixed": k, "n": v} for k, v in r2_ex.most_common(15)],
        "n_self_name_rejections": n_self_rej,
        "n_shape_rejections": sum(1 for r in recs for x in (r.get("rejected") or [])
                                  if not str(x.get("why", "")).startswith("发行人自指")),
    }

    # ---- 与试点 2018 的对照 ----
    pilot_cmp = None
    ppath = os.path.join(ROOT, "reports", "data_gates", "G0", "s3_network_prospectus_pilot_2018.json")
    if os.path.exists(ppath):
        pj = json.load(open(ppath, encoding="utf-8"))
        py = per_year_lift["2018"]
        pfd = (pj["rates"].get("failure_decomposition") or {})
        pilot_cmp = {
            "pilot_artifact": rel(ppath),
            "pilot": {"n_downloaded": pj["rates"]["n_downloaded"], "n_located": pj["rates"]["n_located"],
                      "parse_success_rate": pj["rates"]["parse_success_rate"],
                      "mapping_failure_rate": pj["rates"]["mapping_failure_rate"],
                      "n_unmapped/n_unique": f"{pj['rates']['n_unmapped_names']}/{pj['rates']['n_unique_names']}",
                      "coverage_rate": pj["rates"]["coverage_rate"],
                      "n_covered/cohort": f"{pj['rates']['n_covered']}/{pj['rates']['cohort_n']}",
                      "overlap_share": pj["anchor_readout"]["overlap_share_raw"],
                      "pair_edge/pair_total": f"{pj['anchor_readout']['pair_edge']}/{pj['anchor_readout']['pair_total']}",
                      "lift": pj["anchor_readout"]["lift"],
                      "diag_lift": pj["anchor_readout"]["diagnostic_node_restricted"]["lift"],
                      "network_version": pj["scope"]["network_version"]},
            "full_2018": {"n_downloaded": py["downloaded"], "n_located": py["parsed"],
                          "parse_success_rate": py["parse_success_rate"],
                          "coverage_rate": py["coverage_rate"],
                          "n_covered/cohort": f"{py['n_covered']}/{py['cohort']}",
                          "mapping_failure_rate": fail_year["2018"]["上界（含碎片，主口径）"],
                          "n_unmapped/n_unique": fail_year["2018"]["上界分子分母"],
                          "overlap_share": py["overlap_share"],
                          "pair_edge/pair_total": f"{py['pair_edge']}/{py['pair_total']}",
                          "lift": py["lift"], "diag_lift": py["diag_lift"],
                          "network_version": py["network_version_used"]},
            "note": "试点口径（103 份下载 / 99 定位 / 重合率 0.555985 → lift 26.4573）与全量 2018 行的差异"
                    "来源 = ① 全量轮补 OCR 通道使 2 份无文本层扫描件（300760/300748）进入口径（cohort 105 全量），"
                    "② 抽取/映射规则与通道逐字一致、网络版本同为 v1_2020 → 差异只来自新增 2 份件与"
                    "「全量为 cohort 全量、试点为下载成功口径」的分母差。",
        }

    low_cov = [y for y in years if (per_year[str(y)]["coverage_rate"] or 0) < 0.50]
    known_limitations = [
        "**本轮为登记在案的「先修解析」修复轮**（registry 1.5.15；H1/M1/M2 三处解析缺陷的规则化修复）："
        "**判据零改动**（四项定义 / 停机线 / 匹配口径 / 覆盖率分母 / 网络版本规则 / 边界年处理一律不改）；"
        "修复前基线逐值见 `sensitivity_pre_fix`（含修复前后对照，§0）；修复**不解除任何措辞限制**——"
        "外部锚条件按修复后读数机械重出、不加宽任何门槛。",
        "**覆盖率为下界**：覆盖率分母（2015–2023 cohort 2,656 ∩ 项目 universe）固定，分子只随抽取/映射成功"
        f"而增加；未定位（{overall['no_section']} 份 no_section）、下载未成功（{overall['no_file']} 份 no_file）"
        "与抽到名单但无可映射同行只会让覆盖率更低。",
        f"**逐年覆盖率提示（口径并报）**：全量覆盖率（分母 = 2,656 池化）= {overall['coverage_rate']:.4f} 过线；"
        f"但按 workplan 规划期措辞「**逐年**名单覆盖率 <50% → 停」的字面口径，"
        + ("、".join(f"{y} 年 = {per_year[str(y)]['coverage_rate']:.4f}（{per_year[str(y)]['n_covered']}/"
                    f"{per_year[str(y)]['cohort']}）" for y in low_cov) if low_cov else "无年份低于 0.50")
        + "。registry 冻结的 `stop_rules` 为池化口径「覆盖率 <50%」（判据真源），两类口径均在本报告并报；"
          "无论按哪种口径，**映射失败率一项已独立超线** → 外部锚条件未满足的判定不变。",
        "**lift 方向未证**（沿用试点结论）：漏抽同行同时减少分子（漏掉的同行对）与分母（自报同行对总数），"
        "净方向取决于漏抽同行的落网率相对全体自报同行落网率的高低；全量方向性检查见 anchor_readout.direction_check，"
        f"被剔名-码对落网率 = {(rej_hit / len(rej)) if rej else None}（保留集诊断口径 = "
        f"{pooled_main.get('diag_overlap_share')}）→ 不宣称下界、不宣称上界。",
        "**CI 为指示性**：Wilson 区间按对（pair）二项抽样计算，未校正「同一 focal 内多对相关」的聚类效应；"
        "池化 lift 的 CI 只传递重合率的抽样不确定性，基线密度 d 视为固定。",
        "**SW2 快照与本步无关**：本锚只用「招股书自报名单 ↔ 主口径边表」的对照，不涉及申万二级行业快照"
        "（same_sw2 等诊断指标），故「SW2 快照版本」问题不影响本报告任何读数。",
        f"**OCR 回退件 {len([x for x in chan if 'ocr' in x['channel']])} 件 / CID 确定性还原件 "
        f"{len([x for x in chan if x['channel'].startswith('pdfminer+cid_gid')])} 件**（共 {len(chan)} 件非 pymupdf 通道）："
        "逐件通道/页数/字数/关键词命中见 §通道披露；OCR 为 tesseract 5.3.4 chi_sim（--psm 3、200 DPI），"
        "**非人工、非宽松匹配**；OCR 文本可能含识别噪声（已知：公司与代码识别错误率高于文本层）。",
        f"**下载失败 {dl['n_failed']} 件**（status 概览 = {dl['by_status']}）：逐条 reason 见 §下载失败概览"
        "与 `data/raw/cninfo_prospectus/<year>/download_log.csv`；**未静默剔除**（全部计入 cohort 与覆盖率分母）。",
        "**边界年（2021–2023）锚读数退化**：y_target = min(Y+2, 2023) 中 v1_2023 不存在 → fallback v1_2022；"
        "三 cohort 在该网络版本中节点数 = 0（冻结样本筛选「上市不足 2 年」）→ 无对、lift 不可测；"
        "边界年只报下载/解析/覆盖计数，单列不混入主读数。",
        "**修复轮二（registry 1.5.17，修 1–7）**：**判据零改动**（四项定义 / 停机线 / 匹配口径 / 覆盖率分母 / 网络版本规则 / "
        "边界年处理一律不改）；只修**抽取层触发条件**与**归因判据/文案**，**无人工补录、未放宽匹配、未调分母、未剔失败件**；"
        "修复前基线（上一轮 H1/M1/M2 后的读数）逐值见 `sensitivity_pre_round2`（更早的 v1 基线见 `sensitivity_pre_fix`）；"
        "修复件清单与新增同行对见 `repair_round2_evidence`。",
        "**修 1 修复件（曾漏抽、已修）**：① **300748**（上市 2018，OCR 件）：原抽 0 名（正文「同行业可比公司中科三环、宁波韵升、"
        "正海磁材、英洛华、京磁股份和大地能」无「包括/有：」触发词、末位成员含「和」→ 现由 R4 复合触发 + 和字切分 + R4_tbl_head 抽出）；"
        "② **300737**（上市 2018，CID 件）：原 4 名同行仅抽 1（东方雨虹 002271 因「（证\n\n券代码：002271）」断行 + 左边界语境「上市公司有」"
        "→ 现由 R2 括号内空白容错 + 左边界扩展抽出，凯伦股份 300715 保持）。两件前后对照逐值见 `repair_round2_evidence.focus_cases`。",
        "**残差桶（未过前三类判据）需独立判读（修 2）**：该桶是 `classify_unmapped` 的**兜底 return**，不是判据证据；"
        "复核抽样（n=160）显示 真非上市/境外 ≈62.5% / 抽取伪影 ≈30.0% / A 股名漂移 ≈7.5% → **不得**以该桶断言失败主体为单一来源；"
        "复核量化分解（2027 未映射 ≈ 抽取伪影 844 / 真非 A 股 926 / A 股漂移 257）与**最宽松修正后 0.2378（复核载 0.2379）仍 >0.20**"
        "见 `failure_diagnostics.residual_bucket_audit`。",
        "**抽取噪声判据（修 3）**：判据 = 「整名 ≈ 通用词」（整名 ∈ 词表 或 ≤5 字且含词表词）+ **公司后缀豁免**"
        "（公司/集团/股份/有限/企业 收尾或含 股份/集团）——原「子串命中」判据把 117 条中的 73 条（≥6 字或带 公司/集团/股份，"
        "如 上海弘陆物流设备 / 山东联创节能新材料）误判为抽取噪声，现归入残差桶（**该分列仍为描述性规则判读口径**，主口径失败率不依赖它）。",
        "**字段语义（修 4）**：`first_hit_page` 对**非 pymupdf 通道**（CID 确定性还原 / OCR 回退）置 `not_applicable_channel`"
        "（替换后文本的字符偏移与 pymupdf 页码不对应，不得以页码冒充）；pymupdf 通道仍报真实页码。",
        "**网络节点两口径（修 6）**：`n_network_nodes`（**located 口径**）只数已定位件中的在网家数；"
        "`n_network_nodes_member_scope`（**预登记会员口径**）= cohort 中 firm_id ∈ 该网络版本节点集的家数（含未下载/未定位件）——"
        "两口径并报、不得混用（预登记轮登记值 = 1,537 / 1,592 = 96.5%）。",
        "**恒等式校验（修 7）**：`identity_check_pooled_lift` 与池化 lift **代数恒等**（非独立证据；原名 "
        "`weighted_mean_of_year_lifts`），只作一致性校验，不得当独立读数引用。",
        "**修 1 新增抽取的粘连片段（如实披露，未做事后剔除）**：修复轮二新增抽取的未映射名中约半数属「真名 + 粘连前缀/后缀」"
        "（如 莱克电气显 / 选择兆龙互连 / 定期报告 / 公开资料），与既有已知残余「太原重式截断名」同类（R4 列举型 token 为"
        "无代码锚定的字面切片，遇换行/长词即截断）；该项使**主口径失败率上升**（上一轮 0.3484 → 本轮 0.3670，方向上不利于过线），"
        "且已计入分母与分子——**未按任何判据事后剔除、未人工补录**。",
        "**已知残余**：① 「太原重」式截断名（形态上与真名不可分）；② 表格列错行的码-名错位（名与码都在同行集内但配对错位）；"
        "③ 抽取口径偏保守（未含未上市/境外/港股同业）→ 覆盖率与重合率为下界口径。",
        "**角色限定**：本锚只作外部一致性读数，不进入面板/回归、不改主口径；"
        "「边级语义」表述仍受 #10 证据层级（AI 独立判读，三代理、非人工）限定约束。",
    ]

    # ---- 修复前基线（registry 1.5.15 修复轮 F3：前置/后对照；只读引用，不改本轮任何读数）----
    pre = None
    ppre = args.pre_fix_report
    if ppre and os.path.exists(ppre):
        try:
            p = json.load(open(ppre, encoding="utf-8"))
            pfd = p["rates"]["failure_decomposition_pooled"]
            plp = p["anchor_readout"]["pooled_main_years_2015_2020"]
            pre = {
                "source": rel(ppre), "run_at_before_fix": p["run_at"],
                "note": "修复前（registry 1.5.13/1.5.14 全量第二轮）读数；口径与判据逐字未改，"
                        "差异全部来自解析层修复（H1/M1/M2，registry 1.5.15 登记）。"
                        "修复前分类为**三分类**（碎片/名称漂移/非 A 股实体）——本轮新增第 4 类「抽取噪声」"
                        "（作用位置 = 原「非 A 股实体」桶内细分，additive），故修复前「非 A 股实体」"
                        "1,584 个内含两类（形态为公司名 1,470 / 通用词·科目·句片段 114）。",
                "parse_success_rate": p["rates"]["overall"]["parse_success_rate"],
                "parse_n": f"{p['rates']['overall']['parsed']}/{p['rates']['overall']['downloaded']}",
                "mapping_failure_rate_upper": pfd["上界（含碎片，主口径）"],
                "mapping_failure_n": pfd["上界分子分母"],
                "mapping_failure_ex_frag": pfd["剔碎片后"],
                "mapping_failure_ex_frag_n": pfd["剔碎片后分子分母"],
                "cls3": {"碎片": pfd["碎片"], "名称漂移": pfd["名称漂移"], "非 A 股实体": pfd["非 A 股实体"]},
                "coverage_rate": p["rates"]["overall"]["coverage_rate"],
                "coverage_n": f"{p['rates']['overall']['n_covered']}/{p['rates']['overall']['cohort']}",
                "pooled_lift": plp["lift"], "pooled_pair": f"{plp['pair_edge']}/{plp['pair_total']}",
                "pooled_overlap_share": plp["overlap_share"],
                "external_anchor_satisfied": p["external_anchor"]["external_anchor_satisfied"],
                "per_year": {y: {"parse_success_rate": p["rates"]["per_year"][y]["parse_success_rate"],
                                 "coverage_rate": p["rates"]["per_year"][y]["coverage_rate"],
                                 "mapping_failure_rate": p["rates"]["failure_decomposition_per_year"][y]["上界（含碎片，主口径）"],
                                 "lift": p["anchor_readout"]["per_year"][y]["lift"]}
                             for y in map(str, years)},
            }
        except Exception as e:      # 只读对照，失败不阻断本轮（报告注明缺失）
            pre = {"source": rel(ppre), "error": f"{type(e).__name__}: {str(e)[:120]}"}

    # ---- 修复轮二：上一轮（H1/M1/M2）基线 + 修复件清单/新增同行对（只读对照，不改本轮读数）----
    pre2, repair2 = None, None
    if args.pre_fix_report2 and os.path.exists(args.pre_fix_report2):
        try:
            q = json.load(open(args.pre_fix_report2, encoding="utf-8"))
            qfd = q["rates"]["failure_decomposition_pooled"]
            qlp = q["anchor_readout"]["pooled_main_years_2015_2020"]
            qres = [k for k in qfd.keys() if k.startswith("残差桶") or k == "非 A 股实体"]
            qres_key = qres[0] if qres else "非 A 股实体"
            pre2 = {
                "source": rel(args.pre_fix_report2), "run_at_before_fix2": q["run_at"],
                "note": "上一轮（H1/M1/M2 解析层修复轮，registry 1.5.15/1.5.16）修复后读数 = **本轮修复前基线**；"
                        "本轮（修复轮二，registry 1.5.17）差异全部来自修 1（抽取层触发条件）/ 修 2（命名）/"
                        "修 3（抽取噪声判据）——判据（四项定义/停机线/匹配口径/分母/网络版本规则）逐字未改。",
                "parse_success_rate": q["rates"]["overall"]["parse_success_rate"],
                "parse_n": f"{q['rates']['overall']['parsed']}/{q['rates']['overall']['downloaded']}",
                "mapping_failure_rate_upper": qfd["上界（含碎片，主口径）"],
                "mapping_failure_n": qfd["上界分子分母"],
                "mapping_failure_ex_frag": qfd["剔碎片后"],
                "mapping_failure_ex_frag_n": qfd["剔碎片后分子分母"],
                "mapping_failure_ex_frag_noise": qfd.get("剔碎片与噪声后"),
                "mapping_failure_ex_frag_noise_n": qfd.get("剔碎片与噪声后分子分母"),
                "cls4": {"碎片": qfd["碎片"], "名称漂移": qfd["名称漂移"], "抽取噪声": qfd["抽取噪声"],
                         qres_key: qfd[qres_key]},
                "coverage_rate": q["rates"]["overall"]["coverage_rate"],
                "coverage_n": f"{q['rates']['overall']['n_covered']}/{q['rates']['overall']['cohort']}",
                "pooled_lift": qlp["lift"], "pooled_pair": f"{qlp['pair_edge']}/{qlp['pair_total']}",
                "pooled_overlap_share": qlp["overlap_share"],
                "external_anchor_satisfied": q["external_anchor"]["external_anchor_satisfied"],
                "per_year": {y: {"parse_success_rate": q["rates"]["per_year"][y]["parse_success_rate"],
                                 "coverage_rate": q["rates"]["per_year"][y]["coverage_rate"],
                                 "mapping_failure_rate": q["rates"]["failure_decomposition_per_year"][y]["上界（含碎片，主口径）"],
                                 "lift": q["anchor_readout"]["per_year"][y]["lift"]}
                             for y in map(str, years)},
            }
            # 修复件清单与新增同行对（修 1 定点证据；逐件 diff，只读）
            qf = {x["code"]: x for x in (q.get("per_firm") or [])}
            changed = []
            for r in recs:
                o = qf.get(r["code"])
                if not o:
                    continue
                if not r.get("located"):
                    continue
                op = set(o.get("peers") or [])
                np_ = set(r.get("peers") or [])
                ou = set(o.get("unmapped_names") or [])
                nu = set(r.get("unmapped_names") or [])
                ocode = set(o.get("mapped_peers_codes") or [])
                ncode = set(r.get("mapped_peers_codes") or [])
                if op != np_ or ou != nu or ocode != ncode:
                    changed.append({
                        "year": r["ipo_year"], "code": r["code"], "channel": r.get("channel"),
                        "n_unique_before_after": [o.get("n_unique_names"), r.get("n_unique_names")],
                        "n_mapped_before_after": [o.get("n_mapped"), r.get("n_mapped")],
                        "peers_added": sorted(np_ - op), "peers_removed": sorted(op - np_),
                        "mapped_codes_added": sorted(ncode - ocode), "mapped_codes_removed": sorted(ocode - ncode),
                        "unmapped_added": sorted(nu - ou)[:20], "unmapped_removed": sorted(ou - nu)[:20],
                        "r2_left_repairs": [f"{x['raw']}→{x['fixed']}" for x in (r.get("r2_left_repairs") or [])][:10],
                    })
            changed.sort(key=lambda x: (-len(x["peers_added"]), x["code"]))
            focus = {c["code"]: c for c in changed if c["code"] in ("300748", "300737")}
            repair2 = {
                "note": "修复轮二（修 1）修复件清单与新增同行对：与上一轮（registry 1.5.16）逐份 diff（只读）；"
                        "**语料未动（不重新下载 PDF）**，差异全部来自抽取层触发条件修复。",
                "n_files_changed": len(changed),
                "focus_cases": focus,
                "changed_files": changed[:60],
                "diff_source": rel(args.pre_fix_report2),
            }
        except Exception as e:      # 只读对照，失败不阻断本轮（报告注明缺失）
            pre2 = {"source": rel(args.pre_fix_report2), "error": f"{type(e).__name__}: {str(e)[:160]}"}

    result = {
        "report": "第二梯队 #9 路线 B 全量（2015–2023）招股书「同行业可比公司」自报锚（锚 1 / plan §8 B″11-7）",
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "parameter": "prospectus_peer_anchor_pilot（registry 1.5.17 全量口径 + 修复轮二（修 1–7）；"
                     "其中 H1/M1/M2 解析层修复轮 = registry 1.5.15/1.5.16；四项定义 / 停机线 / 匹配口径逐字沿用"
                     "试点登记值，**判据零改动**）",
        "fix_round": {
            "registry_version": "1.5.15",
            "change": "第二梯队 #9 全量「先修解析」修复轮：H1（年报头部名称注册表「简称+全称同头拼接」错切 + 标签边界切分 + "
                      "cache_version 3→4）/ M1（R2 标签集补证券代码·公司代码 + R2 左边界剥离 + R5 单字噪声判据 → 整名判据）/ "
                      "M2（发行人自名过滤容错：自名集合含代码锚定历史简称 + 同核/编辑距离·OCR 近似/近全包含 + universe 名录守卫）",
            "judgment_unchanged": "四项定义 / 停机线（0.80 / 0.20 / 0.50 逐字不变）/ 匹配口径（精确 + 仅公司后缀归一化 + "
                                  "代码锚定，无模糊匹配）/ 覆盖率分母（cohort 2,656）/ 网络版本规则（min(Y+2,2023) + fallback v1_2022）/ "
                                  "边界年处理；**无人工补录、未放宽匹配、未调分母、未剔除失败件**",
            "attribution_additive": "归因口径增设第 4 类「抽取噪声」（判据 = GEN_SUB 词表；作用位置 = 原「非 A 股实体」桶内细分提为顶层类），"
                                    "另增列「剔碎片与噪声后」；主口径失败率（上界·含碎片）与停机线判定口径不变",
        },
        "fix_round2": {
            "registry_version": "1.5.17",
            "change": "第二梯队 #9 全量第二轮修复轮（修 1–7）：① 修 1 抽取层触发条件（R4 补「可比公司」类复合触发 + "
                      "和/与/及 并列切分 + R2 括号内空白容错 + R2 左边界语境后缀片段/尾随「有」（名录守卫）剥离 + "
                      "新增「可比公司表头列」型 R4_tbl_head；R1 触发条件复核后不改）；② 修 2 unmapped 第 4 类改名"
                      "「残差桶（未过前三类判据）」+ 抽样审计（n=160）与复核量化分解入报告；③ 修 3 GEN_SUB 判据改"
                      "「整名 ≈ 通用词」+ 公司后缀豁免；④ 修 4 first_hit_page 非 pymupdf 通道置 not_applicable_channel；"
                      "⑤ 修 5 报告头 registry 版本改实际登记版本 + short_keys 净减 1,297 登记；⑥ 修 6 网络节点两口径"
                      "并报；⑦ 修 7 恒等式校验改名（非独立证据）",
            "judgment_unchanged": "四项定义 / 停机线（0.80 / 0.20 / 0.50 逐字不变）/ 匹配口径（精确 + 仅公司后缀归一化 + "
                                  "代码锚定，无模糊匹配）/ 覆盖率分母（cohort 2,656）/ 网络版本规则（min(Y+2,2023) + "
                                  "fallback v1_2022）/ 边界年处理；**无人工补录、未放宽匹配、未调分母、未剔除失败件**",
            "re_extract_scope": "用既有 txt / ocr_txt / CID 缓存**重抽**（不重新下载 PDF）；修复件清单与新增同行对见 "
                                "`repair_round2_evidence`",
        },
        "sensitivity_pre_fix": pre,
        "sensitivity_pre_round2": pre2,
        "repair_round2_evidence": repair2,
        "scope": {
            "ipo_years": years, "cohort_n": cohort_all, "sample_scope": "全量（不抽样）",
            "universe": "data/processed/firm_id_map.csv（项目 firm universe）",
            "download_dir": rel(os.path.join(args.base)),
            "network_version_map": {str(y): {"text_year": verb[y][0], "version": verb[y][1],
                                             "boundary_year_subset": verb[y][2]} for y in years},
            "keyword_set": list(KWS), "window": [WIN_PRE, WIN_POST],
            "extraction_rules": ["R1_full", "R2_name_code", "R3_code_name", "R4_enum", "R4_tbl_head",
                                 "R5_table_row"],
            "r1_trigger_recheck": "修 1(g)：R1（全称型）触发词与近邻窗口本轮**已复核、无改动**"
                                  "（复核点名两件的漏抽由 R4 复合触发 / R4_tbl_head / R2 空白容错 / R2 左边界扩展覆盖）",
            "mapping_modes": ["exact_short", "exact_full", "suffix_norm", "adjacent_code"],
            "name_dict": {"firm_universe": len(firm), "short_keys": len(short), "full_keys": len(full)},
            "name_dict_short_keys_note": "修 5：`name_dict.short_keys` 净减 1,297（H1 错切键移除，13,412→12,115；"
                                         "键数下降来自错切键与头部标签残片键移除，**非**人工删除）",
            "channels": "pymupdf → pdfminer+SimSun GID（CID 编码损坏件，确定性还原）→ OCR 回退"
                        "（tesseract 5.3.4 chi_sim，--psm 3、200 DPI）；不足判据 = 公司<20 或 字符数<max(5000,50×页数)",
            "parse_layer_fix": "registry 1.5.15「先修解析」修复轮（H1 名称注册表切分 + M1 抽取规则 + M2 自名容错）；"
                               "locate/通道/匹配口径/判据与修复前逐字一致",
        },
        "rates": {"overall": overall, "thresholds": th, "stop_triggered": bool(stop),
                  "stop_reasons": stop,
                  "failure_decomposition_pooled": fail_all,
                  "failure_decomposition_main_years": fail_main,
                  "failure_decomposition_boundary": fail_bnd,
                  "failure_decomposition_per_year": fail_year,
                  "per_year": per_year},
        "anchor_readout": {
            "definition": "lift = 重合率 ÷ 该网络年无偏对密度 d = E/(N(N−1)/2)；池化 lift = Σ命中对 ÷ Σ(该年对数 × d_年) "
                          "（= 密度加权平均 lift，等价于池化重合率 ÷ 池化密度）",
            "per_year": {str(y): {k: per_year_lift[str(y)][k] for k in
                                  ("network_version_used", "network_text_year", "pair_total", "pair_edge",
                                   "overlap_share", "overlap_share_ci95", "lift", "lift_ci95",
                                   "diag_pair_total", "diag_pair_edge", "diag_overlap_share", "diag_lift",
                                   "n_network_nodes", "network")} for y in years},
            "pooled_main_years_2015_2020": pooled_main,
            "pooled_including_boundary": pooled_all,
            "boundary_subset": boundary,
            "per_firm_overlap_stats": {
                "n_firms_in_network_with_peers": sum(1 for r in recs if r.get("network_node")
                                                    and r.get("overlap_share") is not None),
                "n_zero_overlap": sum(1 for r in recs if r.get("network_node")
                                      and r.get("overlap_share") == 0),
                "n_ge_50pct": sum(1 for r in recs if r.get("network_node")
                                  and (r.get("overlap_share") or 0) >= 0.5),
                "mean": (sum(r["overlap_share"] for r in recs if r.get("overlap_share") is not None)
                         / max(1, sum(1 for r in recs if r.get("overlap_share") is not None))),
            },
            "direction_check": {
                "note": "方向性检查（全量）：被 R3/R2 形态约束或 R1/R2 交叉校验剔除的名-码对（双方均在该网络年节点集内）"
                        "落网率 vs 保留集诊断口径 — 若二者同量级，说明剔除会改变（而非仅压低）重合率 → lift 方向未证。",
                "rejected_pairs_in_network": len(rej), "rejected_pairs_edge": rej_hit,
                "rejected_pairs_edge_rate": (rej_hit / len(rej)) if rej else None,
                "kept_pairs_edge_rate_diag": pooled_main.get("diag_overlap_share"),
                "rejected_examples": sorted({f"{k[0]}:{k[1]}↔{k[2]}（{v}）" for k, v in rej.items()})[:15],
            },
        },
        "external_anchor": external,
        "failure_diagnostics": failure_diag,
        "fix_round_evidence": fix_evidence,
        "ingest_crosscheck": cross,
        "pilot_2018_contrast": pilot_cmp,
        "download_failures": dl,
        "channel_disclosure": chan,
        "known_limitations": known_limitations,
        "per_firm": recs,
    }

    out = args.out or os.path.join(ROOT, "reports", "data_gates", "G0",
                                   "s3_network_prospectus_anchor.json")
    out_md = args.out_md or out.replace(".json", ".md")
    detail_csv = args.detail_csv or out.replace(".json", "_detail.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    def _atomic_write(path, writer):
        """原子写（先写 .tmp 再 os.replace）：防并发/中断造成的半写文件。"""
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    _atomic_write(out, lambda f: json.dump(result, f, ensure_ascii=False, indent=1))
    json.load(open(out, encoding="utf-8"))      # 写后回读校验（防半写/编码损坏）
    with open(detail_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "ipo_year", "status", "channel", "located", "page_count", "char_count",
                    "n_kw_hits", "first_hit_page", "n_unique_names", "n_mapped", "n_unmapped",
                    "network_version_used", "network_text_year", "boundary_year_subset",
                    "network_node", "n_pairs", "n_pairs_edge", "overlap_share",
                    "peers", "mapped_peers_codes", "unmapped"])
        for r in recs:
            w.writerow([r["code"], r["ipo_year"], r["status"], r.get("channel", ""), r.get("located", ""),
                        r.get("page_count", ""), r.get("char_count", ""), r.get("n_kw_hits", ""),
                        r.get("first_hit_page", ""), r.get("n_unique_names", ""), r.get("n_mapped", ""),
                        r.get("n_unmapped", ""), r["network_version_used"], r["network_text_year"],
                        r["boundary_year_subset"], r.get("network_node", ""), r.get("n_pairs", ""),
                        r.get("n_pairs_edge", ""), r.get("overlap_share", ""),
                        "|".join(r.get("peers") or []), "|".join(r.get("mapped_peers_codes") or []),
                        "|".join(r.get("unmapped_names") or [])])
    write_anchor_md(result, out_md)
    with open(out.replace(".json", "_measure.log"), "w", encoding="utf-8") as f:
        f.write(json.dumps({k: result[k] for k in ("run_at", "rates", "anchor_readout",
                                                   "external_anchor", "scope")},
                           ensure_ascii=False, indent=1))
    print(f"[M-3] 报告 → {rel(out)} / {rel(out_md)}；逐份明细 → {rel(detail_csv)}", flush=True)
    print(f"[M-3] 解析 {overall['parse_success_rate']:.4f}（{overall['parsed']}/{overall['downloaded']}）"
          f"/ 映射失败 {fail_all['上界（含碎片，主口径）']:.4f}（{fail_all['上界分子分母']}）"
          f"/ 覆盖 {overall['coverage_rate']:.4f}（{overall['n_covered']}/{cohort_all}）", flush=True)
    print(f"[M-3] 池化 lift（主读数年 {main_years}）= {pooled_main['lift']}；external_anchor_satisfied = "
          f"{external['external_anchor_satisfied']}", flush=True)
    return result

