"""s3_network_prospectus_anchor 的 ingest 模块（逐字搬移，逻辑未改）。"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data_audit'))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data'))

from lib.prospectus.consts import FIRM_MAP
from lib.prospectus.consts import ROOT
from lib.prospectus.text import _insufficient
from lib.prospectus.text import locate
import csv
import json
import os
import re
import time

def resolve_network_version(year, net_dir):
    """逐 IPO 年 Y → 网络版本（registry 1.5.13）：y_target = min(Y+2, 2023)；
    版本文件缺失 → 以「最近可用网络版本」替代（实测 v1_2022；boundary=True 记边界年子集）。"""
    avail = sorted(int(m.group(1)) for m in
                   (re.match(r"(\d{4})_support\.csv$", x) for x in os.listdir(net_dir)) if m)
    yt = min(year + 2, 2023)
    if yt in avail:
        return yt, f"pm_ma_ewcos_teacher_v1_{yt}", False
    ynew = max(avail)
    return ynew, f"pm_ma_ewcos_teacher_v1_{ynew}", True

def _ingest_one(job):
    """逐份解析（仅解析层）：locate 三通道回退 + 逐份 channel/页数/字数/关键词记录。
    返回 dict（全量轮 ingest 明细行）；不静默剔除——失败逐条带 reason。"""
    code, year, net_ver, net_text_year, boundary, pdfd, txtd = job
    pdfp, txtp = os.path.join(pdfd, f"{code}.pdf"), os.path.join(txtd, f"{code}.txt")
    rec = {"code": code, "ipo_year": year, "network_version_used": net_ver,
           "network_text_year": net_text_year, "boundary_year_subset": boundary}
    if not (os.path.exists(pdfp) and os.path.exists(txtp)):
        rec.update(status="no_file", channel="no_file", located=False, page_count=0, char_count=0,
                   n_kw_hits=0, first_hit_page="", n_spans=0, pdf_bytes=0, txt_bytes=0,
                   reason="pdf 或 txt 缺失（下载未成功，详见 download_log.csv）")
        return rec
    t0 = time.time()
    rec.update(pdf_bytes=os.path.getsize(pdfp), txt_bytes=os.path.getsize(txtp))
    try:
        (located, n_hits, counts, first_page, n_pages, spans, text,
         offsets, channel) = locate(txtp, pdfp, code=code)
    except Exception as e:
        rec.update(status="parse_error", channel=f"error:{type(e).__name__}", located=False,
                   page_count=0, char_count=0, n_kw_hits=0, first_hit_page="", n_spans=0,
                   reason=f"locate 异常：{str(e)[:120]}")
        return rec
    insuf = _insufficient(text, n_pages)
    rec.update(status="parsed" if located else "no_section", channel=channel, located=located,
               page_count=n_pages, char_count=len(text), n_kw_hits=n_hits,
               kw_counts=json.dumps(counts, ensure_ascii=False),
               first_hit_page=first_page if first_page else "", n_spans=len(spans),
               text_insufficient=insuf, parse_sec=round(time.time() - t0, 1))
    if not located:
        if insuf:
            rec["reason"] = ("文本层缺失/不足（CID/OCR 回退后仍不足；"
                             f"company_count={text.count('公司')}, chars={len(text)}, pages={n_pages}）")
        else:
            rec["reason"] = f"章节未定位（4 关键词全 0；chars={len(text)}, pages={n_pages}）"
    return rec

def ingest_main(args):
    """全量 ingest：逐年 cohort → 并行 locate（含 OCR 回退）→ 明细 CSV + 汇总 JSON。"""
    years = []
    for part in args.years.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            years += list(range(int(a), int(b) + 1))
        elif part:
            years.append(int(part))
    years = sorted(set(years))
    firm = {}
    with open(FIRM_MAP, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            firm[r["code6"]] = r["ipo_date"]
    verb = {y: resolve_network_version(y, args.net_dir) for y in years}
    print(f"[F-3] 全量 ingest：years={years}；workers={args.workers}", flush=True)
    for y in years:
        ty, ver, bd = verb[y]
        print(f"[F-3]   {y}: cohort={sum(1 for c, d in firm.items() if d[:4] == str(y))}；"
              f"network_version_used={ver}（text_year={ty}；boundary={bd}）", flush=True)

    jobs = []
    for y in years:
        ty, ver, bd = verb[y]
        pdfd = os.path.join(args.base, str(y), "pdf")
        txtd = os.path.join(args.base, str(y), "txt")
        for c, d in sorted(firm.items()):
            if d[:4] == str(y):
                jobs.append((c, y, ver, ty, bd, pdfd, txtd))
    print(f"[F-3] 计划 {len(jobs)} 份；每 {args.progress_every} 份打印进度", flush=True)

    recs, t0 = [], time.time()
    if args.workers <= 1:
        it = ((j, _ingest_one(j)) for j in jobs)
        for i, (_, r) in enumerate(it, 1):
            recs.append(r)
            if i % args.progress_every == 0 or i == len(jobs):
                print(f"[F-3] {i}/{len(jobs)}（parsed={sum(1 for x in recs if x['status'] == 'parsed')} "
                      f"elapsed={time.time() - t0:.0f}s）", flush=True)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_ingest_one, jobs, chunksize=1), 1):
                recs.append(r)
                if i % args.progress_every == 0 or i == len(jobs):
                    print(f"[F-3] {i}/{len(jobs)}（parsed="
                          f"{sum(1 for x in recs if x['status'] == 'parsed')} "
                          f"elapsed={time.time() - t0:.0f}s）", flush=True)

    outdir = args.ingest_outdir or os.path.join(ROOT, "artifacts", "tmp", "tier2_full")
    os.makedirs(outdir, exist_ok=True)
    detail_csv = args.detail_csv or os.path.join(outdir, "ingest_detail.csv")
    fields = ["code", "ipo_year", "status", "channel", "located", "page_count", "char_count",
              "n_kw_hits", "kw_counts", "first_hit_page", "n_spans", "text_insufficient",
              "pdf_bytes", "txt_bytes", "parse_sec", "network_version_used", "network_text_year",
              "boundary_year_subset", "reason"]
    with open(detail_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(recs, key=lambda x: (x["ipo_year"], x["code"])):
            w.writerow(r)
    # 汇总
    summary = {"years": years, "n_planned": len(jobs), "workers": args.workers,
               "per_year": {}, "channel_dist": {}, "status_dist": {},
               "network_version_map": {str(y): {"text_year": verb[y][0], "version": verb[y][1],
                                                "boundary_year_subset": verb[y][2]} for y in years},
               "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "elapsed_sec": round(time.time() - t0, 1)}
    ch_key = lambda c: ("pymupdf" if c == "pymupdf" else
                        "pdfminer+cid_gid→ocr" if (c.startswith("pdfminer+cid_gid")
                                                   and "ocr_tesseract_chi_sim" in c) else
                        "pdfminer+cid_gid" if c.startswith("pdfminer+cid_gid") else
                        "ocr_tesseract_chi_sim" if "ocr_tesseract_chi_sim" in c else
                        "ocr_insufficient" if "ocr_insufficient" in c else
                        "text_layer_invalid" if "text_layer_invalid" in c else
                        "no_file" if c == "no_file" else "other")
    for y in years:
        rs = [r for r in recs if r["ipo_year"] == y]
        n_dl = sum(1 for r in rs if r["status"] != "no_file")
        n_parsed = sum(1 for r in rs if r["status"] == "parsed")
        n_nofile = sum(1 for r in rs if r["status"] == "no_file")
        n_err = sum(1 for r in rs if r["status"] == "parse_error")
        ch = {}
        for r in rs:
            k = ch_key(r.get("channel", ""))
            ch[k] = ch.get(k, 0) + 1
        summary["per_year"][str(y)] = {"cohort": len(rs), "downloaded": n_dl, "parsed": n_parsed,
                                       "no_section": sum(1 for r in rs if r["status"] == "no_section"),
                                       "no_file": n_nofile, "parse_error": n_err,
                                       "parse_success_rate": round(n_parsed / n_dl, 4) if n_dl else None,
                                       "channels": ch,
                                       "boundary_year_subset": verb[y][2]}
        summary["status_dist"][str(y)] = {s: sum(1 for r in rs if r["status"] == s)
                                          for s in sorted({r["status"] for r in rs})}
        for k, v in ch.items():
            summary["channel_dist"][k] = summary["channel_dist"].get(k, 0) + v
    tot_dl = sum(v["downloaded"] for v in summary["per_year"].values())
    tot_parsed = sum(v["parsed"] for v in summary["per_year"].values())
    summary["overall"] = {"downloaded": tot_dl, "parsed": tot_parsed,
                          "parse_success_rate": round(tot_parsed / tot_dl, 4) if tot_dl else None}
    with open(os.path.join(outdir, "ingest_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"[F-3] 完成 {len(recs)}/{len(jobs)}：parsed={tot_parsed}/{tot_dl}"
          f"（{summary['overall']['parse_success_rate']}）channel={summary['channel_dist']} "
          f"detail={os.path.relpath(detail_csv, ROOT)}", flush=True)
    return summary

