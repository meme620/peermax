#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CNINFO 招股说明书下载与文本提取（第二梯队 #9 路线 A 试点；B″11-7 锚 1）。

- 查询: www.cninfo.com.cn/new/hisAnnouncement/query
  **类别码 = `category_sf_szsh`（首发）**——2026-10-13 探针实测确认（`artifacts/tmp/tier2_prosp/probe_cat3.py`）：
  以 601138 为例，该类别返回「首次公开发行A股股票招股说明书 / 招股说明书摘要 / 招股意向书摘要 /
  上市公告书 / 发行公告」等首发类公告；`category_zf_szsh`（增发）返回 0 条，`category_scgkfx_szsh` /
  `category_zb_szsh` 为无效码（返回全量、无过滤）。股票参数沿用 `fetch_cninfo_annual.py` 的既有做法
  （官方 orgId 优先，缺失回退 gss{z|h}+补零规则；两者都过身份校验）。
- 标题选择: 必须含「招股说明书」或「招股意向书」（回退）；排除 摘要/英文/更正/补充/说明/上市公告书/
  发行公告/路演/投资风险；评分优先「招股说明书」正文（非 申报稿/上会稿/注册稿）、同分取披露日期最新。
- 下载: static.cninfo.com.cn/{adjunctUrl}（无代理直连，同 `fetch_cninfo_annual.py`）。
- 校验: 落盘前 ① 公告 `secCode == 目标代码`；② 披露日期落在 [ipo_date−400d, ipo_date+45d] 窗口内
  （并记录 pub_year）；③ 正文前 3 页黄金代码/公司名校验（沿用年报口径）。任一不符 → 试下一候选，
  全不符则不落盘。
- 存储: data/raw/cninfo_prospectus/{ipo_year}/pdf/{code6}.pdf + .../txt/{code6}.txt
  + 追加式日志 .../download_log.csv + 汇总 .../download_summary.json
- 断点续传: txt 存在且 >5000B 跳过（视为已完成）；凭 download_log.csv 跟踪逐代码结果。
  全量轮（2026-10-14，registry 1.5.13）：文本层不足（扫描件）**不再弃件** → status=ok_no_text_layer
  （留存 PDF + txt，交解析层 OCR 通道）；log 末次 status ∈ {ok, ok_no_text_layer, skip_exists}
  且 pdf+txt 在场 → 跳过（`load_done_codes`）。
用法: python scripts/data/fetch_cninfo_prospectus.py --ipo-year 2018
      python scripts/data/fetch_cninfo_prospectus.py --ipo-year 2018 --max-codes 5 --workers 2
      python scripts/data/fetch_cninfo_prospectus.py --ipo-year 2018 --codes 601138,300750
"""
import argparse
import csv
import json
import os
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pymupdf

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
BASE = os.path.join(ROOT, "data", "raw", "cninfo_prospectus")
FIRM_MAP = os.path.join(ROOT, "data", "processed", "firm_id_map.csv")
A_STOCKS = os.path.join(ROOT, "data", "raw", "baostock", "stock_basic", "a_stocks.csv")
ORG_LIST = os.path.join(ROOT, "data", "raw", "cninfo", "cninfo_stock_list.json")

CATEGORY = "category_sf_szsh"          # 首发（2026-10-13 探针确认）
H = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/x-www-form-urlencoded",
     "Referer": "http://www.cninfo.com.cn/new/index"}
EXCLUDE_TOKENS = ("摘要", "英文", "更正", "补充", "上市公告书", "发行公告", "路演",
                  "投资风险", "中签", "配售", "询价", "发行结果", "发行安排", "问询", "反馈")
# 标题必须**以**「招股说明书 / 招股意向书」结尾（允许尾随括号注释）：2026-10-13 实测
# 「控股股东、实际控制人对招股说明书的确认意见」等标题也含关键词 → 必须用结尾判据排除。
TITLE_TAIL_RE = re.compile(r"(招股说明书|招股意向书)(（[^）]*）|\([^)]*\))?$")
DRAFT_TOKENS = ("申报稿", "上会稿", "注册稿", "草稿", "预披露")
CODE_RE = re.compile(r"(?:证券代码|股票代码|公司代码|证券简称及代码|股票简称及代码)[：:\s]*([0-9]{6})")
CODE_PAIR_RE = re.compile(r"([0-9]{6})\s*[,、]\s*g[sz][sh][0-9]{7}")
VALID_PREFIX = ("000", "001", "002", "003", "300", "301", "302",
                "600", "601", "603", "605", "688", "689")

_ORG_IDS = {}
_NAMES = {}
_LOCK = threading.Lock()
# 全量轮（2026-10-14，registry 1.5.13 预登记）：断点续传扩展——log 末次 status ∈ 下列集合且
# pdf+txt 均在场的 code 视为完成（用于无文本层扫描件：其 txt 可能 <5000B，旧 skip 规则会重复下载）。
_DONE_STATUS = ("ok", "ok_no_text_layer", "skip_exists")
_DONE_CODES = set()


def load_org_ids():
    global _ORG_IDS
    if _ORG_IDS:
        return _ORG_IDS
    if os.path.exists(ORG_LIST):
        try:
            with open(ORG_LIST, encoding="utf-8") as f:
                _ORG_IDS = {str(x["code"]): x.get("orgId") for x in json.load(f)}
        except Exception:
            _ORG_IDS = {}
    return _ORG_IDS


def load_names():
    """code6 -> 名称候选（firm_id_map 简称 + a_stocks 简称），供身份校验。"""
    global _NAMES
    if _NAMES:
        return _NAMES
    names = {}
    for path, code_col, name_col in ((FIRM_MAP, 1, 2), (A_STOCKS, 0, 1)):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for r in csv.reader(f):
                if len(r) <= max(code_col, name_col):
                    continue
                if r[code_col] == "code6" or r[code_col] == "code":
                    continue
                c = re.sub(r"^(sh|sz)\.?", "", r[code_col].strip())
                if len(c) == 6 and c.isdigit():
                    names.setdefault(c, [])
                    nm = re.sub(r"[ \u3000]", "", r[name_col])
                    if nm and nm not in names[c]:
                        names[c].append(nm)
    _NAMES = names
    return _NAMES


def org_id(code6):
    if code6.startswith(("600", "601", "603", "605", "688", "689")):
        return "sse"
    if code6.startswith(("000", "001", "002", "003", "300", "301", "302")):
        return "szse"
    return None


def stock_param(code6):
    org = load_org_ids().get(code6)
    if org:
        return f"{code6},{org}"
    if code6.startswith(("600", "601", "603", "605", "688", "689")):
        return f"{code6},gssh{code6.zfill(7)}"
    return f"{code6},gssz{code6.zfill(7)}"


def _query(body):
    req = urllib.request.Request("http://www.cninfo.com.cn/new/hisAnnouncement/query",
                                 headers=H, data=body.encode())
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8", "ignore"))


def query_all(code6, column, se_date, max_pages=8, sleep=0.4):
    """分页取全部候选公告（API 单页上限 30，回退 searchkey 时同样分页）。

    2026-10-13 实测：`pageSize=50` 服务端只回 30 条，且「招股说明书/招股意向书」按披露日期
    排序落在**列表末页**（IPO 最早披露）——必须按 `totalAnnouncement` 翻页到底，否则漏掉正文。
    """
    base = ("pageSize=30&tabName=fulltext&plate=&secid=&trade=&sortName=&sortType=&isHLtitle=true"
            f"&column={column}&category={CATEGORY}&stock={urllib.parse.quote(stock_param(code6))}")
    anns, total = [], None
    for pg in range(1, max_pages + 1):
        d = _query(f"pageNum={pg}&{base}&seDate={se_date}")
        if total is None:
            total = d.get("totalAnnouncement") or 0
        got = d.get("announcements") or []
        anns += got
        if len(anns) >= total or not got:
            break
        time.sleep(sleep)
    # 兜底：stock 参数无结果时按代码全文检索（仍须过身份校验）
    if not any(a.get("secCode") == code6 for a in anns):
        d = _query(f"pageNum=1&pageSize=30&tabName=fulltext&plate=&secid=&trade=&sortName=&sortType="
                   f"&isHLtitle=true&column={column}&category={CATEGORY}&searchkey={code6}&seDate={se_date}")
        anns += (d.get("announcements") or [])
    return anns, total


def pub_date(a):
    t = a.get("announcementTime") or 0
    try:
        return time.strftime("%Y-%m-%d", time.gmtime(int(t) / 1000))
    except Exception:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", a.get("adjunctUrl", "") or "")
        return m.group(1) if m else ""


def pick_targets(anns, code6):
    """返回 [(score, announcement, title)]，分数高者优先（正文型「招股说明书」最优）。"""
    out = []
    for a in anns:
        if a.get("secCode") not in (None, "", code6):
            continue
        t = re.sub(r"<[^>]+>", "", a.get("announcementTitle", "")).replace(" ", "").replace("\n", "")
        if not TITLE_TAIL_RE.search(t):
            continue
        if any(b in t for b in EXCLUDE_TOKENS):
            continue
        s = 0
        if "招股说明书" in t:
            s += 100
        if any(b in t for b in DRAFT_TOKENS):
            s -= 60
        if "更新" in t or "修订" in t:
            s -= 10
        out.append((s, a, t))
    # 优先级：正文型「招股说明书」（score 100）> 招股意向书（0）；同分取披露日期最新
    out.sort(key=lambda x: (x[0], pub_date(x[1])), reverse=True)
    return out


def extract_pdf(pdf_path, txt_path):
    doc = pymupdf.open(pdf_path)
    full = "".join(page.get_text() for page in doc)
    doc.close()
    if not full.strip():
        return False
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full)
    return True


def verify_identity(pdf_path, code6):
    """前 3 页黄金校验：(True,'code'|'name') / (False, seen) / (None,None)。"""
    doc = pymupdf.open(pdf_path)
    head = re.sub(r"[ \u3000\t\r\n]", "", "".join(pg.get_text() for pg in doc[:3]))
    doc.close()
    codes = {m.group(1) for m in CODE_RE.finditer(head)} | {m.group(1) for m in CODE_PAIR_RE.finditer(head)}
    codes |= set(re.findall(r"(?<![0-9])([0-9]{6})(?![0-9])", head))
    codes = {c for c in codes if c.startswith(VALID_PREFIX)}
    if code6 in codes:
        return True, "code"
    for nm in load_names().get(code6, []):
        if len(nm) >= 3 and nm in head:
            return True, "name"
    if not codes:
        return None, None
    return False, ",".join(sorted(codes)[:3])


def load_done_codes(logp):
    """从既有 download_log.csv 读每个 code 的**末次** status；ok / ok_no_text_layer / skip_exists → 完成。"""
    last = {}
    if os.path.exists(logp):
        with open(logp, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("code"):
                    last[r["code"]] = r.get("status")
    return {c for c, s in last.items() if s in _DONE_STATUS}


def download_one(code6, ipo_date, ipo_year, verify=True):
    column = org_id(code6)
    if column is None:
        return {"code": code6, "status": "unsupported_code"}
    idir = os.path.join(BASE, str(ipo_year))
    pdir, tdir = os.path.join(idir, "pdf"), os.path.join(idir, "txt")
    os.makedirs(pdir, exist_ok=True)
    os.makedirs(tdir, exist_ok=True)
    pdfp, txtp = os.path.join(pdir, f"{code6}.pdf"), os.path.join(tdir, f"{code6}.txt")
    tmp_pdf, tmp_txt = pdfp + ".tmp", txtp + ".tmp"
    try:
        if os.path.exists(txtp) and os.path.getsize(txtp) > 5000:
            return {"code": code6, "status": "skip_exists", "pdf_bytes": os.path.getsize(pdfp) if os.path.exists(pdfp) else 0}
        if code6 in _DONE_CODES and os.path.exists(pdfp) and os.path.exists(txtp):
            # 无文本层扫描件等：log 已记 ok/ok_no_text_layer 且 pdf+txt 在场 → 视为完成（全量轮登记）
            return {"code": code6, "status": "skip_exists", "pdf_bytes": os.path.getsize(pdfp)}
        lo = (time.strftime("%Y-%m-%d", time.localtime(time.mktime(time.strptime(ipo_date, "%Y-%m-%d")) - 400 * 86400))
              if ipo_date else f"{ipo_year - 1}-01-01")
        hi = (time.strftime("%Y-%m-%d", time.localtime(time.mktime(time.strptime(ipo_date, "%Y-%m-%d")) + 45 * 86400))
              if ipo_date else f"{ipo_year}-12-31")
        anns, total = query_all(code6, column, f"{lo}~{hi}")
        cands = pick_targets(anns, code6)
        if not cands:
            return {"code": code6, "status": "no_prospectus", "n_ann": len(anns), "total": total}
        tried = []
        for score, a, title in cands[:4]:
            pd_ = pub_date(a)
            if ipo_date and not (lo <= pd_ <= hi):
                tried.append(f"{title[:30]}:pub_out_of_window={pd_}")
                continue
            url = "http://static.cninfo.com.cn/" + (a.get("adjunctUrl") or "")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(req, timeout=180) as r:
                    pdf = r.read()
            except Exception as e:
                tried.append(f"{title[:30]}:dl_err:{str(e)[:40]}")
                continue
            if pdf[:4] != b"%PDF":
                tried.append(f"{title[:30]}:not_pdf")
                continue
            with open(tmp_pdf, "wb") as f:
                f.write(pdf)
            if verify:
                ok, seen = verify_identity(tmp_pdf, code6)
                if ok is False:
                    tried.append(f"{title[:30]}:identity_{seen}")
                    continue
            # 全量轮（registry 1.5.13 预登记）：文本层不足（扫描件）**不再弃件**——留存 PDF + txt（可空），
            # status=ok_no_text_layer，交解析层 OCR 通道（tesseract）覆盖；正常件逻辑不变。
            txt_ok = extract_pdf(tmp_pdf, tmp_txt)
            if not txt_ok:
                with open(tmp_txt, "w", encoding="utf-8"):
                    pass
            tsz = os.path.getsize(tmp_txt) if os.path.exists(tmp_txt) else 0
            status = "ok" if tsz >= 5000 else "ok_no_text_layer"
            os.replace(tmp_pdf, pdfp)
            os.replace(tmp_txt, txtp)
            return {"code": code6, "status": status, "title": title, "publish_date": pd_,
                    "pub_year": int(pd_[:4]) if pd_ else None, "score": score,
                    "pdf_bytes": os.path.getsize(pdfp), "txt_bytes": os.path.getsize(txtp),
                    "adjunct": a.get("adjunctUrl"), "n_ann": len(anns)}
        return {"code": code6, "status": "all_candidates_failed", "n_ann": len(anns),
                "tried": "|".join(tried[:3])}
    except Exception as e:
        return {"code": code6, "status": f"err:{str(e)[:80]}"}
    finally:
        for p in (tmp_pdf, tmp_txt):
            if os.path.exists(p):
                os.remove(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipo-year", type=int, default=2018)
    ap.add_argument("--codes", default="", help="逗号分隔代码（缺省=该年 IPO 且在本项目 firm 名单内的全部公司）")
    ap.add_argument("--max-codes", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--progress-every", type=int, default=20)
    args = ap.parse_args()

    plan = []
    if args.codes:
        for c in args.codes.split(","):
            c = c.strip()
            if c:
                plan.append((c, ""))
    else:
        with open(FIRM_MAP, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["ipo_date"][:4] == str(args.ipo_year):
                    plan.append((r["code6"], r["ipo_date"]))
    if args.max_codes:
        plan = plan[:args.max_codes]
    idir = os.path.join(BASE, str(args.ipo_year))
    os.makedirs(idir, exist_ok=True)
    logp = os.path.join(idir, "download_log.csv")
    global _DONE_CODES
    _DONE_CODES = load_done_codes(logp)
    print(f"[P-1] 计划 {len(plan)} 份招股说明书（ipo_year={args.ipo_year}，workers={args.workers}，"
          f"category={CATEGORY}；既有 log 完成码 {len(_DONE_CODES)}）", flush=True)

    done, n_ok, n_skip = 0, 0, 0
    lock = threading.Lock()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, c, d, args.ipo_year, not args.no_verify): c for c, d in plan}
        for fu in as_completed(futs):
            res = fu.result()
            with lock:
                done += 1
                if res.get("status") in ("ok", "ok_no_text_layer"):
                    n_ok += 1
                elif res.get("status") == "skip_exists":
                    n_skip += 1
                with open(logp, "a", encoding="utf-8", newline="") as f:
                    row = {"code": res.get("code"), "status": res.get("status"),
                           "title": res.get("title", ""), "publish_date": res.get("publish_date", ""),
                           "pdf_bytes": res.get("pdf_bytes", ""), "txt_bytes": res.get("txt_bytes", ""),
                           "n_ann": res.get("n_ann", ""), "tried": res.get("tried", ""),
                           "run_at": time.strftime("%Y-%m-%d %H:%M:%S")}
                    w = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if not os.path.exists(logp) or os.path.getsize(logp) == 0:
                        w.writeheader()
                    elif f.tell() == 0:
                        w.writeheader()
                    w.writerow(row)
                if done % args.progress_every == 0 or done == len(plan):
                    print(f"[P-1] 进度 {done}/{len(plan)}（ok={n_ok} skip={n_skip} "
                          f"elapsed={time.time() - t0:.0f}s）", flush=True)
    # 汇总
    counts = {}
    with open(logp, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            counts[r["status"]] = counts.get(r["status"], 0) + 1
    summary = {"ipo_year": args.ipo_year, "planned": len(plan), "ran": done, "ok_new": n_ok,
               "skip_exists": n_skip, "log_counts": counts, "pdf_dir": os.path.relpath(os.path.join(idir, 'pdf'), ROOT),
               "txt_dir": os.path.relpath(os.path.join(idir, 'txt'), ROOT),
               "n_pdf_files": len([x for x in os.listdir(os.path.join(idir, "pdf")) if x.endswith(".pdf")])
               if os.path.isdir(os.path.join(idir, "pdf")) else 0,
               "n_txt_files": len([x for x in os.listdir(os.path.join(idir, "txt")) if x.endswith(".txt")])
               if os.path.isdir(os.path.join(idir, "txt")) else 0,
               "run_at": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_sec": round(time.time() - t0, 1)}
    with open(os.path.join(idir, "download_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"[P-1] 完成 {done}/{len(plan)}：ok_new={n_ok} skip={n_skip} log={counts} "
          f"pdf={summary['n_pdf_files']} txt={summary['n_txt_files']}", flush=True)


if __name__ == "__main__":
    main()
