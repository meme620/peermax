#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CNINFO 年报下载与文本提取（D 类：network_edges 自建 pipeline 第一步）。

- 查询: www.cninfo.com.cn/new/hisAnnouncement/query
  **stock=<code>,gss{z|h}<7位补零> + column=sse/szse**（2026-09-11 修复：
  searchkey=<code> 全文搜索会返回他司公告，是 4.5% 内容错位的根因，见
  reports/data_gates/G0/mda_text_integrity_report_20260911.md；
  searchkey 仅作 stock 无结果时的兜底，且必须过身份校验）。
- 标题匹配: 必须含 "{y}年年度报告" 且以之结尾（允许"（更新后）"后缀），
  排除 摘要/英文/公告/更正/补充/说明/问询/反馈/进展/提示/延期/意见 等非正文标题。
- 下载: static.cninfo.com.cn/{adjunctUrl}
- 提取: pymupdf（pdftotext 对 CNINFO 年报输出乱码——2026-08-09 实测，弃用）
- 校验: 落盘前黄金代码校验（正文前 3000 字证券代码 == 目标代码），不符则试下一候选，
  全不符则不落盘（旧文件保持原样）。可用 --no-verify 关闭。
- 存储: data/raw/cninfo/annual_pdf/{code}/{year}.pdf + annual_txt/{code}/{year}.txt
- 断点续传: 已存在 txt 且 >1000B 跳过（--pairs 模式下按清单强制重下，旧文件备份到
  data/raw/cninfo/wrong_downloads_backup_v2/）
用法: python scripts/fetch_cninfo_annual.py --codes path --years 2018-2025
      python scripts/fetch_cninfo_annual.py --pairs artifacts/review/mda_redownload_pairs.csv
"""
import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pymupdf

BASE = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "cninfo")
CODES = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "baostock", "stock_basic", "a_stocks.csv")
H = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/x-www-form-urlencoded",
     "Referer": "http://www.cninfo.com.cn/new/index"}

SUSPECT_TOKENS = ("摘要", "英文", "H股", "已取消", "取消")
SOFT_TITLE_TOKENS = ("公告", "更正", "补充", "修订", "说明", "问询", "反馈",
                     "进展", "提示", "延期", "回复", "意见", "独立", "审计", "更新前")
CODE_RE = re.compile(r"(?:证券代码|股票代码|公司代码|证券简称及代码|股票简称及代码)[：:\s]*([0-9]{6})")
CODE_PAIR_RE = re.compile(r"([0-9]{6})\s*[,、]\s*g[sz][sh][0-9]{7}")
VALID_PREFIX = ("000", "001", "002", "003", "300", "301", "302",
                "600", "601", "603", "605", "688", "689")


LIST_JSON = os.path.join(BASE, "cninfo_stock_list.json")
# cninfo 官方股票列表（code -> orgId）。orgId 并非总是 "gss{z|h}+补零"：
# 688001=9900038969、002251=9900004894；用补零规则查询会静默返回他司公告（2026-09-11 实测）。
_ORG_IDS = {}


def load_org_ids(refresh: bool = False):
    """code -> orgId；本地缓存；失败回退空表（此时用 gss 规则并依赖身份校验兜底）。"""
    global _ORG_IDS
    if _ORG_IDS and not refresh:
        return _ORG_IDS
    if os.path.exists(LIST_JSON) and not refresh:
        try:
            with open(LIST_JSON, encoding="utf-8") as f:
                _ORG_IDS = {str(x["code"]): x.get("orgId") for x in json.load(f)}
            if _ORG_IDS:
                return _ORG_IDS
        except Exception:
            pass
    try:
        req = urllib.request.Request("http://www.cninfo.com.cn/new/data/szse_stock.json",
                                     headers={"User-Agent": "Mozilla/5.0"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=30) as r:
            arr = json.loads(r.read().decode("utf-8", "ignore")).get("stockList") or []
        if arr:
            os.makedirs(BASE, exist_ok=True)
            with open(LIST_JSON, "w", encoding="utf-8") as f:
                json.dump(arr, f, ensure_ascii=False)
            _ORG_IDS = {str(x["code"]): x.get("orgId") for x in arr}
    except Exception as e:
        print(f"警告: cninfo 股票列表拉取失败（回退 gss 规则）: {e}", file=sys.stderr)
        _ORG_IDS = {}
    return _ORG_IDS


def stock_param(code6: str) -> str:
    """优先用官方 orgId；缺失时回退 gss 补零规则。"""
    org = load_org_ids().get(code6)
    return f"{code6},{org}" if org else (org_id(code6)[0] or code6)


def org_id(code6: str) -> tuple:
    """6 位代码 -> (cninfo stock 参数, column)。北交所不支持。"""
    if code6.startswith(("600", "601", "603", "605", "688", "689")):
        return f"{code6},gssh{code6.zfill(7)}", "sse"
    if code6.startswith(("000", "001", "002", "003", "300", "301", "302")):
        return f"{code6},gssz{code6.zfill(7)}", "szse"
    return None, None


def _query(body: str):
    req = urllib.request.Request("http://www.cninfo.com.cn/new/hisAnnouncement/query", headers=H, data=body.encode())
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=25) as r:
        d = json.loads(r.read().decode("utf-8", "ignore"))
    return d.get("announcements") or []


EM_NAMES_JSON = os.path.join(BASE, "em_former_names.json")


def em_former_names(code6: str):
    """东财 F10 曾用名 + 全称（本地缓存 em_former_names.json）。

    退市公司不在 cninfo 官方股票列表里，orgId 缺失 -> 只能用历史简称检索
    （2026-09-12 实测：'深大通' 可命中 000038 的年报公告）。
    """
    cache = {}
    if os.path.exists(EM_NAMES_JSON):
        try:
            with open(EM_NAMES_JSON, encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}
    if code6 in cache:
        return cache[code6]
    names = []
    secid = ("SH" if code6.startswith(("6", "9")) else "SZ") + code6
    try:
        req = urllib.request.Request(
            f"https://emweb.securities.eastmoney.com/PC_HSF10/CompanySurvey/PageAjax?code={secid}",
            headers={"User-Agent": "Mozilla/5.0"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=25) as r:
            d = json.loads(r.read().decode("utf-8", "ignore"))
        jb = (d.get("jbzl") or [{}])[0]
        for nm in re.split(r"[→,，;；/]", jb.get("FORMERNAME") or ""):
            nm = re.sub(r"[ \u3000]", "", nm)
            if len(nm) >= 3:
                names.append(nm)
        org = re.sub(r"[ \u3000]", "", jb.get("ORG_NAME") or "")
        if len(org) >= 4:
            names.append(org)
    except Exception:
        pass
    cache[code6] = names
    try:
        os.makedirs(BASE, exist_ok=True)
        with open(EM_NAMES_JSON, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
    except Exception:
        pass
    return names


def load_names(code6: str):
    """公司名候选：现名 + 历史名（name_history）+ 语料回采简称 + 东财曾用名/全称。"""
    names = []
    if os.path.exists(CODES):
        with open(CODES, encoding="utf-8") as f:
            for r in csv.reader(f):
                if len(r) > 1 and r[0].endswith("." + code6):
                    names.append(re.sub(r"[ \u3000]", "", r[1]))
    hist = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "name_history_refined.csv")
    if os.path.exists(hist):
        with open(hist, encoding="utf-8") as f:
            for r in csv.reader(f):
                if len(r) > 3 and (r[0].endswith(code6) or r[0] == code6):
                    names.append(re.sub(r"[ \u3000]", "", r[3]))
    # 语料回采：该代码其他年份正文头部的简称（同一实体的历史名称）
    d = os.path.join(BASE, "annual_txt", code6)
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d))[:4]:
            if not fn.endswith(".txt"):
                continue
            try:
                head = re.sub(r"[ \u3000\t]", "",
                              open(os.path.join(d, fn), encoding="utf-8", errors="replace").read(3000))
            except Exception:
                continue
            names += re.findall(r"(?:证券简称|股票简称|公司简称)[：:]([\u4e00-\u9fa5A-Za-z0-9]{2,12})", head)
    names += em_former_names(code6)
    return [n for n in dict.fromkeys(names) if len(n) >= 3]


def query_annual(code6: str, stock: str, column: str, se_date: str):
    """stock 参数精确查询；无本代码结果时回退 searchkey（代码/公司名）。"""
    base = ("pageNum=1&pageSize=30&tabName=fulltext&plate=&secid="
            "&category=category_ndbg_szsh&trade=&sortName=&sortType=&isHLtitle=true")
    anns = _query(f"{base}&column={column}&stock={urllib.parse.quote(stock)}&seDate={se_date}")
    if any(a.get("secCode") == code6 for a in anns):
        return anns
    extra = _query(f"{base}&column={column}&searchkey={code6}&seDate={se_date}")
    if not any(a.get("secCode") == code6 for a in extra):
        for nm in load_names(code6)[:6]:
            cand = _query(f"{base}&column={column}&searchkey={urllib.parse.quote(nm)}&seDate={se_date}")
            if any(a.get("secCode") == code6 for a in cand):
                extra = extra + cand
                break
    return anns + extra


NON_ANNUAL_TOKENS = ("半年度", "半年报", "中期报告", "季度报告", "一季报", "三季报", "季报")


def _pub_year(a):
    m = re.search(r"(\d{4})-\d{2}-\d{2}", a.get("adjunctUrl", "") or "")
    return int(m.group(1)) if m else None


def pick_targets(anns, y: int):
    """按可信度排序返回年报正文候选。

    标题形态（2026-09-11/12 实测）：深市 '2021年年度报告（更新后）'；沪市常见
    '工商银行2021年度报告'（无"年"字）、'2020年报全文'（"年报"简称）、
    乃至完全不带年份的 '年报全文'/'锦江酒店年报'——后者用披露日期定位财年
    （年报于次年披露，publish_year == y+1）。A+H 公司另有 'H股公告-…'（排除）。
    排序：精确匹配 → 带前后缀 → 无年份按披露日期 → 软可疑（修订/更正/补充）
    → 已取消版本；全部经落盘前身份校验，不通过则试下一候选。
    """
    canons = [f"{y}年年度报告", f"{y}年度报告", f"{y}年报"]
    exact, good, bydate, soft, cancelled = [], [], [], [], []
    for a in anns:
        t = re.sub(r"<[^>]+>", "", a.get("announcementTitle", "")).replace(" ", "").replace("\n", "")
        if any(b in t for b in NON_ANNUAL_TOKENS):
            continue
        matched = next((c for c in canons if c in t), None)
        if matched is not None:
            if "已取消" in t and "摘要" not in t and "英文" not in t:
                cancelled.append(a)
            if any(b in t for b in SUSPECT_TOKENS):
                continue
            tail = t.split(matched)[-1]
            for suf in ("（更新后）", "(更新后)", "(更新后）", "（更新后)", "全文", "（A股）", "(A股)"):
                tail = tail.replace(suf, "")
            if tail in ("", ":", "：", "。"):
                exact.append(a)
            elif any(b in t for b in SOFT_TITLE_TOKENS):
                soft.append(a)
            else:
                good.append(a)
        elif "年报" in t or "年度报告" in t:
            # 标题不带年份：靠披露年份定位财年（年报于次年披露）
            if any(b in t for b in SUSPECT_TOKENS):
                continue
            if _pub_year(a) == y + 1:
                bydate.append(a)
    return exact + good + bydate + soft + cancelled


def verify_text_identity(txt_path: str, code6: str):
    """黄金代码校验：(True, content) / (False, content) / (None, None)（无信号）。

    三重信号（任一命中即通过）：
      1) 头部出现的任一有效代码含目标代码（覆盖合并换码公司，如 000066 年报同列 000748）；
      2) 头部出现本公司名（覆盖封面无代码行、但释义段引用他司代码的年报，如 TCL科技/2024）；
      3) 无信号 -> None（由调用方决定接纳，下载来源已限 secCode 与标题双匹配）。
    """
    with open(txt_path, encoding="utf-8", errors="replace") as f:
        head = re.sub(r"[ \u3000\t]", "", f.read(3000))
    codes = {m.group(1) for m in CODE_RE.finditer(head)} | {m.group(1) for m in CODE_PAIR_RE.finditer(head)}
    codes = {c for c in codes if c.startswith(VALID_PREFIX)}
    if code6 in codes:
        return True, code6
    if any(n in head[:1500] for n in load_names(code6)[:3]):
        return True, "name"
    if not codes:
        return None, None
    return False, ",".join(sorted(codes)[:3])


def extract_pdf(pdf_path: str, txt_path: str) -> bool:
    """pymupdf 提取全文（pdftotext 对 CNINFO 年报中文输出乱码，2026-08-09 实测弃用）。"""
    doc = pymupdf.open(pdf_path)
    full = "".join(page.get_text() for page in doc)
    doc.close()
    if not full.strip():
        return False
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full)
    return True


def download_one(code6: str, y: int, verify: bool = True):
    """返回 (code, year, result)；result True 或失败原因字符串。"""
    stock, column = org_id(code6)
    if column is None:
        return code6, y, "unsupported_code"
    stock = stock_param(code6)
    se = f"{y}-01-01~{y + 1}-12-31"
    pdir = os.path.join(BASE, "annual_pdf", code6)
    tdir = os.path.join(BASE, "annual_txt", code6)
    os.makedirs(pdir, exist_ok=True)
    os.makedirs(tdir, exist_ok=True)
    pdfp = os.path.join(pdir, f"{y}.pdf")
    txtp = os.path.join(tdir, f"{y}.txt")
    tmp_pdf = pdfp + ".tmp"
    tmp_txt = txtp + ".tmp"
    try:
        anns = query_annual(code6, stock, column, se)
        cands = pick_targets(anns, y)
        if not cands:
            return code6, y, "no_annual"
        tried = []
        for a in cands:
            title = re.sub(r"<[^>]+>", "", a.get("announcementTitle", ""))[:50]
            if a.get("secCode") not in (None, "", code6):
                # 前身代码（如 601313→601360、000022→001872）：实体相同但代码不同，
                # 按面板 code 口径不采纳（该 code 在该年无年报），登记原因供上游决策
                tried.append(f"predecessor_code={a.get('secCode')}")
                continue
            try:
                req = urllib.request.Request("http://static.cninfo.com.cn/" + a["adjunctUrl"],
                                             headers={"User-Agent": "Mozilla/5.0"})
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(req, timeout=120) as r:
                    pdf = r.read()
            except Exception as e:
                tried.append(f"{title}:dl_err:{str(e)[:30]}")
                continue
            if pdf[:4] != b"%PDF":
                tried.append(f"{title}:not_pdf")
                continue
            with open(tmp_pdf, "wb") as f:
                f.write(pdf)
            if not extract_pdf(tmp_pdf, tmp_txt) or os.path.getsize(tmp_txt) < 1000:
                tried.append(f"{title}:extract_fail")
                continue
            if verify:
                ok, content = verify_text_identity(tmp_txt, code6)
                if ok is False:
                    tried.append(f"{title}:identity_{content}")
                    continue
            # 校验通过（或 unknown）-> 原子落盘；旧文件先备份
            if os.path.exists(txtp) and os.path.getsize(txtp) > 1000:
                bdir = os.path.join(BASE, "wrong_downloads_backup_v2", code6)
                os.makedirs(bdir, exist_ok=True)
                shutil.move(txtp, os.path.join(bdir, f"{y}.txt"))
                if os.path.exists(pdfp):
                    shutil.move(pdfp, os.path.join(bdir, f"{y}.pdf"))
            os.replace(tmp_pdf, pdfp)
            os.replace(tmp_txt, txtp)
            return code6, y, True
        return code6, y, "all_candidates_failed:" + "|".join(tried[:3])
    except Exception as e:
        return code6, y, f"err:{str(e)[:60]}"
    finally:
        for p in (tmp_pdf, tmp_txt):
            if os.path.exists(p):
                os.remove(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default=CODES)
    ap.add_argument("--years", default="2018-2025")
    ap.add_argument("--pairs", default="", help="CSV(code,year[,reason]) 清单，强制重下这些年份")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-codes", type=int, default=0)
    ap.add_argument("--no-verify", action="store_true", help="关闭黄金代码校验（不推荐）")
    args = ap.parse_args()
    verify = not args.no_verify

    plan = []
    if args.pairs:
        with open(args.pairs, encoding="utf-8") as f:
            r = csv.reader(f)
            head = next(r)
            if head and "year" not in [h.strip().lower() for h in head]:
                plan.append((head[0].strip(), int(head[1])))
            for row in r:
                if row and row[0]:
                    plan.append((row[0].strip(), int(row[1])))
        plan = sorted(set(plan))
    else:
        ys = args.years.split("-")
        years = list(range(int(ys[0]), int(ys[1]) + 1))
        with open(args.codes, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        code_meta = {}
        for r in rows[1:]:
            if r and r[0]:
                code_meta[r[0].split(".")[-1]] = r
        codes = [c for c in code_meta if c]
        if args.max_codes:
            codes = codes[: args.max_codes]
        for c in codes:
            stock, column = org_id(c)
            if stock is None:
                continue
            meta = code_meta.get(c) or []
            ipo_y = int(meta[2][:4]) if len(meta) > 2 and meta[2] else 1990
            out_y = int(meta[3][:4]) if len(meta) > 3 and meta[3] else 2100
            for y in years:
                if y < ipo_y or y > out_y:
                    continue
                txt = os.path.join(BASE, "annual_txt", c, f"{y}.txt")
                if os.path.exists(txt) and os.path.getsize(txt) > 1000:
                    continue
                plan.append((c, y))

    print(f"计划: {len(plan)} (workers {args.workers}, verify {verify})", file=sys.stderr)
    if not plan:
        print("无任务", file=sys.stderr)
        return

    failed = []
    done = 0
    ok_n = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, c, y, verify): (c, y) for c, y in plan}
        for fut in as_completed(futs):
            c, y, res = fut.result()
            done += 1
            if res is True:
                ok_n += 1
            else:
                failed.append((c, y, res))
            if done % 50 == 0:
                el = time.time() - t0
                print(f"进度 {done}/{len(plan)} 成功 {ok_n} 失败 {len(failed)} ({el:.0f}s)", file=sys.stderr)

    print(f"完成: 成功 {ok_n}/{done}, 失败 {len(failed)}, 耗时 {(time.time()-t0)/60:.1f} 分钟", file=sys.stderr)
    out = os.path.join(BASE, "failed_annual_refetch.csv" if args.pairs else "failed_annual.csv")
    if failed:
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["code", "year", "reason"])
            w.writerows(failed)
        print(f"失败清单: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
