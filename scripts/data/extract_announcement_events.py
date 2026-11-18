#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""公告事件提取：停牌区间 / ST 历史 / 名称变更（G7 information_events 基础，universe 三分类停牌核验源）。

输入: data/raw/cninfo/announcements/{YYYYMMDD}_{sse|szse}.csv（按日全市场公告索引）
输出:
  data/raw/institutional/suspension_intervals.csv   (code, sec_name, start_date, end_date, start_title, end_title, source)
  data/raw/institutional/st_history_ann.csv          (code, sec_name, event_date, event_type, title, source)
  data/raw/institutional/name_history_ann.csv        (code, sec_name, event_date, title, source)
幂等: 全量重扫生成（可重复运行）。

规则说明:
- 停牌起点: 标题含"停牌"+"公告"，排除含"进展/延期/继续"（延续公告）；终点: 标题含"复牌"+"公告"，排除"延期复牌"
- ST 事件: 标题含"实施退市风险警示/实施其他风险警示/实施风险警示"（实施）、"撤销退市风险警示/撤销其他风险警示"（撤销）
- 名称变更: 标题含"变更证券简称/证券简称变更/公司名称变更/变更公司名称"
"""
import csv
import glob
import os
import re
import sys
import time
from datetime import datetime, timezone

BASE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "eastmoney", "announcements")
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "institutional")
os.makedirs(OUT, exist_ok=True)


def ts_to_date(ts: str) -> str:
    """CNINFO announcementTime：13 位毫秒或 10 位秒 → YYYY-MM-DD。"""
    if not ts:
        return ""
    try:
        v = int(ts)
    except ValueError:
        return ""
    if v > 10**12:  # 毫秒
        v //= 1000
    return datetime.fromtimestamp(v, tz=timezone.utc).strftime("%Y-%m-%d")


def is_suspend_start(title: str) -> bool:
    if "停牌" not in title or "公告" not in title:
        return False
    for exclude in ("进展", "延期", "继续", "复牌"):
        if exclude in title:
            return False
    return True


def is_resume_end(title: str) -> bool:
    if "复牌" not in title or "公告" not in title:
        return False
    return "延期" not in title


def st_event(title: str):
    if "实施退市风险警示" in title or ("实施" in title and "退市风险警示" in title):
        return "ST_IMPL_STAR"
    if "实施其他风险警示" in title or ("实施" in title and "其他风险警示" in title):
        return "ST_IMPL"
    if "撤销退市风险警示" in title or ("撤销" in title and "退市风险警示" in title):
        return "ST_LIFT_STAR"
    if "撤销其他风险警示" in title:
        return "ST_LIFT"
    return None


def name_event(title: str) -> bool:
    return any(k in title for k in ("变更证券简称", "证券简称变更", "公司名称变更", "变更公司名称"))


def main():
    files = sorted(glob.glob(os.path.join(BASE, "*.csv")))
    print(f"扫描 {len(files)} 个公告索引文件", file=sys.stderr)
    susp = []   # (code, sec_name, start_ts, start_title, end_ts, end_title)
    sts = []
    names = []
    t0 = time.time()
    for fi, p in enumerate(files):
        with open(p, encoding="utf-8", errors="ignore") as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            continue
        hdr = rows[0]
        try:
            i_title = hdr.index("title")
            i_time = hdr.index("notice_date")
        except ValueError:
            continue
        # 代码从文件名 {code}_{year}.csv 取；title 前缀 "code:简称标题"
        fname = os.path.basename(p)
        code = fname.split("_")[0]
        for r in rows[1:]:
            if len(r) <= max(i_title, i_time):
                continue
            title = r[i_title]
            name = ""
            if ":" in title:
                name = title.split(":", 1)[0].split(":")[-1]
            date = (r[i_time] or "")[:10]
            if is_suspend_start(title):
                susp.append((code, name, date, "start", title))
            elif is_resume_end(title):
                susp.append((code, name, date, "end", title))
            ev = st_event(title)
            if ev:
                sts.append((code, name, date, ev, title))
            if name_event(title):
                names.append((code, name, date, title))
        if (fi + 1) % 1000 == 0:
            print(f"进度 {fi+1}/{len(files)} ({(time.time()-t0)/60:.1f} min)", file=sys.stderr)

    # 合并停牌区间：按 code 分组，start 后第一个未配对 end；未闭合 start 记 open 区间
    by_code = {}
    for rec in susp:
        by_code.setdefault(rec[0], []).append(rec)
    intervals = []
    for code, recs in by_code.items():
        recs.sort(key=lambda x: x[2])
        used = set()
        for i, r in enumerate(recs):
            if r[3] != "start" or i in used:
                continue
            end_rec = None
            for j in range(i + 1, len(recs)):
                if j in used:
                    continue
                if recs[j][3] == "end" and recs[j][2] >= r[2]:
                    end_rec = recs[j]
                    used.add(j)
                    break
            used.add(i)
            if end_rec:
                intervals.append((code, r[1], r[2], end_rec[2], r[4], end_rec[4]))
            else:
                intervals.append((code, r[1], r[2], "", r[4], ""))

    # 去重（源索引存在同日同标题重复记录）
    seen = set()
    dedup = []
    for it in intervals:
        key = (it[0], it[2], it[4])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(it)
    intervals = dedup

    with open(os.path.join(OUT, "suspension_intervals.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "sec_name", "start_date", "end_date", "start_title", "end_title", "source"])
        for it in intervals:
            w.writerow(list(it) + ["cninfo_ann"])
    with open(os.path.join(OUT, "st_history_ann.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "sec_name", "event_date", "event_type", "title", "source"])
        for s in sts:
            w.writerow(list(s) + ["cninfo_ann"])
    with open(os.path.join(OUT, "name_history_ann.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "sec_name", "event_date", "title", "source"])
        for n in names:
            w.writerow(list(n) + ["cninfo_ann"])

    print(f"完成: 停牌记录 {len(susp)} (区间 {len(intervals)}), ST 事件 {len(sts)}, 名称变更 {len(names)}", file=sys.stderr)
    print(f"耗时 {(time.time()-t0)/60:.1f} 分钟", file=sys.stderr)


if __name__ == "__main__":
    main()
