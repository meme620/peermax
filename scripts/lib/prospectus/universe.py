"""s3_network_prospectus_anchor 的 universe 模块（逐字搬移，逻辑未改）。"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data_audit'))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data'))

from datetime import datetime, timezone
from lib.prospectus.consts import ANNUAL_TXT
from lib.prospectus.consts import A_STOCKS
from lib.prospectus.consts import BLACK
from lib.prospectus.consts import CACHE
from lib.prospectus.consts import CONN_SUB
from lib.prospectus.consts import EM_NAMES
from lib.prospectus.consts import FIRM_MAP
from lib.prospectus.consts import NAME_HIST
from lib.prospectus.consts import NAME_HIST_CAND
from lib.prospectus.consts import NAME_START_BAD
from lib.prospectus.consts import NOISE_SUB
from lib.prospectus.consts import _WS
from lib.prospectus.names import base_name
from lib.prospectus.names import cut_label_tail
from lib.prospectus.names import norm
from lib.prospectus.names import strip_st
from lib.prospectus.names import trim_dup_name
import csv
import json
import os
import re
import time

def load_firm_universe():
    firm, short, ipo = {}, {}, {}
    with open(FIRM_MAP, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            firm[r["code6"]] = dict(firm_id=r["firm_id"], name=r["name"], ipo_date=r["ipo_date"],
                                    board=r["board"])
            short.setdefault(norm(r["name"]), set()).add(r["code6"])
    with open(A_STOCKS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            c = r["code"].split(".")[-1]
            if len(c) == 6 and c.isdigit() and c in firm:
                short.setdefault(norm(r["code_name"]), set()).add(c)
    for path in (NAME_HIST, NAME_HIST_CAND):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                c = re.sub(r"\D", "", r.get("code", ""))[:6]
                nm = norm(r.get("name", ""))
                if c in firm and nm and len(nm) >= 2:
                    short.setdefault(nm, set()).add(c)
    # 修 2（2026-10-13 修正轮）：universe 名称去 ST/*ST/S*/S 前缀标记建键——试点年（2018）招股书用
    # 当时简称，universe 为当前简称；ST 前缀（如 *ST亿晶 / ST信通）会挡住历史简称的精确匹配。
    for k, v in list(short.items()):
        sk = strip_st(k)
        if sk != k:
            short.setdefault(sk, set()).update(v)
    return firm, short

def load_universe_name_keys():
    """A 股 universe 名称键（firm_id_map 当前简称 + a_stocks 简称 + 各自去 ST 前缀变体）；
    仅用于 unmapped 三分类的「名称漂移」判读（不含年报派生的第三方/子公司名）。"""
    keys = set()
    with open(FIRM_MAP, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n = norm(r["name"])
            keys.add(n)
            keys.add(strip_st(n))
    with open(A_STOCKS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n = norm(r["code_name"])
            keys.add(n)
            keys.add(strip_st(n))
    return {k for k in keys if len(k) >= 2}

def build_full_name_map(firm, use_cache=True):
    """名称注册表：① 年报头部「股票简称/证券简称 + 股票代码」配对（历史简称，代码锚定）；
    ② 年报头部公司全称（最新与最早各一年）；③ 年报「证券简称由 X 变更为 Y」更名披露；
    ④ 年报释义段「X 指 Y（…公司）」缩写定义。缓存 artifacts/tmp/tier2_prosp/full_name_map.json（v3）。

    历史简称是**代码锚定**抽取（简称与 6 位代码相邻，或同属该代码自己的年报），非模糊匹配；用于弥合
    「项目 firm 名单为当前简称、招股书用当时简称」的名称漂移（如 良信电器→良信股份、东方航空→中国东航、
    坚瑞沃能→保力新）。**修 2（2026-10-13 修正轮）**：① 头部窗口 6000→24000 字（「公司股票简况」块常在
    6000 字之后）；② 标签补 `公司代码`/`A股证券简称` 且允许冒号；③ 头部「简称+全称同头拼接」去重；
    ④ 新增更名披露与释义缩写两条代码锚定来源；⑤ 注册名统一去 ST/*ST/S*/S 前缀标记建键。
    **修 5（2026-10-14 修复轮，registry 1.5.15）**：① 「简称+全称同头拼接」切分判据重写
    （`trim_dup_name`：最短合法切分 + rest 去后缀基名与简称头部一致性 + 公司后缀残片起头否决）；
    ② 新增**标签边界切分**（`cut_label_tail`：转债代码/债券代码/港股代码 等标签片段按最早边界切分）；
    ③ 缓存 `cache_version` 3→**4**（强制重算）。历史简称来源与代码锚定机制不变，**无人工补录**。
    """
    if use_cache and os.path.exists(CACHE):
        try:
            with open(CACHE, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("info", {}).get("cache_version") == 4 and d["info"].get("n_codes", 0) >= 4000:
                return ({k: set(v) for k, v in d["full"].items()},
                        {k: set(v) for k, v in d.get("short_hist", {}).items()}, d["info"])
        except Exception:
            pass
    full, short_hist, n_codes, n_files, n_short = {}, {}, 0, 0, 0
    n_change, n_defn = 0, 0
    codes = sorted(firm)
    t0 = time.time()
    P_SHORT_FWD = re.compile(r"(?:股票简称|证券简称|公司简称|中文简称|A股证券简称)[：:]?"
                             r"([一-龥A-Za-z0-9*]{2,10}?)[（(]?(?:股票代码|证券代码|公司代码)[：:]?([0-9]{6})")
    P_SHORT_REV = re.compile(r"(?:股票代码|证券代码|公司代码)[：:]?([0-9]{6})[^一-龥]{0,4}"
                             r"(?:股票简称|证券简称|公司简称|中文简称)[：:]?([一-龥A-Za-z0-9*]{2,10})")
    P_FULL = re.compile(r"[一-龥]{2,20}(?:股份有限公司|有限责任公司|有限公司)")
    # ③ 更名披露：「证券简称由“东方航空”变更为“中国东航”」（该年报所属代码 = 新旧名同属）
    P_NAME_CHANGE = re.compile(r"(?:A股)?(?:证券简称|股票简称)由[“\"']?([一-龥A-Za-z0-9*]{2,10})[”\"']?"
                               r"(?:变更为|更改为)[“\"']?([一-龥A-Za-z0-9*]{2,10})")
    # ④ 释义缩写：「东方航空指中国东方航空股份有限公司」（左名须为右全称的后缀/本身，排除子公司前缀式）
    P_DEFN = re.compile(r"([一-龥A-Za-z0-9]{2,10})指([一-龥A-Za-z0-9]{2,20}(?:股份有限公司|有限责任公司|有限公司))")

    def reg_short(nm, cd):
        """注册短名（含去 ST 前缀变体）。"""
        nonlocal n_short
        if not (cd in firm and 2 <= len(nm) <= 10):
            return False
        short_hist.setdefault(nm, set()).add(cd)
        n_short += 1
        sk = strip_st(nm)
        if sk != nm and 2 <= len(sk) <= 10:
            short_hist.setdefault(sk, set()).add(cd)
        return True

    def trim_dup(seg, start, cap):
        """修 5（registry 1.5.15）：转调模块级 `trim_dup_name`（切分判据 + 标签边界切分）。"""
        return trim_dup_name(seg, start, cap)

    for k, c in enumerate(codes):
        d = os.path.join(ANNUAL_TXT, c)
        if not os.path.isdir(d):
            continue
        years = sorted([x for x in os.listdir(d) if x.endswith(".txt")])
        got = 0
        for fn in years:
            n_files += 1
            try:
                head = re.sub(_WS, "", open(os.path.join(d, fn), encoding="utf-8",
                                            errors="replace").read(24000))
            except Exception:
                continue
            for m in P_SHORT_FWD.finditer(head):
                nm, cd = norm(cut_label_tail(m.group(1))), m.group(2)   # 修 5：标签边界切分
                if reg_short(nm, cd):
                    got += 1
            for m in P_SHORT_REV.finditer(head):
                cd, nm = m.group(1), norm(m.group(2))
                nm = trim_dup(head, m.start(2), nm)                      # 修 5：切分判据重写 + 标签边界切分
                if reg_short(nm, cd):
                    got += 1
            # 本公司全称（头部 1200 字内）——更名披露/释义缩写两条来源须指向**本公司**（防同业/客户/子公司误挂）
            own = {base_name(x.group(0)) for x in P_FULL.finditer(head[:1200])}
            for m in P_NAME_CHANGE.finditer(head):
                old, new = norm(m.group(1)), norm(m.group(2))
                if not own or not any((old in o or o in old or new in o or o in new) for o in own):
                    continue
                for nm in (old, new):     # 旧名与新名都登记（同代码）
                    if reg_short(nm, c):
                        n_change += 1
            for m in P_DEFN.finditer(head):
                left, right = norm(m.group(1)), norm(m.group(2))
                rb = base_name(right)
                if not (rb == left or rb.endswith(left)) or len(left) < 2:
                    continue
                if not own or not any((left in o or o in left) for o in own):
                    continue              # 释义缩写须指向本公司（否则是定义同业/客户）
                if any(x in left for x in CONN_SUB) or any(x in left for x in NOISE_SUB) \
                        or left in BLACK or left.startswith(NAME_START_BAD):
                    continue
                if reg_short(left, c):
                    n_defn += 1
            if fn in (years[-1], years[0]):
                for m in P_FULL.finditer(head):
                    nm = norm(m.group(0))
                    if len(nm) >= 6:
                        full.setdefault(nm, set()).add(c)
                        got += 1
        if got:
            n_codes += 1
        if (k + 1) % 1000 == 0:
            print(f"[P-2] 名称注册表构建 {k + 1}/{len(codes)}（覆盖 {n_codes} 只 / 简称条目 {n_short}，"
                  f"{time.time() - t0:.0f}s）", flush=True)
    if os.path.exists(EM_NAMES):
        try:
            em = json.load(open(EM_NAMES, encoding="utf-8"))
            for c, names in em.items():
                c6 = str(c).zfill(6)
                if c6 not in firm:
                    continue
                for nm in names:
                    nn = norm(nm)
                    if re.search(r"(?:股份有限公司|有限责任公司|有限公司)$", nn):
                        full.setdefault(nn, set()).add(c6)
                    elif 2 <= len(nn) <= 10:
                        reg_short(nn, c6)
        except Exception:
            pass
    # 注册名统一去 ST 前缀建键（修 2）：头部简称可能带 ST（如 ST信通），招股书用无前缀历史名
    for nm in list(short_hist):
        sk = strip_st(nm)
        if sk != nm:
            short_hist.setdefault(sk, set()).update(short_hist[nm])
    info = {"cache_version": 4, "n_codes": n_codes, "n_files": n_files, "n_full_names": len(full),
            "n_short_hist": len(short_hist), "n_short_pairs": n_short,
            "n_name_change_hits": n_change, "n_defn_hits": n_defn, "head_chars": 24000,
            "fix_round": "registry 1.5.15：trim_dup_name（最短合法切分 + 区域前缀一致性 + 后缀残片否决）"
                         "+ cut_label_tail（标签边界切分）",
            "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_sec": round(time.time() - t0, 1),
            "source": "annual_txt 头部（股票简称/公司简称+股票代码 配对（标签/顺序/冒号变体、去 ST 前缀）；"
                      "全称取最新/最早各一年）+ 更名披露 + 释义缩写 + em_former_names.json"}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump({"info": info, "full": {k: sorted(v) for k, v in full.items()},
                   "short_hist": {k: sorted(v) for k, v in short_hist.items()}}, f, ensure_ascii=False)
    return full, short_hist, info

