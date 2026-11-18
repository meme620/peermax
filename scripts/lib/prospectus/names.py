"""s3_network_prospectus_anchor 的 names 模块（逐字搬移，逻辑未改）。"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data_audit'))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data'))

from lib.prospectus.consts import CJK_CH
from lib.prospectus.consts import FULL_NAME_RE
from lib.prospectus.consts import GEN_SUB
from lib.prospectus.consts import GEN_SUB_MAXLEN
from lib.prospectus.consts import GEN_SUB_SET
from lib.prospectus.consts import GEN_SUB_TAIL_EXEMPT
from lib.prospectus.consts import GEO_EXTRA
from lib.prospectus.consts import LABEL_CUT
from lib.prospectus.consts import NAME_BIZ_WORD
from lib.prospectus.consts import NAME_START_BAD
from lib.prospectus.consts import PLACE_PREFIX
from lib.prospectus.consts import R2_CTX_HEAD
from lib.prospectus.consts import R4_MEMBER_RE
from lib.prospectus.consts import R4_MEMBER_STOP
from lib.prospectus.consts import ROOT
from lib.prospectus.consts import ST_PREFIX
from lib.prospectus.consts import SUFFIX_END
from lib.prospectus.consts import _CTX_LINK_TAIL
from lib.prospectus.consts import _CTX_SUFFIX
from lib.prospectus.consts import _GEO
from lib.prospectus.consts import _SELF_IDX
import os
import re

def r4_member_ok(t):
    """修 1(a) 守则①：成员 = 2–6 汉字、非动词/连接词/科目词起头、**全名不含虚词字符**。"""
    if not R4_MEMBER_RE.match(t):
        return False
    if any(t.startswith(x) for x in NAME_START_BAD):
        return False
    if any(c in R4_MEMBER_STOP for c in t):
        return False
    return True

def r4_strip_head(t, uni_keys):
    """修 1(a) 守则④（registry 1.5.17）：名录守卫的**首字修复**——t[1:] 在 A 股在册名录（uni_keys）
    且 t 本身不在册 → 剥首字（切分/拼接残片，如 `有海泰科→海泰科`、`中苏交科→苏交科`、`中广日股份→广日股份`）；
    真名（如 `有研新材` = 自身在册）不剥；剥后若落到在册名，则可被映射为同行（而非计入未映射）。"""
    s = t
    while len(s) > 2 and s not in uni_keys and s[1:] in uni_keys:
        s = s[1:]
    return s

def r4_peer_list_ok(toks, gap, uni_keys):
    """修 1(a) 守则②③（registry 1.5.17）：复合触发词直列**必须是名单**——
    ② 触发词与首位成员之间不得有并列分隔符（散文续写常见「…公司，公司主要…」）；
    ③ 名单须含 **≥1 个 A 股在册名**（`uni_keys` 名录守卫；纯散文/科目/账龄枚举无人名 → 弃）。
    方向 = 从严（只影响**新增**的复合触发路径；既有「包括/分别为/有：+ 同行语境」路径逐字未改）。"""
    if any(ch in gap for ch in "、,，"):     # 并列分隔符出现在触发词与首位成员之间 = 散文续写（；/： 允许）
        return False
    if not any(t in uni_keys for t in toks):
        return False
    return True

def is_generic_name(b):
    """修 3（registry 1.5.17）：抽取噪声判据 = 「整名 ≈ 通用词」+ 公司后缀豁免。

    ① **公司后缀豁免**：名以 公司/集团/股份/有限/企业 收尾，或含「股份」「集团」→ **不得**判为抽取噪声
       （原「子串命中」判据把真实公司名（上海弘陆物流设备 / 山东联创节能新材料 / 国网综合能源服务…）
       误判为噪声：复核实测 117 条中 73 条 ≥6 字或带 公司/集团/股份）；
    ② **整名 ≈ 通用词**：整名 ∈ 词表，或整名长度 ≤ `GEN_SUB_MAXLEN` 且含词表词
       （= 整名近似于通用词/科目词/句片段词，如 一季度 / 差旅费 / 在产品 / 应用领域 / 运输工具）；
    ③ 其余（多为句片段/公司名）**不由本判据判读** → 落入残差桶（需独立判读）。
    """
    if any(b.endswith(t) for t in GEN_SUB_TAIL_EXEMPT) or "股份" in b or "集团" in b:
        return False
    if b in GEN_SUB_SET:
        return True
    return len(b) <= GEN_SUB_MAXLEN and any(t in b for t in GEN_SUB)

def geo_prefix():
    """「区域/中国」前缀集（H1 切分判据用；延迟引用 §P5 的 PLACE_PREFIX，避免前向引用歧义）。"""
    global _GEO
    if _GEO is None:
        _GEO = set(PLACE_PREFIX) | set(GEO_EXTRA)
    return _GEO

def cut_label_tail(cap):
    """H1-(b)：标签边界切分——取捕获串中最早的标签片段位置切分（保留 ≥2 字，否则原样返回）。"""
    best = None
    for lb in LABEL_CUT:
        i = cap.find(lb)
        if i >= 2 and (best is None or i < best):
            best = i
    return cap[:best] if best is not None else cap

def trim_dup_name(seg, start, cap):
    """H1-(a)：年报头部「简称」捕获串规范化 = 标签边界切分 + 「简称+全称同头拼接」最短合法切分。

    切分 L 合法 ⟺ ① rest = ext[L:] 形态为 …股份有限公司/有限责任公司/有限公司；② rb = base_name(rest)
    与简称头部一致——rb 以简称前 2 字起头，**或** rb 以区域/「中国」前缀（2–3 字）起头且其后以简称
    前 2 字起头；③ rb 不以公司后缀残片（股份/有限/责任/公司）起头。取最小合法 L；无合法切分时返回
    标签切分后的捕获串（不改动）。**判据登记 = registry 1.5.15**。
    """
    cap = cut_label_tail(cap)
    if len(cap) < 2:
        return cap
    j, ext = start + len(cap), cap
    while j < len(seg) and len(ext) < 30 and not SUFFIX_END.search(ext):
        if not CJK_CH.match(seg[j]):
            break
        ext += seg[j]
        j += 1
    for L in range(2, min(10, len(ext)) + 1):
        short, rest = ext[:L], ext[L:]
        if not FULL_NAME_RE.match(rest):
            continue
        rb = base_name(rest)
        if rb.startswith(("股份", "有限", "责任", "公司")):
            continue
        head = short[:2]
        ks = [0] + [k for k in (2, 3) if rb[:k] in geo_prefix()]
        if any(len(rb) - k >= 2 and rb[k:].startswith(head) for k in ks):
            return short
    return cap

def ctx_suffix_frags():
    """修 1(e)（registry 1.5.17）：语境词的 **≥2 字后缀片段** 集（R2 捕获窗口 12 字截断时
    捕获串常以语境词的中段起头，如 `行业可比上市公司有东方雨虹` ← 「同行业可比上市公司有」）。"""
    global _CTX_SUFFIX
    if _CTX_SUFFIX is None:
        s = set()
        for p in R2_CTX_HEAD:
            for k in range(2, len(p) + 1):
                s.add(p[-k:])
        _CTX_SUFFIX = s
    return _CTX_SUFFIX

def repair_r2_left(raw, uni_keys=()):
    """M1-(a) + 修 1(e)：R2 左边界修复——语境/连接词前缀逐层剥离 + 语境后缀片段剥离 +
    尾随连接词（「有」）**名录守卫**剥离（修复轮二，registry 1.5.17）。

    守则（顺序）：
      ① 捕获串本身即在 A 股在册名录（uni_keys）→ **整体不剥离**（防误切真名）；
      ② 既有 R2_CTX_HEAD 逐层剥离（≥2 字剩余）；
      ③ 修 1(e)：以任一语境词的 ≥2 字**后缀片段**起头（剩余 ≥3 字）→ 剥离一次；
      ④ 剥离后回到 ② 再剥一层（露出连接词的情形）；
      ⑤ 修 1(e)：以「有」起头且剩余 ≥2 字、**整体不在册** → 剥离该「有」（名录守卫防误切「有研新材」）。
    """
    s = raw
    if raw in uni_keys:          # ① 名录守卫（uni_keys 为集合；在册名整体不剥离）
        return s
    changed = True
    while changed and len(s) > 4:
        changed = False
        for p in R2_CTX_HEAD:
            if s.startswith(p) and len(s) - len(p) >= 2:
                s, changed = s[len(p):], True
                break
    for L in range(min(12, len(s) - 3), 1, -1):     # 修 1(e)：语境后缀片段（保留 ≥3 字）
        if s[:L] in ctx_suffix_frags():
            s = s[L:]
            break
    changed = True
    while changed and len(s) > 4:
        changed = False
        for p in R2_CTX_HEAD:
            if s.startswith(p) and len(s) - len(p) >= 2:
                s, changed = s[len(p):], True
                break
    for t in _CTX_LINK_TAIL:
        if s.startswith(t) and len(s) - len(t) >= 2 and s not in uni_keys:
            s = s[len(t):]
    return s

def lev_leq1(a, b):
    """M2 容错：|Δlen| ≤1 且编辑距离 ≤1 的快速判据（O(n)）。"""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x != y) <= 1
    s, l = (a, b) if len(a) < len(b) else (b, a)
    i = j = 0
    skipped = False
    while i < len(s) and j < len(l):
        if s[i] == l[j]:
            i, j = i + 1, j + 1
            continue
        if skipped:
            return False
        skipped, j = True, j + 1
    return True

def build_self_index(full, short_hist):
    """按 code 倒排名称注册表（H1/M2 修复轮：自名集合须含**代码锚定历史简称**；一次构建、全程复用）。"""
    fu, sh = {}, {}
    for k, v in full.items():
        for c in v:
            fu.setdefault(c, []).append(k)
    for k, v in short_hist.items():
        for c in v:
            sh.setdefault(c, []).append(k)
    _SELF_IDX.update(full=fu, short=sh, cache={})

def self_names_for(firm, code):
    """发行人自名集合 (exact, core)：universe 当前名 + 全称字典键 + **名称注册表历史简称**（均含基名）。

    core = 容错判据比较池 = exact 中长度 ≥2 者（**不设长度上限**——登记 M2 判据为「与**某自名**」，
    长度窗口会漏掉长全称（如「…股份有限公司」>14 字）上的同核/近全包含命中）。**登记 = registry 1.5.15**。
    """
    ent = _SELF_IDX["cache"].get(code)
    if ent is None:
        exact = {norm(firm[code]["name"]), base_name(firm[code]["name"])}
        for k in _SELF_IDX["full"].get(code, ()):
            exact.add(k)
            exact.add(base_name(k))
        for k in _SELF_IDX["short"].get(code, ()):
            exact.add(k)
            exact.add(base_name(k))
        exact = {x for x in exact if x}
        ent = (exact, {x for x in exact if len(x) >= 2})
        _SELF_IDX["cache"][code] = ent
    return ent

def is_self_name(b, raw, self_exact, self_core, uni_keys=()):
    """M2：发行人自指容错判据（登记 = registry 1.5.15）。

    判据（任一命中即判自指）：(i) 精确；(ii) 同核（与某自名共享 **≥3 字前缀，上限 4 字**——逐 k∈{4,3} 比对）；
    (iii) 编辑距离/OCR 近似（|Δlen| ≤1 且 lev ≤1，双方 ≥4 字）；(iv) 近全包含（b ⊆ 自名，残差 ≤4 字，len(b) ≥3）。
    守卫：b ∈ A 股 universe 名录（uni_keys）→ **不**做自指剔除（防误剔真实在册名）。
    """
    if not b:
        return False
    if b in self_exact or raw in self_exact:
        return True
    if b in uni_keys or len(b) < 3:
        return False
    for s in self_core:
        if b == s:
            return True
        if len(b) <= len(s) and b in s and len(s) - len(b) <= 4:
            return True
        for k in (4, 3):                       # 同核：共享 ≥3 字前缀（上限 4 字）
            if len(b) >= k and len(s) >= k and b[:k] == s[:k]:
                return True
        if len(b) >= 4 and len(s) >= 4 and lev_leq1(b, s):
            return True
    return False

def strip_st(name):
    """去 ST/*ST/S*/S 前缀标记（修 2）：universe 当前名与招股书当时名之间的 ST 前缀漂移。"""
    s = ST_PREFIX.sub("", name or "")
    return s if len(s) >= 2 else (name or "")

def r3_shape_ok(b):
    """R3 名称形态约束（修 1）：3–10 汉字 + 非动词/连接词起头 + 无业务描述词。"""
    if not re.match(r"^[\u4e00-\u9fa5]{3,10}$", b):
        return False
    if any(b.startswith(x) for x in NAME_START_BAD):
        return False
    if any(x in b for x in NAME_BIZ_WORD):
        return False
    return True

def rel(p):
    try:
        return os.path.relpath(p, ROOT).replace("\\", "/")
    except ValueError:
        return p

def norm(s):
    s = re.sub(r"[ \u3000\t\r\n\u00a0]", "", s or "")
    s = s.replace("（", "(").replace("）", ")").replace("，", ",").replace("、", ",")
    s = re.sub(r'["\'“”‘’《》<>]', "", s)
    return s

def base_name(s):
    """去公司后缀；保留到「集团」级别（不删集团/科技/股份，避免过度归一化）。"""
    s = norm(s)
    s = re.sub(r"(?:集团)?(?:股份有限公司|有限责任公司|有限公司)$", "", s)
    return s

