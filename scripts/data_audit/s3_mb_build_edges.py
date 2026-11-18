"""S3-M-B 正式构网骨架（B2，2026-09-13）。

设计目标（Plan §8 Phase B2）：
  把「相似度 → 边选择 → 权重归一化 → 合同字段输出」写成可复用管线，
  用现有全期相似度矩阵先跑通到「待 τ」接口；τ 一到（B1 校准后）替换参数即可出正式边表。

与 pilot（s3_build_network.py）的关键差别：
  1. **边选择层与权重归一化层分离**：边选择用阈值（τ 或 percentile），
     权重归一化用**全集有效 source** 的相似度之和作分母，不用 Top-k 截断后的 Σsim；
  2. 输出完整 `network_edges` 合同字段（含 available_at / coverage_flag / network_support_status）；
  3. 分母为 0 → `no-network-support`，权重留 missing（**不填 0**）。

τ 的双模式（骨架期）：
  - `--tau <float>`：显式阈值（正式模式，B1 校准后使用）
  - `--peer-target <int>`：占位校准（中位 peer 数 = target），骨架期用于跑通验收
  二者互斥；未提供任一则**默认用 peer-target=10 占位并标 `tau_mode=placeholder`**，
  产物写入 `network_mb_stub/`（不污染正式目录）。

用法：
  python scripts/data_audit/s3_mb_build_edges.py --year 2021 --tau 0.25
  python scripts/data_audit/s3_mb_build_edges.py --year 2021 --peer-target 10   # 骨架占位
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "data/processed/mda_texts_v2.csv"                      # 自建语料（对照链）
TEACHER = ROOT / "data/processed/teacher_mda_panel.csv"            # 老师导出（M-B 主输入，2026-09-22 起）
OUT_DIR = ROOT / "data/processed/network_mb_stub"   # 骨架目录（正式为 network_mb/）
NET_USE_CASE = "product_market_mb"
SRC_VERSION = "mda_texts_v2"
CORPORA = {
    "self": {"path": V2, "src_version": "mda_texts_v2", "ver_prefix": "pm_mb_tfidf_v1",
             "id_col": "code", "year_col": "year"},
    "teacher": {"path": TEACHER, "src_version": "teacher_mda_2023", "ver_prefix": "pm_mb_tfidf_teacher_v1",
                "id_col": "firm_id", "year_col": "year"},
}
DF_CUTOFF = 0.25

STOPWORDS = set("""的 了 和 与 及 或 在 是 为 对 到 由 于 以 并 等 中 上 下 内 外 该 本 各 其 之
有 无 不 也 都 会 将 已 未 可 能 应 需 要 且 而 但 则 若 如 因 此 等 上述 其中 报告期内 报告期 公司 本公司年度 年度
期间 情况 影响 主要 相关 同比 上年 同期 增加 减少 增长 下降 实现 完成 达到 保持 进一步 持续 不断 继续 推动
加强 提升 优化 完善 推进 落实 开展 进行 通过 按照 根据 由于 以及 同时 此外 另外 目前 未来 下一步 后续 总体
亿元 万元 元 个百分点 家 户 人 项 个 名 只 台 辆 吨 平方米 万千瓦 万 亿 千 百 十""".split())
TOKEN_RE = re.compile(r"^[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9]{0,}$")

COLS = ["network_use_case", "network_version", "focal_id", "source_id",
        "similarity", "weight", "available_at", "source_version",
        "coverage_flag", "missing_reason", "network_support_status"]


def load_texts(text_year, corpus="self"):
    """读入某文本年的 (id, text) 列表。

    corpus="self"    → 自建 CNINFO 抽取（mda_texts_v2.csv，status=="ok" 过滤）
    corpus="teacher" → 老师导出的 MD&A 面板（teacher_mda_panel.csv，status=="ok" 过滤）
    两个面板的行语义一致，仅列名与年份类型不同；返回 (panel_code, text)。
    """
    spec = CORPORA[corpus]
    rows = []
    csv.field_size_limit(10 ** 9)
    with open(spec["path"], encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok":
                continue
            if str(r.get(spec["year_col"], "")) != str(text_year):
                continue
            rows.append((r[spec["id_col"]], r["text"]))
    return rows


def tokenize(text, jieba):
    toks = []
    for w in jieba.cut(text):
        w = w.strip()
        if len(w) < 2 and not re.match(r"^[A-Za-z]{2,}$", w):
            continue
        if not TOKEN_RE.match(w):
            continue
        if w in STOPWORDS:
            continue
        toks.append(w)
    return toks


def build_similarity(text_year, sample=None, corpus="self"):
    """返回 (codes, Xn_csr)。Xn 已 L2 归一化，余弦 = Xn @ Xn.T。"""
    import jieba
    jieba.setLogLevel(60)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    rows = load_texts(text_year, corpus=corpus)
    codes = [c for c, _ in rows]
    toks = [tokenize(t, jieba) for _, t in rows]
    df = Counter()
    for ts in toks:
        df.update(set(ts))
    n_doc = len(toks)
    kept = {w for w, c in df.items() if c / n_doc <= DF_CUTOFF}
    toks2 = [[w for w in ts if w in kept] for ts in toks]
    vec = TfidfVectorizer(analyzer=lambda x: x, token_pattern=None, min_df=1)
    X = vec.fit_transform(toks2)
    Xn = normalize(X)
    return codes, Xn


def calibrate_tau(sims_by_focal, n, peer_target, lo=0.0, hi=1.0, iters=60):
    """求 τ 使**全体** focal 的中位 peer 数 = peer_target。

    B1 复用此机制（把 target 换成申万二级行业的中位 peer 数）。
    注意：中位 peer 数是 τ 的**阶梯函数**，简单二分会在并列处跨过目标
    （实测 target=10 时得 12）。故先二分定位区间，再在区间内细扫取命中点。
    """
    import numpy as _np
    arrs = [sims_by_focal[i][1] for i in sorted(sims_by_focal)]

    def med_at(tau):
        return int(_np.median([(a > tau).sum() for a in arrs]))

    for _ in range(iters):
        mid = (lo + hi) / 2
        if hi - lo < 1e-12:
            break
        m = med_at(mid)
        if m > peer_target:
            lo = mid
        elif m < peer_target:
            hi = mid
        else:
            # 命中：向右收窄上界，保留一个命中点
            hi = mid
    if med_at(lo) == peer_target:
        return float(lo)
    # 阶梯跨越：在 [lo, hi] 内细扫找命中区间的中点
    grid = _np.linspace(lo, hi, 401)
    hits = [g for g in grid if med_at(g) == peer_target]
    if hits:
        return float((hits[0] + hits[-1]) / 2)
    return float((lo + hi) / 2)


def select_and_weight(codes, Xn, tau=None, peer_target=10):
    """边选择（阈值）→ 权重归一化（全集分母）→ 逐 focal 行。

    返回 (rows, support_rows, meta)。
      rows:        [(focal, source, sim, weight)]  仅 sim > tau 的边
      support_rows:[（focal, n_edges, weight_sum, status)]
      meta:        {tau, tau_mode, ...}
    """
    n = len(codes)
    idx = np.arange(n)
    # --- 边选择层：全对全相似度，按 tau 过滤 ---
    sims_by_focal = {}     # focal_idx -> (src_idx array, sim array) 已过滤
    all_mass = np.zeros(n)  # 全集相似度之和（用于诊断：被过滤掉多少 mass）
    B = 256
    for s in range(0, n, B):
        blk = idx[s:s + B]
        S = (Xn[blk] @ Xn.T).toarray()
        for k, i in enumerate(blk):
            row = S[k]
            row[i] = -1.0                       # 排除自连接
            all_mass[i] = row[row > 0].sum()
            sel = np.where(row > (tau if tau is not None else -1.0))[0]
            # 无显式 tau 时先用全正相似度，稍后统一算占位 τ
            sims_by_focal[int(i)] = (sel, row[sel])
    # --- τ 校准（B1 复用此机制）---
    # 目标：使**中位 focal 的 peer 数 = peer_target**。
    # 不能用「正值分位数」近似——相似度分布高度右偏且含大量零值（实测 target=10 时
    # 该近似只给出中位 peer=4）。改为在抽样 focal 上对 τ 求中位数命中（二分）。
    tau_mode = "explicit"
    if tau is None:
        tau = calibrate_tau(sims_by_focal, n, peer_target)
        tau_mode = "placeholder"
    # 用最终 tau 重新过滤
    rows, support_rows = [], []
    for i in range(n):
        sel, sim = sims_by_focal[i]
        keep = sim > tau
        src = sel[keep]; sv = sim[keep]
        wsum = float(sv.sum())
        if len(src) == 0 or wsum <= 0:
            support_rows.append((codes[i], 0, 0.0, "no-network-support"))
            continue
        for j, s_ in zip(src, sv):
            rows.append((codes[i], codes[j], float(s_), float(s_ / wsum)))
        support_rows.append((codes[i], int(len(src)), wsum, "support"))
    n_sup_c = sum(1 for x in support_rows if x[3] == "support")
    meta = {"tau": float(tau), "tau_mode": tau_mode, "peer_target": (peer_target if tau_mode == "placeholder" else None),
            "n_nodes": n, "n_nodes_total": n, "n_support": n_sup_c,
            "n_no_support": len(support_rows) - n_sup_c, "n_edges": len(rows),
            "mass_retained_ratio": float(
                sum(r[2] for r in rows) / all_mass.sum()) if all_mass.sum() > 0 else None}
    return rows, support_rows, meta


def select_network_version(event_date, versions):
    """消费侧规则（plan L1693/L3696/L4313）：给定事件日 d，选 max{available_at_v < d} 的版本。

    versions: [(version_name, available_at_iso_str), ...]
    返回版本名；无可用版本返回 None（此时该事件不得用任何网络 → no-network-support 语义）。
    """
    cand = [(v, a) for v, a in versions if a < event_date]
    if not cand:
        return None
    return max(cand, key=lambda x: x[1])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True, help="文本年（网络版本 = 该年）")
    ap.add_argument("--tau", type=float, default=None, help="显式相似度阈值（正式模式）")
    ap.add_argument("--peer-target", type=int, default=10, help="占位校准：中位 peer 数")
    ap.add_argument("--available-at", default="", help="网络公开可得日（默认取覆盖表最晚报告日）")
    ap.add_argument("--corpus", choices=["self", "teacher"], default="self",
                    help="文本语料源：self=自建 CNINFO 抽取（对照链）；teacher=老师 MD&A 导出（M-B 主输入）")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    if args.tau is not None and args.peer_target != 10:
        raise SystemExit("--tau 与 --peer-target 互斥")

    codes, Xn = build_similarity(args.year, corpus=args.corpus)
    print(f"文本年 {args.year}: {len(codes)} 家入网", flush=True)
    rows, support_rows, meta = select_and_weight(
        codes, Xn, tau=args.tau, peer_target=args.peer_target)
    # available_at 语义（对齐 plan §7 L1693）：该网络版本可被研究流程取得的日期。
    # 采用**年报法定披露截止日**（次年 4 月 30 日）作保守取值——此时该财年年报已全部公开。
    # 消费侧规则：事件的网络版本 = argmax{available_at_v < d}（plan L1693/L4313），
    # 故研究年 Y 的 1-4 月会落到版本 Y-2 上，不产生前视。
    avail = args.available_at or f"{args.year + 1}-04-30"
    outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)
    net_ver = f"{CORPORA[args.corpus]['ver_prefix']}_{args.year}"
    fp = outd / f"{args.year}_similarity.csv"
    with open(fp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for focal, src, sim, wt in rows:
            w.writerow([NET_USE_CASE, net_ver, focal, src, round(sim, 8), round(wt, 8),
                        avail, CORPORA[args.corpus]["src_version"], "covered", "", "support"])
    sp = outd / f"{args.year}_support.csv"
    with open(sp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["focal_id", "n_edges", "weight_sum", "network_support_status",
                    "network_version", "available_at"])
        for focal, ne, ws, status in support_rows:
            w.writerow([focal, ne, round(ws, 8) if ws else "", status, net_ver, avail])
    meta.update({"network_version": net_ver, "available_at": avail,
                 "text_year": args.year,
                 "run_at": datetime.now(timezone.utc).isoformat(),
                 "output": str(fp.relative_to(ROOT)) if fp.is_absolute() else str(fp).replace("\\", "/"),
                 "corpus": args.corpus,
                 "note": ("**正式边表**（tau_mode=explicit，τ 取 registry `m_b_tau_fixed`，可复现）"
                          if meta.get("tau_mode") == "explicit" else
                          "骨架产物（stub）：tau_mode=placeholder，**不得作为 headline**")})
    rp = outd / f"{args.year}_build_report.json"
    rp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    st = Counter(s[3] for s in support_rows)
    print(json.dumps(meta, ensure_ascii=False))
    print("support status:", dict(st), "->",
          fp.relative_to(ROOT) if fp.is_absolute() else fp)


if __name__ == "__main__":
    main()
