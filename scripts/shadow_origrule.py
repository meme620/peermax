"""Phase 3.3（N-2 原文第四步阈值规则）shadow：原文字面规则的**最小证据** + PC 对照 + 「不可比」决策数据底座。

背景（`artifacts/phase3/3.0_plan.md` §2.3；审计 L-2 / N-2）
--------------------------------------------------------------
原文第四步对手集合 = 「相似度 > **团内最小相似度**」的公司，**字面不限定候选域**（可含团外）；
项目口径 = **同团成员**（`network_business_edges_build.py` 已登记偏离：「团内最小相似度阈值法实测退化」）。
既有退化证据只在 2021 年 **self 语料**：K=300 → peer 3,456 / lift 1.14；**K=400 → 3,347 / 1.19**
（`reports/data_gates/G0/s3_ma_K_explore.json`）；**teacher 语料（headline K_y 235–463）从未读过** → 本脚本补齐。
「窄词表重建到原文 ~1.6 万 active_vocab」路线实测不可行（df 阈值收紧对词表/active_vocab 几乎无影响，
见本脚本 `--vocab-probe` 实测与 `3.3_counts.json` 的 `vocab_sensitivity`）。

本脚本做什么（headline 只读；只写 shadow）
--------------------------------------------------------------
1. 逐年 `build_vectors(y, corpus='teacher')`（headline 词表 df≤0.25、同筛选、等权 0/1）→ 全相似度矩阵 C；
2. **原文规则**：`thr_i` = focal 所在团的**团内最小相似度**（团内任意两成员对的最小相似度，团内全员共享；
   单点团 → 1.0，即无 peer），`mask = C > thr_i`（**严格大于**、**候选域 = 全部节点**、排除对角）；
   **项目规则**（对照）：同团成员（peer = 团规模 − 1）；
3. 逐年报：peer 分位 / 占市场比 / lift（申万二级；latest 静态与 as-of 两套标签）/ 与同团规则重合度 /
   thr 分位 / 团内零相似对；
4. **PC 对照**（文本年 2017–2022，y−1 年营收可得）：`pc_origrule = −HHI(原文规则对手集合 ∪ {focal},
   y−1 年销售收入)`（对手集合按原文规则含团外；HHI 分母含 focal 自身，与 headline PC 同构），
   与 headline 同团 PC 并排；2014–2016 营收面板缺 → pc 留空（不填 0，`pc_status=revenue_year_not_in_panel`）；
5. R1 对账：逐年矩阵与 headline `{y}_similarity.csv` 逐对核对；同团规则重算 PC 与 headline `{y}_pc.csv`
   逐值对账（容差 1e-6）；同团规则 peer/lift/K 重算与 freeze 报告逐值对账；
6. 固化解读：`3.3_counts.json`（退化计数/PC 对比/词表敏感度/结论）、`3.3_origrule_readout.csv`、
   `pc_origrule_p3shadow.csv`(+`.meta.json`)，另 `3.3_decision_memo.md` / `3.3_execution.md`（执行期写）。

不改：headline 网络/PC/M1（只读）、`reports/data_gates/G0/*`、`configs/*`、现有脚本与测试；版本串 `p3shadow_v1`。

产物
----
  data/processed/phase3_shadow/pc_origrule_p3shadow.csv       逐 focal×年×规则（原文规则 / 同团对照）
  data/processed/phase3_shadow/pc_origrule_p3shadow.meta.json 口径/列定义/版本
  artifacts/phase3/3.3_origrule_readout.csv                   逐年 × {headline,p3asof} × {2 规则} 读数
  artifacts/phase3/3.3_counts.json                            计数/对账/结论/锚台账
  artifacts/phase3/3.3_anchor_{pre,post}.json                 9 锚 + headline 45 文件 sha 台账（+3.1 产物）
  artifacts/phase3/3.3_output_sha.json                        本次 shadow 产物 sha

用法
----
  python scripts/shadow_origrule.py --steps all
  python scripts/shadow_origrule.py --steps readout --years 2021   # 子集重算
  python scripts/shadow_origrule.py --steps vocab --no-vocab-probe"""

from __future__ import annotations

import sys
from pathlib import Path
from fileio import atomic_write_text as _atomic_write_text

import argparse
import csv
import json
import os
import time
import warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from pricing_lib import assert_m1_inputs_sha, require, sha256_file  # noqa: E402
from network_business_lexicon import build_automaton, load_lexicon  # noqa: E402
from network_business_cluster_grid import build_vectors, load_refs  # noqa: E402
from fileio import json_default as _json_default
from fileio import assert_path_allowed  # noqa: E402



ROOT = Path(__file__).resolve().parents[1]

SHADOW_DIR = ROOT / 'data/processed/phase3_shadow'

HEADLINE_NET = ROOT / 'data/processed/network_ma_teacher'

P3ASOF_NET = ROOT / 'data/processed/network_ma_teacher_p3asof'   # 3.1 产物（可选；只读交叉读数）

TEACHER_LEXICON = ROOT / 'data/processed/product_lexicon_v2_teacher.csv'

REVENUE_PKL = ROOT / 'data/raw/drive/operating_revenue_annual.pkl'

SW2_LATEST = ROOT / 'data/processed/sw2_industry_latest.csv'

ASOF_MAP_PARQUET = SHADOW_DIR / 'sw2_asof_map_p3shadow.parquet'  # 3.1 交接（只读；缺则跳过 as-of lift）

FREEZE_REPORT = ROOT / 'reports/data_gates/G0/s3_b3_ma_param_freeze_teacher.json'   # 只读对账

ART = ROOT / 'artifacts/phase3'

SHADOW_CSV = SHADOW_DIR / 'pc_origrule_p3shadow.csv'

SHADOW_META = SHADOW_DIR / 'pc_origrule_p3shadow.meta.json'

READOUT_CSV = ART / '3.3_origrule_readout.csv'

COUNTS_JSON = ART / '3.3_counts.json'

ANCHOR_PRE = ART / '3.3_anchor_pre.json'

ANCHOR_POST = ART / '3.3_anchor_post.json'

OUTPUT_SHA = ART / '3.3_output_sha.json'

SHADOW_VERSION = 'p3shadow_v1'

RULE_ORIG = 'origrule_minthr_global'

RULE_CLUSTER = 'same_cluster_members'

YEARS = list(range(2014, 2023))                 # 文本年（网络年 y+1）

PC_YEARS = list(range(2017, 2023))              # PC 可得文本年（y−1 年营收 ∈ 面板 FY2016–2024）

SOURCE_VERSION = 'teacher_mda_panel+product_lexicon_v2_teacher'

HEADLINE_K = {2014: 235, 2015: 238, 2016: 322, 2017: 291, 2018: 334,
              2019: 407, 2020: 414, 2021: 434, 2022: 463}

HEADLINE_EDGES = {2014: 157096, 2015: 156974, 2016: 152302, 2017: 214028,
                  2018: 204786, 2019: 238458, 2020: 246728, 2021: 249254,
                  2022: 293938}

HEADLINE_NODES = {2014: 2384, 2015: 2363, 2016: 2503, 2017: 2721, 2018: 2835,
                  2019: 3354, 2020: 3427, 2021: 3570, 2022: 3942}

NOT_DEGENERATE_PEER_MED_MAX = 500.0

NOT_DEGENERATE_LIFT_MIN = 5.0

VOCAB_PROBES = [(2022, 0.25), (2022, 0.002), (2016, 0.002)]

VOCAB_DF_GRID = [0.25, 0.10, 0.05, 0.02, 0.01, 0.005, 0.002]

ORIGINAL_PAPER_EFFECTIVE_WORDS = 16598      # 原文 2016 有效产品词数（3.0_plan §2.3）

SHADOW_CSV_COLS = [
    'shadow_version', 'cluster_source', 'peer_rule', 'focal_id', 'text_year',
    'cluster_id', 'cluster_size', 'n_competitors', 'peer_share_of_market',
    'same_sw2_peer_share_latest', 'thr', 'sw2_lift_latest', 'sw2_lift_asof',
    'base_same_sw2_rate_latest', 'available_at', 'headline_network_version',
    'headline_source_version', 'revenue_year', 'pc', 'hhi', 'n_rev_covered',
    'revenue_coverage', 'pc_status', 'pc_headline', 'pc_matches_headline',
]

READOUT_COLS = [
    'cluster_source', 'peer_rule', 'text_year', 'n_firms', 'n_clusters', 'singleton_share',
    'n_pairs_within_clusters', 'n_zero_sim_within_pairs', 'thr_median', 'thr_p10', 'thr_p90',
    'thr_zero_cluster_share', 'thr_median_non_singleton_firms', 'peer_median', 'peer_mean',
    'peer_p25', 'peer_p90',
    'isolated_share', 'peer_share_of_market_median', 'peer_share_of_market_mean',
    'share_peers_ge_90pct_market', 'same_sw2_edge_share_latest', 'base_same_sw2_rate_latest',
    'sw2_lift_latest', 'same_sw2_edge_share_asof', 'base_same_sw2_rate_asof', 'sw2_lift_asof',
    'overlap_with_cluster_mean', 'overlap_with_cluster_median', 'source_dir',
]

ALLOWED_ROOTS = (SHADOW_DIR, ART)

BANNED_ROOTS = (
    ROOT / 'data/processed/network_ma_teacher',
    P3ASOF_NET,
    ROOT / 'data/processed/network_ma',
    ROOT / 'data/processed/network_mb',
    ROOT / 'data/processed/network_ma_teacher_idf',
    ROOT / 'data/processed/network_ma_teacher_medadj',
    ROOT / 'data/processed/network_ma_teacher_df10',
    ROOT / 'data/processed/network_ma_teacher_df100',
    ROOT / 'data/processed/network_ma_fulltext',
    ROOT / 'data/processed/network_mb_teacher',
    ROOT / 'data/processed/network_ma_teacher_p3yt',
    ROOT / 'data/processed/network_ma_teacher_p3ydf',
    ROOT / 'data/processed/network_ma_teacher_p3yt_ydf',
    ROOT / 'data/processed/network_ma_teacher_p3yt_latest',
    ROOT / 'data/processed/phase2_shadow',
    ROOT / 'data/processed/m1_common_sample_ma_teacher_v2.parquet',
    ROOT / 'reports',
    ROOT / 'configs',
    ROOT / 'scripts',
    ROOT / 'tests',
    ROOT / 'docs',
    ROOT / 'deliverables',
)

READONLY_INPUTS = (TEACHER_LEXICON, REVENUE_PKL, SW2_LATEST, ASOF_MAP_PARQUET)

def assert_output_path_guard(path) -> Path:
    """任何写出路径必须落在 shadow 允许根内，且不命中 headline/G0/configs/只读输入等禁写集合。"""
    extra = [x for x in READONLY_INPUTS if Path(x).exists()]
    return assert_path_allowed(path, ALLOWED_ROOTS, (*BANNED_ROOTS, *extra))

def _atomic_write_text(path, text: str) -> None:
    p = assert_output_path_guard(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, p)

def write_json(path, obj) -> None:
    _atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=1, default=_json_default))

def write_csv(path, rows, header) -> None:
    p = assert_output_path_guard(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    with open(tmp, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    os.replace(tmp, p)

def log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

def within_cluster_min_sim(C: np.ndarray, labels: np.ndarray) -> tuple:
    """原文规则的团内最小相似度 `thr_i`（团内任意两成员对的最小相似度，团内全员共享）。

    - 团规模 ≥ 2：`min_{k≠l ∈ 团} C[k, l]`（对角以 2.0 掩掉，只取团内**无序对**）；
    - 单点团：阈值未定义 → `thr = 1.0`（C ≤ 1 → 无 peer），与 `network_business_cluster_grid.eval_K` /
      `network_business_k_explore.eval_at` 的既有实现一致（留档）。
    返回 `(thr float64 (n,), stats dict)`。
    """
    C = np.asarray(C, dtype=np.float64)
    n = C.shape[0]
    labels = np.asarray(labels)
    thr = np.ones(n, dtype=np.float64)
    n_pairs = 0
    n_zero = 0
    per_cluster_min = []
    n_clusters = 0
    n_singleton = 0
    for k in np.unique(labels):
        idx = np.flatnonzero(labels == k)
        n_clusters += 1
        if len(idx) < 2:
            n_singleton += 1
            continue
        sub = C[np.ix_(idx, idx)].copy()
        np.fill_diagonal(sub, 2.0)
        m = float(sub.min())
        thr[idx] = m
        off = ~np.eye(len(idx), dtype=bool)
        n_pairs += len(idx) * (len(idx) - 1) // 2
        n_zero += int((sub[off] <= 0).sum()) // 2      # 无序对（对称，只数一次）
        per_cluster_min.append(m)
    stats = {
        'n_firms': int(n), 'n_clusters': int(n_clusters), 'n_singleton_clusters': int(n_singleton),
        'singleton_share': round(float(n_singleton / max(1, n_clusters)), 4),
        'n_pairs_within_clusters': int(n_pairs),
        'n_zero_sim_within_pairs': int(n_zero),
        'thr_median': round(float(np.median(thr)), 12),
        'thr_p10': round(float(np.quantile(thr, .1)), 12),
        'thr_p90': round(float(np.quantile(thr, .9)), 12),
        'thr_zero_firm_share': round(float((thr <= 0).mean()), 6),
        'thr_median_non_singleton_clusters': (round(float(np.median(per_cluster_min)), 12)
                                              if per_cluster_min else None),
    }
    return thr, stats

def orig_rule_mask(C: np.ndarray, labels: np.ndarray) -> tuple:
    """原文第四步字面规则：`mask_ij = C_ij > thr_i`（**严格大于**；候选域 = 全部节点；对角排除）。

    返回 `(mask bool (n,n), thr, stats)`；stats 内记录团外有向边数（候选域 = 全部节点的实现证据）。
    """
    C = np.asarray(C, dtype=np.float64)
    thr, stats = within_cluster_min_sim(C, labels)
    mask = C > thr[:, None]
    np.fill_diagonal(mask, False)
    labels = np.asarray(labels)
    same_cluster = labels[:, None] == labels[None, :]
    stats['n_directed_edges_cross_cluster'] = int((mask & ~same_cluster).sum())
    stats['candidate_domain'] = 'all_nodes'
    return mask, thr, stats

def cluster_rule_mask(labels: np.ndarray) -> np.ndarray:
    """项目规则对照：同团成员（含自身对角 True；peer 数 = 团规模 − 1）。"""
    labels = np.asarray(labels)
    return labels[:, None] == labels[None, :]

def edge_share_lift(mask: np.ndarray, l2_idx: np.ndarray) -> dict:
    """有向边份额 / 边际基线 lift（与 `network_business_cluster_grid.eval_K` / `network_business_k_explore.eval_at` 同式）。

    对称 mask（不含对角）下，有向份额与无序对数占比**逐值相等**；
    `base` = 全样本同申万二级无序对占比（含自身对角 True 减去 n）。
    """
    mask = np.asarray(mask, dtype=bool)
    l2_idx = np.asarray(l2_idx)
    n = mask.shape[0]
    same = l2_idx[:, None] == l2_idx[None, :]
    tot = int(mask.sum())
    sm = int((mask & same).sum())
    base = float((same.sum() - n) / max(1, n * (n - 1)))
    share = sm / max(1, tot)
    return {'n_directed_edges': tot, 'n_same_sw2_directed_edges': sm,
            'same_sw2_edge_share': round(share, 4), 'base_same_sw2_rate': round(base, 5),
            'sw2_lift': round(share / max(1e-9, base), 2)}

def peer_stats(peers: np.ndarray, same_l2_count: np.ndarray | None = None) -> dict:
    """peer 数分位 / 占市场比 / 隔离率（可选：行内同申万二级 peer 份额）。"""
    peers = np.asarray(peers, dtype=np.float64)
    n = len(peers)
    out = {
        'n_firms': int(n),
        'peer_median': float(np.median(peers)),
        'peer_mean': round(float(peers.mean()), 1),
        'peer_p25': float(np.quantile(peers, .25)),
        'peer_p90': float(np.quantile(peers, .9)),
        'isolated_share': round(float((peers == 0).mean()), 4),
        'peer_share_of_market_median': round(float(np.median(peers / max(1, n - 1))), 6),
        'peer_share_of_market_mean': round(float((peers / max(1, n - 1)).mean()), 6),
        'share_peers_ge_90pct_market': round(float((peers >= 0.9 * max(1, n - 1)).mean()), 6),
    }
    if same_l2_count is not None and peers.size:
        with np.errstate(invalid='ignore', divide='ignore'):
            share = np.where(peers > 0, np.asarray(same_l2_count, dtype=float) / np.maximum(peers, 1),
                             np.nan)
        out['same_sw2_peer_share_mean'] = round(float(np.nanmean(share)), 6)
        out['same_sw2_peer_share_median'] = round(float(np.nanmedian(share)), 6)
    return out

def hhi_pc(member_vals: list, focal_covered: bool) -> dict:
    """`network_business_competition_hhi` 同式：HHI = Σ(v/Σv)²（仅覆盖成员入分母）；PC = −HHI。

    缺营收（None）或 ≤0 计缺失；覆盖 <2 或 `Σv ≤ 0` 或 focal 缺营收 → pc 空（不填 0）。
    """
    vals = [float(v) for v in member_vals if v is not None and float(v) > 0]
    n_covered = len(vals)
    cov = round(n_covered / max(1, len(member_vals)), 4)
    if n_covered < 2 or not focal_covered:
        return {'hhi': None, 'pc': None, 'n_rev_covered': n_covered - (1 if focal_covered else 0),
                'revenue_coverage': cov}
    tot = float(sum(vals))
    if tot <= 0:
        return {'hhi': None, 'pc': None, 'n_rev_covered': n_covered - 1, 'revenue_coverage': cov}
    hhi = float(sum((v / tot) ** 2 for v in vals))
    return {'hhi': round(hhi, 8), 'pc': round(-hhi, 8), 'n_rev_covered': n_covered - 1,
            'revenue_coverage': cov}

def degeneracy_verdict(year_rows: list) -> dict:
    """由逐年原文规则读数（headline 来源）判退化（3.0_plan §2.3 停机判据的互斥面）。

    判据：**未退化** = (peer 中位 < 500) 且 (lift > 5)；任一不成立 → 退化。
    """
    peer_med = [r['peer_median'] for r in year_rows]
    lifts = [r['sw2_lift_latest'] for r in year_rows]
    med_peer = float(np.median(peer_med))
    med_lift = float(np.median(lifts))
    not_degenerate = bool(med_peer < NOT_DEGENERATE_PEER_MED_MAX and med_lift > NOT_DEGENERATE_LIFT_MIN)
    return {
        'degenerate': bool(not not_degenerate),
        'median_of_year_peer_medians': med_peer,
        'median_of_year_lifts_latest': med_lift,
        'peer_median_by_year': {int(r['text_year']): r['peer_median'] for r in year_rows},
        'peer_mean_by_year': {int(r['text_year']): r['peer_mean'] for r in year_rows},
        'peer_share_of_market_median_by_year': {int(r['text_year']): r['peer_share_of_market_median']
                                                for r in year_rows},
        'share_focals_ge_90pct_market_by_year': {int(r['text_year']): r['share_peers_ge_90pct_market']
                                                 for r in year_rows},
        'sw2_lift_latest_by_year': {int(r['text_year']): r['sw2_lift_latest'] for r in year_rows},
        'sw2_lift_asof_by_year': {int(r['text_year']): r['sw2_lift_asof'] for r in year_rows},
        'rule': f'未退化 = peer 中位 < {NOT_DEGENERATE_PEER_MED_MAX:.0f} 且 lift > '
                f'{NOT_DEGENERATE_LIFT_MIN:.1f}（任一不成立即退化）',
    }

def load_rev(path=REVENUE_PKL) -> dict:
    """`network_business_competition_hhi.load_rev` 同式：{年: {code6: 正营收}}（缺失/非正不入字典）。"""
    d = pd.read_pickle(path)
    d.index = [ts.strftime('%Y') if hasattr(ts, 'strftime') else str(ts)[:4] for ts in d.index]
    rev = {}
    for y, row in d.iterrows():
        rev[str(y)] = {c.split('.')[0]: float(v) for c, v in row.items() if v == v and v > 0}
    return rev

def latest_l2_map() -> dict:
    """headline 静态标签（`sw2_industry_latest.csv`，= `load_sw2()` 口径）→ firm_id: l2_code。"""
    lat = pd.read_csv(SW2_LATEST, dtype={'code6': str, 'firm_id': str, 'l2_code': str})
    lat = lat[lat['firm_id'].notna() & lat['code6'].notna()]
    lat['code6'] = lat['code6'].str.zfill(6)
    return dict(zip(lat['firm_id'], lat['l2_code']))

def asof_l2_map_for_key(key: str) -> dict:
    """as-of 标签（`effective_from <= key`，3.1 映射长表只读）→ firm_id: l2_code。缺件 → {}。"""
    if not ASOF_MAP_PARQUET.is_file():
        return {}
    m = pd.read_parquet(ASOF_MAP_PARQUET,
                        columns=['code6', 'effective_from', 'l2_code', 'firm_id'])
    m = m[m['firm_id'].notna() & (pd.to_datetime(m['effective_from']) <= pd.Timestamp(key))]
    m = m.sort_values('effective_from', kind='stable')
    last = m.drop_duplicates('code6', keep='last')
    return dict(zip(last['firm_id'], last['l2_code']))

def l2_index(firms: list, mapping: dict) -> np.ndarray:
    """firm 序列 → l2 整数编码（缺失 → 'NA'）。"""
    arr = np.array([str(mapping.get(f, 'NA')) for f in firms])
    _, idx = np.unique(arr, return_inverse=True)
    return idx

def matrix_vs_headline_edges(C: np.ndarray, firms: list, y: int) -> dict:
    """R1：全相似度矩阵与 headline `{y}_similarity.csv`（同团正相似对有向边表）逐对核对。"""
    p = HEADLINE_NET / f'{y}_similarity.csv'
    e = pd.read_csv(p, usecols=['focal_id', 'source_id', 'similarity'],
                    dtype={'focal_id': str, 'source_id': str})
    pos = {f: i for i, f in enumerate(firms)}
    ii = e['focal_id'].map(pos)
    jj = e['source_id'].map(pos)
    require(bool(ii.notna().all()) and bool(jj.notna().all()),
            f'[{y}] headline 边表存在矩阵外节点（firms 对齐失败）')
    ii = ii.to_numpy(dtype=np.int64)
    jj = jj.to_numpy(dtype=np.int64)
    dif = np.abs(C[ii, jj] - e['similarity'].to_numpy(dtype=np.float64))
    max_dif = float(dif.max()) if len(dif) else 0.0
    require(max_dif <= 1e-6,
            f'[{y}] 逐对核对失败：矩阵与 headline similarity.csv 最大差 {max_dif:.3e} > 1e-6')
    return {'n_pairs': int(len(dif)), 'max_abs_diff': max_dif,
            'headline_similarity_file': p.relative_to(ROOT).as_posix()}

def read_year(y: int, A, uni, l2m: dict, rev: dict, vocab_stats: dict) -> dict:
    """一年：矩阵（R1 对账）→ 原文/同团规则读数（headline + 可选 p3asof）→ PC → shadow 行。"""
    t0 = time.time()
    firms, X, stats = build_vectors(y, A, uni, corpus='teacher')
    vocab_stats[str(y)] = stats
    Xn = normalize(X)
    C = np.asarray((Xn @ Xn.T).todense(), dtype=np.float64)
    np.fill_diagonal(C, 1.0)
    del Xn
    chk = matrix_vs_headline_edges(C, firms, y)

    l2_lat = l2_index(firms, l2m)
    asof_map = asof_l2_map_for_key(f'{y + 1}-04-30')
    l2_asof = l2_index(firms, asof_map) if asof_map else None

    rep = json.loads((HEADLINE_NET / f'{y}_build_report.json').read_text(encoding='utf-8'))
    require(str(rep['available_at']) == f'{y + 1}-04-30',
            f'[{y}] headline available_at={rep["available_at"]} != {y + 1}-04-30')
    require(int(rep['k_clusters']) == HEADLINE_K[y] and int(rep['n_edges']) == HEADLINE_EDGES[y]
            and int(rep['n_nodes']) == HEADLINE_NODES[y],
            f'[{y}] headline 登记读数（K/边/节点）与 build_report 不符：{rep["k_clusters"]}/'
            f'{rep["n_edges"]}/{rep["n_nodes"]}')

    sources = ['headline'] + (['p3asof'] if P3ASOF_NET.is_dir() else [])
    readout_rows, detail = [], {}
    for src in sources:
        ndir = HEADLINE_NET if src == 'headline' else P3ASOF_NET
        clu = pd.read_csv(ndir / 'clusters' / f'{y}_clusters.csv', dtype={'firm_id': str})
        c2k = dict(zip(clu['firm_id'], clu['cluster_id']))
        require(len(clu) == len(firms) and all(f in c2k for f in firms),
                f'[{y}/{src}] 聚类标签与向量样本不对齐（{len(clu)} vs {len(firms)}）')
        labels = np.array([c2k[f] for f in firms])
        inv = pd.factorize(labels)[0]
        sizes = np.bincount(inv)
        n = len(firms)

        mask, thr, ostats = orig_rule_mask(C, labels)
        require(int(ostats['n_directed_edges_cross_cluster']) > 0,
                f'[{y}/{src}] 原文规则未产生任何团外候选（候选域 ≠ 全部节点）；停机')
        same_cluster = cluster_rule_mask(labels)
        peers = mask.sum(axis=1)
        same_l2_cnt = (mask & (l2_lat[:, None] == l2_lat[None, :])).sum(axis=1).astype(np.int64)
        lift_lat = edge_share_lift(mask, l2_lat)
        lift_asof = edge_share_lift(mask, l2_asof) if l2_asof is not None else None
        ps = peer_stats(peers, same_l2_cnt)
        cl_peer = sizes[inv] - 1
        ov = np.where(cl_peer > 0, (mask & same_cluster).sum(axis=1) / np.maximum(cl_peer, 1), np.nan)
        row = {'cluster_source': src, 'peer_rule': RULE_ORIG, 'text_year': y,
               'n_firms': n, 'n_clusters': ostats['n_clusters'],
               'singleton_share': ostats['singleton_share'],
               'n_pairs_within_clusters': ostats['n_pairs_within_clusters'],
               'n_zero_sim_within_pairs': ostats['n_zero_sim_within_pairs'],
               'thr_median': ostats['thr_median'], 'thr_p10': ostats['thr_p10'],
               'thr_p90': ostats['thr_p90'],
               'thr_zero_cluster_share': ostats['thr_zero_firm_share'],
               'thr_median_non_singleton_firms': (round(float(np.median(thr[cl_peer >= 1])), 12)
                                                  if int((cl_peer >= 1).sum()) else None),
               **{k: ps[k] for k in ('peer_median', 'peer_mean', 'peer_p25', 'peer_p90',
                                     'isolated_share', 'peer_share_of_market_median',
                                     'peer_share_of_market_mean', 'share_peers_ge_90pct_market')},
               'same_sw2_edge_share_latest': lift_lat['same_sw2_edge_share'],
               'base_same_sw2_rate_latest': lift_lat['base_same_sw2_rate'],
               'sw2_lift_latest': lift_lat['sw2_lift'],
               'same_sw2_edge_share_asof': (lift_asof or {}).get('same_sw2_edge_share'),
               'base_same_sw2_rate_asof': (lift_asof or {}).get('base_same_sw2_rate'),
               'sw2_lift_asof': (lift_asof or {}).get('sw2_lift'),
               'overlap_with_cluster_mean': round(float(np.nanmean(ov)), 6),
               'overlap_with_cluster_median': round(float(np.nanmedian(ov)), 6),
               'source_dir': ndir.relative_to(ROOT).as_posix()}
        readout_rows.append(row)

        cl_mask = same_cluster & ~np.eye(n, dtype=bool)
        cl_lift_lat = edge_share_lift(cl_mask, l2_lat)
        cl_lift_asof = edge_share_lift(cl_mask, l2_asof) if l2_asof is not None else None
        cl_same_l2_cnt = (same_cluster & (l2_lat[:, None] == l2_lat[None, :])).sum(axis=1) - 1
        cps = peer_stats(cl_peer, cl_same_l2_cnt)
        readout_rows.append({
            'cluster_source': src, 'peer_rule': RULE_CLUSTER, 'text_year': y,
            'n_firms': n, 'n_clusters': ostats['n_clusters'],
            'singleton_share': ostats['singleton_share'],
            'n_pairs_within_clusters': ostats['n_pairs_within_clusters'],
            'n_zero_sim_within_pairs': ostats['n_zero_sim_within_pairs'],
            'thr_median': None, 'thr_p10': None, 'thr_p90': None, 'thr_zero_cluster_share': None,
            'thr_median_non_singleton_firms': None,
            **{k: cps[k] for k in ('peer_median', 'peer_mean', 'peer_p25', 'peer_p90',
                                   'isolated_share', 'peer_share_of_market_median',
                                   'peer_share_of_market_mean', 'share_peers_ge_90pct_market')},
            'same_sw2_edge_share_latest': cl_lift_lat['same_sw2_edge_share'],
            'base_same_sw2_rate_latest': cl_lift_lat['base_same_sw2_rate'],
            'sw2_lift_latest': cl_lift_lat['sw2_lift'],
            'same_sw2_edge_share_asof': (cl_lift_asof or {}).get('same_sw2_edge_share'),
            'base_same_sw2_rate_asof': (cl_lift_asof or {}).get('base_same_sw2_rate'),
            'sw2_lift_asof': (cl_lift_asof or {}).get('sw2_lift'),
            'overlap_with_cluster_mean': None, 'overlap_with_cluster_median': None,
            'source_dir': ndir.relative_to(ROOT).as_posix()})
        detail[src] = {'labels': labels, 'inv': inv, 'sizes': sizes, 'thr': thr,
                       'mask_orig': mask, 'same_cluster': same_cluster, 'peers': peers,
                       'cl_peer': cl_peer, 'lift_lat': lift_lat, 'cl_lift_lat': cl_lift_lat,
                       'lift_asof': lift_asof, 'cl_lift_asof': cl_lift_asof, 'l2_lat': l2_lat,
                       'n_clusters': ostats['n_clusters'],
                       'peer_median': row['peer_median'], 'cl_peer_median': cps['peer_median'],
                       'sw2_lift_latest': row['sw2_lift_latest'],
                       'cl_sw2_lift_latest': cl_lift_lat['sw2_lift']}
        log(f'[{y}/{src}] n={n} 团={ostats["n_clusters"]} thr中位={ostats["thr_median"]:.4f} '
            f'原文peer中位={row["peer_median"]:.0f}（占市场 {row["peer_share_of_market_median"]:.3f}）'
            f'lift={row["sw2_lift_latest"]} | 同团peer中位={cps["peer_median"]:.0f} '
            f'lift={cl_lift_lat["sw2_lift"]}')

    # ---- PC 对照（原文规则 vs headline 同团；headline 侧逐值对账）
    pc, pc_summary = None, None
    cl = detail['headline']
    if y in PC_YEARS:
        n = len(firms)
        ry = str(y - 1)
        rmap = rev.get(ry, {})
        require(bool(rmap), f'[{y}] 营收年 {ry} 不在营收面板内（PC 应可得）')
        r = np.array([rmap.get(f[2:], np.nan) for f in firms], dtype=np.float64)
        focal_cov = ~np.isnan(r)
        hp = pd.read_csv(HEADLINE_NET / f'{y}_pc.csv',
                         dtype={'focal_id': str, 'revenue_year': str})
        require(len(hp) == n and set(hp['focal_id']) == set(firms),
                f'[{y}] headline pc.csv 行集与向量样本不一致')
        require(bool((hp['revenue_year'] == ry).all()), f'[{y}] headline pc.csv revenue_year != {ry}')
        hp = hp.set_index('focal_id').reindex(firms)
        pc_head = pd.to_numeric(hp['pc'], errors='coerce').to_numpy()
        hhi_head = pd.to_numeric(hp['hhi'], errors='coerce').to_numpy()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            R = np.where(cl['mask_orig'], r[None, :], np.nan)
            tot = np.nansum(R, axis=1) + np.where(focal_cov, r, 0.0)
            with np.errstate(invalid='ignore', divide='ignore'):
                hhi_o = np.nansum((R / tot[:, None]) ** 2, axis=1) + (r / tot) ** 2
            ok_o = (cl['peers'] >= 1) & focal_cov & (tot > 0)
            hhi_o = np.where(ok_o, hhi_o, np.nan)
            n_cov_o = np.sum(~np.isnan(R), axis=1) + focal_cov.astype(np.int64)
            Rc = np.where(cl['same_cluster'], r[None, :], np.nan)
            totc = np.nansum(Rc, axis=1)
            with np.errstate(invalid='ignore', divide='ignore'):
                hhi_c = np.nansum((Rc / totc[:, None]) ** 2, axis=1)
            ok_c = (cl['cl_peer'] >= 1) & focal_cov & (totc > 0)
            hhi_c = np.where(ok_c, hhi_c, np.nan)
            n_cov_c = np.sum(~np.isnan(Rc), axis=1)
        require(bool(((~np.isnan(hhi_c)) == (~np.isnan(hhi_head))).all()),
                f'[{y}] 同团规则 PC 缺失模式与 headline pc.csv 不一致')
        both = ~np.isnan(hhi_c) & ~np.isnan(hhi_head)
        max_dif = float(np.max(np.abs(hhi_c[both] - hhi_head[both]))) if both.any() else 0.0
        require(max_dif <= 1e-6,
                f'[{y}] 同团规则 PC 重算与 headline pc.csv 不符（最大差 {max_dif:.3e}）')
        pc = {'revenue_year': ry, 'orig': {'pc': -hhi_o, 'hhi': hhi_o, 'n_cov': n_cov_o},
              'cluster': {'pc': -hhi_c, 'hhi': hhi_c, 'n_cov': n_cov_c},
              'headline': {'pc': pc_head, 'hhi': hhi_head}, 'focal_cov': focal_cov}
        pc_summary = {'text_year': y, 'revenue_year': ry, 'n_firms': n,
                      'orig': {'n_firms_with_pc': int(np.sum(~np.isnan(hhi_o))),
                               'n_competitors_median': float(np.median(cl['peers'])),
                               'pc_median': round(float(np.nanmedian(-hhi_o)), 6),
                               'pc_mean': round(float(np.nanmean(-hhi_o)), 6),
                               'pc_p10': round(float(np.nanquantile(-hhi_o, .1)), 6),
                               'pc_p90': round(float(np.nanquantile(-hhi_o, .9)), 6),
                               'revenue_coverage_median': round(float(np.median(
                                   n_cov_o / (cl['peers'] + 1))), 4)},
                      'cluster_recompute': {'n_firms_with_pc': int(np.sum(~np.isnan(hhi_c))),
                                            'pc_median': round(float(np.nanmedian(-hhi_c)), 6),
                                            'pc_mean': round(float(np.nanmean(-hhi_c)), 6)},
                      'headline_file': {'n_firms_with_pc': int(np.sum(~np.isnan(pc_head))),
                                        'pc_median': round(float(np.nanmedian(pc_head)), 6),
                                        'pc_mean': round(float(np.nanmean(pc_head)), 6),
                                        'pc_p10': round(float(np.nanquantile(pc_head, .1)), 6),
                                        'pc_p90': round(float(np.nanquantile(pc_head, .9)), 6)},
                      'match': {'n_both_nonmissing': int(both.sum()), 'max_abs_diff_hhi': max_dif,
                                'all_match_tol_1e-6': bool(max_dif <= 1e-6)}}
        if both.sum() >= 2 and np.nanstd(pc_head[both]) > 0 and np.nanstd(-hhi_o[both]) > 0:
            mb = both & ~np.isnan(hhi_o)      # 相关只在两列都非缺的行上算（原文规则含 thr=1 → 无 peer 的行）
            if mb.sum() >= 2 and np.nanstd(pc_head[mb]) > 0 and np.nanstd(-hhi_o[mb]) > 0:
                pc_summary['corr'] = {
                    'orig_vs_headline': round(float(np.corrcoef(-hhi_o[mb], pc_head[mb])[0, 1]), 6),
                    'abs_magnitude_ratio_median': round(float(
                        np.median(np.abs(hhi_o[mb])) / max(1e-12, np.median(np.abs(pc_head[mb])))), 6),
                    'n': int(mb.sum())}
        log(f'[{y}] PC 对照：原文规则 中位={np.nanmedian(-hhi_o):.6f}（n={int(ok_o.sum())}）'
            f' vs headline 同团 中位={np.nanmedian(pc_head):.6f}；同团重算对账 max|Δhhi|={max_dif:.2e}')

    # ---- shadow 行（仅 headline 来源）
    m1_obj = {'sw2_lift_latest': cl['lift_lat']['sw2_lift'],
              'cl_sw2_lift_latest': cl['cl_lift_lat']['sw2_lift'],
              'sw2_lift_asof': (cl['lift_asof'] or {}).get('sw2_lift'),
              'cl_sw2_lift_asof': (cl['cl_lift_asof'] or {}).get('sw2_lift'),
              'base_latest': cl['lift_lat']['base_same_sw2_rate']}
    shadow_rows = build_shadow_rows(y, firms, cl, rep, pc, m1_obj)
    del C, detail
    return {'readout_rows': readout_rows, 'shadow_rows': shadow_rows, 'pc_summary': pc_summary,
            'matrix_check': chk, 'sec': round(time.time() - t0, 1)}

def build_shadow_rows(y: int, firms: list, cl: dict, rep: dict, pc, m1_obj) -> list:
    """逐 focal × 规则 → shadow CSV 行（原文规则 + 同团对照；PC 仅 2017–2022）。"""
    n = len(firms)
    labels = cl['labels']
    inv, sizes = cl['inv'], cl['sizes']
    thr, mask, same_cluster, l2_lat = cl['thr'], cl['mask_orig'], cl['same_cluster'], cl['l2_lat']
    avail, hver = str(rep['available_at']), str(rep['network_version'])
    ry = pc['revenue_year'] if pc is not None else str(y - 1)
    same_l2 = l2_lat[:, None] == l2_lat[None, :]
    cl_peer = sizes[inv] - 1
    out = []
    for rule in (RULE_ORIG, RULE_CLUSTER):
        if rule == RULE_ORIG:
            peers_arr = mask.sum(axis=1)
            cnt_arr = (mask & same_l2).sum(axis=1)
            pc_arr = pc['orig']['pc'] if pc is not None else None
            ncov_arr = pc['orig']['n_cov'] if pc is not None else None
        else:
            peers_arr = cl_peer
            cnt_arr = (same_cluster & same_l2).sum(axis=1) - 1
            pc_arr = pc['cluster']['pc'] if pc is not None else None
            ncov_arr = pc['cluster']['n_cov'] if pc is not None else None
        for i, f in enumerate(firms):
            if pc is None:
                status = 'revenue_year_not_in_panel'
            elif not bool(pc['focal_cov'][i]):
                status = 'focal_revenue_missing'
            elif peers_arr[i] < 1:
                status = 'singleton_cluster' if rule == RULE_CLUSTER else 'no_peer_above_thr'
            else:
                status = 'support'
            pv = pc_arr[i] if pc is not None else np.nan
            hv = -pv if pc is not None else np.nan
            fc = bool(pc['focal_cov'][i]) if pc is not None else False
            pc_head = pc['headline']['pc'][i] if pc is not None else np.nan
            nrev = int(ncov_arr[i] - (1 if pc is not None and fc else 0)) if pc is not None else ''
            cov = (round(float(ncov_arr[i]) / max(1, int(peers_arr[i]) + 1), 4) if rule == RULE_ORIG
                   else round(float(ncov_arr[i]) / max(1, int(sizes[inv[i]])), 4)) if pc is not None else ''
            match = ''
            if pc is not None and rule == RULE_CLUSTER:
                # 对账：本文件重算 **pc**（= −hhi）vs headline pc.csv 的 pc（两者同为 PC 口径）
                match = int((not np.isnan(pv)) and (not np.isnan(pc_head))
                            and abs(pv - pc_head) <= 1e-6)
            out.append([
                SHADOW_VERSION, 'headline', rule, f, y, int(labels[i]), int(sizes[inv[i]]),
                int(peers_arr[i]), round(float(peers_arr[i] / max(1, n - 1)), 6),
                (round(float(cnt_arr[i] / peers_arr[i]), 6) if peers_arr[i] > 0 else ''),
                (round(float(thr[i]), 12) if rule == RULE_ORIG else ''),
                m1_obj['sw2_lift_latest'] if rule == RULE_ORIG else m1_obj['cl_sw2_lift_latest'],
                (m1_obj['sw2_lift_asof'] if rule == RULE_ORIG else m1_obj['cl_sw2_lift_asof']),
                m1_obj['base_latest'], avail, hver, SOURCE_VERSION, ry,
                ('' if (pv is None or np.isnan(pv)) else round(float(pv), 8)),
                ('' if (hv is None or np.isnan(hv)) else round(float(hv), 8)),
                nrev, cov, status,
                ('' if pc is None or np.isnan(pc_head) else round(float(pc_head), 8)),
                match,
            ])
    return out

def sha_ledger_dir(d: Path) -> dict:
    out = {}
    for p in sorted(Path(d).rglob('*')):
        if p.is_file():
            out[p.relative_to(ROOT).as_posix()] = sha256_file(p)
    return out

def anchors_snapshot(tag: str) -> dict:
    ledger = assert_m1_inputs_sha()
    extra = {k: {'file': str(Path(v).relative_to(ROOT)).replace('\\', '/'), 'sha256': sha256_file(v)}
             for k, v in {'product_lexicon_v2_teacher': TEACHER_LEXICON,
                          'operating_revenue_annual': REVENUE_PKL,
                          'sw2_industry_latest': SW2_LATEST,
                          'sw2_asof_map_p3shadow': ASOF_MAP_PARQUET}.items() if Path(v).is_file()}
    snap = {
        'tag': tag, 'run_at': datetime.now(timezone.utc).isoformat(),
        'm1_inputs_9': {r['input']: r['sha256'] for r in ledger},
        'm1_inputs_ledger': ledger,
        'extra_inputs': extra,
        'headline_network_sha': sha_ledger_dir(HEADLINE_NET),
        'p3asof_network_sha': sha_ledger_dir(P3ASOF_NET) if P3ASOF_NET.is_dir() else {},
        'note': 'headline 网络目录（data/processed/network_ma_teacher/）全文件 sha + 3.1 产物目录只读复验；'
                '本步不写任何 headline/3.1 文件（输出路径守卫硬门）',
    }
    snap['headline_network_n_files'] = len(snap['headline_network_sha'])
    return snap

def compare_anchors(pre: dict, post: dict) -> dict:
    diffs = []
    for sect in ('m1_inputs_9', 'extra_inputs', 'headline_network_sha', 'p3asof_network_sha'):
        a, b = pre.get(sect) or {}, post.get(sect) or {}
        if isinstance(a, dict) and a and isinstance(list(a.values())[0], dict):
            a = {k: v.get('sha256') for k, v in a.items()}
            b = {k: v.get('sha256') for k, v in b.items()}
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                diffs.append({'section': sect, 'key': k, 'pre': a.get(k), 'post': b.get(k)})
    return {'status': 'MATCH' if not diffs else 'CHANGED', 'n_diff': len(diffs), 'diffs': diffs}

def output_sha_ledger() -> dict:
    """本次 shadow 产物 sha；若存在上次的台账，附数据产物（CSV/meta/readout）逐字节复现对照。"""
    files = [SHADOW_CSV, SHADOW_META, READOUT_CSV, ANCHOR_PRE, ANCHOR_POST]
    sha, absent = {}, []
    for p in files:
        if Path(p).is_file():
            sha[Path(p).relative_to(ROOT).as_posix()] = sha256_file(p)
        else:
            absent.append(Path(p).relative_to(ROOT).as_posix())
    out = {'n_files': len(sha), 'absent_files': absent, 'sha': sha,
           'note': 'anchor_{pre,post}.json 含 run_at → 逐次不同；数据产物应逐字节一致'}
    if OUTPUT_SHA.is_file():
        try:
            prior = json.loads(OUTPUT_SHA.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            prior = None
        if isinstance(prior, dict) and isinstance(prior.get('sha'), dict):
            data_files = [Path(SHADOW_CSV).relative_to(ROOT).as_posix(),
                          Path(SHADOW_META).relative_to(ROOT).as_posix(),
                          Path(READOUT_CSV).relative_to(ROOT).as_posix()]
            cmp_ = {k: {'prior_sha': prior['sha'].get(k), 'sha': sha.get(k),
                        'match': prior['sha'].get(k) == sha.get(k)} for k in data_files}
            out['data_artifact_reproducibility'] = {
                'prior_ledger_n_files': prior.get('n_files'),
                'per_file': cmp_,
                'all_match': all(v['match'] for v in cmp_.values())}
    return out

def vocab_sensitivity(probe: bool) -> dict:
    """词表/active_vocab 对 df 阈值的敏感度（对表 3.0_plan §2.3：窄词表路线不可行）。"""
    d = pd.read_csv(TEACHER_LEXICON, usecols=['term', 'df_ratio'])
    d['df_ratio'] = pd.to_numeric(d['df_ratio'], errors='coerce').fillna(0.0)
    total_rows = int(len(d))
    kept = {str(th): int(d.loc[d['df_ratio'] <= th, 'term'].nunique()) for th in VOCAB_DF_GRID}
    freeze = json.loads(FREEZE_REPORT.read_text(encoding='utf-8')) if FREEZE_REPORT.is_file() else {}
    freeze_av = {y: freeze.get('years', {}).get(y, {}).get('filter_stats', {}).get('active_vocab')
                 for y in freeze.get('years', {})} if freeze else {}
    measured = []
    if probe:
        for y, dfm in VOCAB_PROBES:
            terms, lmeta = load_lexicon(TEACHER_LEXICON, df_max=dfm)
            A = build_automaton(terms)
            uni = load_refs()
            firms, _X, st = build_vectors(y, A, uni, corpus='teacher', df_max=dfm)
            measured.append({'text_year': y, 'df_max': dfm, 'terms_kept': lmeta['terms_kept'],
                             'n_firms': len(firms), 'active_vocab': st['active_vocab'],
                             'mean_terms': st['mean_terms']})
            del A, uni, _X
            log(f'[vocab-probe] {y} df≤{dfm}: active_vocab={st["active_vocab"]:,} '
                f'mean_terms={st["mean_terms"]}')
    return {
        'lexicon': str(TEACHER_LEXICON.relative_to(ROOT)).replace('\\', '/'),
        'terms_total_rows': total_rows,
        'terms_kept_unique_at_df_max': kept,
        'kept_share_at_df_max': {k: round(v / max(1, total_rows), 4) for k, v in kept.items()},
        'df_rule': 'df_ratio <= df_max（load_lexicon 同式；df_ratio = 9 年上确界 max_y doc_count_y / n_docs_y）',
        'active_vocab_headline_df025_by_year': freeze_av,
        'measured_active_vocab_probes': measured,
        'original_paper_effective_words_fy2016': ORIGINAL_PAPER_EFFECTIVE_WORDS,
        'plan_reported_active_vocab_df_grid_2022': {'0.25': 145100, '0.10': 143400, '0.05': 141300,
                                                    '0.02': 136900, '0.01': 131600, '0.005': 123800,
                                                    '0.002': 108300},
        'plan_reported_active_vocab_df_grid_2016': {'0.25': 89700, '0.002': 66000},
        'plan_source': 'artifacts/phase3/3.0_plan.md §2.3（Phase 3 规划期实测，本次仅引用；'
                       '本步实测探针值见 measured_active_vocab_probes，与 plan 引用值同量级但非逐值相同——'
                       '口径差异未追（不影响结论：两口径均 ≥2.6× 原文词表规模））',
        'conclusion': 'df 阈值从 0.25 收紧到 0.002 只减 ~3.2% 词条（1,353,479→1,312,192）；'
                      '本步实测 active_vocab：2022 年 132,968（df≤0.25）→ 91,701（df≤0.002）、'
                      '2016 年 → 43,078（mean_terms 444.8→29.7，向量被阈值破坏）——即最严阈值下仍约 '
                      '2.6×（2016）/5.5×（2022）原文 ~1.66 万量级，且以「每文档命中词数崩塌」为代价；'
                      '用 df 微调不可能到原文词表规模，要到原文量级须**重建词表本身**'
                      '（换词源/加产品性过滤），不在 Phase 3 范围。',
    }


# --- scripts/lib bootstrap ---
# --- end scripts/lib bootstrap ---




# ---------------------------------------------------------------- 路径与常量



# headline 登记读数（只读对账；`3.0_plan.md` §1.1）
# 退化停机判据（3.0_plan §2.3）：peer 中位 < 500 **且** lift > 5 = 未退化 → 停机升级项目决策
# 词表敏感度探针（(文本年, df_max) → build_vectors 实测 active_vocab；对表 3.0_plan §2.3 读数）


# ---------------------------------------------------------------- 输出路径守卫（同 3.1/3.2 型）












# ---------------------------------------------------------------- 纯函数（合成可测）














# ---------------------------------------------------------------- 只读装载










# ---------------------------------------------------------------- 逐年读数




# ---------------------------------------------------------------- 锚台账








# ---------------------------------------------------------------- vocab 敏感度


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description='Phase 3.3 原文第四步阈值规则 shadow（最小证据 + PC 对照）')
    ap.add_argument('--steps', default='all',
                    help='all 或逗号分隔子集：preflight,readout,vocab,accept,postflight')
    ap.add_argument('--years', type=int, nargs='*', default=None, help='readout 文本年（缺省全 9 年）')
    ap.add_argument('--net-dir', default=None, help='headline 网络目录（缺省 data/processed/network_ma_teacher）')
    ap.add_argument('--no-vocab-probe', action='store_true',
                    help='跳过 active_vocab 实测探针（只做词表层 df 阈值表）')
    args = ap.parse_args()
    global HEADLINE_NET
    if args.net_dir:
        HEADLINE_NET = Path(args.net_dir)
    steps = (['preflight', 'readout', 'vocab', 'accept', 'postflight']
             if args.steps == 'all' else [s.strip() for s in args.steps.split(',') if s.strip()])
    years = [int(y) for y in (args.years or YEARS)]
    t_all = time.time()
    SHADOW_DIR.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    counts = {'run_at': datetime.now(timezone.utc).isoformat(), 'steps': steps,
              'shadow_version': SHADOW_VERSION, 'years': years, 'pc_years': PC_YEARS,
              'run_history': []}
    if COUNTS_JSON.is_file():
        try:
            prev = json.loads(COUNTS_JSON.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            prev = None
        if isinstance(prev, dict):
            hist = prev.get('run_history') if isinstance(prev.get('run_history'), list) else []
            hist = hist + [{'run_at': prev.get('run_at'), 'steps': prev.get('steps'),
                            'elapsed_sec_total': prev.get('elapsed_sec_total')}]
            prev = {k: v for k, v in prev.items() if k != 'run_history'}
            prev.update({'run_at': counts['run_at'], 'steps': steps, 'years': years,
                         'run_history': hist})
            counts = prev

    if 'preflight' in steps:
        counts['anchor_pre'] = anchors_snapshot('pre')
        write_json(ANCHOR_PRE, counts['anchor_pre'])
        log(f'preflight：9 锚 + {counts["anchor_pre"]["headline_network_n_files"]} 个 headline 网络文件 sha'
            f' + {len(counts["anchor_pre"]["p3asof_network_sha"])} 个 3.1 产物 sha 已记录')

    if 'readout' in steps:
        terms, lmeta = load_lexicon(TEACHER_LEXICON)
        A = build_automaton(terms)
        uni = load_refs()
        l2m = latest_l2_map()
        rev = load_rev()
        log(f'词表：kept={lmeta["terms_kept"]:,}（df≤{lmeta["df_max"]}）；营收面板年 {sorted(rev)}')
        readout_rows, shadow_rows = [], []
        pc_compare, matrix_chk, vocab_stats, secs = {}, {}, {}, {}
        for y in years:
            res = read_year(y, A, uni, l2m, rev, vocab_stats)
            readout_rows += res['readout_rows']
            shadow_rows += res['shadow_rows']
            matrix_chk[str(y)] = res['matrix_check']
            secs[str(y)] = res['sec']
            if res['pc_summary'] is not None:
                pc_compare[str(y)] = res['pc_summary']
            log(f'[{y}] readout 完成（{res["sec"]}s；矩阵核对 {res["matrix_check"]["n_pairs"]:,} 对，'
                f'max|Δ|={res["matrix_check"]["max_abs_diff"]:.2e}）')
        del A, uni
        write_csv(READOUT_CSV, [[r[c] if r[c] is not None else '' for c in READOUT_COLS]
                                for r in readout_rows], READOUT_COLS)
        write_csv(SHADOW_CSV, shadow_rows, SHADOW_CSV_COLS)
        meta = {
            'shadow_version': SHADOW_VERSION,
            'script': 'scripts/shadow_origrule.py',
            'rule_definitions': {
                'origrule_minthr_global': '原文第四步字面规则：thr_i = focal 所在团的团内最小相似度'
                                          '（团内任意两成员对的最小相似度，团内全员共享；单点团 → 1.0 无 peer）；'
                                          'peer 集合 = {j: sim(i,j) > thr_i}（严格大于；候选域 = 全部节点；'
                                          '排除 focal 自身）；含团外公司。',
                'same_cluster_members': '项目口径（对照）：peer 集合 = 同团成员（peer 数 = 团规模 − 1）。',
            },
            'origrule_asymmetry': '原文规则按**行阈值**判定（mask_ij = C_ij > thr_i），thr 逐团不同 → '
                                  'mask **不必对称**（(i,j) 入选而 (j,i) 不入选是合法结果）；lift 用有向份额'
                                  '（与 eval_K / eval_at 同式）。',
            'pc_definition': 'pc = −HHI(y−1 年销售收入份额)；HHI 分母含 focal 自身（与 headline '
                             'network_business_competition_hhi 同构）；对手集合 = 规则给出的 peer 集合；缺营收不计入分母但报告覆盖；'
                             '覆盖 <2 或 Σv ≤ 0 或 focal 缺营收 → pc 留空（不填 0）。',
            'revenue_year_rule': 'PC_{i,y} 需 y−1 年营收（面板 FY2016–2024）→ 文本年 2017–2022；'
                                 '2014–2016 → pc_status=revenue_year_not_in_panel（留空）。',
            'available_at_rule': '(y+1)-04-30（= headline build_report available_at，逐行对账）',
            'cluster_source': 'headline（data/processed/network_ma_teacher/clusters/，只读）',
            'lexicon': lmeta, 'corpus': 'teacher_mda_panel（headline 词表 df≤0.25）',
            'p3asof_note': 'network_ma_teacher_p3asof 的聚类只进 3.3_origrule_readout.csv（交叉读数）；'
                           '不进本 CSV',
            'columns': SHADOW_CSV_COLS,
            'column_notes': {
                'peer_share_of_market': 'n_competitors / (n_firms − 1)',
                'same_sw2_peer_share_latest': '本 focal 的对手集合中与 focal 同申万二级的比例（latest 静态标签）',
                'sw2_lift_latest': '年层面 lift（同申万二级有向边份额 / 边际基线；eval_K 同式），同规则行内为常数',
                'sw2_lift_asof': '同上，行业标签用 as-of（effective_from ≤ (y+1)-04-30，来自 3.1 映射）；'
                                 '3.1 映射缺件时为空',
                'pc_headline': 'headline {y}_pc.csv 的 pc（同团成员口径；两规则行都带，便于并排）',
                'pc_matches_headline': '仅 same_cluster_members 行：本文件重算 pc 与 headline pc.csv 一致（容差 1e-6）',
            },
            'years': years, 'pc_years': sorted(int(k) for k in pc_compare),
        }
        write_json(SHADOW_META, meta)
        counts['readout'] = {'rows': readout_rows, 'matrix_check': matrix_chk,
                             'vocab_filter_stats_by_year': vocab_stats,
                             'n_shadow_rows': len(shadow_rows),
                             'shadow_csv': str(SHADOW_CSV.relative_to(ROOT)),
                             'secs': secs}
        counts['pc_compare'] = pc_compare
        log(f'readout 完成：readout {len(readout_rows)} 行 / shadow {len(shadow_rows)} 行')

    if 'vocab' in steps:
        counts['vocab_sensitivity'] = vocab_sensitivity(not args.no_vocab_probe)
        log('vocab：df 阈值敏感度表 + 探针完成')

    if 'accept' in steps:
        require('readout' in counts, 'accept 需要 readout 计数（先跑 --steps readout）')
        rr = counts['readout']['rows']
        hl = [r for r in rr if r['cluster_source'] == 'headline']
        orig_rows = [r for r in hl if r['peer_rule'] == RULE_ORIG]
        cl_rows = [r for r in hl if r['peer_rule'] == RULE_CLUSTER]
        verdict = degeneracy_verdict(orig_rows)
        freeze = json.loads(FREEZE_REPORT.read_text(encoding='utf-8'))
        fr = {}
        for r in cl_rows:
            y = str(r['text_year'])
            fz = freeze['years'].get(y, {})
            fr[y] = {'peer_median_recomputed': r['peer_median'],
                     'peer_median_freeze': fz.get('peer_median'),
                     'sw2_lift_recomputed': r['sw2_lift_latest'],
                     'sw2_lift_freeze': fz.get('sw2_lift'),
                     'K_recomputed': r['n_clusters'], 'K_freeze': fz.get('K'),
                     'isolated_share_recomputed': r['isolated_share'],
                     'isolated_share_freeze': fz.get('isolated_share')}
        require(all(abs(v['peer_median_recomputed'] - v['peer_median_freeze']) <= 0
                    for v in fr.values()), '同团规则 peer 中位与 freeze 报告不符；停机')
        require(all(v['K_recomputed'] == v['K_freeze'] for v in fr.values()),
                '同团规则团数与 freeze 报告不符；停机')
        require(all(abs(v['sw2_lift_recomputed'] - v['sw2_lift_freeze']) <= 0.005
                    for v in fr.values()), '同团规则 sw2_lift 与 freeze 报告不符（容差 0.005）；停机')
        pcc = {}
        for k, v in (counts.get('pc_compare') or {}).items():
            if not isinstance(v, dict) or 'orig' not in v:
                continue
            pcc[k] = {'revenue_year': v['revenue_year'], 'n_firms': v['n_firms'],
                      'orig_pc_median': v['orig']['pc_median'],
                      'headline_pc_median': v['headline_file']['pc_median'],
                      'abs_magnitude_ratio_median': (v.get('corr') or {}).get(
                          'abs_magnitude_ratio_median'),
                      'corr_orig_vs_headline': (v.get('corr') or {}).get('orig_vs_headline'),
                      'match': v['match']}
        counts['origrule_readout'] = {'per_year': {int(r['text_year']): r for r in orig_rows},
                                      'per_year_p3asof': {int(r['text_year']): r for r in rr
                                                          if r['cluster_source'] == 'p3asof'
                                                          and r['peer_rule'] == RULE_ORIG}}
        counts['project_rule_readout_vs_freeze'] = fr
        counts['degeneracy'] = verdict
        counts['pc_compare_summary'] = pcc
        counts['conclusion'] = {
            'status': '不可比/未复现（Phase 0.2 结论维持）',
            'verdict_wording': '原文规则在当前词表下不可构造/退化',
            'degenerate_under_current_lexicon': bool(verdict['degenerate']),
            'statements': [
                '原文第四步字面规则（相似度 > 团内最小相似度，候选域不受限）在当前 teacher 词表下**退化** —— '
                '即按本步实现（严格 >、候选域 = 全节点、行阈值）该口径的对手集合**不可构造**为「数十家量级」的'
                '对手集合：peer ≈ 全市场（见 degeneracy.peer_median_by_year 与 '
                'degeneracy.peer_share_of_market_median_by_year）；',
                'PC 与 headline 同团 PC **不可比**（量级与横截面排序均不同；见 pc_compare_summary）；',
                '窄词表路线不可行：df 阈值收紧对词表/active_vocab 几乎无影响，无法到原文 ~1.66 万词量级'
                '（见 vocab_sensitivity）；',
                '若要与原文对表，须**同时**满足「原文规则 + 近似原文词表规模」——本项目两条都不满足；'
                '本步不启动窄词表重建。',
            ],
            'wording_note': '本步避免「复现失败/符号相反/近似一致/可对表」表述；判定用词 = 不可比/未复现',
        }
        write_json(COUNTS_JSON, counts)
        log(f'accept：退化裁决 = {verdict["degenerate"]}（peer 中位 '
            f'{verdict["median_of_year_peer_medians"]:.0f}，lift {verdict["median_of_year_lifts_latest"]}）')
        if not verdict['degenerate']:
            print('[HARD-GATE] 原文规则**未**退化（peer 中位 <500 且 lift >5）→ 停机，升级项目决策；'
                  '不得自行启动窄词表重建', file=sys.stderr, flush=True)
            write_json(OUTPUT_SHA, output_sha_ledger())
            raise SystemExit(3)

    if 'postflight' in steps:
        post = anchors_snapshot('post')
        write_json(ANCHOR_POST, post)
        pre = json.loads(ANCHOR_PRE.read_text(encoding='utf-8')) if ANCHOR_PRE.is_file() else None
        cmp_ = compare_anchors(pre, post) if pre else {'status': 'NO_PRE'}
        counts['anchor_post'] = post
        counts['anchor_compare'] = cmp_
        counts['output_sha_rerun'] = output_sha_ledger()
        write_json(OUTPUT_SHA, counts['output_sha_rerun'])
        if cmp_['status'] != 'MATCH':
            write_json(COUNTS_JSON, counts)
            print('[HARD-GATE] headline/输入锚发生变化；停机（禁止重算覆盖）', file=sys.stderr, flush=True)
            raise SystemExit(2)
        log(f'postflight：锚对比 {cmp_["status"]}（headline {post["headline_network_n_files"]} 文件 + '
            f'9 锚 + 3.1 产物 {len(post["p3asof_network_sha"])} 文件）')

    counts['elapsed_sec_total'] = round(time.time() - t_all, 1)
    write_json(COUNTS_JSON, counts)
    log(f'完成：总耗时 {counts["elapsed_sec_total"]}s → {COUNTS_JSON}')


if __name__ == '__main__':
    main()
