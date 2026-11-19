#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3 Phase B″5：换章节对照审计（文本稳健性列 network_ma ↔ 主口径 network_ma_teacher）。

目的：把「同方法、换章节/语料」（业务描述 vs 老师 MD&A）这件事量化为可登记的对照结论，
并用机械判据回答「章节输入是否改变网络结构结论方向」。蓝本 = `corpus_bridge_reconcile.py`
（Phase B′4 的 M-B 换语料对账），本脚本改目录对与判据块。

v2（2026-09-29，判据修订，依据用户裁定）：
  - ⓐ **median peer 口径统一为「当年网络全部公司（含零对手）」**：分母 = 当年 `{y}_support.csv`
    的 focal 集合（= 入网全部公司 = build_report.n_nodes），peer 数 = 该 focal 边数，无支持 = 0。
    与 `network_business_cluster_freeze.cluster_stats` 的 `peers = sizes[inv] − 1` 逐值一致，也与老师方法说明
    原文一致：「We calculate the median number of peers **for all firms** … the median number of
    text-based peers **across the pooled sample** equals N_SW2」
    （`ref/teacher-materials/Methodology Description.pdf` §2）。
  - ⓒ 边 Jaccard **降级为描述性**（不再计入 pass/fail）。理由：两章语料层的产品词集本身只重合
    10–22%（`s3_teacher_corpus_chapter_scan.json → lexicon_comparison`，Jaccard 0.119–0.132），
    边不重合是预期；且 0.20 这条线是从阈值法（M-B）跨域借来的（换语料 0.600 / 换方法 0.047–0.129），
    在聚类法域内连「换方法」都只有 4.7–12.9%。实测边 Jaccard 0.1175–0.1818 属预期。
  - ⓓ 聚类一致性 **重标注为实现自检（bug detector），非稳健性证据**（不参与 pass/fail）：两侧
    各自 1.000 只说明边表与聚类标签自洽，不构成"换章节不改结论"的证据。
  - **ⓔ 下游方向一致性（描述性对照，**非闸门**）**：主口径与文本稳健性两套网络各自算出的
    PeerMAXraw / PeerReturn（及组合层五分位）是否**反向**；数值由 `network_business_downstream_consistency.py`
    产出，经 `--downstream-report` 读入。**定性（2026-09-29 二修）**：该口径在观察到本审计 v1 结果
    （2026-09-28 停机）之后设计，2026-09-28 预先写下的替代处置口径中不含"下游方向"这一项 →
    **属探索性/事后（post hoc）判据，不构成 outcome-blind 预登记**；其作用**仅限于**描述性对照，
    **不得**作为 M1/M2 输入合格性或"换章节稳健性通过"的证据。结论措辞为"**两网下游未反向（描述性）**"。
  - ⓑ 同申万二级 lift（两侧均 >1 且同向）保持不变；**lift 基线口径本条唯一化**为
    「同样本边际基线」（节点全域边际分布），旧报告的 SW2 lift 用不同基线，**口径不同，勿直接比较**。
  - **本审计的闸门判据只有 ⓐ（统一口径，逐年 + 池化）与 ⓑ**；ⓒ/ⓓ/ⓔ 均为描述性或实现自检。

输入：data/processed/network_ma/（自建业务描述，pm_ma_ewcos_v1_y，文本年 2017-2025）
      data/processed/network_ma_teacher/（老师 MD&A，pm_ma_ewcos_teacher_v1_y，文本年 2014-2022）
      data/processed/sw2_industry_latest.csv（同申万二级占比基线）
      两侧 clusters/{y}_clusters.csv（聚类一致性自检）
重叠文本年（缺省）：2017–2022（→ 网络年 2018–2023）。

指标（逐年 + 池化，两侧**同定义**）：
  - 节点/边规模、隔离率（no-network-support focal 占比）
  - peer 规模（统一口径）：median_peer = 全部入网公司（含零对手）；另报旧口径（仅含有边 focal）
    与均值/p90（同统一口径）
  - 同申万二级边占比 + lift（同样本边际基线：Σ_k n_k(n_k−1) / N(N−1)，节点 = 当年入网公司）
  - 边 Jaccard（限共同节点集）＋密度诊断（**描述性**）
  - 聚类一致性：每条边 focal/source 同团率（**实现自检**）

判据（v2，重叠网络年 2018–2023；**闸门只有 ⓐ+ⓑ**）：
  ⓐ peer 规模：|median_peer_teacher − median_peer_self| / median_peer_self ≤ 0.30（逐年 + 池化，统一口径）
  ⓑ 同申万二级 lift：两侧均 > 1 且方向一致（同样本边际基线）
  ⓒ 边 Jaccard ≥ 0.20 —— **已降级为描述性，不参与 pass/fail**（阈值保留仅作参考线）
  ⓓ 聚类一致性 = 1.000 —— **实现自检（bug detector），不参与 pass/fail**
  ⓔ 下游方向一致性 —— **探索性/事后（post hoc）描述性对照，不参与 pass/fail**（详见文首定性）

产物：reports/data_gates/G0/s3_ma_chapter_cross_audit.json + .md
用法：
  python scripts/network_business_chapter_cross_audit.py [--years 2017 ... 2022]
  python scripts/network_business_chapter_cross_audit.py \
      --downstream-report reports/data_gates/G0/s3_ma_downstream_consistency.json
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SELF_DIR = ROOT / 'data/processed/network_ma'
TEACH_DIR = ROOT / 'data/processed/network_ma_teacher'
SW2 = ROOT / 'data/processed/sw2_industry_latest.csv'
REPD = ROOT / 'reports/data_gates/G0'
YEARS = [2017, 2018, 2019, 2020, 2021, 2022]          # 文本年（重叠段）
PEER_REL_TOL = 0.30                                    # ⓐ（统一口径）
JACCARD_MIN = 0.20                                     # ⓒ 参考线（描述性，不判）
CLUSTER_CONSISTENCY_EXPECT = 1.0                       # ⓓ 实现自检期望值（不判）
MIN_LIFT = 1.0                                         # ⓑ
PEER_MEDIAN_DEF = ('median_peer = 当年网络全部公司（含零对手）：分母 = {y}_support.csv 的 focal 集合'
                   '（= 入网公司 = build_report.n_nodes），peer 数 = 该 focal 边数，无支持 = 0；'
                   '与 network_business_cluster_freeze.cluster_stats 的 peers = sizes[inv]-1 逐值可比；'
                   '依据 = ref/teacher-materials/Methodology Description.pdf §2'
                   '（median number of peers for all firms / across the pooled sample）')
LIFT_DEF = ('lift = 同申万二级边占比 / 同样本边际基线：base = Σ_k n_k(n_k−1) / (N(N−1))，'
            'n_k = 当年网络节点（support 表全部 focal，含零对手）中申万二级 = k 的计数（仅计有映射者），'
            'N = Σ_k n_k；分子 = 边表中两端均可映射的边里同二级占比。'
            '**口径不同，勿直接比较**：旧报告 s3_b3_ma_param_freeze{,teacher}.json 的 sw2_lift 用'
            '「节点全域两两配对、缺映射记为同类」的基线，network_business_chapter_cross_audit v1 用「边端点边际'
            '分布」基线 → 同一网络同年可出现 4.43 / 5.96 / 5.96+ 等不同数值。')
csv.field_size_limit(10**9)


def read_edges(d: Path, y: int) -> pd.DataFrame:
    fp = d / f'{y}_similarity.csv'
    return pd.read_csv(fp, dtype={'focal_id': str, 'source_id': str},
                       usecols=['focal_id', 'source_id', 'similarity', 'weight'])


def read_support(d: Path, y: int) -> pd.DataFrame:
    return pd.read_csv(d / f'{y}_support.csv', dtype={'focal_id': str})


def load_sw2() -> dict[str, str]:
    sw = pd.read_csv(SW2, dtype={'code6': str, 'l2_code': str})
    sw['code6'] = sw['code6'].str.zfill(6)
    return dict(zip(sw['code6'], sw['l2_code'].astype(str)))


def l2_of(firm: str, m: dict[str, str]):
    if not isinstance(firm, str):
        return None
    return m.get(firm[2:].zfill(6))


def peer_stats(sup: pd.DataFrame, df: pd.DataFrame) -> dict:
    """**两侧同定义（统一口径）**的 peer 规模统计 = 全部入网公司（含零对手）。

    分母 = support 表 focal 集合；无支持/零对手 = 0。旧口径（仅含有边 focal）另报，作描述性。
    """
    c = df.groupby('focal_id').size()
    peers = sup['focal_id'].map(lambda f: int(c.get(f, 0))).astype('int64')
    return {'focal_nodes_with_edges': int(df['focal_id'].nunique()),
            'focal_nodes_total': int(len(sup)),
            'edges': int(len(df)),
            'peer_median': float(peers.median()),
            'peer_median_edges_only': float(c.median()) if len(c) else None,
            'peer_mean': round(float(peers.mean()), 2),
            'peer_p90': float(peers.quantile(0.9)),
            'isolated_share': round(float((peers == 0).mean()), 4)}


def sw2_lift(df: pd.DataFrame, sw2: dict, nodes: set[str]) -> dict:
    """同申万二级边占比 + lift（**同样本边际基线**，两侧同式）。

    分子：边表（focal→source）中两端均可映射的边里同二级占比。
    分母：当年网络节点集 nodes（support 表全部 focal，含零对手）在申万二级上的边际分布
          给出的随机配对期望占比 Σ_k n_k(n_k−1) / (N(N−1))。
    另报旧口径（边端点边际分布）作诊断——**口径不同，勿与 lift 直接比较**。
    """
    fa = df['focal_id'].map(lambda f: l2_of(f, sw2))
    fb = df['source_id'].map(lambda f: l2_of(f, sw2))
    ok = fa.notna() & fb.notna()
    codes = [c for c in (l2_of(f, sw2) for f in nodes) if c is not None]
    vc = pd.Series(codes).value_counts() if codes else pd.Series(dtype=int)
    N = int(vc.sum())
    base_node = float((vc * (vc - 1)).sum()) / max(1, N * (N - 1))
    out = {'same_sw2_share': None, 'base_rate_canonical': round(base_node, 5),
           'n_nodes_mapped': N, 'base_rate_edge_endpoints_legacy': None,
           'lift': None, 'lift_edge_endpoint_baseline_legacy': None, 'n_edges_mapped': int(ok.sum())}
    if not ok.any():
        return out
    same = float((fa[ok] == fb[ok]).mean())
    ep = pd.concat([fa[ok], fb[ok]])
    vce = ep.value_counts()
    Ne = int(vce.sum())
    base_edge = float((vce * (vce - 1)).sum()) / max(1, Ne * (Ne - 1))
    out.update({'same_sw2_share': round(same, 4),
                'base_rate_edge_endpoints_legacy': round(base_edge, 5),
                'lift': round(same / base_node, 3) if base_node > 0 else None,
                'lift_edge_endpoint_baseline_legacy': round(same / base_edge, 3) if base_edge > 0 else None})
    return out


def cluster_consistency(net_dir: Path, y: int, edges: pd.DataFrame) -> dict:
    """每条边 focal/source 同团率（**实现自检 / bug detector**，不作为稳健性证据）。"""
    clu = pd.read_csv(net_dir / 'clusters' / f'{y}_clusters.csv', dtype={'firm_id': str})
    c2k = dict(zip(clu['firm_id'], clu['cluster_id']))
    ek = edges[['focal_id', 'source_id']].drop_duplicates()
    same = sum(1 for a, b in zip(ek['focal_id'], ek['source_id']) if c2k.get(a) == c2k.get(b))
    return {'n_edge_pairs': int(len(ek)), 'same_cluster': int(same),
            'consistency': round(same / max(1, len(ek)), 6)}


def isolation(edges: pd.DataFrame, sup: pd.DataFrame) -> dict:
    """隔离率（无支持 = no-network-support focal 占比；两侧同定义）。"""
    n_ns = int((sup['network_support_status'] == 'no-network-support').sum())
    n_tot = int(len(sup))
    return {'nodes_total': n_tot, 'nodes_support': n_tot - n_ns, 'nodes_no_support': n_ns,
            'isolation_rate': round(n_ns / max(1, n_tot), 4)}


def density_diagnostics(a: pd.DataFrame, b: pd.DataFrame, common: set) -> dict:
    """**诊断（描述性，不参与判据）**：把密度差从 Jaccard 里剥离出来。

    - 包含率：|A∩B|/|B|、|A∩B|/|A|
    - 独立性 lift：观测交集 / 独立随机图期望交集（|A|·|B|/U，U = 共同节点有序对数）
    - top-k Jaccard：两侧各按权重取每 focal 前 k 条边后再算 Jaccard（密度对齐口径）
    """
    A = a.loc[a.focal_id.isin(common) & a.source_id.isin(common)]
    B = b.loc[b.focal_id.isin(common) & b.source_id.isin(common)]
    sa = set(map(tuple, A[['focal_id', 'source_id']].values))
    sb = set(map(tuple, B[['focal_id', 'source_id']].values))
    n = len(common)
    U = max(1, n * (n - 1))
    inter = len(sa & sb)
    out = {'n_common_nodes': n, 'n_pairs_universe': U,
           'edges_self': len(sa), 'edges_teacher': len(sb), 'intersection': inter,
           'containment_teacher_in_self': round(inter / max(1, len(sb)), 4),
           'containment_self_in_teacher': round(inter / max(1, len(sa)), 4),
           'density_self': round(len(sa) / U, 5), 'density_teacher': round(len(sb) / U, 5),
           'overlap_lift_vs_independence': round(inter / max(1e-9, len(sa) * len(sb) / U), 2)}
    for k in (30, 50):
        ta = set(map(tuple, A.sort_values('weight', ascending=False)
                     .groupby('focal_id').head(k)[['focal_id', 'source_id']].values))
        tb = set(map(tuple, B.sort_values('weight', ascending=False)
                     .groupby('focal_id').head(k)[['focal_id', 'source_id']].values))
        out[f'top{k}_jaccard_density_matched'] = round(len(ta & tb) / max(1, len(ta | tb)), 4)
    return out


def corpus_richness(net_dir: Path, y: int) -> dict:
    """诊断：两侧该年词向量密度（mean_terms / active_vocab / kept firms）。"""
    p = net_dir / f'{y}_build_report.json'
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding='utf-8'))
    fs = d.get('filter_stats', {})
    return {'mean_terms': fs.get('mean_terms'), 'active_vocab': fs.get('active_vocab'),
            'firms_kept': fs.get('firms_kept'), 'K_clusters': d.get('k_clusters')}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, nargs='*', default=YEARS, help='重叠文本年（缺省 2017-2022）')
    ap.add_argument('--self-dir', default=str(SELF_DIR))
    ap.add_argument('--teacher-dir', default=str(TEACH_DIR))
    ap.add_argument('--downstream-report', default=str(REPD / 's3_ma_downstream_consistency.json'),
                    help='ⓔ 下游方向一致性报告（缺省 reports/data_gates/G0/s3_ma_downstream_consistency.json；'
                         '不存在则 ⓔ 记 pending）')
    ap.add_argument('--out-json', default=str(REPD / 's3_ma_chapter_cross_audit.json'))
    ap.add_argument('--out-md', default=str(REPD / 's3_ma_chapter_cross_audit.md'))
    args = ap.parse_args()

    self_dir = Path(args.self_dir).resolve()
    teach_dir = Path(args.teacher_dir).resolve()
    sw2 = load_sw2()

    per_year, pool_self, pool_teach = [], [], []
    for y in args.years:
        if not (self_dir / f'{y}_similarity.csv').exists() or not (teach_dir / f'{y}_similarity.csv').exists():
            per_year.append({'text_year': y, 'status': 'missing'})
            continue
        a = read_edges(self_dir, y)          # 自建（文本稳健性列）
        b = read_edges(teach_dir, y)         # 老师 MD&A（主口径）
        sa, sb = read_support(self_dir, y), read_support(teach_dir, y)
        na = set(a.focal_id) | set(a.source_id)
        nb = set(b.focal_id) | set(b.source_id)
        common = na & nb

        pa, pb = peer_stats(sa, a), peer_stats(sb, b)
        # 池化用的 (year, focal, peer) 记录：统一口径（全部入网公司，含零对手）
        ca = a.groupby('focal_id').size(); cb = b.groupby('focal_id').size()
        pool_self.extend([(y, f, int(ca.get(f, 0))) for f in sa['focal_id']])
        pool_teach.extend([(y, f, int(cb.get(f, 0))) for f in sb['focal_id']])

        la, lb = sw2_lift(a, sw2, set(sa['focal_id'])), sw2_lift(b, sw2, set(sb['focal_id']))
        # 共同节点集上的 Jaccard（与 B′4 基线可比；**描述性**）
        A = set(map(tuple, a.loc[a.focal_id.isin(common) & a.source_id.isin(common),
                                 ['focal_id', 'source_id']].values))
        B = set(map(tuple, b.loc[b.focal_id.isin(common) & b.source_id.isin(common),
                                 ['focal_id', 'source_id']].values))
        jac = len(A & B) / max(1, len(A | B))

        ka = cluster_consistency(self_dir, y, a)
        kb = cluster_consistency(teach_dir, y, b)
        ia = isolation(a, sa)
        ib = isolation(b, sb)

        rel = abs(pb['peer_median'] - pa['peer_median']) / pa['peer_median'] if pa['peer_median'] else None
        crit_a = bool(rel is not None and rel <= PEER_REL_TOL)
        crit_b = bool(la['lift'] and lb['lift'] and la['lift'] > MIN_LIFT and lb['lift'] > MIN_LIFT
                      and np.sign(la['lift'] - MIN_LIFT) == np.sign(lb['lift'] - MIN_LIFT))
        # 描述性（不判）
        crit_c = bool(jac >= JACCARD_MIN)
        crit_d = bool(ka['consistency'] == CLUSTER_CONSISTENCY_EXPECT
                      and kb['consistency'] == CLUSTER_CONSISTENCY_EXPECT)
        per_year.append({
            'text_year': y, 'network_year': y + 1, 'status': 'ok',
            'nodes_self': ia['nodes_total'], 'nodes_teacher': ib['nodes_total'],
            'nodes_common': len(common), 'nodes_self_only': len(na - nb), 'nodes_teacher_only': len(nb - na),
            'edges_self': pa['edges'], 'edges_teacher': pb['edges'],
            'isolation_rate_self': ia['isolation_rate'], 'isolation_rate_teacher': ib['isolation_rate'],
            'peer_median_self': pa['peer_median'], 'peer_median_teacher': pb['peer_median'],
            'peer_median_edges_only_self': pa['peer_median_edges_only'],
            'peer_median_edges_only_teacher': pb['peer_median_edges_only'],
            'peer_mean_self': pa['peer_mean'], 'peer_mean_teacher': pb['peer_mean'],
            'peer_p90_self': pa['peer_p90'], 'peer_p90_teacher': pb['peer_p90'],
            'isolated_share_self': pa['isolated_share'], 'isolated_share_teacher': pb['isolated_share'],
            'peer_median_rel_diff': round(rel, 4) if rel is not None else None,
            'same_sw2_self': la['same_sw2_share'], 'same_sw2_teacher': lb['same_sw2_share'],
            'sw2_lift_self': la['lift'], 'sw2_lift_teacher': lb['lift'],
            'sw2_base_canonical_self': la['base_rate_canonical'],
            'sw2_base_canonical_teacher': lb['base_rate_canonical'],
            'sw2_lift_legacy_baseline_self': la['lift_edge_endpoint_baseline_legacy'],
            'sw2_lift_legacy_baseline_teacher': lb['lift_edge_endpoint_baseline_legacy'],
            'edge_jaccard_common_nodes': round(jac, 4),
            'cluster_consistency_self': ka['consistency'], 'cluster_consistency_teacher': kb['consistency'],
            'corpus_richness_self': corpus_richness(self_dir, y),
            'corpus_richness_teacher': corpus_richness(teach_dir, y),
            'density_diagnostics': density_diagnostics(a, b, common),
            'criteria': {'a_peer_unified': crit_a, 'b_lift_direction': crit_b},
            'descriptive': {'c_jaccard_ge_0.20': crit_c,
                            'd_cluster_consistency_1.000_implementation_selfcheck': crit_d},
        })

    ok = [r for r in per_year if r.get('status') == 'ok']
    ps = pd.DataFrame(pool_self, columns=['year', 'focal', 'n'])
    pt = pd.DataFrame(pool_teach, columns=['year', 'focal', 'n'])
    pooled = {
        'n_focal_year_self': int(len(ps)), 'n_focal_year_teacher': int(len(pt)),
        'peer_median_self': float(ps['n'].median()) if len(ps) else None,
        'peer_median_teacher': float(pt['n'].median()) if len(pt) else None,
    }
    pooled['peer_median_rel_diff'] = (
        round(abs(pooled['peer_median_teacher'] - pooled['peer_median_self'])
              / pooled['peer_median_self'], 4)
        if pooled['peer_median_self'] else None)
    pooled['criteria'] = {'a_peer_unified': bool(pooled['peer_median_rel_diff'] is not None
                                                 and pooled['peer_median_rel_diff'] <= PEER_REL_TOL)}
    # ⓔ 下游方向一致性（**描述性对照，非闸门**；读对照报告；缺失 → pending）
    dp = Path(args.downstream_report)
    e_block = {'status': 'pending', 'gating': False,
               'report': str(dp.relative_to(ROOT)).replace('\\', '/') if dp.exists() else None,
               'note': '探索性/事后（post hoc）描述性对照：不构成 outcome-blind 预登记，非闸门；'
                       '不得作为 M1/M2 输入合格性或"换章节稳健性通过"的证据'}
    e_ok = None
    if dp.exists():
        dd = json.loads(dp.read_text(encoding='utf-8'))
        h = dd.get('cross_audit_handoff', {})
        e_ok = h.get('e_downstream_direction')
        e_block = {'status': ('未反向（描述性）' if e_ok else '反向（描述性）') if e_ok is not None else 'pending',
                   'gating': False,
                   'report': str(dp.relative_to(ROOT)).replace('\\', '/'),
                   'run_at': dd.get('run_at'),
                   'qualitative_status': dd.get('qualitative_status'),
                   'conclusion': dd.get('conclusion'),
                   'descriptive_checks': h.get('descriptive_checks'),
                   'overlap_months': h.get('overlap_months'),
                   'effective_paired_months': h.get('effective_paired_months'),
                   'peer_maxraw': h.get('peer_maxraw'), 'peer_return': h.get('peer_return'),
                   'quintile_agreement': h.get('quintile_agreement'),
                   'strength': h.get('strength'),
                   'portfolio': h.get('portfolio'),
                   'note': h.get('note')}
    a_all = all(r['criteria']['a_peer_unified'] for r in ok) if ok else False
    b_all = all(r['criteria']['b_lift_direction'] for r in ok) if ok else False
    # 闸门只有 ⓐ+ⓑ（ⓒ 描述性、ⓓ 实现自检、ⓔ post hoc 描述性）
    gating = [a_all, b_all, pooled['criteria']['a_peer_unified']]
    status = 'PASS' if (ok and len(ok) == len(args.years) and all(gating)) else 'FAIL'
    rep = {
        'step': 'S3 Phase B″5 换章节对照审计（自建业务描述 ↔ 老师 MD&A，同方法 M-A）',
        'audit_version': 'v2 (2026-09-29, 判据修订)',
        'superseded_artifact': 'artifacts/repro/phase_bpp_snapshot/'
                               's3_ma_chapter_cross_audit_superseded_20260929.{json,md}（v1，status=FAIL，保留原判据）',
        'run_at': datetime.now(timezone.utc).isoformat(),
        'years_text': args.years, 'years_network': [y + 1 for y in args.years],
        'self_dir': str(self_dir.relative_to(ROOT)).replace('\\', '/'),
        'teacher_dir': str(teach_dir.relative_to(ROOT)).replace('\\', '/'),
        'peer_metric_definition': PEER_MEDIAN_DEF,
        'peer_metric_definition_legacy_descriptive':
            'median_peer_edges_only = edges.groupby(focal_id).size().median()（仅含有边 focal；描述性字段，不判）',
        'lift_definition': LIFT_DEF,
        'thresholds': {'a_peer_rel_tol': PEER_REL_TOL, 'b_min_lift': MIN_LIFT,
                       'c_jaccard_reference_line_descriptive': JACCARD_MIN,
                       'd_cluster_consistency_expected_selfcheck': CLUSTER_CONSISTENCY_EXPECT},
        'criteria_set': {
            'gating': ['a_peer_unified', 'b_lift_direction'],
            'descriptive_not_gating': ['c_jaccard_ge_0.20',
                                       'd_cluster_consistency_1.000_implementation_selfcheck',
                                       'e_downstream_direction_post_hoc_descriptive'],
            'status_note': ('闸门 = ⓐ（统一口径，逐年+池化）+ ⓑ（同样本边际基线 lift 同向）；'
                            'ⓒ 描述性、ⓓ 实现自检、ⓔ 探索性/事后（post hoc）描述性，均不计入 pass/fail。'
                            'ⓔ 报告状态：' + e_block['status']),
            'e_post_hoc_note': 'ⓔ 在观察到本审计 v1 结果之后设计（2026-09-29），不构成 outcome-blind '
                               '预登记；作用限于"两网下游是否反向"的描述性对照，**不得**作为 M1/M2 '
                               '输入合格性或"换章节稳健性通过"的证据。',
        },
        'baselines': {'same_corpus_other_chapter_reference_jaccard_mb': 0.600,
                      'change_method_jaccard_range': [0.047, 0.129],
                      'two_chapter_lexicon_jaccard_range': [0.119, 0.1322],
                      'sources': ['reports/data_gates/G0/s3_corpus_bridge_report.json',
                                  'reports/data_gates/G0/s3_network_cross_audit.json',
                                  'reports/data_gates/G0/s3_teacher_corpus_chapter_scan.json'
                                  ' → lexicon_comparison']},
        'note': '两侧网络均在同一冻结方法参数下构建（average 链 + 逐年 K 校准 + 同团对手集合 + '
                '行归一化权重），只差文本章节/语料；不改变任何冻结值。ⓒ/ⓓ 已按 2026-09-29 裁定'
                '降级/重标注，不再作为闸门。',
        'per_year': per_year, 'pooled': pooled, 'e_downstream_direction': e_block,
        'criteria_summary': {
            'gating': {'a_peer_unified_per_year_all': a_all,
                       'a_peer_unified_pooled': pooled['criteria']['a_peer_unified'],
                       'b_lift_direction_all': b_all},
            'descriptive': {
                'e_downstream_direction_post_hoc': e_ok,
                'e_status': e_block['status'], 'e_gating': False,
                'c_jaccard_all': all(r['descriptive']['c_jaccard_ge_0.20'] for r in ok) if ok else False,
                'c_jaccard_min_observed': min((r['edge_jaccard_common_nodes'] for r in ok), default=None),
                'c_jaccard_max_observed': max((r['edge_jaccard_common_nodes'] for r in ok), default=None),
                'd_cluster_consistency_all': all(
                    r['descriptive']['d_cluster_consistency_1.000_implementation_selfcheck'] for r in ok)
                if ok else False,
                'reading': 'ⓒ 实测 0.1175–0.1818 属两章语料层产品词集只重合 10–22%'
                           '（lexicon Jaccard 0.119–0.132）的预期结果，降级为描述性；'
                           'ⓓ 两侧 1.000 只说明边表与聚类标签自洽（实现自检），不构成稳健性证据。'},
        },
        'diagnostics_summary': {
            'failing_years_a_peer_unified': [r['text_year'] for r in ok
                                             if not r['criteria']['a_peer_unified']],
            'jaccard_range': [min((r['edge_jaccard_common_nodes'] for r in ok), default=None),
                              max((r['edge_jaccard_common_nodes'] for r in ok), default=None)],
            'same_sw2_share_range_self': [min((r['same_sw2_self'] for r in ok), default=None),
                                          max((r['same_sw2_self'] for r in ok), default=None)],
            'same_sw2_share_range_teacher': [min((r['same_sw2_teacher'] for r in ok), default=None),
                                             max((r['same_sw2_teacher'] for r in ok), default=None)],
            'same_sw2_ratio_teacher_over_self_range': [
                min((round(r['same_sw2_teacher'] / max(1e-9, r['same_sw2_self']), 2) for r in ok), default=None),
                max((round(r['same_sw2_teacher'] / max(1e-9, r['same_sw2_self']), 2) for r in ok), default=None)],
            'industry_clustering_note': '主口径（老师 MD&A）同申万二级边占比 0.194–0.349，'
                                        '约为文本稳健性列（业务描述）0.085–0.122 的 1.7–4.1 倍 → '
                                        '行业吸收检验（CF 与行业动量同入回归，看行业系数是否塌缩）'
                                        '已列为 M1（C2）计划项（参照 S3A 9H 模板）。',
            'top50_jaccard_density_matched_range': [
                min((r['density_diagnostics']['top50_jaccard_density_matched'] for r in ok), default=None),
                max((r['density_diagnostics']['top50_jaccard_density_matched'] for r in ok), default=None)],
            'overlap_lift_vs_independence_range': [
                min((r['density_diagnostics']['overlap_lift_vs_independence'] for r in ok), default=None),
                max((r['density_diagnostics']['overlap_lift_vs_independence'] for r in ok), default=None)],
            'containment_teacher_in_self_range': [
                min((r['density_diagnostics']['containment_teacher_in_self'] for r in ok), default=None),
                max((r['density_diagnostics']['containment_teacher_in_self'] for r in ok), default=None)],
            'density_ratio_self_over_teacher_range': [
                min((round(r['density_diagnostics']['density_self'] /
                           max(1e-9, r['density_diagnostics']['density_teacher']), 2) for r in ok), default=None),
                max((round(r['density_diagnostics']['density_self'] /
                           max(1e-9, r['density_diagnostics']['density_teacher']), 2) for r in ok), default=None)],
            'corpus_richness_mean_terms_self_vs_teacher': {
                str(r['text_year']): [r['corpus_richness_self'].get('mean_terms'),
                                      r['corpus_richness_teacher'].get('mean_terms')] for r in ok},
            'reading': '两列边密度比约 1.2–2.9 倍（设计差 + 语料丰度差，MD&A 长文本 mean_terms 约 2 倍）；'
                       '按权重对齐的 top-50 Jaccard 仍 < 0.20，但两网非随机同向（独立性 lift 6–10×），'
                       '即边不重合主要是"同义不同词"的语料层差异，不是随机噪声。',
        },
        'cross_reference_evidence': {
            'source': 'reports/data_gates/G0/s3_teacher_corpus_chapter_scan.json → lexicon_comparison'
                      '（同公司同年份逐对：MD&A 章 产品词命中集合 vs 业务描述章 产品词命中集合）',
            'firm_year_term_set_jaccard_mda_vs_business': {'2017': 0.119, '2018': 0.121,
                                                           '2019': 0.1242, '2020': 0.1322},
            'business_term_coverage_by_mda_median': {'2017': 0.4882, '2018': 0.4831,
                                                     '2019': 0.4775, '2020': 0.4615},
            'reading': '同一公司同一年份的**底层文本特征**（产品词命中集合）在两章之间的 Jaccard 仅 '
                       '0.119–0.132；本章节对照的边 Jaccard 观测值 0.1175–0.1818 并不低于该底层一致度'
                       '（网络构造未额外损失一致度）→ 边重合率在本设计下不是有效闸门（已降级为描述性）。',
        },
        'status': status,
        'interpretation_rule': 'ⓐ（统一口径，逐年+池化）与 ⓑ（同样本边际基线下 lift 同向）全过 → '
                               '换章节的**结构层闸门**通过；ⓒ/ⓓ/ⓔ 均非闸门（ⓔ 为探索性/事后描述性对照，'
                               '结论措辞"两网下游未反向（描述性）"，不得作为 M1/M2 输入合格性证据）。',
    }
    outp = Path(args.out_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')

    md = ['# S3 换章节对照审计（Phase B″5，v2 判据修订 + 2026-09-29 二修）', '',
          f'生成时间：{rep["run_at"]}　重叠文本年：{args.years}（网络年 {[y + 1 for y in args.years]}）', '',
          '**闸门判据（只有两项）**：ⓐ peer 相对差 ≤ 0.30（**统一口径：当年网络全部公司含零对手**，逐年+池化）；'
          'ⓑ 同样本边际基线下两侧同申万二级 lift > 1 且同向。', '',
          '**非闸门**：ⓒ 边 Jaccard（描述性，参考线 0.20）；ⓓ 聚类一致性（**实现自检 / bug detector**）；'
          'ⓔ 下游方向一致性（**探索性/事后（post hoc）描述性对照**，不构成 outcome-blind 预登记，'
          '不得作为 M1/M2 输入合格性或"换章节稳健性通过"的证据）。', '',
          f'**ⓔ 状态：{e_block["status"]}**'
          + (f'（报告：`{e_block["report"]}`；有效配对月份 {e_block.get("effective_paired_months")}）'
             if e_block.get('report') else '（下游面板与对照报告尚未产出）'), '',
          '| 文本年 | 网络年 | 节点(自建/老师/共) | 隔离率(自建/老师) | peer中位统一口径(自建/老师) | ⓐ相对差 | '
          'lift 同基线(自建/老师) | 边Jaccard(描述) | ⓓ自检(自建/老师) | ⓐⓑ |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for r in per_year:
        if r.get('status') != 'ok':
            md.append(f'| {r["text_year"]} | — | — | — | — | — | — | — | — | 缺失 |')
            continue
        c = r['criteria']
        flag = 'PASS' if all(c.values()) else 'FAIL'
        md.append(f'| {r["text_year"]} | {r["network_year"]} | {r["nodes_self"]}/{r["nodes_teacher"]}/'
                  f'{r["nodes_common"]} | {r["isolation_rate_self"]:.3f}/{r["isolation_rate_teacher"]:.3f} | '
                  f'{r["peer_median_self"]:.0f}/{r["peer_median_teacher"]:.0f} | '
                  f'{r["peer_median_rel_diff"]:.3f} | {r["sw2_lift_self"]}/{r["sw2_lift_teacher"]} | '
                  f'{r["edge_jaccard_common_nodes"]} | {r["cluster_consistency_self"]:.3f}/'
                  f'{r["cluster_consistency_teacher"]:.3f} | {flag} |')
    md += ['', f'池化：peer 中位（统一口径；自建/老师）= {pooled["peer_median_self"]:.0f}/'
               f'{pooled["peer_median_teacher"]:.0f}（n_firm-year = {pooled["n_focal_year_self"]}/'
               f'{pooled["n_focal_year_teacher"]}），相对差 = {pooled["peer_median_rel_diff"]:.3f} '
               f'（ⓐ {"PASS" if pooled["criteria"]["a_peer_unified"] else "FAIL"}）'
               '。年份聚合口径说明：本项目实现为**逐年中位 + 逐年 K 校准**（沿用 freeze 层设计，非 pooled）；'
               '若按各列全期池化（主口径 2014–2022 / 文本稳健性 2017–2025）则 ⓐ = 0.196，同样 ≤0.30'
               '（后者年份集不同，仅作敏感性参考）。',
           '', f'**总闸门状态：{status}**（{rep["criteria_set"]["status_note"]}）', '',
           '## ⓔ 下游方向一致性（探索性/事后（post hoc）描述性对照，**非闸门**）', '']
    if e_ok is None:
        md += ['待产出：`reports/data_gates/G0/s3_ma_downstream_consistency.json`（B″6 面板后运行'
               '`network_business_downstream_consistency.py`）。']
    else:
        md += [f'`{e_block["report"]}`（run_at {e_block.get("run_at")}；'
               f'定性状态 {e_block.get("qualitative_status")}）', '',
               f'- **结论：{e_block.get("conclusion")}**',
               f'- 名义重叠 {e_block.get("overlap_months")}（72 个月），**有效配对月份 = '
               f'{e_block.get("effective_paired_months")}**（2018-01~04 因文本稳健性列最早版本 '
               f'`available_at=2018-04-30`、严格 `<` 规则而无配对）',
               f'- PeerMAXraw：{e_block.get("peer_maxraw")}',
               f'- PeerReturn：{e_block.get("peer_return")}',
               f'- 五分位一致性：{e_block.get("quintile_agreement")}',
               f'- 组合层：{e_block.get("portfolio")}',
               f'- 强度：{e_block.get("strength")}',
               '', '**定性声明**：ⓔ 在观察到本审计 v1 结果之后设计，**不构成 outcome-blind 预登记**；'
               '作用限于"两网下游是否反向"的**描述性**对照——**不得**作为 M1/M2 输入合格性或'
               '"换章节稳健性通过"的证据。']
    md += ['', '## 描述性与诊断（不参与判据）', '',
           '判读参考基线：M-B 侧「换语料」边 Jaccard 中位 = 0.600（`s3_corpus_bridge_report.json`）；'
           '「换方法」= 0.047–0.129（`s3_network_cross_audit.json`）；两章语料层产品词集 Jaccard = '
           '0.119–0.132（`s3_teacher_corpus_chapter_scan.json`）→ **边 Jaccard 实测 0.1175–0.1818 '
           '属语料层只重合 10–22% 的预期结果，故降级为描述性**。', '',
           'lift 口径：本条为**同样本边际基线**（当年网络节点在申万二级上的边际分布计算期望同行业占比）。'
           '旧报告（`s3_b3_ma_param_freeze{,teacher}.json` 的 `sw2_lift`）用不同基线，**口径不同，'
           '勿直接比较**——同一网络同年可差出 4.43 / 5.96 这类量级。', '',
           '| 文本年 | 边密度(自建/老师) | 密度比 | 包含率(老师⊂自建) | 独立性lift | top50 Jaccard | '
           'mean_terms(自建/老师) |',
           '|---|---|---|---|---|---|---|']
    for r in per_year:
        if r.get('status') != 'ok':
            continue
        dd, cs, ct = r['density_diagnostics'], r['corpus_richness_self'], r['corpus_richness_teacher']
        md.append(f'| {r["text_year"]} | {dd["density_self"]}/{dd["density_teacher"]} | '
                  f'{dd["density_self"] / max(1e-9, dd["density_teacher"]):.2f} | '
                  f'{dd["containment_teacher_in_self"]:.3f} | {dd["overlap_lift_vs_independence"]:.1f}x | '
                  f'{dd["top50_jaccard_density_matched"]:.4f} | {cs.get("mean_terms")}/{ct.get("mean_terms")} |')
    _ts = rep['diagnostics_summary']['same_sw2_share_range_teacher']
    _ss = rep['diagnostics_summary']['same_sw2_share_range_self']
    _rr = rep['diagnostics_summary']['same_sw2_ratio_teacher_over_self_range']
    md += ['', f'行业扎堆：主口径同申万二级边占比 {_ts[0]:.3f}–{_ts[1]:.3f}'
               f'（逐年 {_ts}）vs 文本稳健性 {_ss[0]:.3f}–{_ss[1]:.3f}（逐年 {_ss}）'
               f'（比值 {_rr}）→ 行业吸收检验列为 M1（C2）计划项。']
    Path(args.out_md).write_text('\n'.join(md), encoding='utf-8')
    print(json.dumps(rep['criteria_summary'], ensure_ascii=False, indent=1))
    print('status =', status)
    print('->', outp)


if __name__ == '__main__':
    main()
