#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3 第一梯队 ④：2021 断点分段稳健性（结构层）。

**只读复算**：不重估 K、不重建网络、不碰面板/回归、不改任何 headline_value / 冻结参数；只新建
`reports/data_gates/G0/network_business_segment_robustness.{json,md}`。复用 `network_business_chapter_cross_audit.py`
的纯函数（read_edges / read_support / load_sw2 / peer_stats / sw2_lift / isolation / corpus_richness）。

口径真源：
  - `reports/audits/s3_remaining_workplan_20261011.md` 第一梯队 4
  - 主仓库 methodology_execution_plan §8 B″11-3
  - `reports/audits/s3_network_open_items_plan_20261006.md` §3-①A（含 2026-10-06 二修：两版本切分、
    版本 2 后段 2 年停机例外、下游只能 2018–2020 vs 2021–2023、措辞「两网下游未反向（描述性）」）

切分（网络年；段内用各自年度 K，不重估、不重建）：
  - 版本 1（网络年对齐，主口径分段版本）：2015–2020（文本年 2014–2019） vs 2021–2023（文本年 2020–2022）
  - 版本 2（文本年对齐，更贴断点机制）：文本年 2014–2020（网络年 2015–2021） vs 文本年 2021–2022
    （网络年 2022–2023）；**后段仅 2 个网络年 → 按 §3-①A 二修显式例外，只作描述性**
  - 网络年错位提示：网络年 y 使用文本年 y−1（每年 1–4 月沿用 y−2），语料断点发生在文本年 2021 ⇒
    版本 1 的后段混入 1 个断点前年份（文本年 2020）。

逐段指标：peer 规模分布（中位/p90/均值，统一口径含零对手）、`same_sw2`、`lift`（同样本边际基线
base = Σ_k n_k(n_k−1)/(N(N−1))）、段间边 Jaccard（共同节点集内外各一）、`mean_terms`、隔离率、K。

下游段对照：**只能 2018–2020 vs 2021–2023**，且只作描述性（措辞「两网下游未反向（描述性）」）；
数值直接引用既有 `s3_ma_downstream_consistency.json → segments`，本步不重算。

停机：某段 <3 个网络年（版本 2 后段按登记例外除外），或段内隔离率跳变 >2 倍 → 停，先定位语料。
用法：python scripts/network_business_segment_robustness.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import network_business_chapter_cross_audit as CA  # noqa: E402

REPD = ROOT / 'reports/data_gates/G0'
TEACH_DIR = ROOT / 'data/processed/network_ma_teacher'
DOWNSTREAM = REPD / 's3_ma_downstream_consistency.json'
TEXT_YEARS = list(range(2014, 2023))         # 主口径 9 年：文本年 2014–2022 → 网络年 2015–2023
SEG_V1 = {'2015_2020': [2015, 2016, 2017, 2018, 2019, 2020], '2021_2023': [2021, 2022, 2023]}
SEG_V2 = {'text_2014_2020': [2015, 2016, 2017, 2018, 2019, 2020, 2021], 'text_2021_2022': [2022, 2023]}
MIN_YEARS_PER_SEG = 3
ISO_JUMP_TOL = 2.0                           # 段内隔离率跳变上限（max/min）


def pooled_peer(pairs: list[tuple[int, pd.DataFrame, pd.DataFrame]]) -> dict:
    """段内池化的 peer 分布（统一口径 = 全部入网公司含零对手；**按 (年, 公司) 记录池化**）。"""
    recs = []
    for _y, sup, edges in pairs:
        c = edges.groupby('focal_id').size()
        recs.extend(int(c.get(f, 0)) for f in sup['focal_id'])
    peers = np.asarray(recs, dtype='int64')
    return {'n_firm_year': int(len(peers)),
            'peer_median': float(np.median(peers)),
            'peer_mean': round(float(peers.mean()), 3),
            'peer_p90': float(np.quantile(peers, 0.9)),
            'peer_p99': float(np.quantile(peers, 0.99)),
            'isolated_share': round(float((peers == 0).mean()), 4)}


def share_counts(edges: pd.DataFrame, sw2: dict) -> tuple[float, int, int]:
    """全精度同申万二级边占比（**未舍入**）+ 分子/分母计数（L-1：池化必须用未舍入 share）。

    与 `CA.sw2_lift` 同一口径（两端均可映射的边里同二级占比），仅不做 4dp 舍入；
    返回 (share_full, same_edges_mapped, n_edges_mapped)。逐 focal 映射用同一 `CA.l2_of`。
    """
    fa = edges['focal_id'].map(lambda f: CA.l2_of(f, sw2)).to_numpy()
    fb = edges['source_id'].map(lambda f: CA.l2_of(f, sw2)).to_numpy()
    ok = pd.notna(fa) & pd.notna(fb)
    n = int(ok.sum())
    if n == 0:
        return float('nan'), 0, 0
    same = int((fa[ok] == fb[ok]).sum())
    return same / n, same, n


def segment_block(years: list[int], data: dict, common_core: set) -> dict:
    yrs = [y for y in years if y in data]
    if not yrs:
        return {'status': 'missing'}
    peer = pooled_peer([(y, data[y]['sup'], data[y]['edges']) for y in yrs])
    # 池化 same_sw2 / base / lift（同样本边际基线；pooled 口径 = Σ 同类对/Σ 全对 + Σ 同业边/Σ 可映射边）
    # L-1：same_sw2 池化用**未舍入**全精度计数（逐年 4dp 存储值只作逐年读数，不参与池化）
    K = sum(data[y]['K'] for y in yrs)
    U = sum(data[y]['U'] for y in yrs)
    Em = sum(data[y]['n_edges_mapped'] for y in yrs)
    same_edges = sum(data[y]['same_edges'] for y in yrs)
    base = K / U if U else float('nan')
    share = same_edges / Em if Em else float('nan')
    iso = {'nodes_total': int(sum(data[y]['iso']['nodes_total'] for y in yrs)),
           'nodes_no_support': int(sum(data[y]['iso']['nodes_no_support'] for y in yrs)),
           'isolation_rate_pooled': round(sum(data[y]['iso']['nodes_no_support'] for y in yrs)
                                          / max(1, sum(data[y]['iso']['nodes_total'] for y in yrs)), 4),
           'isolation_rate_by_year': {str(y): data[y]['iso']['isolation_rate'] for y in yrs},
           'isolation_rate_max_over_min': (round(max(data[y]['iso']['isolation_rate'] for y in yrs)
                                                 / max(1e-12, min(data[y]['iso']['isolation_rate'] for y in yrs)), 4)),
           'isolation_rate_max_consecutive_jump': (round(max(
               [data[b]['iso']['isolation_rate'] / max(1e-12, data[a]['iso']['isolation_rate'])
                for a, b in zip(yrs, yrs[1:])] or [1.0]), 4)),
           'isolation_note': ('隔离率 = no-network-support focal 占比 = **singleton 团（无同团对手）**占比'
                             '（`network_business_edges_build.py`：无同团成员 → no-network-support，权重不填 0）')}
    mt = {str(y): data[y]['mean_terms'] for y in yrs}
    mt_vals = [v for v in mt.values() if v is not None]
    Ks = {str(y): data[y]['K_clusters'] for y in yrs}
    # 段内边集（并集；有向边 focal→source）
    seg_edges = set().union(*[data[y]['edge_set'] for y in yrs])
    seg_edges_core = {(a, b) for (a, b) in seg_edges if a in common_core and b in common_core}
    return {'status': 'ok', 'network_years': yrs, 'text_years': [y - 1 for y in yrs],
            'n_network_years': len(yrs), **peer,
            'same_sw2_pooled': share, 'base_canonical_pooled': base, 'lift_pooled': share / base,
            'same_sw2_pooled_precision_note': ('池化用**全精度（未舍入）**计数 Σ 同业边 / Σ 可映射边（L-1）；'
                                              '逐年 `same_sw2_by_year` 仍为 4dp 存储读数，两者不得互相对账'),
            'same_edges_mapped_sum': same_edges, 'n_edges_mapped_sum': Em,
            'same_sw2_by_year': {str(y): data[y]['share'] for y in yrs},
            'lift_by_year': {str(y): data[y]['lift'] for y in yrs},
            'n_edges_sum': int(sum(data[y]['n_edges'] for y in yrs)),
            'isolation': iso, 'mean_terms': mt,
            'mean_terms_segment_mean': round(float(np.mean(mt_vals)), 3) if mt_vals else None,
            'mean_terms_segment_median': float(np.median(mt_vals)) if mt_vals else None,
            'K_clusters': Ks,
            'edge_set_sizes': {'union_all_nodes': len(seg_edges),
                               'restricted_common_core': len(seg_edges_core)},
            '_seg_edges': seg_edges, '_seg_edges_core': seg_edges_core}


def jaccard(a: set, b: set) -> dict:
    inter = len(a & b)
    return {'n_a': len(a), 'n_b': len(b), 'intersection': inter,
            'jaccard': round(inter / max(1, len(a | b)), 4),
            'containment_a_in_b': round(inter / max(1, len(a)), 4),
            'containment_b_in_a': round(inter / max(1, len(b)), 4)}


def downstream_block() -> dict:
    """下游段对照（只作描述性；直接引用既有产物，不重算）。"""
    if not DOWNSTREAM.exists():
        return {'status': 'pending', 'gating': False,
                'note': '下游对照报告缺失（reports/data_gates/G0/s3_ma_downstream_consistency.json）'}
    d = json.loads(DOWNSTREAM.read_text(encoding='utf-8'))
    h = d.get('cross_audit_handoff', {})
    segs = d.get('segments', {})
    return {'status': 'ok', 'gating': False, 'descriptive_only': True,
            'year_axis': 'calendar_month（日历月；非文本年/网络年）',
            'allowed_comparison': '只能**日历月** 2018-01..2020-12 vs 2021-01..2023-12（= 下游月历段；'
                                  '文本稳健性列最早 available_at=2018-04-30，严格 < 规则下 2018-05 起才可用 → '
                                  '2015–2020 段缺可比月份）',
            'report': str(DOWNSTREAM.relative_to(ROOT)).replace('\\', '/'),
            'run_at': d.get('run_at'),
            'conclusion': h.get('conclusion') or d.get('conclusion'),
            'wording_required': '两网下游未反向（描述性）',
            'qualitative_status': d.get('qualitative_status'),
            'effective_paired_months': h.get('effective_paired_months'),
            'segments': segs,
            'reading': ('逐段看（**日历月段**，非文本年/网络年）：2018-01–2020-12 两网多空同为正'
                        '（+0.004149 / +0.004202），2021-01–2023-12 两网多空同为负'
                        '（−0.009676 / −0.007864）→ **两网（主口径 ↔ 文本稳健性）在段内方向一致、未反向**；'
                        '跨段符号翻转属描述性事实，**不得**据此写成「分段稳健性通过」，亦不得作 M1/M2 '
                        '输入合格性证据（ⓔ 类探索性/事后对照）'),
            'note': '本块为 post hoc 描述性对照（非 outcome-blind 预登记、非闸门）；数值原样引用既有产物，'
                    '本步未重算任何下游量'}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--net-dir', default=str(TEACH_DIR))
    ap.add_argument('--text-years', type=int, nargs='*', default=TEXT_YEARS)
    ap.add_argument('--out-json', default=str(REPD / 's3_ma_segment_robustness.json'))
    ap.add_argument('--out-md', default=str(REPD / 's3_ma_segment_robustness.md'))
    args = ap.parse_args()

    net_dir = Path(args.net_dir)
    sw2 = CA.load_sw2()
    data, per_year, node_sets = {}, [], []
    for ty in args.text_years:
        y = ty + 1
        if not (net_dir / f'{ty}_similarity.csv').exists():
            per_year.append({'text_year': ty, 'network_year': y, 'status': 'missing'})
            continue
        sup = CA.read_support(net_dir, ty)
        edges = CA.read_edges(net_dir, ty)
        blk = CA.sw2_lift(edges, sw2, set(sup['focal_id']))
        sh_full, same_edges, n_map = share_counts(edges, sw2)   # L-1：未舍入 share + 计数
        assert n_map == blk['n_edges_mapped'] and round(sh_full, 4) == blk['same_sw2_share'], (
            f'全精度 share 与 CA.sw2_lift 不一致（文本年 {ty}）：n_mapped {n_map} vs {blk["n_edges_mapped"]}、'
            f'round4 {round(sh_full, 4)} vs {blk["same_sw2_share"]}')
        peers = CA.peer_stats(sup, edges)
        iso = CA.isolation(edges, sup)
        cr = CA.corpus_richness(net_dir, ty)
        brp = net_dir / f'{ty}_build_report.json'
        br = json.loads(brp.read_text(encoding='utf-8')) if brp.exists() else {}
        fs = br.get('filter_stats', {})
        corpus = {'docs': fs.get('docs'), 'drop_menlei': fs.get('drop_menlei'),
                  'drop_ipo': fs.get('drop_ipo'), 'drop_min': fs.get('drop_min'),
                  'firms_kept': fs.get('firms_kept'), 'active_vocab': fs.get('active_vocab'),
                  'mean_terms': fs.get('mean_terms'), 'n_nodes': br.get('n_nodes'),
                  'n_support': br.get('n_support'), 'n_no_support': br.get('n_no_support'),
                  'k_clusters': br.get('k_clusters'),
                  'network_version': br.get('network_version'), 'available_at': br.get('available_at'),
                  'params_K_rule': (br.get('params') or {}).get('K_rule'),
                  'params_min_product_terms': (br.get('params') or {}).get('min_product_terms')}
        codes = pd.Series([CA.l2_of(f, sw2) for f in sup['focal_id']]).dropna()
        vc = codes.value_counts()
        data[y] = {'sup': sup, 'edges': edges, 'share': blk['same_sw2_share'] / 1.0,
                   'share_full': sh_full, 'same_edges': same_edges,
                   'base_5dp': blk['base_rate_canonical'], 'lift': blk['lift'],
                   'n_edges': int(len(edges)), 'n_edges_mapped': blk['n_edges_mapped'],
                   'K': int((vc * (vc - 1) / 2).sum()),
                   'U': float(len(codes) * (len(codes) - 1) / 2),
                   'iso': iso, 'mean_terms': cr.get('mean_terms'), 'K_clusters': cr.get('K_clusters'),
                   'corpus': corpus,
                   'edge_set': set(map(tuple, edges[['focal_id', 'source_id']].values))}
        node_sets.append(set(sup['focal_id']))
        per_year.append({'text_year': ty, 'network_year': y, 'status': 'ok',
                         'n_nodes': len(sup), 'n_edges': int(len(edges)),
                         'metadata_lag_note': '网络年 y 使用文本年 y−1（1–4 月沿用 y−2）',
                         **{k: peers[k] for k in ('peer_median', 'peer_mean', 'peer_p90',
                                                 'focal_nodes_total', 'edges', 'isolated_share')},
                         'isolation_rate': iso['isolation_rate'],
                         'same_sw2': blk['same_sw2_share'], 'lift': blk['lift'],
                         'base_canonical_5dp': blk['base_rate_canonical'],
                         'mean_terms': cr.get('mean_terms'), 'K_clusters': cr.get('K_clusters'),
                         'mean_terms_note': '由 {ty}_build_report.json → filter_stats.mean_terms（语料丰度）'})
        print(f'[T1-④] 网络年 {y}（文本年 {ty}）: peer中位={peers["peer_median"]:.0f} '
              f'same_sw2={blk["same_sw2_share"]:.4f} lift={blk["lift"]:.3f} '
              f'mean_terms={cr.get("mean_terms")} 隔离率={iso["isolation_rate"]:.4f} '
              f'K={cr.get("K_clusters")}', flush=True)

    common_core = set.intersection(*node_sets) if node_sets else set()
    segs = {}
    for name, yrs in (('v1_network_year_aligned', SEG_V1), ('v2_text_year_aligned', SEG_V2)):
        segs[name] = {}
        for sname, ylist in yrs.items():
            segs[name][sname] = segment_block(ylist, data, common_core)
    v1a, v1b = segs['v1_network_year_aligned']['2015_2020'], segs['v1_network_year_aligned']['2021_2023']
    jac_all = jaccard(v1a['_seg_edges'], v1b['_seg_edges'])
    jac_core = jaccard(v1a['_seg_edges_core'], v1b['_seg_edges_core'])
    # 描述性：段内相邻年 Jaccard（各段内），用于说明段内稳定性
    def within(ylist):
        out = {}
        for a, b in zip(ylist, ylist[1:]):
            if a in data and b in data:
                ca = data[a]['edge_set'] & data[b]['edge_set']
                out[f'{a}->{b}'] = round(len(ca) / max(1, len(data[a]['edge_set'] | data[b]['edge_set'])), 4)
        return out
    within_seg = {k: within(v) for k, v in (('2015_2020', SEG_V1['2015_2020']),
                                           ('2021_2023', SEG_V1['2021_2023']))}

    # 停机检查
    stops, checks = [], {}
    for name, yy in segs.items():
        for sname, b in yy.items():
            if b.get('status') != 'ok':
                continue
            n_min_ok = b['n_network_years'] >= MIN_YEARS_PER_SEG
            exception = (name == 'v2_text_year_aligned' and sname == 'text_2021_2022')
            jump = max(b['isolation']['isolation_rate_max_over_min'],
                       b['isolation']['isolation_rate_max_consecutive_jump'])
            b['isolation']['isolation_rate_jump_ratio_used'] = jump
            checks[f'{name}.{sname}.min_3_years'] = bool(n_min_ok or exception)
            checks[f'{name}.{sname}.isolation_jump_le_2x'] = bool(jump <= ISO_JUMP_TOL)
            if not n_min_ok and not exception:
                stops.append(f'{name}.{sname} 仅 {b["n_network_years"]} 个网络年（<3）→ 停，先定位语料')
            if not checks[f'{name}.{sname}.isolation_jump_le_2x']:
                stops.append(f'{name}.{sname} 段内隔离率跳变 {jump:.2f}× >2× → 停，先定位语料')
    # 语料定位（只读：build_report 的 filter_stats / n_* / K；用于「先定位语料」停产要求）
    corpus_rows = []
    for ty in args.text_years:
        y = ty + 1
        if y not in data:
            continue
        c = data[y]['corpus']
        corpus_rows.append({'text_year': ty, 'network_year': y, 'docs': c['docs'], 'firms_kept': c['firms_kept'],
                            'drop_menlei': c['drop_menlei'], 'drop_ipo': c['drop_ipo'],
                            'drop_min': c['drop_min'], 'active_vocab': c['active_vocab'],
                            'mean_terms': c['mean_terms'], 'K_clusters': c['k_clusters'],
                            'n_support': c['n_support'], 'n_no_support': c['n_no_support'],
                            'isolation_rate': data[y]['iso']['isolation_rate']})
    # 自动定位：隔离率**最大单年跳变**（相邻文本年比值最大者）
    jump_pairs = []
    for a, b in zip(corpus_rows, corpus_rows[1:]):
        r = b['isolation_rate'] / max(1e-12, a['isolation_rate'])
        jump_pairs.append({'text_years': [a['text_year'], b['text_year']],
                           'network_years': [a['network_year'], b['network_year']],
                           'isolation_rate': [a['isolation_rate'], b['isolation_rate']],
                           'ratio': round(r, 4),
                           'firms_kept': [a['firms_kept'], b['firms_kept']],
                           'firms_kept_rel_change': round(b['firms_kept'] / a['firms_kept'] - 1, 4),
                           'K_clusters': [a['K_clusters'], b['K_clusters']],
                           'K_rel_change': round(b['K_clusters'] / a['K_clusters'] - 1, 4),
                           'active_vocab': [a['active_vocab'], b['active_vocab']],
                           'mean_terms': [a['mean_terms'], b['mean_terms']],
                           'drop_ipo': [a['drop_ipo'], b['drop_ipo']],
                           'drop_min': [a['drop_min'], b['drop_min']],
                           'n_no_support': [a['n_no_support'], b['n_no_support']]})
    worst = max(jump_pairs, key=lambda j: j['ratio']) if jump_pairs else None
    # L-2：隔离率跳变必须**同时并报水平值**（比值受低基数放大；只报倍数不合格）
    lvl = [{'text_year': r['text_year'], 'network_year': r['network_year'],
            'isolation_rate_level': r['n_no_support'] / max(1, r['firms_kept']),
            'isolation_rate_level_4dp': round(r['n_no_support'] / max(1, r['firms_kept']), 4),
            'n_no_support': r['n_no_support'], 'K_clusters': r['K_clusters'],
            'singleton_share_of_K': round(r['n_no_support'] / max(1, r['K_clusters']), 4)}
           for r in corpus_rows]
    rates9 = [x['isolation_rate_level'] for x in lvl]
    mean_arith9 = float(np.mean(rates9)) if rates9 else float('nan')
    median9 = float(np.median(rates9)) if rates9 else float('nan')
    pooled9 = (sum(x['n_no_support'] for x in lvl) / max(1, sum(r['firms_kept'] for r in corpus_rows))
               if lvl else float('nan'))
    lo9 = min(lvl, key=lambda x: x['isolation_rate_level']) if lvl else None
    hi9 = max(lvl, key=lambda x: x['isolation_rate_level']) if lvl else None
    rank9 = sorted(lvl, key=lambda x: x['isolation_rate_level']).index(lo9) + 1 if lo9 else None
    lvl_by_ny = {x['network_year']: x for x in lvl}
    post_pooled = segs['v1_network_year_aligned'].get('2021_2023', {}).get('isolation', {}) \
        .get('isolation_rate_pooled')
    level_report: dict = {
        'purpose': ('L-2：隔离率跳变与**水平值**同时并报（比值 = max/min 会被低基数放大；不得只报 2.09×）'),
        'per_year_level': lvl,
        'nine_year_level_context': {
            'min': {'network_year': lo9['network_year'], 'text_year': lo9['text_year'],
                    'isolation_rate_level': lo9['isolation_rate_level']} if lo9 else None,
            'max': {'network_year': hi9['network_year'], 'text_year': hi9['text_year'],
                    'isolation_rate_level': hi9['isolation_rate_level']} if hi9 else None,
            'mean_arith_by_year': round(mean_arith9, 6),
            'median_by_year': round(median9, 6),
            'pooled': round(pooled9, 6),
            'min_year_rank_ascending': rank9,
        },
    }
    level_statement = '无相邻年对'
    if worst:
        jl = [lvl_by_ny.get(ny) for ny in worst['network_years']]
        lo_ny, hi_ny = worst['network_years']
        lo_rate, hi_rate = jl[0]['isolation_rate_level'], jl[1]['isolation_rate_level']
        level_report['jump_pair_levels'] = {
            'text_years': worst['text_years'], 'network_years': worst['network_years'],
            'low_year_rate': lo_rate, 'high_year_rate': hi_rate,
            'ratio': worst['ratio'],
            'abs_level_change_pp': round(100 * (hi_rate - lo_rate), 4),
            'low_year_rate_is_9y_min': bool(lo9 and lo9['network_year'] == lo_ny),
            'low_year_rate_rank_ascending': (
                [x['network_year'] for x in sorted(lvl, key=lambda x: x['isolation_rate_level'])].index(lo_ny) + 1),
            'high_year_rate_vs_9y_mean_arith_rel': round(hi_rate / mean_arith9 - 1, 4),
            'high_year_rate_vs_9y_median_rel': round(hi_rate / median9 - 1, 4),
            'high_year_rate_vs_9y_pooled_rel': round(hi_rate / pooled9 - 1, 4),
            'high_year_rate_vs_post_break_segment_pooled_rel': (
                round(hi_rate / post_pooled - 1, 4) if post_pooled else None),
            'singleton_share_of_K_pair': [jl[0]['singleton_share_of_K'], jl[1]['singleton_share_of_K']],
            'singleton_share_of_K_change_pp': round(100 * (jl[1]['singleton_share_of_K']
                                                           - jl[0]['singleton_share_of_K']), 2),
            'singleton_count_pair': [jl[0]['n_no_support'], jl[1]['n_no_support']],
            'K_clusters_pair': [jl[0]['K_clusters'], jl[1]['K_clusters']],
            'reading': ('L-2 水平值：低值年（文本年 {ty_lo} = 网络年 {ny_lo}）**{r_lo:.4f}（{s_lo}/{f_lo}）'
                        '为 9 年最低**（9 年区间 {r_min:.4f}–{r_max:.4f}）；高值年（文本年 {ty_hi} = 网络年 {ny_hi}）'
                        '{r_hi:.4f}（{s_hi}/{f_hi}）——水平差仅 {dpp:.4f}pp；相对 9 年逐年算术均值仅 {rel_mean:+.2%}、'
                        '相对 9 年池化率仅 {rel_pooled:+.2%}、相对 9 年中位数仅 {rel_med:+.2%} '
                        '⇒ {ratio:.2f}× 的比值主要由**低基数**造成。同期 **K 中 singleton 占比 '
                        '{sk_lo:.1%}（{s_lo}/{k_lo}）→ {sk_hi:.1%}（{s_hi}/{k_hi}）**（{dsk:+.2f}pp）。'
                        ).format(ty_lo=worst['text_years'][0], ny_lo=lo_ny, r_lo=lo_rate,
                                 s_lo=jl[0]['n_no_support'], f_lo=worst['firms_kept'][0],
                                 r_min=min(rates9), r_max=max(rates9),
                                 ty_hi=worst['text_years'][1], ny_hi=hi_ny, r_hi=hi_rate,
                                 s_hi=jl[1]['n_no_support'], f_hi=worst['firms_kept'][1],
                                 dpp=100 * (hi_rate - lo_rate), rel_mean=hi_rate / mean_arith9 - 1,
                                 rel_pooled=hi_rate / pooled9 - 1, rel_med=hi_rate / median9 - 1,
                                 ratio=worst['ratio'], sk_lo=jl[0]['singleton_share_of_K'],
                                 k_lo=jl[0]['K_clusters'], sk_hi=jl[1]['singleton_share_of_K'],
                                 k_hi=jl[1]['K_clusters'],
                                 dsk=100 * (jl[1]['singleton_share_of_K'] - jl[0]['singleton_share_of_K'])),
            'approximation_note': ('复核草案原文记「2016 仅 +11% vs 9 年均值」（草案**未标年份轴**）：'
                                   '若指跳变对高值年（文本年 2016 = 网络年 2017）则实测 {rel_hi_mean:+.2%}'
                                   '（逐年算术均值基准）/ {rel_hi_pooled:+.2%}（9 年池化基准）；'
                                   '若指 9 年最低年（文本年 2015 = 网络年 2016）则为 {rel_lo_mean:+.2%}'
                                   '（同上基准）。本产物不采用近似值，水平值一律按实测并报'
                                   '（逐年算术均值 {mean_arith:.6f}、池化 {pooled9:.6f}、中位数 {median9:.6f}）'
                                   '——三种基准的共同结论 = 水平变化远小于 {ratio:.2f}× 比值所暗示的幅度'
                                   ).format(rel_hi_mean=hi_rate / mean_arith9 - 1,
                                            rel_hi_pooled=hi_rate / pooled9 - 1,
                                            rel_lo_mean=lo_rate / mean_arith9 - 1,
                                            mean_arith=mean_arith9, pooled9=pooled9, median9=median9,
                                            ratio=worst['ratio']),
        }
        level_statement = ('**水平值并报**（L-2）：' +
                           level_report['jump_pair_levels']['reading'])
    base_statement = (('隔离率的最大单年跳变为文本年 {}→{}（网络年 {}→{}）：{:.4f}→{:.4f}（{:.2f}×，'
                      'singleton 数 {}→{}），同期 **K（团数）变化 {:+.1%}**、入网公司数变化 {:+.1%}、'
                      'active_vocab {:+.1%}、mean_terms {:+.1%}、drop_ipo {}→{}、drop_min {}→{}。'
                      '机制：逐年 K 按「目标中位 peer = 44」校准，公司数增长（+{:.1%}）远小于 K 增长（{:+.1%}）'
                      '⇒ 平均团规模下降、singleton 占比上升 —— 属**语料规模/聚类规模效应**，'
                      '非数据缺失、非映射错误（该年 drop_min = 0、映射缺失 0–1 家）。'
                      '按登记停机规则本项仍记 STOP（不得改判据）；是否放宽由项目拍板。'
                      ).format(worst['text_years'][0], worst['text_years'][1], worst['network_years'][0],
                               worst['network_years'][1], worst['isolation_rate'][0], worst['isolation_rate'][1],
                               worst['ratio'], worst['n_no_support'][0], worst['n_no_support'][1],
                               worst['K_rel_change'], worst['firms_kept_rel_change'],
                               (worst['active_vocab'][1] / worst['active_vocab'][0] - 1),
                               (worst['mean_terms'][1] / worst['mean_terms'][0] - 1),
                               worst['drop_ipo'][0], worst['drop_ipo'][1],
                               worst['drop_min'][0], worst['drop_min'][1],
                               worst['firms_kept_rel_change'], worst['K_rel_change']) if worst else '无相邻年对')
    corpus_loc = {
        'purpose': '停产要求「先定位语料」：隔离率 = singleton 团占比，其年际跳变是否由语料筛选/聚类规模引起',
        'per_year': corpus_rows, 'consecutive_year_jump_pairs': jump_pairs,
        'max_jump_pair': worst,
        'level_report': level_report,
        'statement': base_statement + level_statement,
        'note': ('隔离率、K、语料丰度口径不同（singleton 团 / 团数 / 词表规模），本块只做**同一 build_report '
                 '产物内的一致性定位**，不做因果推断；隔离率水平 = n_no_support / firms_kept（与 4dp 存储值同源、'
                 '此处不四舍五入）；年份一律按**网络年（文本年 y−1）**并报（L-3）')}
    for name, yy in segs.items():
        for sname, b in yy.items():
            b.pop('_seg_edges', None)
            b.pop('_seg_edges_core', None)

    deltas = {'peer_median': [v1a['peer_median'], v1b['peer_median'],
                              round(v1b['peer_median'] - v1a['peer_median'], 3)],
              'peer_mean': [v1a['peer_mean'], v1b['peer_mean'],
                            round(v1b['peer_mean'] - v1a['peer_mean'], 3)],
              'peer_p90': [v1a['peer_p90'], v1b['peer_p90'], round(v1b['peer_p90'] - v1a['peer_p90'], 3)],
              'same_sw2_pooled': [v1a['same_sw2_pooled'], v1b['same_sw2_pooled'],
                                  round(v1b['same_sw2_pooled'] - v1a['same_sw2_pooled'], 4)],
              'lift_pooled': [v1a['lift_pooled'], v1b['lift_pooled'],
                              round(v1b['lift_pooled'] - v1a['lift_pooled'], 3)],
              'mean_terms_segment_mean': [v1a['mean_terms_segment_mean'], v1b['mean_terms_segment_mean'],
                                          round(v1b['mean_terms_segment_mean']
                                                - v1a['mean_terms_segment_mean'], 1)],
              'isolation_rate_pooled': [v1a['isolation']['isolation_rate_pooled'],
                                        v1b['isolation']['isolation_rate_pooled'],
                                        round(v1b['isolation']['isolation_rate_pooled']
                                              - v1a['isolation']['isolation_rate_pooled'], 4)]}
    mech = (f'断点机制检查：断点后段（网络年 2021–2023 = 文本年 2020–2022）mean_terms 高于断点前段'
            f'（网络年 2015–2020 = 文本年 2014–2019）（'
            f'{v1a["mean_terms_segment_mean"]} → {v1b["mean_terms_segment_mean"]}）→ 与「准则改版使业务描述'
            '并入 MD&A、正文篇幅上升」一致' if v1b['mean_terms_segment_mean']
            > v1a['mean_terms_segment_mean'] else
            '断点机制检查：断点后段（网络年 2021–2023 = 文本年 2020–2022）mean_terms **未**高于前段'
            '（网络年 2015–2020 = 文本年 2014–2019），须登记说明（不得反向解读为「无断点」）')
    rep = {
        'step': 'S3 第一梯队 ④ 断点分段稳健性（文本年 2021 语料断点；结构层；不重估 K、不重建网络）',
        'run_at': datetime.now(timezone.utc).isoformat(),
        'net_dir': str(net_dir.relative_to(ROOT)).replace('\\', '/'),
        'nature': '只读复算：不重估 K、不重建网络、不碰面板/回归、不改任何 headline_value / 冻结参数；'
                  '产物为新建文件',
        'year_axis_note': ('**年份轴（L-3 消歧）**：`segments.*.network_years` 与全部逐段/逐年字典键 = **网络年**'
                           '（即「网络年（文本年 y−1）」）；`text_years` = **文本年**（网络年 y 使用文本年 y−1 的'
                           '相似度网络，每年 1–4 月沿用 y−2）。段名 `v1_network_year_aligned` = 按网络年切段、'
                           '`v2_text_year_aligned` = 按文本年切段（其 `text_*` 前缀即文本年）；'
                           '`corpus_localization` 与 `per_year` 两列并报。下游段对照（`downstream_segment_descriptive`）'
                           '的 2018–2020 / 2021–2023 = **日历月段**（月历，非文本年/网络年，见该块 year_axis）。'
                           '全文不再出现无轴名的裸年份。'),
        'splits': {'v1_network_year_aligned': {k: {'text_years': [y - 1 for y in v], 'network_years': v}
                                               for k, v in SEG_V1.items()},
                   'v2_text_year_aligned': {'text_2014_2020': {'text_years': list(range(2014, 2021)),
                                                               'network_years': SEG_V2['text_2014_2020']},
                                            'text_2021_2022': {'text_years': [2021, 2022],
                                                               'network_years': SEG_V2['text_2021_2022']}}},
        'metadata_lag_note': '网络年 y 使用文本年 y−1（每年 1–4 月沿用 y−2）⇒ 语料断点（文本年 2021）落在'
                             '网络年 2022；版本 1 的「网络年 2021–2023」段混入 1 个断点前年份（文本年 2020）',
        'v2_exception_note': '版本 2 的后段（文本年 2021–2022 = 网络年 2022–2023）仅 2 个网络年 → 按 '
                             'open_items §3-①A 二修**显式例外**：不触发「<3 年即停」，但只能作**描述性**报告'
                             '（标注「样本量不足、仅描述性」，不得作为稳健性通过的依据）',
        'pooling_precision_note': ('L-1：段内 `same_sw2_pooled` / `lift_pooled` 用**全精度（未舍入）**份额计数'
                                   '重算（Σ 同业边 / Σ 可映射边，见 `segments.*.same_edges_mapped_sum` 与 '
                                   '`n_edges_mapped_sum`）；逐年 `same_sw2_by_year` 仍为 4dp 存储读数，'
                                   '两者不得互相对账'),
        'metric_definitions': {
            'peer_median_peer_mean_peer_p90': '统一口径 = 当年网络全部入网公司（含零对手）；段内 = 该段'
                                              '全部 firm-year 记录池化（`network_business_chapter_cross_audit.peer_stats`）',
            'same_sw2': '边层面：边表两端均可映射的边中同申万二级占比；段内池化 = Σ 同业边 / Σ 可映射边'
                        '（**池化用全精度未舍入计数**，L-1；逐年读数仍为 4dp 存储值）',
            'lift': '同样本边际基线：base = Σ_k n_k(n_k−1)/(N(N−1))（段内池化 = Σ 同类对/Σ 全对）',
            'mean_terms': '语料丰度（build_report.filter_stats.mean_terms），断点机制的直接证据',
            'jaccard': '段级边集（段内各年边并集；有向 focal→source）Jaccard；`restricted_common_core` = '
                       '限定两端均为「9 年共同节点」（每列节点集交集）',
            'year_axis': '年份轴（L-3）：`network_years` / 逐年字典键 = **网络年（文本年 y−1）**；`text_years` = '
                         '**文本年**；下游段月份 = **日历月**'},
        'per_year': per_year, 'common_core_nodes': len(common_core),
        'segments': segs,
        'segment_deltas_v1': deltas,
        'segment_jaccard_v1': {'all_nodes': jac_all, 'restricted_common_core': jac_core,
                               'within_segment_consecutive_year_jaccard': within_seg,
                               'note': '段间 Jaccard 为**描述性**（两段年份集不同、节点集不同）；'
                                       '段内相邻年 Jaccard 作参照'},
        'break_mechanism_check': mech,
        'corpus_localization': corpus_loc,
        'downstream_segment_descriptive': downstream_block(),
        'acceptance': {'checks': checks, 'stop_conditions_triggered': stops,
                       'min_years_per_segment': MIN_YEARS_PER_SEG, 'isolation_jump_tol_x': ISO_JUMP_TOL,
                       'status_semantics': ('STOP = 触发登记停机条件（隔离率段内跳变 >2×）；结构指标仍全部'
                                            '登记在案，语料定位见 corpus_localization；**不改判据、不放宽**，'
                                            '是否放宽/后续处置由项目拍板')},
        'status': 'PASS' if not stops else 'STOP_WITH_CORPUS_LOCALIZATION',
    }
    outp, outm = Path(args.out_json), Path(args.out_md)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')

    md = ['# S3 断点分段稳健性（文本年 2021 语料断点；结构层；第一梯队 ④ / B″11-3、§3-①A）', '',
          f'生成时间：{rep["run_at"]}；网络目录：`{rep["net_dir"]}`（主口径，9 个网络年 2015–2023）', '',
          '**只读复算**：段内用各自年度 K、不重估、不重建网络；不改任何 headline_value / 冻结参数。', '',
          f'**年份轴（L-3）**：本文件除注明外，年份一律为**网络年（文本年 y−1）**（`text_years` 列与「文本年」'
          '字样处为文本年；下游段对照的月份为**日历月**）——不再出现无轴名的裸年份。', '',
          f'**池化精度（L-1）**：段内 `same_sw2(池化)` / `lift(池化)` 用**全精度（未舍入）**份额计数重算'
          f'（Σ 同业边 / Σ 可映射边）；「逐年」表的 `same_sw2` 仍为 4dp 存储读数，两者不得互相对账。', '',
          f'错位提示：网络年 y 使用**文本年 y−1**（1–4 月沿用 y−2）；语料断点（文本年 2021）落在网络年 2022'
          f'→ 版本 1 后段混入 1 个断点前年份（文本年 2020）。9 年共同节点数 = {len(common_core)}。', '',
          '## 逐年指标（主口径）', '',
          '| 文本年 | 网络年 | 节点 | 边 | peer 中位 | peer 均值 | peer p90 | 隔离率 | same_sw2（4dp 存储） | '
          'lift | mean_terms | K |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in per_year:
        if r.get('status') != 'ok':
            md.append(f'| {r["text_year"]} | {r["network_year"]} | — | — | — | — | — | — | — | — | — | 缺失 |')
            continue
        md.append(f'| {r["text_year"]} | {r["network_year"]} | {r["n_nodes"]} | {r["n_edges"]} | '
                  f'{r["peer_median"]:.0f} | {r["peer_mean"]:.2f} | {r["peer_p90"]:.0f} | '
                  f'{r["isolation_rate"]:.4f} | {r["same_sw2"]:.4f} | {r["lift"]:.3f} | '
                  f'{r["mean_terms"]} | {r["K_clusters"]} |')
    md += ['', '## 分段指标（版本 1：网络年对齐，主口径分段版本）', '',
           '| 段 | 网络年 | 年数 | peer 中位 | peer 均值 | peer p90 | 隔离率(池化) | same_sw2(池化，全精度) | '
           'lift(池化，全精度) | mean_terms 段均 | K 逐年 |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for sname, b in segs['v1_network_year_aligned'].items():
        md.append(f'| {sname} | {b["network_years"]} | {b["n_network_years"]} | {b["peer_median"]:.0f} | '
                  f'{b["peer_mean"]:.2f} | {b["peer_p90"]:.0f} | {b["isolation"]["isolation_rate_pooled"]:.4f} | '
                  f'{b["same_sw2_pooled"]:.4f} | {b["lift_pooled"]:.3f} | {b["mean_terms_segment_mean"]} | '
                  f'{list(b["K_clusters"].values())} |')
    md += ['', '## 分段指标（版本 2：文本年对齐，后段 2 年 → **仅描述性**）', '',
           '| 段 | 网络年 | 年数 | peer 中位 | peer 均值 | peer p90 | 隔离率(池化) | same_sw2(池化，全精度) | '
           'lift(池化，全精度) | mean_terms 段均 |', '|---|---|---|---|---|---|---|---|---|---|']
    for sname, b in segs['v2_text_year_aligned'].items():
        md.append(f'| {sname} | {b["network_years"]} | {b["n_network_years"]} | {b["peer_median"]:.0f} | '
                  f'{b["peer_mean"]:.2f} | {b["peer_p90"]:.0f} | {b["isolation"]["isolation_rate_pooled"]:.4f} | '
                  f'{b["same_sw2_pooled"]:.4f} | {b["lift_pooled"]:.3f} | {b["mean_terms_segment_mean"]} |')
    md += ['', '> 版本 2 后段（文本年 2021–2022 = 网络年 2022–2023）仅 2 个网络年 → 按 §3-①A 二修**显式例外**：'
               '不触发「<3 年即停」，但**只作描述性**（样本量不足），**不得**作为稳健性通过的依据。', '',
           '## 段间差异（版本 1；逐项登记；年份 = 网络年（文本年 y−1））', '',
           '| 指标 | 网络年 2015–2020 | 网络年 2021–2023 | Δ（后−前） |', '|---|---|---|---|']
    for k, v in deltas.items():
        md.append(f'| {k} | {v[0]} | {v[1]} | {v[2]} |')
    md += ['', f'**断点机制检查**：{mech}', '',
           '## 段间边 Jaccard（描述性）', '',
           f'- 全节点集（网络年 2015–2020 vs 2021–2023）：Jaccard = {jac_all["jaccard"]}'
           f'（|A| = {jac_all["n_a"]}、|B| = {jac_all["n_b"]}、交集 = {jac_all["intersection"]}）',
           f'- 限定 9 年共同节点（同上两段网络年）：Jaccard = {jac_core["jaccard"]}（|A| = {jac_core["n_a"]}、'
           f'|B| = {jac_core["n_b"]}、交集 = {jac_core["intersection"]}）',
           f'- 段内相邻**网络年** Jaccard：2015–2020 = {within_seg["2015_2020"]}；'
           f'2021–2023 = {within_seg["2021_2023"]}',
           '', '## 下游段对照（描述性；**只能日历月 2018-01..2020-12 vs 2021-01..2023-12**）', '']
    ds = rep['downstream_segment_descriptive']
    if ds.get('status') != 'ok':
        md.append(f'状态：{ds.get("status")}（{ds.get("note")}）')
    else:
        md += [f'来源：`{ds["report"]}`（run_at {ds["run_at"]}；有效配对月份 {ds["effective_paired_months"]}）',
               '', f'**结论：{ds["conclusion"]}**（必须写作「两网下游未反向（描述性）」）', '',
               '| 段 | 月份 | n_rows | PeerMAXraw ρ | PeerReturn ρ | 多空 主口径 | 多空 文本稳健性 |',
               '|---|---|---|---|---|---|---|']
        for k, v in ds['segments'].items():
            md.append(f'| {k} | {v["months"][0]}~{v["months"][1]} | {v["n_rows_merged"]} | '
                      f'{v["peer_maxraw"]["pearson"]} | {v["peer_return"]["pearson"]} | '
                      f'{v["long_short_main"]} | {v["long_short_robust"]} |')
        md += ['', f'{ds["reading"]}', '', f'限定：{ds["note"]}']
    md += ['', '## 语料定位（停产要求「先定位语料」）', '',
           '| 文本年 | 网络年 | docs | firms_kept | drop_menlei | drop_ipo | drop_min | active_vocab | '
           'mean_terms | K | n_no_support | 隔离率 | singleton/K 占比 |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    _lv = {x['network_year']: x for x in corpus_loc['level_report']['per_year_level']}
    for r in corpus_loc['per_year']:
        md.append(f'| {r["text_year"]} | {r["network_year"]} | {r["docs"]} | {r["firms_kept"]} | '
                  f'{r["drop_menlei"]} | {r["drop_ipo"]} | {r["drop_min"]} | {r["active_vocab"]} | '
                  f'{r["mean_terms"]} | {r["K_clusters"]} | {r["n_no_support"]} | {r["isolation_rate"]:.4f} | '
                  f'{_lv[r["network_year"]]["singleton_share_of_K"]:.4f} |')
    md += ['', f'**定位结论**：{corpus_loc["statement"]}', '',
           f'> {corpus_loc["note"]}', '',
           '## 停机检查', '',
           f'- 段最少网络年数 = {MIN_YEARS_PER_SEG}（版本 2 后段例外见上）；段内隔离率跳变上限 = {ISO_JUMP_TOL}×'
           '（跳变 = 段内**网络年**隔离率 max/min 与相邻年比值）',
           '- 检查结果：' + json.dumps(checks, ensure_ascii=False),
           f'- **停机条件触发：{stops if stops else "无"}**', '',
           f'**status = {rep["status"]}**（STOP 表示触发登记停机条件；指标仍全部登记，语料定位见上节；'
           '**不改判据、不放宽**，是否放宽由项目拍板）', '',
           '> 本产物只作**结构层分段对照**：不替代主口径全样本结果，不重估 K，不改冻结值；'
           '下游段只作描述性（ⓔ 类）。']
    outm.write_text('\n'.join(md), encoding='utf-8')
    print(f'[T1-④] status={rep["status"]} stops={stops}', flush=True)
    print(f'[T1-④] L-1 池化（全精度）: v1 2015–2020 same_sw2={v1a["same_sw2_pooled"]!r} '
          f'lift={v1a["lift_pooled"]!r} | 2021–2023 same_sw2={v1b["same_sw2_pooled"]!r} '
          f'lift={v1b["lift_pooled"]!r}', flush=True)
    if worst:
        jpl = corpus_loc['level_report']['jump_pair_levels']
        print(f'[T1-④] L-2 跳变对水平值: 网络年 {jpl["network_years"][0]}={jpl["low_year_rate"]:.4f}'
              f'（9 年最低 rank={jpl["low_year_rate_rank_ascending"]}）→ 网络年 {jpl["network_years"][1]}='
              f'{jpl["high_year_rate"]:.4f}（vs 9 年算术均值 {jpl["high_year_rate_vs_9y_mean_arith_rel"]:+.2%}、'
              f'vs 9 年池化 {jpl["high_year_rate_vs_9y_pooled_rel"]:+.2%}）；singleton/K '
              f'{jpl["singleton_share_of_K_pair"][0]:.1%}→{jpl["singleton_share_of_K_pair"][1]:.1%}', flush=True)
    print(f'[T1-④] v1 段: 2015–2020 peer中位={v1a["peer_median"]:.0f} same_sw2={v1a["same_sw2_pooled"]:.4f} '
          f'lift={v1a["lift_pooled"]:.3f} mt={v1a["mean_terms_segment_mean"]} | '
          f'2021–2023 peer中位={v1b["peer_median"]:.0f} same_sw2={v1b["same_sw2_pooled"]:.4f} '
          f'lift={v1b["lift_pooled"]:.3f} mt={v1b["mean_terms_segment_mean"]}', flush=True)
    print(f'[T1-④] 段间 Jaccard: all={jac_all["jaccard"]} core={jac_core["jaccard"]} -> {outp}', flush=True)


if __name__ == '__main__':
    main()
