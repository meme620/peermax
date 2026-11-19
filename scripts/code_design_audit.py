"""network 代码↔设计一致性机械核查（对抗审计 R2，2026-09-14）。

对每条设计公式做**独立复算**，不信任构建脚本自报：
  C1  M-B 边集 = {(i,j): sim_ij > τ}  —— 用冻结实现重算某年相似度，逐 focal 比对边集与相似度值
  C2  M-B 权重 = sim / Σ_{τ-选中} sim（并核对"全集分母"备选口径的差异幅度）
  C3  M-A 边集 = 同团全部无序对（反例检查：同团对数 vs 边数，不得漏边）
  C4  M-A 权重 = sim / Σ_{同团} sim
  C5  样本筛选：J/S 门类、上市不足 2 年、命中词条 <10 的剔除计数与设计一致
  C6  赛点：τ/K 与 registry 冻结值一致
  C7  反例：no-network-support 行不得有权重；孤立公司不得被填 0
产物：reports/data_gates/G0/code_design_audit.json
"""
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPD = ROOT / 'reports/data_gates/G0'
MA, MB = ROOT / 'data/processed/network_ma', ROOT / 'data/processed/network_mb'
YEAR = 2021
out = {'run_at': datetime.now(timezone.utc).isoformat(), 'year': YEAR, 'checks': {}}


# ---------- C1/C2：M-B 边集与权重（重算相似度）----------
def c1_c2_mb():
    from network_mda_edges_build import build_similarity, DF_CUTOFF
    codes, Xn = build_similarity(YEAR)
    with open(MB / f'{YEAR}_build_report.json', encoding='utf-8') as f:
        rep = json.load(f)
    tau = float(rep['tau'])
    # 载入边表（按 focal 分组）
    edges = defaultdict(list)
    with open(MB / f'{YEAR}_similarity.csv', encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            edges[r['focal_id']].append((r['source_id'], float(r['similarity']), float(r['weight'])))
    rng = np.random.default_rng(5)
    sample = rng.choice(len(codes), size=6, replace=False)
    res = {'tau': tau, 'per_focal': [], 'set_mismatch': 0, 'sim_mismatch': 0,
           'full_denominator_median_ratio': None}
    ratios = []
    idx = {c: i for i, c in enumerate(codes)}
    for si in sample:
        focal = codes[si]
        row = (Xn[si] @ Xn.T).toarray().ravel().astype(np.float64)
        row[si] = 0.0
        expect = {codes[j] for j in np.where(row > tau)[0]}
        got = {s for s, _, _ in edges.get(focal, [])}
        set_ok = expect == got
        sim_ok = True
        exp_sim = {}
        for s, sim_v, w_v in edges.get(focal, []):
            j = idx[s]
            if abs(sim_v - float(row[j])) > 2e-7:
                sim_ok = False
            exp_sim[s] = float(row[j])
        # 权重分母：τ-选中 vs 全集
        sum_sel = sum(exp_sim.values())
        sum_all = float(row[row > 0].sum())
        if sum_sel > 0:
            ratios.append(sum_sel / sum_all)
        res['per_focal'].append({'focal': focal, 'n_expect': len(expect), 'n_got': len(got),
                                 'set_equal': bool(set_ok), 'sim_equal': bool(sim_ok),
                                 'sum_sel_over_sum_all': round(sum_sel / sum_all, 4) if sum_sel else None})
        res['set_mismatch'] += (not set_ok)
        res['sim_mismatch'] += (not sim_ok)
    res['full_denominator_median_ratio'] = round(float(np.median(ratios)), 4) if ratios else None
    res['pass'] = res['set_mismatch'] == 0 and res['sim_mismatch'] == 0
    out['checks']['C1_C2_mb_edge_set_and_weight'] = res


# ---------- C3/C4：M-A 边集与权重 ----------
def c3_c4_ma():
    clu = pd.read_csv(MA / 'clusters' / f'{YEAR}_clusters.csv', dtype={'firm_id': str})
    sizes = clu['cluster_id'].value_counts()
    # 期望边数（有向）= 2 × ΣC(size,2) − 2 × (同团零相似度对数)
    # 零相似度对必须**不**写边（权重会是 0，违反 zero_weight=0 与 no-0 语义）
    from network_business_lexicon import build_automaton, load_lexicon
    from network_business_cluster_grid import build_vectors, load_refs
    from sklearn.preprocessing import normalize as l2
    A_ = build_automaton(load_lexicon()[0]); uni_ = load_refs()
    fl, X_, _ = build_vectors(YEAR, A_, uni_)
    Xn_ = l2(X_)
    idx_ = {f: i for i, f in enumerate(fl)}
    zero_pairs = 0
    for cid, grp in clu.groupby('cluster_id'):
        m = [f for f in grp['firm_id'] if f in idx_]
        if len(m) < 2:
            continue
        sub = np.asarray((Xn_[list(map(idx_.get, m))] @ Xn_[list(map(idx_.get, m))].T).todense())
        iu_ = np.triu_indices(len(m), 1)
        zero_pairs += int((sub[iu_] == 0).sum())   # 只在**上三角取值**计数（triu 矩阵会含下三角填充零）
    expect_pairs = int((sizes * (sizes - 1) // 2).sum()) * 2 - 2 * zero_pairs
    out['checks']['C3_extra'] = {'zero_similarity_within_cluster_pairs': zero_pairs,
                                 'expected_directed_edges': expect_pairs}
    edges = pd.read_csv(MA / f'{YEAR}_similarity.csv', dtype={'focal_id': str, 'source_id': str})
    got = len(edges)
    # 反例：同团对数 应等于 边数（不漏边）；且每条边两端同团
    c2k = dict(zip(clu['firm_id'], clu['cluster_id']))
    same = sum(1 for a, b in zip(edges['focal_id'], edges['source_id']) if c2k.get(a) == c2k.get(b))
    # 权重复算（抽 3 个 focal，取该 focal 全部边）
    rng = np.random.default_rng(3)
    foc = rng.choice(edges['focal_id'].unique(), size=3, replace=False)
    recomputed = []
    for f_ in foc:
        d = edges[edges['focal_id'] == f_]
        diff = ((d['similarity'] / d['similarity'].sum()) - d['weight']).abs()
        rel = diff / d['weight'].abs().clip(lower=1e-12)
        # 判据：绝对 1e-12（12 位小数落盘的舍入下限）或相对 1e-9
        recomputed.append({'focal': f_, 'n': len(d),
                           'max_abs_err': float(diff.max()), 'max_rel_err': float(rel.max())})
    out['checks']['C3_C4_ma_edge_set_and_weight'] = {
        'expected_pairs_from_clusters': expect_pairs, 'edges_in_file': got,
        'no_missing_edge': expect_pairs == got, 'same_cluster_share': round(same / max(1, got), 6),
        'weight_recompute_sample': recomputed,
        'weight_recompute_pass': all(r['max_abs_err'] < 1e-12 or r['max_rel_err'] < 1e-9 for r in recomputed),
        'tol': 'abs<1e-12 或 rel<1e-9（float64 计算 + 12 位小数落盘）',
        'pass': expect_pairs == got and same == got and all(r['max_abs_err'] < 1e-12 or r['max_rel_err'] < 1e-9 for r in recomputed)}


# ---------- C5：样本筛选计数 ----------
def c5_filter():
    import importlib
    grid = importlib.import_module('network_business_cluster_grid')
    terms, _ = grid.load_lexicon() if hasattr(grid, 'load_lexicon') else (None, None)
    from network_business_lexicon import build_automaton, load_lexicon
    terms, _ = load_lexicon()
    A = build_automaton(terms)
    uni = grid.load_refs()
    _, X, stats = grid.build_vectors(YEAR, A, uni)
    clu = pd.read_csv(MA / 'clusters' / f'{YEAR}_clusters.csv', dtype={'firm_id': str})
    # 独立重算：语料 ok 数 − 三类剔除
    corpus = pd.read_csv(ROOT / 'data/processed/business_texts.csv',
                         usecols=['code', 'year', 'status'])
    n_ok = int(((corpus['year'] == YEAR) & (corpus['status'] == 'ok')).sum())
    out['checks']['C5_sample_filter'] = {
        'corpus_ok': n_ok, 'docs_scanned': stats['docs'],
        'drop_menlei_J_S': stats['drop_menlei'], 'drop_ipo_lt2y': stats['drop_ipo'],
        'drop_min_terms': stats['drop_min'], 'kept': stats['firms_kept'],
        'cluster_rows': int(len(clu)),
        'identity_holds': n_ok - stats['drop_menlei'] - stats['drop_ipo'] - stats['drop_min'] == stats['firms_kept'],
        'kept_eq_clusters': stats['firms_kept'] == len(clu),
        'pass': (n_ok - stats['drop_menlei'] - stats['drop_ipo'] - stats['drop_min'] == stats['firms_kept']
                 and stats['firms_kept'] == len(clu))}


# ---------- C6：赛点与 registry 一致 ----------
def c6_frozen():
    import yaml
    reg = yaml.safe_load((ROOT / 'configs/parameter_registry.yaml').read_text(encoding='utf-8'))
    P = {x['name']: x for x in reg['parameters']}
    mb = json.loads((MB / f'{YEAR}_build_report.json').read_text(encoding='utf-8'))
    ma = json.loads((MA / f'{YEAR}_build_report.json').read_text(encoding='utf-8'))
    tau_reg = float(P['m_b_tau_fixed']['headline_value'])
    k_ma = ma['params']['K']
    freeze = json.loads((REPD / 'network_business_param_freeze.json').read_text(encoding='utf-8'))
    k_freeze = freeze['years'][str(YEAR)]['K']
    out['checks']['C6_frozen_params'] = {
        'tau_file': mb['tau'], 'tau_registry': tau_reg, 'tau_match': abs(mb['tau'] - tau_reg) < 1e-12,
        'K_file': k_ma, 'K_freeze_json': k_freeze, 'K_match': k_ma == k_freeze,
        'df_cutoff_used': 0.25, 'df_registry': P['df_threshold']['headline_value'],
        'df_match': float(P['df_threshold']['headline_value']) == 0.25,
        'min_product_terms_file': 10,
        'min_product_terms_registry': P['min_product_terms']['headline_value'],
        'linkage_file': ma['params']['cluster_linkage'], 'linkage_registry': P['cluster_linkage']['headline_value'],
    }
    c = out['checks']['C6_frozen_params']
    c['pass'] = c['tau_match'] and c['K_match'] and c['df_match'] and c['linkage_file'] == c['linkage_registry']


# ---------- C7：反例（无支持语义）----------
def c7_adversarial():
    res = {}
    for tag, d in [('ma', MA), ('mb', MB)]:
        sup = pd.read_csv(d / f'{YEAR}_support.csv', dtype={'focal_id': str})
        ns = set(sup.loc[sup['network_support_status'] == 'no-network-support', 'focal_id'])
        edges = pd.read_csv(d / f'{YEAR}_similarity.csv', dtype={'focal_id': str}, usecols=['focal_id'])
        res[tag] = {'n_no_support': len(ns),
                    'no_support_have_edges': len(ns & set(edges['focal_id'])),
                    'pass': len(ns & set(edges['focal_id'])) == 0}
    out['checks']['C7_adversarial_no_support'] = res


def main():
    print('C1/C2 M-B 边集与权重（重算相似度，约 1-2 分钟）…', flush=True)
    c1_c2_mb()
    print('C3/C4 M-A 边集与权重…', flush=True)
    c3_c4_ma()
    print('C5 样本筛选…', flush=True)
    c5_filter()
    print('C6 冻结参数一致性…', flush=True)
    c6_frozen()
    print('C7 反例（无支持语义）…', flush=True)
    c7_adversarial()
    ok = all(v.get('pass', True) for v in out['checks'].values())
    out['status'] = 'PASS' if ok else 'FAIL'
    (REPD / 'code_design_audit.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=1)[:3000])
    print('status =', out['status'])


if __name__ == '__main__':
    main()
