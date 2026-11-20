"""network-B1：τ 校准（申万二级中位 peer 数目标，2026-09-14）。

Plan §8 Phase B1 的可执行实现：
  1. 目标值：申万二级同行业 peer 数（组内 N-1）在**入网公司**上的**公司层面中位数**，
     跨 9 个文本年池化 → `sw2_median_peer_target`（registry 冻结值）。
  2. 校准：TF-IDF 余弦（df<=0.25，mda_texts ok 行）在**抽样 focal**上求 τ，
     使池化中位 peer 数 = 目标（中位 peer 数是 τ 的阶梯函数 → 二分 + 区间细扫）。
  3. 诊断：逐文本年的中位/均值 peer 数、孤立公司、质量保留比；并给出
     「组层面中位规模」目标下的备选 τ（次级口径，供稳健性）。

产物：
  reports/data_gates/G0/network_tau_calibration.json
  data/processed/network_mb/tau_calibration_probe.csv（τ-中位 peer 曲线，抽稀存档）

用法：
  python scripts/network_tau_calibrate.py                 # 全期 9 年
  python scripts/network_tau_calibrate.py --years 2021    # 单年快速
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from network_mda_edges_build import build_similarity

V2 = ROOT / 'data/processed/mda_texts.csv'
LAT = ROOT / 'data/processed/sw2_industry_latest.csv'
REPD = ROOT / 'reports/data_gates/G0'
PROBE = ROOT / 'data/processed/network_mb' / 'tau_calibration_probe.csv'
FOCAL_SAMPLE = 1000          # 每年抽样 focal 数（peer 计数仍用全部 source）
SEED = 20260914
TXTS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
TXTS_TEACHER = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]   # 老师语料（B′1）
CORPORA = {
    'self': {'path': V2, 'years': TXTS, 'id_col': 'code'},
    'teacher': {'path': ROOT / 'data/processed/teacher_mda_panel.csv',
                'years': TXTS_TEACHER, 'id_col': 'firm_id'},
}


# ---------- 1) SW2 目标 ----------
def sw2_target(years, corpus='self'):
    import pandas as pd
    lat = pd.read_csv(LAT, dtype={'code6': str})
    lat['code6'] = lat['code6'].str.zfill(6)
    spec = CORPORA[corpus]
    mda = pd.read_csv(spec['path'], usecols=[spec['id_col'], 'year', 'status'])
    mda = mda[mda['status'] == 'ok'].copy()
    # 两个面板的 id 均为项目面板码（sz/sh + 六位），统一剥前缀取六位代码
    mda['code6'] = mda[spec['id_col']].astype(str).str[2:].str.zfill(6)
    mda = mda[mda['year'].isin(years)]
    m = mda.merge(lat[['code6', 'l2_code']], on='code6', how='left')
    # 未匹配到申万分类的行不进 peer 计数（其 peer 数无定义）；np.median 遇 NaN 会整体变 NaN，必须显式剔除并登记
    n_unmatched = int(m['l2_code'].isna().sum())
    m = m.dropna(subset=['l2_code'])
    per_year, pooled, pooled_group = [], [], []
    for y, g in m.groupby('year'):
        sizes = g.groupby('l2_code')['code6'].size()
        peers = (g['l2_code'].map(sizes) - 1).astype(float)
        per_year.append({
            'text_year': int(y), 'network_year': int(y) + 1, 'firms': int(len(g)),
            'sw2_groups': int(sizes.size),
            'median_peer': float(peers.median()), 'mean_peer': round(float(peers.mean()), 2),
            'group_size_median': float(sizes.median()),
        })
        pooled.append(peers.values)
        pooled_group.append(sizes.values.astype(float))
    return {
        'per_year': per_year,
        'median_peer_pooled_firm_level': float(np.median(np.concatenate(pooled))),
        'median_group_size_pooled': float(np.median(np.concatenate(pooled_group))),
        'n_firm_years': int(sum(len(p) for p in pooled)),
        'n_firm_years_unmatched': n_unmatched,
    }


# ---------- 2) 相似度抽样 ----------
def sampled_sims(years, k=FOCAL_SAMPLE, seed=SEED, corpus='self'):
    """每年返回 (focal_idx_sorted_sims_desc, n_nodes, codes)。"""
    out = {}
    rng = np.random.default_rng(seed)
    for y in years:
        codes, Xn = build_similarity(y, corpus=corpus)
        n = len(codes)
        idx = rng.choice(n, size=min(k, n), replace=False)
        idx.sort()
        sims = {}
        B = 200
        for s in range(0, len(idx), B):
            blk = idx[s:s + B]
            S = (Xn[blk] @ Xn.T).toarray().astype(np.float32)
            for r, i in enumerate(blk):
                row = S[r]
                row[i] = -1.0                      # 排除自连接
                sims[int(i)] = np.sort(row)[::-1]  # 降序，便于 searchsorted
        out[y] = {'n': n, 'sims': sims, 'codes': codes}
        print(f"  [{y}] n={n} sampled={len(sims)}", flush=True)
        del Xn, codes
    return out


def pooled_median_peer(sims_by_year, tau):
    vals = []
    for y, d in sims_by_year.items():
        for arr in d['sims'].values():
            vals.append(int((arr > tau).sum()))
    return float(np.median(vals)), vals


def calibrate(sims_by_year, target, lo=0.0, hi=0.5, iters=50):
    """二分 + 区间细扫（中位 peer 数是阶梯函数）。"""
    def med_at(t):
        v, _ = pooled_median_peer(sims_by_year, t)
        return v
    for _ in range(iters):
        if hi - lo < 1e-10:
            break
        mid = (lo + hi) / 2
        m = med_at(mid)
        if m > target:
            lo = mid
        elif m < target:
            hi = mid
        else:
            hi = mid
    if med_at(lo) == target:
        return float(lo)
    grid = np.linspace(lo, hi, 401)
    hits = [g for g in grid if med_at(g) == target]
    if hits:
        return float((hits[0] + hits[-1]) / 2)
    return float((lo + hi) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, nargs='*', default=None)
    ap.add_argument('--corpus', choices=['self', 'teacher'], default='self')
    args = ap.parse_args()
    years = args.years if args.years else CORPORA[args.corpus]['years']
    corpus = args.corpus

    print('=== 1) 申万二级目标 ===', flush=True)
    tgt = sw2_target(years, corpus=corpus)
    target = round(tgt['median_peer_pooled_firm_level'])
    print(json.dumps({k: v for k, v in tgt.items() if k != 'per_year'}, ensure_ascii=False))
    for r in tgt['per_year']:
        print('   ', r)

    print('\n=== 2) 相似度抽样 ===', flush=True)
    sims = sampled_sims(years, corpus=corpus)

    print('\n=== 3) τ 校准 ===', flush=True)
    tau = calibrate(sims, target)
    med, vals = pooled_median_peer(sims, tau)
    print(f'τ = {tau:.10f} | 池化中位 peer = {med} (目标 {target}) | n={len(vals)}')

    # 备选口径：组层面中位规模
    alt_target = round(tgt['median_group_size_pooled'])
    tau_alt = calibrate(sims, alt_target) if alt_target != target else None

    # 诊断曲线
    grid = np.round(np.arange(0.02, 0.32, 0.01), 4)
    curve = [{'tau': float(g), 'pooled_median_peer': pooled_median_peer(sims, float(g))[0]} for g in grid]

    # 逐年诊断（在选定 τ 下）
    per_year = []
    for y, d in sims.items():
        peers = np.array([int((a > tau).sum()) for a in d['sims'].values()])
        yr = tgt['per_year'][[r['text_year'] for r in tgt['per_year']].index(y)]
        per_year.append({
            'text_year': y, 'network_year': y + 1, 'n_nodes': d['n'],
            'sampled_focals': len(peers),
            'median_peer_at_tau': float(np.median(peers)),
            'mean_peer_at_tau': round(float(peers.mean()), 2),
            'isolated_at_tau': int((peers == 0).sum()),
            'sw2_median_peer': yr['median_peer'],
            'sw2_groups': yr['sw2_groups'],
        })

    rep = {
        'method': 'B1 τ 校准（申万二级中位 peer 数目标）',
        'run_at': datetime.now(timezone.utc).isoformat(),
        'inputs': {
            'texts': str(CORPORA[corpus]['path'].relative_to(ROOT)).replace('\\', '/'),
            'corpus': corpus,
            'sw2_map': str(LAT.relative_to(ROOT)).replace('\\', '/'),
            'df_threshold': 0.25,
            'focal_sample_per_year': FOCAL_SAMPLE,
            'seed': SEED,
        },
        'sw2_target': tgt,
        'frozen_target_primary': target,
        'target_definition': '公司层面中位 peer 数（组内 N-1），跨年池化；peer 计数在同一文本年的全部入网公司内',
        'tau_primary': tau,
        'tau_primary_check_median': med,
        'alt_secondary': ({'target_group_level_median': alt_target, 'tau': tau_alt}
                          if tau_alt is not None else None),
        'per_year_at_tau': per_year,
        'curve': curve,
        'note': '正式 network_edges 由 B2 用 --tau 产出；本文件的 τ 为 B1 校准结果，注册 freeze 后使用。',
    }
    REPD.mkdir(parents=True, exist_ok=True)
    suffix = '' if corpus == 'self' else '_' + corpus
    out_json = REPD / ('network_tau_calibration' + suffix + '.json')
    probe_fp = PROBE.with_name('tau_calibration_probe' + suffix + '.csv')
    out_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    probe_fp.parent.mkdir(parents=True, exist_ok=True)
    with open(probe_fp, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['tau', 'pooled_median_peer'])
        for r in curve:
            w.writerow([r['tau'], r['pooled_median_peer']])
    print('\nper-year at tau:')
    for r in per_year:
        print('   ', r)
    print(f"\nwrote {out_json} and {probe_fp}")


if __name__ == '__main__':
    main()
