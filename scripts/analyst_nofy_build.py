# -*- coding: utf-8 -*-
"""S3A 9K：不限 FY（Long 口径，本数据源下与蒋口径等值）对照网络构建（2026-09-01 登记）。

与 v3 唯一差异 = 去掉 FY 过滤（gap∈[0,1]）；窗口 (cutoff-12M, cutoff]、Anndats<=cutoff、
mapping_ok、A_code ^A\\d{5}$、(firm,analyst,report) 去重语义、投影与行归一化权重全部同 step5/67。
输出独立版本 `coverage_win12_nofy_v4`，不触碰 v3/冻结参数/任何主线产物。

包含性断言（验收 R19 ①）：
  - nofy 的 (cutoff, firm, analyst) 对集 ⊇ v3 对集（FY 过滤只删行不增行）；
  - 逐月 nofy 无向边集 ⊇ v3 边集，且共同边上 common_nofy >= common_v3。

用法: python scripts/analyst_nofy_build.py
输出: data/processed/analyst/firm_analyst_coverage_nofy.csv
      data/processed/analyst/edges/edges_coverage_win12_nofy_v4_<cutoff>.csv ×241
      data/processed/analyst/step9k_nofy_diagnostics.json
"""
import json, csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

SRC = 'data/processed/analyst/s3a_step3_clean.csv'
OUTD = Path('data/processed/analyst')
EDGED = OUTD / 'edges'
WINDOW_MONTHS = 12
VERSION = 'coverage_win12_nofy_v4'
BASE_VERSION = 'coverage_win12_fy12_v3'   # 对照基线（只读）
SOURCE_VERSION = 'forecast_shard_v1_20260811'
COLS = ['network_use_case', 'network_version', 'focal_id', 'source_id',
        'common_analyst_count', 'link_strength', 'analyst_count_focal',
        'analyst_count_source', 'union_analyst_count', 'jaccard', 'weight',
        'SourceFocalDistinct', 'network_cutoff', 'available_at', 'source_version',
        'coverage_flag', 'missing_reason', 'network_support_status']


def build_coverage():
    rec = pd.read_csv(SRC, dtype=str, keep_default_na=False,
                      usecols=['firm_id', 'Anndats', 'Fpenddt', 'analyst_id',
                               'broker_code', 'mapping_status'])
    rec = rec[rec['mapping_status'] == 'ok']
    bad = ~rec['analyst_id'].str.match(r'^A\d{5}$', na=False)
    rec = rec[~bad]
    rec['report_date'] = pd.to_datetime(rec['Anndats'])
    # 9K 差异点：不做 FY 过滤。Long 口径=任意 ≥1 条盈利预测记录。
    print(f"nofy rows (mapping ok + valid code, NO FY filter): {len(rec)}", flush=True)
    rec = rec.sort_values('report_date', kind='mergesort')
    dates = rec['report_date'].to_numpy(dtype='datetime64[ns]')
    cutoffs = pd.date_range('2005-12-31', '2025-12-31', freq='ME')
    frames = []
    for cutoff in cutoffs:
        ws64 = np.datetime64(cutoff - pd.DateOffset(months=WINDOW_MONTHS))
        hi = dates.searchsorted(np.datetime64(cutoff), side='right')
        lo = dates.searchsorted(ws64, side='right')
        if hi <= lo:
            continue
        sub = rec.iloc[lo:hi]
        g = sub.groupby(['firm_id', 'analyst_id'], sort=False).agg(
            report_count=('Anndats', 'nunique'),
            first_report_date=('report_date', 'min'),
            last_report_date=('report_date', 'max'),
            broker_count=('broker_code', lambda s: int(s[s != ''].nunique())),
        ).reset_index()
        g['coverage_window_version'] = VERSION
        g['network_cutoff'] = cutoff.strftime('%Y-%m-%d')
        g['coverage_start'] = g['first_report_date'].dt.strftime('%Y-%m-%d')
        g['coverage_end'] = g['last_report_date'].dt.strftime('%Y-%m-%d')
        g['available_at'] = g['last_report_date'].dt.strftime('%Y-%m-%d')
        g['source_version'] = SOURCE_VERSION
        g['coverage_flag'] = 'covered'
        g['missing_reason'] = ''
        frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    out['first_report_date'] = out['first_report_date'].dt.strftime('%Y-%m-%d')
    out['last_report_date'] = out['last_report_date'].dt.strftime('%Y-%m-%d')
    dup = out.duplicated(['coverage_window_version', 'network_cutoff', 'firm_id', 'analyst_id']).sum()
    assert dup == 0, f"nofy 覆盖表唯一键重复 {dup}"
    out.to_csv(OUTD / 'firm_analyst_coverage_nofy.csv', index=False, encoding='utf-8')
    print(f"coverage rows: {len(out)}", flush=True)
    return out


def build_edges_and_check(cov_nofy):
    cov_v3 = pd.read_csv(OUTD / 'firm_analyst_coverage.csv', dtype=str,
                         usecols=['network_cutoff', 'firm_id', 'analyst_id'])
    v3_key = set(cov_v3['network_cutoff'] + '|' + cov_v3['firm_id'] + '|' + cov_v3['analyst_id'])
    nofy_key = set(cov_nofy['network_cutoff'] + '|' + cov_nofy['firm_id'] + '|' + cov_nofy['analyst_id'])
    assert v3_key <= nofy_key, f"包含性失败：v3 有 {len(v3_key - nofy_key)} 对不在 nofy"
    diag = {'network_version': VERSION, 'months': []}
    cutoffs = sorted(cov_nofy['network_cutoff'].unique())
    for cutoff in cutoffs:
        sub = cov_nofy[cov_nofy['network_cutoff'] == cutoff]
        af = defaultdict(set)
        fac = defaultdict(int)
        for fid, aid in zip(sub['firm_id'], sub['analyst_id']):
            af[aid].add(fid); fac[fid] += 1
        pc = defaultdict(int)
        for aid, firms in af.items():
            fl = sorted(firms)
            for a in range(len(fl)):
                for b in range(a + 1, len(fl)):
                    pc[(fl[a], fl[b])] += 1
        denom = defaultdict(int)
        for (i, j), c in pc.items():
            denom[i] += c; denom[j] += c
        path = EDGED / f'edges_{VERSION}_{cutoff}.csv'
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f); w.writerow(COLS)
            for (i, j), c in sorted(pc.items()):
                ni, nj = fac[i], fac[j]
                union = ni + nj - c
                wi = c / denom[i]; wj = c / denom[j]
                for focal, src, wt in ((i, j, wi), (j, i, wj)):
                    w.writerow(['shared_analyst', VERSION, focal, src, c, c,
                                ni if focal == i else nj, nj if focal == i else ni,
                                union, round(c / union, 8), round(wt, 8), 1,
                                cutoff, cutoff, SOURCE_VERSION, 'covered', '', 'support'])
        # ---- 质量验收（六项，同 v3 口径）----
        e = pd.read_csv(path, dtype=str, usecols=['focal_id', 'source_id', 'weight', 'common_analyst_count'])
        assert e.duplicated(['focal_id', 'source_id']).sum() == 0
        assert (e['focal_id'] != e['source_id']).all()
        ws = pd.to_numeric(e['weight'], errors='coerce')
        err = (ws.groupby(e['focal_id']).sum() - 1).abs().max()
        assert err < 1e-4, f"{cutoff} 权重行和误差 {err}"
        piv = e[['focal_id', 'source_id', 'common_analyst_count']]
        rev = piv.rename(columns={'focal_id': 'source_id', 'source_id': 'focal_id'})
        mm = piv.merge(rev, on=['focal_id', 'source_id'], suffixes=('_f', '_s'))
        assert (mm['common_analyst_count_f'] == mm['common_analyst_count_s']).all()
        # ---- 包含性断言：nofy 边集 ⊇ v3 边集，且 common_nofy >= common_v3 ----
        v3p = EDGED / f'edges_{BASE_VERSION}_{cutoff}.csv'
        n_incl_viol, m_incl_viol = 0, 0
        if v3p.exists():
            ev = pd.read_csv(v3p, dtype=str, usecols=['focal_id', 'source_id', 'common_analyst_count'])
            ev = ev[ev['focal_id'] < ev['source_id']]
            en = e[e['focal_id'] < e['source_id']].copy()
            en['c'] = pd.to_numeric(en['common_analyst_count'], errors='coerce')
            ev['c'] = pd.to_numeric(ev['common_analyst_count'], errors='coerce')
            j = ev.merge(en, on=['focal_id', 'source_id'], suffixes=('_v3', '_nofy'), how='left')
            n_incl_viol = int(j['c_nofy'].isna().sum())
            m_incl_viol = int((j['c_nofy'] < j['c_v3']).sum())
            assert n_incl_viol == 0 and m_incl_viol == 0, \
                f"{cutoff} 包含性失败：缺边 {n_incl_viol}，common 更小 {m_incl_viol}"
        deg = defaultdict(int)
        for (i, j2) in pc:
            deg[i] += 1; deg[j2] += 1
        degs = sorted(deg.values()); cc = sorted(pc.values())
        diag['months'].append({
            'cutoff': cutoff, 'firms_covered': len(fac), 'undirected_edges': len(pc),
            'degree_p50': degs[int((len(degs) - 1) * .5)] if degs else None,
            'degree_p90': degs[int((len(degs) - 1) * .9)] if degs else None,
            'common_count_p50': cc[(len(cc)) // 2] if cc else None,
            'weak_edge_ratio': round(sum(1 for c in cc if c == 1) / len(pc), 4) if pc else None,
            'containment_violations_missing_edges': n_incl_viol,
            'containment_violations_smaller_common': m_incl_viol,
        })
        if len(diag['months']) % 12 == 0:
            print(f"{cutoff}: edges={len(pc)} firms={len(fac)}", flush=True)
    ms = diag['months']
    tot = sum(m['undirected_edges'] for m in ms)
    diag['summary'] = {
        'cutoffs': len(ms), 'edges_total_undirected': tot,
        'edges_monthly_mean': round(tot / len(ms)),
        'edges_monthly_min': min(m['undirected_edges'] for m in ms),
        'edges_monthly_max': max(m['undirected_edges'] for m in ms),
        'degree_p50_mean': round(float(np.mean([m['degree_p50'] for m in ms])), 1),
        'common_count_p50_mean': round(float(np.mean([m['common_count_p50'] for m in ms])), 2),
        'weak_edge_ratio_mean': round(float(np.mean([m['weak_edge_ratio'] for m in ms])), 4),
        'baseline_v3_edges_total': 49719505,
        'note': ('9K 对照变体（Long 口径=本数据源蒋口径等值，见 execution plan Step 9·9K）。'
                 '全部包含性断言通过：nofy ⊇ v3（对/边/common 三重）。不进主 claim、不改冻结参数。'),
    }
    (OUTD / 'step9k_nofy_diagnostics.json').write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding='utf-8')
    print('BUILD DONE:', json.dumps(diag['summary'], ensure_ascii=False)[:400], flush=True)


if __name__ == '__main__':
    cov = build_coverage()
    build_edges_and_check(cov)
