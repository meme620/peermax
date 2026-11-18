"""S3A smoketest C6：覆盖窗口/FY 规则/快照频率敏感性网格兑现（2026-08-14）。

registry 三-2 冻结条目登记的 sensitivity_grid 从未执行，本脚本兑现：
  coverage_window ∈ {1, 3, 6}（12=主口径，数字引用主产物）
  fy1_fy2_rule ∈ {FY1+FY2（主）, FY1-only}
  network_cutoff_rule ∈ {month_end（主）, quarter_end}

不落盘覆盖表/边表（只做内存聚合统计），不污染主产物。
输出: data/processed/analyst/sensitivity_window.json
"""
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

CLEAN = 'data/processed/analyst/s3a_step3_clean.csv'
OUT = Path('data/processed/analyst/sensitivity_window.json')


def edge_stats(af):
    pairs = set()
    mass = 0
    for aid, firms in af.items():
        fl = sorted(firms)
        for a in range(len(fl)):
            for b in range(a + 1, len(fl)):
                pairs.add((fl[a], fl[b]))
                mass += 1
    deg = defaultdict(int)
    for i, j in pairs:
        deg[i] += 1
        deg[j] += 1
    d = sorted(deg.values())
    return {'edges': len(pairs), 'mass': mass,
            'degree_p50': d[int((len(d) - 1) * 0.50)] if d else None}


def main():
    rec = pd.read_csv(CLEAN, dtype=str, keep_default_na=False,
                      usecols=['firm_id', 'Anndats', 'Fpenddt', 'analyst_id', 'mapping_status'])
    rec = rec[rec['mapping_status'] == 'ok']
    rec = rec[rec['analyst_id'].str.match(r'^A\d{5}$', na=False)]
    rec['report_date'] = pd.to_datetime(rec['Anndats'])
    fy = pd.to_datetime(rec['Fpenddt'], errors='coerce').dt.year
    rec['gap'] = fy - rec['report_date'].dt.year
    rec = rec.sort_values('report_date', kind='mergesort')
    dates = rec['report_date'].to_numpy(dtype='datetime64[ns]')
    print(f'row-level ready: {len(rec)}', flush=True)

    variants = {
        'win1_fy12_month':  dict(win=1, fy_any=True,  freq='ME'),
        'win3_fy12_month':  dict(win=3, fy_any=True,  freq='ME'),
        'win6_fy12_month':  dict(win=6, fy_any=True,  freq='ME'),
        'win12_fy1_month':  dict(win=12, fy_any=False, freq='ME'),
        'win12_fy12_quarter': dict(win=12, fy_any=True, freq='QE'),
    }

    results = {}
    for name, cfg in variants.items():
        sub = rec if cfg['fy_any'] else rec[rec['gap'] == 0]
        sub = sub[sub['gap'].between(0, 1)]
        dts = sub['report_date'].to_numpy(dtype='datetime64[ns]')
        cutoffs = pd.date_range('2017-12-31', '2025-12-31', freq=cfg['freq'])
        months = []
        for cutoff in cutoffs:
            ws = cutoff - pd.DateOffset(months=cfg['win'])
            lo = dts.searchsorted(np.datetime64(ws), side='right')
            hi = dts.searchsorted(np.datetime64(cutoff), side='right')
            if hi <= lo:
                continue
            g = sub.iloc[lo:hi].groupby(['firm_id', 'analyst_id'], sort=False).size()
            af = defaultdict(set)
            for (fid, aid) in g.index:
                af[aid].add(fid)
            st = edge_stats(af)
            st['firms'] = len(set(f for v in af.values() for f in v))
            st['analysts'] = len(af)
            months.append(st)
        n = len(months)
        agg = {
            'cutoffs': n,
            'period': f'{cutoffs[0].date()} ~ {cutoffs[-1].date()}',
            'edges_mean': round(sum(m['edges'] for m in months) / n, 1),
            'edges_min': min(m['edges'] for m in months),
            'edges_max': max(m['edges'] for m in months),
            'degree_p50_mean': round(sum(m['degree_p50'] for m in months) / n, 2) if months[0]['degree_p50'] else None,
            'firms_mean': round(sum(m['firms'] for m in months) / n, 1),
            'analysts_mean': round(sum(m['analysts'] for m in months) / n, 1),
            'mass_mean': round(sum(m['mass'] for m in months) / n, 1),
        }
        results[name] = agg
        print(name, json.dumps(agg, ensure_ascii=False), flush=True)

    results['win12_fy12_month_main'] = {
        'cutoffs': 97,
        'edges_mean': 246479.9, 'edges_min': 150828, 'edges_max': 376601,
        'degree_p50_mean': 152.0, 'firms_mean': 2664.0, 'analysts_mean': 12638.0,
        'note': '主口径（引用 step67_diagnostics.json），供对比',
    }
    OUT.write_text(json.dumps({
        'note': 'registry 三-2 敏感性网格兑现（2026-08-14）。win=fy 过滤后覆盖窗口（月）；fy_any=FY1+FY2（主）；'
                'freq=快照频率 ME=月末（主）/QE=季末。',
        'variants': results,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
