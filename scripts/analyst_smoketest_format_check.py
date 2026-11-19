r"""analyst-network smoketest A1：A_code 格式校验 ^A\d{5}$ 的影响对比（2026-08-14）。

自设计：step5 剔除 6,600 行非法 A_code（'0'/'S31388'/'A3162' 等）。
外部方案：不校验（直接使用全部 ok 记录）。
抽查 3 个月（2018-06/2021-06/2024-06），比较两种方案的覆盖与网络统计。
用法: python scripts/analyst_smoketest_format_check.py
输出: data/processed/analyst/smoketest_a1_format_check.json
"""
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

CLEAN = 'data/processed/analyst/analyst_step3_clean.csv'
OUT = Path('data/processed/analyst/smoketest_a1_format_check.json')
CUTOFFS = ['2018-06-30', '2021-06-30', '2024-06-30']


def main():
    rec = pd.read_csv(CLEAN, dtype=str, keep_default_na=False,
                      usecols=['firm_id', 'Anndats', 'Fpenddt', 'analyst_id', 'mapping_status'])
    rec = rec[rec['mapping_status'] == 'ok']
    rec['report_date'] = pd.to_datetime(rec['Anndats'])
    fy = pd.to_datetime(rec['Fpenddt'], errors='coerce').dt.year
    rec = rec[(fy - rec['report_date'].dt.year).between(0, 1)]
    rec = rec.sort_values('report_date', kind='mergesort')
    dates = rec['report_date'].to_numpy(dtype='datetime64[ns]')
    fmt_ok = rec['analyst_id'].str.match(r'^A\d{5}$', na=False)
    print(f'row-level: total={len(rec)} bad_code_rows={int((~fmt_ok).sum())} ({(~fmt_ok).mean()*100:.4f}%)', flush=True)

    def stats(sub):
        af = defaultdict(set)
        for fid, aid in zip(sub['firm_id'], sub['analyst_id']):
            af[aid].add(fid)
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
        return {'pairs': int(len(sub)), 'analysts': len(af),
                'firms': len(set(f for v in af.values() for f in v)),
                'edges': len(pairs), 'mass': mass,
                'degree_p50': d[int((len(d) - 1) * 0.50)] if d else None}

    out = {}
    for c in CUTOFFS:
        ws = pd.Timestamp(c) - pd.DateOffset(months=12)
        lo = dates.searchsorted(np.datetime64(ws), side='right')
        hi = dates.searchsorted(np.datetime64(c), side='right')
        sub = rec.iloc[lo:hi]
        s_all = sub
        s_fmt = sub[fmt_ok[sub.index]]
        out[c] = {'with_format_check': stats(s_fmt), 'without_check': stats(s_all)}
        print(c, json.dumps(out[c]), flush=True)

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
