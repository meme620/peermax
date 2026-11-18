"""S3A smoketest C4：h12 复合窗口的月数要求对比（2026-08-14）。

自设计：min_count=1（有 1 个月可得就算累计收益）。
外部方案：要求窗口内可得月数 >= 8 或 >= 11（完整窗口，Jegadeesh-Titman 式严格口径）。
一次运行输出 1/8/11 三个要求的 ln/h12 EW 结果。
用法: python scripts/analyst_smoketest_min_months.py
输出: data/processed/analyst/smoketest_c4_min_months.json
"""
# --- scripts/lib bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from dates import next_month as _next_month
from statsutil import nw_t
# --- end scripts/lib bootstrap ---

import json
from pathlib import Path
import pandas as pd
import numpy as np

EDGES = Path('data/processed/analyst/edges')
RET = 'data/processed/monthly_returns_clean.csv'
OUT = Path('data/processed/analyst/smoketest_c4_min_months.json')
NET_VER = 'coverage_win12_fy12_v2'
MIN_MONTHS_LIST = [1, 8, 11]


def main():
    ret = pd.read_csv(RET, dtype=str, keep_default_na=False)
    ret['firm_id'] = ret['firm_id'].str.replace(r'^(sz|sh)', '', regex=True)
    ret['R'] = pd.to_numeric(ret['R'], errors='coerce')
    ret = ret[ret['R'].notna()]
    months = sorted(ret['month'].unique())
    ret_pivot = ret.pivot(index='firm_id', columns='month', values='R')

    # 每个 (focal, month) 的 cf（ln/h12）与可得月数
    rows = []
    for cf in sorted(EDGES.glob(f'edges_{NET_VER}_*.csv')):
        cutoff = cf.stem.replace(f'edges_{NET_VER}_', '')
        pred_month = _next_month(cutoff[:7])
        if pred_month not in months:
            continue
        t_idx = months.index(pred_month)
        if t_idx < 12:
            continue
        win = months[t_idx - 12:t_idx - 1]  # t-12 .. t-2（11 个月）
        sub = ret_pivot[win]
        avail = sub.notna().sum(axis=1).rename('n_avail')
        cum = (1 + sub).prod(axis=1, min_count=1) - 1
        r_sig = cum.reset_index().rename(columns={0: 'r_sig'})
        edges = pd.read_csv(cf, dtype=str, usecols=['focal_id', 'source_id', 'common_analyst_count'])
        c = pd.to_numeric(edges['common_analyst_count'], errors='coerce').fillna(0)
        lnw = np.log1p(c)
        denom = lnw.groupby(edges['focal_id']).transform('sum')
        edges['weight'] = lnw / denom.replace(0, np.nan)
        edges = edges[edges['weight'].notna() & (edges['weight'] > 0)]
        r_pred = ret[ret['month'] == pred_month][['firm_id', 'R']].rename(columns={'R': 'r_pred'})
        m = edges.merge(r_sig, left_on='source_id', right_on='firm_id', how='inner')
        m = m.merge(r_pred, left_on='focal_id', right_on='firm_id', how='inner', suffixes=('', '_p'))
        if m.empty:
            continue
        g = m.groupby('focal_id').apply(
            lambda d: pd.Series({'cf': float(np.average(d['r_sig'], weights=d['weight'])),
                                 'r_pred': float(d['r_pred'].iloc[0])}),
            include_groups=False).reset_index()
        g['month'] = pred_month
        avail_df = avail.reset_index()
        avail_df.columns = ['firm_id', 'n_avail']
        g = g.merge(avail_df, left_on='focal_id', right_on='firm_id', how='left')
        rows.append(g)

    panel = pd.concat(rows, ignore_index=True)
    print(f'panel rows: {len(panel)}', flush=True)

    results = {}
    for mm in MIN_MONTHS_LIST:
        sub = panel[panel['n_avail'] >= mm]
        monthly = []
        for mp, s in sub.groupby('month'):
            if len(s) < 10:
                continue
            s = s.copy()
            s['q'] = pd.qcut(s['cf'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            rec = {'month': mp}
            for q in range(1, 6):
                gq = s[s['q'] == q]
                rec[f'ew_q{q}'] = gq['r_pred'].mean()
            rec['ew_ls'] = rec['ew_q5'] - rec['ew_q1']
            monthly.append(rec)
        df = pd.DataFrame(monthly)
        ew = df['ew_ls'].dropna()
        results[f'n_avail>={mm}'] = {
            'n_focal_months': int(len(sub)),
            'n_months': len(df),
            'mean_ew_ls_pct': round(float(ew.mean()) * 100, 4),
            't_iid': round(float(ew.mean() / ew.std() * np.sqrt(len(ew))), 2) if len(ew) > 2 else None,
            't_nw1': round(float(nw_t(ew.values, L=1)), 2) if len(ew) > 2 else None,
        }
        print(mm, json.dumps(results[f'n_avail>={mm}']), flush=True)

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
