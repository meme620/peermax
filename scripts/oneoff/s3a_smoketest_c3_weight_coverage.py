"""S3A smoketest C3：缺失同行收益的处理方案对比（2026-08-14）。

自设计：内 join 后对可得同行重归一化（无覆盖阈值要求）。
外部方案：要求可得同行权重和 >= 阈值（0.5 / 0.8 / 1.0），不足则 focal 该月剔除。
一次运行输出 4 个阈值（0=当前口径 / 0.5 / 0.8 / 1.0）的 ln/h1 EW 结果。
用法: python scripts/data_audit/s3a_smoketest_c3_weight_coverage.py
输出: data/processed/analyst/smoketest_c3_weight_coverage.json
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np

EDGES = Path('data/processed/analyst/edges')
RET = 'data/processed/monthly_returns_clean.csv'
OUT = Path('data/processed/analyst/smoketest_c3_weight_coverage.json')
NET_VER = 'coverage_win12_fy12_v2'
THRESHOLDS = [0.0, 0.5, 0.8, 1.0]


def nw_t(x, L=1):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    T = len(x)
    if T < 3:
        return np.nan
    mu = x.mean()
    se2 = np.sum((x - mu) ** 2) / T / T
    for l in range(1, min(L, T - 1) + 1):
        gamma_l = np.sum((x[:-l] - mu) * (x[l:] - mu)) / T
        se2 += 2 * (1 - l / (L + 1)) * gamma_l / T
    return mu / np.sqrt(se2)


def _next_month(ym):
    y, m = int(ym[:4]), int(ym[5:7])
    m += 1
    if m == 13:
        y, m = y + 1, 1
    return f'{y:04d}-{m:02d}'


def main():
    ret = pd.read_csv(RET, dtype=str, keep_default_na=False)
    ret['firm_id'] = ret['firm_id'].str.replace(r'^(sz|sh)', '', regex=True)
    ret['R'] = pd.to_numeric(ret['R'], errors='coerce')
    ret = ret[ret['R'].notna()]
    months = sorted(ret['month'].unique())

    # 每个 (focal, month) 存 cf 与可得权重和
    rows = []
    for cf in sorted(EDGES.glob(f'edges_{NET_VER}_*.csv')):
        cutoff = cf.stem.replace(f'edges_{NET_VER}_', '')
        pred_month = _next_month(cutoff[:7])
        signal_month = cutoff[:7]
        if pred_month not in months or signal_month not in months:
            continue
        r_sig = ret[ret['month'] == signal_month][['firm_id', 'R']].rename(columns={'R': 'r_sig'})
        edges = pd.read_csv(cf, dtype=str, usecols=['focal_id', 'source_id', 'weight'])
        edges['weight'] = pd.to_numeric(edges['weight'], errors='coerce')
        edges = edges[edges['weight'].notna() & (edges['weight'] > 0)]
        r_pred = ret[ret['month'] == pred_month][['firm_id', 'R']].rename(columns={'R': 'r_pred'})
        m = edges.merge(r_sig, left_on='source_id', right_on='firm_id', how='inner')
        m = m.merge(r_pred, left_on='focal_id', right_on='firm_id', how='inner', suffixes=('', '_p'))
        if m.empty:
            continue
        g = m.groupby('focal_id').apply(
            lambda d: pd.Series({'cf': float(np.average(d['r_sig'], weights=d['weight'])),
                                 'r_pred': float(d['r_pred'].iloc[0]),
                                 'w_avail': float(d['weight'].sum())}),
            include_groups=False).reset_index()
        g['month'] = pred_month
        rows.append(g)

    panel = pd.concat(rows, ignore_index=True)
    print(f'panel rows: {len(panel)}', flush=True)

    results = {}
    for thr in THRESHOLDS:
        sub = panel if thr == 0.0 else panel[panel['w_avail'] >= thr]
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
        results[f'wcoverage>={thr}'] = {
            'n_focal_months': int(len(sub)),
            'n_months': len(df),
            'mean_ew_ls_pct': round(float(ew.mean()) * 100, 4),
            't_iid': round(float(ew.mean() / ew.std() * np.sqrt(len(ew))), 2) if len(ew) > 2 else None,
            't_nw1': round(float(nw_t(ew.values, L=1)), 2) if len(ew) > 2 else None,
        }
        print(thr, json.dumps(results[f'wcoverage>={thr}']), flush=True)

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
