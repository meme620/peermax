"""analyst-network smoketest 补跑A：剔除超级分析师（跨月覆盖>300 家的 6 人）后 CF MOM 是否仍显著（2026-08-14）。

问题：sens_specs 的 no_top1 变体只对比了网络统计（边数 −4.5%），没有对比 CF 动量。
本脚本：①从 v2 覆盖表计算跨月覆盖>300 家的分析师（应=6）；②剔除后重建网络（不落盘边表，
逐月构建）；③跑 ln/h1 CF MOM（microcap+次新股过滤与 cf_mom_replication 一致）；④FF4/FF5 alpha。
用法: python scripts/analyst_smoketest_cf_mom_no_super.py
输出: data/processed/analyst/smoketest_cf_mom_no_super.json
"""
from dates import next_month as _next_month
from statsutil import nw_t
from statsutil import nw_ols

import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

COV = 'data/processed/analyst/firm_analyst_coverage.csv'
RET = 'data/processed/monthly_returns_clean.csv'
MEQ = 'data/processed/market_equity_panel.csv'
FAC = 'data/processed/factor_returns_daily.csv'
REV_FAC = 'data/processed/rev_factor_monthly.csv'
OUT = Path('data/processed/analyst/smoketest_cf_mom_no_super.json')
TOP_EXCLUDE_THRESHOLD = 300


def main():
    cov = pd.read_csv(COV, dtype=str, keep_default_na=False,
                      usecols=['network_cutoff', 'firm_id', 'analyst_id'])
    # 超级分析师：全样本跨月 distinct 覆盖公司数 > 300（与 sens_specs 同口径）
    ac = cov.groupby('analyst_id')['firm_id'].nunique()
    top = set(ac[ac > TOP_EXCLUDE_THRESHOLD].index)
    print(f'top-analyst exclusion set (> {TOP_EXCLUDE_THRESHOLD} firms across months): {len(top)}', flush=True)
    print('ids:', sorted(top), flush=True)

    ret = pd.read_csv(RET, dtype=str, keep_default_na=False)
    ret['firm_id'] = ret['firm_id'].str.replace(r'^(sz|sh)', '', regex=True)
    ret['R'] = pd.to_numeric(ret['R'], errors='coerce')
    ret = ret[ret['R'].notna()]
    months = sorted(ret['month'].unique())

    meq = pd.read_csv(MEQ, dtype=str, keep_default_na=False)
    meq['date'] = pd.to_datetime(meq['date'])
    meq['month'] = meq['date'].dt.strftime('%Y-%m')
    meq['me'] = pd.to_numeric(meq['market_equity'], errors='coerce')
    meq = meq[['firm_id', 'month', 'me']].dropna()

    ipo = pd.read_csv('data/processed/firm_id_map.csv', dtype=str)
    ipo_map = dict(zip(ipo['code6'].str.strip(), ipo['ipo_date'].str.strip()))

    rows = []
    for cutoff in sorted(cov['network_cutoff'].unique()):
        pred_month = _next_month(cutoff[:7])
        signal_month = cutoff[:7]
        if pred_month not in months or signal_month not in months:
            continue
        sub = cov[cov['network_cutoff'] == cutoff]
        sub = sub[~sub['analyst_id'].isin(top)]   # 剔除超级分析师
        af = defaultdict(set)
        for fid, aid in zip(sub['firm_id'], sub['analyst_id']):
            af[aid].add(fid)
        # pair 计数 + ln 权重（与 cf_mom ln 口径一致）
        pair_count = defaultdict(int)
        for aid, firms in af.items():
            fl = sorted(firms)
            for a in range(len(fl)):
                for b in range(a + 1, len(fl)):
                    pair_count[(fl[a], fl[b])] += 1
        denom = defaultdict(float)
        for (i, j), c in pair_count.items():
            denom[i] += np.log1p(c)
            denom[j] += np.log1p(c)
        edges = []
        for (i, j), c in pair_count.items():
            wi = np.log1p(c) / denom[i] if denom[i] > 0 else 0.0
            wj = np.log1p(c) / denom[j] if denom[j] > 0 else 0.0
            if wi > 0:
                edges.append((i, j, wi))
            if wj > 0:
                edges.append((j, i, wj))
        ed = pd.DataFrame(edges, columns=['focal_id', 'source_id', 'weight'])
        r_sig = ret[ret['month'] == signal_month][['firm_id', 'R']].rename(columns={'R': 'r_sig'})
        r_pred = ret[ret['month'] == pred_month][['firm_id', 'R']].rename(columns={'R': 'r_pred'})
        m = ed.merge(r_sig, left_on='source_id', right_on='firm_id', how='inner')
        m = m.merge(r_pred, left_on='focal_id', right_on='firm_id', how='inner', suffixes=('', '_p'))
        if m.empty:
            continue
        g = m.groupby('focal_id').apply(
            lambda d: pd.Series({'cf': float(np.average(d['r_sig'], weights=d['weight'])),
                                 'r_pred': float(d['r_pred'].iloc[0])}),
            include_groups=False).reset_index()
        g['month_pred'] = pred_month
        g['month_form'] = cutoff[:7]
        rows.append(g)

    panel = pd.concat(rows, ignore_index=True)
    meq_p = meq.rename(columns={'me': 'me_form'})
    panel = panel.merge(meq_p, left_on=['focal_id', 'month_form'], right_on=['firm_id', 'month'], how='left')
    panel['ipo_date'] = panel['focal_id'].map(ipo_map)
    panel['months_since_ipo'] = (pd.to_datetime(panel['month_pred']) - pd.to_datetime(panel['ipo_date'], errors='coerce')).dt.days / 30.4
    panel['microcap'] = panel.groupby('month_pred')['me_form'].transform(lambda s: s < s.quantile(0.10))
    panel['recent_ipo'] = panel['months_since_ipo'] < 6
    n_pre = len(panel)
    panel = panel[~(panel['microcap'].fillna(False) | panel['recent_ipo'].fillna(False))]
    print(f'price filters: {n_pre} -> {len(panel)}', flush=True)

    monthly = []
    for mp, sub in panel.groupby('month_pred'):
        if len(sub) < 10:
            continue
        sub = sub.copy()
        sub['q'] = pd.qcut(sub['cf'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        rec = {'month': mp}
        for q in range(1, 6):
            gq = sub[sub['q'] == q]
            rec[f'ew_q{q}'] = gq['r_pred'].mean()
        rec['ew_ls'] = rec['ew_q5'] - rec['ew_q1']
        monthly.append(rec)
    df = pd.DataFrame(monthly)
    ew = df['ew_ls'].dropna()

    # FF4/FF5 alpha
    f = pd.read_csv(FAC, dtype=str, keep_default_na=False)
    f['date'] = pd.to_datetime(f['date'])
    f['month'] = f['date'].dt.strftime('%Y-%m')
    for col in ['MKT_d', 'SMB_d', 'HML_d', 'MOM_d']:
        f[col] = pd.to_numeric(f[col], errors='coerce')
    fm = f.groupby('month').apply(lambda d: pd.Series({
        'MKT': float((1 + d['MKT_d']).prod() - 1),
        'SMB': float((1 + d['SMB_d']).prod() - 1),
        'HML': float((1 + d['HML_d']).prod() - 1),
        'MOM': float((1 + d['MOM_d']).prod() - 1),
    }), include_groups=False)
    df = df.merge(fm, left_on='month', right_index=True, how='inner')
    rev = pd.read_csv(REV_FAC, dtype=str, keep_default_na=False)
    rev['REV'] = pd.to_numeric(rev['REV'], errors='coerce')
    df = df.merge(rev[['month', 'REV']], on='month', how='left').sort_values('month').reset_index(drop=True)

    d5 = df.dropna(subset=['ew_ls', 'MKT', 'SMB', 'HML', 'MOM', 'REV'])
    y5 = d5['ew_ls'].values
    X5 = np.column_stack([np.ones(len(y5)), d5['MKT'], d5['SMB'], d5['HML'], d5['MOM'], d5['REV']])
    b5, spec_no_turnover, t5 = nw_ols(X5, y5, L=1)
    d4 = df.dropna(subset=['ew_ls', 'MKT', 'SMB', 'HML', 'MOM'])
    y4 = d4['ew_ls'].values
    X4 = np.column_stack([np.ones(len(y4)), d4['MKT'], d4['SMB'], d4['HML'], d4['MOM']])
    b4, se4, t4 = nw_ols(X4, y4, L=1)

    res = {
        'excluded_top_analysts': sorted(top),
        'n_excluded': len(top),
        'threshold': TOP_EXCLUDE_THRESHOLD,
        'n_months': len(df),
        'sample': f"{df['month'].min()} ~ {df['month'].max()}",
        'mean_ew_ls_pct': round(float(ew.mean()) * 100, 4),
        't_iid': round(float(ew.mean() / ew.std() * np.sqrt(len(ew))), 2),
        't_nw1': round(float(nw_t(ew.values, L=1)), 2),
        'ff5_alpha_pct': round(float(b5[0]) * 100, 4),
        'ff5_alpha_t': round(float(t5[0]), 2),
        'ff4_alpha_pct': round(float(b4[0]) * 100, 4),
        'ff4_alpha_t': round(float(t4[0]), 2),
        'baseline_ln_h1': {'mean_ew_ls_pct': 0.7687, 't_nw1': 1.66,
                           'ff5_alpha_pct': 1.5048, 'ff5_alpha_t': 4.00},
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=2), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
