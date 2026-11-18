"""S3A smoketest：REV 因子构造变体对 CF MOM alpha 的影响（2026-08-14）。

对抗审查 P0-3：headline α=1.50%（t=4.00）依赖自建 REV 因子（10 组 EW，无过滤）。
本脚本构造 4 个 REV 变体，逐一与 CF 多空月收益 + FF4 因子做回归（NW L=1），
比较 FF5 口径 alpha 的稳定性。

变体:
  rev10_ew         : 10 组 EW，输家-赢家（当前口径）
  rev10_vw         : 10 组 VW（形成期市值加权）
  rev5_ew          : 5 组 EW
  rev10_ew_nomicro : 10 组 EW，先剔除当月市值最低 10% 分位（与 CF 面板过滤一致）

用法: python scripts/analyst_smoketest_rev_variants.py
输出: data/processed/analyst/smoketest_rev_variants.json
"""
from statsutil import nw_ols

import json
from pathlib import Path
import pandas as pd
import numpy as np

RET = 'data/processed/monthly_returns_clean.csv'
MEQ = 'data/processed/market_equity_panel.csv'
CF = 'data/processed/analyst/cf_mom_replication_ln_h1.json'
FAC = 'data/processed/factor_returns_daily.csv'
REV_BASE = 'data/processed/rev_factor_monthly.csv'
OUT = Path('data/processed/analyst/smoketest_rev_variants.json')


def main():
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

    rev_base = pd.read_csv(REV_BASE, dtype=str, keep_default_na=False)
    rev_base['REV'] = pd.to_numeric(rev_base['REV'], errors='coerce')
    rev_base = rev_base.set_index('month')['REV']

    cf = json.loads(Path(CF).read_text(encoding='utf-8'))
    df = pd.DataFrame(cf['monthly'])
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
    print(f'CF monthly series: {len(df)} months', flush=True)

    variants = {}
    for name, n_groups, vw, drop_micro in [
        ('rev10_ew', 10, False, False),
        ('rev10_vw', 10, True, False),
        ('rev5_ew', 5, False, False),
        ('rev10_ew_nomicro', 10, False, True),
    ]:
        rows = []
        for i in range(len(months) - 1):
            sig_m, ret_m = months[i], months[i + 1]
            sig = ret[ret['month'] == sig_m][['firm_id', 'R']].rename(columns={'R': 'sig'})
            r = ret[ret['month'] == ret_m][['firm_id', 'R']]
            d = sig.merge(r, on='firm_id', how='inner')
            if drop_micro:
                me_sig = meq[meq['month'] == sig_m][['firm_id', 'me']]
                if len(me_sig):
                    thr = me_sig['me'].quantile(0.10)
                    keep = set(me_sig.loc[me_sig['me'] >= thr, 'firm_id'])
                    d = d[d['firm_id'].isin(keep)]
            if len(d) < 30:
                continue
            d['q'] = pd.qcut(d['sig'], n_groups, labels=False, duplicates='drop')
            loser = d[d['q'] == d['q'].min()]
            winner = d[d['q'] == d['q'].max()]
            if vw:
                me_sig = meq[meq['month'] == sig_m][['firm_id', 'me']].rename(columns={'me': 'me_s'})
                loser = loser.merge(me_sig, on='firm_id', how='left')
                winner = winner.merge(me_sig, on='firm_id', how='left')
                lw = loser['me_s'].fillna(0)
                ww = winner['me_s'].fillna(0)
                lr = float(np.average(loser['R'], weights=lw)) if lw.sum() > 0 else loser['R'].mean()
                wr = float(np.average(winner['R'], weights=ww)) if ww.sum() > 0 else winner['R'].mean()
            else:
                lr = loser['R'].mean()
                wr = winner['R'].mean()
            rows.append({'month': ret_m, 'REV': lr - wr})
        rev = pd.DataFrame(rows)
        rev_s = rev.set_index('month')['REV']
        m2 = df.merge(rev, on='month', how='left')
        d5 = m2.dropna(subset=['ew_ls', 'MKT', 'SMB', 'HML', 'MOM', 'REV'])
        y = d5['ew_ls'].values
        X = np.column_stack([np.ones(len(y)), d5['MKT'], d5['SMB'], d5['HML'], d5['MOM'], d5['REV']])
        beta, se, t = nw_ols(X, y, L=1)
        d4 = m2.dropna(subset=['ew_ls', 'MKT', 'SMB', 'HML', 'MOM'])
        y4 = d4['ew_ls'].values
        X4 = np.column_stack([np.ones(len(y4)), d4['MKT'], d4['SMB'], d4['HML'], d4['MOM']])
        b4, s4, t4 = nw_ols(X4, y4, L=1)
        corr = float(rev_s.corr(rev_base.reindex(rev_s.index)))
        variants[name] = {
            'n_months': int(len(d5)),
            'rev_mean_pct': round(float(rev['REV'].mean()) * 100, 4),
            'corr_with_rev10_ew': round(corr, 4),
            'ff5_alpha_pct': round(float(beta[0]) * 100, 4),
            'ff5_alpha_t': round(float(t[0]), 2),
            'ff5_rev_beta_pct': round(float(beta[5]) * 100, 2),
            'ff5_rev_t': round(float(t[5]), 2),
            'ff4_alpha_pct': round(float(b4[0]) * 100, 4),
            'ff4_alpha_t': round(float(t4[0]), 2),
        }
        print(name, json.dumps(variants[name], ensure_ascii=False), flush=True)

    OUT.write_text(json.dumps(variants, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
