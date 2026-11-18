"""S3A NPM 复刻（Long et al. 2025 Table 3 锚点，2026-08-16）。

NPM(i, t-1) = CF(i, t-1) − R(i, t-1)   （论文式 1：连接公司收益 − 焦点自身收益）
分组变量 NPM，预测 R(i, t)；其余口径与 analyst_cf_mom_replication.py 完全一致
（同一网络版本=NET_VER（默认来自统一 S3A spec）、同一 microcap/次新股过滤、同一 5 组分位）。

用法: python analyst_npm_replication.py [weight=count|ln]
锚点: Long 2025 Table 3 —— NPM EW 多空 1.28%/月, t=5.20（CSMAR 2004-2022）
输出: data/processed/analyst/npm_replication[_{weight}][_network].json
"""
import json, sys
from pathlib import Path
import hashlib
import pandas as pd
import numpy as np

from analyst_cf_mom_replication import nw_t, _next_month, EDGES, RET, MEQ, NET_VER, SPEC, _SPEC
from analyst_meta import attach_meta

OUT = Path('data/processed/analyst/npm_replication.json')
_SPEC_HASH = hashlib.sha256(SPEC.read_bytes()).hexdigest()
_INPUT_HASHES = _SPEC['provenance']['input_hashes']

def main():
    global NET_VER
    weight_mode = sys.argv[1] if len(sys.argv) > 1 else 'count'
    if weight_mode not in ('count', 'ln'):
        raise ValueError('weight must be count or ln')
    if len(sys.argv) > 2:
        if sys.argv[2] not in {'coverage_win12_fy12_v3', 'coverage_win12_nofy_v4'}:
            raise ValueError('network must be coverage_win12_fy12_v3 or coverage_win12_nofy_v4')
        NET_VER = sys.argv[2]
    # 2026-09-10 Gate 4.7（9F）：收益口径显式变体（raw 主口径 / adj 后复权敏感性）
    returns_mode = sys.argv[3] if len(sys.argv) > 3 else 'raw'
    if returns_mode not in {'raw', 'adj'}:
        raise ValueError('returns must be raw or adj')
    _RET = Path(_SPEC['inputs']['returns_post_adjusted']) if returns_mode == 'adj' else RET
    ret = pd.read_csv(_RET, dtype=str, keep_default_na=False)
    ret['firm_id'] = ret['firm_id'].str.replace(r'^(sz|sh)', '', regex=True)
    ret['R'] = pd.to_numeric(ret['R'], errors='coerce')
    ret = ret[ret['R'].notna()]
    ret = ret[['firm_id', 'month', 'R']]
    months = sorted(ret['month'].unique())
    meq = pd.read_csv(MEQ, dtype=str, keep_default_na=False)
    meq['date'] = pd.to_datetime(meq['date'])
    meq['month'] = meq['date'].dt.strftime('%Y-%m')
    meq['me'] = pd.to_numeric(meq['market_equity'], errors='coerce')
    meq = meq[['firm_id', 'month', 'me']].dropna()
    # 2026-09-06 Step 10/Gate 1: full forming-month market q10,
    # computed before the network/return research join.
    meq_q10 = (meq.groupby('month', as_index=False)['me']
                  .quantile(0.10)
                  .rename(columns={'me': 'me_q10_form'}))
    ipo = pd.read_csv('data/processed/firm_id_map.csv', dtype=str)
    ipo_map = dict(zip(ipo['code6'].str.strip(), ipo['ipo_date'].str.strip()))

    rows = []
    for cf in sorted(EDGES.glob(f'edges_{NET_VER}_*.csv')):
        cutoff = cf.stem.replace(f'edges_{NET_VER}_', '')
        pred_month = _next_month(cutoff[:7])
        if pred_month not in months:
            continue
        signal_month = cutoff[:7]
        if signal_month not in months:
            continue
        r_sig = ret[ret['month'] == signal_month][['firm_id', 'R']].rename(columns={'R': 'r_sig'})
        r_form = r_sig.rename(columns={'r_sig': 'r_form'})   # focal 信号月收益（NPM 的减项）

        edges = pd.read_csv(cf, dtype=str, usecols=['focal_id', 'source_id', 'common_analyst_count', 'weight'])
        if weight_mode == 'ln':
            c = pd.to_numeric(edges['common_analyst_count'], errors='coerce').fillna(0)
            lnw = np.log1p(c)
            denom = lnw.groupby(edges['focal_id']).transform('sum')
            edges['weight'] = lnw / denom.replace(0, np.nan)
        else:
            edges['weight'] = pd.to_numeric(edges['weight'], errors='coerce')
        edges = edges[edges['weight'].notna() & (edges['weight'] > 0)]

        r_pred = ret[ret['month'] == pred_month][['firm_id', 'R']].rename(columns={'R': 'r_pred'})
        m = edges.merge(r_sig, left_on='source_id', right_on='firm_id', how='inner')
        m = m.merge(r_pred, left_on='focal_id', right_on='firm_id', how='inner', suffixes=('', '_p'))
        if m.empty:
            continue
        g = m.groupby('focal_id').apply(
            lambda d: pd.Series({'cf': float(np.average(d['r_sig'], weights=d['weight'])),
                                 'r_pred': float(d['r_pred'].iloc[0])}), include_groups=False).reset_index()
        g = g.merge(r_form, left_on='focal_id', right_on='firm_id', how='left')
        g['npm'] = g['cf'] - g['r_form']
        g['month_pred'] = pred_month
        g['month_form'] = signal_month
        rows.append(g)
        if len(rows) % 24 == 0:
            print(f"signal months done: {len(rows)} (at {pred_month})", flush=True)

    panel = pd.concat(rows, ignore_index=True)
    meq_p = meq.rename(columns={'me': 'me_form'})
    panel = panel.merge(meq_p, left_on=['focal_id', 'month_form'], right_on=['firm_id', 'month'], how='left')
    panel = panel.merge(meq_q10.rename(columns={'month': 'month_form_q10'}),
                        left_on='month_form', right_on='month_form_q10', how='left')
    panel['ipo_date'] = panel['focal_id'].map(ipo_map)
    panel['months_since_ipo'] = (pd.to_datetime(panel['month_pred']) - pd.to_datetime(panel['ipo_date'], errors='coerce')).dt.days / 30.4
    panel['microcap'] = panel['me_form'] < panel['me_q10_form']
    panel['recent_ipo'] = panel['months_since_ipo'] < 6
    n_pre = len(panel)
    panel = panel[~(panel['microcap'].fillna(False) | panel['recent_ipo'].fillna(False))]
    print(f"价格/规模过滤: {n_pre} -> {len(panel)}", flush=True)

    monthly = []
    for mp, sub in panel.groupby('month_pred'):
        if len(sub) < 10:
            continue
        sub = sub.copy()
        sub['q'] = pd.qcut(sub['npm'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        rec = {'month': mp}
        for q in range(1, 6):
            gg = sub[sub['q'] == q]
            rec[f'ew_q{q}'] = gg['r_pred'].mean()
            gw = gg.dropna(subset=['me_form'])
            rec[f'vw_q{q}'] = float(np.average(gw['r_pred'], weights=gw['me_form'])) if len(gw) else np.nan
        rec['ew_ls'] = rec['ew_q5'] - rec['ew_q1']
        rec['vw_ls'] = rec['vw_q5'] - rec['vw_q1']
        monthly.append(rec)
    df = pd.DataFrame(monthly)

    results = {'weight_mode': weight_mode, 'signal': 'NPM = CF - R_form (Long 2025 式1)',
               'n_months': len(df),
               'sample': f"{df['month'].min()} ~ {df['month'].max()}" if len(df) else 'EMPTY',
               'mean_monthly_ret_pct': {}}
    for col in ['ew_q1', 'ew_q2', 'ew_q3', 'ew_q4', 'ew_q5', 'ew_ls', 'vw_ls']:
        s = df[col].dropna()
        results['mean_monthly_ret_pct'][col] = round(float(s.mean()) * 100, 4) if len(s) else None
    ew = df['ew_ls'].dropna(); vw = df['vw_ls'].dropna()
    results['t_stats'] = {
        'ew_ls_iid': round(float(ew.mean() / ew.std() * np.sqrt(len(ew))), 2) if len(ew) > 2 else None,
        'ew_ls_nw1': round(float(nw_t(ew.values, L=1)), 2) if len(ew) > 2 else None,
        'vw_ls_iid': round(float(vw.mean() / vw.std() * np.sqrt(len(vw))), 2) if len(vw) > 2 else None,
        'vw_ls_nw1': round(float(nw_t(vw.values, L=1)), 2) if len(vw) > 2 else None,
        'n_valid_months': {'ew': int(len(ew)), 'vw': int(len(vw))},
    }
    results['network_version'] = NET_VER
    results['benchmark'] = {
        'long2025_npm_ls_ew_pct': 1.28, 'long2025_npm_t': 5.20,
        'long2025_note': 'Table 3 Panel A：NPM 五分位 EW 多空（Q5-Q1），CSMAR 2004-2022；'
                         '本项目 CNRDS 全市场 + 覆盖判定=NET_VER 标注版本（fy12=严口径 / nofy=Long 等值口径）+ 样本期见本文件 sample 字段；'
                         '对照时注意数据源与样本期差异（9K 已消除覆盖判定一项，见 execution plan Step 9）。',
    }
    results['monthly'] = df.to_dict(orient='records')

    attach_meta(results, __file__, network_version=NET_VER, returns_mode=returns_mode)
    suffix = '' if weight_mode == 'count' else f'_{weight_mode}'
    if NET_VER == 'coverage_win12_nofy_v4':
        suffix += '_nofy'
    if returns_mode == 'adj':
        suffix += '_adj'
    out_path = OUT.with_name(OUT.stem + suffix + OUT.suffix)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"weight={weight_mode} -> {out_path.name}")
    print(json.dumps({k: v for k, v in results.items() if k != 'monthly'}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
