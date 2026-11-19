"""analyst-network 9D：ST 剔除敏感性——CF/NPM 复刻剔除 ST 股（st_flag，Drive 批次1）。
口径：与 analyst_npm_replication 完全一致，仅形成期股票若为 ST（月末 flag=True）则剔除。
"""
import pickle, sys, warnings
import pandas as pd, numpy as np
from pathlib import Path
from fileio import read_pickle
warnings.filterwarnings('ignore')
from dates import next_month as _next_month
from statsutil import nw_t
from analyst_cf_mom_replication import EDGES, RET, MEQ, NET_VER

ST = read_pickle('data/raw/drive/stock_st_flag_20260630.pkl')
ST.columns = [c.split('.')[0] for c in ST.columns]

def st_month_map():
    m = ST.resample('ME').last().T   # 月末状态，行=股票
    m.columns = [c.strftime('%Y-%m') for c in m.columns]
    m.index.name = 'firm_id'
    return m

def main():
    weight_mode = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ('count','ln') else 'count'
    ret = pd.read_csv(RET, dtype=str, keep_default_na=False)
    ret['firm_id'] = ret['firm_id'].str.replace(r'^(sz|sh)','',regex=True)
    ret['R'] = pd.to_numeric(ret['R'], errors='coerce')
    ret = ret[ret['R'].notna()][['firm_id','month','R']]
    months = sorted(ret['month'].unique())
    meq = pd.read_csv(MEQ, dtype=str, keep_default_na=False)
    meq['month'] = pd.to_datetime(meq['date']).dt.strftime('%Y-%m')
    meq['me'] = pd.to_numeric(meq['market_equity'], errors='coerce')
    meq = meq[['firm_id','month','me']].dropna()
    ipo_map = dict(zip(pd.read_csv('data/processed/firm_id_map.csv',dtype=str)['code6'],
                       pd.read_csv('data/processed/firm_id_map.csv',dtype=str)['ipo_date']))
    stm = st_month_map()

    rows = []
    for cf in sorted(EDGES.glob(f'edges_{NET_VER}_*.csv')):
        cutoff = cf.stem.replace(f'edges_{NET_VER}_','')
        pred_month = _next_month(cutoff[:7])
        if pred_month not in months: continue
        signal_month = cutoff[:7]
        if signal_month not in months: continue
        r_sig = ret[ret['month']==signal_month][['firm_id','R']].rename(columns={'R':'r_sig'})
        r_form = r_sig.rename(columns={'r_sig':'r_form'})
        edges = pd.read_csv(cf, dtype=str, usecols=['focal_id','source_id','common_analyst_count','weight'])
        if weight_mode=='ln':
            c = pd.to_numeric(edges['common_analyst_count'], errors='coerce').fillna(0)
            lnw = np.log1p(c); edges['weight'] = lnw/lnw.groupby(edges['focal_id']).transform('sum').replace(0,np.nan)
        else:
            edges['weight'] = pd.to_numeric(edges['weight'], errors='coerce')
        edges = edges[edges['weight'].notna() & (edges['weight']>0)]
        # ST 剔除（形成期月末状态）
        st_row = stm.get(signal_month, pd.Series(dtype=bool)).fillna(False)
        st_firms = set(st_row[st_row].index)
        edges = edges[~edges['focal_id'].isin(st_firms)]
        r_pred = ret[ret['month']==pred_month][['firm_id','R']].rename(columns={'R':'r_pred'})
        m = edges.merge(r_sig, left_on='source_id', right_on='firm_id', how='inner')
        m = m.merge(r_pred, left_on='focal_id', right_on='firm_id', how='inner', suffixes=('','_p'))
        if m.empty: continue
        g = m.groupby('focal_id').apply(lambda d: pd.Series({'cf': float(np.average(d['r_sig'], weights=d['weight'])),
            'r_pred': float(d['r_pred'].iloc[0])}), include_groups=False).reset_index()
        g = g.merge(r_form, left_on='focal_id', right_on='firm_id', how='left')
        g['npm'] = g['cf'] - g['r_form']
        g['month_pred'] = pred_month; g['month_form'] = signal_month
        rows.append(g)
    panel = pd.concat(rows, ignore_index=True)
    meq_p = meq.rename(columns={'me':'me_form'})
    panel = panel.merge(meq_p, left_on=['focal_id','month_form'], right_on=['firm_id','month'], how='left')
    # 2026-09-10 Gate 4.5：microcap 阈值取完整形成月市场横截面 q10（与主复刻同口径）；
    # 旧版用研究子样本 transform(quantile) 会把阈值绑定到研究样本边界。
    q10_full = meq.groupby('month')['me'].quantile(0.10)
    panel['ipo_date'] = panel['focal_id'].map(ipo_map)
    panel['msi'] = (pd.to_datetime(panel['month_pred'])-pd.to_datetime(panel['ipo_date'], errors='coerce')).dt.days/30.4
    panel['microcap'] = panel['me_form'] < panel['month_form'].map(q10_full)
    panel['recent_ipo'] = panel['msi'] < 6
    panel = panel[~(panel['microcap'].fillna(False)|panel['recent_ipo'].fillna(False))]
    monthly = []
    for mp, sub in panel.groupby('month_pred'):
        if len(sub) < 10: continue
        sub = sub.copy(); sub['q'] = pd.qcut(sub['npm'], 5, labels=[1,2,3,4,5], duplicates='drop')
        rec = {'month': mp}
        for q in range(1,6):
            gg = sub[sub['q']==q]; rec[f'ew_q{q}'] = gg['r_pred'].mean()
        rec['ew_ls'] = rec['ew_q5']-rec['ew_q1']; monthly.append(rec)
    df = pd.DataFrame(monthly)
    res = {'weight_mode': weight_mode, 'st_excluded': True, 'n_months': len(df),
           'mean_ew_ls_pct': round(float(df['ew_ls'].mean())*100, 4),
           't_iid': round(float(df['ew_ls'].mean()/df['ew_ls'].std()*np.sqrt(len(df))), 2),
           't_nw1': round(float(nw_t(df['ew_ls'].values, L=1)), 2)}
    import json
    p = Path(f'data/processed/analyst/npm_st_excluded_{weight_mode}.json')
    p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"ST 剔除 {weight_mode}: EW LS={res['mean_ew_ls_pct']}% t={res['t_iid']} (nw1 {res['t_nw1']}) -> {p.name}")

if __name__ == '__main__':
    main()
