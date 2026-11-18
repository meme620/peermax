"""S3A 9E：大小盘分层（csi300/csi500 成分内/外）的 NPM 五分组 + Drive 市值 VW。
复用 9C 的 NPM 面板逻辑；分层按形成期月末成分静态归属（简化）。
"""
import pickle, sys, warnings
import pandas as pd, numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')
from analyst_cf_mom_replication import _next_month, EDGES, RET, MEQ, NET_VER
from analyst_meta import attach_meta

d = Path('data/raw/drive')
c300 = pickle.load(open(d/'daily_csi300_components.pkl','rb')); c300.columns=[c.split('.')[0] for c in c300.columns]
c500 = pickle.load(open(d/'daily_csi500_components.pkl','rb')); c500.columns=[c.split('.')[0] for c in c500.columns]
mv = pickle.load(open(d/'daily_a_share_market_val_3_20260630.pkl','rb')); mv.columns=[c.split('.')[0] for c in mv.columns]
def month_end(df):
    m = df.resample('ME').last(); m.index=[c.strftime('%Y-%m') for c in m.index]
    return m
# R9（2026-09-01 审核）已知口径：成分数据起点前（csi300 2005/csi500 2007 以前）或缺失月，
# fillna(False) 将其归"成分外（小盘）"层；影响限于早期月份归属，判读按此口径解读。
c300m, c500m, mvm = month_end(c300), month_end(c500), month_end(mv)

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
rows = []
for cf in sorted(EDGES.glob(f'edges_{NET_VER}_*.csv')):
    cutoff = cf.stem.replace(f'edges_{NET_VER}_',''); pred = _next_month(cutoff[:7])
    if pred not in months: continue
    sig = cutoff[:7]
    if sig not in months: continue
    r_sig = ret[ret['month']==sig][['firm_id','R']].rename(columns={'R':'r_sig'})
    r_form = r_sig.rename(columns={'r_sig':'r_form'})
    edges = pd.read_csv(cf, dtype=str, usecols=['focal_id','source_id','weight'])
    edges['weight'] = pd.to_numeric(edges['weight'], errors='coerce')
    edges = edges[edges['weight'].notna() & (edges['weight']>0)]
    r_pred = ret[ret['month']==pred][['firm_id','R']].rename(columns={'R':'r_pred'})
    m = edges.merge(r_sig, left_on='source_id', right_on='firm_id', how='inner')
    m = m.merge(r_pred, left_on='focal_id', right_on='firm_id', how='inner', suffixes=('','_p'))
    if m.empty: continue
    g = m.groupby('focal_id').apply(lambda d: pd.Series({'cf': float(np.average(d['r_sig'], weights=d['weight'])),
        'r_pred': float(d['r_pred'].iloc[0])}), include_groups=False).reset_index()
    g = g.merge(r_form, left_on='focal_id', right_on='firm_id', how='left')
    g['npm'] = g['cf'] - g['r_form']
    g['month_pred'] = pred; g['month_form'] = sig
    rows.append(g)
panel = pd.concat(rows, ignore_index=True)
# 市值 + 成分分层
panel = panel.merge(mvm.stack().rename('mvd').reset_index().rename(columns={'level_0':'month','level_1':'firm_id'}),
                    left_on=['firm_id','month_form'], right_on=['firm_id','month'], how='left')
panel = panel.merge(c300m.stack().rename('in300').reset_index().rename(columns={'level_0':'month','level_1':'firm_id'}),
                    left_on=['firm_id','month_form'], right_on=['firm_id','month'], how='left')
panel = panel.merge(c500m.stack().rename('in500').reset_index().rename(columns={'level_0':'month','level_1':'firm_id'}),
                    left_on=['firm_id','month_form'], right_on=['firm_id','month'], how='left')
panel['ipo_date'] = panel['focal_id'].map(ipo_map)
panel['msi'] = (pd.to_datetime(panel['month_pred'])-pd.to_datetime(panel['ipo_date'],errors='coerce')).dt.days/30.4
# 2026-09-10 Gate 4.6：microcap 改用**完整形成月市场横截面 q10**（与主复刻同口径），
# 不再用研究子样本分位；市场价值基础 = market_equity_panel（总市值，登记偏离）。
meq_q10 = meq.groupby('month')['me'].quantile(0.10).rename('me_q10_form')
panel = panel.merge(meq[['firm_id', 'month', 'me']].rename(columns={'month': 'month_form', 'me': 'me_form'}),
                    on=['firm_id', 'month_form'], how='left')
panel = panel.merge(meq_q10.reset_index().rename(columns={'month': 'month_form'}),
                    on='month_form', how='left')
panel['microcap'] = panel['me_form'] < panel['me_q10_form']
panel = panel[~(panel['microcap'].fillna(False)|(panel['msi']<6))]
# 分层：大（in300）/ 中（in500 且非 in300）/ 小（其余已知成分）/ unknown（成分信息缺失）
panel['mem_known'] = panel['in300'].notna() | panel['in500'].notna()
panel['in300'] = panel['in300'].fillna(False)
panel['in500'] = panel['in500'].fillna(False)
panel['layer'] = np.select([panel['in300'], panel['in500']], ['large','mid'], default='small')
panel.loc[~panel['mem_known'], 'layer'] = 'unknown'
print('layer 分布：', panel['layer'].value_counts().to_dict(), flush=True)

res = {}
for layer in ['large','mid','small','unknown']:
    sub = panel[panel['layer']==layer]
    monthly = []
    for mp, s in sub.groupby('month_pred'):
        if len(s) < 10: continue
        s = s.copy(); s['q'] = pd.qcut(s['npm'], 5, labels=[1,2,3,4,5], duplicates='drop')
        ew = s.groupby('q')['r_pred'].mean()
        monthly.append(ew[5]-ew[1])
    a = np.array(monthly)
    res[layer] = {'n_months': len(a), 'ew_ls_pct': round(float(a.mean())*100, 4),
                  't_iid': round(float(a.mean()/a.std()*np.sqrt(len(a))), 2)}
    print(f"{layer}: n={len(a)} EW LS={a.mean()*100:.4f}% t={a.mean()/a.std()*np.sqrt(len(a)):.2f}", flush=True)
# Drive 市值 VW（全样本）
vw = []
for mp, s in panel.groupby('month_pred'):
    if len(s) < 10: continue
    s = s.copy(); s['q'] = pd.qcut(s['npm'], 5, labels=[1,2,3,4,5], duplicates='drop')
    def vw_ret(q):
        g = s[s['q']==q].dropna(subset=['mvd'])
        return float(np.average(g['r_pred'], weights=g['mvd'])) if len(g) else np.nan
    vw.append(vw_ret(5)-vw_ret(1))
a = np.array([v for v in vw if v==v])
res['vw_drive'] = {'n_months': len(a), 'ew_ls_pct': round(float(a.mean())*100, 4), 't_iid': round(float(a.mean()/a.std()*np.sqrt(len(a))), 2)}
print(f"VW(Drive 市值): n={len(a)} LS={a.mean()*100:.4f}% t={a.mean()/a.std()*np.sqrt(len(a)):.2f}", flush=True)
import json
attach_meta(res, __file__, NET_VER)
Path('reports/data_gates/G0/s3a_9e_subsample.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
print("DONE -> reports/data_gates/G0/s3a_9e_subsample.json")
