"""S3A 9C：NPM 的 Fama-MacBeth 回归（对标 Long Table 5, p.49）。
控制变量：size(Drive市值 log)、BM(TTM)、REV(上月收益)、MOM(12-2 累计)、TURN(官方换手率)、ILLIQ(Amihud 月|R|/成交额)。
手写 FM：每月横截面 OLS -> 系数序列 -> 均值 + NW(t) 与 iid(t)。
"""
import pickle, sys, warnings
import pandas as pd, numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
from analyst_cf_mom_replication import _next_month, EDGES, RET, MEQ, NET_VER
from analyst_meta import attach_meta

d = Path('data/raw/drive')
# Drive 日频宽表 -> 月末
def month_end(df):
    m = df.resample('ME').last()
    m.index = [c.strftime('%Y-%m') for c in m.index]
    m.columns = [c.split('.')[0] for c in m.columns]
    return m
mv = month_end(pickle.load(open(d/'daily_a_share_market_val_3_20260630.pkl','rb')))
bm = month_end(pickle.load(open(d/'daily_book_to_market_ratio_ttm_20260630.pkl','rb')))
tr = month_end(pickle.load(open(d/'daily_turnover_rate_20260630.pkl','rb')))/100.0
am = month_end(pickle.load(open(d/'daily_total_amount_20260630.pkl','rb')))
print("Drive 控制变量月末表就绪", flush=True)

ret = pd.read_csv(RET, dtype=str, keep_default_na=False)
ret['firm_id'] = ret['firm_id'].str.replace(r'^(sz|sh)','',regex=True)
ret['R'] = pd.to_numeric(ret['R'], errors='coerce')
ret = ret[ret['R'].notna()][['firm_id','month','R']]
months = sorted(ret['month'].unique())
meq = pd.read_csv(MEQ, dtype=str, keep_default_na=False)
meq['month'] = pd.to_datetime(meq['date']).dt.strftime('%Y-%m')
meq['me'] = pd.to_numeric(meq['market_equity'], errors='coerce')
meq = meq[['firm_id','month','me']].dropna()
meq_q10 = (meq.groupby('month', as_index=False)['me'].quantile(0.10)
           .rename(columns={'me': 'me_q10_form'}))
ipo_map = dict(zip(pd.read_csv('data/processed/firm_id_map.csv',dtype=str)['code6'],
                   pd.read_csv('data/processed/firm_id_map.csv',dtype=str)['ipo_date']))

rows = []
for cf in sorted(EDGES.glob(f'edges_{NET_VER}_*.csv')):
    cutoff = cf.stem.replace(f'edges_{NET_VER}_','')
    pred_month = _next_month(cutoff[:7])
    if pred_month not in months: continue
    sig = cutoff[:7]
    if sig not in months: continue
    r_sig = ret[ret['month']==sig][['firm_id','R']].rename(columns={'R':'r_sig'})
    r_form = r_sig.rename(columns={'r_sig':'r_form'})
    edges = pd.read_csv(cf, dtype=str, usecols=['focal_id','source_id','weight'])
    edges['weight'] = pd.to_numeric(edges['weight'], errors='coerce')
    edges = edges[edges['weight'].notna() & (edges['weight']>0)]
    r_pred = ret[ret['month']==pred_month][['firm_id','R']].rename(columns={'R':'r_pred'})
    m = edges.merge(r_sig, left_on='source_id', right_on='firm_id', how='inner')
    m = m.merge(r_pred, left_on='focal_id', right_on='firm_id', how='inner', suffixes=('','_p'))
    if m.empty: continue
    g = m.groupby('focal_id').apply(lambda d: pd.Series({
        'cf': float(np.average(d['r_sig'], weights=d['weight'])),
        'r_pred': float(d['r_pred'].iloc[0])}), include_groups=False).reset_index()
    g = g.merge(r_form, left_on='focal_id', right_on='firm_id', how='left')
    g['npm'] = g['cf'] - g['r_form']
    g['month_pred'] = pred_month; g['month_form'] = sig
    rows.append(g)
panel = pd.concat(rows, ignore_index=True)
# 控制变量合并
panel = panel.merge(mv.rename(columns=lambda c: c).stack().rename('mv_').reset_index().rename(
    columns={'level_0':'month','level_1':'firm_id'}), left_on=['firm_id','month_form'], right_on=['firm_id','month'], how='left', suffixes=('','_x'))
# 简化：直接逐变量 merge
def merge_drive(series, name):
    s = series.stack().rename(name).reset_index().rename(columns={'level_0':'month','level_1':'firm_id'})
    return panel.merge(s, left_on=['firm_id','month_form'], right_on=['firm_id','month'], how='left', suffixes=('', '_d'))
panel = merge_drive(mv, 'size_mv'); panel['size'] = np.log(panel['size_mv'])
panel = merge_drive(bm, 'bm_t')
panel = merge_drive(tr, 'turn_t')
panel = merge_drive(am, 'amt_t')
# MOM（12-2 累计，跳过最近月）与 ILLIQ
mom_vals = {}
for i in range(12, len(months)-1):
    w = months[i-12:i-1]  # t-12..t-2
    mom_vals[months[i]] = ret[ret['month'].isin(w)].groupby('firm_id')['R'].apply(lambda x: np.prod(1+x)-1)
mom_df = pd.concat(mom_vals, names=['month','firm_id']).rename('mom').reset_index()
panel = panel.merge(mom_df, left_on=['firm_id','month_form'], right_on=['firm_id','month'], how='left', suffixes=('','_m'))
panel = panel.drop(columns=['month_m'])
panel['illiq'] = panel['r_form'].abs() / panel['amt_t'].replace(0, np.nan)
# 过滤：控制变量缺失行剔除
cols = ['npm','size','bm_t','r_form','mom','turn_t','illiq']
panel = panel.dropna(subset=cols+['r_pred'])
# 2026-09-10 Gate 4.4（spec: 9c_extreme_form_return_filter=sensitivity_only_abs_r_form_lt_0_5）：
# 主口径**不**施加 |r_form|<0.5 极端收益过滤；该过滤仅作显式敏感性（python analyst_fm_regression.py extreme）。
EXTREME_FILTER = len(sys.argv) > 1 and sys.argv[1] == 'extreme'
if EXTREME_FILTER:
    panel = panel[panel['r_form'].abs() < 0.5]
# --- 2026-09-01 R5（review 总账）：对齐主复刻样本过滤 = microcap 形成期市值月内最低 10% + 上市<6 月 ---
_mf = meq.rename(columns={'me':'me_form','month':'month_form'})
panel = panel.merge(_mf[['firm_id','month_form','me_form']], on=['firm_id','month_form'], how='left')
panel = panel.merge(meq_q10.rename(columns={'month':'month_form_q10'}),
                    left_on='month_form', right_on='month_form_q10', how='left')
panel['ipo_date'] = panel['focal_id'].map(ipo_map)
panel['months_since_ipo'] = (pd.to_datetime(panel['month_pred'], errors='coerce')
                             - pd.to_datetime(panel['ipo_date'], errors='coerce')).dt.days/30.4
_micro = (panel['me_form'] < panel['me_q10_form']).fillna(False)
_young = (panel['months_since_ipo'] < 6).fillna(False)
_n0 = len(panel)
panel = panel[~(_micro | _young)]
print(f"主样本过滤（R5）：{_n0} -> {len(panel)} 行", flush=True)

# FM 回归
print(f"panel: {len(panel)} 行 | 各列非空: " + ", ".join(f"{c}={panel[c].notna().sum()}" for c in ['npm','size','bm_t','r_form','mom','turn_t','illiq','r_pred']), flush=True)
X_names = ['npm','size','bm_t','r_form','mom','turn_t','illiq']
coefs = {k: [] for k in X_names}
n_months = 0
_first = True
for mp, sub in panel.groupby('month_pred'):
    if len(sub) < 30: continue
    y = pd.to_numeric(sub['r_pred'], errors='coerce').values.astype(float)
    X = np.column_stack([pd.to_numeric(sub[k], errors='coerce').values.astype(float) for k in X_names]
                        + [np.ones(len(sub))])
    fin = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if _first:
        for j, k in enumerate(X_names+['const']):
            nf = int((~np.isfinite(X[:, j])).sum())
            if nf: print(f'  col {k}: non-finite {nf}', flush=True)
        print(f'  first month {mp}: sub={len(sub)} fin={int(fin.sum())}/{len(fin)}', flush=True)
        _first = False
    X, y = X[fin], y[fin]
    if len(y) < 30:
        continue
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except Exception as e:
        print(f'lstsq err: {e}', flush=True)
        continue
    n_months += 1
    for k, b in zip(X_names+['const'], beta):
        coefs.setdefault(k, []).append(b)
def fm_report(vals):
    a = np.array(vals); T = len(a)
    m = a.mean(); se = a.std(ddof=1)/np.sqrt(T)
    # NW L=1，Bartlett 权 (1-1/(L+1))=0.5——2026-09-01 修复 R4：原 2*gam 等价权重 1，与 nw_t 不一致
    resid = a - m
    var = resid.var(ddof=1) if T>1 else np.nan
    gam = np.cov(resid[:-1], resid[1:], ddof=1)[0,1] if T>2 else 0
    if T < 2:
        return m, np.nan, np.nan, T
    nw_se = np.sqrt((var + 2*0.5*gam)/T)
    return m, m/se if se>0 else np.nan, m/nw_se if nw_se>0 else np.nan, T
print(f'panel month_pred: nunique={panel[chr(39)+chr(39)] if False else panel["month_pred"].nunique()}, isna={panel["month_pred"].isna().sum()}, sample={panel["month_pred"].dropna().unique()[:3]}', flush=True)
print(f'n_months={n_months}', flush=True)
res = {'n_months': n_months, 'sample': f"{panel['month_pred'].min()}~{panel['month_pred'].max()}",
       'sample_filter': ('microcap 形成月完整市场 q10 + 上市<6月（与主复刻同）；主口径不含 |r_form|<0.5 极端收益过滤'
                         + ('（本文件为 extreme 敏感性：额外施加 |r_form|<0.5）' if EXTREME_FILTER else '')),
       'extreme_form_filter_applied': bool(EXTREME_FILTER),
       'coefs': {}, 'benchmark': {'long2025_table5_npm': 'NPM 系数 0.012~0.023, t 2.60~6.26',
                                  'note': '本项目控制变量：size(Drive市值)/BM(TTM)/REV/MOM/TURN(官方)/ILLIQ(Amihud |R_form|/amt)'}}
if EXTREME_FILTER:
    res['output_path'] = 'reports/data_gates/G0/s3a_9c_fm_regression_extreme.json'
for k in X_names+['const']:
    m, t_iid, t_nw, T = fm_report(coefs[k])
    res['coefs'][k] = {'beta': round(float(m), 5), 't_iid': round(float(t_iid), 2),
                       't_nw1': round(float(t_nw), 2) if t_nw==t_nw else None, 'n_months': T}
    print(f"  {k:8s}: beta={m:+.5f} t_iid={t_iid:+.2f} t_nw={t_nw:+.2f}", flush=True)
import json
attach_meta(res, __file__, NET_VER)
_out = Path('reports/data_gates/G0/s3a_9c_fm_regression_extreme.json') if EXTREME_FILTER \
    else Path('reports/data_gates/G0/s3a_9c_fm_regression.json')
_out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
print(f"DONE -> {_out}")
