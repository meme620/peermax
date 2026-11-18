# -*- coding: utf-8 -*-
"""S3A 数据使用审计（2026-09-03）：退市/停牌缺失月对 NPM LS 的暴露量化。

复刻主线 NPM 面板（同 9e：microcap Drive mv<10% 分位 + IPO<6 月过滤），
但把 r_pred 内连接丢掉的 focal 行保留，量化：
  1) 每月因 focal 无预测月收益而丢行的数量分布；
  2) 丢行在"全行集"NPM 五分位中的位置（Q5 丢多 → LS 被高估）；
  3) 丢行中"准退市"（pred 起 6 个月无收益）的占比；
  4) 上界模拟：把丢行以 -20%/-30%/-50% 收益计入所在分位，看 EW LS 变化。
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from analyst_cf_mom_replication import _next_month, EDGES, RET, NET_VER

d = Path('data/raw/drive')
mv = pd.read_pickle(d / 'daily_a_share_market_val_3_20260630.pkl')
mv.columns = [c.split('.')[0] for c in mv.columns]
mvm = mv.resample('ME').last()
mvm.index = [c.strftime('%Y-%m') for c in mvm.index]
mvd = mvm.stack().rename('mvd').reset_index()
mvd.columns = ['month', 'firm_id', 'mvd']
mvd['firm_id'] = mvd['firm_id'].astype(str)

ret = pd.read_csv(RET, dtype=str, keep_default_na=False)
ret['firm_id'] = ret['firm_id'].str.replace(r'^(sz|sh)', '', regex=True)
ret['R'] = pd.to_numeric(ret['R'], errors='coerce')
ret = ret[ret['R'].notna()][['firm_id', 'month', 'R']]
months = sorted(ret['month'].unique())
month_idx = {m: i for i, m in enumerate(months)}
last_panel = ret.groupby('firm_id')['month'].max().to_dict()
ret_by_month = {m: g.set_index('firm_id')['R'] for m, g in ret.groupby('month')}

ipo = pd.read_csv('data/processed/firm_id_map.csv', dtype=str)
ipo_map = dict(zip(ipo['code6'], ipo['ipo_date']))

rows = []
for cf in sorted(EDGES.glob(f'edges_{NET_VER}_*.csv')):
    cutoff = cf.stem.replace(f'edges_{NET_VER}_', '')
    pred, sig = _next_month(cutoff[:7]), cutoff[:7]
    if pred not in months or sig not in months:
        continue
    edges = pd.read_csv(cf, dtype=str, usecols=['focal_id', 'source_id', 'weight'])
    edges['weight'] = pd.to_numeric(edges['weight'], errors='coerce')
    edges = edges[edges['weight'].notna() & (edges['weight'] > 0)]
    r_sig = ret_by_month[sig]
    r_pred_s = ret_by_month[pred]
    m = edges.merge(r_sig.rename('r_sig'), left_on='source_id', right_index=True, how='inner')
    if m.empty:
        continue
    g = m.groupby('focal_id').apply(
        lambda x: np.average(x['r_sig'], weights=x['weight']), include_groups=False
    ).rename('cf').reset_index()
    g['r_form'] = g['focal_id'].map(r_sig.rename('r_form'))
    g['r_pred'] = g['focal_id'].map(r_pred_s.rename('r_pred'))
    g = g[g['r_form'].notna()]
    g['npm'] = g['cf'] - g['r_form']
    g['month_pred'], g['month_form'] = pred, sig
    rows.append(g[['focal_id', 'npm', 'r_pred', 'r_form', 'month_pred', 'month_form']])
    if len(rows) % 60 == 0:
        print(f'{len(rows)} cutoffs done', flush=True)

panel = pd.concat(rows, ignore_index=True)
# microcap（形成月末 Drive mv < 当月 10% 分位）+ IPO<6 月，与 9e/主线一致
panel = panel.merge(mvd, left_on=['focal_id', 'month_form'], right_on=['firm_id', 'month'],
                    how='left', suffixes=('', '_mv'))
panel['microcap'] = panel.groupby('month_pred')['mvd'].transform(lambda s: s < s.quantile(0.10))
panel['ipo_date'] = panel['focal_id'].map(ipo_map)
panel['msi'] = (pd.to_datetime(panel['month_pred']) - pd.to_datetime(panel['ipo_date'], errors='coerce')).dt.days / 30.4
panel = panel[~(panel['microcap'].fillna(False) | (panel['msi'] < 6))].copy()

panel['has_pred'] = panel['r_pred'].notna()
# 准退市：pred 起 6 个月内 focal 无面板收益
def _exit(f, pred):
    i = month_idx[pred]
    tail = months[i:i + 6]
    lp = last_panel.get(f)
    return lp is not None and lp < pred  # 最后收益月早于 pred => pred 起全无收益
panel['exit_like'] = [not h and _exit(f, p) for h, f, p in
                      zip(panel['has_pred'], panel['focal_id'], panel['month_pred'])]

n_drop = int((~panel['has_pred']).sum())
n_tot = len(panel)
n_exit = int(panel['exit_like'].sum())
print(f'focal-rows={n_tot}  dropped(no r_pred)={n_drop} ({n_drop/n_tot*100:.2f}%)  exit-like={n_exit} ({n_exit/n_tot*100:.2f}%)', flush=True)

# 丢行在 NPM 五分位中的分布（按全行集在月内排序）
panel['q'] = panel.groupby('month_pred')['npm'].transform(
    lambda s: pd.qcut(s, 5, labels=[1, 2, 3, 4, 5], duplicates='drop'))
dist = panel[~panel['has_pred']].groupby('q', observed=True).size()
tot_q = panel.groupby('q', observed=True).size()
print('dropped by quintile:', {str(k): int(v) for k, v in dist.items()})
print('all by quintile   :', {str(k): int(v) for k, v in tot_q.items()})

# 基线 EW LS（丢行版 = 主线口径）
def ls_with(fill_value=None):
    out = []
    for mp, s in panel.groupby('month_pred'):
        if len(s) < 10:
            continue
        ss = s.copy()
        if fill_value is not None:
            ss.loc[~ss['has_pred'], 'r_pred'] = fill_value
        ew = ss.groupby('q', observed=True)['r_pred'].mean()
        if 1 in ew.index and 5 in ew.index:
            out.append(ew[5] - ew[1])
    a = np.array(out)
    return len(a), a.mean() * 100, a.mean() / a.std() * np.sqrt(len(a))

res = {}
n, m_, t = ls_with(None)
res['observed_dropped'] = {'n_months': n, 'ls_pct': round(m_, 4), 't': round(t, 2)}
print(f'observed LS (drops excluded) = {m_:.4f}% t={t:.2f}', flush=True)
for fv in (-0.20, -0.30, -0.50):
    n, m_, t = ls_with(fv)
    res[f'imputed_{int(fv*100)}pct'] = {'n_months': n, 'ls_pct': round(m_, 4), 't': round(t, 2)}
    print(f'imputed {fv:+.0%}: LS={m_:.4f}% t={t:.2f}', flush=True)

# 丢行 r_form 特征（形成期崩跌 → NPM 高 → Q5）
print('dropped r_form mean =', round(panel.loc[~panel['has_pred'], 'r_form'].mean(), 4),
      ' kept r_form mean =', round(panel.loc[panel['has_pred'], 'r_form'].mean(), 4), flush=True)
print('dropped npm mean =', round(panel.loc[~panel['has_pred'], 'npm'].mean(), 4),
      ' kept npm mean =', round(panel.loc[panel['has_pred'], 'npm'].mean(), 4), flush=True)

# |R|>30% 分布（收益面板质量）
big = ret[ret['R'].abs() > 0.3].copy()
res['absR_gt30_by_year'] = big['month'].str[:4].value_counts().sort_index().to_dict()
res['panel_rows'] = n_tot
res['drop_pct'] = round(n_drop / n_tot * 100, 3)
res['exit_pct'] = round(n_exit / n_tot * 100, 3)
res['dropped_by_q'] = {str(k): int(v) for k, v in dist.items()}
res['all_by_q'] = {str(k): int(v) for k, v in tot_q.items()}
Path('artifacts/s3a_delisting_exposure_20260903.json').write_text(
    json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
print('DONE -> artifacts/s3a_delisting_exposure_20260903.json')
