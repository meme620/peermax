"""S3A smoketest：CF MOM 价格/规模过滤变体对比（2026-08-14）。

对抗审查 P1-5：当前过滤 = 样本内市值最低 10% 分位 + 上市<6 个月（fillna(False) 保留）。
A&H 原文 = 前月末股价 <$5 剔除。本脚本复刻 cf_mom_replication.py ln/h1 逻辑，
仅替换过滤规则，对比多空收益与 t 值：

  filter=price5        : 前月末收盘价 < 5 元剔除（A&H $5 的 A 股近似）
  filter=market_micro  : 全市场（MEQ 面板）当月市值最低 10% 分位剔除
  filter=none          : 不过滤（参考系）
  （当前口径 baseline 直接取 cf_mom_replication_ln_h1.json）

用法: python scripts/analyst_smoketest_cf_filters.py <price5|market_micro|none>
输出: data/processed/analyst/smoketest_cf_filter_<mode>.json
"""
from dates import next_month as _next_month
from statsutil import nw_t

import json, sys
from pathlib import Path
import pandas as pd
import numpy as np

EDGES = Path('data/processed/analyst/edges')
RET = 'data/processed/monthly_returns_clean.csv'
MEQ = 'data/processed/market_equity_panel.csv'
OUT = Path('data/processed/analyst')
NET_VER = 'coverage_win12_fy12_v2'

MODE = sys.argv[1] if len(sys.argv) > 1 else 'price5'


def main():
    ret = pd.read_csv(RET, dtype=str, keep_default_na=False)
    ret['firm_id'] = ret['firm_id'].str.replace(r'^(sz|sh)', '', regex=True)
    ret['R'] = pd.to_numeric(ret['R'], errors='coerce')
    ret = ret[ret['R'].notna()]
    months = sorted(ret['month'].unique())
    ret_pivot = ret.pivot(index='firm_id', columns='month', values='R')

    meq = pd.read_csv(MEQ, dtype=str, keep_default_na=False)
    meq['date'] = pd.to_datetime(meq['date'])
    meq['month'] = meq['date'].dt.strftime('%Y-%m')
    meq['me'] = pd.to_numeric(meq['market_equity'], errors='coerce')
    meq['close'] = pd.to_numeric(meq['close'], errors='coerce')
    meq = meq[['firm_id', 'month', 'me', 'close']].dropna(subset=['me'])

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
                                 'r_pred': float(d['r_pred'].iloc[0])}),
            include_groups=False).reset_index()
        g['month_pred'] = pred_month
        g['month_form'] = cutoff[:7]
        rows.append(g)

    panel = pd.concat(rows, ignore_index=True)
    meq_p = meq.rename(columns={'me': 'me_form', 'close': 'close_form'})
    panel = panel.merge(meq_p, left_on=['focal_id', 'month_form'], right_on=['firm_id', 'month'], how='left')

    # ---- 过滤规则 ----
    n_pre = len(panel)
    if MODE == 'price5':
        # A&H 近似：前月末（cutoff 月）收盘价 < 5 元剔除；未匹配价格不剔除（保留）
        panel = panel[~(panel['close_form'].fillna(999.0) < 5.0)]
    elif MODE == 'market_micro':
        # 全市场（MEQ 面板当月所有公司）市值最低 10% 分位
        me_snap = meq[['month', 'me']].groupby('month')['me'].quantile(0.10).rename('me_q10')
        panel = panel.merge(me_snap, left_on='month_form', right_index=True, how='left')
        panel = panel[~(panel['me_form'].fillna(1e30) < panel['me_q10'])]
    elif MODE == 'none':
        pass
    else:
        raise SystemExit(f'unknown mode {MODE}')
    # 次新股过滤保持（三种模式一致，price5 之外可对比）
    panel['ipo_date'] = panel['focal_id'].map(ipo_map)
    panel['months_since_ipo'] = (pd.to_datetime(panel['month_pred']) - pd.to_datetime(panel['ipo_date'], errors='coerce')).dt.days / 30.4
    panel['recent_ipo'] = panel['months_since_ipo'] < 6
    panel = panel[~panel['recent_ipo'].fillna(False)]
    print(f'filter={MODE}: {n_pre} -> {len(panel)} ({(1 - len(panel) / n_pre) * 100:.1f}% removed)', flush=True)

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
    out = {
        'mode': MODE,
        'n_months': len(df),
        'sample': f"{df['month'].min()} ~ {df['month'].max()}",
        'mean_ew_ls_pct': round(float(ew.mean()) * 100, 4),
        't_iid': round(float(ew.mean() / ew.std() * np.sqrt(len(ew))), 2) if len(ew) > 2 else None,
        't_nw1': round(float(nw_t(ew.values, L=1)), 2) if len(ew) > 2 else None,
        'note': 'ln 权重 h1 口径；与 baseline(cf_mom_replication_ln_h1: 0.7687%, t=1.63/1.66) 对比',
    }
    (OUT / f'smoketest_cf_filter_{MODE}.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
