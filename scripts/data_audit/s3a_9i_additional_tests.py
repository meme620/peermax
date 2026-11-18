# -*- coding: utf-8 -*-
"""S3A 补做检验 P1（对标 AH/Long/蒋丹凌）：
1. 双变量分层：按反转(REV)或行业动量分 3 组，组内按 NPM 分 5 组的多空（对标 AH bivariate / 蒋表 5 注 4）
2. 十分位：NPM 分 10 组多空（AH untabulated decile）
3. 累计收益曲线：NPM 排序后 t+1..t+12 持有收益（蒋图 1 / AH Figure 1）

输入: v3 边表 + 统一 spec 指定的全期收益面板
输出: reports/data_gates/G0/s3a_9i_bivariate_decile_cumret.json
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from s3a_cf_mom_replication import nw_t, _next_month, EDGES, RET, NET_VER
from s3a_meta import attach_meta

OUT = Path('reports/data_gates/G0/s3a_9i_bivariate_decile_cumret.json')
OUT.parent.mkdir(parents=True, exist_ok=True)

MAX_HOLD = 12  # 累计收益最长持有月


def main():
    ret = pd.read_csv(RET, dtype=str, keep_default_na=False)
    ret['firm_id'] = ret['firm_id'].str.replace(r'^(sz|sh)', '', regex=True)
    ret['R'] = pd.to_numeric(ret['R'], errors='coerce')
    ret = ret[ret['R'].notna()][['firm_id', 'month', 'R']]
    months = sorted(ret['month'].unique())
    ret_pivot = ret.pivot(index='firm_id', columns='month', values='R')
    # 2026-09-01 R5：对齐主复刻样本过滤（microcap 形成期市值月内最低 10% + 上市<6 月）
    from s3a_cf_mom_replication import MEQ
    meq = pd.read_csv(MEQ, dtype=str, keep_default_na=False)
    meq['month'] = pd.to_datetime(meq['date']).dt.strftime('%Y-%m')
    meq['me'] = pd.to_numeric(meq['market_equity'], errors='coerce')
    meq = meq[['firm_id', 'month', 'me']].dropna()
    # 2026-09-10 Gate 4.10：microcap 阈值取完整形成月市场横截面 q10（与主复刻同口径），
    # 替换旧的研究子样本 quantile(0.10)。
    meq_q10 = meq.groupby('month')['me'].quantile(0.10).rename('me_q10_form')
    _ipo = pd.read_csv('data/processed/firm_id_map.csv', dtype=str)
    _ipo_map = dict(zip(_ipo['code6'].str.strip(), _ipo['ipo_date'].str.strip()))

    rows = []           # 单变量/双变量/十分位逐月
    cum_rows = []       # 累计收益：每月排序后持有 1..12 月
    n_month_used = 0
    files = sorted(EDGES.glob(f'edges_{NET_VER}_*.csv'))
    print(f"边表 {len(files)} 个", flush=True)
    for cf_path in files:
        cutoff = cf_path.stem.replace(f'edges_{NET_VER}_', '')
        sig = cutoff[:7]
        pred = _next_month(sig)
        if pred not in months or sig not in months:
            continue
        edges = pd.read_csv(cf_path, dtype=str, usecols=['focal_id', 'source_id', 'weight'])
        edges['weight'] = pd.to_numeric(edges['weight'], errors='coerce')
        edges = edges[edges['weight'].notna() & (edges['weight'] > 0)]
        r_sig = ret[ret['month'] == sig][['firm_id', 'R']].rename(columns={'R': 'r_sig'})
        r_pred = ret[ret['month'] == pred][['firm_id', 'R']].rename(columns={'R': 'r_pred'})
        m = edges.merge(r_sig, left_on='source_id', right_on='firm_id', how='inner')
        m = m.merge(r_pred, left_on='focal_id', right_on='firm_id', how='inner', suffixes=('', '_p'))
        if m.empty:
            continue
        m['wcf'] = m['r_sig'] * m['weight']
        snum = m.groupby('focal_id')['wcf'].sum()
        sden = m.groupby('focal_id')['weight'].sum()
        g = pd.DataFrame({'cf': (snum / sden).to_numpy()}, index=snum.index).reset_index()
        g.columns = ['firm_id', 'cf']
        g = g.merge(r_pred, on='firm_id', how='left')
        g = g.merge(r_sig.rename(columns={'r_sig': 'rev'}), on='firm_id', how='left')
        g['npm'] = g['cf'] - g['rev']
        g['month_pred'] = pred
        g = g.dropna(subset=['npm', 'r_pred'])
        # R5 主样本过滤
        g = g.merge(meq[meq['month'] == sig][['firm_id', 'me']].rename(columns={'me': 'me_form'}),
                    on='firm_id', how='left')
        _thr = float(meq_q10.get(sig, np.nan))
        mos = (pd.to_datetime(pred, errors='coerce')
               - pd.to_datetime(g['firm_id'].map(_ipo_map), errors='coerce')).dt.days / 30.4
        g = g[~((g['me_form'] < _thr).fillna(False) | (mos < 6).fillna(False))]
        if len(g) < 200:
            continue
        n_month_used += 1

        # 1) 十分位（Q10-Q1）
        try:
            g['q10'] = pd.qcut(g['npm'], 10, labels=False, duplicates='drop')
            d10 = g.groupby('q10')['r_pred'].mean()
            rows.append({'type': 'decile', 'month': pred, 'ls': d10.iloc[-1] - d10.iloc[0]})
        except ValueError:
            pass
        # 2) 双变量：按 rev（反转）3 组，组内 NPM 5 组
        try:
            g['qrev'] = pd.qcut(g['rev'], 3, labels=False, duplicates='drop')
            g['q5'] = g.groupby('qrev')['npm'].transform(
                lambda s: pd.qcut(s, 5, labels=False, duplicates='drop'))
            gd = g.dropna(subset=['q5'])
            ls_rev = gd.groupby('qrev').apply(
                lambda d: d[d['q5'] == d['q5'].max()]['r_pred'].mean() - d[d['q5'] == d['q5'].min()]['r_pred'].mean(),
                include_groups=False)
            rows.append({'type': 'bivar_rev', 'month': pred, 'ls': float(np.mean(ls_rev))})
        except (ValueError, TypeError):
            pass
        # 3) 双变量：按自身动量（r_pred 未知，用 npm 分层后控制 cf）——改用"按 cf 3 组，组内 npm 5 组"
        try:
            g['qcf'] = pd.qcut(g['cf'], 3, labels=False, duplicates='drop')
            g['q5cf'] = g.groupby('qcf')['npm'].transform(
                lambda s: pd.qcut(s, 5, labels=False, duplicates='drop'))
            gd2 = g.dropna(subset=['q5cf'])
            ls_cf = gd2.groupby('qcf').apply(
                lambda d: d[d['q5cf'] == d['q5cf'].max()]['r_pred'].mean() - d[d['q5cf'] == d['q5cf'].min()]['r_pred'].mean(),
                include_groups=False)
            rows.append({'type': 'bivar_cf', 'month': pred, 'ls': float(np.mean(ls_cf))})
        except (ValueError, TypeError):
            pass

        # 4) 累计收益：本月排序的 NPM 五分位，跟踪未来 1..12 月收益
        try:
            g['q5s'] = pd.qcut(g['npm'], 5, labels=False, duplicates='drop')
        except ValueError:
            continue
        top = g[g['q5s'] == g['q5s'].max()]['firm_id'].tolist()
        bot = g[g['q5s'] == g['q5s'].min()]['firm_id'].tolist()
        idx = months.index(pred)
        for h in range(1, MAX_HOLD + 1):
            if idx + h - 1 >= len(months):
                break
            hm = months[idx + h - 1]
            hret = ret_pivot[hm]
            t = hret.reindex(top).mean()
            b = hret.reindex(bot).mean()
            if t == t and b == b:
                cum_rows.append({'month': pred, 'hold': h, 'ls': float(t - b)})
        if n_month_used % 24 == 0:
            print(f"处理 {n_month_used} 个月（{pred}）", flush=True)

    # 汇总
    df = pd.DataFrame(rows)
    def summ(typ):
        d = df[df['type'] == typ]['ls']
        return {'mean_pct': round(float(d.mean()) * 100, 4),
                't_iid': round(float(np.mean(d) / (d.std(ddof=1) / np.sqrt(len(d)))), 2),
                'n_months': int(len(d))}
    cum = pd.DataFrame(cum_rows)
    cum_agg = cum.groupby('hold')['ls'].agg(['mean', 'count'])
    cum_curve = []
    running = 0.0
    for h in sorted(cum_agg.index):
        running += cum_agg.loc[h, 'mean']
        cum_curve.append({'hold': int(h), 'monthly_ls_pct': round(float(cum_agg.loc[h, 'mean']) * 100, 4),
                          'cumulative_pct': round(running * 100, 4), 'n_months': int(cum_agg.loc[h, 'count'])})
    res = {
        'note': '补做检验 P1：十分位/双变量分层（控制反转、控制 CF）/累计收益曲线（NPM 五分位多空，持有 1-12 月）',
        'n_months_used': n_month_used,
        'decile': summ('decile'),
        'bivariate_control_rev': summ('bivar_rev'),
        'bivariate_control_cf': summ('bivar_cf'),
        'cumulative_return_curve': cum_curve,
        'benchmark': {
            'ah2020': '双变量：控制行业/反转后 CF 仍预测；十分位 alpha 大于五分位；Figure 1 累计 12 月持续漂移',
            'jiang2023_fig1': '多空组合 12 个月累计超额 EW 4.76%、VW 7.26%',
        },
        'network_version': NET_VER,
        'returns_panel': str(RET),
    }
    attach_meta(res, __file__, NET_VER)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False)[:600], flush=True)


if __name__ == '__main__':
    main()
