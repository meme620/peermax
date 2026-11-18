# -*- coding: utf-8 -*-
"""S3A Step 9H：CF vs 行业动量吸收回归（对标 AH Table 6 / 蒋丹凌 2023 表 5）。

逻辑（对标蒋丹凌等 2023 表 5 列 1-3/11）：
  - 因变量：Ret_{i,t+1}（下月收益）
  - 解释变量：CF_{i,t}（连接公司加权收益，=AH 式 1）、IndRet_{i,t}（同行业其他公司等权收益）
  - FM 三列：(1) ~CF (2) ~IndRet (3) ~CF+IndRet，均含控制变量（size/BM/反转/动量/换手/流动性）
  - 判读：CF 加入后 IndRet 系数缩小或消失 → CF 动量吸收行业动量（独特性证据升级，对标蒋丹凌表 5：
    IndRet 0.951***→0.433** 缩小约一半；CFRet 系数保持 0.92-1.21***）

行业来源：Drive ind_code（30 类两位数，非申万二级，登记口径）；地区待数据源确认（本版仅行业）。
输出: reports/data_gates/G0/s3a_9h_absorb.json
用法: python scripts/analyst_industry_absorb.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

from analyst_cf_mom_replication import nw_t, _next_month, EDGES, RET, MEQ, NET_VER
from analyst_meta import attach_meta

DRIVE = Path('data/raw/drive')
OUT = Path('reports/data_gates/G0/s3a_9h_absorb.json')
OUT.parent.mkdir(parents=True, exist_ok=True)


def month_end(df):
    s = df.copy()
    s.index = pd.to_datetime(s.index)
    return s.resample('ME').last()


def load_ind():
    s = pickle.load(open(DRIVE / 'ind_code.pkl', 'rb'))
    # index '000001.SZ' -> 6 位码（与 ret 清洗后的 firm_id 对齐）
    firm = s.index.astype(str).str.split('.').str.get(0)
    s2 = pd.Series(s.values, index=firm, name='ind')
    return s2[~s2.index.duplicated(keep='first')]


def main():
    ind = load_ind()
    ret = pd.read_csv(RET, dtype=str, keep_default_na=False)
    ret['firm_id'] = ret['firm_id'].str.replace(r'^(sz|sh)', '', regex=True)  # 与 edges 6 位码对齐（同 cf_mom）
    ret = ret[ret['R'].notna()][['firm_id', 'month', 'R']]
    ret['R'] = pd.to_numeric(ret['R'], errors='coerce')
    ret = ret.dropna()
    months = sorted(ret['month'].unique())

    # 行业动量：同行业其他公司等权收益（总量法避免逐公司循环）
    ind2 = ind.rename_axis('firm_id').reset_index()
    ret_ind = ret.merge(ind2, on='firm_id', how='left')
    grp = ret_ind.groupby(['month', 'ind'])['R'].agg(['sum', 'count']).reset_index()
    # IndRet_i = (行业总收益 - 自己收益) / (n-1)
    ind_ret = ret_ind.merge(grp, on=['month', 'ind'], how='left')
    ind_ret['ind_ret'] = (ind_ret['sum'] - ind_ret['R']) / (ind_ret['count'] - 1).clip(lower=1)
    ind_ret = ind_ret[['firm_id', 'month', 'ind_ret']]
    print(f"ind_ret 覆盖: {ind_ret['ind_ret'].notna().mean():.2%} 行", flush=True)

    # Drive 控制变量（同 9C 口径）
    mv = month_end(pickle.load(open(DRIVE / 'daily_a_share_market_val_3_20260630.pkl', 'rb')))
    bm = month_end(pickle.load(open(DRIVE / 'daily_book_to_market_ratio_ttm_20260630.pkl', 'rb')))
    tr = month_end(pickle.load(open(DRIVE / 'daily_turnover_rate_20260630.pkl', 'rb'))) / 100.0
    am = month_end(pickle.load(open(DRIVE / 'daily_total_amount_20260630.pkl', 'rb')))

    def stack_drive(s, name):
        st = s.stack().rename(name).reset_index()
        st.columns = ['month', 'firm_id', name]  # 强制列名（不依赖 index 命名）
        st['firm_id'] = st['firm_id'].astype(str).str.split('.').str.get(0)  # order_book_id -> 6 位码
        st['month'] = pd.to_datetime(st['month']).dt.strftime('%Y-%m')
        return st

    mv_s = stack_drive(mv, 'mv_').copy()
    mv_s['month'] = pd.to_datetime(mv_s['month']).dt.strftime('%Y-%m')
    bm_s = stack_drive(bm, 'bm_t').copy()
    bm_s['month'] = pd.to_datetime(bm_s['month']).dt.strftime('%Y-%m')
    tr_s = stack_drive(tr, 'turn_t').copy()
    tr_s['month'] = pd.to_datetime(tr_s['month']).dt.strftime('%Y-%m')
    am_s = stack_drive(am, 'amt_t').copy()
    am_s['month'] = pd.to_datetime(am_s['month']).dt.strftime('%Y-%m')

    # 逐月构造面板：cf（连接公司收益）+ ind_ret + 控制
    rows = []
    mom_vals = {}
    for i in range(12, len(months) - 1):
        w = months[i - 12:i - 1]
        mom_vals[months[i]] = ret[ret['month'].isin(w)].groupby('firm_id')['R'].apply(
            lambda x: np.prod(1 + x) - 1)
    mom_df = pd.concat(mom_vals, names=['month', 'firm_id']).rename('mom').reset_index()

    for cf_path in sorted(EDGES.glob(f'edges_{NET_VER}_*.csv')):
        cutoff = cf_path.stem.replace(f'edges_{NET_VER}_', '')
        sig = cutoff[:7]
        pred = _next_month(sig)
        if pred not in months or sig not in months:
            continue
        edges = pd.read_csv(cf_path, dtype=str, usecols=['focal_id', 'source_id', 'weight'])
        edges['weight'] = pd.to_numeric(edges['weight'], errors='coerce')
        edges = edges[edges['weight'].notna() & (edges['weight'] > 0)]
        r_sig = ret[ret['month'] == sig][['firm_id', 'R']].rename(columns={'R': 'r_sig'})
        r_pred_df = ret[ret['month'] == pred][['firm_id', 'R']].rename(columns={'R': 'r_pred'})
        r_form_df = ret[ret['month'] == sig][['firm_id', 'R']].rename(columns={'R': 'r_form'})
        m = edges.merge(r_sig, left_on='source_id', right_on='firm_id', how='inner')
        m = m.merge(r_pred_df, left_on='focal_id', right_on='firm_id', how='inner')
        m['wcf'] = m['r_sig'] * m['weight']
        snum = m.groupby('focal_id', sort=False)['wcf'].sum()
        sden = m.groupby('focal_id', sort=False)['weight'].sum()
        sub = pd.DataFrame({'firm_id': snum.index, 'cf': (snum / sden).to_numpy()})
        sub['month'] = sig
        sub = sub.merge(r_pred_df, on='firm_id', how='left')
        sub = sub.merge(r_form_df, on='firm_id', how='left')
        sub = sub.merge(ind_ret[ind_ret['month'] == sig], on=['firm_id', 'month'], how='left')
        sub = sub.merge(mv_s[mv_s['month'] == sig], on=['firm_id', 'month'], how='left')
        sub = sub.merge(bm_s[bm_s['month'] == sig], on=['firm_id', 'month'], how='left')
        sub = sub.merge(tr_s[tr_s['month'] == sig], on=['firm_id', 'month'], how='left')
        sub = sub.merge(am_s[am_s['month'] == sig], on=['firm_id', 'month'], how='left')
        sub = sub.merge(mom_df[mom_df['month'] == sig], on=['firm_id', 'month'], how='left')
        sub['size'] = np.log(sub['mv_'])
        sub['illiq'] = sub['r_form'].abs() / sub['amt_t'].replace(0, np.nan)  # 2026-09-01 R6：统一 Amihud |R_form|/amt（原 amt/mv 为换手/规模代理且与 turn_t 共线）
        sub['month_pred'] = pred
        rows.append(sub[['firm_id', 'month', 'month_pred', 'r_pred', 'cf', 'ind_ret', 'r_form',
                         'size', 'bm_t', 'mom', 'turn_t', 'illiq']])  # 2026-09-01：保留形成月 month（R5 过滤需要）
    panel = pd.concat(rows, ignore_index=True)
    print(f"panel: {len(panel)} 行 | 月份 {panel['month_pred'].nunique()}", flush=True)

    # --- 2026-09-01 R5（review 总账）：对齐主复刻样本过滤（microcap 形成期最低 10% + 上市<6 月）---
    meq = pd.read_csv(MEQ, dtype=str, keep_default_na=False)
    meq['month'] = pd.to_datetime(meq['date']).dt.strftime('%Y-%m')
    meq['me'] = pd.to_numeric(meq['market_equity'], errors='coerce')
    meq = meq[['firm_id', 'month', 'me']].dropna()
    # 2026-09-10 Gate 4.9：microcap 阈值取**完整形成月市场横截面 q10**（与主复刻/CF-NPM 同口径），
    # 不再用研究子样本分位（旧口径会把阈值绑定到研究样本边界）。
    meq_q10 = meq.groupby('month')['me'].quantile(0.10).rename('me_q10_form')
    _ipo = pd.read_csv('data/processed/firm_id_map.csv', dtype=str)
    _ipo_map = dict(zip(_ipo['code6'].str.strip(), _ipo['ipo_date'].str.strip()))
    panel = panel.merge(meq.rename(columns={'me': 'me_form'}), on=['firm_id', 'month'], how='left')
    panel['months_since_ipo'] = (pd.to_datetime(panel['month_pred'], errors='coerce')
                                 - pd.to_datetime(panel['firm_id'].map(_ipo_map), errors='coerce')).dt.days / 30.4
    _micro = (panel['me_form'] < panel['month'].map(meq_q10)).fillna(False)
    _young = (panel['months_since_ipo'] < 6).fillna(False)
    _n0 = len(panel)
    panel = panel[~(_micro | _young)]
    print(f"主样本过滤（R5）：{_n0} -> {len(panel)} 行", flush=True)

    def fm(x_names, y='r_pred'):
        coefs = {x: [] for x in x_names}
        t_iid = {x: [] for x in x_names}
        months_used = []
        for mp, sub in panel.groupby('month_pred'):
            d = sub[[y] + x_names].replace([np.inf, -np.inf], np.nan).dropna()
            if len(d) < 100:
                continue
            X = np.column_stack([np.ones(len(d))] + [d[x].astype(float) for x in x_names])
            yy = d[y].astype(float).to_numpy()
            try:
                beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
            except np.linalg.LinAlgError:
                continue
            resid = yy - X @ beta
            s2 = resid @ resid / max(len(yy) - X.shape[1], 1)
            cov = s2 * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(cov))
            for k, x in enumerate(x_names):
                coefs[x].append(beta[k + 1])
                t_iid[x].append(beta[k + 1] / se[k + 1])
            months_used.append(mp)
        out = {}
        for x in x_names:
            if not coefs[x]:
                out[x] = None
                continue
            a = np.array(coefs[x])
            # 2026-09-01 R1：t_iid=FM 时序 mean/SE（同 9C）；旧"单月 t 均值"保留为参考
            c = float(np.mean(coefs[x]))
            t = float(c / (a.std(ddof=1) / np.sqrt(len(a)))) if len(a) > 2 else float('nan')
            out[x] = {'beta': round(c, 4), 't_iid': round(t, 2),
                      't_mean_monthly_ref': round(float(np.mean(t_iid[x])), 2),
                      't_nw1': round(float(nw_t(np.array(coefs[x]))), 2)}
        out['n_months'] = len(months_used)
        out['sample'] = f"{months_used[0]}~{months_used[-1]}" if months_used else None
        return out

    ctrl = ['size', 'bm_t', 'r_form', 'mom', 'turn_t', 'illiq']
    col1 = fm(['cf'] + ctrl)
    col2 = fm(['ind_ret'] + ctrl)
    col3 = fm(['cf', 'ind_ret'] + ctrl)
    res = {
        'note': 'FM 回归：Ret_{t+1} ~ CF/IndRet + 控制(size/BM/反转/动量/换手/流动性)；对标蒋丹凌 2023 表 5 列(1)(2)(3)',
        'col1_cf_only': col1,
        'col2_ind_only': col2,
        'col3_cf_ind': col3,
        'benchmark': {
            'jiang2023_table5': 'CFRet(t-1)=1.146*** (t=5.06)；IndRet(t-1)=0.951*** (t=3.93)；同入后 CFRet 0.92-1.21*** 不变、IndRet 缩小至 0.433**（约一半）→ CF 吸收行业动量',
            'criterion': '同入后 cf 系数保持显著 + ind_ret 系数明显缩小 = 独特性成立',
        },
        'industry_source': 'Drive ind_code（30 类两位数，非申万二级）',
        'network_version': NET_VER,
    }
    attach_meta(res, __file__, NET_VER)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False)[:900], flush=True)


if __name__ == '__main__':
    main()
