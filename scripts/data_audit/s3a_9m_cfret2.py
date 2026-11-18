# -*- coding: utf-8 -*-
"""S3A 9M：CFRET2 二阶网络动量的正式规格（对标 AH Table 8 完整版；review R20，2026-09-01）。

旧版只有 hi-lo 中位分组诊断（cf_mom_replication.py 内嵌）；AH Table 8 的完整规格=
逐月横截面 FM：r_{t+1} ~ CFRET2 + CF + controls，要求**控制 CF 后 CFRET2 仍显著**
（信息沿"同行的同行"二阶传播）。样本过滤同主复刻（R5）。
输出: reports/data_gates/G0/s3a_9m_cfret2.json
用法: python scripts/data_audit/s3a_9m_cfret2.py
"""
import json, sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from s3a_cf_mom_replication import nw_t, _next_month, EDGES, RET, MEQ, NET_VER
from s3a_meta import attach_meta

# 与主复刻一致：网络版本和输入由统一 spec 决定
DRIVE = Path('data/raw/drive')


def month_end(df):
    m = df.copy(); m.index = pd.to_datetime(m.index)
    return m.resample('ME').last()


def stack(s, name):
    st = s.stack().rename(name).reset_index()
    st.columns = ['month', 'firm_id', name]
    st['firm_id'] = st['firm_id'].astype(str).str.split('.').str.get(0)
    st['month'] = pd.to_datetime(st['month']).dt.strftime('%Y-%m')
    return st


def main():
    ret = pd.read_csv(RET, dtype=str, keep_default_na=False)
    ret['firm_id'] = ret['firm_id'].str.replace(r'^(sz|sh)', '', regex=True)
    ret['R'] = pd.to_numeric(ret['R'], errors='coerce')
    ret = ret[ret['R'].notna()][['firm_id', 'month', 'R']]
    months = sorted(ret['month'].unique())
    meq = pd.read_csv(MEQ, dtype=str, keep_default_na=False)
    meq['month'] = pd.to_datetime(meq['date']).dt.strftime('%Y-%m')
    meq['me'] = pd.to_numeric(meq['market_equity'], errors='coerce')
    meq = meq[['firm_id', 'month', 'me']].dropna()
    _ipo = pd.read_csv('data/processed/firm_id_map.csv', dtype=str)
    _ipo_map = dict(zip(_ipo['code6'].str.strip(), _ipo['ipo_date'].str.strip()))
    mv = stack(month_end(pickle.load(open(DRIVE / 'daily_a_share_market_val_3_20260630.pkl', 'rb'))), 'mv')
    bm = stack(month_end(pickle.load(open(DRIVE / 'daily_book_to_market_ratio_ttm_20260630.pkl', 'rb'))), 'bm')
    tr = stack(month_end(pickle.load(open(DRIVE / 'daily_turnover_rate_20260630.pkl', 'rb'))), 'turn')
    # 2026-09-10 Gate 4.14：完整形成月市场横截面 q10（与主复刻同口径）
    q10_full = meq.groupby('month')['me'].quantile(0.10)
    rows = []
    cov_rows = []
    for cf_path in sorted(EDGES.glob(f'edges_{NET_VER}_*.csv')):
        cutoff = cf_path.stem.replace(f'edges_{NET_VER}_', '')
        sig = cutoff[:7]; pred = _next_month(sig)
        if pred not in months or sig not in months:
            continue
        e = pd.read_csv(cf_path, dtype=str, usecols=['focal_id', 'source_id', 'weight'])
        e['weight'] = pd.to_numeric(e['weight'], errors='coerce')
        e = e[e['weight'].notna() & (e['weight'] > 0)]
        r_sig = ret[ret['month'] == sig][['firm_id', 'R']].rename(columns={'R': 'r_sig'})
        r_pred = ret[ret['month'] == pred][['firm_id', 'R']].rename(columns={'R': 'r_pred'})
        m = e.merge(r_sig, left_on='source_id', right_on='firm_id', how='inner')
        num = m.assign(wr=m['weight'] * m['r_sig']).groupby('focal_id')['wr'].sum()
        den = m.groupby('focal_id')['weight'].sum()
        cf = (num / den).rename('cf').reset_index().rename(columns={'focal_id': 'firm_id'})
        # CFRET2 = Σ_j w_ij·CF_j
        m2 = e.merge(cf.rename(columns={'firm_id': 'source_id', 'cf': 'cf_j'}), on='source_id', how='inner')
        n2 = m2.assign(wr=m2['weight'] * m2['cf_j']).groupby('focal_id')['wr'].sum()
        den2 = m2.groupby('focal_id')['weight'].sum()
        cfret2 = (n2 / den2).rename('cfret2').reset_index().rename(columns={'focal_id': 'firm_id'})
        coverage2 = (den2 / den).rename('second_order_weight_coverage').reset_index().rename(columns={'focal_id': 'firm_id'})
        share_total = den2.rename('second_order_share_of_total').reset_index().rename(columns={'focal_id': 'firm_id'})
        g = (cf.merge(cfret2, on='firm_id')
               .merge(coverage2, on='firm_id', how='left')
               .merge(share_total, on='firm_id', how='left')
               .merge(r_pred, on='firm_id', how='inner'))
        g = g.merge(meq[meq['month'] == sig][['firm_id', 'me']].rename(columns={'me': 'me_form'}), on='firm_id', how='left')
        g = g.merge(mv[mv['month'] == sig][['firm_id', 'mv']], on='firm_id', how='left')
        g = g.merge(bm[bm['month'] == sig][['firm_id', 'bm']], on='firm_id', how='left')
        g = g.merge(tr[tr['month'] == sig][['firm_id', 'turn']], on='firm_id', how='left')
        g = g.merge(r_sig.rename(columns={'r_sig': 'r_form'}), on='firm_id', how='left')
        g['month_pred'] = pred
        # 主样本过滤：microcap 阈值取**完整形成月市场横截面 q10**（2026-09-10 Gate 4.14 修正；
        # 旧版用研究子样本 quantile(0.10) 会把阈值绑定到研究样本边界）
        thr = float(q10_full.get(sig, np.nan))
        mos = (pd.to_datetime(pred, errors='coerce') - pd.to_datetime(g['firm_id'].map(_ipo_map), errors='coerce')).dt.days / 30.4
        g = g[~((g['me_form'] < thr).fillna(False) | (mos < 6).fillna(False))]
        g = g.dropna(subset=['cf', 'cfret2', 'r_pred', 'mv', 'bm', 'turn', 'r_form'])
        if len(g) < 100:
            continue
        cov_rows.append({'month': pred,
                         'ratio_vs_first_order_available': float(g['second_order_weight_coverage'].mean()),
                         'min_ratio': float(g['second_order_weight_coverage'].min()),
                         'max_ratio': float(g['second_order_weight_coverage'].max()),
                         'share_of_total_weight': float(g['second_order_share_of_total'].mean()),
                         'n': int(len(g))})
        d = g[['r_pred', 'cfret2', 'cf', 'mv', 'bm', 'r_form', 'turn', 'second_order_weight_coverage']].astype(float)
        X = np.column_stack([np.ones(len(d)), d['cfret2'], d['cf'], np.log(d['mv']), d['bm'], d['r_form'], d['turn']])
        y = d['r_pred'].to_numpy()
        fin = np.isfinite(X).all(1) & np.isfinite(y)
        X, y = X[fin], y[fin]
        if len(y) < 100:
            continue
        try:
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        rows.append({'month': pred, 'cfret2': b[1], 'cf': b[2]})
    df = pd.DataFrame(rows)
    out = {}
    for k in ['cfret2', 'cf']:
        v = df[k].to_numpy()
        out[k] = {'beta': round(float(v.mean()), 5),
                  't_iid': round(float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))), 2),
                  't_nw1': round(float(nw_t(v)), 2)}
    out['n_months'] = int(len(df))
    out['sample'] = f"{df['month'].min()}~{df['month'].max()}"
    cov = pd.DataFrame(cov_rows)
    out['second_order_weight_coverage'] = {
        'n_months': int(len(cov)),
        'mean_ratio_vs_first_order_available': round(float(cov['ratio_vs_first_order_available'].mean()), 6) if len(cov) else None,
        'min_ratio': round(float(cov['min_ratio'].min()), 6) if len(cov) else None,
        'max_ratio': round(float(cov['max_ratio'].max()), 6) if len(cov) else None,
        'mean_share_of_total_weight': round(float(cov['share_of_total_weight'].mean()), 6) if len(cov) else None,
        'monthly': cov.to_dict(orient='records'),
        'definition': ('ratio = Σ二阶可得 source 权重 / Σ一阶可得 source 权重（两个可得集合不同，可 >1）；'
                       'share_of_total_weight = Σ二阶可得 source 权重（行归一化下即 focal 网络权重占比，恒 <=1）'),
    }
    out['note'] = ('AH Table 8 完整规格（R20）：r~CFRET2+CF+控制（logmv/bm/r_form/turn），主样本过滤=R5。'
                   '判读：控制一阶 CF 后 CFRET2 仍显著=二阶传播独立信息（对照 [AH p.30 式3] Table 8 与 cf_mom hi-lo 简化诊断 0.45%/t2.50）。')
    out['benchmark'] = {'ah2020_table8': 'CF RET2（二阶）在控制 CF RET 后仍显著', 'hi_lo_simplified': 'cf_mom_replication.json cfret2_diagnostic'}
    attach_meta(out, __file__, NET_VER)
    Path('reports/data_gates/G0/s3a_9m_cfret2.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False)[:500])


if __name__ == '__main__':
    main()
