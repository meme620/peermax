# -*- coding: utf-8 -*-
"""S3A Step 9G：CFFR 盈余预测修正滞后检验（对标 AH Table 9 / 蒋丹凌 2023 表 9）。

逻辑（对标蒋丹凌等 2023 表 9）：
  - consensus_{i,t} = 焦点公司 i 在月末 t 的分析师 FY1 EPS 一致预测（当月及之前已知预测的均值，月末截点）
  - FR_{i,t} = (consensus_t - consensus_{t-1}) / max(|consensus_{t-1}|, 0.01)
  - CFFR_{i,t} = Σ_j w_ij,t · FR_{j,t}（网络权重加权，w 来自 v3 边表）
  - FM 回归三列：FR_t ~ CFFR_t + FR_{t-1} | FR_t ~ CFFR_{t-1} + FR_{t-1} | FR_t ~ CFFR_{t-2} + FR_{t-1}
  - 判读：当期 CFFR_t 显著正 + 滞后 1 月 CFFR_{t-1} 仍显著正 + 滞后 2 月不显著
    → 分析师对关联公司信息迟缓反应约 1 个月（有限注意力机制证据）

输出: reports/data_gates/G0/s3a_9g_cffr.json
用法: python scripts/data_audit/s3a_9g_cffr.py
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from s3a_cf_mom_replication import nw_t, _next_month, EDGES, RET, NET_VER
from s3a_meta import attach_meta

SRC = 'data/processed/analyst/analyst_forecast_clean.csv'
OUT = Path('reports/data_gates/G0/s3a_9g_cffr.json')
OUT.parent.mkdir(parents=True, exist_ok=True)
EPS_MIN = 0.01  # 分母下界（蒋丹凌表 9 口径）


def build_consensus():
    """FY1 月末一致预测（firm × month）。2026-09-01 R2：与主口径 step5 同源清洗——
    mapping_status==ok + A_code 合法格式 ^A\\d{5}$（原实现 usecols 漏 mapping_status，
    过滤条件恒 False 静默失效，且坏分析师码未剔除，review 总账 R2）。"""
    df = pd.read_csv(SRC, dtype=str, keep_default_na=False,
                     usecols=['firm_id', 'Anndats', 'Fpenddt', 'analyst_id', 'feps', 'mapping_status'])
    df = df[df['mapping_status'] == 'ok']
    n_bad = int((~df['analyst_id'].str.match(r'^A\d{5}$', na=False)).sum())
    df = df[df['analyst_id'].str.match(r'^A\d{5}$', na=False)]
    print(f"consensus: bad A_code rows dropped: {n_bad}", flush=True)
    df['report_date'] = pd.to_datetime(df['Anndats'], errors='coerce')
    df['fpend'] = pd.to_datetime(df['Fpenddt'], errors='coerce')
    # FY1：gap = 0
    df['gap'] = df['fpend'].dt.year - df['report_date'].dt.year
    df = df[df['gap'] == 0]
    df['month'] = df['report_date'].dt.strftime('%Y-%m')
    df['feps'] = pd.to_numeric(df['feps'], errors='coerce')
    df = df[df['feps'].notna()].dropna(subset=['month'])
    # 同月同分析师多条取最后一条（该月最新）
    df = df.sort_values(['firm_id', 'analyst_id', 'month', 'report_date'])
    df = df.drop_duplicates(['firm_id', 'analyst_id', 'month'], keep='last')
    # firm × month 均值（当月已知预测）
    cons = df.groupby(['firm_id', 'month'])['feps'].mean().reset_index()
    cons = cons.rename(columns={'feps': 'consensus'})
    return cons


def fm_one(panel, x_name, y='fr', controls=('fr_lag',)):
    """单列 FM 回归：每月横截面，系数为时间序列均值，NW1 校正。"""
    xs = [x_name] + list(controls)
    coefs, t_iid, t_nw = [], [], []
    months = []
    for mp, sub in panel.groupby('month'):
        d = sub[[y] + xs].replace([np.inf, -np.inf], np.nan).dropna()
        if len(d) < 50 or y not in d.columns:
            continue
        X = d[xs].astype(float)
        X = np.column_stack([np.ones(len(X)), X])
        yy = d[y].astype(float).to_numpy()
        try:
            beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid = yy - X @ beta
        s2 = resid @ resid / max(len(yy) - X.shape[1], 1)
        cov = s2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        coefs.append(beta[1])
        t_iid.append(beta[1] / se[1])
        months.append(mp)
    if not coefs:
        return None
    c = float(np.mean(coefs))
    a = np.array(coefs)
    # 2026-09-01 R1：t_iid=FM 时序标准定义（逐月 β 序列 mean/SE，同 9C）；
    # 原实现"t_iid=单月 t 均值"不是显著性统计量，曾把显著项误标为不显著。
    t_i = float(c / (a.std(ddof=1) / np.sqrt(len(a)))) if len(a) > 2 else float('nan')
    t_m = float(np.mean(t_iid))            # 旧口径保留为参考（单月 t 均值）
    t_n = float(nw_t(np.array(coefs)))
    return {'beta': round(c, 4), 't_iid': round(t_i, 2), 't_nw1': round(t_n, 2),
            't_mean_monthly_ref': round(t_m, 2),
            'n_months': len(months), 'sample': f"{months[0]}~{months[-1]}"}


def main():
    cons = build_consensus()
    # firm × 完整自然月面板 → FR；先保留缺失月份，再按日历位置构造 lag。
    # 不得在 lag 前删除缺失月份，否则 shift(1) 会变成上一条非缺失记录。
    all_months = pd.date_range(cons['month'].min(), cons['month'].max(), freq='MS').strftime('%Y-%m').tolist()
    firms = pd.Index(cons['firm_id'].unique(), name='firm_id')
    full_idx = pd.MultiIndex.from_product([firms, all_months], names=['firm_id', 'month'])
    cons_full = (cons.set_index(['firm_id', 'month'])['consensus']
                    .reindex(full_idx).unstack('month'))
    cons_ff = cons_full.ffill(axis=1)
    prev = cons_ff.shift(1, axis=1)
    fr_full = (cons_ff - prev) / prev.abs().clip(lower=EPS_MIN)
    fr = fr_full.stack(dropna=False).rename('fr').reset_index()
    fr['fr_lag'] = fr.groupby('firm_id')['fr'].shift(1)
    fr['fr_lag2'] = fr.groupby('firm_id')['fr'].shift(2)
    fr['calendar_gap_lag1'] = fr.groupby('firm_id')['month'].transform(
        lambda s: pd.to_datetime(s).diff().dt.days)
    panel = fr

    # CFFR：逐月读 v3 边表，CFFR_{i,t} = Σ w_ij · FR_{j,t}
    months = sorted(panel['month'].unique())
    cffr_rows = []
    files = sorted(EDGES.glob(f'edges_{NET_VER}_*.csv'))
    print(f"edge files: {len(files)}", flush=True)
    for cf_path in files:
        cutoff = cf_path.stem.replace(f'edges_{NET_VER}_', '')
        sig = cutoff[:7]
        if sig not in months:
            continue
        edges = pd.read_csv(cf_path, dtype=str, usecols=['focal_id', 'source_id', 'weight'])
        edges['weight'] = pd.to_numeric(edges['weight'], errors='coerce')
        edges = edges[edges['weight'].notna() & (edges['weight'] > 0)]
        fr_sig = (panel[panel['month'] == sig][['firm_id', 'fr']]
                  .dropna(subset=['fr'])
                  .rename(columns={'fr': 'fr_j'}))
        m = edges.merge(fr_sig, left_on='source_id', right_on='firm_id', how='inner')
        g = m.groupby('focal_id', sort=False).apply(
            lambda d: np.average(d['fr_j'], weights=d['weight']), include_groups=False)
        cffr_rows.append(pd.DataFrame({'firm_id': g.index, 'month': sig, 'cffr': g.values}))
    cffr = pd.concat(cffr_rows, ignore_index=True)
    cffr_full = (cffr.set_index(['firm_id', 'month'])['cffr']
                 .reindex(full_idx).rename('cffr').reset_index())
    panel = panel.drop(columns=['calendar_gap_lag1']).merge(
        cffr_full, on=['firm_id', 'month'], how='left')
    panel['cffr_lag'] = panel.groupby('firm_id')['cffr'].shift(1)
    panel['cffr_lag2'] = panel.groupby('firm_id')['cffr'].shift(2)

    col1 = fm_one(panel, 'cffr')
    col2 = fm_one(panel, 'cffr_lag')
    col3 = fm_one(panel, 'cffr_lag2')
    res = {
        'note': ('FR_t ~ CFFR(当期/滞后1/滞后2) + FR_{t-1}；FM 回归；consensus=月末 FY1 一致预测均值(ffill)；'
                 '2026-09-01 修订：共识按 step5 同源清洗（mapping_ok+合法码）；t_iid=FM 时序 mean/SE（旧"单月 t 均值"降级为 t_mean_monthly_ref）；'
                 'FR 口径=FY1 财年共识（蒋=未来 12 个月一致预测，预测期限不完全同构，差异已声明）'),
        'col1_cffr_t': col1,
        'col2_cffr_lag1': col2,
        'col3_cffr_lag2': col3,
        'benchmark': {
            'jiang2023_table9': 'CFFR_t=0.206*** (t=7.99)；CFFR_{t-1}=0.061*** (t=2.63)；CFFR_{t-2}=0.034 不显著 (t=1.43) → 迟缓约 1 个月',
            'criterion': '当期显著正 + 滞后1 仍显著 + 滞后2 不显著 = 分析师迟缓反应机制成立（蒋表 9 判据）',
        },
        'network_version': NET_VER,
    }
    attach_meta(res, __file__, NET_VER)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False)[:800], flush=True)


if __name__ == '__main__':
    main()
