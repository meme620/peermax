# -*- coding: utf-8 -*-
"""analyst-network 补做检验 B（对标 AH/蒋丹凌）：
A. n_ij ~ 基本面相似度（蒋表 3 简化版）：同行业 + |Δsize| + |Δbm| + |Δgrowth| 的 FM 回归
B. 关联复杂性交互（蒋表 10）：Ret_{t+1} ~ CF + NetworkDegree×CF + NetworkDegree + 控制
C. 小盘连接→大盘焦点（AH p.17）：焦点 csi300 内，按连接公司市值分小/大两组 CF
D. 基本面与收益（AH Table 12）：focal 收益 ~ peer 加权营收增速 + 控制

输入: v3 边表 + 全期面板 + Drive（size/bm/turn/amt/ind_code/财务）
输出: reports/data_gates/G0/analyst_extra_tests.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from dates import next_month as _next_month
from analyst_cf_mom_replication import EDGES, RET, NET_VER
from statsutil import nw_t
from analyst_meta import attach_meta
from fileio import read_pickle

DRIVE = Path('data/raw/drive')
OUT = Path('reports/data_gates/G0/analyst_extra_tests.json')
OUT.parent.mkdir(parents=True, exist_ok=True)


def month_end(df):
    s = df.copy()
    s.index = pd.to_datetime(s.index)
    return s.resample('ME').last()


def load_controls():
    mv = month_end(read_pickle(DRIVE / 'daily_a_share_market_val_3_20260630.pkl'))
    bm = month_end(read_pickle(DRIVE / 'daily_book_to_market_ratio_ttm_20260630.pkl'))
    tr = month_end(read_pickle(DRIVE / 'daily_turnover_rate_20260630.pkl')) / 100.0
    am = month_end(read_pickle(DRIVE / 'daily_total_amount_20260630.pkl'))

    def stack_drive(s, name):
        st = s.stack().rename(name).reset_index()
        st.columns = ['month', 'firm_id', name]
        st['firm_id'] = st['firm_id'].astype(str).str.split('.').str.get(0)
        st['month'] = pd.to_datetime(st['month']).dt.strftime('%Y-%m')
        return st

    c = stack_drive(mv, 'mv')
    c = c.merge(stack_drive(bm, 'bm'), on=['firm_id', 'month'], how='left')
    c = c.merge(stack_drive(tr, 'turn'), on=['firm_id', 'month'], how='left')
    c = c.merge(stack_drive(am, 'amt'), on=['firm_id', 'month'], how='left')
    c['size'] = np.log(pd.to_numeric(c['mv'], errors='coerce'))
    c['amt_mv'] = pd.to_numeric(c['amt'], errors='coerce') / pd.to_numeric(c['mv'], errors='coerce')  # 换手/规模代理（2026-09-01 R6 更名：原名 illiq 非 Amihud；本脚本回归实际未使用该列，仅保留备查）
    c['mv_q10'] = c.groupby('month')['mv'].transform(lambda s: pd.to_numeric(s, errors='coerce').quantile(0.10))
    return c


def main():
    ret = pd.read_csv(RET, dtype=str, keep_default_na=False)
    ret['firm_id'] = ret['firm_id'].str.replace(r'^(sz|sh)', '', regex=True)
    ret['R'] = pd.to_numeric(ret['R'], errors='coerce')
    ret = ret[ret['R'].notna()][['firm_id', 'month', 'R']]
    months = sorted(ret['month'].unique())
    ctrl = load_controls()
    # 2026-09-01 R5：focal 面板（B/C/D）对齐主复刻样本过滤（microcap 形成期最低 10% + 上市<6 月）；
    # A 段（n_ij~相似度）是配对级网络诊断、非收益预测检验，保持不过滤并在输出 note 声明。
    _ipo = pd.read_csv('data/processed/firm_id_map.csv', dtype=str)
    _ipo_map = dict(zip(_ipo['code6'].str.strip(), _ipo['ipo_date'].str.strip()))

    ind = read_pickle(DRIVE / 'ind_code.pkl')
    ind.index = ind.index.astype(str).str.split('.').str.get(0)
    ind = ind[~ind.index.duplicated(keep='first')]
    ind = ind.rename('ind')

    # 财务（营收同比，用于相似度与基本面-收益）
    fin = pd.read_csv('data/processed/analyst/financial_panel.csv', dtype=str,
                      usecols=['firm_id', 'notice_date', 'TOTALOPERATEREVETZ'])
    fin['notice_date'] = pd.to_datetime(fin['notice_date'], errors='coerce')
    fin['grow'] = pd.to_numeric(fin['TOTALOPERATEREVETZ'], errors='coerce')
    fin = fin.dropna(subset=['notice_date', 'grow'])
    fin = fin.sort_values('notice_date')

    files = sorted(EDGES.glob(f'edges_{NET_VER}_*.csv'))
    rows_a, rows_b, rows_c, rows_d = [], [], [], []
    n_rank_deficient = 0        # 奇异设计矩阵（lstsq 抛 LinAlgError）被跳过的月-规格数
    share_unknown_mean = []
    c300 = read_pickle(DRIVE / 'daily_csi300_components.pkl')
    c300.columns = [c.split('.')[0] for c in c300.columns]
    c300m = c300.resample('ME').last().stack().reset_index()
    c300m.columns = ['date', 'firm_id', 'in300']
    c300m['month'] = pd.to_datetime(c300m['date']).dt.strftime('%Y-%m')
    c300m = c300m[c300m['in300'] > 0][['firm_id', 'month']]

    n_done = 0
    for cf_path in files:
        cutoff = cf_path.stem.replace(f'edges_{NET_VER}_', '')
        sig, pred = cutoff[:7], _next_month(cutoff[:7])
        if pred not in months or sig not in months:
            continue
        edges = pd.read_csv(cf_path, dtype=str, usecols=['focal_id', 'source_id', 'weight', 'common_analyst_count'])
        edges['weight'] = pd.to_numeric(edges['weight'], errors='coerce')
        edges['cnt'] = pd.to_numeric(edges['common_analyst_count'], errors='coerce').fillna(1)
        edges = edges[edges['weight'].notna() & (edges['weight'] > 0)]
        if edges.empty:
            continue
        r_sig = ret[ret['month'] == sig][['firm_id', 'R']].rename(columns={'R': 'r_sig'})
        r_pred = ret[ret['month'] == pred][['firm_id', 'R']].rename(columns={'R': 'r_pred'})
        c_sig = ctrl[ctrl['month'] == sig]

        # ---- A: n_ij ~ 相似度（抽样每月全期 FM）----
        ea = edges.merge(c_sig[['firm_id', 'size', 'bm']], left_on='source_id', right_on='firm_id', how='left',
                         suffixes=('', '_s'))
        ea = ea.merge(c_sig[['firm_id', 'size', 'bm']], left_on='focal_id', right_on='firm_id', how='left',
                      suffixes=('_s', '_f'))
        ea = ea.merge(ind.rename('ind_s'), left_on='source_id', right_index=True, how='left')
        ea = ea.merge(ind.rename('ind_f'), left_on='focal_id', right_index=True, how='left')
        ea['same_ind'] = (ea['ind_s'] == ea['ind_f']).astype(float)
        ea['d_size'] = (ea['size_s'] - ea['size_f']).abs()
        ea['d_bm'] = (ea['bm_s'] - ea['bm_f']).abs()
        ea = ea.dropna(subset=['d_size', 'd_bm'])
        if len(ea) > 200:
            ea['ln_cnt'] = np.log1p(ea['cnt'])
            X = np.column_stack([np.ones(len(ea)), ea['same_ind'], -ea['d_size'], -ea['d_bm']])
            yy = ea['ln_cnt'].to_numpy()
            try:
                beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
                rows_a.append({'month': sig, 'beta_ind': beta[1], 'beta_size': beta[2], 'beta_bm': beta[3]})
            except np.linalg.LinAlgError:
                n_rank_deficient += 1

        # ---- B/C/D: focal 面板 ----
        m = edges.merge(r_sig, left_on='source_id', right_on='firm_id', how='inner')
        m = m.merge(r_pred, left_on='focal_id', right_on='firm_id', how='inner', suffixes=('', '_p'))
        if m.empty:
            continue
        m['wcf'] = m['r_sig'] * m['weight']
        snum = m.groupby('focal_id')['wcf'].sum()
        sden = m.groupby('focal_id')['weight'].sum()
        g = pd.DataFrame({'cf': (snum / sden).to_numpy(), 'deg': snum.index.map(
            m.groupby('focal_id')['weight'].count())}, index=snum.index).reset_index()
        g.columns = ['firm_id', 'cf', 'deg']
        g = g.merge(r_pred, on='firm_id', how='left')
        g = g.merge(r_sig.rename(columns={'r_sig': 'rev'}), on='firm_id', how='left')
        g['month_pred'] = pred
        g = g.merge(c_sig[['firm_id', 'size', 'bm', 'turn', 'mv', 'mv_q10']], on='firm_id', how='left')  # 2026-09-06/09-10：形成月完整市值横截面 q10（mv 必须一并合并，否则过滤抛 KeyError）
        g = g.merge(c300m[c300m['month'] == sig], on='firm_id', how='left')
        # 基本面（peer 加权营收增速）：focal 最近公告增速
        fsub = fin[fin['notice_date'] <= cutoff].sort_values('notice_date')
        latest = fsub.drop_duplicates('firm_id', keep='last')[['firm_id', 'grow']]
        merged_edges = edges.merge(latest, left_on='source_id', right_on='firm_id', how='inner')
        merged_edges['wgr'] = merged_edges['grow'] * merged_edges['weight']
        gsum = merged_edges.groupby('focal_id')['wgr'].sum()
        gden = merged_edges.groupby('focal_id')['weight'].sum()
        peer_g = pd.DataFrame({'firm_id': gsum.index, 'peer_grow': (gsum / gden).to_numpy()})
        g = g.merge(peer_g, on='firm_id', how='left')
        g = g.merge(latest.rename(columns={'grow': 'focal_grow'}), on='firm_id', how='left')
        g = g.dropna(subset=['cf', 'r_pred'])
        # 2026-09-01 R5：focal 面板套主复刻同款过滤（形成月 Drive 市值最低 10% microcap + 上市<6 月）
        _mos = (pd.to_datetime(pred, errors='coerce')
                - pd.to_datetime(g['firm_id'].map(_ipo_map), errors='coerce')).dt.days / 30.4
        _drop = ((g['mv'] < g['mv_q10']).fillna(False) | (_mos < 6).fillna(False))
        g = g[~_drop]
        if len(g) < 200:
            continue

        # B: 关联复杂性交互
        gb = g.dropna(subset=['size', 'bm', 'turn', 'deg'])
        if len(gb) > 200:
            Xb = np.column_stack([np.ones(len(gb)), gb['cf'], gb['deg'] * gb['cf'], gb['deg'],
                                  gb['size'], gb['bm'], gb['rev'], gb['turn']])
            yb = gb['r_pred'].to_numpy()
            try:
                bb, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
                rows_b.append({'month': pred, 'beta_cf': bb[1], 'beta_cfxdeg': bb[2], 'beta_deg': bb[3]})
            except np.linalg.LinAlgError:
                n_rank_deficient += 1

        # C: 小盘连接→大盘焦点（焦点 csi300，连接公司按完整形成月市值分组）
        gc = g[g['firm_id'].isin(c300m[c300m['month'] == sig]['firm_id'])]
        if len(gc) > 100:
            mc = edges.merge(c_sig[['firm_id', 'mv']], left_on='source_id', right_on='firm_id', how='left')
            mc['small'] = np.where(mc['mv'].isna(), np.nan,
                                   (pd.to_numeric(mc['mv'], errors='coerce') <=
                                    pd.to_numeric(c_sig['mv'], errors='coerce').median()).astype(float))
            mc['large'] = np.where(mc['mv'].isna(), np.nan, 1 - mc['small'])
            mc['wcf_s'] = mc['r_sig'] if 'r_sig' in mc.columns else 0
            # 需要 r_sig 并入 mc
            mc2 = mc.merge(r_sig, left_on='source_id', right_on='firm_id', how='left', suffixes=('', '_r'))
            mc2['wcf_s'] = mc2['r_sig'] * mc2['weight'] * mc2['small']
            mc2['wcf_l'] = mc2['r_sig'] * mc2['weight'] * mc2['large']
            num = mc2.groupby('focal_id')[['wcf_s', 'wcf_l']].sum(min_count=1)
            den = mc2.assign(w_small=mc2['weight'] * mc2['small'],
                             w_large=mc2['weight'] * mc2['large']).groupby('focal_id')[['w_small', 'w_large']].sum(min_count=1)
            den_tot = mc2.groupby('focal_id')['weight'].sum()
            # 2026-09-10 Gate 4.11：同时报告 (a) 子组平均 subgroup-average、(b) contribution（分母=全部可得 source 权重）、
            # (c) unknown 权重占比（source 市值缺失者既不入 small 也不入 large）。
            _known = den['w_small'].fillna(0) + den['w_large'].fillna(0)
            cf_ab = pd.DataFrame({
                'firm_id': num.index,
                'cf_small': num['wcf_s'] / den['w_small'],
                'cf_large': num['wcf_l'] / den['w_large'],
                'share_small': den['w_small'],
                'share_large': den['w_large'],
                'share_unknown': (den_tot - _known) / den_tot,
                'contrib_small': num['wcf_s'] / den_tot,
                'contrib_large': num['wcf_l'] / den_tot,
            }).reset_index(drop=True)
            share_unknown_mean.append(float(cf_ab['share_unknown'].mean()))
            gcs = gc.merge(cf_ab, on='firm_id', how='left').dropna(subset=['cf_small', 'cf_large'])
            if len(gcs) > 100:
                for x in ['cf_small', 'cf_large', 'contrib_small', 'contrib_large']:
                    Xc = np.column_stack([np.ones(len(gcs)), gcs[x], gcs['size'], gcs['rev']])
                    yc = gcs['r_pred'].to_numpy()
                    try:
                        bc, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
                        rows_c.append({'month': pred, 'var': x, 'beta': bc[1]})
                    except np.linalg.LinAlgError:
                        n_rank_deficient += 1

        # D: 基本面与收益（focal 收益 ~ peer 加权营收增速）
        # 2026-09-10 Gate 4.11：BM 缺失时**不得**用全零列替代控制变量；改为按当月实际
        # 可得控制集合估计（BM 可得月才纳入 BM 列），并逐月记录 controls_used。
        gd = g.dropna(subset=['peer_grow', 'size'])
        if len(gd) > 200:
            use_bm = int(gd['bm'].notna().sum()) > 100
            cols_d = [np.ones(len(gd)), gd['peer_grow'], gd['size']]
            if use_bm:
                cols_d.append(gd['bm'])
            Xd = np.column_stack(cols_d)
            yd = gd['r_pred'].to_numpy()
            try:
                bd, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
                rows_d.append({'month': pred, 'beta_peer_grow': bd[1],
                               'controls_used': 'peer_grow,size,bm' if use_bm else 'peer_grow,size'})
            except np.linalg.LinAlgError:
                n_rank_deficient += 1
        n_done += 1
        if n_done % 24 == 0:
            print(f"处理 {n_done} 个月（{pred}）", flush=True)

    def fm_summary(rows, keys):
        df = pd.DataFrame(rows)
        out = {'n_months': len(df)}
        for k in keys:
            if k in df.columns:
                v = df[k].dropna().to_numpy()
                out[k] = {'beta': round(float(v.mean()), 4),
                          't_iid': round(float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))), 2),
                          't_nw1': round(float(nw_t(v)), 2)} if len(v) > 2 else None
        return out

    def fm_summary_by_var(rows):
        """2026-09-01 R10：C 检验的 cf_small/cf_large 分开报告（原混池平均不代表任何单一系数）。"""
        df = pd.DataFrame(rows)
        out = {}
        for x, sub in df.groupby('var'):
            v = sub['beta'].dropna().to_numpy()
            out[x] = {'beta': round(float(v.mean()), 4),
                      't_iid': round(float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))), 2),
                      't_nw1': round(float(nw_t(v)), 2), 'n_months': int(len(v))} if len(v) > 2 else None
        return out

    res = {
        'note': '补做检验 B：A n_ij~相似度(蒋表3，配对级不做 focal 样本过滤) / B 关联复杂性交互(蒋表10) / C 小盘连接→大盘(AH p.17，2026-09-01 R10 起 cf_small/cf_large 分列) / D 基本面与收益(AH Table 12)。B/C/D 已套主复刻 microcap+次新过滤(R5，2026-09-01)。',
        'A_nij_similarity': fm_summary(rows_a, ['beta_ind', 'beta_size', 'beta_bm']),
        'B_link_complexity': fm_summary(rows_b, ['beta_cf', 'beta_cfxdeg', 'beta_deg']),
        'C_small_link_large_focal': fm_summary_by_var(rows_c),
        'D_fundamental_to_return': fm_summary(rows_d, ['beta_peer_grow']),
        'benchmark': {
            'jiang2023_table3': '同行业 0.524***/t13、-|ΔBM| 0.224***、-|ΔGrowth| 0.008***（相似度↑共同覆盖↑）',
            'jiang2023_table10': '#Connected×CFRet=0.003**（t2.56）——关联越复杂效应越强',
            'ah2020_p17': '小盘连接公司收益预测大盘焦点（五因子 α=0.76%/t4.69）',
        },
        'network_version': NET_VER,
    }
    if share_unknown_mean:
        res['C_unknown_weight_share'] = {
            'mean': round(float(np.mean(share_unknown_mean)), 6),
            'n_months': len(share_unknown_mean),
            'definition': 'source 市值缺失（无法判定 small/large）的权重占全部可得 source 权重之比',
            'note': 'unknown 权重既不进 small 也不进 large；subgroup-average 分母仅含已知组，contribution 分母含全部可得权重',
        }
    if rows_d:
        _d = pd.DataFrame(rows_d)
        res['D_controls_used_counts'] = {str(k): int(v) for k, v in
                                         _d['controls_used'].value_counts().items()}
        res['D_controls_note'] = ('BM 可得月纳入 BM 列；BM 缺失月仅用 peer_grow,size，'
                                  '不做零列替代（2026-09-10 Gate 4.11）')
    res['n_rank_deficient_skips'] = int(n_rank_deficient)
    res['n_rank_deficient_note'] = ('lstsq 遇奇异设计矩阵（LinAlgError）的月-规格被跳过，'
                                    '计数在此；对应 beta 不进入月均，非静默丢弃')
    attach_meta(res, __file__, NET_VER)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False)[:800], flush=True)


if __name__ == '__main__':
    main()
