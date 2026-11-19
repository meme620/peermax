"""analyst-network CF MOM 复现诊断（三轮对抗 review 实证闭环，2026-08-12）。

用法: python analyst_cf_mom_replication.py [weight=count|ln] [horizon=1|12]
  weight: count=共同分析师数行归一化（主口径）；ln=ln(1+count) 行归一化（Long et al. 2025）
  horizon: 1=上个月 CF 动量；12=过去 12 个月（跳过最近 1 个月）CF 动量（AH 2020 主口径）

口径对齐：
  - 月末 t-1 形成网络（cutoff=t-1 月末，只使用 Anndats <= cutoff 的 FY1/FY2 覆盖）；
  - horizon=1 : CF_ret(i,t-1) = sum_j w_ij * R(j,t-1)
  - horizon=12: CF_ret(i,t-1) = sum_j w_ij * cumret(j, t-12..t-2)（AH "12-month skipping most recent"）
  - 预测 focal 在 t 月的收益 R(i,t)；
  - 每月横截面按 CF_ret 分 5 组，报告 EW/VW 未来收益、多空（高分位减低分位）与 t 统计（iid + NW L=1）。

对照基准：
  - Jansen et al. 2021（IBES 稀疏）：CF MOM(1m) EW 0.46%/月 t=1.54 不显著
  - Long et al. 2025（CSMAR 全市场）：CF(1m) alpha=0.97%/月 t=3.52 显著
  - Ali & Hirshleifer 2020（美国）：主口径为上月同行加权收益（式 1），EW 五因子 α=2.10%/月（t=11.88）

输出: data/processed/analyst/cf_mom_replication[_{weight}][_h{horizon}].json
"""
from dates import next_month as _next_month
from statsutil import nw_t

import json, sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

from analyst_meta import attach_meta
from fileio import file_size

EDGES = Path('data/processed/analyst/edges')
SPEC = Path('configs/analyst_spec.yaml')
_SPEC = yaml.safe_load(SPEC.read_text(encoding='utf-8'))
RET = Path(_SPEC['inputs']['returns'])
MEQ = Path(_SPEC['inputs']['market_equity'])
OUT = Path('data/processed/analyst/cf_mom_replication.json')
NET_VER = _SPEC['network']['primary_version']
H12_MIN_AVAIL = int(_SPEC['horizon']['h12_min_available_months'])
if not 1 <= H12_MIN_AVAIL <= 11:
    # H12 窗口 t-12..t-2 共 11 个月：门槛大于 11 时过滤恒为空，h12 变体整体无产出
    raise SystemExit(f'h12_min_available_months={H12_MIN_AVAIL} 超出 11 个月窗口，不可满足')
_SPEC_HASH = file_size(SPEC)
_INPUT_HASHES = _SPEC['provenance']['input_hashes']


def main():
    global NET_VER, RET
    weight_mode = sys.argv[1] if len(sys.argv) > 1 else 'count'
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    if weight_mode not in {'count', 'ln'}:
        raise ValueError('weight must be count or ln')
    if horizon not in {1, 12}:
        raise ValueError('horizon must be 1 or 12')
    if len(sys.argv) > 3:
        if sys.argv[3] not in {'coverage_win12_fy12_v3', 'coverage_win12_nofy_v4'}:
            raise ValueError('network must be coverage_win12_fy12_v3 or coverage_win12_nofy_v4')
        NET_VER = sys.argv[3]
    # 2026-09-10 Gate 4.7（9F）：收益口径为**显式**变体，禁止环境变量静默切换。
    returns_mode = sys.argv[4] if len(sys.argv) > 4 else 'raw'
    if returns_mode not in {'raw', 'adj'}:
        raise ValueError('returns must be raw or adj')
    if returns_mode == 'adj':
        RET = Path(_SPEC['inputs']['returns_post_adjusted'])
    suffix = '' if (weight_mode == 'count' and horizon == 1) else f'_{weight_mode}_h{horizon}'
    if NET_VER == 'coverage_win12_nofy_v4':
        suffix += '_nofy'
    if returns_mode == 'adj':
        suffix += '_adj'

    ret = pd.read_csv(RET, dtype=str, keep_default_na=False)
    ret['firm_id'] = ret['firm_id'].str.replace(r'^(sz|sh)', '', regex=True)
    ret['R'] = pd.to_numeric(ret['R'], errors='coerce')
    ret = ret[ret['R'].notna()]
    ret = ret[['firm_id', 'month', 'R']]
    months = sorted(ret['month'].unique())
    print(f"monthly returns: {len(ret)} rows, months {months[0]}~{months[-1]} ({len(months)})", flush=True)
    ret_pivot = ret.pivot(index='firm_id', columns='month', values='R')

    meq = pd.read_csv(MEQ, dtype=str, keep_default_na=False)
    meq['date'] = pd.to_datetime(meq['date'])
    meq['month'] = meq['date'].dt.strftime('%Y-%m')
    meq['me'] = pd.to_numeric(meq['market_equity'], errors='coerce')
    meq = meq[['firm_id', 'month', 'me']].dropna()
    # 2026-09-06 Step 10/Gate 1: derive microcap thresholds from the
    # complete forming-month market cross-section before joining research rows.
    meq_q10 = (meq.groupby('month', as_index=False)['me']
                  .quantile(0.10)
                  .rename(columns={'me': 'me_q10_form'}))

    rows = []
    for cf in sorted(EDGES.glob(f'edges_{NET_VER}_*.csv')):
        cutoff = cf.stem.replace(f'edges_{NET_VER}_', '')
        pred_month = _next_month(cutoff[:7])
        if pred_month not in months:
            continue
        if horizon == 1:
            signal_month = cutoff[:7]
            if signal_month not in months:
                continue
            r_sig = ret[ret['month'] == signal_month][['firm_id', 'R']].rename(columns={'R': 'r_sig'})
        else:
            t_idx = months.index(pred_month)
            if t_idx < horizon + 1:
                continue
            # H12 是“12-2 位置窗口”：使用 t-12..t-2，共 11 个自然月收益，
            # 明确跳过最近信号月 t-1；不要把 11 个月收益误称为 12 个月收益。
            win = months[t_idx - horizon:t_idx - 1]
            if horizon == 12 and len(win) != 11:
                raise RuntimeError(f'H12 calendar window error at {pred_month}: {len(win)} months')
            if len(win) < 2:
                continue
            sub = ret_pivot[win]
            if horizon == 12:
                # H12_MIN_AVAIL：窗口内可得月数 >= 8 才进入 CF 累计（2026-08-14 修改，
                # 依据 smoketest_c4_min_months.json：≥1/≥8/≥11 个月结果几乎相同 0.470~0.483%/月、
                # t 0.88~0.91；取 ≥8 避免仅 1 个月可得的股票以极短窗口进排序）
                avail = sub.notna().sum(axis=1)
                sub = sub[avail >= H12_MIN_AVAIL]
            cum = (1 + sub).prod(axis=1, min_count=1) - 1   # 可得月累计：跳过缺失月（停牌不假设 0 收益），全缺失->NaN 由 inner join 丢弃
            r_sig = cum.reset_index().rename(columns={0: 'r_sig'})

        edges = pd.read_csv(cf, dtype=str, usecols=['focal_id', 'source_id', 'common_analyst_count', 'weight'])
        if weight_mode == 'ln':
            c = pd.to_numeric(edges['common_analyst_count'], errors='coerce').fillna(0)
            lnw = np.log1p(c)
            denom = lnw.groupby(edges['focal_id']).transform('sum')
            edges['weight'] = lnw / denom.replace(0, np.nan)
        else:
            edges['weight'] = pd.to_numeric(edges['weight'], errors='coerce')
        edges = edges[edges['weight'].notna() & (edges['weight'] > 0)]

        r_pred = ret[ret['month'] == pred_month][['firm_id', 'R']].rename(columns={'R': 'r_pred'})
        # 注意：inner join 会丢弃缺收益的 source（停牌/无数据），随后 groupby 内 np.average 对剩余
        # source 的权重自动重新归一化——这是"用可得同行加权平均"的标准处理，声明于此（2026-08-12）。
        m = edges.merge(r_sig, left_on='source_id', right_on='firm_id', how='inner')
        m = m.merge(r_pred, left_on='focal_id', right_on='firm_id', how='inner', suffixes=('', '_p'))
        if m.empty:
            continue
        g = m.groupby('focal_id').apply(
            lambda d: pd.Series({'cf': float(np.average(d['r_sig'], weights=d['weight'])),
                                 'r_pred': float(d['r_pred'].iloc[0])}), include_groups=False).reset_index()
        # 修改 6（2026-08-12）：CFRET2 二阶同行动量（AH 论文式 3，Table 8 链接复杂度维度）
        #   CFRET2(i) = sum_j w_ij * CF(j)，即用网络权重对"同行的 CF"再加权一次
        cf_map = dict(zip(g['focal_id'], g['cf']))
        e2 = edges.copy()
        e2['cf_src'] = e2['source_id'].map(cf_map)
        e2 = e2.dropna(subset=['cf_src'])
        if len(e2):
            g2 = e2.groupby('focal_id').apply(
                lambda d: float(np.average(d['cf_src'], weights=d['weight'])),
                include_groups=False).reset_index()
            g2.columns = ['focal_id', 'cfret2']
            g = g.merge(g2, on='focal_id', how='left')
        else:
            g['cfret2'] = np.nan
        g['month_pred'] = pred_month
        g['month_form'] = cutoff[:7]   # 信号/网络形成月（cutoff 月），用于形成期市值加权
        rows.append(g)
        if len(rows) % 12 == 0:
            print(f"signal months done: {len(rows)} (at {pred_month})", flush=True)

    if not rows:
        raise SystemExit('CF 面板为空：无任何月份产出（检查 edges glob、收益月份覆盖与 H12 门槛）')
    panel = pd.concat(rows, ignore_index=True)
    # v2 修复（对抗检查 R2-2）：VW 权重改用形成期（cutoff 月）月末市值，
    # 不再用收益实现月（pred_month）月末市值，消除组合权重对当月收益的轻微前视。
    meq_p = meq.rename(columns={'me': 'me_form'})
    panel = panel.merge(meq_p, left_on=['focal_id', 'month_form'], right_on=['firm_id', 'month'], how='left')

    # 修改 5（2026-08-12）：价格/规模过滤，对齐 methodology §4.2 与 AH 排除低价股。
    # 排除：① microcap（每月形成期市值最低 10%）；② 次新股（上市不足 6 个月）。
    # limit-hit（涨跌停）需日度数据，留待日度层，此处声明不实现。
    ipo = pd.read_csv('data/processed/firm_id_map.csv', dtype=str)
    ipo_map = dict(zip(ipo['code6'].str.strip(), ipo['ipo_date'].str.strip()))
    panel['ipo_date'] = panel['focal_id'].map(ipo_map)
    panel['months_since_ipo'] = (pd.to_datetime(panel['month_pred']) - pd.to_datetime(panel['ipo_date'], errors='coerce')).dt.days / 30.4
    panel = panel.merge(meq_q10.rename(columns={'month': 'month_form_q10'}),
                        left_on='month_form', right_on='month_form_q10', how='left')
    panel['microcap'] = panel['me_form'] < panel['me_q10_form']
    panel['recent_ipo'] = panel['months_since_ipo'] < 6
    n_pre = len(panel)
    panel = panel[~(panel['microcap'].fillna(False) | panel['recent_ipo'].fillna(False))]
    print(f"价格/规模过滤: {n_pre} -> {len(panel)}（剔除 microcap+次新股 {(1 - len(panel) / n_pre) * 100:.1f}%）", flush=True)

    results = {'network_version': NET_VER}
    monthly = []
    for mp, sub in panel.groupby('month_pred'):
        if len(sub) < 10:
            continue
        sub = sub.copy()
        sub['q'] = pd.qcut(sub['cf'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        rec = {'month': mp}
        for q in range(1, 6):
            g = sub[sub['q'] == q]
            rec[f'ew_q{q}'] = g['r_pred'].mean()
            gw = g.dropna(subset=['me_form'])
            rec[f'vw_q{q}'] = float(np.average(gw['r_pred'], weights=gw['me_form'])) if len(gw) else np.nan
        rec['ew_ls'] = rec['ew_q5'] - rec['ew_q1']
        rec['vw_ls'] = rec['vw_q5'] - rec['vw_q1']
        # CFRET2 诊断（AH Table 8 链接复杂度维度简化版）：高二阶 vs 低二阶组的未来收益差
        cm_sample = sub.dropna(subset=['cfret2'])
        if len(cm_sample) >= 20:
            med = cm_sample['cfret2'].median()
            rec['cfret2_hi_lo'] = cm_sample[cm_sample['cfret2'] >= med]['r_pred'].mean() - cm_sample[cm_sample['cfret2'] < med]['r_pred'].mean()
        else:
            rec['cfret2_hi_lo'] = np.nan
        monthly.append(rec)
    df = pd.DataFrame(monthly)
    results['weight_mode'] = weight_mode
    c2 = df['cfret2_hi_lo'].dropna()
    results['cfret2_diagnostic'] = {
        'n_months': int(len(c2)),
        'mean_hi_lo_pct': round(float(c2.mean()) * 100, 4) if len(c2) else None,
        't_iid': round(float(c2.mean() / c2.std() * np.sqrt(len(c2))), 2) if len(c2) > 2 else None,
        'note': '高二阶（同行的同行涨幅高）组未来收益 - 低二阶组；正 = 网络深度传播存在（AH Table 8 复杂度维度）',
    }
    results['horizon_months'] = horizon
    results['h12_min_avail_months'] = H12_MIN_AVAIL if horizon == 12 else None
    results['n_months'] = len(df)
    results['sample'] = f"{df['month'].min()} ~ {df['month'].max()}" if len(df) else 'EMPTY'
    results['mean_monthly_ret_pct'] = {}
    for col in ['ew_q1', 'ew_q2', 'ew_q3', 'ew_q4', 'ew_q5', 'ew_ls', 'vw_ls']:
        s = df[col].dropna()
        results['mean_monthly_ret_pct'][col] = round(float(s.mean()) * 100, 4) if len(s) else None
    ew = df['ew_ls'].dropna()
    vw = df['vw_ls'].dropna()   # v2：t 统计前剔除 NaN 月（旧版 NW 对含 NaN 序列输出 NaN）
    results['t_stats'] = {
        'ew_ls_iid': round(float(ew.mean() / ew.std() * np.sqrt(len(ew))), 2) if len(ew) > 2 else None,
        'ew_ls_nw1': round(float(nw_t(ew.values, L=1)), 2) if len(ew) > 2 else None,
        'vw_ls_iid': round(float(vw.mean() / vw.std() * np.sqrt(len(vw))), 2) if len(vw) > 2 else None,
        'vw_ls_nw1': round(float(nw_t(vw.values, L=1)), 2) if len(vw) > 2 else None,
        'n_valid_months': {'ew': int(len(ew)), 'vw': int(len(vw))},
    }
    results['benchmark'] = {
        'jansen2021_cf_mom_1m_ew_pct': 0.46, 'jansen2021_t': 1.54,
        'long2025_cf_alpha_pct': 0.97, 'long2025_t': 3.52,
        'ah2020_core': 'CF RET（上月同行加权收益，式 1）为 AH 主口径；EW 五因子 α=2.10%/月 t=11.88',
        'note': 'Jansen 用 IBES 稀疏覆盖不显著；Long 0.97%/t=3.52 为六因子（FF2018）CF factor 的 spanning α（Table 7 Panel D），与项目五因子组合 α 口径不同；本项目 CNRDS 全市场 + FY1/FY2 限定。'
    }
    results['monthly'] = df.to_dict(orient='records')

    attach_meta(results, __file__, network_version=NET_VER, returns_mode=returns_mode)
    out_path = OUT.with_name(OUT.stem + suffix + OUT.suffix)
    (out_path).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"weight={weight_mode} horizon={horizon} -> {out_path.name}")
    print(json.dumps({k: v for k, v in results.items() if k != 'monthly'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
