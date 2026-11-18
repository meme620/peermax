"""S3A M1 面板合并缺失成分分析（待办 3，2026-08-14）。

背景：step10 把 6 个 analyst 控制变量并入月度定价面板，merge_rate=60.62%；
剩余 39.4% 为无共同分析师覆盖/无网络支撑的公司（不进 M1 主回归样本）。
本脚本对比 covered（有 analyst 控制，进主回归）与 missing（无，被排除）
两组在以下维度的分布，判断排除 missing 组是否构成系统性样本选择：
  - 市值（log me，月末市值）
  - 上市月龄（ipo_date 起）
  - 自身 MAX（OwnMAXraw，彩票特征）
  - 同行收益（PeerReturn）与当期收益（R）
  - 产品网络字段（support_mass/covered_mass/coverage_ratio）可得性

用法: python scripts/data_audit/s3a_panel_missing_analysis.py
输出: data/processed/analyst/panel_missing_analysis.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PANEL = 'data/processed/monthly_pricing_panel_with_analyst.csv'
MEQ = 'data/processed/market_equity_panel.csv'
IDMAP = 'data/processed/firm_id_map.csv'
RET = 'data/processed/monthly_returns_clean.csv'
OUT = Path('data/processed/analyst/panel_missing_analysis.json')


def main():
    panel = pd.read_csv(PANEL, dtype=str, keep_default_na=False)
    panel['covered'] = panel['AnalystCoverage'].str.strip() != ''
    panel['code6'] = panel['focal_id'].str.replace(r'^(sz|sh)', '', regex=True)

    # 市值：market_equity_panel（firm_id 纯 6 位，date 为月末）
    meq = pd.read_csv(MEQ, dtype=str, keep_default_na=False)
    meq['me'] = pd.to_numeric(meq['market_equity'], errors='coerce')
    meq['month'] = pd.to_datetime(meq['date']).dt.strftime('%Y-%m')
    panel = panel.merge(meq[['firm_id', 'month', 'me']], left_on=['code6', 'month'],
                        right_on=['firm_id', 'month'], how='left')

    # 上市月龄（firm_id_map code6 为纯 6 位）
    idm = pd.read_csv(IDMAP, dtype=str, keep_default_na=False)
    ipo_map = dict(zip(idm['code6'], idm['ipo_date']))
    panel['ipo_date'] = panel['code6'].map(ipo_map)
    panel['age_months'] = ((pd.to_datetime(panel['month']) - pd.to_datetime(panel['ipo_date'], errors='coerce'))
                           .dt.days / 30.4)

    # 当期收益 R（monthly_returns_clean firm_id 带前缀，转 code6 与 panel 匹配）
    ret = pd.read_csv(RET, dtype=str, keep_default_na=False, usecols=['firm_id', 'month', 'R'])
    ret['R'] = pd.to_numeric(ret['R'], errors='coerce')
    ret['code6'] = ret['firm_id'].str.replace(r'^(sz|sh)', '', regex=True)
    panel = panel.merge(ret[['code6', 'month', 'R']], on=['code6', 'month'], how='left')

    for c in ['OwnMAXraw', 'PeerReturn', 'support_mass', 'covered_mass', 'coverage_ratio']:
        panel[c] = pd.to_numeric(panel[c], errors='coerce')

    def stats(g):
        out = {'n_rows': int(len(g)), 'n_firms': int(g['code6'].nunique()), 'n_months': int(g['month'].nunique())}
        out['log_me_mean'] = round(float(np.log(g['me']).mean()), 4) if g['me'].notna().any() else None
        out['log_me_median'] = round(float(np.log(g['me']).median()), 4) if g['me'].notna().any() else None
        out['me_nonmissing_pct'] = round(float(g['me'].notna().mean()) * 100, 2)
        out['age_months_mean'] = round(float(g['age_months'].mean()), 1) if g['age_months'].notna().any() else None
        out['ownmax_mean'] = round(float(g['OwnMAXraw'].mean()), 4) if g['OwnMAXraw'].notna().any() else None
        out['peerret_mean'] = round(float(g['PeerReturn'].mean()), 4) if g['PeerReturn'].notna().any() else None
        out['ret_mean'] = round(float(g['R'].mean()), 4) if g['R'].notna().any() else None
        out['support_mass_nonmissing_pct'] = round(float(g['support_mass'].notna().mean()) * 100, 2)
        return out

    cov = panel[panel['covered']]
    mis = panel[~panel['covered']]
    rep = {
        'note': 'M1 面板合并缺失成分分析：covered=有 analyst 控制（进主回归）；missing=无覆盖/无网络支撑（被排除）。',
        'covered_pct': round(float(panel['covered'].mean()) * 100, 2),
        'missing_pct': round(float((~panel['covered']).mean()) * 100, 2),
        'covered': stats(cov),
        'missing': stats(mis),
        'diff_mean': {},
    }
    for k in ['log_me_mean', 'age_months_mean', 'ownmax_mean', 'peerret_mean', 'ret_mean']:
        a, b = rep['covered'].get(k), rep['missing'].get(k)
        rep['diff_mean'][k] = round(a - b, 4) if a is not None and b is not None else None
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print('DONE')


if __name__ == '__main__':
    main()
