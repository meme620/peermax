"""S3A Step 10：分析师网络派生变量接入月度定价面板（控制变量落地，2026-08-12）。

把 firm_analyst_derived.csv（firm-month 派生变量）合并进 monthly_pricing_panel，
作为 M1 Fama-MacBeth 回归的 analyst 控制变量（methodology §4.1 的 X 中
AnalystCoverage / NetworkDegree 等；execution_plan S5 主回归样本要求全部注册 controls）。

时间对齐（无前视）：
  - 面板 month = focal 月 t（PeerMAXraw 用 t 期可得信息，R_next = t+1 月收益）；
  - 控制变量用网络 cutoff = t 月末（Anndats <= 月末，t 月末可得），即 derived 的
    month=cutoff 转成 'YYYY-MM' 后与面板同月 merge。

输入:
  data/processed/monthly_pricing_panel_partial.csv
  data/processed/analyst/firm_analyst_derived.csv
输出:
  data/processed/monthly_pricing_panel_with_analyst.csv
  data/processed/analyst/step10_panel_merge.json（覆盖/缺失诊断）
"""
import json
from pathlib import Path
import pandas as pd

PANEL = 'data/processed/monthly_pricing_panel_partial.csv'
DER = 'data/processed/analyst/firm_analyst_derived.csv'
OUTP = 'data/processed/monthly_pricing_panel_with_analyst.csv'
OUTD = Path('data/processed/analyst')
CTL = ['AnalystCoverage', 'BrokerCoverage',
       'NetworkDegree', 'AnalystNetworkMass', 'EffectivePeerCount', 'Top1Weight']


def main():
    panel = pd.read_csv(PANEL, dtype=str, keep_default_na=False)
    panel['focal_id_prefix'] = panel['focal_id']   # 保留原始带 sz/sh 前缀的 ID（输出时恢复，保持与输入面板 ID 约定一致）
    panel['focal_id'] = panel['focal_id'].str.replace(r'^(sz|sh)', '', regex=True)
    n_panel = len(panel)
    print(f"panel rows: {n_panel}, months {panel['month'].min()}~{panel['month'].max()}", flush=True)

    der = pd.read_csv(DER, dtype=str, keep_default_na=False,
                      usecols=['firm_id', 'month'] + CTL)
    der['month'] = der['month'].str[:7]          # '2018-01-31' -> '2018-01'
    der = der.rename(columns={'firm_id': 'focal_id'})
    # 同一 (focal, month) 只保留一行（derived 唯一键应为 firm+month）
    dup = der.duplicated(['focal_id', 'month']).sum()
    if dup:
        print(f"WARN derived 重复 (focal,month): {dup}，取首行", flush=True)
        der = der.drop_duplicates(['focal_id', 'month'], keep='first')

    out = panel.merge(der, on=['focal_id', 'month'], how='left')
    out['focal_id'] = out['focal_id_prefix']   # 恢复带前缀 ID（与输入 monthly_pricing_panel_partial 一致）
    out = out.drop(columns=['focal_id_prefix'])
    for c in CTL:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    out.to_csv(OUTP, index=False, encoding='utf-8')

    # 诊断
    nonmiss = out[CTL].notna()
    rep = {
        'panel_rows': n_panel,
        'panel_months': out['month'].nunique(),
        'merged_rows': int(nonmiss['AnalystCoverage'].sum()),
        'merge_rate_pct': round(float(nonmiss['AnalystCoverage'].mean()) * 100, 2),
        'control_nonmissing_pct': {c: round(float(nonmiss[c].mean()) * 100, 2) for c in CTL},
        'monthly_merge_rate': out.groupby('month')['AnalystCoverage'].apply(
            lambda s: round(float(s.notna().mean()) * 100, 1)).head(6).to_dict(),
        'note': '缺失行 = 无共同分析师覆盖/无网络支持的公司（不进 M1 主回归样本，见 execution_plan S5）',
    }
    (OUTD / 'step10_panel_merge.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
