"""S3A Step 9：firm-month 派生变量表（v2，2026-08-12）。

从冻结口径的覆盖表/边表生成 methodology §3.2A 第 144 行与 execution_plan S3A Step 6 定义的派生变量：
  AnalystCoverage       : 覆盖该公司的分析师数
  BrokerCoverage        : 覆盖该公司的去重券商数（与网络同口径：FY1/FY2 + 12 月窗口）
  NetworkDegree         : 网络度数（连接 peer 数）
  AnalystNetworkMass    : sum_j common_analyst_count(i,j)（权重分母）
  EffectivePeerCount    : 1 / sum_j w_ij^2
  Top1Weight            : max_j w_ij

v2 变更（2026-08-12，对抗检查 R2-3/P2-1 修复）:
  - network_version 列由 cutoff 日期改为真实版本串（与 step5 VERSION 同源）；
  - BrokerCoverage 由"全历史 records 的 broker 明细"改为与覆盖表完全同口径：
    行级 clean 数据 -> ok/格式校验 -> FY1/FY2 行级过滤 -> 当 cutoff 的 12 个月窗口
    -> 非空券商码去重计数。修复旧版两处虚高来源（换券商分析师的旧券商计入当月、
    空串 broker_code 被 nunique 当作一个券商）。

输入: firm_analyst_coverage.csv（v2 冻结口径）、s3a_step3_clean.csv（broker 明细）、
      edges/edges_<version>_<cutoff>.csv
输出: data/processed/analyst/firm_analyst_derived.csv（firm_id, month, 派生变量）
      data/processed/analyst/step9_derived.json（月度覆盖诊断）
"""
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

COV = 'data/processed/analyst/firm_analyst_coverage.csv'
CLEAN = 'data/processed/analyst/s3a_step3_clean.csv'
EDGES = Path('data/processed/analyst/edges')
OUTD = Path('data/processed/analyst')
NET_VER = 'coverage_win12_fy12_v3'  # 与 step5/step67 同源
WINDOW_MONTHS = 12
COLS = ['firm_id', 'month', 'AnalystCoverage', 'BrokerCoverage',
        'NetworkDegree', 'AnalystNetworkMass', 'EffectivePeerCount', 'Top1Weight',
        'network_version', 'source_version']


def main():
    cov = pd.read_csv(COV, dtype=str, keep_default_na=False,
                      usecols=['network_cutoff', 'firm_id', 'analyst_id', 'source_version'])
    cov['month'] = cov['network_cutoff']

    # broker 明细（行级 clean，与覆盖表同口径：ok + A_code 格式 + FY1/FY2 行级过滤）
    rec = pd.read_csv(CLEAN, dtype=str, keep_default_na=False,
                      usecols=['firm_id', 'Anndats', 'Fpenddt', 'analyst_id', 'broker_code', 'mapping_status'])
    rec = rec[rec['mapping_status'] == 'ok']
    rec = rec[rec['analyst_id'].str.match(r'^A\d{5}$', na=False)]
    rec['report_date'] = pd.to_datetime(rec['Anndats'])
    fpenddt_year = pd.to_datetime(rec['Fpenddt'], errors='coerce').dt.year
    rec = rec[(fpenddt_year - rec['report_date'].dt.year).between(0, 1)]
    rec = rec[rec['broker_code'] != '']
    rec = rec.sort_values('report_date', kind='mergesort')
    rec_dates = rec['report_date'].to_numpy(dtype='datetime64[ns]')
    print(f"coverage rows: {len(cov)}, broker 明细行(FY+格式过滤后): {len(rec)}", flush=True)

    cutoffs = sorted(cov['month'].unique())
    cutoff_ts = pd.Timestamp(cutoffs[0])
    frames = []
    diag = []
    for k, cutoff in enumerate(cutoffs):
        cts = pd.Timestamp(cutoff)
        cs = cov[cov['month'] == cutoff]
        # 覆盖类派生变量
        g = cs.groupby('firm_id').agg(
            AnalystCoverage=('analyst_id', 'nunique'),
        ).reset_index()
        # BrokerCoverage：当 cutoff 的 12 个月窗口内、FY1/FY2 行级过滤后的公司级去重券商数
        ws = cts - pd.DateOffset(months=WINDOW_MONTHS)
        lo = rec_dates.searchsorted(np.datetime64(ws), side='right')
        hi = rec_dates.searchsorted(np.datetime64(cts), side='right')
        bk = rec.iloc[lo:hi]
        bk_cov = bk.groupby('firm_id')['broker_code'].nunique().rename('BrokerCoverage')
        g = g.merge(bk_cov, left_on='firm_id', right_index=True, how='left')
        # 2026-08-14 修改（B4）：不再 fillna(0)——窗口内无任何非空券商码的公司 BrokerCoverage 保留
        # missing（NaN），区分"0 家券商"与"券商明细缺失"，符合 missing 不转 0 原则；
        # 空串 broker_code 已在 rec 过滤阶段剔除。
        # 网络类派生变量
        ef = pd.read_csv(EDGES / f'edges_{NET_VER}_{cutoff}.csv', dtype=str,
                         usecols=['focal_id', 'common_analyst_count', 'weight'])
        ef['c'] = pd.to_numeric(ef['common_analyst_count'], errors='coerce').fillna(0)
        ef['w'] = pd.to_numeric(ef['weight'], errors='coerce').fillna(0)
        net = ef.groupby('focal_id').apply(
            lambda d: pd.Series({
                'NetworkDegree': int(len(d)),
                'AnalystNetworkMass': float(d['c'].sum()),
                'EffectivePeerCount': float(1.0 / max((d['w'] ** 2).sum(), 1e-12)),
                'Top1Weight': float(d['w'].max()),
            }), include_groups=False).reset_index()
        g = g.merge(net, left_on='firm_id', right_on='focal_id', how='left')
        g['month'] = cutoff
        g['network_version'] = NET_VER
        g['source_version'] = cov['source_version'].iloc[0]
        frames.append(g[COLS])
        diag.append({'month': cutoff, 'firms': int(len(g)),
                     'covered_firms': int(g['AnalystCoverage'].notna().sum()),
                     'network_firms': int(g['NetworkDegree'].notna().sum())})
        if (k + 1) % 12 == 0:
            print(f"{cutoff} done ({k + 1}/{len(cutoffs)})", flush=True)

    out = pd.concat(frames, ignore_index=True)
    for col in ['NetworkDegree', 'AnalystNetworkMass', 'EffectivePeerCount', 'Top1Weight']:
        out[col] = pd.to_numeric(out[col], errors='coerce')
    out.to_csv(OUTD / 'firm_analyst_derived.csv', index=False, encoding='utf-8')
    rep = {
        'rows': int(len(out)),
        'months': len(cutoffs),
        'network_version': NET_VER,
        'unique_firms': int(out['firm_id'].nunique()),
        'monthly': diag,
        'coverage_means': {c: round(float(out[c].mean()), 4) if out[c].notna().any() else None
                           for c in ['AnalystCoverage', 'BrokerCoverage',
                                     'NetworkDegree', 'AnalystNetworkMass', 'EffectivePeerCount', 'Top1Weight']},
        'broker_coverage_missing_pct': round(float(out['BrokerCoverage'].isna().mean()) * 100, 4),
    }
    (OUTD / 'step9_derived.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in rep.items() if k != 'monthly'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
