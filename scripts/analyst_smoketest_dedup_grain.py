"""analyst-network smoketest A2：报告级去重粒度的影响验证（2026-08-14）。

自设计：step4 按 (firm, analyst, report_date) 报告级去重后再聚合覆盖。
外部方案：不按报告日去重，直接对行级数据做窗口聚合（同日多预测行不合并）。
预期：两种口径的覆盖集合一致（因为 v2 的 step5 本来就读行级数据，按 (firm, analyst)
聚合，report_count 用 nunique(report_date)——同日多行天然只算 1 个报告日）。
本脚本在 3 个抽查月验证"覆盖集合是否完全一致 + report_count 是否有差异"。
用法: python scripts/analyst_smoketest_dedup_grain.py
输出: data/processed/analyst/smoketest_a2_dedup_grain.json
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np

CLEAN = 'data/processed/analyst/analyst_step3_clean.csv'
COV = 'data/processed/analyst/firm_analyst_coverage.csv'
OUT = Path('data/processed/analyst/smoketest_a2_dedup_grain.json')
CUTOFFS = ['2018-06-30', '2021-06-30', '2024-06-30']


def main():
    rec = pd.read_csv(CLEAN, dtype=str, keep_default_na=False,
                      usecols=['firm_id', 'Anndats', 'Fpenddt', 'analyst_id', 'mapping_status'])
    rec = rec[rec['mapping_status'] == 'ok']
    rec = rec[rec['analyst_id'].str.match(r'^A\d{5}$', na=False)]
    rec['report_date'] = pd.to_datetime(rec['Anndats'])
    fy = pd.to_datetime(rec['Fpenddt'], errors='coerce').dt.year
    rec = rec[(fy - rec['report_date'].dt.year).between(0, 1)]
    rec = rec.sort_values('report_date', kind='mergesort')
    dates = rec['report_date'].to_numpy(dtype='datetime64[ns]')

    cov = pd.read_csv(COV, dtype=str, keep_default_na=False,
                      usecols=['network_cutoff', 'firm_id', 'analyst_id', 'report_count'])
    cov['report_count'] = pd.to_numeric(cov['report_count'], errors='coerce')

    out = {}
    for c in CUTOFFS:
        ws = pd.Timestamp(c) - pd.DateOffset(months=12)
        lo = dates.searchsorted(np.datetime64(ws), side='right')
        hi = dates.searchsorted(np.datetime64(c), side='right')
        sub = rec.iloc[lo:hi]
        # 行级直接聚合（不按报告日去重）：覆盖集合 + 报告日 nunique
        g = sub.groupby(['firm_id', 'analyst_id'])['Anndats'].nunique().reset_index()
        g.columns = ['firm_id', 'analyst_id', 'report_count_row']
        row_set = set(zip(g['firm_id'], g['analyst_id']))
        # 当前覆盖表（step4 报告日去重后 + step5 聚合）
        cc = cov[cov['network_cutoff'] == c]
        cov_set = set(zip(cc['firm_id'], cc['analyst_id']))
        m = cc.merge(g, on=['firm_id', 'analyst_id'], how='outer')
        out[c] = {
            'pairs_row_level': len(row_set),
            'pairs_coverage_table': len(cov_set),
            'sets_identical': row_set == cov_set,
            'only_in_row_level': len(row_set - cov_set),
            'only_in_coverage_table': len(cov_set - row_set),
            'report_count_identical': int((m['report_count'] == m['report_count_row']).all()),
        }
        print(c, json.dumps(out[c]), flush=True)

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
