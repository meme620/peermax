"""S3A smoketest B2：派生控制变量之间的冗余度（2026-08-14）。

自设计：6 个控制变量（AnalystCoverage/BrokerCoverage/NetworkDegree/
AnalystNetworkMass/EffectivePeerCount/Top1Weight）。
问题：BrokerCoverage 是否只是 AnalystCoverage 的冗余代理？
外部方案：只用 AnalystCoverage（文献标准）或换用 degree。
本脚本直接读 firm_analyst_derived.csv 算 Spearman 相关矩阵 + 缺失率。
用法: python scripts/analyst_smoketest_control_redundancy.py
输出: data/processed/analyst/smoketest_b2_control_redundancy.json
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np

DER = 'data/processed/analyst/firm_analyst_derived.csv'
OUT = Path('data/processed/analyst/smoketest_b2_control_redundancy.json')
VARS = ['AnalystCoverage', 'BrokerCoverage',
        'NetworkDegree', 'AnalystNetworkMass', 'EffectivePeerCount', 'Top1Weight']


def main():
    d = pd.read_csv(DER, dtype=str, keep_default_na=False)
    for c in VARS:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    corr = d[VARS].corr(method='spearman').round(4)
    out = {
        'rows': len(d),
        'nonmissing_pct': {c: round(float(d[c].notna().mean()) * 100, 2) for c in VARS},
        'spearman_corr': corr.to_dict(),
    }
    print(json.dumps({'nonmissing_pct': out['nonmissing_pct']}, ensure_ascii=False, indent=1), flush=True)
    print(corr.to_string(), flush=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
