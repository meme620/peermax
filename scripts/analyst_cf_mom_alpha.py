"""CF MOM 多空组合的因子 alpha 调整（对齐 Long et al. 2025 报告口径，2026-08-12）。

输入: data/processed/analyst/cf_mom_replication_ln.json（ln 权重 1 个月口径，最优口径）
      data/processed/factor_returns_daily.csv（FF4 日度因子，S2.5 自建）
回归: 多空月收益 ~ alpha + MKT + SMB + HML + MOM（FF4）
输出: 月度 alpha（%）、Newey-West L=1 t 统计、因子暴露、年化 alpha
"""
# --- scripts/lib bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from statsutil import nw_ols
# --- end scripts/lib bootstrap ---

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

SPEC = Path('configs/s3a_final_spec.yaml')
_SPEC = yaml.safe_load(SPEC.read_text(encoding='utf-8'))
DEFAULT_CF = 'data/processed/analyst/cf_mom_replication_ln_h1.json'
FAC = 'data/processed/factor_returns_daily.csv'
REV_FAC = 'data/processed/rev_factor_monthly.csv'
DEFAULT_OUT = 'data/processed/analyst/cf_mom_alpha.json'


def main():
    cf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_CF)
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(DEFAULT_OUT)
    if len(sys.argv) > 3:
        raise ValueError('usage: python analyst_cf_mom_alpha.py [cf_json] [output_json]')
    if not cf_path.exists():
        raise FileNotFoundError(cf_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cf = json.loads(cf_path.read_text(encoding='utf-8'))
    df = pd.DataFrame(cf['monthly'])
    f = pd.read_csv(FAC, dtype=str, keep_default_na=False)
    f['date'] = pd.to_datetime(f['date'])
    f['month'] = f['date'].dt.strftime('%Y-%m')
    for col in ['rf_d', 'MKT_d', 'SMB_d', 'HML_d', 'MOM_d']:
        f[col] = pd.to_numeric(f[col], errors='coerce')
    fm = f.groupby('month').apply(lambda d: pd.Series({
        'MKT': float((1 + d['MKT_d']).prod() - 1),
        'SMB': float((1 + d['SMB_d']).prod() - 1),
        'HML': float((1 + d['HML_d']).prod() - 1),
        'MOM': float((1 + d['MOM_d']).prod() - 1),
    }), include_groups=False)
    df = df.merge(fm, left_on='month', right_index=True, how='inner').dropna(subset=['ew_ls', 'MKT'])
    # REV（短期反转，AH 五因子第 5 个；2026-08-12 build_rev_factor.py 构造）
    rev = pd.read_csv(REV_FAC, dtype=str, keep_default_na=False)
    rev['REV'] = pd.to_numeric(rev['REV'], errors='coerce')
    df = df.merge(rev[['month', 'REV']], on='month', how='left')
    df = df.sort_values('month').reset_index(drop=True)
    print(f"多空序列与因子对齐: {len(df)} 个月 ({df['month'].min()} ~ {df['month'].max()})", flush=True)

    # FF5（含 REV）为主，FF4 对照；weight_mode 从输入序列读取（2026-09-03：支持 count 权重 α 补跑）
    res = {'weight_mode': cf.get('weight_mode', 'ln'), 'horizon': 1,
           'factor_model': 'FF5 (MKT SMB HML MOM REV) + alpha, FF4 对照'}
    for dep in ['ew_ls', 'vw_ls']:
        y = df[dep].values
        d5 = df.dropna(subset=[dep, 'MKT', 'SMB', 'HML', 'MOM', 'REV'])
        y5 = d5[dep].values
        X5 = np.column_stack([np.ones(len(y5)), d5['MKT'], d5['SMB'], d5['HML'], d5['MOM'], d5['REV']])
        beta5, se5, t5 = nw_ols(X5, y5, L=1)
        names5 = ['alpha', 'MKT', 'SMB', 'HML', 'MOM', 'REV']
        X4 = np.column_stack([np.ones(len(df)), df['MKT'], df['SMB'], df['HML'], df['MOM']])
        beta4, se4, t4 = nw_ols(X4, y, L=1)
        names4 = ['alpha', 'MKT', 'SMB', 'HML', 'MOM']
        res[dep] = {
            'ff5': {n: {'beta_pct': round(float(b) * 100, 4), 't': round(float(tt), 2)}
                    for n, b, tt in zip(names5, beta5, t5)},
            'ff4': {n: {'beta_pct': round(float(b) * 100, 4), 't': round(float(tt), 2)}
                    for n, b, tt in zip(names4, beta4, t4)},
            'ff5_annualized_alpha_pct': round(float(beta5[0]) * 1200, 2),
            'ff4_annualized_alpha_pct': round(float(beta4[0]) * 1200, 2),
            'ff5_raw_mean_pct': round(float(y5.mean()) * 100, 4),
            'ff5_n_months': int(len(y5)),
        }

    res.update({
        'spec_id': _SPEC['spec_id'],
        'spec_hash': __import__('hashlib').sha256(SPEC.read_bytes()).hexdigest(),
        'input_hashes': _SPEC['provenance']['input_hashes'],
        'input_cf': str(cf_path),
        'run_at': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        'command': ' '.join(sys.argv),
        'status': 'GENERATED',
    })
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
