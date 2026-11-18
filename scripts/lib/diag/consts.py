"""s3_pc_sign_diag_h1h8 的 consts 模块（逐字搬移，逻辑未改）。"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data_audit'))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data'))

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DIAG = ROOT / 'artifacts/tmp/pc_diag'

PROGRESS = ROOT / 'artifacts/tmp/pc_sign_diag_progress.md'

PANEL = ROOT / 'data/processed/m1_common_sample_ma_teacher_v2.parquet'

PC_DIR = ROOT / 'data/processed/network_ma_teacher'

PC_YEARS = list(range(2017, 2023))

CLUSTER_DIR = PC_DIR / 'clusters'

REV_PKL = ROOT / 'data/raw/drive/operating_revenue_annual.pkl'

RETURNS = ROOT / 'data/processed/monthly_returns_clean_v2.csv'

FACTORS = ROOT / 'data/processed/factor_returns_daily.csv'

SW2_MAP = ROOT / 'data/processed/sw2_industry_latest.csv'

TIER_DIR = ROOT / 'artifacts/tmp/pc_rc'          # 既有（rootcause 轮）K 档缓存目录（只读复用）

OUT_JSON = ROOT / 'reports/audits/s3_pc_sign_diagnostics_20261021.json'

OUT_MD = ROOT / 'reports/audits/s3_pc_sign_diagnostics_20261021.md'

CONTROLS = ['size', 'bm', 'mom', 'rev', 'illiq']

PAPER_CONTROLS = {  # 原文表 4 列(2)（收益单位 %）
    'MKT_beta': (0.481, 2.82), 'size': (0.339, 23.42), 'bm': (-0.358, -8.79),
    'mom': (1.791, 15.28), 'rev': (1.835, 8.27), 'illiq': (0.028, 4.68),
    'ia': (0.085, 7.72), 'roe': (2.278, 13.63), 'lev': (1.074, 13.26),
    'disp': (-0.378, -8.53), 'ivol': (14.964, 8.23),
}

PAPER_T2 = {'mean': -0.230, 'sd': 0.337, 'p25': -0.416, 'median': -0.180, 'p75': -0.076, 'n': 15482}

TARGETS = {'c1': (0.152, 7.76), 'c2': (0.130, 5.40), 'hl_a3_ew': (0.218, 6.27)}

PAPER_TARGET_EFFECT = 0.152  # 原文表 4 列(1) PC 系数（%/单位），REV2 功效分析参照效应

SEED = 20261021

NW = 6

NW_PAPER = 12

