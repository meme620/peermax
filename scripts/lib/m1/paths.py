"""s3_m1_lib 的 paths 实现（逐字搬移，逻辑未改）。"""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data_audit'))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data'))


from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]

REGISTRY = ROOT / 'configs/parameter_registry.yaml'

MANIFEST = ROOT / 'configs/version_manifest.yaml'

SPEC = ROOT / 'configs/s3a_final_spec_v1.yaml'

PANELS = {
    'ma_teacher': ROOT / 'data/processed/monthly_pricing_panel_ma_teacher.csv',
    'mb_teacher': ROOT / 'data/processed/monthly_pricing_panel_mb_teacher.csv',
    'ma': ROOT / 'data/processed/monthly_pricing_panel_ma.csv',
    'mb': ROOT / 'data/processed/monthly_pricing_panel_mb.csv',
}

PANEL_PATHS_V4 = {
    'ma_teacher': ROOT / 'data/processed/monthly_pricing_panel_ma_teacher_v4.csv',
    'mb_teacher': ROOT / 'data/processed/monthly_pricing_panel_mb_teacher_v4.csv',
    'ma': ROOT / 'data/processed/monthly_pricing_panel_ma_v4.csv',
    'mb': ROOT / 'data/processed/monthly_pricing_panel_mb_v4.csv',
}

NET_DIRS = {
    'ma_teacher': ROOT / 'data/processed/network_ma_teacher',
    'mb_teacher': ROOT / 'data/processed/network_mb_teacher',
    'ma': ROOT / 'data/processed/network_ma',
    'mb': ROOT / 'data/processed/network_mb',
}

RETURNS_V2 = ROOT / 'data/processed/monthly_returns_clean_v2.csv'

RETURNS_POSTADJ = ROOT / 'data/processed/monthly_returns_adj_full_2006_2025.csv'

SRC_MONTH = ROOT / 'data/processed/source_month_argmax_panel.csv'

ME_PANEL = ROOT / 'data/processed/market_equity_panel.csv'

CHARS = ROOT / 'data/processed/firm_chars_clean.csv'

ANALYST_DERIVED = ROOT / 'data/processed/analyst/firm_analyst_derived.csv'

SW2 = ROOT / 'data/processed/sw2_industry_latest.csv'

MDA_PANEL = ROOT / 'data/processed/teacher_mda_panel.csv'

AMOUNT_PKL = ROOT / 'data/raw/drive/daily_total_amount_20260630.pkl'

TURNOVER_PKL = ROOT / 'data/raw/drive/daily_turnover_rate_20260630.pkl'

ST_PKL = ROOT / 'data/raw/drive/stock_st_flag_20260630.pkl'

OUT_DIR = ROOT / 'reports/data_gates/G0'

SAMPLE_DIR = ROOT / 'data/processed'

PROGRESS = ROOT / 'artifacts/tmp/phaseA_progress.md'

ILLIQ_FORMULA_ID = 'ILLIQ_MONTHLY_ABS_RET_OVER_AMOUNT_v1'

AMOUNT_CACHE_NAME = 'm1_cache_amount_100m.parquet'

AMOUNT_CACHE_META_NAME = 'm1_cache_amount_100m.meta.json'

LEGACY_AMOUNT_CACHE_NAME = 'm1_cache_illiq.parquet'

AMOUNT_CACHE_COLUMNS = ['firm_id', 'month', 'amount_100m']

AMOUNT_CACHE_DIR = ROOT / 'artifacts/tmp'

AMOUNT_CACHE_META_REQUIRED = (
    'cache_name', 'liquidity_formula_id', 'formula', 'amount_100m_unit',
    'source_path', 'source_sha256', 'row_count', 'columns', 'dtypes',
    'script_path', 'script_sha256', 'created_at_utc',
)

M1_SAMPLE_VERSION_V1 = 'm1_s5_v1_2015_2023'

M1_SAMPLE_VERSION_V2 = 'm1_s5_v2_2015_2023'

M1_SAMPLE_VERSION_V3 = 'm1_s5_v3_20261115'

M1_SAMPLE_VERSION_V4 = 'm1_s5_v4_20261116'

M1_SAMPLE_VERSION_V4_RENORM = 'm1_s5_v4_renorm_20261116'

M1_SAMPLE_VERSION = M1_SAMPLE_VERSION_V3

_MONTH_RE = re.compile(r'^\d{4}-(0[1-9]|1[0-2])$')

IQR_POOL_RULE_ID = 'IQR_POOL_TREATMENT_ONLY_v1'

IQR_POOL_REGISTERED = 0.02341287

IQR_POOL_REGISTERED_SOURCE = ('configs/parameter_registry.yaml:'
                              'm1_sample_and_spec_registry.iqr_scale_rule_id'
                              '（规则 ID；数值见 change_log 2026-10-10 正文，无独立数值参数）')

SAMPLE_MONTHS = None  # 由 months_between 生成

M1_INPUTS = {
    'monthly_returns_clean_v2': 'data/processed/monthly_returns_clean_v2.csv',
    'monthly_returns_adj_full_2006_2025': 'data/processed/monthly_returns_adj_full_2006_2025.csv',
    'source_month_argmax_panel': 'data/processed/source_month_argmax_panel.csv',
    'market_equity_panel_full_2006_2025': 'data/processed/market_equity_panel_full_2006_2025.csv',
    'sw2_industry_latest': 'data/processed/sw2_industry_latest.csv',
    'firm_analyst_derived': 'data/processed/analyst/firm_analyst_derived.csv',
    'firm_chars_clean': 'data/processed/firm_chars_clean.csv',
    'firm_id_map': 'data/processed/firm_id_map.csv',
    'market_equity_panel': 'data/processed/market_equity_panel.csv',
}

_SHA256_HEX = re.compile(r'^[0-9a-f]{64}$')

HARD_GATE_KEYS = ('rows_match_registered', 'pit_violations_strict_lt',
                  'duplicate_focal_month_keys', 'missing_filled_with_zero')

ROWS_NOT_APPLICABLE = 'not_applicable_custom_panel'

_M1_RETURN_COLS = ('PeerMAXraw', 'OwnMAXraw', 'PeerReturn')

_SAMPLE_EXCLUSION_VALUE_COL = {'microcap_q10_exclusion': 'market_equity',
                               'no_network_support': 'PeerMAXraw',
                               'own_max_missing': 'OwnMAXraw',
                               'peer_return_missing': 'PeerReturn',
                               'no_outcome_next_month': 'R_next'}

PANEL_COLS = ['focal_id', 'month', 'network_use_case', 'network_version', 'available_at',
              'PeerMAXraw', 'OwnMAXraw', 'PeerReturn', 'n_sources', 'n_max_sources',
              'n_ret_sources', 'support_mass', 'covered_mass_max', 'covered_mass_ret',
              'coverage_ratio_max', 'coverage_ratio_ret', 'limit_hit_weight_share',
              'network_support_status', 'data_status']

PANEL_COLS_V4 = ['focal_id', 'month', 'network_use_case', 'network_version',
                 'network_version_label', 'corpus_id', 'panel_version', 'available_at',
                 'PeerMAXraw', 'OwnMAXraw', 'PeerReturn', 'PeerMAXraw_partial',
                 'PeerReturn_partial', 'n_sources', 'n_max_sources', 'n_ret_sources',
                 'n_max_incomplete', 'n_ret_incomplete', 'n_max_missing_statistic',
                 'n_max_excluded_limit_unknown', 'n_ret_missing_statistic',
                 'n_ret_zero_trading_days', 'support_mass', 'covered_mass_max',
                 'covered_mass_ret', 'incomplete_mass_max', 'incomplete_mass_ret',
                 'missing_mass_max', 'missing_mass_ret', 'coverage_ratio_max',
                 'coverage_ratio_ret', 'limit_hit_weight_share', 'support_status_max',
                 'support_status_ret', 'missing_reason_max', 'missing_reason_ret',
                 'network_support_status', 'data_status', 'coverage_flag',
                 'missing_reason', 'formula_id_max', 'formula_id_ret',
                 'source_version', 'PeerMAXraw_renorm', 'formula_id_max_renorm']

