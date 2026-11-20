"""pricing月度定价共享库（Phase A / C2，2026-10-10）。

只提供纯计算与装载工具；不含任何 outcome 相关的口径选择（口径一律来自 registry：
alpha_pricing / K_Econ_Pricing / K_Spread_Pricing / K_Consist_Pricing / nw_lag / winsorization_pct /
portfolio_breakpoints / random_seed / sample_start / sample_end / sample_and_spec_registry /
pricing_formula_ids）。

关键约定（先登记后跑，见 configs/parameter_registry.yaml）：
  - PIT：max{available_at < month}（面板既有映射，严格小于）
  - outcome：紧邻下一自然月 R，complete_month=1（n_valid_days>=15），不跨缺失月
  - winsor：winsorization_pct 月度截面（收益类变量；当前 registry [0.01,0.99]）
  - 样本窗口：sample_start/sample_end 截月（当前 registry 2015-01/2023-12）
  - FM：逐月 OLS/WLS -> 系数序列 -> 均值 + Newey-West(lag=6, Bartlett) SE
  - 单侧 p：H0 beta>=0 vs H1 beta<0（主方向 beta<0）时 p = Phi(t)
  - TOST：d = beta_EW - beta_VW，等价界 = +/- K_Consist_Pricing，alpha = alpha_pricing"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd


from pathlib import Path
import re
from fileio import file_size, read_pickle

ROOT = Path(__file__).resolve().parents[1]

REGISTRY = ROOT / 'configs/parameter_registry.yaml'

MANIFEST = ROOT / 'configs/version_manifest.yaml'

SPEC = ROOT / 'configs/analyst_spec.yaml'
CONTRACTS = ROOT / 'configs/pricing_inputs.yaml'

# 样本两列口径：只有老师 MD&A 语料（ma_teacher / mb_teacher）。自建 CNINFO 语料两列
# （ma / mb）按 CO-2026-11-20-teacher-only 退役——网络目录、面板与语料提取链都不再属于本仓
# 计算链；键位保留即会让 check_panel_contracts 遍历到不存在的文件而抛 FileNotFoundError。
PANELS = {
    'ma_teacher': ROOT / 'data/processed/monthly_pricing_panel_ma_teacher.csv',
    'mb_teacher': ROOT / 'data/processed/monthly_pricing_panel_mb_teacher.csv',
}

PANEL_PATHS_V4 = {
    'ma_teacher': ROOT / 'data/processed/monthly_pricing_panel_ma_teacher_v4.csv',
    'mb_teacher': ROOT / 'data/processed/monthly_pricing_panel_mb_teacher_v4.csv',
}

NET_DIRS = {
    'ma_teacher': ROOT / 'data/processed/network_ma_teacher',
    'mb_teacher': ROOT / 'data/processed/network_mb_teacher',
}

RETURNS_V2 = ROOT / 'data/processed/monthly_returns_clean_v2.csv'

RETURNS_RAW = ROOT / 'data/processed/monthly_returns_clean_raw.csv'

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

AMOUNT_CACHE_NAME = 'pricing_cache_amount_100m.parquet'

AMOUNT_CACHE_META_NAME = 'pricing_cache_amount_100m.meta.json'

LEGACY_AMOUNT_CACHE_NAME = 'pricing_cache_illiq.parquet'

AMOUNT_CACHE_COLUMNS = ['firm_id', 'month', 'amount_100m']

AMOUNT_CACHE_DIR = ROOT / 'artifacts/tmp'

AMOUNT_CACHE_META_REQUIRED = (
    'cache_name', 'liquidity_formula_id', 'formula', 'amount_100m_unit',
    'source_path', 'source_bytes', 'row_count', 'columns', 'dtypes',
    'script_path', 'script_bytes', 'created_at_utc',
)

# 样本版本一律不可变（spec/design.yaml:immutable_versions 登记 v1..v5）：v1–v4 保留为
# 已落盘 parquet 的读取标识与 v1/v2 敏感性路径，v5 是当前 headline（只有老师 MD&A 两列）。
SAMPLE_VERSION_V1 = 'pricing_sample_v1_2015_2023'
SAMPLE_VERSION_V2 = 'pricing_sample_v2_2015_2023'
SAMPLE_VERSION_V3 = 'pricing_sample_v3_20261115'
SAMPLE_VERSION_V4 = 'pricing_sample_v4_20261116'
SAMPLE_VERSION_V5 = 'pricing_sample_v5_teacher_only_20261120'

SAMPLE_VERSION = SAMPLE_VERSION_V5

_MONTH_RE = re.compile(r'^\d{4}-(0[1-9]|1[0-2])$')

IQR_POOL_RULE_ID = 'IQR_POOL_TREATMENT_ONLY_v1'

IQR_POOL_REGISTERED = 0.01693548

IQR_POOL_REGISTERED_SOURCE = ('configs/parameter_registry.yaml:'
                              'sample_and_spec_registry.iqr_scale_rule_id'
                              '（规则 ID；数值见 change_log 2026-10-10 正文，无独立数值参数）')

SAMPLE_MONTHS = None  # 由 months_between 生成

PRICING_INPUTS = {
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


HARD_GATE_KEYS = ('rows_match_registered', 'pit_violations_strict_lt',
                  'duplicate_focal_month_keys', 'missing_filled_with_zero')

ROWS_NOT_APPLICABLE = 'not_applicable_custom_panel'

_PRICING_RETURN_COLS = ('PeerMAXraw', 'OwnMAXraw', 'PeerReturn')

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
from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
import os
import pandas as pd
import sys

# 可解析的样本版本：headline v5 + shadow 仍登记的 legacy v2
# （退役条件见 registry 的 shadow_legacy_sample_version）。
_SAMPLE_SUFFIX = {'v5': 'v5', SAMPLE_VERSION_V5: 'v5',
                  'v2': 'v2', 'pricing_sample_v2_2015_2023': 'v2'}


def sample_path(key: str, version: str | None = None):
    """样本件路径。缺省 = headline；显式版本只接受 headline 与 shadow legacy，越界即报错。"""
    if version is None:
        version = str(param_value('sample_and_spec_registry')['sample_version'])
    suffix = _SAMPLE_SUFFIX.get(version)
    if suffix is None:
        raise SystemExit(f'[STOP] 未知样本版本 {version!r}；可解析 {sorted(set(_SAMPLE_SUFFIX))}')
    return SAMPLE_DIR / f'pricing_sample_{key}_{suffix}.parquet'

def progress(msg: str) -> None:
    print(f'[A] {msg}', flush=True)
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, 'a', encoding='utf-8') as f:
        f.write(f'- {msg}\n')

def write_json(path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, p)
    print(f'  -> {p.relative_to(ROOT)}', flush=True)

def load_panel(key: str, version: str = 'v3') -> pd.DataFrame:
    """v3（默认）= 现行面板；v4 = W1 新面板（新文件名，不覆盖 v3）。"""
    path = PANEL_PATHS_V4[key] if version in ('v4', 'panel_v4') else PANELS[key]
    df = pd.read_csv(path, dtype={'focal_id': str, 'month': str})
    return df

def code6(firm_id: str) -> str:
    s = str(firm_id)
    return s[2:] if s[:2] in ('sz', 'sh') else s

def load_returns_v2() -> pd.DataFrame:
    """主口径月收益（后复权）+ 交易日月质量列；R 与 monthly_returns_clean.csv 逐值一致。"""
    d = pd.read_csv(RETURNS_V2, usecols=['firm_id', 'month', 'R', 'n_valid_days',
                                         'complete_month', 'return_status'],
                    dtype={'firm_id': str, 'month': str})
    return d

def load_returns_raw() -> pd.DataFrame:
    """未复权月收益（raw close 环比），仅作 outcome 口径敏感性。

    与主口径 RETURNS_V2 不是同一条序列：主口径含公司行为调整，本序列除权日断崖保留。
    """
    d = pd.read_csv(RETURNS_RAW, usecols=['firm_id', 'month', 'R'],
                    dtype={'firm_id': str, 'month': str})
    return d.dropna(subset=['R'])

def load_postadj() -> pd.DataFrame:
    d = pd.read_csv(RETURNS_POSTADJ, dtype={'firm_id': str, 'month': str})
    return d

def load_src_month(cols=None) -> pd.DataFrame:
    cols = cols or ['source_id', 'month', 'MAXraw', 'R', 'qualification_status',
                    'limit_hit_at_argmax', 'st_flag_at_argmax', 'n_valid_days',
                    'complete_month', 'max_status', 'return_status',
                    'source_month_status', 'missing_reason', 'headline_max_eligible']
    return pd.read_csv(SRC_MONTH, usecols=cols, dtype={'source_id': str, 'month': str})

def load_me() -> pd.DataFrame:
    d = pd.read_csv(ME_PANEL, dtype={'firm_id': str})
    d['firm_id'] = d['firm_id'].str.zfill(6)
    d['month'] = pd.to_datetime(d['date']).dt.strftime('%Y-%m')
    return d[['firm_id', 'month', 'market_equity']]

def load_sw2() -> pd.DataFrame:
    s = pd.read_csv(SW2, dtype={'firm_id': str, 'l1_code': str, 'l2_code': str})
    s = s[s['firm_id'].notna() & (s['is_current_version'] == True)]  # noqa: E712
    s = s.drop_duplicates('firm_id')
    return s[['firm_id', 'l1_code', 'l2_code']]

def load_st() -> pd.DataFrame:
    import warnings
    warnings.filterwarnings('ignore')
    d = read_pickle(ST_PKL)
    d.index = pd.to_datetime(d.index)
    d = d.loc['2014-01-01':'2024-01-31']
    me = d.resample('ME').max()          # 当月是否曾 ST（保守：曾 ST 即标 1）
    me.index = [c.strftime('%Y-%m') for c in me.index]
    long = me.stack().rename('st_flag').reset_index()
    long.columns = ['month', 'code6', 'st_flag']
    # 2026-10-10 I6 缺陷修复：Drive 列名为 '000002.SZ' 形式，此前 .str.lower() 得到 '000002.sz'
    # 与面板 focal_id（'sz000002'）不匹配 → st_flag_focal 恒为 False（ST 分层结构性为空）。
    # 改用 drive_code_to_firm_id（'000002.SZ' -> 'sz000002'）。
    long['firm_id'] = long['code6'].map(drive_code_to_firm_id)
    return long[['firm_id', 'month', 'st_flag']]

def drive_code_to_firm_id(code: str) -> str:
    """'000002.SZ' / 'sz000002' / '000002' -> 'sz000002'。"""
    s = str(code).lower()
    if '.' in s:
        num, ex = s.split('.')
        return ex + num
    if s[:2] in ('sz', 'sh'):
        return s
    return ('sh' if s[:1] == '6' else 'sz') + s

def _drive_monthly_wide(pkl, how: str) -> pd.DataFrame:
    """Drive 日频 pkl -> 月频长表（firm_id/month/value，2014-06..2024-01 切片）。

    how 分支显式（Phase 4 BL-03 修复；此前 `how != 'sum'` 全部静默落 `.last()`）：
      - ``'sum'``  : ``resample('ME').sum(min_count=1)``（保持历史口径）
      - ``'last'`` : ``resample('ME').last()``（保持历史口径）
      - ``'mean'`` : ``groupby(月度期).mean()``——部分 NaN 月 = 非 NaN 日均值；
        全 NaN 月在**重采样宽表层**为 NaN（其后 ``.stack()`` 默认 dropna，长表不含该月）
      - 其他      : ``raise ValueError(f"unsupported how={how!r}")``（禁止静默落 `.last()`）
    注意：不得写 ``resample('ME').mean(min_count=1)``——``DatetimeIndexResampler.mean()``
    不接受 ``min_count``，pandas 2.3.1 抛 ``pandas.errors.UnsupportedFunctionCall``。
    """
    import warnings
    warnings.filterwarnings('ignore')
    d = read_pickle(pkl)
    d.index = pd.to_datetime(d.index)
    d = d.loc['2014-06-01':'2024-01-31']
    if how == 'sum':
        m = d.resample('ME').sum(min_count=1)
    elif how == 'last':
        m = d.resample('ME').last()
    elif how == 'mean':
        m = d.groupby(d.index.to_period('M')).mean()
    else:
        raise ValueError(f"unsupported how={how!r}")
    m.index = [c.strftime('%Y-%m') for c in m.index]
    long = m.stack().rename('value').reset_index()
    long.columns = ['month', 'code', 'value']
    long['firm_id'] = long['code'].map(drive_code_to_firm_id)
    return long[['firm_id', 'month', 'value']]

def load_amount_100m(months=None) -> pd.DataFrame:
    """成交额源表（缓存名与内容一致：只有 amount_100m，不含 illiq）。

    Amount = Drive daily_total_amount(千元) 当月求和 / 1e5，单位亿元。
    下游事件月实际流动性公式（不改动、不新增）：illiq = |own_R_t| / amount_100m，
    liquidity_formula_id = ILLIQ_MONTHLY_ABS_RET_OVER_AMOUNT_v1。
    注：既有登记名 ILLIQ_MONTHLY_AMIHUD_CNY100M_v1 保持不变；本函数只澄清实际口径，
    不做经典 Amihud 的日频比值/年均步骤。
    """
    long = _drive_monthly_wide(AMOUNT_PKL, 'sum')
    long['amount_100m'] = long['value'] / 1e5
    return long[['firm_id', 'month', 'amount_100m']]

def load_illiq(months=None) -> pd.DataFrame:
    """Deprecated 兼容别名：历史调用方仍可 import load_illiq；返回列/数值保持不变。"""
    return load_amount_100m(months=months)

def _rel_path(path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)

def _cache_fail(msg: str):
    print(f'[CACHE] {msg}', file=sys.stderr, flush=True)
    raise SystemExit(2)

def _check_amount_frame(df: pd.DataFrame, where: str) -> None:
    if list(df.columns) != AMOUNT_CACHE_COLUMNS:
        _cache_fail(f'{where} 列不符：期望 {AMOUNT_CACHE_COLUMNS}，实际 {list(df.columns)}')

def _write_json_atomic(path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, p)

def _write_parquet_atomic(df: pd.DataFrame, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    df.to_parquet(tmp, index=False)
    os.replace(tmp, p)

def amount_cache_meta(df: pd.DataFrame, source_path=None, script_path=None) -> dict:
    """缓存 sidecar：输入路径/sha、公式、单位、行数、dtype、脚本身份、生成时间。"""
    src = Path(source_path) if source_path is not None else AMOUNT_PKL
    scr = Path(script_path) if script_path is not None else Path(__file__)
    if not src.is_file():
        _cache_fail(f'缓存输入不存在，拒绝生成指纹: {src}')
    _check_amount_frame(df, '缓存数据')
    return {
        'cache_name': AMOUNT_CACHE_NAME,
        'liquidity_formula_id': ILLIQ_FORMULA_ID,
        'formula': 'illiq = |R_t| / amount_100m',
        'amount_100m_unit': '亿元（100 million CNY）；amount_100m = 当月 Σ daily_total_amount(千元) / 1e5',
        'source_path': _rel_path(src),
        'source_bytes': file_size(src),
        'row_count': int(len(df)),
        'columns': list(df.columns),
        'dtypes': {c: str(t) for c, t in df.dtypes.items()},
        'script_path': _rel_path(scr),
        'script_bytes': file_size(scr),
        'created_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }

def write_amount_cache(df: pd.DataFrame, cache_path=None, meta_path=None,
                       source_path=None, script_path=None) -> dict:
    """原子写 amount 缓存 + sidecar（先 parquet 后 meta，避免 meta 指向旧内容）。"""
    cache_path = (Path(cache_path) if cache_path is not None
                  else AMOUNT_CACHE_DIR / AMOUNT_CACHE_NAME)
    meta_path = (Path(meta_path) if meta_path is not None
                 else cache_path.with_name(AMOUNT_CACHE_META_NAME))
    _check_amount_frame(df, '缓存数据')
    meta = amount_cache_meta(df, source_path=source_path, script_path=script_path)
    _write_parquet_atomic(df, cache_path)
    _write_json_atomic(meta_path, meta)
    return meta

def _read_amount_cache_meta(meta_path) -> dict:
    p = Path(meta_path)
    try:
        meta = json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError) as e:
        _cache_fail(f'sidecar 元数据不可读: {p} ({e})')
    if not isinstance(meta, dict):
        _cache_fail(f'sidecar 元数据不是 JSON 对象: {p}')
    missing = [k for k in AMOUNT_CACHE_META_REQUIRED if k not in meta]
    if missing:
        _cache_fail(f'sidecar 元数据缺字段 {missing}: {p}')
    return meta

def _cache_contract_problems(meta: dict, df: pd.DataFrame, source_bytes: str) -> list:
    problems = []
    if meta.get('source_bytes') != source_bytes:
        problems.append(f'source_bytes sidecar={meta.get("source_bytes")} != 当前输入={source_bytes}')
    if meta.get('liquidity_formula_id') != ILLIQ_FORMULA_ID:
        problems.append(f'liquidity_formula_id sidecar={meta.get("liquidity_formula_id")!r} '
                        f'!= {ILLIQ_FORMULA_ID!r}')
    try:
        meta_rows = int(meta.get('row_count', -1))
    except (TypeError, ValueError):
        meta_rows = -1
    if meta_rows != len(df):
        problems.append(f'row_count sidecar={meta.get("row_count")} != 缓存实际={len(df)}')
    if meta.get('columns') != list(df.columns):
        problems.append(f'columns sidecar={meta.get("columns")} != 缓存实际={list(df.columns)}')
    return problems

def load_amount_100m_cached(cache_dir=None, cache_path=None, meta_path=None,
                            legacy_path=None, source_path=None, recompute=None,
                            rebuild=False):
    """读取/迁移/重建 amount_100m 缓存，返回 (df, meta)。

    明确策略（不静默复用）：
      - 新缓存 + 有效 sidecar + source_bytes/行数/列一致 → 命中，不重算；
      - 新缓存存在但 sidecar 缺失/不可读 → 从 source 重建并写 sidecar；
      - sidecar 存在但 source_bytes / formula id / 行数 / 列不匹配 → 默认 SystemExit(2)；
        只有显式 `rebuild=True` 才重算覆盖（输入漂移不得静默沿用旧值）；
      - 新缓存不存在、旧 `pricing_cache_illiq.parquet` 存在且列正确 → 一次性只读迁移
        （读旧值原样写新缓存+sidecar，旧文件保留）；
      - 两者都不存在 → 从 source 计算并落盘。
    `recompute`/`source_path` 仅测试注入；生产默认 `load_amount_100m` + AMOUNT_PKL。
    """
    if cache_path is None:
        base = Path(cache_dir) if cache_dir is not None else AMOUNT_CACHE_DIR
        cache_path = base / AMOUNT_CACHE_NAME
    cache_path = Path(cache_path)
    meta_path = (Path(meta_path) if meta_path is not None
                 else cache_path.with_name(AMOUNT_CACHE_META_NAME))
    legacy_path = (Path(legacy_path) if legacy_path is not None
                   else cache_path.with_name(LEGACY_AMOUNT_CACHE_NAME))
    src = Path(source_path) if source_path is not None else AMOUNT_PKL
    if recompute is None:
        recompute = load_amount_100m
    if not src.is_file():
        _cache_fail(f'缓存输入不存在，无法核验/生成指纹: {src}')

    def _rebuild(status: str, reason: str):
        print(f'[CACHE] {status} reason={reason} -> 重算并写 {cache_path.name}',
              file=sys.stderr, flush=True)
        df = recompute()
        _check_amount_frame(df, '重建缓存')
        meta = write_amount_cache(df, cache_path, meta_path, source_path=src)
        out = dict(meta)
        out['cache_status'] = status
        return df, out

    if cache_path.exists():
        if not meta_path.exists():
            return _rebuild('rebuilt_missing_meta', f'缺少 {meta_path.name}')
        meta = _read_amount_cache_meta(meta_path)
        df = pd.read_parquet(cache_path)
        _check_amount_frame(df, f'新缓存 {cache_path.name}')
        problems = _cache_contract_problems(meta, df, file_size(src))
        if problems:
            if not rebuild:
                _cache_fail('缓存指纹/契约不匹配，拒绝复用：' + '；'.join(problems) +
                            '（显式重建请传 rebuild=True，或删除缓存/sidecar 后重跑）')
            return _rebuild('rebuilt_cache_contract_mismatch', '; '.join(problems))
        out = dict(meta)
        out['cache_status'] = 'hit'
        print(f'[CACHE] amount_cache_hit rows={len(df):,} source_bytes={meta["source_bytes"]}',
              file=sys.stderr, flush=True)
        return df, out

    if legacy_path.exists():
        df = pd.read_parquet(legacy_path)
        _check_amount_frame(df, f'旧缓存 {legacy_path.name}')
        meta = write_amount_cache(df, cache_path, meta_path, source_path=src)
        out = dict(meta)
        out['cache_status'] = 'legacy_migrated'
        print(f'[CACHE] legacy_cache_migrated old={legacy_path.name} new={cache_path.name} '
              f'rows={len(df):,}（旧文件保留，数值原样迁移）', file=sys.stderr, flush=True)
        return df, out

    return _rebuild('created', '无缓存')

def load_turnover_drive() -> pd.DataFrame:
    """Turnover = Drive daily_turnover_rate(官方 %) 月末值 / 100（与 analyst-network 9C 同口径）。

    登记原因：firm_chars_clean.turnover 在 2015–2016 覆盖率为 0（仅 2017 起可得），
    故主控制集的 Turnover 改读已登记 spec 输入 Drive daily_turnover_rate。
    """
    long = _drive_monthly_wide(TURNOVER_PKL, 'last')
    long['turnover'] = long['value'] / 100.0
    return long[['firm_id', 'month', 'turnover']]

def load_chars() -> pd.DataFrame:
    d = pd.read_csv(CHARS, usecols=['firm_id', 'month', 'size', 'bm', 'mom', 'rev',
                                    'turnover', 'missing_reason'],
                    dtype={'firm_id': str, 'month': str})
    d['firm_id'] = d['firm_id'].str.zfill(6)
    return d

def load_analyst_derived() -> pd.DataFrame:
    d = pd.read_csv(ANALYST_DERIVED, usecols=['firm_id', 'month', 'AnalystCoverage',
                                              'NetworkDegree', 'EffectivePeerCount'],
                    dtype={'firm_id': str})
    d['month'] = pd.to_datetime(d['month']).dt.strftime('%Y-%m')
    d['firm_id'] = d['firm_id'].str.zfill(6)
    return d

def load_edges(net_key: str, text_year: int) -> pd.DataFrame:
    p = NET_DIRS[net_key] / f'{text_year}_similarity.csv'
    e = pd.read_csv(p, usecols=['focal_id', 'source_id', 'similarity', 'weight'],
                    dtype={'focal_id': str, 'source_id': str})
    return e

def load_support(net_key: str, text_year: int) -> pd.DataFrame:
    p = NET_DIRS[net_key] / f'{text_year}_support.csv'
    return pd.read_csv(p, dtype={'focal_id': str})

def add_weight_rules(e: pd.DataFrame) -> pd.DataFrame:
    """one_sided_significance.2 三套权重（固定边集，只换权重；行内归一化）。"""
    e = e.copy()
    g = e.groupby('focal_id', sort=False)['similarity']
    s_sum = g.transform('sum')
    n_i = g.transform('size')
    l = np.log1p(e['similarity'].to_numpy(dtype=float))
    e['_l'] = l
    l_sum = e.groupby('focal_id', sort=False)['_l'].transform('sum')
    e['w_equal'] = 1.0 / n_i
    e['w_linear'] = np.where(s_sum > 0, e['similarity'] / s_sum, np.nan)
    e['w_log_sim'] = np.where(l_sum > 0, e['_l'] / l_sum, np.nan)
    return e.drop(columns=['_l'])

def aggregate_edges(edges: pd.DataFrame, values: pd.DataFrame, wcols, value_col: str,
                    key_col: str = 'firm_id', focal_col: str = 'focal_id',
                    source_col: str = 'source_id', month_col: str = 'month') -> pd.DataFrame:
    """固定边集上的加权平均（覆盖源上重新归一化，缺失不填 0）。

    edges: focal/source/wcols；values: [key_col(=source 身份), month, value_col]
    注意：本函数是历史 A9 权重变体/placebo 使用的 **renormalized** 实现（诊断用途）；
    headline PeerMAXraw/PeerReturn 聚合使用 `aggregate_stat`（未归一化 complete-only，
    07 sign.2-01 A2/A3），两者不得混用。
    """
    m = edges.merge(values.rename(columns={key_col: source_col}), on=source_col, how='left')
    m = m[m[value_col].notna()]
    out = None
    for wc in wcols:
        mm = m[m[wc].notna()].copy()
        mm['_wx'] = mm[wc] * mm[value_col]
        g = mm.groupby([month_col, focal_col], sort=False)
        agg = g.agg(_num=('_wx', 'sum'), _den=(wc, 'sum')).reset_index()
        agg['_v'] = np.where(agg['_den'] > 0, agg['_num'] / agg['_den'], np.nan)
        agg = agg[[month_col, focal_col, '_v']].rename(columns={'_v': f'{value_col}_{wc}'})
        out = agg if out is None else out.merge(agg, on=[month_col, focal_col], how='outer')
    return out

def aggregate_stat(edges: pd.DataFrame, src: pd.DataFrame, stat_col: str,
                   eligible_mask, suffix: str, w_col: str = 'weight',
                   excluded_mask=None, zero_trading_days_mask=None,
                   incomplete_mask=None, extra_weight=None) -> pd.DataFrame:
    """**唯一共享聚合函数**（唯一实现，禁止第二套）。

    语义（07 sign.2-01 A2/A3；plan :4739-4761/:4968）：
      support_mass = sum(w)（全部边）
      covered_mass = sum(w where class=='covered')
      missing_mass = sum(w where class!='covered')  ← 直接累加缺失权重，不用 support-covered 反算
      weighted_value = sum(w*x over covered)，仅当 support>0 且 missing==0.0；否则 NaN
      partial_value  = sum(w*x over covered|incomplete)，只作诊断
      coverage_ratio = covered/support（support=0 → NaN）
      support_status = not_applicable | complete | partial | no_covered_source_{suffix}
    类别优先级：excluded > zero_trading_days > incomplete_month > covered > missing_statistic。
    不允许任何 `/covered`、`/w_max`、`/w_ret` 出现在 headline 返回值路径。

    参数
    ----
    edges : DataFrame，含 `focal_id`,`month`,`w_col`（与 src 行对齐）
    src   : DataFrame，与 edges 等长；含 `stat_col` 与该行掩码
    eligible_mask : complete/合格谓词（与 src 对齐）；covered = eligible & stat.notna() 且未被其他掩码占用
    excluded_mask / zero_trading_days_mask / incomplete_mask : 可选，互斥类
    extra_weight  : 可选，与 edges 对齐的额外权重（只对 covered 行求和；如 MAX 的 limit-hit 权重）

    返回：按 (month, focal_id) 的 DataFrame；列名带 `_{suffix}` 后缀。
    """
    n = len(edges)
    if len(src) != n:
        raise ValueError(f'aggregate_stat: edges({n}) 与 src({len(src)}) 行数不一致')
    if w_col not in edges.columns:
        raise ValueError(f'aggregate_stat: edges 缺权重列 {w_col!r}')
    if stat_col not in src.columns:
        raise ValueError(f'aggregate_stat: src 缺统计量列 {stat_col!r}')

    def _bool(x):
        if x is None:
            return np.zeros(n, dtype=bool)
        s = pd.Series(x)
        if len(s) != n:
            raise ValueError('aggregate_stat: 掩码长度与 edges 不一致')
        return s.fillna(False).astype(bool).to_numpy()

    w_series = pd.to_numeric(pd.Series(edges[w_col]), errors='coerce')
    if w_series.isna().any():
        raise ValueError(f'aggregate_stat: 权重列 {w_col!r} 有 {int(w_series.isna().sum())} 行 NaN/非数值')
    w = w_series.to_numpy(float)
    x = pd.to_numeric(pd.Series(src[stat_col]), errors='coerce').to_numpy(float)
    xok = ~np.isnan(x)
    m_elig = _bool(eligible_mask)
    m_excl = _bool(excluded_mask)
    m_zero = _bool(zero_trading_days_mask)
    m_inc = _bool(incomplete_mask)
    covered = m_elig & xok & ~m_excl & ~m_zero & ~m_inc

    cls = np.full(n, 'missing_statistic', dtype=object)
    cls[m_inc] = 'incomplete_month'
    cls[m_zero] = 'zero_trading_days'
    cls[m_excl] = 'excluded'
    cls[covered] = 'covered'
    is_cov = cls == 'covered'
    is_inc = cls == 'incomplete_month'
    is_zero = cls == 'zero_trading_days'
    is_excl = cls == 'excluded'
    partial_mask = (is_cov | is_inc) & xok

    df = pd.DataFrame({
        'month': pd.Series(edges['month']).to_numpy(),
        'focal_id': pd.Series(edges['focal_id']).to_numpy(),
        '_w': w,
        '_w_cov': np.where(is_cov, w, 0.0),
        '_w_missing': np.where(~is_cov, w, 0.0),
        '_w_inc': np.where(is_inc, w, 0.0),
        '_w_zero': np.where(is_zero, w, 0.0),
        '_w_excl': np.where(is_excl, w, 0.0),
        '_wx_cov': np.where(is_cov, w * np.nan_to_num(x, nan=0.0), 0.0),
        '_wx_part': np.where(partial_mask, w * np.nan_to_num(x, nan=0.0), 0.0),
        '_n_cov': is_cov.astype(np.int64),
        '_n_inc': is_inc.astype(np.int64),
        '_n_zero': is_zero.astype(np.int64),
        '_n_excl': is_excl.astype(np.int64),
        '_n_miss': (cls == 'missing_statistic').astype(np.int64),
        '_n_part': partial_mask.astype(np.int64),
    })
    if extra_weight is None:
        df['_extra'] = 0.0
    else:
        ew = pd.to_numeric(pd.Series(extra_weight), errors='coerce').fillna(0.0).to_numpy(float)
        if len(ew) != n:
            raise ValueError('aggregate_stat: extra_weight 长度与 edges 不一致')
        df['_extra'] = np.where(is_cov, ew, 0.0)

    g = df.groupby(['month', 'focal_id'], sort=False)
    agg = g.agg(
        support_mass=('_w', 'sum'),
        covered_mass=('_w_cov', 'sum'),
        missing_mass=('_w_missing', 'sum'),
        incomplete_mass=('_w_inc', 'sum'),
        excluded_mass=('_w_excl', 'sum'),
        zero_trading_days_mass=('_w_zero', 'sum'),
        weighted_sum=('_wx_cov', 'sum'),
        partial_sum=('_wx_part', 'sum'),
        n_sources=('_w', 'size'),
        n_covered=('_n_cov', 'sum'),
        n_incomplete=('_n_inc', 'sum'),
        n_zero_trading_days=('_n_zero', 'sum'),
        n_excluded=('_n_excl', 'sum'),
        n_missing_statistic=('_n_miss', 'sum'),
        n_partial=('_n_part', 'sum'),
        extra_mass=('_extra', 'sum'),
    ).reset_index()

    sup = agg['support_mass'].to_numpy(float)
    cov = agg['covered_mass'].to_numpy(float)
    miss = agg['missing_mass'].to_numpy(float)
    extra = agg['extra_mass'].to_numpy(float)
    complete = (sup > 0) & (miss == 0.0)
    agg['weighted_value'] = np.where(complete, agg['weighted_sum'], np.nan)
    agg['partial_value'] = np.where(agg['n_partial'] > 0, agg['partial_sum'], np.nan)
    agg['coverage_ratio'] = np.where(sup > 0, cov / np.where(sup > 0, sup, 1.0), np.nan)
    agg['limit_hit_weight_share'] = np.where((sup > 0) & (cov > 0), extra / np.where(cov > 0, cov, 1.0), np.nan)
    status = np.select(
        [sup == 0, complete, cov > 0],
        ['not_applicable', 'complete', 'partial'],
        default=f'no_covered_source_{suffix}' if suffix else 'no_covered_source')
    agg['support_status'] = status

    # missing_reason：按 §0.4B 的 suffix 专用词表（多因 → mixed；空串 = 无缺失）
    if suffix in ('max', 'max_la'):
        candidates = [(agg['n_missing_statistic'] > 0, 'peer_max_not_in_source_panel'),
                      (agg['n_excluded'] > 0, 'limit_status_unknown_excluded'),
                      (agg['n_zero_trading_days'] > 0, 'zero_trading_days')]
    elif suffix == 'ret':
        candidates = [(agg['n_missing_statistic'] > 0, 'no_return_record'),
                      (agg['n_zero_trading_days'] > 0, 'zero_trading_days'),
                      (agg['n_incomplete'] > 0, 'incomplete_month')]
    else:
        candidates = [(agg['n_missing_statistic'] > 0, 'missing_statistic'),
                      (agg['n_excluded'] > 0, 'excluded'),
                      (agg['n_zero_trading_days'] > 0, 'zero_trading_days'),
                      (agg['n_incomplete'] > 0, 'incomplete_month')]
    lab = np.full(len(agg), '', dtype=object)
    multi = np.zeros(len(agg), dtype=bool)
    for mask, label in candidates:
        mask = mask.to_numpy(bool)
        conflict = mask & (lab != '') & (lab != label)
        lab = np.where(mask & (lab == ''), label, lab)
        multi |= conflict
    agg['missing_reason'] = np.where(miss > 0, np.where(multi, 'mixed', lab), '')

    rename = {}
    for base in ('support_mass', 'covered_mass', 'missing_mass', 'incomplete_mass',
                 'zero_trading_days_mass',
                 'n_sources', 'n_covered', 'n_incomplete', 'n_zero_trading_days',
                 'n_excluded', 'n_missing_statistic', 'weighted_value', 'partial_value',
                 'coverage_ratio', 'support_status', 'missing_reason', 'extra_mass',
                 'limit_hit_weight_share'):
        rename[base] = f'{base}_{suffix}'
    agg = agg.rename(columns=rename)
    keep = (['month', 'focal_id'] + [f'{c}_{suffix}' for c in (
        'support_mass', 'covered_mass', 'missing_mass', 'incomplete_mass',
        'zero_trading_days_mass',
        'n_sources', 'n_covered', 'n_incomplete', 'n_zero_trading_days', 'n_excluded',
        'n_missing_statistic', 'weighted_value', 'partial_value', 'coverage_ratio',
        'support_status', 'missing_reason', 'extra_mass', 'limit_hit_weight_share')])
    return agg[keep]
from pathlib import Path
import json
import numpy as np
import pandas as pd
import sys

def _registry_fail(msg: str):
    print(f'[REGISTRY] {msg}', file=sys.stderr, flush=True)
    raise SystemExit(2)

def _read_registry_doc(registry_path=None) -> dict:
    """只读 registry YAML；不做任何写回。"""
    import yaml
    p = Path(registry_path) if registry_path is not None else REGISTRY
    if not p.is_file():
        _registry_fail(f'registry 不存在: {p}')
    doc = yaml.safe_load(p.read_text(encoding='utf-8'))
    if not isinstance(doc, dict) or not isinstance(doc.get('parameters'), list):
        _registry_fail(f'registry 结构非法（缺 parameters 列表）: {p}')
    return doc

def _registry_param(name: str, registry_path=None):
    """按 name 取唯一参数的 headline_value；命中数 != 1 / 缺 headline_value → SystemExit(2)。"""
    plist = _read_registry_doc(registry_path)['parameters']
    hits = [p for p in plist if isinstance(p, dict) and p.get('name') == name]
    if len(hits) != 1:
        _registry_fail(f'参数 {name!r} 命中 {len(hits)} 条 != 1；拒绝猜测/回退')
    if 'headline_value' not in hits[0]:
        _registry_fail(f'参数 {name!r} 缺 headline_value')
    return hits[0]['headline_value']

def _registry_month(name: str, registry_path=None) -> str:
    raw = str(_registry_param(name, registry_path)).strip()
    if not _MONTH_RE.fullmatch(raw[:7]) or (len(raw) > 7 and raw[7] not in '-/'):
        _registry_fail(f'{name} 不是 YYYY-MM[-DD] 格式: {raw!r}')
    return raw[:7]

def sample_window_bounds(registry_path=None) -> tuple:
    """registry sample_start/sample_end -> ('YYYY-MM', 'YYYY-MM')；start<=end，否则 SystemExit(2)。"""
    start = _registry_month('sample_start', registry_path)
    end = _registry_month('sample_end', registry_path)
    if start > end:
        _registry_fail(f'sample_start {start} > sample_end {end}')
    return start, end

def load_registry(registry_path=None) -> dict:
    return _read_registry_doc(registry_path)

def params(registry_path=None) -> dict:
    reg = load_registry(registry_path)
    return {p['name']: p for p in reg['parameters']}

def param_value(name: str, registry_path=None):
    """registry headline_value；registry_path 仅测试注入，生产用 configs 真源。"""
    return _registry_param(name, registry_path)

def require(cond, msg: str) -> bool:
    if not bool(cond):
        print(f'[HARD-GATE] {msg}', file=sys.stderr, flush=True)
        raise SystemExit(2)
    return True

def manifest_input_key(path) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()

def load_manifest_entries(manifest_path=None) -> list:
    import yaml
    p = Path(manifest_path) if manifest_path is not None else MANIFEST
    require(p.is_file(), f'manifest 不存在: {p}')
    doc = yaml.safe_load(p.read_text(encoding='utf-8'))
    require(isinstance(doc, dict) and isinstance(doc.get('deliverables'), list),
            f'manifest 结构非法: {p}')
    return doc['deliverables']

def load_input_contracts(contracts_path=None) -> dict:
    import yaml
    p = Path(contracts_path) if contracts_path is not None else CONTRACTS
    require(p.is_file(), f'输入合同不存在: {p}')
    doc = yaml.safe_load(p.read_text(encoding='utf-8'))
    entries = doc.get('pricing_inputs') if isinstance(doc, dict) else None
    require(isinstance(entries, dict) and bool(entries), f'输入合同结构非法: {p}')
    return entries

def file_shape(path):
    p = Path(path)
    ext = p.suffix.lower()
    if ext == '.csv':
        header = _csv_header(p)
        return _csv_row_count(p), header
    if ext == '.parquet':
        import pyarrow.parquet as pq
        meta = pq.ParquetFile(p)
        return meta.metadata.num_rows, list(meta.schema.names)
    raise SystemExit(f'[STOP] 未登记的输入格式: {p}')

def _csv_header(path) -> list:
    """首行表头（只读一行，不扫全表）。"""
    import csv as _csv
    with open(path, encoding='utf-8', newline='') as f:
        return next(_csv.reader(f), [])


def _csv_row_count(path) -> int:
    """数据行数：换行数即数据行数，仅当换行落在引号内时回退 csv.reader。

    输入合同每次核对行数。换行计数在 763MB 面板上是 0.04s，csv.reader 逐行解析要 0.8s
    且随文件线性增长。只有「引号内出现换行」（跨行字段）才会让换行计数失真，判据因此是
    「是否存在被引号包住、且内含换行的字段」，不是「文件里出现过引号」。
    """
    import csv as _csv
    n = 0
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(1 << 22)
            if not chunk:
                break
            n += chunk.count(b'\n')
    if n and not _has_multiline_field(path):
        return n - 1 if _ends_with_newline(path) else n
    with open(path, encoding='utf-8', newline='') as f:
        reader = _csv.reader(f)
        next(reader, [])
        return sum(1 for _ in reader)


def _has_multiline_field(path) -> bool:
    """是否存在引号内换行（跨行字段）。mmap 里配对查找引号，只扫到第一个命中为止。

    CSV 的引号内转义是双写（""），配对查找会把它当成「闭合 + 重开」两次，开合状态不变。
    """
    import mmap
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            pos = mm.find(b'"')
            while pos != -1:
                nxt = mm.find(b'"', pos + 1)
                if nxt == -1:
                    return True
                if mm.find(b'\n', pos, nxt) != -1:
                    return True
                pos = mm.find(b'"', nxt + 1)
    return False


def _ends_with_newline(path) -> bool:
    with open(path, 'rb') as f:
        f.seek(-1, 2)
        return f.read(1) == b'\n'


def check_input_contract(path, name=None, key=None, contracts_path=None) -> dict:
    p = Path(path)
    require(p.is_file(), f'输入不存在: {p}')
    contracts = load_input_contracts(contracts_path)
    k = key or manifest_input_key(p)
    entry = contracts.get(k)
    if entry is None:
        # 生产合同按逻辑名键控（pricing_inputs.<name>.file=登记路径）；key/路径都未命中时
        # 经 PRICING_INPUTS 反查逻辑名（测试临时合同按路径键控，走上一分支）
        by_path = {rel: nm for nm, rel in PRICING_INPUTS.items()}
        entry = contracts.get(by_path.get(k, '\x00'))
    entry = entry or {}
    rows, columns = file_shape(p)
    reg_rows = entry.get('rows')
    reg_cols = entry.get('columns')
    missing = [c for c in (reg_cols or []) if c not in columns]
    require(reg_rows is not None,
            f'输入合同未登记行数（file={k}）；先填 configs/pricing_inputs.yaml 的 rows')
    require(int(reg_rows) == rows,
            f'输入行数不匹配（file={k}）期望={reg_rows} 实际={rows}')
    require(not missing, f'输入缺列（file={k}）缺={missing}')
    return {'input': name or k, 'file': k, 'path': str(p), 'rows': rows,
            'rows_registered': reg_rows,
            'rows_match_registered': reg_rows is None or int(reg_rows) == rows,
            'columns_missing': missing, 'match': True}

def assert_named_pricing_input(name: str, path=None, manifest_path=None) -> dict:
    require(name in PRICING_INPUTS, f'pricing_gate 输入: 未登记逻辑名 {name!r}')
    p = Path(path) if path is not None else ROOT / PRICING_INPUTS[name]
    return check_input_contract(p, name=name, key=PRICING_INPUTS[name])

def assert_pricing_inputs(inputs=None, manifest_path=None, spec_path=None, *,
                         _allow_subset: bool = False, contracts_path=None) -> list:
    if inputs is None:
        inputs = {name: ROOT / rel for name, rel in PRICING_INPUTS.items()}
    else:
        inputs = dict(inputs)
        require(bool(inputs), 'pricing_gate 输入: inputs 为空；拒绝 0/0 PASS')
        if not _allow_subset:
            expected_names = set(PRICING_INPUTS)
            require(set(inputs) == expected_names,
                    f'pricing_gate 输入: 非测试调用必须校验全部 {len(expected_names)} 个输入；'
                    f'缺少={sorted(expected_names - set(inputs))} '
                    f'多余={sorted(set(inputs) - expected_names)}')
    ledger = []
    for name, path in inputs.items():
        rec = check_input_contract(path, name=name, key=PRICING_INPUTS[name],
                                   contracts_path=contracts_path)
        print(f"[INPUT] {name}: rows={rec['rows']} registered={rec['rows_registered']}", flush=True)
        ledger.append(rec)
    return ledger

def _is_zero_int(value) -> bool:
    try:
        return int(value) == 0
    except (TypeError, ValueError):
        return False

def contract_from_frame(df: pd.DataFrame, expected_rows=None, kind: str = 'auto') -> dict:
    """从面板/样本 DataFrame 现算四条硬门（只读，不写文件）。

    kind='panel'：missing_filled_with_zero 按 network_support_status != 'support' 的
                  PeerMAXraw/OwnMAXraw/PeerReturn 零值统计（与 pricing_sample_build A1 面板合同同口径）。
    kind='sample'：按 pricing_exclusion_reason -> 对应值列统计（与 pricing_sample_build _integrity_checks 同口径）。
    expected_rows=None：rows_match_registered 记 ROWS_NOT_APPLICABLE（仅自定义面板使用，
    表示“行数门未施加”，不是通过）。
    """
    if kind == 'auto':
        if 'pricing_exclusion_reason' in df.columns:
            kind = 'sample'
        elif 'network_support_status' in df.columns:
            kind = 'panel'
        else:
            kind = 'sample'
    c = {'rows': int(len(df)), 'kind': kind}
    if expected_rows is None:
        c['rows_match_registered'] = ROWS_NOT_APPLICABLE
    else:
        c['rows_registered'] = int(expected_rows)
        c['rows_match_registered'] = bool(len(df) == int(expected_rows))
    if {'focal_id', 'month'}.issubset(df.columns):
        c['duplicate_focal_month_keys'] = int(df.duplicated(['focal_id', 'month']).sum())
    else:
        c['duplicate_focal_month_keys'] = None
    if {'available_at', 'month'}.issubset(df.columns):
        av = pd.to_datetime(df['available_at'], errors='coerce')
        mo = pd.to_datetime(df['month'].astype(str) + '-01', errors='coerce')
        c['pit_violations_strict_lt'] = int((av >= mo).sum())
    else:
        c['pit_violations_strict_lt'] = None
    zeros = None
    if kind == 'sample' and 'pricing_exclusion_reason' in df.columns:
        exc = df['pricing_exclusion_reason'].fillna('')
        zeros = 0
        for name, col in _SAMPLE_EXCLUSION_VALUE_COL.items():
            if col in df.columns:
                zeros += int((df.loc[exc == name, col].fillna(-999) == 0).sum())
    elif kind == 'panel' and 'network_support_status' in df.columns:
        sup = df['network_support_status'] == 'support'
        cols = [c2 for c2 in _PRICING_RETURN_COLS if c2 in df.columns]
        if cols:
            zeros = int((df.loc[~sup, cols].fillna(-999) == 0).sum().sum())
    c['missing_filled_with_zero'] = None if zeros is None else int(zeros)
    return c

def _contract_entries(source, where: str, expected_rows, kind: str) -> dict:
    """把 report dict / contract dict / 路径 / DataFrame 归一化为 {key: contract}。"""
    if isinstance(source, pd.DataFrame):
        return {'<frame>': contract_from_frame(source, expected_rows=expected_rows, kind=kind)}
    if isinstance(source, (str, Path)):
        p = Path(source)
        require(p.exists(), f'{where}: 契约来源不存在: {p}')
        suffix = p.suffix.lower()
        if suffix == '.json':
            return _contract_entries(json.loads(p.read_text(encoding='utf-8')),
                                     where, expected_rows, kind)
        if suffix == '.parquet':
            return {'<parquet>': contract_from_frame(pd.read_parquet(p),
                                                     expected_rows=expected_rows, kind=kind)}
        if suffix == '.csv':
            return {'<csv>': contract_from_frame(pd.read_csv(p),
                                                 expected_rows=expected_rows, kind=kind)}
        require(False, f'{where}: 不支持的契约来源类型: {p}')
    if isinstance(source, dict):
        if 'contracts' in source:
            contracts = source.get('contracts')
            require(isinstance(contracts, dict) and bool(contracts),
                    f'{where}: 报告中 contracts 为空/缺失，拒绝继续')
            return contracts
        if any(k in source for k in HARD_GATE_KEYS) or 'missing_filled_with_zero_total' in source:
            return {'<contract>': source}
        require(False, f'{where}: dict 既非 build report（无 contracts）也无四条硬门字段')
    require(False, f'{where}: 无法识别的契约来源（{type(source).__name__}）；'
                   '需要 build report dict / contract dict / json|parquet|csv 路径 / DataFrame')
    return {}

def assert_pricing_contract(source, expected_rows=None, where: str = 'pricing_contract',
                       kind: str = 'auto') -> dict:
    """硬门入口：核验四条输入契约，任一失败即 SystemExit(2)。

    source:
      - pricing_sample_report(.json) 的 report dict（使用其 contracts）
      - 单列 contract dict（直接含四条硬门字段）
      - report/parquet/csv 路径
      - DataFrame（此时 expected_rows 给登记行数；None 表示行数门不适用，仅自定义面板）
    返回摘要 dict（便于报告/测试断言）；不写任何文件。
    """
    contracts = _contract_entries(source, where, expected_rows, kind)
    summary = {'where': where, 'n_contracts': len(contracts), 'status': 'PASS', 'contracts': {}}
    for key, c in contracts.items():
        require(isinstance(c, dict), f'{where}/{key}: 契约不是 dict')
        if 'rows_match_registered' in c:
            rows_val = c['rows_match_registered']
        elif isinstance(expected_rows, dict) and key in expected_rows and 'rows' in c:
            rows_val = bool(int(c['rows']) == int(expected_rows[key]))
        elif expected_rows is not None and not isinstance(expected_rows, dict) and 'rows' in c:
            rows_val = bool(int(c['rows']) == int(expected_rows))
        else:
            require(False, f'{where}/{key}: 缺 rows_match_registered，且无法由 rows+期望行数复算')
        rows_enforced = rows_val != ROWS_NOT_APPLICABLE
        if rows_enforced:
            require(isinstance(rows_val, (bool, np.bool_)) and bool(rows_val),
                    f'{where}/{key}: rows_match_registered={rows_val!r}'
                    f'（rows={c.get("rows")}, 登记={c.get("rows_registered")}）')
        pit = c.get('pit_violations_strict_lt', c.get('pit_violations'))
        require(pit is not None, f'{where}/{key}: 缺 pit_violations_strict_lt')
        require(_is_zero_int(pit), f'{where}/{key}: PIT 违规 {pit} != 0')
        dup = c.get('duplicate_focal_month_keys')
        require(dup is not None, f'{where}/{key}: 缺 duplicate_focal_month_keys')
        require(_is_zero_int(dup), f'{where}/{key}: 重复 (focal_id, month) 键 {dup} != 0')
        zero = c.get('missing_filled_with_zero', c.get('missing_filled_with_zero_total'))
        require(zero is not None, f'{where}/{key}: 缺 missing_filled_with_zero')
        require(_is_zero_int(zero), f'{where}/{key}: 缺失被填 0 的行 {zero} != 0')
        summary['contracts'][key] = {
            'rows': c.get('rows'), 'rows_registered': c.get('rows_registered'),
            'rows_match_registered': rows_val, 'rows_gate_enforced': rows_enforced,
            'pit_violations_strict_lt': pit,
            'duplicate_focal_month_keys': dup, 'missing_filled_with_zero': zero}
    print(f'[pricing_gate-CONTRACT] {where}: PASS（{len(contracts)} 个契约 × 4 条硬门）', flush=True)
    return summary
import math
import numpy as np
import pandas as pd

def winsor_bounds(registry_path=None) -> tuple:
    """registry winsorization_pct -> (lo, hi)；长度 2、0<=lo<hi<=1，否则 SystemExit(2)。"""
    raw = _registry_param('winsorization_pct', registry_path)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        _registry_fail(f'winsorization_pct 必须是长度 2 序列，当前={raw!r}')
    try:
        lo, hi = float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        _registry_fail(f'winsorization_pct 含非数值: {raw!r}')
    if not (0.0 <= lo < hi <= 1.0):
        _registry_fail(f'winsorization_pct 需满足 0<=lo<hi<=1，当前=({lo}, {hi})')
    return lo, hi

def winsorize_bounds_by_month(df: pd.DataFrame, col: str, lo: float = 0.01, hi: float = 0.99,
                              month_col: str = 'month'):
    """月度截面 [lo, hi] 分位边界（在该 frame 的宇宙上计算；缺月返回 NaN 边界 = 不裁剪）。"""
    g = df.groupby(month_col, sort=False)[col]
    return g.transform(lambda s: s.quantile(lo)), g.transform(lambda s: s.quantile(hi))

def clip_by_month(df: pd.DataFrame, cols, lo: float = 0.01, hi: float = 0.99,
                  month_col: str = 'month') -> pd.DataFrame:
    """I1 修正：对指定列做与主规格同规则的月度截面 [lo,hi] winsorize（宇宙 = 传入 frame）。

    用于 A9 中由 aggregate_edges 重算的权重变体 / placebo 网络的 PeerMAXraw 与 PeerReturn：
    修正前这些列是未 winsorize 值，与主规格（面板构造期已 winsorize）口径不对称。
    """
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        b_lo, b_hi = winsorize_bounds_by_month(df, c, lo=lo, hi=hi, month_col=month_col)
        df[c] = df[c].clip(lower=b_lo, upper=b_hi)
    return df

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def newey_west(series, lag: int = 6):
    """系数序列的均值 + Newey-West(Bartlett) SE。返回 dict。"""
    a = np.asarray([v for v in series if v is not None and np.isfinite(v)], dtype=float)
    T = len(a)
    out = {'T': T, 'mean': None, 'se_nw': None, 't_nw': None, 'se_iid': None, 't_iid': None}
    if T < 2:
        return out
    m = float(a.mean())
    d = a - m
    gamma0 = float(np.dot(d, d) / T)
    nw_var = gamma0
    for l in range(1, min(lag, T - 1) + 1):
        w = 1.0 - l / (lag + 1.0)
        cov = float(np.dot(d[l:], d[:-l]) / T)
        nw_var += 2.0 * w * cov
    se_nw = math.sqrt(max(nw_var, 0.0) / T)
    se_iid = math.sqrt(float(d.var(ddof=1)) / T) if T > 1 else float('nan')
    out.update({'mean': m,
                'se_nw': se_nw if np.isfinite(se_nw) else None,
                't_nw': (m / se_nw) if se_nw > 0 else None,
                'se_iid': se_iid if np.isfinite(se_iid) else None,
                't_iid': (m / se_iid) if se_iid and se_iid > 0 else None})
    return out

def one_sided_p(t, direction: str = 'negative'):
    """H0: beta >= 0 vs H1: beta < 0（主方向 beta<0）-> p = Phi(t)。"""
    if t is None or not np.isfinite(t):
        return None
    return norm_cdf(t) if direction == 'negative' else 1.0 - norm_cdf(t)

def two_sided_p(t):
    if t is None or not np.isfinite(t):
        return None
    return 2.0 * (1.0 - norm_cdf(abs(t)))

def tost_equivalence(diff_series, bound: float, alpha: float, lag: int = 6):
    """TOST：H0 |d| >= bound vs H1 |d| < bound（两单侧 p 均 < alpha 记等价）。"""
    nw = newey_west(diff_series, lag=lag)
    d, se = nw['mean'], nw['se_nw']
    out = {'mean_diff': d, 'se_nw': se, 'bound': bound, 'alpha': alpha,
           'p_lower': None, 'p_upper': None, 'equivalent': False,
           'status': 'NOT_COMPUTABLE'}
    if d is None or not se or se <= 0:
        return out
    t1 = (d + bound) / se          # H0: d <= -bound
    t2 = (bound - d) / se          # H0: d >= +bound
    p_low = 1.0 - norm_cdf(t1)
    p_high = 1.0 - norm_cdf(t2)
    out.update({'p_lower': p_low, 'p_upper': p_high,
                'equivalent': bool(p_low < alpha and p_high < alpha),
                'status': 'PASS' if (p_low < alpha and p_high < alpha) else 'FAIL'})
    return out

def winsorize_by_month(df: pd.DataFrame, cols, lo: float = 0.01, hi: float = 0.99,
                       month_col: str = 'month') -> pd.DataFrame:
    """月度截面 winsorize（只动给定列；缺失保持缺失）。"""
    df = df.copy()
    g = df.groupby(month_col, sort=False)
    for c in cols:
        if c not in df.columns:
            continue
        q_lo = g[c].transform(lambda s: s.quantile(lo))
        q_hi = g[c].transform(lambda s: s.quantile(hi))
        df[c] = df[c].clip(lower=q_lo, upper=q_hi)
    return df

def treatment_only_mask(df: pd.DataFrame) -> pd.Series:
    """treatment-only eligible 宇宙（outcome-blind）：
    通过全部 treatment/controls 可得性屏幕的行（exclusion 顺序中 outcome 屏之前）；
    v2 样本的 `pricing_exclusion_reason ∈ {'', 'no_outcome_next_month'}` 即该宇宙。"""
    if 'pricing_exclusion_reason' not in df.columns:
        return pd.Series(True, index=df.index)
    return df['pricing_exclusion_reason'].fillna('').isin(['', 'no_outcome_next_month'])

def demean_by(df: pd.DataFrame, cols, group_cols) -> pd.DataFrame:
    """组内去均值（within 变换）。

    group_cols 含 month 时等价于「行业×月固定效应」：每个 (month, industry) 组是
    饱和虚拟变量，组内去均值后的 OLS 就是带该组固定效应的估计量。缺失保持缺失。
    """
    out = df.copy()
    g = out.groupby(list(group_cols), sort=False)
    for c in cols:
        if c in out.columns:
            out[c] = out[c] - g[c].transform('mean')
    return out


def rank_average_groups(x: pd.Series, n_groups: int) -> pd.Series:
    """冻结 tie rule：rank(method='average') 后按 ceil(rank/N*g) 分组（跨 cutoff 的 tie 同组）。"""
    r = x.rank(method='average')
    n = r.notna().sum()
    if n == 0:
        return pd.Series(np.nan, index=x.index)
    g = np.ceil(r / n * n_groups)
    g = g.clip(lower=1, upper=n_groups)
    g[x.isna()] = np.nan
    return g.astype(float)

def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Harrell restricted cubic spline 基（含线性项 + k-1 个非线性项）。"""
    x = np.asarray(x, dtype=float)
    k = np.asarray(knots, dtype=float)
    nk = len(k)
    out = [x]
    for j in range(nk - 2):
        def cube(t):
            return np.where(t > 0, t ** 3, 0.0)
        term = (cube(x - k[j])
                - cube(x - k[nk - 2]) * (k[nk - 1] - k[j]) / (k[nk - 1] - k[nk - 2])
                + cube(x - k[nk - 1]) * (k[nk - 2] - k[j]) / (k[nk - 1] - k[nk - 2]))
        out.append(term / (k[nk - 1] - k[1]) ** 2)
    return np.column_stack(out)

# 设计矩阵条件数上限：超过即判该月不可估（不进入系数序列）。实测主规格条件数
# p50≈11、p95≈19、max≈20（列归一化后）；阈值取 1e10 留 9 个数量级余量，
# 只拦数值上已退化的月份（近共线噪声 <1e-9 量级）。
COND_MAX = 1e10


def _ols(X: np.ndarray, y: np.ndarray, w: np.ndarray | None):
    n, p = X.shape
    if w is None:
        Xw, yw = X, y
    else:
        sw = np.sqrt(w)
        Xw = X * sw[:, None]
        yw = y * sw
    XtX = Xw.T @ Xw
    # 秩与条件数都必须在**列归一化**后判定：原始列量纲可差十几个数量级
    # （市值 ~1e10、MAX ~0.04），未归一化的 matrix_rank 会把健康的满秩矩阵误判成秩 1。
    cn = np.linalg.norm(Xw, axis=0)
    zero_var = [int(j) for j in range(p) if cn[j] == 0.0]
    if zero_var:
        return {'status': 'ZERO_VARIANCE', 'columns': zero_var, 'n': n}
    Xn = Xw / cn
    rank = int(np.linalg.matrix_rank(Xn))
    if rank < p:
        return None
    # 近共线：满秩但条件数爆表时伪逆给出数值爆炸的 β，仍会被计入 FM 均值。
    # 条件数在**列归一化**后算（原始列量纲差几个数量级）。零方差列单独判退化，
    # 不并进条件数：某些月在分层变量上恒为常数（如 limit_hit_weight_share 全 0），
    # 那是该月无变异而不是共线。
    sv_n = np.linalg.svd(Xn, compute_uv=False)
    cond = float(sv_n[0] / sv_n[-1]) if sv_n[-1] > 0 else float('inf')
    if cond > COND_MAX:
        return {'status': 'ILL_CONDITIONED', 'cond': cond, 'rank': rank, 'n': n}
    XtXi = np.linalg.pinv(XtX)
    beta = XtXi @ (Xw.T @ yw)
    resid = yw - Xw @ beta
    df = n - p
    if df <= 0:
        return None
    sigma2 = float(resid @ resid / df)
    cov = sigma2 * XtXi
    finite = bool(np.isfinite(beta).all() and np.isfinite(cov).all() and np.isfinite(sigma2))
    return {'beta': beta, 'cov': cov, 'rank': rank, 'df': df, 'n': n,
            'sigma2': sigma2, 'finite': finite}

def fm_regression(df: pd.DataFrame, ycol: str, xcols, month_col: str = 'month',
                  weight_col: str | None = None, min_n: int = 100, nw_lag: int = 6,
                  direction: str = 'negative'):
    """逐月横截面回归 -> 系数序列 -> NW 汇总。返回 (summary, per_month)。"""
    coefs = {c: [] for c in xcols}
    coefs['const'] = []
    per_month = []
    n_months_used = 0
    for m, sub in df.groupby(month_col, sort=True):
        sub = sub.dropna(subset=[ycol] + list(xcols))
        if weight_col:
            sub = sub[sub[weight_col].notna() & (sub[weight_col] > 0)]
        n = len(sub)
        if n < min_n:
            per_month.append({'month': m, 'n': n, 'used': False, 'reason': 'below_min_n'})
            continue
        y = sub[ycol].to_numpy(dtype=float)
        X = np.column_stack([sub[c].to_numpy(dtype=float) for c in xcols]
                            + [np.ones(n)])
        w = sub[weight_col].to_numpy(dtype=float) if weight_col else None
        fit = _ols(X, y, w)
        if fit is None:
            per_month.append({'month': m, 'n': n, 'used': False,
                              'reason': 'rank_or_df_or_covariance_failure'})
            continue
        if fit.get('status') in ('ILL_CONDITIONED', 'ZERO_VARIANCE'):
            per_month.append({'month': m, 'n': n, 'used': False,
                              'reason': fit['status'].lower(),
                              'condition_number': fit.get('cond')})
            continue
        if not fit.get('finite'):
            per_month.append({'month': m, 'n': n, 'used': False,
                              'reason': 'rank_or_df_or_covariance_failure'})
            continue
        n_months_used += 1
        for j, c in enumerate(list(xcols) + ['const']):
            coefs[c].append(float(fit['beta'][j]))
        per_month.append({'month': m, 'n': n, 'used': True, 'rank': fit['rank'],
                          'n_params': len(xcols) + 1, 'residual_df': fit['df'],
                          'covariance_status': 'finite',
                          'rank_status': 'FULL' if fit['rank'] == len(xcols) + 1 else 'RANK_DEFICIENT',
                          'beta': {c: float(fit['beta'][j]) for j, c in enumerate(list(xcols) + ['const'])}})
    summary = {'n_months_used': n_months_used, 'min_n': min_n, 'weight_col': weight_col,
               'nw_lag': nw_lag, 'coefs': {}, 'per_month': per_month}
    for c in list(xcols) + ['const']:
        nw = newey_west(coefs[c], lag=nw_lag)
        s = {'beta': nw['mean'], 'se_nw': nw['se_nw'], 't_nw': nw['t_nw'],
             'se_iid': nw['se_iid'], 't_iid': nw['t_iid'], 'n_months': nw['T'],
             'p_one_sided': one_sided_p(nw['t_nw'], direction),
             'p_two_sided': two_sided_p(nw['t_nw'])}
        if s['beta'] is not None and s['se_nw']:
            s['ci95_low'] = s['beta'] - 1.96 * s['se_nw']
            s['ci95_high'] = s['beta'] + 1.96 * s['se_nw']
        summary['coefs'][c] = s
    return summary, per_month

def fm_cluster_se(df: pd.DataFrame, ycol: str, xcols, cluster_col: str,
                  weight_col: str | None = None, month_col: str = 'month',
                  min_n: int = 100, nw_lag: int = 6) -> dict:
    """Fama-MacBeth + 月内聚类稳健抽样误差（hybrid：跨月 NW × 月内聚类设计效应）。

    逐月横截面 OLS（与 fm_regression 同口径）；对 treatment 列给出：
      se_sampling_iid_m     = sqrt(sigma2 * (X'WX)^{-1}_jj)   当月同方差抽样误差
      se_sampling_cluster_m = sqrt(sum_g psi_gj^2)            当月团聚类抽样误差
      psi_g                 = (X'WX)^{-1} X_g' W e_g          团得分函数
    T 个月合并后：
      se_sampling = sqrt(mean(se_m^2) / T)
      se_hybrid   = sqrt(se_nw^2 + se_sampling_cluster^2 - se_sampling_iid^2)
    deff = se_sampling_cluster / se_sampling_iid（月内聚类的设计效应）。
    """
    j = list(xcols).index('PeerMAXraw') if 'PeerMAXraw' in xcols else 0
    betas = []
    var_iid, var_cl = [], []
    for m, sub in df.groupby(month_col, sort=True):
        sub = sub.dropna(subset=[ycol] + list(xcols) + [cluster_col])
        if weight_col:
            sub = sub[sub[weight_col].notna() & (sub[weight_col] > 0)]
        n = len(sub)
        if n < min_n:
            continue
        X = np.column_stack([sub[c].to_numpy(dtype=float) for c in xcols] + [np.ones(n)])
        y = sub[ycol].to_numpy(dtype=float)
        w = sub[weight_col].to_numpy(dtype=float) if weight_col else None
        fit = _ols(X, y, w)
        if (fit is None or not fit.get('finite')
                or fit.get('status') in ('ILL_CONDITIONED', 'ZERO_VARIANCE')):
            continue
        sw = np.sqrt(w) if w is not None else np.ones(n)
        Xw, yw = X * sw[:, None], y * sw
        resid = yw - Xw @ fit['beta']
        bread = np.linalg.pinv(Xw.T @ Xw)
        betas.append(float(fit['beta'][j]))
        var_iid.append(float(fit['cov'][j, j]))
        g = pd.factorize(sub[cluster_col].to_numpy())[0]
        s = 0.0
        for gi in np.unique(g):
            sel = (g == gi)
            psi = bread @ (Xw[sel].T @ resid[sel])
            s += float(psi[j]) ** 2
        var_cl.append(s)
    T = len(betas)
    out = {'T': T, 'cluster_col': cluster_col, 'nw_lag': nw_lag,
           'status': 'NOT_COMPUTABLE', 'beta': None}
    if T < 2:
        return out
    nw = newey_west(betas, lag=nw_lag)
    se_iid_s = math.sqrt(float(np.mean(var_iid)) / T)
    se_cl_s = math.sqrt(float(np.mean(var_cl)) / T)
    hybrid = math.sqrt(max(nw['se_nw'] ** 2 + se_cl_s ** 2 - se_iid_s ** 2, 0.0))
    out.update({
        'status': 'COMPUTED',
        'beta': nw['mean'], 'se_nw': nw['se_nw'], 't_nw': nw['t_nw'],
        'se_sampling_iid': se_iid_s,
        'se_sampling_cluster': se_cl_s,
        'deff_within_month': (se_cl_s / se_iid_s) if se_iid_s > 0 else None,
        'se_hybrid': hybrid,
        't_hybrid': (nw['mean'] / hybrid) if hybrid > 0 else None,
        'ratio_hybrid_vs_nw': (hybrid / nw['se_nw']) if nw['se_nw'] else None,
        'p_one_sided_hybrid': one_sided_p((nw['mean'] / hybrid) if hybrid > 0 else None,
                                          'negative'),
        'reading': 'hybrid 为「跨月 NW + 月内聚类设计效应」；deff>1 表示同月同团内相关使'
                   '同方差 SE 偏小',
    })
    return out


def nw_cov_matrix(vecs, lag: int = 6):
    """系数向量序列的均值 + Newey-West 协方差矩阵（Bartlett 权）。"""
    A = np.asarray([v for v in vecs if v is not None and np.isfinite(v).all()], dtype=float)
    T, k = A.shape
    if T < 2:
        return None, None, T
    mean = A.mean(axis=0)
    D = A - mean
    S = (D.T @ D) / T
    for l in range(1, min(lag, T - 1) + 1):
        w = 1.0 - l / (lag + 1.0)
        G = (D[l:].T @ D[:-l]) / T
        S = S + w * (G + G.T)
    cov = S / T
    return mean, cov, T

def joint_wald(vecs, idx, lag: int = 6):
    """对系数向量子集做联合 Wald 检验（H0: 子集全部为 0）。"""
    mean, cov, T = nw_cov_matrix(vecs, lag=lag)
    if mean is None:
        return {'status': 'NOT_COMPUTABLE'}
    sub = cov[np.ix_(idx, idx)]
    b = mean[idx]
    try:
        inv = np.linalg.pinv(sub)
        stat = float(b @ inv @ b)
    except Exception:
        return {'status': 'NOT_COMPUTABLE'}
    from math import erfc, sqrt
    k = len(idx)
    # Wilson-Hilferty 近似 p 值（chi2_k 上尾）
    if stat <= 0:
        p = 1.0
    else:
        z = ((stat / k) ** (1.0 / 3.0) - (1 - 2.0 / (9 * k))) / np.sqrt(2.0 / (9 * k))
        p = 0.5 * erfc(z / sqrt(2.0))
    return {'status': 'COMPUTED', 'wald_chi2': stat, 'df': int(k), 'p_value': float(p),
            'T': int(T), 'note': 'NW 协方差 + Wilson-Hilferty 近似（诊断用）'}

def vif_table(df: pd.DataFrame, xcols, month_col: str = 'month') -> dict:
    """逐月 VIF 的均值（对设计矩阵列，含常数项以外）；条件数同时给出。"""
    vifs = {c: [] for c in xcols}
    conds = []
    for m, sub in df.groupby(month_col, sort=True):
        sub = sub.dropna(subset=list(xcols))
        if len(sub) < 50:
            continue
        X = sub[list(xcols)].to_numpy(dtype=float)
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        Xs = (X - mu) / sd
        Xs = np.column_stack([Xs, np.ones(len(Xs))])   # 常数项单列（不标准化，避免零列退化）
        sv = np.linalg.svd(Xs, compute_uv=False)
        if sv[-1] > 0:
            conds.append(float(sv[0] / sv[-1]))
        for j, c in enumerate(xcols):
            others = [k for k in range(X.shape[1]) if k != j]
            Xo = Xs[:, others]
            fit = _ols(Xo, Xs[:, j], None)
            if (fit is None or not fit.get('finite')
                    or fit.get('status') in ('ILL_CONDITIONED', 'ZERO_VARIANCE')):
                vifs[c].append(np.nan)
                continue
            resid = Xs[:, j] - Xo @ fit['beta']
            ss_tot = float(((Xs[:, j] - Xs[:, j].mean()) ** 2).sum())
            r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
            vifs[c].append(1.0 / (1.0 - r2) if np.isfinite(r2) and r2 < 1 else np.inf)
    return {'vif_mean': {c: (float(np.nanmean(v)) if len(v) else None) for c, v in vifs.items()},
            'vif_max': {c: (float(np.nanmax(v)) if len(v) else None) for c, v in vifs.items()},
            'condition_number_mean': float(np.mean(conds)) if conds else None,
            'condition_number_max': float(np.max(conds)) if conds else None,
            'n_months': len(conds)}


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --- scripts/lib bootstrap ---
# --- end scripts/lib bootstrap ---


import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# W1 bundle（sign.2-01）：v4 面板路径（新文件名；不覆盖 v3）。默认行为仍走 PANELS。

# ---------------------------------------------------------------- I-14 amount 缓存契约
# 缓存名必须与内容一致：本缓存只存 firm_id/month/amount_100m（Drive 日成交额千元当月求和 / 1e5），
# **不含 illiq 列**。下游（pricing_sample_build）事件月实际公式：illiq = |own_R_t| / amount_100m。
# 既有 registry/formula 登记名 ILLIQ_MONTHLY_AMIHUD_CNY100M_v1 保持不动（冻结登记），
# 本常量仅用于报告/缓存元数据澄清实际公式，不新增 Amihud 公式、不改数值。

# 样本版本（Phase A 复核修复批 2026-10-10）：
#   v1 = pricing_sample_v1_2015_2023：未施加 microcap 筛除、ma/mb 未施加 sample 窗口过滤 →
#        status = superseded_registered_deviation（保留作「未施加 microcap 筛除」敏感性）
#   v2 = pricing_sample_v2_2015_2023：microcap + 次新股筛除；2026-11-15 sign-01 起标 superseded
#        （旧 complete_month 用价格非空日数，非有效交易日）
#   v3 = pricing_sample_v3_20261115：headline（sign-01：complete_month/n_valid_days = volume>0 交易日 >=15）
# W1 bundle 版本占位：v4 样本只在 W2 创建；W1 仅让 sample_path/CLI 可解析（默认行为不变）。

# ---------------------------------------------------------------- registry 参数绑定（Phase 1 I-5）
# 本块的注册表读取放在模块级常量之前：SAMPLE_WINDOW_* / winsor_bounds() 均以 registry 为准，
# 导入即 fail-closed（registry 缺失、参数缺失/重复、值域非法 → SystemExit(2)）。
# registry_path 仅作测试注入；生产调用一律用 configs/parameter_registry.yaml。


# sample 窗口（registry sample_start/sample_end，事件月口径；当前 2015-01 / 2023-12）
SAMPLE_WINDOW_START, SAMPLE_WINDOW_END = sample_window_bounds()

# I4：IQR_pool 登记值（treatment-only 宇宙、单一池化、全规格共用）。
# 2026-11-19 审核收口：数值唯一来源改为 registry 的 iqr_pool_value 参数；模块常量与
# registry 不一致即 fail-closed，禁止两处各写一份。outcome-blind 事前冻结，不按结果调整。
_IQR_REGISTRY = float(param_value('iqr_pool_value'))
if abs(_IQR_REGISTRY - IQR_POOL_REGISTERED) > 1e-12:
    raise SystemExit(
        f"[STOP] IQR_POOL_REGISTERED={IQR_POOL_REGISTERED} 与 registry "
        f"iqr_pool_value={_IQR_REGISTRY} 不一致；以 registry 为准并同步常量")
IQR_POOL_REGISTERED = _IQR_REGISTRY

# 2026-11-19 参数口径核查（CO-2026-11-19-parameters）：
# headline 样本指针与 ew_vw_consistency 等价界同样只认 registry，禁止各脚本再写字面量。
_SAMPLE_REG = param_value('sample_and_spec_registry')
HEADLINE_SAMPLE_VERSION = str(_SAMPLE_REG['sample_version'])
# headline 只许 v5；shadow 仍登记在 v2（退役条件见下），因此 label 表需同时容纳两者。
# 只保留 headline（v5）与 shadow 登记的 legacy 版本（v2）；v1/v3/v4 已退役，不再解析。
_VERSION_LABELS = {SAMPLE_VERSION_V5: 'v5'}
if HEADLINE_SAMPLE_VERSION not in _VERSION_LABELS:
    raise SystemExit(f"[STOP] registry sample_version={HEADLINE_SAMPLE_VERSION!r} 不在已登记版本集")
SAMPLE_VERSION_LABEL = _VERSION_LABELS[HEADLINE_SAMPLE_VERSION]

# shadow_* 全套尚未重跑到 headline 版本；其样本版本单独登记，避免与 headline 混用。
# 退役条件：shadow 全部重跑到 HEADLINE_SAMPLE_VERSION 后删除本参数与 shadow_sample_version()。
_SHADOW_REG = param_value('shadow_legacy_sample_version')
if _SHADOW_REG not in _VERSION_LABELS:
    raise SystemExit(f"[STOP] registry shadow_legacy_sample_version={_SHADOW_REG!r} 不在已登记版本集")
SHADOW_SAMPLE_VERSION_LABEL = _VERSION_LABELS[_SHADOW_REG]


def shadow_sample_path(key: str):
    """shadow_* 专用：读 registry 登记的 legacy 样本版本（当前 v2），不随 headline 漂移。"""
    return sample_path(key, SHADOW_SAMPLE_VERSION_LABEL)

_K_ECON = float(param_value('K_Econ_Pricing'))
_K_CONSIST = float(param_value('K_Consist_Pricing'))
_K_CONSIST_DERIVED = _K_ECON / IQR_POOL_REGISTERED
if abs(_K_CONSIST - _K_CONSIST_DERIVED) > 1e-6:
    raise SystemExit(
        f"[STOP] K_Consist_Pricing={_K_CONSIST} 与 K_Econ_Pricing/IQR_pool={_K_CONSIST_DERIVED:.6f} 不一致；"
        "以派生关系为准并同步 registry")

import yaml as _yaml
_SPEC_DOC = _yaml.safe_load(SPEC.read_text(encoding='utf-8')) or {}
_REG_VERSION = load_registry().get('registry_version')
_SPEC_REG_VERSION = ((_SPEC_DOC.get('governance') or {}).get('registry_version'))
if _REG_VERSION != _SPEC_REG_VERSION:
    raise SystemExit(
        f"[STOP] registry_version={_REG_VERSION!r} 与 analyst_spec governance.registry_version="
        f"{_SPEC_REG_VERSION!r} 不一致；改 registry 必须同步 configs/analyst_spec.yaml")


# ---------------------------------------------------------------- 基础工具


# ---------------------------------------------------------------- pricing_gate 主输入合同（Phase 1 I-3）
# 9 个 pricing_gate 主输入（以 pricing_sample_build / pricing_lib 实际读取路径为准；market_equity_panel_full
# 为 market_equity_panel 的字节相同孪生件，按方案一并纳入账本；postadj 为 pricing_1
# 读取的后复权敏感性收益面板，R1 复核后加入）。
# 合同只读 configs/pricing_inputs.yaml 的 rows / columns；rows 未登记或行数、
# 列名不匹配一律 SystemExit(2)，禁止回退实测值或自动刷新合同。


# ---------------------------------------------------------------- 输入契约硬门（Phase 1 I-2）
# 四条硬门（字段名 == pricing_sample_report.json:contracts.<column> 的实际键名）：
#   rows_match_registered / pit_violations_strict_lt /
#   duplicate_focal_month_keys / missing_filled_with_zero


def months_between(start: str, end: str) -> list:
    s = pd.Period(start, freq='M')
    e = pd.Period(end, freq='M')
    return [str(p) for p in pd.period_range(s, e, freq='M')]


def ym_add(month: str, k: int) -> str:
    return str(pd.Period(month, freq='M') + k)


# ---------------------------------------------------------------- 统计工具


# ---------------------------------------------------------------- FM 回归


# ---------------------------------------------------------------- 装载

# W1 bundle（sign.2-01 A1 + D-08 B）v4 面板列集（43 + PeerMAXraw_renorm/formula_id_max_renorm）


# ---------------------------------------------------------------- 边表与权重变体
