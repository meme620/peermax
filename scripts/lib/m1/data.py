"""s3_m1_lib 的 data 实现（逐字搬移，逻辑未改）。"""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data_audit'))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data'))


from datetime import datetime, timezone
from hashlib import sha256
from lib.m1.paths import AMOUNT_CACHE_COLUMNS
from lib.m1.paths import AMOUNT_CACHE_DIR
from lib.m1.paths import AMOUNT_CACHE_META_NAME
from lib.m1.paths import AMOUNT_CACHE_META_REQUIRED
from lib.m1.paths import AMOUNT_CACHE_NAME
from lib.m1.paths import AMOUNT_PKL
from lib.m1.paths import ANALYST_DERIVED
from lib.m1.paths import CHARS
from lib.m1.paths import ILLIQ_FORMULA_ID
from lib.m1.paths import LEGACY_AMOUNT_CACHE_NAME
from lib.m1.paths import M1_SAMPLE_VERSION_V4
from lib.m1.paths import M1_SAMPLE_VERSION_V4_RENORM
from lib.m1.paths import ME_PANEL
from lib.m1.paths import NET_DIRS
from lib.m1.paths import PANELS
from lib.m1.paths import PANEL_PATHS_V4
from lib.m1.paths import PROGRESS
from lib.m1.paths import RETURNS_POSTADJ
from lib.m1.paths import RETURNS_V2
from lib.m1.paths import ROOT
from lib.m1.paths import SAMPLE_DIR
from lib.m1.paths import SRC_MONTH
from lib.m1.paths import ST_PKL
from lib.m1.paths import SW2
from lib.m1.paths import TURNOVER_PKL
from pathlib import Path
import json
import numpy as np
import os
import pandas as pd
import sys

def sample_path(key: str, version: str = 'v3'):
    """v1 = ..._{key}.parquet；v2 = ..._{key}_v2.parquet；v3 = ..._{key}_v3.parquet；
    v4 = ..._{key}_v4.parquet（W2 才创建）；v4 renorm = ..._{key}_v4_renorm.parquet（仅 D-08 B）。
    """
    if version in ('v4', M1_SAMPLE_VERSION_V4):
        return SAMPLE_DIR / f'm1_common_sample_{key}_v4.parquet'
    if version in ('v4_renorm', M1_SAMPLE_VERSION_V4_RENORM):
        return SAMPLE_DIR / f'm1_common_sample_{key}_v4_renorm.parquet'
    if version in ('v3', 'm1_s5_v3_20261115'):
        return SAMPLE_DIR / f'm1_common_sample_{key}_v3.parquet'
    if version in ('v2', 'm1_s5_v2_2015_2023'):
        return SAMPLE_DIR / f'm1_common_sample_{key}_v2.parquet'
    return SAMPLE_DIR / f'm1_common_sample_{key}.parquet'

def progress(msg: str) -> None:
    print(f'[A] {msg}', flush=True)
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, 'a', encoding='utf-8') as f:
        f.write(f'- {msg}\n')

def write_json(path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'  -> {p.relative_to(ROOT)}', flush=True)

def sha256_file(path) -> str:
    h = sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def load_panel(key: str, version: str = 'v3') -> pd.DataFrame:
    """v3（默认）= 现行面板；v4 = W1 新面板（新文件名，不覆盖 v3）。"""
    path = PANEL_PATHS_V4[key] if version in ('v4', 's4_panel_v4') else PANELS[key]
    df = pd.read_csv(path, dtype={'focal_id': str, 'month': str})
    return df

def code6(firm_id: str) -> str:
    s = str(firm_id)
    return s[2:] if s[:2] in ('sz', 'sh') else s

def load_returns_v2() -> pd.DataFrame:
    d = pd.read_csv(RETURNS_V2, usecols=['firm_id', 'month', 'R', 'n_valid_days',
                                         'complete_month', 'return_status'],
                    dtype={'firm_id': str, 'month': str})
    return d

def load_returns_raw() -> pd.DataFrame:
    """面板同源的 source-month 原始月收益（与 monthly_returns_clean.csv R 逐值一致）。"""
    d = pd.read_csv(RETURNS_V2, usecols=['firm_id', 'month', 'R'],
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
    import pickle
    import warnings
    warnings.filterwarnings('ignore')
    d = pickle.load(open(ST_PKL, 'rb'))
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
    import pickle
    import warnings
    warnings.filterwarnings('ignore')
    d = pickle.load(open(pkl, 'rb'))
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
        'source_sha256': sha256_file(src),
        'row_count': int(len(df)),
        'columns': list(df.columns),
        'dtypes': {c: str(t) for c, t in df.dtypes.items()},
        'script_path': _rel_path(scr),
        'script_sha256': sha256_file(scr),
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

def _cache_contract_problems(meta: dict, df: pd.DataFrame, source_sha: str) -> list:
    problems = []
    if meta.get('source_sha256') != source_sha:
        problems.append(f'source_sha256 sidecar={meta.get("source_sha256")} != 当前输入={source_sha}')
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
      - 新缓存 + 有效 sidecar + 输入 sha/行数/列一致 → 命中，不重算；
      - 新缓存存在但 sidecar 缺失/不可读 → 从 source 重建并写 sidecar；
      - sidecar 存在但输入 sha / formula id / 行数 / 列不匹配 → 默认 SystemExit(2)；
        只有显式 `rebuild=True` 才重算覆盖（输入漂移不得静默沿用旧值）；
      - 新缓存不存在、旧 `m1_cache_illiq.parquet` 存在且列正确 → 一次性只读迁移
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
        problems = _cache_contract_problems(meta, df, sha256_file(src))
        if problems:
            if not rebuild:
                _cache_fail('缓存指纹/契约不匹配，拒绝复用：' + '；'.join(problems) +
                            '（显式重建请传 rebuild=True，或删除缓存/sidecar 后重跑）')
            return _rebuild('rebuilt_cache_contract_mismatch', '; '.join(problems))
        out = dict(meta)
        out['cache_status'] = 'hit'
        print(f'[CACHE] amount_cache_hit rows={len(df):,} source_sha256={meta["source_sha256"]}',
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
    """Turnover = Drive daily_turnover_rate(官方 %) 月末值 / 100（与 S3A 9C 同口径）。

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
    """P2.2 三套权重（固定边集，只换权重；行内归一化）。"""
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
    07 P1.2-01 A2/A3），两者不得混用。
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
    """**唯一共享聚合函数**（P1.2-01 为 owner；P1.4-05/M2 复用，禁止第二套）。

    语义（07 P1.2-01 A2/A3；plan :4739-4761/:4968）：
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

    w = pd.to_numeric(pd.Series(edges[w_col]), errors='coerce').fillna(0.0).to_numpy(float)
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
    if suffix == 'max':
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

