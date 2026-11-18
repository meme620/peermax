#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 4 BL-01（I-16 / I-17）：M1 v2 样本与月度收益面板 metadata-only 重写（D1 路径）。

范围（写死，不做 D2 全量重建）：
  ① 4 个 `m1_common_sample_{key}_v2.parquet`：原 44 列原序逐值不变，仅末尾追加
     `window_requested`（全行 2015-01..2023-12）与 `sample_window_actual`
     （teacher ma/mb = 2015-05..2023-12；ma/mb = 2018-05..2023-12），
     其中 actual 现算（`network_version.notna() | m1_in_fm_sample`），不等期望值即停机。
  ② `data/processed/monthly_returns_clean_v2.csv`：**只改** `sample_version`
     一列（`mainline_2018_2025_fullperiod` → `min(month)..max(month)`），10 列原序、
     不新增列；其余 9 列按字节保留（字段级替换，不做 pandas 重格式化）。

不变量：行数 / 列序 / 全部旧列逐值（数值 max|Δ|=0、字符串精确）/ FM 行数与月数 /
headline EW·VW 系数（只读内存重算，≤1e-12）。任一违反 → 恢复备份并 SystemExit(2)。
不跑 M1/PC、不写 M1/PC 产物、不改 headline 与 shadow 网络、不改 registry/claim。

用法：
  python scripts/data_audit/p4_1_rewrite_metadata.py --v2-rewrite --dry-run
  python scripts/data_audit/p4_1_rewrite_metadata.py --v2-rewrite --apply \
      --backup-dir artifacts/phase4/backup/bl01_<timestamp>
  python scripts/data_audit/p4_1_rewrite_metadata.py --verify \
      --backup-dir artifacts/phase4/backup/bl01_<timestamp>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
G0_DIR = ROOT / 'reports' / 'data_gates' / 'G0'
SAMPLE_BUILD_REPORT = G0_DIR / 's3_m1_sample_report.json'
PRICING_TABLE_V2 = G0_DIR / 's3_m1_pricing_table_v2.json'
DEFAULT_BACKUP_ROOT = ROOT / 'artifacts' / 'phase4' / 'backup'

SCRIPT_REL = 'scripts/data_audit/p4_1_rewrite_metadata.py'
WINDOW_REQUESTED = '2015-01..2023-12'
NO_NETWORK_VERSION = 'NO_NETWORK_VERSION'
NEW_COLS = ['window_requested', 'sample_window_actual']

PARQUET_TARGETS = [
    ('ma_teacher', 'data/processed/m1_common_sample_ma_teacher_v2.parquet', 307_503, '2015-05..2023-12'),
    ('mb_teacher', 'data/processed/m1_common_sample_mb_teacher_v2.parquet', 384_303, '2015-05..2023-12'),
    ('ma', 'data/processed/m1_common_sample_ma_v2.parquet', 226_210, '2018-05..2023-12'),
    ('mb', 'data/processed/m1_common_sample_mb_v2.parquet', 282_570, '2018-05..2023-12'),
]
FM_EXPECTED = {'ma_teacher': (181_178, 104), 'mb_teacher': (225_264, 104),
               'ma': (118_620, 68), 'mb': (151_702, 68)}

CSV_REL = 'data/processed/monthly_returns_clean_v2.csv'
CSV_EXPECTED_ROWS = 856_771
CSV_COLUMNS = ['firm_id', 'month', 'R', 'n_valid_days', 'complete_month', 'return_status',
               'month_start', 'month_end', 'sample_version', 'available_at']
CSV_LEGACY_LABEL = 'mainline_2018_2025_fullperiod'
CSV_FORBIDDEN_COLS = ['n_price_obs', 'n_trade_days', 'n_valid_days_semantics',
                      'sample_version_source', 'sample_version_legacy']
MONTHLY_STATUS_REPORT = G0_DIR / 'monthly_returns_status_report.json'
CSV_EXPECTED_LABEL = '2000-02..2026-06'
CSV_NUMERIC_COLS = ['R', 'n_valid_days', 'complete_month']

TWIN_PAIRS = [('data/processed/monthly_returns_clean.csv',
               'data/processed/monthly_returns_full_2006_2025.csv'),
              ('data/processed/market_equity_panel.csv',
               'data/processed/market_equity_panel_full_2006_2025.csv')]

HEADLINE_TOL = 1e-12


def log(msg: str) -> None:
    print(f'[p4.1] {msg}', file=sys.stderr, flush=True)


def fail(msg: str) -> None:
    print(f'[p4.1] HARD-FAIL: {msg}', file=sys.stderr, flush=True)
    raise SystemExit(2)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """同目录临时文件 + os.replace；失败不留半成品。"""
    tmp = path.with_name(path.name + '.bl01_tmp')
    try:
        with open(tmp, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_bytes(path, json.dumps(obj, ensure_ascii=False, indent=1).encode('utf-8'))


# ------------------------------------------------------------------ 只读计划（dry-run / apply 共用）
def actual_window_from_frame(df: pd.DataFrame) -> str:
    """与 s3_m1_1_build_sample.compute_actual_sample_window 同口径（不修改入参）。"""
    mask = df['network_version'].notna()
    if 'm1_in_fm_sample' in df.columns:
        mask = mask | df['m1_in_fm_sample'].fillna(False).astype(bool)
    months = df.loc[mask, 'month'].dropna()
    if months.empty:
        return NO_NETWORK_VERSION
    return f'{months.min()}..{months.max()}'


def plan_parquet(key: str, rel: str, expected_rows: int, expected_actual: str) -> dict:
    path = ROOT / rel
    if not path.is_file():
        fail(f'缺少输入: {rel}')
    names = list(pq.ParquetFile(path).schema_arrow.names)
    present = [c for c in NEW_COLS if c in names]
    if present and len(present) != len(NEW_COLS):
        fail(f'{rel}: I-16 新列仅部分存在 {present}（拒绝在半个状态下重写）')
    rec = {'path': rel, 'kind': 'parquet', 'key': key, 'old_sha256': sha256_file(path),
           'rows_before': int(pq.ParquetFile(path).metadata.num_rows),
           'n_cols_before': len(names), 'columns_before': names,
           'expected_rows': expected_rows, 'expected_sample_window_actual': expected_actual}
    if rec['rows_before'] != expected_rows:
        fail(f'{rel}: 行数 {rec["rows_before"]} != 期望 {expected_rows}')
    legacy = None
    if 'sample_window' in names:
        legacy = pd.read_parquet(path, columns=['sample_window'])['sample_window'].dropna().unique().tolist()
    rec['legacy_sample_window_values'] = legacy
    df = pd.read_parquet(path, columns=['month', 'network_version', 'm1_in_fm_sample'])
    actual = actual_window_from_frame(df)
    rec['computed_sample_window_actual'] = actual
    if actual != expected_actual:
        fail(f'{rel}: 现算实际窗 {actual} != 期望 {expected_actual}（停机，见任务书验收 2）')
    fm = df['m1_in_fm_sample'].fillna(False).astype(bool)
    rec['fm_rows_before'] = int(fm.sum())
    rec['fm_months_before'] = int(df.loc[fm, 'month'].nunique())
    if (rec['fm_rows_before'], rec['fm_months_before']) != FM_EXPECTED[key]:
        fail(f'{rel}: FM 行/月 {(rec["fm_rows_before"], rec["fm_months_before"])} '
             f'!= 期望 {FM_EXPECTED[key]}')
    start = expected_actual.split('..')[0]
    pre = df[df['month'] < start]
    rec['pre_window_rows'] = int(len(pre))
    rec['pre_window_fm_true'] = int(pre['m1_in_fm_sample'].fillna(False).astype(bool).sum())
    rec['pre_window_network_version_notna'] = int(pre['network_version'].notna().sum())
    if rec['pre_window_fm_true'] or rec['pre_window_network_version_notna']:
        fail(f'{rel}: 窗口前 {start} 存在 FM/网络行，实际窗口径不成立')
    if present:
        chk = pd.read_parquet(path, columns=NEW_COLS)
        got_req = chk['window_requested'].dropna().unique().tolist()
        got_act = chk['sample_window_actual'].dropna().unique().tolist()
        rec.update({'status': 'NOOP', 'new_col_values': {'window_requested': got_req,
                                                                   'sample_window_actual': got_act}})
        if got_req != [WINDOW_REQUESTED] or got_act != [expected_actual]:
            fail(f'{rel}: 新列值不达目标（{got_req} / {got_act}）')
    else:
        rec.update({'status': 'NEEDS_REWRITE', 'new_col_values': None})
    return rec


def _read_csv_raw(path: Path):
    raw = path.read_bytes()
    if b'\r\n' in raw:
        sep = b'\r\n'
        if raw.replace(b'\r\n', b'').count(b'\n'):
            fail(f'{path.name}: 混合行尾（\\r\\n 与裸 \\n 并存），拒绝字段级重写')
    else:
        sep = b'\n'
    lines = raw.split(sep)
    trailing = lines and lines[-1] == b''
    if trailing:
        lines = lines[:-1]
    return raw, sep, lines, trailing


def plan_csv(rel: str) -> dict:
    path = ROOT / rel
    if not path.is_file():
        fail(f'缺少输入: {rel}')
    raw, sep, lines, trailing = _read_csv_raw(path)
    header = lines[0].split(b',')
    header_txt = [h.decode('utf-8') for h in header]
    if header_txt != CSV_COLUMNS:
        fail(f'{rel}: 列名/顺序 != 期望 10 列（实际 {header_txt}）')
    if b'"' in raw:
        fail(f'{rel}: 含引号字符，字段级重写不安全（停机）')
    n_rows = len(lines) - 1
    if n_rows != CSV_EXPECTED_ROWS:
        fail(f'{rel}: 数据行数 {n_rows} != 期望 {CSV_EXPECTED_ROWS}')
    idx = header_txt.index('sample_version')
    i_month, i_end, i_avail = (header_txt.index('month'), header_txt.index('month_end'),
                              header_txt.index('available_at'))
    old_vals, months = set(), []
    avail_mismatch = 0
    new_lines = [lines[0]]
    for line in lines[1:]:
        fields = line.split(b',')
        if len(fields) != len(CSV_COLUMNS):
            fail(f'{rel}: 数据行列数 {len(fields)} != {len(CSV_COLUMNS)}（坏行，停机）')
        old_vals.add(fields[idx].decode('utf-8'))
        months.append(fields[i_month].decode('utf-8'))
        if fields[i_avail] != fields[i_end]:
            avail_mismatch += 1
        new_lines.append(line)
    if avail_mismatch:
        fail(f'{rel}: available_at != month_end 的行数 = {avail_mismatch}（期望 0）')
    target = f'{min(months)}..{max(months)}'
    if target != CSV_EXPECTED_LABEL:
        fail(f'{rel}: 现算 sample_version {target} != 期望 {CSV_EXPECTED_LABEL}')
    tgt = target.encode('utf-8')
    if old_vals - {CSV_LEGACY_LABEL, target}:
        fail(f'{rel}: sample_version 出现未登记旧值 {sorted(old_vals)}')
    status = 'NOOP' if old_vals == {target} else 'NEEDS_REWRITE'
    if status == 'NEEDS_REWRITE':
        # 逐行字段级替换（保持行序、行尾与字节分隔符不变；不做 pandas 重格式化）
        rebuilt = []
        for line in lines[1:]:
            fields = line.split(b',')
            fields[idx] = tgt
            rebuilt.append(b','.join(fields))
        new_lines = [lines[0]] + rebuilt
    new_raw = sep.join(new_lines) + (sep if trailing else b'')
    if not MONTHLY_STATUS_REPORT.is_file():
        fail(f'缺少 P1-01 审计报告: {MONTHLY_STATUS_REPORT.relative_to(ROOT)}（先跑 '
             f'scripts/data/build_monthly_returns_status.py）')
    mstatus = json.loads(MONTHLY_STATUS_REPORT.read_text(encoding='utf-8'))
    if mstatus.get('n_valid_days_semantics') != 'trade_days_v1':
        fail(f'{MONTHLY_STATUS_REPORT.name}: n_valid_days_semantics != trade_days_v1')
    if not mstatus.get('n_valid_days_equals_n_trade_days'):
        fail(f'{MONTHLY_STATUS_REPORT.name}: n_valid_days != n_trade_days（拒绝旧语义交付）')
    if int(mstatus.get('complete_month_violations_n_trade_days_lt_min', -1)) != 0:
        fail(f'{MONTHLY_STATUS_REPORT.name}: complete_month=1 且 n_trade_days<15 的行数 != 0')
    return {'path': rel, 'kind': 'csv', 'old_sha256': sha256_file(path), 'rows_before': n_rows,
            'n_cols_before': len(header_txt), 'columns_before': header_txt,
            'expected_rows': CSV_EXPECTED_ROWS, 'line_sep': 'CRLF' if sep == b'\r\n' else 'LF',
            'sample_version_old_values': sorted(old_vals),
            'sample_version_new_value': target,
            'available_at_equals_month_end_rows': n_rows - avail_mismatch,
            'available_at_rule': 'available_at = month_end（月末可得，保守；本项不改其值）',
            'forbidden_new_columns_absent': all(c not in header_txt for c in CSV_FORBIDDEN_COLS),
            'monthly_status_audit': {
                'report': str(MONTHLY_STATUS_REPORT.relative_to(ROOT)),
                'n_valid_days_semantics': mstatus.get('n_valid_days_semantics'),
                'n_valid_days_equals_n_trade_days': mstatus.get('n_valid_days_equals_n_trade_days'),
                'n_price_obs_legacy': {
                    'rule': mstatus.get('n_price_obs_legacy_rule'),
                    'rows_price_obs_ne_trade_days': mstatus.get('n_price_obs_ne_n_trade_days_rows'),
                    'rows_both_zero': mstatus.get('n_price_obs_and_n_trade_days_both_zero_rows'),
                },
                'n_trade_days': {
                    'rule': mstatus.get('n_valid_days_rule'),
                    'rows': mstatus.get('rows'),
                    'complete_month_rows': mstatus.get('complete_month_rows'),
                    'complete_month_violations': mstatus.get(
                        'complete_month_violations_n_trade_days_lt_min'),
                    'legacy_definition': mstatus.get('legacy_definition'),
                },
                'no_legacy_semantics_restored': True,
                'delivered_csv_columns_unchanged': CSV_COLUMNS == header_txt,
            },
            'status': status, '_new_bytes': new_raw if status == 'NEEDS_REWRITE' else None}


def build_plan() -> list:
    plans = [plan_parquet(k, rel, rows, act) for k, rel, rows, act in PARQUET_TARGETS]
    plans.append(plan_csv(CSV_REL))
    return plans


def print_plan(plans: list) -> None:
    for rec in plans:
        if rec['kind'] == 'parquet':
            log(f"{rec['path']}: rows={rec['rows_before']} cols={rec['n_cols_before']}->"
                f"{rec['n_cols_before'] + len(NEW_COLS)} legacy_sample_window="
                f"{rec['legacy_sample_window_values']} actual={rec['computed_sample_window_actual']} "
                f"FM={rec['fm_rows_before']}/{rec['fm_months_before']} "
                f"pre_window_rows={rec['pre_window_rows']}(fm={rec['pre_window_fm_true']}) "
                f"status={rec['status']}")
        else:
            log(f"{rec['path']}: rows={rec['rows_before']} cols={rec['n_cols_before']} "
                f"sample_version {rec['sample_version_old_values']} -> "
                f"{rec['sample_version_new_value']} available_at==month_end rows="
                f"{rec['available_at_equals_month_end_rows']} status={rec['status']}")


# ------------------------------------------------------------------ 备份
def ensure_backup(backup_dir: Path, plans: list) -> dict:
    backup_dir.mkdir(parents=True, exist_ok=True)
    pre_path = backup_dir / 'pre_sha256.json'
    pre = json.loads(pre_path.read_text(encoding='utf-8')) if pre_path.is_file() else {}
    if not pre:
        pre = {'script': SCRIPT_REL, 'created_at': datetime.now(timezone.utc).isoformat(),
               'note': 'BL-01 pre-state（首次 --apply 落盘；重复 apply 不覆盖）', 'files': {}}
    copied = {}
    for rec in plans:
        src = ROOT / rec['path']
        dst = backup_dir / Path(rec['path']).name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied[rec['path']] = 'copied'
        else:
            copied[rec['path']] = 'reused'
        pre['files'][rec['path']] = {'backup_copy': str(dst.relative_to(ROOT)),
                                      'sha256': sha256_file(dst)}
    atomic_write_json(pre_path, pre)
    return {'pre_sha256': {k: v['sha256'] for k, v in pre['files'].items()},
            'backup_actions': copied}


def restore_from_backup(backup_dir: Path, plans: list) -> list:
    restored = []
    for rec in plans:
        if rec.get('status') != 'REWRITTEN':
            continue
        dst = ROOT / rec['path']
        src = backup_dir / Path(rec['path']).name
        if not src.is_file():
            fail(f'回滚失败：备份缺 {src}')
        atomic_write_bytes(dst, src.read_bytes())
        restored.append(rec['path'])
    return restored


# ------------------------------------------------------------------ 写盘
def apply_one(rec: dict) -> None:
    path = ROOT / rec['path']
    if rec['status'] != 'NEEDS_REWRITE':
        rec['new_sha256'] = sha256_file(path)
        return
    if rec['kind'] == 'parquet':
        df = pd.read_parquet(path)
        before = list(df.columns)
        if before != rec['columns_before']:
            fail(f"{rec['path']}: 读入列序与计划不一致（并发改动？）")
        df = df.copy()
        df['window_requested'] = WINDOW_REQUESTED
        df['sample_window_actual'] = rec['computed_sample_window_actual']
        out = df[before + NEW_COLS]
        tmp = path.with_name(path.name + '.bl01_tmp')
        try:
            out.to_parquet(tmp, index=False, engine='pyarrow')
            post = pd.read_parquet(tmp)
            if list(post.columns) != before + NEW_COLS or len(post) != rec['rows_before']:
                fail(f"{rec['path']}: 临时件结构校验失败（列序/行数）")
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
    else:
        atomic_write_bytes(path, rec['_new_bytes'])
    rec['new_sha256'] = sha256_file(path)
    rec['status'] = 'REWRITTEN'


# ------------------------------------------------------------------ 验证（逐列对拍 + headline 复算）
def _col_evidence(name: str, s_old: pd.Series, s_new: pd.Series) -> dict:
    ev = {'column': name, 'dtype_old': str(s_old.dtype), 'dtype_new': str(s_new.dtype),
          'dtype_equal': str(s_old.dtype) == str(s_new.dtype)}
    if len(s_old) != len(s_new):
        fail(f'{name}: 行数不等 {len(s_old)} != {len(s_new)}')
    if pd.api.types.is_numeric_dtype(s_old) and pd.api.types.is_numeric_dtype(s_new):
        a = s_old.astype('float64').to_numpy()
        b = s_new.astype('float64').to_numpy()
        nan_o, nan_n = np.isnan(a), np.isnan(b)
        nan_mismatch = int((nan_o ^ nan_n).sum())
        d = np.abs(a - b)
        d[nan_o & nan_n] = 0.0
        finite = np.isfinite(d)
        max_delta = float(d[finite].max()) if finite.any() else 0.0
        n_mismatch = int((d > 0).sum()) + nan_mismatch
        ev.update({'kind': 'numeric', 'n_mismatch': n_mismatch,
                   'max_abs_delta': max_delta, 'n_nan_mismatch': nan_mismatch})
    else:
        a = s_old.astype('object').where(s_old.notna(), '__NA__').astype(str).to_numpy()
        b = s_new.astype('object').where(s_new.notna(), '__NA__').astype(str).to_numpy()
        n_mismatch = int((a != b).sum())
        ev.update({'kind': 'string_or_bool', 'n_mismatch': n_mismatch, 'max_abs_delta': None})
    ev['ok'] = ev['n_mismatch'] == 0 and ev['dtype_equal']
    return ev


def _csv_byte_evidence(old_bytes: bytes, new_bytes: bytes, label_col: str) -> dict:
    """证明「除 sample_version 字段外逐字节相同」。"""
    _, sep_o, lines_o, trail_o = _read_csv_raw_bytes(old_bytes)
    _, sep_n, lines_n, trail_n = _read_csv_raw_bytes(new_bytes)
    header_o = lines_o[0].split(b',')
    idx = header_o.index(label_col.encode('utf-8'))
    rebuilt = [lines_o[0]]
    for line in lines_o[1:]:
        fields = line.split(b',')
        fields[idx] = b'PLACEHOLDER_NOT_USED'
        rebuilt.append(b','.join(fields))
    # 用新文件的 sample_version 值回填，验证其他字段逐字节相同
    used_votes = set()
    for line in lines_n[1:]:
        used_votes.add(line.split(b',')[idx])
    if len(used_votes) != 1:
        fail(f'CSV 新值非唯一: {sorted(v.decode() for v in used_votes)}')
    repl = list(used_votes)[0]
    rebuilt = [lines_o[0]]
    for line in lines_o[1:]:
        fields = line.split(b',')
        fields[idx] = repl
        rebuilt.append(b','.join(fields))
    candidate = sep_o.join(rebuilt) + (sep_o if trail_o else b'')
    same = candidate == new_bytes
    return {'byte_identical_except_sample_version_field': bool(same),
            'line_sep_preserved': sep_o == sep_n, 'n_data_lines_old': len(lines_o) - 1,
            'n_data_lines_new': len(lines_n) - 1,
            'new_sample_version_value': repl.decode('utf-8')}


def _read_csv_raw_bytes(raw: bytes):
    if b'\r\n' in raw:
        sep = b'\r\n'
        if raw.replace(b'\r\n', b'').count(b'\n'):
            fail('CSV 混合行尾（字节证据失败）')
    else:
        sep = b'\n'
    lines = raw.split(sep)
    trailing = bool(lines) and lines[-1] == b''
    if trailing:
        lines = lines[:-1]
    return raw, sep, lines, trailing


def headline_recompute() -> dict:
    """只读内存重算 headline（ma_teacher v2 主规格 EW/VW），不写任何 M1/PC 产物。"""
    sys.path.insert(0, str(ROOT / 'scripts' / 'data_audit'))
    import s3_m1_2_fm_main as fm  # noqa: E402（延迟导入：模块级仅读 registry）
    from s3_m1_lib import IQR_POOL_REGISTERED  # noqa: E402

    sample_report = json.loads(SAMPLE_BUILD_REPORT.read_text(encoding='utf-8'))
    df = fm.load_checked_sample('ma_teacher', 'v2', sample_report)
    blk, _, _ = fm.spec_block(df, 'ma_teacher', IQR_POOL_REGISTERED)
    old = json.loads(PRICING_TABLE_V2.read_text(encoding='utf-8'))['main']['ma_teacher']
    rows, max_delta = {}, 0.0
    for w in ('ew', 'vw'):
        for stat in ('beta', 't_nw'):
            new_v = blk[w]['coefs']['PeerMAXraw'][stat]
            old_v = old[w]['coefs']['PeerMAXraw'][stat]
            if new_v is None or old_v is None:
                fail(f'headline 复算空值: {w}.PeerMAXraw.{stat} new={new_v} old={old_v}')
            d = abs(float(new_v) - float(old_v))
            max_delta = max(max_delta, d)
            rows[f'{w}.PeerMAXraw.{stat}'] = {'old_registered': float(old_v),
                                              'recomputed': float(new_v), 'abs_delta': d}
    ok = max_delta <= HEADLINE_TOL
    return {'source': 'reports/data_gates/G0/s3_m1_pricing_table_v2.json main.ma_teacher',
            'spec': 'S1 主规格（PeerMAXraw + OwnMAXraw + PeerReturn + 8 controls；EW/VW；NW lag 6）',
            'in_memory_only': True, 'wrote_m1_or_pc_artifacts': False,
            'rows': rows, 'max_abs_delta': max_delta, 'tolerance': HEADLINE_TOL, 'ok': bool(ok)}


def verify_rewrite(backup_dir: Path, plans: list, with_headline: bool = True) -> dict:
    """对拍备份原件 vs 现盘；任何违反 → SystemExit(2)。"""
    out = {'script': SCRIPT_REL, 'run_at': datetime.now(timezone.utc).isoformat(),
           'backup_dir': str(backup_dir.relative_to(ROOT)), 'files': {}}
    for rec in plans:
        orig = backup_dir / Path(rec['path']).name
        if not orig.is_file():
            fail(f'验证缺少备份原件: {orig}')
        cur = ROOT / rec['path']
        if rec['kind'] == 'parquet':
            o = pd.read_parquet(orig)
            n = pd.read_parquet(cur)
            o_cols = list(o.columns)
            idem = all(c in o_cols for c in NEW_COLS)
            if idem:
                # 幂等重跑：备份原件本身已含新列（本轮之前已重写）→ 只要求列序一致
                if list(n.columns) != o_cols:
                    fail(f"{rec['path']}: 现盘列序 != 备份原件列序（幂等场景）")
                base_cols = [c for c in o_cols if c not in NEW_COLS]
            else:
                if list(n.columns) != o_cols + NEW_COLS:
                    fail(f"{rec['path']}: 现盘列序 != 原 {len(o_cols)} 列 + {NEW_COLS}（实际尾部 "
                         f"{list(n.columns)[-3:]}）")
                base_cols = o_cols
            if len(o) != len(n) or len(o) != rec['rows_before']:
                fail(f"{rec['path']}: 行数变化 {len(o)} -> {len(n)}")
            cols_ev = [_col_evidence(c, o[c], n[c]) for c in base_cols]
            worst = max((e['max_abs_delta'] or 0.0) for e in cols_ev)
            n_bad = sum(e['n_mismatch'] for e in cols_ev)
            key_ev = {'m1_in_fm_sample': _col_evidence('m1_in_fm_sample', o['m1_in_fm_sample'],
                                                       n['m1_in_fm_sample']),
                      'network_version': _col_evidence('network_version', o['network_version'],
                                                       n['network_version'])}
            fm_old = o['m1_in_fm_sample'].fillna(False).astype(bool)
            fm_new = n['m1_in_fm_sample'].fillna(False).astype(bool)
            freq = {'window_requested': n['window_requested'].dropna().unique().tolist(),
                    'sample_window_actual': n['sample_window_actual'].dropna().unique().tolist()}
            ok = (n_bad == 0 and worst == 0.0 and all(e['dtype_equal'] for e in cols_ev)
                  and freq['window_requested'] == [WINDOW_REQUESTED]
                  and freq['sample_window_actual'] == [rec['expected_sample_window_actual']]
                  and int(fm_new.sum()) == FM_EXPECTED[rec['key']][0]
                  and int(n.loc[fm_new, 'month'].nunique()) == FM_EXPECTED[rec['key']][1])
            out['files'][rec['path']] = {
                'kind': 'parquet', 'rows': int(len(n)),
                'columns_new': list(n.columns), 'n_columns_new': len(n.columns),
                'original_columns_mismatches_total': n_bad,
                'original_columns_max_abs_delta': worst,
                'key_columns_mismatch': {k: v['n_mismatch'] for k, v in key_ev.items()},
                'new_column_values': freq,
                'fm_rows': int(fm_new.sum()), 'fm_months': int(n.loc[fm_new, 'month'].nunique()),
                'fm_rows_expected': FM_EXPECTED[rec['key']][0],
                'fm_months_expected': FM_EXPECTED[rec['key']][1],
                'old_sha256': sha256_file(orig), 'new_sha256': sha256_file(cur),
                'columns': cols_ev, 'ok': bool(ok)}
            if not ok:
                fail(f"{rec['path']}: 逐列对拍失败（mismatch={n_bad} max|Δ|={worst}）")
        else:
            o = pd.read_csv(orig, dtype={'firm_id': str, 'month': str, 'return_status': str,
                                         'month_start': str, 'month_end': str,
                                         'sample_version': str, 'available_at': str})
            n = pd.read_csv(cur, dtype={'firm_id': str, 'month': str, 'return_status': str,
                                        'month_start': str, 'month_end': str,
                                        'sample_version': str, 'available_at': str})
            if list(o.columns) != CSV_COLUMNS or list(n.columns) != CSV_COLUMNS:
                fail(f'{rec["path"]}: 列名/顺序变化')
            if len(o) != CSV_EXPECTED_ROWS or len(n) != CSV_EXPECTED_ROWS:
                fail(f'{rec["path"]}: 行数变化 {len(o)} -> {len(n)}')
            cols_ev = [_col_evidence(c, o[c], n[c]) for c in CSV_COLUMNS]
            changed = [e['column'] for e in cols_ev if e['n_mismatch']]
            avail_mismatch = int((n['available_at'] != n['month_end']).sum())
            label_old = o['sample_version'].dropna().unique().tolist()
            label_new = n['sample_version'].dropna().unique().tolist()
            byte_ev = _csv_byte_evidence(orig.read_bytes(), cur.read_bytes(), 'sample_version')
            ok = (changed in ([], ['sample_version']) and label_new == [CSV_EXPECTED_LABEL]
                  and avail_mismatch == 0
                  and byte_ev['byte_identical_except_sample_version_field'])
            out['files'][rec['path']] = {
                'kind': 'csv', 'rows': int(len(n)), 'columns': list(n.columns),
                'n_columns': len(n.columns),
                'columns_with_mismatch': changed,
                'other_columns_max_abs_delta': max((e['max_abs_delta'] or 0.0) for e in cols_ev
                                                   if e['column'] != 'sample_version'),
                'sample_version_old_values': label_old,
                'sample_version_new_values': label_new,
                'available_at_equals_month_end_rows': int(len(n) - avail_mismatch),
                'forbidden_new_columns_absent': all(c not in n.columns for c in CSV_FORBIDDEN_COLS),
                'bytes': byte_ev,
                'old_sha256': sha256_file(orig), 'new_sha256': sha256_file(cur),
                'columns_evidence': cols_ev, 'ok': bool(ok)}
            if not ok:
                fail(f'{rec["path"]}: CSV 对拍失败（changed={changed}）')
    twins = {}
    for a, b in TWIN_PAIRS:
        pa, pb = ROOT / a, ROOT / b
        twins[f'{a} vs {b}'] = {'sha256_a': sha256_file(pa), 'sha256_b': sha256_file(pb),
                                'byte_identical': sha256_file(pa) == sha256_file(pb)}
        if not twins[f'{a} vs {b}']['byte_identical']:
            fail(f'孪生对不再字节相同: {a} vs {b}')
    out['twins'] = twins
    out['twins_ok'] = all(v['byte_identical'] for v in twins.values())
    if with_headline:
        hl = headline_recompute()
        out['headline'] = hl
        log(f"headline 复算 max|Δ| = {hl['max_abs_delta']:.3e}（容差 {HEADLINE_TOL:.0e}）")
        if not hl['ok']:
            fail(f"headline 系数漂移 {hl['max_abs_delta']:.3e} > {HEADLINE_TOL:.0e}")
    out['ok'] = all(v['ok'] for v in out['files'].values()) and out['twins_ok'] and out.get(
        'headline', {'ok': True})['ok']
    return out


# ------------------------------------------------------------------ CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='Phase 4 BL-01：I-16/I-17 数据 metadata 重写（D1）')
    ap.add_argument('--v2-rewrite', action='store_true', help='启用 4 个 v2 parquet + clean_v2.csv 重写')
    ap.add_argument('--dry-run', action='store_true', help='只读：打印目标列与差异摘要，不写盘')
    ap.add_argument('--apply', action='store_true', help='写盘：先备份，再临时文件 + os.replace 原子替换')
    ap.add_argument('--verify', action='store_true', help='独立验证：备份原件 vs 现盘逐列对拍 + headline 复算')
    ap.add_argument('--backup-dir', default=None,
                    help='备份目录（默认 artifacts/phase4/backup/bl01_<UTC timestamp>）')
    return ap.parse_args(argv)


def resolve_backup_dir(arg) -> Path:
    if arg:
        p = Path(arg)
        return p if p.is_absolute() else (ROOT / p)
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return DEFAULT_BACKUP_ROOT / f'bl01_{ts}'


def main(argv=None) -> int:
    args = parse_args(argv)
    if not (args.v2_rewrite or args.verify):
        fail('需显式 --v2-rewrite 或 --verify（默认不做任何事）')
    if args.v2_rewrite and args.dry_run == args.apply:
        fail('--v2-rewrite 必须且只能指定 --dry-run 或 --apply 之一')
    plans = build_plan()
    print_plan(plans)
    if args.dry_run and not args.apply:
        summary = {'mode': 'dry-run', 'wrote_files': False,
                   'files': {r['path']: r['status'] for r in plans}}
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return 0

    backup_dir = resolve_backup_dir(args.backup_dir)
    bk = ensure_backup(backup_dir, plans)
    log(f"备份目录 {backup_dir}（actions={bk['backup_actions']}）")

    if args.apply:
        for rec in plans:
            apply_one(rec)
        report = {'script': SCRIPT_REL, 'phase': 'Phase 4 BL-01（I-16/I-17 metadata-only 重写，D1）',
                  'mode': 'apply', 'run_at': datetime.now(timezone.utc).isoformat(),
                  'backup_dir': str(backup_dir.relative_to(ROOT)),
                  'pre_sha256': bk['pre_sha256'], 'backup_actions': bk['backup_actions'],
                  'files': [{k: v for k, v in r.items() if not k.startswith('_')} for r in plans],
                  'csv_no_new_columns': True,
                  'legacy_sample_window_note': '旧 sample_window 列 = 申请窗（2015-01..2023-12），'
                                               '语义未改；实际窗 = sample_window_actual（新列）',
                  'available_at_rule': 'CSV 的 available_at 逐行 = month_end（月末可得，保守）；'
                                       '本项不改其值',
                  'notes': ['不重跑 M1/PC；headline 系数不变由逐列对拍 + 只读内存复算证明',
                            '原 44 / 10 列顺序不变，新列只追加在 parquet 末尾，CSV 不新增列']}
        rep_path = backup_dir / 'rewrite_report.json'
        if rep_path.is_file():   # 幂等重跑：保留历次 apply 记录，不覆盖证据
            old = json.loads(rep_path.read_text(encoding='utf-8'))
            prior = {k: v for k, v in old.items() if k != 'previous_runs'}
            report['previous_runs'] = list(old.get('previous_runs') or []) + [prior]
        atomic_write_json(rep_path, report)
        log('rewrite_report.json 已写')
        try:
            ver = verify_rewrite(backup_dir, plans, with_headline=True)
        except SystemExit:
            log('验证失败 → 恢复备份原件（回滚）')
            restored = restore_from_backup(backup_dir, plans)
            log(f'已恢复 {len(restored)} 个文件: {restored}')
            raise
        atomic_write_json(backup_dir / 'verify_report.json', ver)
        log('验证通过；verify_report.json 已写')
        print(json.dumps({'mode': 'apply', 'backup_dir': str(backup_dir.relative_to(ROOT)),
                          'files': {r['path']: {'status': r['status'],
                                                'old_sha256': r['old_sha256'],
                                                'new_sha256': r['new_sha256']} for r in plans},
                          'verify_ok': ver['ok'],
                          'headline_max_abs_delta': ver['headline']['max_abs_delta'],
                          'twins_ok': ver['twins_ok']}, ensure_ascii=False, indent=1))
        return 0

    ver = verify_rewrite(backup_dir, plans, with_headline=True)
    atomic_write_json(backup_dir / 'verify_report.json', ver)
    print(json.dumps({'mode': 'verify', 'backup_dir': str(backup_dir.relative_to(ROOT)),
                      'ok': ver['ok'], 'headline_max_abs_delta': ver['headline']['max_abs_delta'],
                      'files': {k: v['ok'] for k, v in ver['files'].items()}},
                     ensure_ascii=False, indent=1))
    log('verify_report.json 已写')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
