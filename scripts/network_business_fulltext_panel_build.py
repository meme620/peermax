#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F1：年报全文语料面板构建（对表语料敏感性列 `ma_fulltext`，2026-10-19）。

语料真源 = `data/raw/cninfo/annual_txt/<code6>/<year>.txt`（5,406 家公司目录 / 39,778 件 / 财年 2017-2025）。

产物：
  ① `data/processed/annual_fulltext_panel.parquet`
     列 = firm_id / year / text / length / sha256（**来源文件**）/ src_file / status / reason /
          src_identity / content_code / dup_n / year_signal / doc_type / md5_match_audit / n_hits
     —— **身份与年份校验沿用既有三轮审计口径**（`artifacts/review/mda_src_identity.csv`，由
     `corpus_mda_integrity_audit.py` → `corpus_mda_identity_second_pass.py` → `corpus_mda_identity_confirm.py` 产出），
     status 判定与 `text_mda_extract.py` 的 MD&A 面板**同规则**：
        identity=mismatch 或 doc_type=announcement            -> bad_source（文本置空）
        year_signal 以 mismatch 开头                          -> suspect_source（正文声明的年度与目录名不一致）
        identity=mismatch_weak                                -> suspect_source
        dup_n>1 且 identity=unknown                           -> suspect_source
        其余                                                  -> ok
     **不一致一律打标、不静默丢弃**（非 ok 行保留在面板中、带 reason，仅在构网 `status=='ok'` 处被排除）。
  ② `data/processed/annual_fulltext_hits/{year}_hits.npz`
     逐年词条命中缓存（Aho-Corasick 全子串、`network_business_lexicon.normalize` 口径、词表 = `product_lexicon_v2.csv` 全表
     1,355,237 条），供 F2（df 重算）与 F3（构向量）复用，**避免对 18 GB 文本重复扫描**。
  ③ `reports/data_gates/G0/annual_fulltext_panel_report.json`（年份覆盖 / status 分布 / 长度分布 /
     md5 漂移核验 / 命中统计 / 停机线）。

用法：
  python scripts/network_business_fulltext_panel_build.py [--workers 10] [--years 2017 ... 2025]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

TXT = ROOT / 'data/raw/cninfo/annual_txt'
IDENT = ROOT / 'artifacts/review/mda_src_identity.csv'
LEXICON = ROOT / 'data/processed/product_lexicon_v2.csv'
IDMAP = ROOT / 'data/processed/firm_id_map.csv'
OUT_PANEL = ROOT / 'data/processed/annual_fulltext_panel.parquet'
HITS_DIR = ROOT / 'data/processed/annual_fulltext_hits'
REPORT = ROOT / 'reports/data_gates/G0/annual_fulltext_panel_report.json'
PROGRESS = ROOT / 'artifacts/tmp/fulltext_corr_progress.md'
YEARS = list(range(2017, 2026))

# 模块级（子进程继承）：词表 / 自动机 / 身份表 / firm_id 映射由 initializer 装载
_TERMS: list = []
_A = None
_ID: dict = {}
_IDMAP: dict = {}


def psay(msg: str) -> None:
    line = f'[FT-1] {msg}'
    print(line, flush=True)
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, 'a', encoding='utf-8', newline='\n') as f:
        f.write(f'- {line}\n')


def rel(p) -> str:
    """相对 ROOT 的 POSIX 路径；ROOT 外/相对路径回退为绝对路径。"""
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def to_panel_code(num: str) -> str:
    """与 text_mda_extract.to_panel_code 同规则（firm_id_map 缺失时回退）。"""
    if num.startswith(('600', '601', '603', '605', '688', '689')):
        return 'sh' + num
    if num.startswith(('000', '001', '002', '003', '300', '301', '302')):
        return 'sz' + num
    return num


def load_meta() -> tuple[dict, dict]:
    ident = {}
    with open(IDENT, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            ident[(r['code6'], int(r['year']))] = r
    fmap = {}
    if IDMAP.exists():
        for r in csv.reader(open(IDMAP, encoding='utf-8')):
            if r and r[0] != 'firm_id':
                fmap[r[1]] = r[0]
    return ident, fmap


def classify(meta: dict) -> tuple[str, str]:
    """返回 (status, reason) —— 文本置空规则与 text_mda_extract 一致（bad_source 置空）。"""
    src_identity = meta.get('identity', 'unknown') or 'unknown'
    doc_type = meta.get('doc_type', '') or ''
    year_signal = meta.get('year_signal', '') or ''
    dup_n = meta.get('dup_n', '1') or '1'
    status, reason = 'ok', ''
    if src_identity == 'mismatch' or doc_type == 'announcement':
        status, reason = 'bad_source', f'identity:{src_identity};doc:{doc_type}'
    elif str(year_signal).startswith('mismatch'):
        status, reason = 'suspect_source', f'year_signal:{year_signal}（源文件年份错位，待重下）'
    elif src_identity == 'mismatch_weak':
        status, reason = 'suspect_source', 'identity:mismatch_weak（名称确认歧义，保留文本待复核）'
    elif int(dup_n or 1) > 1 and src_identity == 'unknown':
        status, reason = 'suspect_source', f'dup_n:{dup_n};identity:unknown'
    return status, reason


def _init_worker(lex_path: str, ident_path: str, idmap_path: str):
    global _TERMS, _A, _ID, _IDMAP
    from network_business_lexicon import build_automaton
    d = pd.read_csv(lex_path, dtype=str)
    _TERMS = [str(t) for t in d['term']]
    assert len(_TERMS) == len(set(_TERMS)), '词表有重复词条'
    _A = build_automaton(_TERMS)
    _ID, _IDMAP = load_meta()


def _work(item):
    """读单文件 -> (row dict, hits int32 array)。文本置空规则与 text_mda_extract 一致。"""
    from network_business_lexicon import normalize
    code6, year, rel = item
    p = TXT / code6 / f'{year}.txt'
    try:
        b = p.read_bytes()
    except Exception as exc:                                   # pragma: no cover
        return ({'firm_id': to_panel_code(code6), 'year': year, 'text': '', 'length': 0,
                 'sha256': '', 'src_file': rel, 'status': 'read_error', 'reason': str(exc)[:200],
                 'src_identity': '', 'content_code': '', 'dup_n': 0, 'year_signal': '',
                 'doc_type': '', 'md5_match_audit': False, 'md5_audit_present': False,
                 'n_hits': -1}, np.zeros(0, np.int32))
    sha = hashlib.sha256(b).hexdigest()
    md5 = hashlib.md5(b).hexdigest()
    text = b.decode('utf-8', errors='replace')
    meta = _ID.get((code6, year), {})
    status, reason = classify(meta)
    md5_audit = meta.get('md5', '')
    md5_ok = bool(md5_audit) and md5 == md5_audit
    if status != 'ok':
        text = ''
    hits = np.zeros(0, np.int32)
    if status == 'ok' and text:
        # 集合语义（与 network_business_lexicon.match_terms 一致：重复/嵌套命中只计一次）→ unique 排序
        hits = np.unique(np.fromiter((i for i, _v in _hits_iter(text)), dtype=np.int32))
    elif status == 'ok' and not text:
        status, reason = 'bad_source', 'empty_text'
    return ({'firm_id': _IDMAP.get(code6, to_panel_code(code6)), 'year': year, 'text': text,
             'length': len(text), 'sha256': sha, 'src_file': rel, 'status': status, 'reason': reason,
             'src_identity': meta.get('identity', ''), 'content_code': meta.get('content_code', ''),
             'dup_n': int(meta.get('dup_n') or 0), 'year_signal': meta.get('year_signal', ''),
             'doc_type': meta.get('doc_type', ''), 'md5_match_audit': md5_ok,
             'md5_audit_present': bool(md5_audit),
             'n_hits': int(len(hits))}, hits)


_TIDX = None


def _hits_iter(text):
    """产出 (词条行号, 词条)；首次调用时建 term->行号表。"""
    global _TIDX
    from network_business_lexicon import normalize
    if _TIDX is None:
        _TIDX = {t: i for i, t in enumerate(_TERMS)}
    for _pos, t in _A.iter(normalize(text)):
        yield _TIDX[t], t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--years', type=int, nargs='*', default=YEARS)
    ap.add_argument('--out', default=str(OUT_PANEL))
    ap.add_argument('--hits-dir', default=str(HITS_DIR))
    ap.add_argument('--report', default=str(REPORT))
    ap.add_argument('--lexicon', default=str(LEXICON))
    ap.add_argument('--limit-files', type=int, default=0, help='仅处理前 N 件（冒烟用；0 = 全部）')
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    from multiprocessing import Pool

    t_all = time.time()
    lex_sha = hashlib.sha256(Path(args.lexicon).read_bytes()).hexdigest()
    d = pd.read_csv(args.lexicon, dtype=str)
    n_terms = len(d)
    psay(f'状态 = 开始；语料 = {TXT.relative_to(ROOT)}（年报全文）；词表 = {Path(args.lexicon).name}'
         f'（{n_terms:,} 条，sha256 {lex_sha[:12]}…）；worker = {args.workers}')

    # 文件清单：(year, code6) 排序
    items, per_year_files = [], Counter()
    for code6 in sorted(os.listdir(TXT)):
        ydir = TXT / code6
        if not ydir.is_dir():
            continue
        for y in args.years:
            p = ydir / f'{y}.txt'
            if p.exists():
                items.append((code6, y, f'data/raw/cninfo/annual_txt/{code6}/{y}.txt'))
                per_year_files[y] += 1
    items.sort(key=lambda x: (x[1], x[0]))
    if args.limit_files:
        items = items[:args.limit_files]
        per_year_files = Counter(y for _c, y, _r in items)
    psay(f'文件清单状态 = ok：{len(items):,} 件 / {len(per_year_files)} 年 '
         f'{min(per_year_files)}-{max(per_year_files)}（逐年 {dict(sorted(per_year_files.items()))}）')

    schema = pa.schema([('firm_id', pa.string()), ('year', pa.int16()), ('text', pa.string()),
                        ('length', pa.int32()), ('sha256', pa.string()), ('src_file', pa.string()),
                        ('status', pa.string()), ('reason', pa.string()), ('src_identity', pa.string()),
                        ('content_code', pa.string()), ('dup_n', pa.int16()), ('year_signal', pa.string()),
                        ('doc_type', pa.string()), ('md5_match_audit', pa.bool_()),
                        ('md5_audit_present', pa.bool_()), ('n_hits', pa.int32())])
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    hitsd = Path(args.hits_dir)
    hitsd.mkdir(parents=True, exist_ok=True)

    stats = {'per_year': {}, 'n_files': len(items), 'md5_mismatch': [], 'n_dupe_key': 0}
    writer = pq.ParquetWriter(outp, schema, compression='zstd', compression_level=3)
    seen = set()
    cur_year, cur_rows = None, []           # 面板写缓冲（按年/批刷）
    yr_firms, yr_hits, yr_rows = [], [], 0  # 当年命中缓存（**仅在换年时落盘一次**）
    n_done = 0
    t0 = time.time()

    def flush_rows():
        """把面板缓冲写盘并清空（不触碰当年命中缓存）。"""
        nonlocal cur_rows
        if cur_rows:
            writer.write_table(pa.Table.from_pylist(cur_rows, schema=schema))
            cur_rows = []

    def close_year():
        """当年结束：落盘命中缓存 npz（一次且仅一次）。"""
        nonlocal cur_year, yr_firms, yr_hits, yr_rows
        if cur_year is None:
            return
        assert len(yr_firms) == yr_rows, f'{cur_year}: 缓存行数不符'
        off = np.zeros(len(yr_firms) + 1, dtype=np.int64)
        np.cumsum([len(h) for h in yr_hits], out=off[1:])
        cat = (np.concatenate(yr_hits).astype(np.int32) if yr_hits and any(len(h) for h in yr_hits)
               else np.zeros(0, np.int32))
        np.savez(hitsd / f'{cur_year}_hits.npz', firm_id=np.array(yr_firms), offsets=off,
                 hits_idx=cat, lex_sha256=np.array(lex_sha), lex_rows=np.array(n_terms))
        psay(f'[{cur_year}] 命中缓存落盘状态 = ok（{yr_rows:,} 行 / {len(cat):,} 命中 / '
             f'{(hitsd / f"{cur_year}_hits.npz").stat().st_size / 2**20:.1f} MiB）')
        cur_year, yr_firms, yr_hits, yr_rows = None, [], [], 0

    with Pool(args.workers, initializer=_init_worker,
              initargs=(args.lexicon, str(IDENT), str(IDMAP))) as pool:
        for row, hits in pool.imap(_work, items, chunksize=8):
            if row['year'] != cur_year:
                flush_rows()
                close_year()
                cur_year = row['year']
                psay(f'[{cur_year}] 面板写入状态 = 开始（{per_year_files[cur_year]:,} 件）')
            key = (row['firm_id'], row['year'])
            if key in seen:
                stats['n_dupe_key'] += 1
            seen.add(key)
            if not row['md5_match_audit'] and row['md5_audit_present'] and row['status'] != 'read_error':
                stats['md5_mismatch'].append(row['src_file'])
                if row['status'] == 'ok':
                    row['status'], row['reason'] = 'suspect_source', 'md5_drift_vs_audit'
            cur_rows.append(row)
            yr_firms.append(row['firm_id'])
            yr_hits.append(hits)
            yr_rows += 1
            n_done += 1
            if len(cur_rows) >= 400:
                flush_rows()
            if n_done % 2000 == 0:
                el = time.time() - t0
                psay(f'进度 = {n_done:,}/{len(items):,}（{el:.0f}s，{n_done / max(el, 1e-9):.1f} 件/s）')
    flush_rows()
    close_year()
    writer.close()
    psay(f'面板落盘状态 = ok（{rel(outp)}，{outp.stat().st_size / 2**30:.2f} GiB，'
         f'{n_done:,} 行，{time.time() - t_all:.0f}s）')

    # ---------------- 复核：逐 kb 回读 + 统计 ----------------
    df = pq.read_table(outp).to_pandas()
    per_year = {}
    for y, sub in df.groupby('year'):
        st = Counter(sub['status'])
        ok = sub[sub['status'] == 'ok']
        per_year[int(y)] = {
            'n_files': int(len(sub)), 'status': dict(st), 'n_ok': int(len(ok)),
            'length_median_chars': int(ok['length'].median()) if len(ok) else None,
            'length_p10': int(ok['length'].quantile(.1)) if len(ok) else None,
            'length_p90': int(ok['length'].quantile(.9)) if len(ok) else None,
            'n_firms_ok': int(ok['firm_id'].nunique()),
            'hits_median': float(ok['n_hits'].median()) if len(ok) else None,
            'hits_mean': round(float(ok['n_hits'].mean()), 1) if len(ok) else None,
            'zero_hit_docs': int((ok['n_hits'] == 0).sum()) if len(ok) else None,
            'md5_audit_present': int(sub['md5_audit_present'].sum()),
            'md5_match_audit': int(sub['md5_match_audit'].sum()),
        }
    stats['per_year'] = per_year
    # 命中缓存与面板逐行对齐核验（行序 / 行数 / 每行命中数）
    hits_align, bad_align = {}, []
    for y in sorted(int(v) for v in df['year'].unique()):
        z = np.load(hitsd / f'{y}_hits.npz')
        sub = df[df['year'] == y]
        ok = (len(z['firm_id']) == len(sub)
              and bool((z['firm_id'] == sub['firm_id'].to_numpy()).all())
              and bool((np.diff(z['offsets']) == sub['n_hits'].to_numpy()).all())
              and str(z['lex_sha256']) == lex_sha)
        hits_align[y] = ok
        if not ok:
            bad_align.append(y)
    stats['hits_alignment'] = hits_align
    stats['hits_alignment_bad_years'] = bad_align
    stats.update({
        'corpus': 'annual_fulltext（自建年报全文面板）',
        'source_dir': 'data/raw/cninfo/annual_txt',
        'identity_source': 'artifacts/review/mda_src_identity.csv（三轮审计口径：integrity_audit → pass2 → confirm）',
        'status_rule': 'identity=mismatch 或 doc_type=announcement -> bad_source（文本置空）；'
                       'year_signal 以 mismatch 开头 / identity=mismatch_weak / dup_n>1 且 identity=unknown '
                       '-> suspect_source（保留文本、打标）；其余 ok；md5 与审计不符 -> suspect_source(md5_drift_vs_audit)',
        'no_silent_drop': True,
        'empty_text_docs': int(((df['status'] == 'bad_source') & (df['reason'] == 'empty_text')).sum()),
        'n_rows': int(len(df)), 'n_ok': int((df['status'] == 'ok').sum()),
        'status_total': dict(Counter(df['status'])),
        'n_md5_mismatch': len(stats['md5_mismatch']),
        'n_md5_audit_missing': int((~df['md5_audit_present']).sum()),
        'n_dupe_firm_year_key': int(stats['n_dupe_key']),
        'panel_bytes': int(outp.stat().st_size),
        'hits_lexicon': {'path': f'data/processed/{Path(args.lexicon).name}', 'n_terms': n_terms,
                         'sha256': lex_sha},
        'hits_dir': rel(hitsd),
        'hits_bytes': int(sum(f.stat().st_size for f in hitsd.glob('*_hits.npz'))),
        'year_coverage': {'min': int(df['year'].min()), 'max': int(df['year'].max()),
                          'years': sorted(int(y) for y in df['year'].unique())},
        'run_at': datetime.now(timezone.utc).isoformat(),
        'elapsed_sec': round(time.time() - t_all, 1),
    })
    stats['stop_condition'] = {
        'rule': '身份/年份打标不静默丢弃 + 面板可复算（sha256 逐件）+ 零命中文档占比可解释；'
                '对表可用年（PC 可得 = 文本年 2017-2022）>= 3',
        'pc_usable_years': [y for y in (2017, 2018, 2019, 2020, 2021, 2022) if y in per_year],
        'pass': len([y for y in (2017, 2018, 2019, 2020, 2021, 2022) if y in per_year]) >= 3,
    }
    stats['status'] = 'PASS' if (stats['n_md5_mismatch'] == 0 and stats['n_dupe_key'] == 0
                                 and not bad_align and stats['stop_condition']['pass']) else 'FAIL'
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding='utf-8')
    psay(f'状态 = {stats["status"]}：面板 {len(df):,} 行 / ok {(df["status"] == "ok").sum():,} / '
         f'打标 {len(df) - (df["status"] == "ok").sum():,}；md5 漂移 {stats["n_md5_mismatch"]}；'
         f'年份 {stats["year_coverage"]["min"]}-{stats["year_coverage"]["max"]}；'
         f'PC 可用年 {stats["stop_condition"]["pc_usable_years"]}')
    print(json.dumps({k: v for k, v in stats.items() if k != 'per_year'}, ensure_ascii=False, indent=1))
    return 0 if stats['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
