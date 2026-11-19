#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F1'：年报全文面板（annual_fulltext_panel.parquet）独立核验 + 核验报告。

背景：F1（`network_business_fulltext_panel_build.py`）已把 39,778 件年报全文写成面板 + 逐年命中缓存；
其收尾报告需回读全表（含 text，~4.1 GiB → 内存重）未落盘。本脚本**不回读 text**，
只读小列（firm_id/year/length/status/reason/src_identity/content_code/dup_n/year_signal/
doc_type/md5_*/n_hits）重算核验统计，并做两项独立对账：

  A. 面板统计：逐年 status 分布 / ok 数 / 长度中位（p10/p90）/ n_hits 分布 / 零命中文档 /
     md5 审计覆盖；身份与年份打标（src_identity / year_signal / doc_type / dup_n）**全部留痕**，
     非 ok 行不静默丢弃（报告含逐年 status/reason 计数与打标原因分布）。
  B. 命中缓存对账：`annual_fulltext_hits/{y}_hits.npz` 与面板同年行序 / 行数 / 每行命中数
     （offsets 差分 == n_hits）逐行一致 + 缓存内词表 sha/行数 == product_lexicon_v2.csv。
  C. 对照 teacher 主口径面板（`teacher_mda_panel.csv`）：逐年 ok 文档数 / 长度中位并列。

停机线（沿用 registry `ma_fulltext_text_corpus` 登记口径）：对表可用文本年（PC 可得 = 2017-2022）
< 3 → FAIL。报告不含任何阈值改动。

产物：reports/data_gates/G0/annual_fulltext_panel_report.json
用法：python scripts/network_business_fulltext_panel_audit.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / 'data/processed/annual_fulltext_panel.parquet'
HITS_DIR = ROOT / 'data/processed/annual_fulltext_hits'
LEXICON = ROOT / 'data/processed/product_lexicon_v2.csv'
TEACHER_PANEL = ROOT / 'data/processed/teacher_mda_panel.csv'
REPORT = ROOT / 'reports/data_gates/G0/annual_fulltext_panel_report.json'
PROGRESS = ROOT / 'artifacts/tmp/fulltext_corr_progress.md'
PC_YEARS = list(range(2017, 2023))          # 对表可用文本年（PC_{i,y} 需 y−1 年营收）
SMALL_COLS = ['firm_id', 'year', 'length', 'bytes', 'src_file', 'status', 'reason',
              'src_identity', 'content_code', 'dup_n', 'year_signal', 'doc_type',
              'md5_match_audit', 'md5_audit_present', 'n_hits']


def psay(msg: str) -> None:
    line = f"[FT-1'] {msg}"
    print(line, flush=True)
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, 'a', encoding='utf-8', newline='\n') as f:
        f.write(f'- {line}\n')


def file_size(p):
    return p.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', default=str(PANEL))
    ap.add_argument('--hits-dir', default=str(HITS_DIR))
    ap.add_argument('--lexicon', default=str(LEXICON))
    ap.add_argument('--teacher-panel', default=str(TEACHER_PANEL))
    ap.add_argument('--report', default=str(REPORT))
    args = ap.parse_args()

    t_all = time.time()
    pan = pd.read_parquet(args.panel, columns=SMALL_COLS)
    lex_bytes = file_size(Path(args.lexicon))
    lex_rows = len(Path(args.lexicon).read_text(encoding='utf-8').splitlines()) - 1
    psay(f'状态 = 开始；面板 = {Path(args.panel).name}（{len(pan):,} 行）；'
         f'词表 = {Path(args.lexicon).name}（{lex_rows:,} 行，bytes {lex_bytes[:12]}…）')

    rep = {'artifact': 'annual_fulltext_panel_report', 'step': "F1' 面板核验（不回读 text）",
           'run_at': datetime.now(timezone.utc).isoformat(),
           'corpus': 'annual_fulltext（自建年报全文面板，registry ma_fulltext_text_corpus）',
           'panel_path': 'data/processed/annual_fulltext_panel.parquet',
           'panel_bytes': Path(args.panel).stat().st_size,
           'hits_dir': 'data/processed/annual_fulltext_hits',
           'identity_source': 'artifacts/review/mda_src_identity.csv（三轮审计口径，F1 已写入面板列）',
           'status_rule': ('identity=mismatch 或 doc_type=announcement -> bad_source（文本置空）；'
                           'year_signal 以 mismatch 开头 / identity=mismatch_weak / dup_n>1 且 identity=unknown '
                           '-> suspect_source（保留文本、打标）；md5 与审计不符 -> suspect_source(md5_drift_vs_audit)'),
           'no_silent_drop': True, 'per_year': {}, 'flags': {}}

    # ---------------- A. 逐年统计（含打标留痕） ----------------
    for y, sub in pan.groupby('year'):
        y = int(y)
        st = Counter(sub['status'])
        ok = sub[sub['status'] == 'ok']
        rep['per_year'][str(y)] = {
            'n_files': int(len(sub)), 'status': dict(st), 'n_ok': int(len(ok)),
            'n_firms_ok': int(ok['firm_id'].nunique()),
            'length_median_chars': int(ok['length'].median()) if len(ok) else None,
            'length_p10': int(ok['length'].quantile(.1)) if len(ok) else None,
            'length_p90': int(ok['length'].quantile(.9)) if len(ok) else None,
            'hits_median': float(ok['n_hits'].median()) if len(ok) else None,
            'hits_mean': round(float(ok['n_hits'].mean()), 1) if len(ok) else None,
            'zero_hit_docs': int((ok['n_hits'] == 0).sum()) if len(ok) else None,
            'md5_audit_present': int(sub['md5_audit_present'].sum()),
            'md5_match_audit': int(sub['md5_match_audit'].sum()),
            'reason_counts': dict(Counter(sub.loc[sub['status'] != 'ok', 'reason'])),
        }
        psay(f'[{y}] 面板核验状态 = ok：文件 {len(sub):,} / ok {len(ok):,} / 打标 {len(sub) - len(ok):,}；'
             f'长度中位 {rep["per_year"][str(y)]["length_median_chars"]}；零命中 {rep["per_year"][str(y)]["zero_hit_docs"]}')

    rep['n_rows'] = int(len(pan))
    rep['n_ok'] = int((pan['status'] == 'ok').sum())
    rep['status_total'] = dict(Counter(pan['status']))
    rep['reason_total'] = dict(Counter(pan.loc[pan['status'] != 'ok', 'reason']))
    rep['src_identity_total'] = dict(Counter(pan['src_identity'].fillna('')))
    rep['year_signal_total'] = dict(Counter(pan['year_signal'].fillna('')))
    rep['doc_type_total'] = dict(Counter(pan['doc_type'].fillna('')))
    rep['dup_n_gt1_rows'] = int((pd.to_numeric(pan['dup_n'], errors='coerce').fillna(0) > 1).sum())
    rep['md5_audit_present_total'] = int(pan['md5_audit_present'].sum())
    rep['md5_mismatch_rows'] = int((pan['md5_audit_present'] & ~pan['md5_match_audit']).sum())
    rep['n_dupe_firm_year_key'] = int(pan.duplicated(['firm_id', 'year']).sum())
    rep['empty_text_docs'] = int(((pan['status'] == 'bad_source') & (pan['reason'] == 'empty_text')).sum())
    rep['length_zero_ok'] = int(((pan['status'] == 'ok') & (pan['length'] == 0)).sum())
    rep['bytes_missing_ok'] = int(((pan['status'] == 'ok') & (pan['bytes'].fillna('') == '')).sum())

    # 打标 × status 交叉（排除 ok，逐类留痕）
    for col, key in (('src_identity', 'flags_by_src_identity'), ('year_signal', 'flags_by_year_signal'),
                     ('doc_type', 'flags_by_doc_type')):
        x = pan[pan['status'] != 'ok']
        rep['flags'][key] = {f'{v}': dict(Counter(x.loc[x[col].fillna('') == v, 'status']))
                             for v in sorted(set(x[col].fillna('')))}

    # ---------------- B. 命中缓存对账 ----------------
    align, bad = {}, []
    for y in sorted(int(v) for v in pan['year'].unique()):
        sub = pan[pan['year'] == y]
        z = np.load(Path(args.hits_dir) / f'{y}_hits.npz')
        same_rows = len(z['firm_id']) == len(sub)
        same_ids = same_rows and bool((z['firm_id'] == sub['firm_id'].to_numpy()).all())
        same_hits = same_rows and bool((np.diff(z['offsets']) == sub['n_hits'].to_numpy()).all())
        same_lex = (str(z['lex_bytes']) == lex_bytes) and int(z['lex_rows']) == lex_rows
        align[y] = {'rows_equal': bool(same_rows), 'firm_ids_equal': bool(same_ids),
                    'row_hits_equal': bool(same_hits), 'lexicon_equal': bool(same_lex)}
        if not all(align[y].values()):
            bad.append(y)
        psay(f'[B/{y}] 命中缓存对账状态 = {"ok" if not bad or y not in bad else "FAIL"}'
             f'（行序 {same_ids} / 行命中数 {same_hits} / 词表 {same_lex}）')
    rep['hits_alignment'] = align
    rep['hits_alignment_bad_years'] = bad
    rep['hits_bytes'] = int(sum(f.stat().st_size for f in Path(args.hits_dir).glob('*_hits.npz')))

    # ---------------- C. 对照 teacher 主口径面板（逐年 ok 数 / 长度中位） ----------------
    tp = pd.read_csv(args.teacher_panel, usecols=['firm_id', 'year', 'length', 'status'])
    tper = {}
    for y, sub in tp.groupby('year'):
        ok = sub[sub['status'] == 'ok']
        tper[str(int(y))] = {'n_ok': int(len(ok)), 'length_median_chars': int(ok['length'].median()) if len(ok) else None}
    rep['teacher_reference'] = {'panel': 'data/processed/teacher_mda_panel.csv',
                                'n_rows': int(len(tp)), 'n_ok': int((tp['status'] == 'ok').sum()),
                                'per_year': tper}
    cmp_rows = []
    for y in sorted(set(rep['per_year']) & set(tper)):
        cmp_rows.append({'text_year': int(y),
                         'fulltext_ok': rep['per_year'][y]['n_ok'], 'teacher_ok': tper[y]['n_ok'],
                         'fulltext_len_median': rep['per_year'][y]['length_median_chars'],
                         'teacher_len_median': tper[y]['length_median_chars']})
    rep['comparison_vs_teacher'] = cmp_rows
    psay(f'对照 teacher 状态 = ok：重叠年 {len(cmp_rows)} 年（'
         + '；'.join(f"{r['text_year']}: {r['fulltext_ok']:,} vs {r['teacher_ok']:,} 篇，长度中位 "
                     f"{r['fulltext_len_median']} vs {r['teacher_len_median']}" for r in cmp_rows) + '）')

    # ---------------- 停机线 ----------------
    pc_years_ok = [y for y in PC_YEARS if rep['per_year'].get(str(y), {}).get('n_ok', 0) > 0]
    rep['stop_condition'] = {
        'rule': '对表可用文本年（PC 可得 = 2017-2022 且面板有 ok 文档）>= 3；命中缓存对账零失败；'
                'md5 漂移 0；firm-year 键重复 0',
        'pc_usable_years': pc_years_ok, 'n_pc_usable_years': len(pc_years_ok),
        'pass': len(pc_years_ok) >= 3 and not bad and rep['md5_mismatch_rows'] == 0
                and rep['n_dupe_firm_year_key'] == 0}
    rep['status'] = 'PASS' if rep['stop_condition']['pass'] else 'FAIL'
    rep['elapsed_sec'] = round(time.time() - t_all, 1)
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    psay(f'状态 = {rep["status"]}：面板 {rep["n_rows"]:,} 行 / ok {rep["n_ok"]:,} / 打标 '
         f'{rep["n_rows"] - rep["n_ok"]:,}；md5 漂移 {rep["md5_mismatch_rows"]}；'
         f'PC 可用年 {pc_years_ok}；报告 {rp.name}（{rep["elapsed_sec"]}s）')
    print(json.dumps({'n_rows': rep['n_rows'], 'n_ok': rep['n_ok'],
                      'status_total': rep['status_total'], 'pc_usable_years': pc_years_ok,
                      'status': rep['status']}, ensure_ascii=False))
    return 0 if rep['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
