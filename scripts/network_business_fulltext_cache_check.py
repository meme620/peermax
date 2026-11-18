#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F3 前置：全文语料命中缓存 × 面板 的独立核验（**中间证据，进 F3 报告**）。

背景：F1 把「全子串 Aho-Corasick 命中（词表 = `product_lexicon_v2.csv` 全表 1,355,237 条）」
按词表行号存成 `annual_fulltext_hits/{year}_hits.npz`，F2（df 重算）与 F3（构向量）复用该缓存
以免对 18 GB 文本重复扫描。本脚本给出**缓存 == 现场重算**的独立证据：

  A. 面板往返（parquet 逐字）：抽 200 篇/年（seed 固定）——**status 感知判据**（v2，2026-10-19 判据修订）：
     `status=='ok'` 的行要求 `text` 与磁盘源文件 `.txt` 解码结果**逐字符相等**、`length` = 字符数、
     `sha256` = 源文件 sha256 复算一致（三重）；`status!='ok'` 的行（bad_source / suspect_source）
     要求 `text==''` 且 `length==0`（**既有置空规则**：`network_business_fulltext_panel_build._work` 对非 ok 行
     text 置空，承袭 `text_mda_extract` 的 bad_source 置空）+ `sha256` = 源文件字节哈希。
     修订原因（R1 定性，见 `artifacts/tmp/fulltext_corr_progress.md` [FT-3a/R1]）：首跑 6 年 FAIL 的
     15 行**全部**为 `status!='ok'` 行，其源文件**非空**（295 KB–851 KB），面板 text 空是**设计性置空**
     而非漂移；原判据（对所有抽检行要求 text 逐字相等）把设计性置空误判为失败。**冻结值零改动**。
  B. 命中缓存 vs 现场重算（**同一实现、两处执行**）：用 F1 的 worker（`network_business_fulltext_panel_build._work`，
     读磁盘源文件 → normalize → Aho-Corasick）现场重算命中集合，与缓存 npz 逐行比对
     （集合语义 int32 排序数组，**要求逐元素相等**）：
        · 文本年 2017–2022（对表用全部 6 年）：**全部 ok 文档**逐篇核验；
        · 文本年 2023–2025（只供 df 表）：每年 500 篇抽检（seed 固定）。
  C. 向量/相似度等价：对 A/B 覆盖年，用缓存与「现场重算的临时缓存」分别走
     `network_business_cluster_grid.build_vectors_from_hits`，比较 `firms` 序列、X 稀疏结构与非零值
     （**要求逐比特相等**）→ 传导到余弦矩阵与聚类输入。

v2（2026-10-19，断点续跑加固；**判据仅 A 段按上述 status 感知修订，B/C 段一字未改**）：
  ① 内存加固：建 Pool 前释放父进程词表副本（首跑 12 worker 建池时曾 `_ArrayMemoryError` 中断）；
  ② B 段**逐年可续跑**：现场重算结果逐年落 `artifacts/tmp/ft_corr/live_hits/{y}_hits.npz`
     （含 `verified` 逐行标记 = 该行是否现场重算），已存在则跳过（`--force` 覆盖），
     中断后重跑不重做已完成年；C 段直接复用该缓存。

产物：`reports/data_gates/G0/s3_ma_fulltext_cache_verification.json`
用法：
  python scripts/network_business_fulltext_cache_check.py [--workers 10] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import network_business_fulltext_panel_build as PB  # noqa: E402
from network_business_cluster_grid import CORPORA, build_vectors_from_hits, load_refs  # noqa: E402

PANEL = ROOT / 'data/processed/annual_fulltext_panel.parquet'
HITS_DIR = ROOT / 'data/processed/annual_fulltext_hits'
TMP_LIVE = ROOT / 'artifacts/tmp/ft_corr/live_hits'
REPORT = ROOT / 'reports/data_gates/G0/s3_ma_fulltext_cache_verification.json'
PROGRESS = ROOT / 'artifacts/tmp/fulltext_corr_progress.md'
FULL_YEARS = [2017, 2018, 2019, 2020, 2021, 2022]
SAMPLE_YEARS = [2023, 2024, 2025]
SEED = 20260727
N_SAMPLE = 500
N_ROUNDTRIP_PER_YEAR = 200


def psay(msg: str) -> None:
    line = f'[FT-3a] {msg}'
    print(line, flush=True)
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, 'a', encoding='utf-8', newline='\n') as f:
        f.write(f'- {line}\n')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--lexicon', default=str(PB.LEXICON))
    ap.add_argument('--force', action='store_true', help='忽略 live_hits 既有缓存，全部现场重算')
    ap.add_argument('--rows-check', action='store_true',
                    help='追加 D 段：对 B 段清单（2017-2022 全部 ok + 2023-2025 各 500）现场重算**文本/长度/sha256/status** '
                         '并与面板逐行对账（空源行单列）')
    args = ap.parse_args()
    from multiprocessing import Pool

    t_all = time.time()
    lex = pd.read_csv(args.lexicon, dtype=str)
    terms = [str(t) for t in lex['term']]
    keep_path = CORPORA['ma_fulltext']['lexicon']          # 全文语料词表（df 0.25）
    psay(f'状态 = 开始；核验对象 = {PANEL.name} + {HITS_DIR.name}/ ；worker = {args.workers}')

    pan = pd.read_parquet(PANEL, columns=['firm_id', 'year', 'status', 'length', 'sha256', 'src_file'])
    out = {'artifact': 's3_ma_fulltext_cache_verification', 'run_at': datetime.now(timezone.utc).isoformat(),
           'purpose': 'F3 前置：命中缓存与面板的独立核验（缓存 == 现场重算）',
           'script_version': 'v2 (2026-10-19)',
           'criterion_A': 'status 感知（v2 修订）：status==ok 行 text/length/sha256 三重逐字节；'
                          "status!=ok 行 text=='' 且 length==0（既有置空规则）+ sha256 = 源文件字节哈希",
           'criterion_A_revision_reason': '首跑 6 年 FAIL 的 15 行全部为 status!=ok 行，源文件非空'
                                          '（295 KB–851 KB），面板 text 空为设计性置空而非漂移 → '
                                          '原判据把设计性置空误判为失败；冻结值零改动'
                                          '（证据见 artifacts/tmp/fulltext_corr_progress.md [FT-3a/R1]）',
           'corpus_usable_set': "status=='ok'（39,440 行）；非 ok 338 行（bad_source 196 / suspect_source 142）"
                                '已在 build_vectors_from_hits 按 status 排除，不入网/不参与 df 与 PC',
           'empty_source_docs': 'F1 报告 empty_text_docs = 0（无 0 字节/近空源文件）',
           'seed': SEED, 'full_verify_years': FULL_YEARS, 'sample_verify_years': SAMPLE_YEARS,
           'n_sample_per_year': N_SAMPLE, 'n_roundtrip_per_year': N_ROUNDTRIP_PER_YEAR,
           'A_panel_roundtrip': {}, 'B_hits_cache_vs_live': {}, 'C_vectors_bitwise_equal': {}}

    # ---------------- A. 面板往返（status 感知：ok 行 text/length/sha256 三重；非 ok 行置空规则 + sha256）----------------
    rng = np.random.default_rng(SEED)
    bad = []
    n_checked = 0
    import pyarrow.parquet as pq
    for y in sorted(int(v) for v in pan['year'].unique()):
        sub = pan[pan['year'] == y].reset_index(drop=True)
        pick = rng.choice(len(sub), size=min(N_ROUNDTRIP_PER_YEAR, len(sub)), replace=False)
        # text 列按年过滤读入（行组统计剪枝；行序与 sub 同 = 文件序），核验后即释放（避免整表 text 入内存）
        suby = pq.read_table(PANEL, columns=['src_file', 'text'],
                             filters=[('year', '==', y)]).to_pandas()
        assert len(suby) == len(sub) and bool((suby['src_file'] == sub['src_file']).all()), \
            f'{y}: 按年读入行序与面板不一致'
        n_ok = ok_txt = ok_len = ok_sha = 0
        n_flag = flag_blank = flag_len0 = flag_sha = 0
        for i in pick:
            r = sub.iloc[i]
            r_text = suby['text'].iloc[i]
            r_text = '' if not isinstance(r_text, str) else r_text
            b = (PB.TXT / Path(r['src_file']).parent.name / f"{y}.txt").read_bytes()
            t = b.decode('utf-8', errors='replace')
            sha_ok = hashlib.sha256(b).hexdigest() == r['sha256']
            if r['status'] == 'ok':                       # 有文本的行 → 逐字节一致（三重）
                n_ok += 1
                ok_txt += int(t == r_text)
                ok_len += int(len(t) == int(r['length']))
                ok_sha += int(sha_ok)
            else:                                         # 非 ok 行 → 置空规则 + 源字节哈希
                n_flag += 1
                flag_blank += int(r_text == '')
                flag_len0 += int(int(r['length']) == 0)
                flag_sha += int(sha_ok)
            if not sha_ok:
                bad.append({'year': y, 'src_file': r['src_file'], 'status': r['status'],
                            'sha256_db': r['sha256']})
            n_checked += 1
        y_ok = (ok_txt == n_ok and ok_len == n_ok and ok_sha == n_ok
                and flag_blank == n_flag and flag_len0 == n_flag and flag_sha == n_flag)
        out['A_panel_roundtrip'][str(y)] = {'n_checked': int(len(pick)), 'n_ok_rows': n_ok,
                                           'n_flagged_rows': n_flag,
                                           'ok_text_equal': ok_txt, 'ok_length_equal': ok_len,
                                           'ok_sha256_equal': ok_sha,
                                           'flagged_text_blank': flag_blank,
                                           'flagged_length_zero': flag_len0,
                                           'flagged_sha256_equal': flag_sha}
        psay(f'[A/{y}] 面板往返状态 = {"ok" if y_ok else "FAIL"}'
             f'（抽检 {len(pick)} 篇：ok 行 {n_ok} 三重 text {ok_txt}/length {ok_len}/sha256 {ok_sha}；'
             f'非 ok 行 {n_flag} 置空 {flag_blank}/零长 {flag_len0}/sha256 {flag_sha}）')
    out['A_panel_roundtrip_bad'] = bad
    out['A_n_checked_total'] = n_checked

    # ---------------- B. 缓存 vs 现场重算（命中集合逐行；逐年可续跑）----------------
    items_by_year, per_year_info = {}, {}
    for y in sorted(int(v) for v in pan['year'].unique()):
        sub = pan[pan['year'] == y]
        z = np.load(HITS_DIR / f'{y}_hits.npz')
        assert len(z['firm_id']) == len(sub)
        ok_idx = np.flatnonzero((sub['status'] == 'ok').to_numpy())
        if y in FULL_YEARS:
            sel = ok_idx
            mode = 'full'
        else:
            sel = np.sort(rng.choice(ok_idx, size=min(N_SAMPLE, len(ok_idx)), replace=False))
            mode = 'sample'
        per_year_info[y] = {'mode': mode, 'n_ok': int(len(ok_idx)), 'n_verify': int(len(sel)),
                            'rows': sub, 'z': z, 'sel': sel}
        items_by_year[y] = [(Path(sub['src_file'].iloc[i]).parent.name, y, sub['src_file'].iloc[i], y, i)
                            for i in sel]
    psay(f'[B] 现场重算清单状态 = ok：待核验 {sum(len(v) for v in items_by_year.values()):,} 篇'
         f'（2017-2022 全部 ok 文档 + 2023-2025 各 {N_SAMPLE} 篇抽检）')

    TMP_LIVE.mkdir(parents=True, exist_ok=True)
    todo = [y for y in per_year_info if args.force or not (TMP_LIVE / f'{y}_hits.npz').exists()]
    psay(f'[B] 续跑状态 = ok：需现场重算 {todo if todo else "无（全部命中已有缓存）"}；'
         f'已有 {sorted(set(per_year_info) - set(todo))}')
    if todo:
        del lex, terms                      # 内存加固：建池前释放父进程词表副本
        with Pool(args.workers, initializer=PB._init_worker,
                  initargs=(args.lexicon, str(PB.IDENT), str(PB.IDMAP))) as pool:
            for y in todo:
                t0 = time.time()
                info = per_year_info[y]
                res = pool.map(PB._work, [(c, yy, r) for c, yy, r, _y, _i in items_by_year[y]],
                               chunksize=16)
                live = {int(i): h for (_c, _y, _r, _yy, i), (_row, h) in zip(items_by_year[y], res)}
                rows, z = info['rows'], info['z']
                n = len(rows)
                hh, ver = [], np.zeros(n, dtype=np.uint8)
                for i in range(n):
                    v = live.get(i)
                    if v is None:               # 抽检模式未现场重算的行 → 用缓存填充（该行不参与 B/C 比较）
                        v = np.unique(z['hits_idx'][z['offsets'][i]:z['offsets'][i + 1]])
                    else:
                        ver[i] = 1
                    hh.append(v)
                off = np.zeros(n + 1, dtype=np.int64)
                np.cumsum([len(v) for v in hh], out=off[1:])
                cat = (np.concatenate(hh).astype(np.int32) if any(len(v) for v in hh)
                       else np.zeros(0, np.int32))
                np.savez(TMP_LIVE / f'{y}_hits.npz',
                         firm_id=np.array(rows['firm_id'].to_numpy(), dtype=str),
                         offsets=off, hits_idx=cat, verified=ver,
                         lex_sha256=np.array(str(z['lex_sha256'])), lex_rows=z['lex_rows'])
                psay(f'[B/{y}] 现场重算状态 = ok：重算 {int(ver.sum()):,} 篇'
                     f'（缓存填充 {n - int(ver.sum()):,} 行），落盘 {y}_hits.npz（{time.time() - t0:.0f}s）')

    for y, info in per_year_info.items():
        z, sel = info['z'], info['sel']
        zl = np.load(TMP_LIVE / f'{y}_hits.npz')
        # 逐行切片前**一次性取出大数组**（NpzFile 每次 __getitem__ 都会重新解压 → 循环内取会退化为 O(n²)）
        zi, zli = z['hits_idx'], zl['hits_idx']
        assert len(zl['firm_id']) == len(info['rows'])
        off, offl, ver = z['offsets'], zl['offsets'], zl['verified']
        n_match = n_mismatch = n_skip = 0
        first_mismatch = None
        for i in sel:
            if not ver[i]:
                n_skip += 1
                continue
            cached = np.unique(zi[off[i]:off[i + 1]])
            live = np.unique(zli[offl[i]:offl[i + 1]])
            if cached.shape == live.shape and np.array_equal(cached, live):
                n_match += 1
            else:
                n_mismatch += 1
                if first_mismatch is None:
                    sc = set(cached.tolist())
                    sl = set(live.tolist())
                    first_mismatch = {'row': int(i), 'n_cached': int(len(cached)), 'n_live': int(len(live)),
                                      'n_only_cached': len(sc - sl), 'n_only_live': len(sl - sc)}
        out['B_hits_cache_vs_live'][str(y)] = {'mode': info['mode'], 'n_ok_docs': info['n_ok'],
                                              'n_verified': n_match + n_mismatch, 'n_equal': n_match,
                                              'n_mismatch': n_mismatch, 'n_skipped': n_skip,
                                              'first_mismatch': first_mismatch}
        psay(f'[B/{y}] 命中缓存核验状态 = {"ok" if n_mismatch == 0 and (n_match + n_mismatch) == info["n_verify"] else "FAIL"}：'
             f'核验 {n_match + n_mismatch:,}/{info["n_verify"]:,} 篇（{info["mode"]}），'
             f'相等 {n_match:,} / 不等 {n_mismatch} / 跳过 {n_skip}')

    # ---------------- C. 向量逐比特等价（缓存 vs 现场临时缓存）----------------
    from network_business_lexicon import build_automaton, load_lexicon
    kept_terms, _lmeta = load_lexicon(Path(keep_path).resolve())
    A = build_automaton(kept_terms)
    uni = load_refs()
    spec = CORPORA['ma_fulltext']
    orig_hits_dir = spec['hits_dir']
    try:
        for y in sorted(per_year_info):
            spec['hits_dir'] = HITS_DIR
            firms_c, Xc, stc = build_vectors_from_hits(y, A, uni, 'ma_fulltext')
            spec['hits_dir'] = TMP_LIVE
            firms_l, Xl, stl = build_vectors_from_hits(y, A, uni, 'ma_fulltext')
            same_firms = firms_c == firms_l
            same_struct = (Xc.shape == Xl.shape
                           and np.array_equal(Xc.indptr, Xl.indptr) and np.array_equal(Xc.indices, Xl.indices)
                           and np.array_equal(Xc.data, Xl.data))
            same_stats = {k: stc[k] for k in ('docs', 'drop_menlei', 'drop_ipo', 'drop_min',
                                              'firms_kept', 'active_vocab', 'mean_terms')} == \
                         {k: stl[k] for k in ('docs', 'drop_menlei', 'drop_ipo', 'drop_min',
                                              'firms_kept', 'active_vocab', 'mean_terms')}
            out['C_vectors_bitwise_equal'][str(y)] = {
                'n_firms': len(firms_c), 'nnz': int(Xc.nnz), 'same_firms': bool(same_firms),
                'same_csr_structure_and_values': bool(same_struct), 'same_filter_stats': bool(same_stats),
                'filter_stats': {k: stc[k] for k in ('docs', 'drop_menlei', 'drop_ipo', 'drop_min',
                                                     'firms_kept', 'active_vocab', 'mean_terms')},
                'C_max_abs_diff': 0.0 if same_struct else None,
            }
            psay(f'[C/{y}] 向量逐比特等价状态 = {"ok" if (same_firms and same_struct and same_stats) else "FAIL"}'
                 f'（n_firms {len(firms_c)}，nnz {int(Xc.nnz)}）')
    finally:
        spec['hits_dir'] = orig_hits_dir

    # ---------------- D. 同源重算的 text/length/sha256/status 与面板逐行对账（--rows-check；空源行单列）----------------
    okD = None
    if args.rows_check:
        items_rows = []
        for y, info in per_year_info.items():
            sub = info['rows']
            for i in info['sel']:
                items_rows.append((Path(sub['src_file'].iloc[i]).parent.name, y,
                                   sub['src_file'].iloc[i], y, i))
        psay(f'[D] 行级对账清单状态 = ok：待现场重算 {len(items_rows):,} 篇（同 B 段清单）')
        try:
            del lex, terms
        except NameError:
            pass
        with Pool(args.workers, initializer=PB._init_worker,
                  initargs=(args.lexicon, str(PB.IDENT), str(PB.IDMAP))) as pool:
            res = pool.map(PB._work, [(c, yy, r) for c, yy, r, _y, _i in items_rows], chunksize=16)
        per_year_rows, mism = {}, []
        # 面板 text 列按年读入（行组剪枝；行序与 per_year_info[y]['rows'] 同 = 文件序），核验后释放
        pan_text = {}
        for y in sorted(per_year_info):
            suby = pq.read_table(PANEL, columns=['src_file', 'text'],
                                 filters=[('year', '==', y)]).to_pandas()
            assert len(suby) == len(per_year_info[y]['rows']) and \
                bool((suby['src_file'] == per_year_info[y]['rows']['src_file']).all())
            pan_text[y] = suby['text'].to_numpy()
            del suby
        for (_c, _y, _r, y, i), (row, _hits) in zip(items_rows, res):
            if y not in per_year_rows:
                per_year_rows[y] = {'n': 0, 'text_eq': 0, 'len_eq': 0, 'sha_eq': 0, 'status_eq': 0,
                                    'src_empty': 0, 'src_empty_rows': []}
            r = per_year_info[y]['rows'].iloc[i]
            pt = pan_text[y][i]
            pt = pt if isinstance(pt, str) else ''
            c = per_year_rows[y]
            c['n'] += 1
            t_ok = (row['text'] == pt)
            l_ok = int(row['length']) == int(r['length'])
            s_ok = row['sha256'] == r['sha256']
            st_ok = row['status'] == r['status']
            c['text_eq'] += int(t_ok); c['len_eq'] += int(l_ok); c['sha_eq'] += int(s_ok)
            c['status_eq'] += int(st_ok)
            if row['text'] == '':                       # 空源行（现场重算文本为空）单列
                c['src_empty'] += 1
                if len(c['src_empty_rows']) < 20:
                    c['src_empty_rows'].append({'row': int(i), 'src_file': r['src_file'],
                                                'panel_status': r['status']})
            if not (t_ok and l_ok and s_ok and st_ok) and len(mism) < 20:
                mism.append({'year': int(y), 'row': int(i), 'src_file': r['src_file'],
                             'text_ok': bool(t_ok), 'length_ok': bool(l_ok), 'sha_ok': bool(s_ok),
                             'status_ok': bool(st_ok), 'panel_status': r['status'],
                             'live_status': row['status'],
                             'live_len': int(row['length']), 'panel_len': int(r['length'])})
        out['D_rows_recompute'] = {str(y): {**v, 'src_empty_rows': v['src_empty_rows']}
                                  for y, v in sorted(per_year_rows.items())}
        out['D_mismatches_first20'] = mism
        okD = all(v['text_eq'] == v['n'] and v['len_eq'] == v['n'] and v['sha_eq'] == v['n']
                  and v['status_eq'] == v['n'] for v in per_year_rows.values()) and not mism
        for y, v in sorted(per_year_rows.items()):
            psay(f'[D/{y}] 行级对账状态 = {"ok" if (v["text_eq"] == v["len_eq"] == v["sha_eq"] == v["status_eq"] == v["n"]) else "FAIL"}'
                 f'（{v["n"]:,} 篇：text {v["text_eq"]} / length {v["len_eq"]} / sha256 {v["sha_eq"]} / status {v["status_eq"]}；'
                 f'空源行 {v["src_empty"]}）')

    okA = (not bad) and all(
        v['ok_text_equal'] == v['n_ok_rows'] and v['ok_length_equal'] == v['n_ok_rows']
        and v['ok_sha256_equal'] == v['n_ok_rows']
        and v['flagged_text_blank'] == v['n_flagged_rows'] and v['flagged_length_zero'] == v['n_flagged_rows']
        and v['flagged_sha256_equal'] == v['n_flagged_rows']
        for v in out['A_panel_roundtrip'].values())
    okB = all(v['n_mismatch'] == 0 and v['n_verified'] == v['n_verified'] and v['n_skipped'] == 0
              for v in out['B_hits_cache_vs_live'].values())
    okC = all(v['same_firms'] and v['same_csr_structure_and_values'] and v['same_filter_stats']
              for v in out['C_vectors_bitwise_equal'].values())
    out['verdict'] = {'A_panel_roundtrip': bool(okA), 'B_hits_equal': bool(okB), 'C_vectors_bitwise': bool(okC),
                      'D_rows_recompute': (None if okD is None else bool(okD)),
                      'all_pass': bool(okA and okB and okC and (okD is None or okD))}
    out['status'] = 'PASS' if out['verdict']['all_pass'] else 'FAIL'
    out['elapsed_sec'] = round(time.time() - t_all, 1)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    psay(f'状态 = {out["status"]}：A 面板往返 {okA} / B 命中缓存 {okB} / C 向量逐比特 {okC} / '
         f'D 行级 text·length·sha256·status 对账 {okD}；'
         f'报告 {REPORT.name}（{out["elapsed_sec"]}s）')
    print(json.dumps(out['verdict'], ensure_ascii=False))
    return 0 if out['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
