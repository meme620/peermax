#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3.1：申万 **as-of 分类口径** shadow —— as-of 映射 / 8 个行业基线 / K 目标 / N1 网络。

范围（`artifacts/phase3/3.0_plan.md` §2.1 + 项目已定决策）：
  - 只写 shadow：`data/processed/phase3_shadow/*`、`data/processed/network_ma_teacher_p3asof/*`、
    `artifacts/phase3/3.1_*`。**不改** headline 边表/样本/M1/PC/gate/configs；不重跑 headline M1/PC。
  - 本脚本不重建窄词表（3.3 决策）、不建 M-B shadow（留 Phase 4）、不做逐年 latest 目标变体（N5 → 3.2）。

as-of 定义（**口径限定，不得写成严格 PIT**）：
  as-of 标签 = 由「**分类生效日**」(`sw2_industry_map.csv.effective_from`) 取的当期申万分类：
    月频基线：`effective_from <= 月首日`，取 `max(effective_from)`；
    K/τ 目标：`effective_from <= (文本年+1)-04-30`（与网络 `available_at` 规则一致）。
  `sw2_industry_map.last_updated` 是**快照批量刷新字段**（与 `effective_from` 中位滞后 1,343 天、
  最早仅到 2015、22.8% 的行在 180 天内刷新）→ 记录可得日口径不可行，本脚本**不用** `last_updated`
  作 PIT 门。因此本口径是「分类生效日口径」，**不是严格 PIT**。
  已知断点：申万 2021 版自 **2021-07-30** 生效 → L2 组数 2014–2020 ≈102–104、2021–2022 = 131；
  两版码表拼接不做事后映射（不做无据的跨版本码表映射）。

步骤（`--steps`，缺省 all = map,baselines,targets,network,accept）：
  preflight   9 个 M1 输入锚 + 额外只读输入 + headline 网络全文件 sha 台账（`3.1_anchor_pre.json`）
  map         as-of 映射（`phase3_shadow/sw2_asof_map_p3shadow.parquet`）+ 边界/缺失/2021 切换审计
  baselines   8 个行业基线（LOO 同算法，仅换连接键）+ 与 headline 对比
              （`phase3_shadow/ind_baselines_p3asof.parquet`、`3.1_baseline_diff*.csv`）
  targets     as-of 与 latest 两套池化/逐年 K 目标（`3.1_k_targets.json`）
  network     N1 `p3asof` shadow 网络：复用 `s3_b3_ma_freeze.py` / `s3_ma_build_edges.py`
              （**不改磁盘脚本**；运行时把 `CORPORA['teacher']['ver_prefix']` 注入
              `pm_ma_ewcos_teacher_p3asof_v1`，并把 freeze 的默认 out_dir/report 也指向 shadow）
  accept      逐年边数/K/peer/isolated/lift（latest + as-of 两套）、vs headline 的 Jaccard/密度
  postflight  再次核验锚与 headline 网络 sha（`3.1_anchor_post.json`）

用法：
  python scripts/data_audit/p3_1_build_asof_shadow.py                       # 全链
  python scripts/data_audit/p3_1_build_asof_shadow.py --steps map,baselines
  python scripts/data_audit/p3_1_build_asof_shadow.py --steps network --years 2014 2015
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'data_audit'))

from s3_m1_lib import (  # noqa: E402
    SAMPLE_WINDOW_END, SAMPLE_WINDOW_START, assert_m1_inputs_sha, load_me, load_sw2,
    months_between, require, sha256_file,
)
from s4_step2_peermax_formal import (  # noqa: E402
    NETWORK_AVAILABLE_AT_RULE, load_versions, select_version,
)

# ---------------------------------------------------------------- 路径与常量
SHADOW_DIR = ROOT / 'data/processed/phase3_shadow'
NET_DIR = ROOT / 'data/processed/network_ma_teacher_p3asof'
ART = ROOT / 'artifacts/phase3'
HEADLINE_NET = ROOT / 'data/processed/network_ma_teacher'
SW2_MAP = ROOT / 'data/processed/sw2_industry_map.csv'
SW2_LATEST = ROOT / 'data/processed/sw2_industry_latest.csv'
TEACHER_PANEL = ROOT / 'data/processed/teacher_mda_panel.csv'
TEACHER_LEXICON = ROOT / 'data/processed/product_lexicon_v2_teacher.csv'
SAMPLE_PARQUET = ROOT / 'data/processed/m1_common_sample_ma_teacher_v2.parquet'
HEADLINE_IND_CACHE = ROOT / 'artifacts/tmp/m1_cache_industry.parquet'

ASOF_MAP = SHADOW_DIR / 'sw2_asof_map_p3shadow.parquet'
ASOF_BASELINES = SHADOW_DIR / 'ind_baselines_p3asof.parquet'
SAMPLE_ASOF_BASELINES = SHADOW_DIR / 'm1_common_sample_ma_teacher_p3asof_baselines.parquet'
K_TARGETS_JSON = ART / '3.1_k_targets.json'
BASELINE_DIFF_CSV = ART / '3.1_baseline_diff.csv'
BASELINE_DIFF_YEAR_CSV = ART / '3.1_baseline_diff_by_year.csv'
NETWORK_COMPARE_CSV = ART / '3.1_network_compare.csv'
COUNTS_JSON = ART / '3.1_counts.json'
MAP_AUDIT_JSON = ART / '3.1_map_audit.json'
OUTPUT_SHA = ART / '3.1_output_sha.json'
ANCHOR_PRE = ART / '3.1_anchor_pre.json'
ANCHOR_POST = ART / '3.1_anchor_post.json'

SHADOW_VERSION_PREFIX = 'pm_ma_ewcos_teacher_p3asof_v1'
HEADLINE_VERSION_PREFIX = 'pm_ma_ewcos_teacher_v1'
SHADOW_VERSION_TAG = 'p3shadow_v1_p3asof'
YEARS = list(range(2014, 2023))                 # 文本年 → 网络年 y+1
VINTAGE_SWITCH_DATE = '2021-07-30'              # 申万 2021 版生效日（L2 组数断点）
# K 目标（as-of L2 口径；`3.0_plan.md` §1.2 E2 预登记值，容差 0）
K_ASOF_POOLED = 49
K_ASOF_PER_YEAR = {2014: 38, 2015: 40, 2016: 44, 2017: 49, 2018: 50,
                   2019: 52, 2020: 68, 2021: 57, 2022: 64}
K_LATEST_POOLED = 44
K_LATEST_PER_YEAR = {2014: 26, 2015: 27, 2016: 30, 2017: 38, 2018: 37,
                     2019: 46, 2020: 52, 2021: 58, 2022: 62}
# headline 登记读数（只读对账；`3.0_plan.md` §1.1）
HEADLINE_K = {2014: 235, 2015: 238, 2016: 322, 2017: 291, 2018: 334,
              2019: 407, 2020: 414, 2021: 434, 2022: 463}
HEADLINE_EDGES = {2014: 157096, 2015: 156974, 2016: 152302, 2017: 214028,
                  2018: 204786, 2019: 238458, 2020: 246728, 2021: 249254,
                  2022: 293938}
HEADLINE_NODES = {2014: 2384, 2015: 2363, 2016: 2503, 2017: 2721, 2018: 2835,
                  2019: 3354, 2020: 3427, 2021: 3570, 2022: 3942}
IND_COLS = ['ind_l1_max_ew', 'ind_l1_max_vw', 'ind_l1_ret_ew', 'ind_l1_ret_vw',
            'ind_l2_max_ew', 'ind_l2_max_vw', 'ind_l2_ret_ew', 'ind_l2_ret_vw']
EDGE_COLS = ['focal_id', 'source_id', 'similarity', 'weight']

# 只读输入（sha 台账用；sw2_industry_latest 已在 9 锚内）
EXTRA_INPUTS = {
    'sw2_industry_map': SW2_MAP,
    'teacher_mda_panel': TEACHER_PANEL,
    'product_lexicon_v2_teacher': TEACHER_LEXICON,
}


# ---------------------------------------------------------------- 输出路径守卫
ALLOWED_ROOTS = (SHADOW_DIR, NET_DIR, ART)
BANNED_ROOTS = (
    HEADLINE_NET,
    ROOT / 'data/processed/network_ma',
    ROOT / 'data/processed/network_mb',
    ROOT / 'data/processed/network_ma_teacher_idf',
    ROOT / 'data/processed/network_ma_teacher_medadj',
    ROOT / 'data/processed/network_ma_teacher_df10',
    ROOT / 'data/processed/network_ma_teacher_df100',
    ROOT / 'data/processed/network_ma_fulltext',
    ROOT / 'data/processed/network_mb_teacher',
    ROOT / 'data/processed/phase2_shadow',
    ROOT / 'reports',
    ROOT / 'configs',
    ROOT / 'scripts',
    ROOT / 'tests',
    ROOT / 'docs',
    ROOT / 'deliverables',
)


def assert_output_path_guard(path) -> Path:
    """任何写出路径必须落在 shadow 允许根内，且不命中 headline/G0/configs 等禁写集合。"""
    p = Path(path).resolve()
    allowed = any(p == Path(a).resolve() or Path(a).resolve() in p.parents
                  for a in ALLOWED_ROOTS)
    banned = [Path(b).resolve() for b in BANNED_ROOTS]
    banned += [Path(SAMPLE_PARQUET).resolve(), Path(SW2_MAP).resolve()]
    hit = [b for b in banned if p == b or b in p.parents]
    if not allowed or hit:
        print(f'[HARD-GATE] 输出路径命中禁写集合: {p}（allowed={allowed} banned={hit}）',
              file=sys.stderr, flush=True)
        raise SystemExit(2)
    return p


def _atomic_write_text(path, text: str) -> None:
    p = assert_output_path_guard(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, p)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return str(o)


def write_json(path, obj) -> None:
    _atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=1, default=_json_default))


def write_csv(path, rows, header) -> None:
    p = assert_output_path_guard(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    with open(tmp, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    os.replace(tmp, p)


def write_parquet(path, df: pd.DataFrame) -> None:
    p = assert_output_path_guard(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp.parquet')
    df.to_parquet(tmp, index=False)
    os.replace(tmp, p)


def log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ---------------------------------------------------------------- 锚台账
def sha_ledger_headline_net() -> dict:
    """headline 网络目录全文件 sha（含 clusters/；只读）。"""
    out = {}
    for p in sorted(HEADLINE_NET.rglob('*')):
        if p.is_file():
            out[p.relative_to(ROOT).as_posix()] = sha256_file(p)
    return out


def anchors_snapshot(tag: str) -> dict:
    ledger = assert_m1_inputs_sha()
    extra = {k: {'file': str(v.relative_to(ROOT)).replace('\\', '/'), 'sha256': sha256_file(v)}
             for k, v in EXTRA_INPUTS.items()}
    snap = {
        'tag': tag, 'run_at': datetime.now(timezone.utc).isoformat(),
        'm1_inputs_9': {r['input']: r['sha256'] for r in ledger},
        'm1_inputs_ledger': ledger,
        'extra_inputs': extra,
        'headline_network_sha': sha_ledger_headline_net(),
    }
    snap['headline_network_n_files'] = len(snap['headline_network_sha'])
    return snap


def compare_anchors(pre: dict, post: dict) -> dict:
    diffs = []
    for sect in ('m1_inputs_9', 'extra_inputs', 'headline_network_sha'):
        a, b = pre.get(sect) or {}, post.get(sect) or {}
        if isinstance(a, dict) and a and isinstance(list(a.values())[0], dict):
            a = {k: v.get('sha256') for k, v in a.items()}
            b = {k: v.get('sha256') for k, v in b.items()}
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                diffs.append({'section': sect, 'key': k, 'pre': a.get(k), 'post': b.get(k)})
    return {'status': 'MATCH' if not diffs else 'CHANGED', 'n_diff': len(diffs), 'diffs': diffs}


# ---------------------------------------------------------------- as-of 映射
def load_sw2_map() -> pd.DataFrame:
    """`sw2_industry_map.csv` 长表 + firm_id（经 `sw2_industry_latest.csv`）。

    硬门：无 `effective_from` 缺失、`(code6, effective_from)` 键唯一（不同 vintage 的多版本行合法）。
    """
    m = pd.read_csv(SW2_MAP, dtype={'code6': str, 'l1_code': str, 'l2_code': str, 'l3_code': str})
    m['code6'] = m['code6'].str.zfill(6)
    require(m['effective_from'].notna().all(), 'as-of 映射：effective_from 存在缺失')
    m['effective_from'] = pd.to_datetime(m['effective_from'])
    dup = int(m.duplicated(['code6', 'effective_from']).sum())
    require(dup == 0, f'as-of 映射：键 (code6, effective_from) 重复 {dup} 行；停机（不许就近取值）')
    lat = pd.read_csv(SW2_LATEST, dtype=str)
    lat = lat[lat['firm_id'].notna() & lat['code6'].notna()]
    dup2 = int(lat['code6'].duplicated().sum())
    require(dup2 == 0, f'as-of 映射：sw2_industry_latest 中 code6 重复 {dup2} 行')
    firm = dict(zip(lat['code6'].str.zfill(6), lat['firm_id']))
    m['firm_id'] = m['code6'].map(firm)
    m['is_current_version'] = m['is_current_version'].astype(str).str.lower().isin(['true', '1'])
    m['source_version'] = 'sw2_classification_effective_' + m['effective_from'].dt.strftime('%Y%m%d')
    m = m.sort_values(['code6', 'effective_from'], kind='stable').reset_index(drop=True)
    return m


def asof_labels_for_months(m: pd.DataFrame, firm_months: pd.DataFrame) -> pd.DataFrame:
    """`firm_months`（firm_id, month='YYYY-MM', code6）→ as-of 标签（月首日口径，backward）。"""
    left = firm_months.copy()
    left['_key'] = pd.to_datetime(left['month'] + '-01')
    left = left.sort_values('_key', kind='stable')
    right = m[['code6', 'effective_from', 'l1_code', 'l2_code', 'l3_code',
               'source_version', 'is_current_version']].sort_values('effective_from', kind='stable')
    out = pd.merge_asof(left, right, left_on='_key', right_on='effective_from', by='code6',
                        direction='backward', allow_exact_matches=True)
    return out


def asof_labels_for_key(m: pd.DataFrame, key: str) -> pd.DataFrame:
    """`effective_from <= key`（单一日期，供 K 目标与网络 lift 用）→ code6 的 as-of 标签。"""
    sub = m[m['effective_from'] <= pd.Timestamp(key)]
    return (sub.sort_values('effective_from', kind='stable')
            .drop_duplicates('code6', keep='last')[['code6', 'l1_code', 'l2_code',
                                                    'effective_from', 'source_version']])


def headline_latest_l2_map() -> dict:
    """headline 静态标签（`sw2_industry_latest.csv`，= `load_sw2()` 口径）→ firm_id: l2_code。"""
    lat = pd.read_csv(SW2_LATEST, dtype={'code6': str, 'firm_id': str, 'l2_code': str})
    lat = lat[lat['firm_id'].notna() & lat['code6'].notna()]
    return dict(zip(lat['firm_id'], lat['l2_code']))


def l2_map_for_key(m: pd.DataFrame, key: str) -> dict:
    """as-of 标签（`effective_from <= key`）→ firm_id: l2_code（无映射公司不入字典）。"""
    a = asof_labels_for_key(m, key)
    code2firm = (m[m['firm_id'].notna()].sort_values('effective_from', kind='stable')
                 .drop_duplicates('code6', keep='last')
                 .set_index('code6')['firm_id'].to_dict())
    return {code2firm[c]: v for c, v in zip(a['code6'], a['l2_code']) if c in code2firm}


# ---------------------------------------------------------------- 行业基线（LOO）
def build_industry_baselines_labelled(months, labels: pd.DataFrame, srcm: pd.DataFrame,
                                      me: pd.DataFrame, lvl_cols=('l1_code', 'l2_code'),
                                      suffix: str = '', join_on=('firm_id', 'month')) -> pd.DataFrame:
    """与 `s3_m1_1_build_sample.build_industry_baselines` **同一 LOO 算法**，只换连接键。

    headline 版：`srcm.merge(sw2, on='firm_id')`（静态 latest 标签，join_on=('firm_id',)）；
    本版：`srcm.merge(labels, on=['firm_id','month'])`（as-of 标签）。
    """
    v = srcm[['source_id', 'month', 'MAXraw', 'R']].copy()
    v = v[v['month'].isin(months)]
    v = v.rename(columns={'source_id': 'firm_id'})
    v = v.merge(labels[list(join_on) + list(lvl_cols)], on=list(join_on), how='inner')
    v['code6'] = v['firm_id'].str[2:].str.zfill(6)
    v = v.merge(me.rename(columns={'firm_id': 'code6'}), on=['code6', 'month'], how='left')
    v['v_ew'] = 1.0
    v['v_vw'] = v['market_equity']
    out = []
    for lvl in lvl_cols:
        for kind in ['MAXraw', 'R']:
            for wname, vcol in [('ew', 'v_ew'), ('vw', 'v_vw')]:
                sub = v[v[vcol].notna() & v[kind].notna()].copy()
                sub['_num'] = sub[vcol] * sub[kind]
                g = sub.groupby(['month', lvl], sort=False)
                tot = (g.agg(_num=('_num', 'sum'), _den=(vcol, 'sum')).reset_index()
                       .rename(columns={'_num': '_num_sum', '_den': '_den_sum'}))
                sub = sub.merge(tot, on=['month', lvl], how='left')
                own_num = np.where(sub['_num'].notna(), sub['_num'], 0.0)
                sub['_num_loo'] = sub['_num_sum'] - own_num
                sub['_den_loo'] = sub['_den_sum'] - sub[vcol]
                sub['_val'] = np.where(sub['_den_loo'] > 0,
                                       sub['_num_loo'] / sub['_den_loo'], np.nan)
                name = f'ind_{lvl[:2]}_{"max" if kind == "MAXraw" else "ret"}_{wname}{suffix}'
                out.append(sub[['firm_id', 'month', '_val']].rename(columns={'_val': name}))
    res = out[0]
    for o in out[1:]:
        res = res.merge(o, on=['firm_id', 'month'], how='outer')
    return res


def loo_handcheck(labels: pd.DataFrame, srcm: pd.DataFrame, me: pd.DataFrame, months) -> dict:
    """抽 200 个 (month, l2) 组手算 LOO 均值，与向量化实现逐值比对（R1 ≤1e-12）。"""
    v = srcm[['source_id', 'month', 'MAXraw']].rename(columns={'source_id': 'firm_id'})
    v = v[v['month'].isin(months)].merge(labels[['firm_id', 'month', 'l2_code']],
                                         on=['firm_id', 'month'], how='inner')
    v = v[v['MAXraw'].notna()]
    vec = build_industry_baselines_labelled(months, labels, srcm, me)
    vec = vec[['firm_id', 'month', 'ind_l2_max_ew']]
    keys = (v.groupby(['month', 'l2_code']).size().sort_values(ascending=False).head(200)
            .reset_index()[['month', 'l2_code']])
    worst = 0.0
    n_checked = 0
    for _, row in keys.iterrows():
        g = v[(v['month'] == row['month']) & (v['l2_code'] == row['l2_code'])]
        for f in g['firm_id'].head(3):
            rest = g[g['firm_id'] != f]['MAXraw']
            if len(rest) == 0:
                continue
            manual = float(rest.mean())
            got = vec[(vec['firm_id'] == f) & (vec['month'] == row['month'])]['ind_l2_max_ew']
            if got.empty or not np.isfinite(got.iloc[0]):
                continue
            worst = max(worst, abs(float(got.iloc[0]) - manual))
            n_checked += 1
    return {'n_checked': n_checked, 'max_abs_diff': worst,
            'pass': bool(worst <= 1e-12 and n_checked > 0)}


# ---------------------------------------------------------------- K 目标
def sw2_k_target(m: pd.DataFrame, panel: pd.DataFrame, years, mode: str) -> dict:
    """`s3_b1_calibrate_tau.sw2_target` 同构（文本年 ok 行 + 组内 N−1 中位 peer）。

    mode='asof'：`effective_from <= (y+1)-04-30` 的最近一条分类；
    mode='latest'：`sw2_industry_latest.csv` 的静态最新分类（对照列）。
    """
    p = panel.copy()
    p['code6'] = p['firm_id'].astype(str).str[2:].str.zfill(6)
    p['_key'] = pd.to_datetime((p['year'] + 1).astype(str) + '-04-30')
    p = p.sort_values('_key', kind='stable')
    if mode == 'asof':
        right = (m[['code6', 'effective_from', 'l2_code']]
                 .sort_values('effective_from', kind='stable'))
        j = pd.merge_asof(p, right, left_on='_key', right_on='effective_from', by='code6',
                          direction='backward')
    elif mode == 'latest':
        lat = pd.read_csv(SW2_LATEST, dtype={'code6': str, 'l2_code': str})
        lat['code6'] = lat['code6'].str.zfill(6)
        j = p.merge(lat[['code6', 'l2_code']], on='code6', how='left')
    else:
        raise ValueError(mode)
    n_unmatched = int(j['l2_code'].isna().sum())
    j = j.dropna(subset=['l2_code'])
    per_year, pooled = [], []
    for y, g in j.groupby('year'):
        sizes = g.groupby('l2_code')['code6'].size()
        peers = (g['l2_code'].map(sizes) - 1).astype(float)
        per_year.append({'text_year': int(y), 'network_year': int(y) + 1,
                         'firms': int(len(g)), 'sw2_groups': int(sizes.size),
                         'median_peer': float(peers.median()),
                         'mean_peer': round(float(peers.mean()), 4),
                         'group_size_median': float(sizes.median()),
                         'isolated_share': round(float((peers == 0).mean()), 4)})
        pooled.append(peers.to_numpy())
    pooled_peers = np.concatenate(pooled)
    big = j.groupby('l2_code')['code6'].size()
    return {
        'mode': mode, 'asof_rule': ASOF_RULE[0] if mode == 'asof' else 'static_latest_row',
        'per_year': per_year,
        'median_peer_pooled_firm_level': float(np.median(pooled_peers)),
        'median_peer_pooled_rounded': int(round(float(np.median(pooled_peers)))),
        'mean_peer_pooled': round(float(pooled_peers.mean()), 4),
        'pooled_group_size_median': float(big.median()),
        'n_firm_years': int(len(pooled_peers)), 'n_firm_years_unmatched': n_unmatched,
    }


ASOF_RULE = ('effective_from <= key，取 max(effective_from)（分类生效日口径，非严格 PIT）',
             'effective_from <= 月首日，取最近一条（月频基线）')


# ---------------------------------------------------------------- 网络构建（复用现脚本）
def _variant_modules():
    cg = importlib.import_module('s3_b3_ma_cluster_grid')
    frz = importlib.import_module('s3_b3_ma_freeze')
    be = importlib.import_module('s3_ma_build_edges')
    return cg, frz, be


def patch_corpora(target: int = K_ASOF_POOLED) -> dict:
    """运行时注入 shadow 版本前缀 / K 规则标签 / shadow 默认输出路径（**不改磁盘脚本**）。"""
    cg, frz, be = _variant_modules()
    require(be.CORPORA['teacher']['ver_prefix'] == HEADLINE_VERSION_PREFIX,
            f'edges CORPORA[teacher].ver_prefix 非 {HEADLINE_VERSION_PREFIX}；拒绝继续')
    require(cg.CORPORA['teacher']['ver_prefix'] == HEADLINE_VERSION_PREFIX,
            f'cluster_grid CORPORA[teacher].ver_prefix 非 {HEADLINE_VERSION_PREFIX}；拒绝继续')
    require(SHADOW_VERSION_PREFIX != HEADLINE_VERSION_PREFIX,
            'shadow 版本前缀与 headline 同名；停机（禁止 N-9/I-9 型同名不同义）')
    cg.CORPORA['teacher']['ver_prefix'] = SHADOW_VERSION_PREFIX
    be.CORPORA['teacher']['ver_prefix'] = SHADOW_VERSION_PREFIX
    # K 规则标签随目标改写（否则 shadow build_report 仍写 headline 的 44 → 同名不同义）
    require(str(be.CORPORA['teacher']['k_rule']).endswith('(44)'),
            f'edges CORPORA[teacher].k_rule 意外值 {be.CORPORA["teacher"]["k_rule"]!r}')
    be.CORPORA['teacher']['k_rule'] = f'per_year_target_peer_match({target})'
    # 防御纵深：即便有人漏传 CLI 参数，缺省输出也落到 shadow 而非 headline
    frz.SPECS['teacher']['out_dir'] = NET_DIR / 'clusters'
    frz.SPECS['teacher']['report'] = ART / '3.1_freeze_p3asof_default.json'
    be.CORPORA['teacher']['net_dir'] = NET_DIR
    return {'shadow_ver_prefix': SHADOW_VERSION_PREFIX,
            'headline_ver_prefix': HEADLINE_VERSION_PREFIX,
            'k_rule_label': be.CORPORA['teacher']['k_rule'],
            'freeze_out_dir_default': str(frz.SPECS['teacher']['out_dir']),
            'edges_net_dir_default': str(be.CORPORA['teacher']['net_dir'])}


def _call_main(mod, argv) -> None:
    old = sys.argv
    try:
        sys.argv = [Path(mod.__file__).name] + list(argv)
        mod.main()
    finally:
        sys.argv = old


def build_network_year(y: int, target: int) -> dict:
    """单文本年：freeze（K 校准 + 聚类标签）→ edges（边表/support/build_report）。"""
    cg, frz, be = _variant_modules()
    log(f'[{y}] freeze：corpus=teacher target={target} → {NET_DIR.name}/clusters')
    _call_main(frz, ['--corpus', 'teacher', '--years', str(y), '--target', str(target),
                     '--lexicon', str(TEACHER_LEXICON),
                     '--out-dir', str(NET_DIR / 'clusters'),
                     '--out-report', str(ART / f'3.1_freeze_p3asof_{y}.json')])
    log(f'[{y}] edges：corpus=teacher ver_prefix={SHADOW_VERSION_PREFIX} → {NET_DIR.name}')
    _call_main(be, ['--corpus', 'teacher', '--years', str(y), '--net-dir', str(NET_DIR),
                    '--cluster-dir', str(NET_DIR / 'clusters'),
                    '--lexicon', str(TEACHER_LEXICON)])
    rep = json.loads((NET_DIR / f'{y}_build_report.json').read_text(encoding='utf-8'))
    expected_ver = f'{SHADOW_VERSION_PREFIX}_{y}'
    require(rep['network_version'] == expected_ver,
            f'[{y}] shadow 边表 network_version={rep["network_version"]!r} != {expected_ver!r}')
    assert_path = NET_DIR / f'{y}_similarity.csv'
    assert_output_path_guard(assert_path)
    return {'text_year': y, 'network_year': y + 1, 'n_edges': int(rep['n_edges']),
            'n_nodes': int(rep['n_nodes']), 'K': int(rep['k_clusters']),
            'network_version': rep['network_version'], 'available_at': rep['available_at'],
            'n_support': int(rep['n_support']), 'n_no_support': int(rep['n_no_support'])}


# ---------------------------------------------------------------- 网络读数 / 对比
def read_edges(net_dir: Path, y: int) -> pd.DataFrame:
    return pd.read_csv(net_dir / f'{y}_similarity.csv', usecols=EDGE_COLS,
                       dtype={'focal_id': str, 'source_id': str})


def read_clusters(net_dir: Path, y: int) -> pd.DataFrame:
    return pd.read_csv(net_dir / 'clusters' / f'{y}_clusters.csv', dtype={'firm_id': str})


def cluster_lift(clusters: pd.DataFrame, labels: dict) -> dict:
    """同一 `cluster_stats` 口径：团内同申万二级无序对数占比 / 全样本边际基线。"""
    lab = clusters['cluster_id'].to_numpy()
    l2 = pd.Series([labels.get(f, 'NA') for f in clusters['firm_id']], index=clusters.index)
    _, l2_idx = np.unique(l2.astype(str).to_numpy(), return_inverse=True)
    n_l2 = int(l2_idx.max()) + 1
    n = len(lab)
    ks, inv = np.unique(lab, return_inverse=True)
    sizes = np.bincount(inv)
    comb = np.bincount(inv * n_l2 + l2_idx, minlength=len(ks) * n_l2).reshape(len(ks), n_l2)
    same = int((comb * (comb - 1) // 2).sum())
    iu = np.triu_indices(n, k=1)
    base = float((l2_idx[iu[0]] == l2_idx[iu[1]]).mean())
    tot = int((sizes * (sizes - 1) // 2).sum())
    share = same / max(1, tot)
    return {'same_sw2_edge_share': round(share, 4),
            'base_same_sw2_rate': round(base, 5),
            'sw2_lift': round(share / max(1e-9, base), 2),
            'peer_median': float(np.median(sizes[inv] - 1)),
            'isolated_share': round(float(((sizes[inv] - 1) == 0).mean()), 4),
            'n_clusters': int(len(ks)), 'size_max': int(sizes.max())}


def jaccard(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    sa = set(zip(a['focal_id'], a['source_id']))
    sb = set(zip(b['focal_id'], b['source_id']))
    inter, union = len(sa & sb), len(sa | sb)
    return {'n_shadow': len(sa), 'n_headline': len(sb), 'n_inter': inter, 'n_union': union,
            'jaccard': round(inter / union, 6) if union else None}


def density(n_edges: int, n_nodes: int) -> float:
    return round(n_edges / max(1, n_nodes * (n_nodes - 1)), 8)


def acceptance(m: pd.DataFrame, years) -> tuple:
    latest_map = headline_latest_l2_map()
    # headline 网络登记读数复核（只读）
    headline_ref = {}
    for y in years:
        rep = json.loads((HEADLINE_NET / f'{y}_build_report.json').read_text(encoding='utf-8'))
        require(int(rep['n_edges']) == HEADLINE_EDGES[y],
                f'[{y}] headline 边数 {rep["n_edges"]} != 登记 {HEADLINE_EDGES[y]}')
        require(int(rep['k_clusters']) == HEADLINE_K[y],
                f'[{y}] headline K {rep["k_clusters"]} != 登记 {HEADLINE_K[y]}')
        require(int(rep['n_nodes']) == HEADLINE_NODES[y],
                f'[{y}] headline 节点数 {rep["n_nodes"]} != 登记 {HEADLINE_NODES[y]}')
        headline_ref[y] = rep
    rows, detail = [], {}
    for y in years:
        eref = read_edges(HEADLINE_NET, y)
        esha = read_edges(NET_DIR, y)
        crefs = read_clusters(HEADLINE_NET, y)
        csha = read_clusters(NET_DIR, y)
        # as-of 标签（K 目标口径：effective_from <= (y+1)-04-30）
        asof_map = l2_map_for_key(m, f'{y + 1}-04-30')
        for net_name, cdf in (('headline', crefs), ('p3asof', csha)):
            detail.setdefault(y, {})[net_name] = {}
            detail[y][net_name]['lift_latest'] = cluster_lift(cdf, latest_map)
            detail[y][net_name]['lift_asof'] = cluster_lift(cdf, asof_map)
        hrep = headline_ref[y]
        srep = json.loads((NET_DIR / f'{y}_build_report.json').read_text(encoding='utf-8'))
        jc = jaccard(esha, eref)
        row = {
            'text_year': y, 'network_year': y + 1,
            'headline_edges': int(hrep['n_edges']), 'p3asof_edges': int(srep['n_edges']),
            'edge_delta_pct': round(100.0 * (int(srep['n_edges']) / int(hrep['n_edges']) - 1), 3),
            'headline_nodes': int(hrep['n_nodes']), 'p3asof_nodes': int(srep['n_nodes']),
            'headline_K': int(hrep['k_clusters']), 'p3asof_K': int(srep['k_clusters']),
            'p3asof_network_version': srep['network_version'],
            'p3asof_available_at': srep['available_at'],
            'headline_density': density(int(hrep['n_edges']), int(hrep['n_nodes'])),
            'p3asof_density': density(int(srep['n_edges']), int(srep['n_nodes'])),
            'jaccard_vs_headline': jc['jaccard'], 'n_overlap_edges': jc['n_inter'],
            'headline_lift_asof': detail[y]['headline']['lift_asof']['sw2_lift'],
            'p3asof_lift_asof': detail[y]['p3asof']['lift_asof']['sw2_lift'],
            'headline_lift_latest': detail[y]['headline']['lift_latest']['sw2_lift'],
            'p3asof_lift_latest': detail[y]['p3asof']['lift_latest']['sw2_lift'],
            'headline_isolated_share': detail[y]['headline']['lift_asof']['isolated_share'],
            'p3asof_isolated_share': detail[y]['p3asof']['lift_asof']['isolated_share'],
            'headline_peer_median': detail[y]['headline']['lift_latest']['peer_median'],
            'p3asof_peer_median': detail[y]['p3asof']['lift_latest']['peer_median'],
            'headline_size_max': detail[y]['headline']['lift_latest']['size_max'],
            'p3asof_size_max': detail[y]['p3asof']['lift_latest']['size_max'],
            'p3asof_n_support': int(srep['n_support']),
            'p3asof_n_no_support': int(srep['n_no_support']),
            # 3.2 依赖读数：密度楔子（**两个定义不同，不得混用**）
            # - wedge_p3asof_vs_asof_ind  = 本 shadow 网络自身 realized 中位 peer / 同年 as-of 行业中位 peer
            # - wedge_headline_vs_asof_ind= headline 网络 realized 中位 peer / 同年 as-of 行业中位 peer
            #   （后者才是 plan §2.2 的「as-of 标签下楔子 ≈1.16→0.69」预投影定义）
            # - wedge_headline_vs_latest_ind = plan §2.2 的 latest 标签楔子（1.69→0.71）
            'asof_ind_median_peer_y': K_ASOF_PER_YEAR[y],
            'headline_ind_median_peer_y_latest': K_LATEST_PER_YEAR[y],
            'wedge_p3asof_vs_asof_ind': round(
                detail[y]['p3asof']['lift_latest']['peer_median'] / K_ASOF_PER_YEAR[y], 4),
            'wedge_headline_vs_asof_ind': round(
                detail[y]['headline']['lift_latest']['peer_median'] / K_ASOF_PER_YEAR[y], 4),
            'wedge_headline_vs_latest_ind': round(
                detail[y]['headline']['lift_latest']['peer_median'] / K_LATEST_PER_YEAR[y], 4),
        }
        rows.append(row)
    return rows, detail


def freeze_report_value(y: int, field: str):
    p = ART / f'3.1_freeze_p3asof_{y}.json'
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding='utf-8'))
    return d.get('years', {}).get(str(y), {}).get(field)


# ---------------------------------------------------------------- steps
def step_map(months, panel_srcm) -> dict:
    t0 = time.time()
    m = load_sw2_map()
    log(f'map：{len(m)} 行 / {m["code6"].nunique()} code / {int(m["firm_id"].notna().sum())} 行有 firm_id')
    fm = (panel_srcm[['source_id', 'month']].drop_duplicates()
          .rename(columns={'source_id': 'firm_id'}))
    fm = fm[fm['month'].isin(months)].copy()
    fm['code6'] = fm['firm_id'].str[2:].str.zfill(6)
    n_fm = len(fm)
    lab = asof_labels_for_months(m, fm)
    covered = int(lab['l2_code'].notna().sum())
    n_no_eff = int(lab['effective_from'].isna().sum())
    n_no_firm = int(m.groupby('code6')['firm_id'].first().isna().sum())
    mid_month = int((lab['effective_from'].dt.day != 1).sum())
    exact_first = int((lab['effective_from'].dt.day == 1).sum())
    predates_window = int((lab['effective_from'] < pd.Timestamp(months[0] + '-01')).sum())
    switch = {}
    for mo in ('2021-06', '2021-07', '2021-08', '2021-09'):
        sub = lab[lab['month'] == mo]
        switch[mo] = {
            'firm_months': int(len(sub)),
            'eff_eq_switch_date': int((sub['effective_from'] == pd.Timestamp(VINTAGE_SWITCH_DATE)).sum()),
            'n_l2_groups': int(sub['l2_code'].nunique()),
            'median_eff': str(sub['effective_from'].median())[:10] if sub['effective_from'].notna().any() else None,
        }
    # 标签不一致率（M1 样本 307,503 行；月首日口径 + (y+1)-04-30 口径）
    smp = pd.read_parquet(SAMPLE_PARQUET, columns=['focal_id', 'month'])
    smp['code6'] = smp['focal_id'].str[2:].str.zfill(6)
    smp_key = smp.rename(columns={'focal_id': 'firm_id'}).sort_values('month', kind='stable')
    labs_m = asof_labels_for_months(m, smp_key)
    lat = pd.read_csv(SW2_LATEST, dtype={'code6': str, 'l1_code': str, 'l2_code': str})
    lat['code6'] = lat['code6'].str.zfill(6)
    lat_map = lat.drop_duplicates('code6').set_index('code6')
    labs_m['latest_l1'] = labs_m['code6'].map(lat_map['l1_code'])
    labs_m['latest_l2'] = labs_m['code6'].map(lat_map['l2_code'])
    ok = labs_m['l2_code'].notna() & labs_m['latest_l2'].notna()
    mis_l2 = float((labs_m.loc[ok, 'l2_code'] != labs_m.loc[ok, 'latest_l2']).mean())
    mis_l1 = float((labs_m.loc[ok, 'l1_code'] != labs_m.loc[ok, 'latest_l1']).mean())
    by_year = {}
    for yy, g in labs_m[ok].groupby(labs_m.loc[ok, 'month'].str[:4]):
        by_year[yy] = {'n': int(len(g)),
                       'l2_mismatch': round(float((g['l2_code'] != g['latest_l2']).mean()), 4),
                       'l1_mismatch': round(float((g['l1_code'] != g['latest_l1']).mean()), 4)}
    keep = ['code6', 'firm_id', 'month', 'l1_code', 'l2_code', 'l3_code',
            'effective_from', 'source_version', 'is_current_version']
    out = lab[keep].copy()
    write_parquet(ASOF_MAP, out)
    audit = {
        'run_at': datetime.now(timezone.utc).isoformat(),
        'script': 'scripts/data_audit/p3_1_build_asof_shadow.py',
        'scope': 'p3shadow_v1_p3asof',
        'definition': ('as-of = 分类生效日口径（effective_from <= 月首日，取 max(effective_from)）；'
                       '**非严格 PIT**：不使用 last_updated（快照批量刷新字段）作记录可得日门'),
        'map_source_rows': int(len(m)), 'map_source_codes': int(m['code6'].nunique()),
        'codes_with_firm_id': int(m.groupby('code6')['firm_id'].first().notna().sum()),
        'codes_without_firm_id': n_no_firm,
        'key_unique_code6_eff': True,
        'firm_months': n_fm, 'firm_months_covered': covered,
        'coverage': round(covered / max(1, n_fm), 6),
        'firm_months_without_effective_from': n_no_eff,
        'rows_with_effective_from_after_month_first_day': mid_month,
        'rows_with_effective_from_equal_month_first_day': exact_first,
        'firm_months_using_classification_predating_window': predates_window,
        'boundary_note': (f'effective_from 落在月中（day != 1）的 firm-month = {mid_month}；月首日口径下'
                          f'这些公司在当月按既往 vintage 归类，次月首日起才用新 vintage。'
                          f'（恰落在月首日 = {exact_first}；分类生效日早于样本首月 = {predates_window}）'),
        'vintage_switch': {'switch_date': VINTAGE_SWITCH_DATE, 'per_month': switch},
        'mismatch_vs_latest_on_m1_sample': {
            'n_sample_rows': int(len(smp)),
            'n_comparable': int(ok.sum()),
            'l2_mismatch_month_first_day': round(mis_l2, 6),
            'l1_mismatch_month_first_day': round(mis_l1, 6),
            'per_calendar_year': by_year,
        },
        'asof_map_rows_written': int(len(out)),
        'output': str(ASOF_MAP.relative_to(ROOT)).replace('\\', '/'),
        'elapsed_sec': round(time.time() - t0, 1),
    }
    write_json(MAP_AUDIT_JSON, audit)
    log(f'map：覆盖率 {audit["coverage"]:.6f}（未匹配 {n_no_eff} 行）；L2 不一致率 {mis_l2:.4%}；'
        f'L1 {mis_l1:.4%}；2021-07-30 切换行 {switch["2021-08"]["eff_eq_switch_date"]}')
    return audit


def step_baselines(months) -> dict:
    t0 = time.time()
    srcm = load_src_month_safe()
    me = load_me()
    asof = pd.read_parquet(ASOF_MAP)
    lab2 = asof[['firm_id', 'month', 'l1_code', 'l2_code']].rename(
        columns={'l1_code': 'l1_asof', 'l2_code': 'l2_asof'})
    base = build_industry_baselines_labelled(months, lab2.rename(
        columns={'l1_asof': 'l1_code', 'l2_asof': 'l2_code'}), srcm, me, suffix='_p3asof')
    base = base.merge(lab2, on=['firm_id', 'month'], how='left')
    write_parquet(ASOF_BASELINES, base)
    log(f'baselines：as-of 基线 {len(base):,} 行（8 列 _p3asof）→ {ASOF_BASELINES.name}')
    # 复算保真门：同一函数 + latest 标签（同 headline 连接键）应逐值复现 headline 基线缓存
    sw2_latest = load_sw2()
    chk = build_industry_baselines_labelled(months, sw2_latest, srcm, me, suffix='',
                                            join_on=('firm_id',))
    ref = pd.read_parquet(HEADLINE_IND_CACHE)
    req = ref.merge(chk, on=['firm_id', 'month'], how='left', suffixes=('_ref', '_chk'))
    require(len(req) == len(ref), f'baselines：比较行长 {len(req)} != headline 缓存 {len(ref)}')
    worst, per_col = 0.0, {}
    for c in IND_COLS:
        a, b = req[c + '_ref'], req[c + '_chk']
        d = float(np.nanmax(np.abs(a - b))) if a.notna().any() else None
        per_col[c] = d
        worst = max(worst, d or 0.0)
    fidelity = {'rows_ref': int(len(ref)), 'rows_chk': int(len(chk)),
                'matched_rows': int(len(req)), 'max_abs_diff_vs_headline_cache': worst,
                'per_column_max_abs_diff': per_col,
                'pass': bool(worst <= 1e-10 and len(chk) == len(ref))}
    require(fidelity['pass'],
            f'baselines：与 headline 基线缓存的最大差 {worst} > 1e-10（或行数不符）；复算不忠实，停机')
    log(f'baselines：LOO 复算保真门 PASS（max|Δ| = {worst:.2e} vs headline 缓存 {len(ref):,} 行）')
    # LOO 手算抽查
    hand = loo_handcheck(lab2.rename(columns={'l1_asof': 'l1_code', 'l2_asof': 'l2_code'}),
                         srcm, me, months)
    require(hand['pass'], f'baselines：LOO 手算抽查失败 {hand}；停机')
    log(f'baselines：LOO 手算抽查 {hand["n_checked"]} 点 max|Δ| = {hand["max_abs_diff"]:.2e}')
    # 样本键（与 headline 样本列对比用；含 FM 标记）
    smp = pd.read_parquet(SAMPLE_PARQUET,
                          columns=['focal_id', 'month', 'm1_in_fm_sample'] + IND_COLS)
    # 3.5/A12 依赖件：headline 样本副本 + 8 个 `_p3asof` 列（headline 列一字不动）
    full = pd.read_parquet(SAMPLE_PARQUET)
    full = full.merge(base.rename(columns={c: c + '_p3asof' for c in IND_COLS})
                      [['firm_id', 'month'] + [c + '_p3asof' for c in IND_COLS]].rename(
                          columns={'firm_id': 'focal_id'}),
                      on=['focal_id', 'month'], how='left')
    require(len(full) == len(smp), f'as-of 样本副本行数 {len(full)} != 样本 {len(smp)}')
    write_parquet(SAMPLE_ASOF_BASELINES, full)
    log(f'baselines：样本副本 + 8 个 _p3asof 列 → {SAMPLE_ASOF_BASELINES.name}（{len(full):,} 行）')
    # 与 headline 样本列对比（两套 scope：ALL 307,503 / FM 181,178；plan §1.2 E4 以 FM 为准）
    smp = smp.rename(columns={'focal_id': 'firm_id'})
    j = smp.merge(base.rename(columns={c: c + '_p3asof' for c in IND_COLS})
                  [['firm_id', 'month'] + [c + '_p3asof' for c in IND_COLS]],
                  on=['firm_id', 'month'], how='left')
    j['_scope_all'] = True
    rows, yrows = [], []
    fm_vals = {}
    for scope, mask in (('ALL_307503', pd.Series(True, index=j.index)),
                        ('FM_181178', j['m1_in_fm_sample'].astype(bool))):
        g0 = j[mask]
        for c in IND_COLS:
            a, b = g0[c], g0[c + '_p3asof']
            both = a.notna() & b.notna()
            corr = float(np.corrcoef(a[both], b[both])[0, 1]) if both.sum() > 1 else None
            mad = float(np.median(np.abs(a[both] - b[both]))) if both.any() else None
            sign = float((np.sign(a[both]) == np.sign(b[both])).mean()) if both.any() else None
            miss_h = float(1 - a.notna().mean())
            miss_a = float(1 - b.notna().mean())
            rows.append({
                'scope': scope, 'column': c, 'n_rows': int(len(g0)),
                'n_both_nonnull': int(both.sum()),
                'corr_asof_vs_headline': round(corr, 6) if corr is not None else None,
                'median_abs_diff': mad,
                'sign_agree_share': round(sign, 6) if sign is not None else None,
                'missing_headline': round(miss_h, 6), 'missing_asof': round(miss_a, 6),
                'coverage_headline': round(float(a.notna().mean()), 6),
                'coverage_asof': round(float(b.notna().mean()), 6),
                'mean_diff_asof_minus_headline': float((b[both] - a[both]).mean()) if both.any() else None,
            })
            if scope == 'FM_181178':
                fm_vals[c] = {'corr': corr, 'median_abs_diff': mad, 'missing_headline': miss_h,
                              'missing_asof': miss_a}
            # 逐年读数（与 scope 同口径；含 sign / missing，8 列全部逐年输出）
            gy = g0.assign(_y=g0['month'].str[:4])
            for yy, gg in gy.groupby('_y'):
                aa, bb = gg[c], gg[c + '_p3asof']
                bb2 = aa.notna() & bb.notna()
                yrows.append({
                    'scope': scope, 'year': yy, 'column': c, 'n_rows': int(len(gg)),
                    'n_both_nonnull': int(bb2.sum()),
                    'corr': round(float(np.corrcoef(aa[bb2], bb[bb2])[0, 1]), 6) if bb2.sum() > 1 else None,
                    'median_abs_diff': float(np.median(np.abs(aa[bb2] - bb[bb2]))) if bb2.any() else None,
                    'sign_agree_share': round(float((np.sign(aa[bb2]) == np.sign(bb[bb2])).mean()), 6)
                    if bb2.any() else None,
                    'missing_headline': round(float(1 - aa.notna().mean()), 6),
                    'missing_asof': round(float(1 - bb.notna().mean()), 6),
                })
    # 预登记对账（plan §1.2 E4，FM 口径）：corr L1 0.9564 / L2 0.9005、中位差 0.00111 / 0.00231、
    # headline 缺 1.07%/1.10% → as-of 0.00%/0.04%（MAX_EW 列；容差 = corr ±0.002 / mad ±0.0002 / 覆盖 ±0.0005）
    reg = {'ind_l1_max_ew': dict(corr=0.9564, mad=0.00111, miss_h=0.0107, miss_a=0.0000),
           'ind_l2_max_ew': dict(corr=0.9005, mad=0.00231, miss_h=0.0110, miss_a=0.0004)}
    checks = {}
    for c, exp in reg.items():
        got = fm_vals[c]
        ok = (abs(got['corr'] - exp['corr']) <= 0.002
              and abs(got['median_abs_diff'] - exp['mad']) <= 0.0002
              and abs(got['missing_headline'] - exp['miss_h']) <= 0.0005
              and abs(got['missing_asof'] - exp['miss_a']) <= 0.0005)
        checks[c] = {'registered': exp, 'measured': {k: (round(v, 6) if v is not None else None)
                                                    for k, v in got.items()}, 'pass': bool(ok)}
        require(ok, f'baselines：{c} 与 E4 预登记值不符 {checks[c]}；停机')
    log(f'baselines：E4 预登记对账 PASS（FM：L1 corr {fm_vals["ind_l1_max_ew"]["corr"]:.4f}、'
        f'L2 corr {fm_vals["ind_l2_max_ew"]["corr"]:.4f}）')
    write_csv(BASELINE_DIFF_CSV, [list(r.values()) for r in rows], list(rows[0].keys()))
    write_csv(BASELINE_DIFF_YEAR_CSV, [list(r.values()) for r in yrows], list(yrows[0].keys()))
    log(f'baselines：对比表 → {BASELINE_DIFF_CSV.name}（样本行 {len(j):,}，ALL + FM 两 scope）')
    return {'fidelity_vs_headline_cache': fidelity, 'loo_handcheck': hand,
            'registered_e4_check': checks,
            'rows_baseline_written': int(len(base)),
            'n_rows_with_asof_label': int(base['l2_asof'].notna().sum()),
            'sample_copy': {'path': str(SAMPLE_ASOF_BASELINES.relative_to(ROOT)).replace('\\', '/'),
                            'rows': int(len(full)), 'added_columns': [c + '_p3asof' for c in IND_COLS]},
            'sample_compare': [r for r in rows if r['scope'] == 'FM_181178'],
            'sample_compare_all_scope': [r for r in rows if r['scope'] == 'ALL_307503'],
            'compare_csv': str(BASELINE_DIFF_CSV.relative_to(ROOT)).replace('\\', '/'),
            'compare_by_year_csv': str(BASELINE_DIFF_YEAR_CSV.relative_to(ROOT)).replace('\\', '/'),
            'elapsed_sec': round(time.time() - t0, 1)}


def load_src_month_safe() -> pd.DataFrame:
    from s3_m1_lib import load_src_month
    return load_src_month()


def step_targets(panel: pd.DataFrame) -> dict:
    t0 = time.time()
    m = load_sw2_map()
    asof = sw2_k_target(m, panel, YEARS, 'asof')
    latest = sw2_k_target(m, panel, YEARS, 'latest')
    # 逐值对账（容差 0，plan §1.2 E2 预登记）
    require(asof['median_peer_pooled_rounded'] == K_ASOF_POOLED,
            f'as-of 池化 K 目标 {asof["median_peer_pooled_rounded"]} != {K_ASOF_POOLED}；停机')
    got = {p['text_year']: int(round(p['median_peer'])) for p in asof['per_year']}
    require(got == K_ASOF_PER_YEAR, f'as-of 逐年 K 目标 {got} != {K_ASOF_PER_YEAR}；停机')
    require(latest['median_peer_pooled_rounded'] == K_LATEST_POOLED,
            f'latest 池化 K 目标 {latest["median_peer_pooled_rounded"]} != {K_LATEST_POOLED}')
    got_l = {p['text_year']: int(round(p['median_peer'])) for p in latest['per_year']}
    require(got_l == K_LATEST_PER_YEAR,
            f'latest 逐年 K 目标 {got_l} != {K_LATEST_PER_YEAR}；停机')
    out = {
        'run_at': datetime.now(timezone.utc).isoformat(),
        'script': 'scripts/data_audit/p3_1_build_asof_shadow.py',
        'method': 's3_b1_calibrate_tau.sw2_target 同构（文本年 ok 行 + 组内 N−1 中位 peer，跨年池化）',
        'target_key': 'effective_from <= (文本年+1)-04-30（网络 available_at 规则）',
        'asof': asof, 'latest': latest,
        'registered_asof_pooled': K_ASOF_POOLED, 'registered_asof_per_year': K_ASOF_PER_YEAR,
        'registered_latest_pooled': K_LATEST_POOLED, 'registered_latest_per_year': K_LATEST_PER_YEAR,
        'headline_k_target_used': 44,
        'planned_network_variants': {
            'N1_p3asof': {'dir': str(NET_DIR.relative_to(ROOT)).replace('\\', '/'),
                          'K_target': K_ASOF_POOLED, 'ver_prefix': SHADOW_VERSION_PREFIX},
            'N2_p3yt': {'K_target_per_year': K_ASOF_PER_YEAR, 'owner': '3.2（本项只提供目标）'},
            'N5_p3yt_latest': {'K_target_per_year': K_LATEST_PER_YEAR, 'owner': '3.2（可选轴）'},
        },
        'tau': {'status': 'deferred', 'reason': 'τ 属 M-B（TF-IDF）链；M-B shadow 留 Phase 4（项目决策）'},
        'elapsed_sec': round(time.time() - t0, 1),
    }
    write_json(K_TARGETS_JSON, out)
    log(f'targets：as-of 池化 {asof["median_peer_pooled_firm_level"]}（逐年 {list(got.values())}）；'
        f'latest 池化 {latest["median_peer_pooled_firm_level"]}')
    return out


def step_network(years, target: int) -> dict:
    t0 = time.time()
    injected = patch_corpora(target)
    res = {'injected_config': injected, 'target_used': target, 'years': {}}
    for y in years:
        res['years'][str(y)] = build_network_year(y, target)
    # version_spec 注入验证（3.6 依赖：shadow 目录不在 manifest 白名单，必须注入）
    spec = {'version_prefix': SHADOW_VERSION_PREFIX, 'years': [int(y) for y in years],
            'available_at_rule': NETWORK_AVAILABLE_AT_RULE}
    vers = load_versions(NET_DIR, version_spec=spec)
    probe_months = ['2015-01', '2016-06', '2021-06', '2024-01']
    res['version_spec_used'] = spec
    res['version_load_check'] = {
        'n_text_years': len(vers),
        'selected_text_year_by_month': {
            mo: (select_version(vers, mo) or {}).get('text_year') for mo in probe_months},
        'all_available_at': sorted({v['available_at'] for v in vers.values()}),
        'version_strings': sorted({f'{SHADOW_VERSION_PREFIX}_{y}' for y in vers}),
    }
    res['elapsed_sec'] = round(time.time() - t0, 1)
    log(f'network：{len(years)} 年完成（load_versions(version_spec) 通过，{len(vers)} 个年份）')
    return res


def step_sha_ledger() -> dict:
    """确定性产物（边表/support/聚类/映射/基线）sha 台账；与上次比对 = 重跑逐字节一致证据。

    注：`*_build_report.json` / 审计 JSON 含 `run_at`/`elapsed_sec`，不参与逐字节比对。
    """
    files = (sorted(NET_DIR.glob('*_similarity.csv')) + sorted(NET_DIR.glob('*_support.csv'))
             + sorted((NET_DIR / 'clusters').glob('*.csv'))
             + sorted(ART.glob('3.1_freeze_p3asof_*.json'))
             + [ASOF_MAP, ASOF_BASELINES, SAMPLE_ASOF_BASELINES])
    absent = [Path(p).relative_to(ROOT).as_posix() for p in files if not Path(p).is_file()]
    cur = {Path(p).relative_to(ROOT).as_posix(): sha256_file(p) for p in files if Path(p).is_file()}
    res = {'n_files': len(cur), 'absent_files': absent, 'sha': cur}
    if OUTPUT_SHA.is_file():
        prev = json.loads(OUTPUT_SHA.read_text(encoding='utf-8'))
        ps = prev.get('sha') or {}
        prev_absent = set(prev.get('absent_files') or [])
        new_files = sorted(k for k in cur if k in prev_absent)
        diff = sorted(k for k in set(ps) | set(cur) if ps.get(k) != cur.get(k))
        diff = [k for k in diff if '3.1_freeze_p3asof_' not in k and k not in new_files]
        res['rerun_check'] = {'status': 'IDENTICAL' if not diff else 'DIFFERENT',
                              'n_compared': len([k for k in cur if '3.1_freeze_p3asof_' not in k
                                                 and k not in new_files]),
                              'n_diff': len(diff), 'diff_keys': diff,
                              'new_files_since_last_ledger': new_files,
                              'note': 'freeze 报告（含 run_at/目标 K 记录）与上次台账缺失的新文件不参与'
                                      '逐字节比对'}
    write_json(OUTPUT_SHA, res)
    log(f'sha 台账：{res["n_files"]} 个确定性文件'
        + (f'；重跑比对 {res["rerun_check"]["status"]}（diff={res["rerun_check"]["n_diff"]}）'
           if 'rerun_check' in res else '（首次运行，无对比基线）'))
    return res


def step_ladder(years) -> dict:
    """K-中位 peer 阶梯诊断：解释 `closest` 命中区间与 |realized − target| 偏差（只读现算）。

    复用 `s3_b3_ma_cluster_grid.build_vectors`（teacher 语料 + headline 词表/筛选）与
    `s3_b3_ma_freeze` 同款 average linkage；报告 `K ∈ [min(K_y)−6, max(K_y)+6]` 的中位 peer，
    用于判定「目标 49 在该年阶梯上不可达（跳到 50 或停在 44）」是否为构造性事实。
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.preprocessing import normalize
    from s3_b3_ma_cluster_grid import build_vectors, load_refs
    from s3_ma_lexicon import build_automaton, load_lexicon

    terms, lmeta = load_lexicon(TEACHER_LEXICON)
    A = build_automaton(terms)
    uni = load_refs()
    rows = []
    t0 = time.time()
    for y in years:
        firms, X, stats = build_vectors(y, A, uni, corpus='teacher')
        C = (normalize(X) @ normalize(X).T).toarray().astype(np.float32)
        np.fill_diagonal(C, 1.0)
        D = np.clip(1.0 - C, 0, 2)
        np.fill_diagonal(D, 0.0)
        Z = linkage(squareform(D, checks=False), method='average')
        K_shadow = int(json.loads((ART / f'3.1_freeze_p3asof_{y}.json')
                                  .read_text(encoding='utf-8'))['years'][str(y)]['K'])
        K_head = HEADLINE_K[y]
        lo, hi = min(K_shadow, K_head) - 6, max(K_shadow, K_head) + 6
        for K in range(lo, hi + 1):
            lab = fcluster(Z, K, criterion='maxclust')
            _, inv = np.unique(lab, return_inverse=True)
            sizes = np.bincount(inv)
            peers = sizes[inv] - 1          # 公司层面 peer（与 cluster_stats 同口径）
            rows.append({'text_year': y, 'K': int(K), 'median_peer': float(np.median(peers)),
                         'mean_peer': round(float(peers.mean()), 2), 'size_max': int(sizes.max()),
                         'n_clusters': int(len(sizes)),
                         'is_shadow_K': bool(K == K_shadow), 'is_headline_K': bool(K == K_head)})
        log(f'ladder[{y}] n={len(firms)} K∈[{lo},{hi}] 已扫（shadow K={K_shadow}, '
            f'headline K={K_head}, {round(time.time() - t0, 1)}s）')
        del C, D, Z
    write_csv(ART / '3.1_k_ladder.csv', [list(r.values()) for r in rows], list(rows[0].keys()))
    reach = {}
    for y in years:
        med = {r['K']: r['median_peer'] for r in rows if r['text_year'] == y}
        hits = [k for k, v in med.items() if v == K_ASOF_POOLED]
        reach[str(y)] = {
            'K_at_median_peer_49': sorted(hits),
            'target_49_reachable': bool(hits),
            'closest_attainable': sorted({v for v in med.values()}, key=lambda v: (abs(v - K_ASOF_POOLED), v))[:4],
            'ladder_slice': {str(k): med[k] for k in sorted(med)},
        }
    res = {'years': [int(y) for y in years], 'lexicon': lmeta, 'target': K_ASOF_POOLED,
           'rows': len(rows), 'reachability': reach, 'elapsed_sec': round(time.time() - t0, 1)}
    write_json(ART / '3.1_k_ladder.json', res)
    log(f'ladder：{len(rows)} 行 → 3.1_k_ladder.csv；目标 49 可达性 '
        f'{ {k: v["target_49_reachable"] for k, v in reach.items()} }')
    return res


def step_accept(m: pd.DataFrame, years) -> tuple:
    rows, detail = acceptance(m, years)
    header = list(rows[0].keys())
    write_csv(NETWORK_COMPARE_CSV, [list(r.values()) for r in rows], header)
    lift_rows = []
    for y in years:
        for net in ('headline', 'p3asof'):
            for tag in ('lift_latest', 'lift_asof'):
                d = detail[y][net][tag]
                lift_rows.append({'text_year': y, 'network': net, 'label_basis': tag, **d})
    write_csv(ART / '3.1_lift_detail.csv', [list(r.values()) for r in lift_rows],
              list(lift_rows[0].keys()))
    log(f'accept：对比表 {len(rows)} 年；Jaccard {[r["jaccard_vs_headline"] for r in rows]}')
    return rows, detail


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description='Phase 3.1 as-of 行业 + K 目标 + p3asof shadow 网络')
    ap.add_argument('--steps', default='all',
                    help='all 或逗号分隔子集：preflight,map,baselines,targets,network,accept,ladder,postflight')
    ap.add_argument('--years', type=int, nargs='*', default=None, help='network 文本年（缺省全 9 年）')
    args = ap.parse_args()
    steps = (['preflight', 'map', 'baselines', 'targets', 'network', 'accept', 'postflight']
             if args.steps == 'all' else [s.strip() for s in args.steps.split(',') if s.strip()])
    years = args.years if args.years else YEARS
    t_all = time.time()
    SHADOW_DIR.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()
    counts = {'run_at': now, 'steps': steps,
              'shadow_version_tag': SHADOW_VERSION_TAG, 'years': years, 'run_history': []}
    # 增量运行合并：本次未执行的 step 保留上次结果（不丢审计轨迹），并记录 run_history
    if COUNTS_JSON.is_file():
        try:
            prev = json.loads(COUNTS_JSON.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            prev = None
        if isinstance(prev, dict):
            hist = prev.get('run_history') if isinstance(prev.get('run_history'), list) else []
            hist = hist + [{'run_at': prev.get('run_at'), 'steps': prev.get('steps'),
                            'elapsed_sec_total': prev.get('elapsed_sec_total')}]
            prev = {k: v for k, v in prev.items() if k != 'run_history'}
            prev.update({'run_at': now, 'steps': steps, 'years': years, 'run_history': hist})
            counts = prev

    if 'preflight' in steps:
        counts['anchor_pre'] = anchors_snapshot('pre')
        write_json(ANCHOR_PRE, counts['anchor_pre'])
        log(f'preflight：9 锚 + {len(counts["anchor_pre"]["headline_network_sha"])} 个 headline '
            f'网络文件 sha 已记录')

    months = months_between(SAMPLE_WINDOW_START, SAMPLE_WINDOW_END)
    m = load_sw2_map()
    if 'map' in steps:                      # 未执行的 step 保留上次结果（不写 None 覆盖）
        counts['map_audit'] = step_map(months, load_src_month_safe())

    if 'targets' in steps:
        teach = pd.read_csv(TEACHER_PANEL, usecols=['firm_id', 'year', 'status'])
        panel_teach = teach[teach['status'] == 'ok'].copy()
        panel_teach['firm_id'] = panel_teach['firm_id'].astype(str)
        counts['k_targets'] = step_targets(panel_teach)

    if 'baselines' in steps:
        counts['baselines'] = step_baselines(months)
    if 'network' in steps:
        net_res = step_network([int(y) for y in years],
                               (counts.get('k_targets') or {}).get('asof', {}).get(
                                   'median_peer_pooled_rounded', K_ASOF_POOLED))
        prev_net = counts.get('network') or {}
        if isinstance(prev_net.get('years'), dict):        # 增量运行：保留其它年份的读数
            merged = dict(prev_net['years'])
            merged.update(net_res['years'])
            net_res['years'] = merged
            net_res['years_rebuilt_this_run'] = [str(y) for y in years]
        counts['network'] = net_res
    if 'accept' in steps:
        rows, detail = step_accept(m, [int(y) for y in years])
        counts['network_compare'] = rows
        counts['lift_detail'] = {str(k): v for k, v in detail.items()}
        counts['output_sha_rerun'] = step_sha_ledger()
    if 'ladder' in steps:
        counts['k_ladder'] = step_ladder([int(y) for y in years])

    if 'postflight' in steps:
        post = anchors_snapshot('post')
        write_json(ANCHOR_POST, post)
        pre = json.loads(ANCHOR_PRE.read_text(encoding='utf-8')) if ANCHOR_PRE.is_file() else None
        cmp_ = compare_anchors(pre, post) if pre else {'status': 'NO_PRE'}
        counts['anchor_compare'] = cmp_
        log(f'postflight：锚对比 {cmp_["status"]}（diff={cmp_["n_diff"]}）')
        if cmp_['status'] != 'MATCH':
            write_json(COUNTS_JSON, counts)
            print('[HARD-GATE] headline/输入锚发生变化；停机（禁止重算覆盖）', file=sys.stderr, flush=True)
            raise SystemExit(2)

    counts['elapsed_sec_total'] = round(time.time() - t_all, 1)
    write_json(COUNTS_JSON, counts)
    log(f'完成：总耗时 {counts["elapsed_sec_total"]}s → {COUNTS_JSON}')


if __name__ == '__main__':
    main()
