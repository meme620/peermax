"""Phase 3.2：L-2 / L-6 校准 shadow 变体（逐年 target / 逐年 df / 组合 / latest 对照）。

范围（`artifacts/phase3/3.0_plan.md` §2.2 + 项目 3.2 任务书）：
  - 只写 shadow：`data/processed/phase3_shadow/lexicon_p3ydf_{y}.csv`、
    `data/processed/network_ma_teacher_<变体>/*`、`artifacts/phase3/*`。
  - **不改** headline 边表/样本/网络/pricing_gate/PC/gate/configs；不重跑 headline pricing_gate/PC；不写 G0；
    不碰 3.1 产物（`artifacts/phase3/3.1_*`、`network_ma_teacher_p3asof/`）——只读依赖。

四个变体（两轴：词表 df × K 目标映射；矩阵见 `VARIANTS`）：

| 变体 | 词表 | K 目标 | 隔离的效应 |
|---|---|---|---|
| `p3yt`        | headline（9 年上确界 df≤0.25） | 逐年 as-of 目标 38/40/44/49/50/52/68/57/64 | L-2：K 目标「逐年 vs 池化 44」 |
| `p3ydf`       | 逐年 df≤0.25（9 个 shadow 词表） | 池化 44（= headline 目标） | L-6：词表 df「逐年 vs 9 年上确界」 |
| `p3yt_ydf`    | 逐年 df | 逐年 as-of 目标 | 组合（两轴交互） |
| `p3yt_latest` | headline | 逐年 latest 目标 26/27/30/38/37/46/52/58/62 | 目标映射（latest vs as-of 逐年目标） |

**口径限定（不得改写）**：
  1. as-of = **申万分类生效日口径**（`effective_from <= (文本年+1)-04-30`），**不是严格 PIT**（见 3.1 §0）。
  2. **L2 映射不进入网络构造**：`build_vectors` / 聚类 / 边表只吃（语料 × 词表 × K 目标）；`l2_code` 只出现在
     (a) 逐年 K 目标的来源（as-of vs latest）、(b) lift 分式的行业标签。故 `p3yt` 与 `p3yt_latest` 的网络
     差异 = 目标向量差异（不是「标签进入网络」）；标签口径对网络的**构造性影响 = 0**（可证伪点：
     `test_l2_labels_do_not_enter_network_construction`）。
  3. 逐年词表 df：`df_ratio_y = doc_count_{y} / n_docs_{y}`（`product_lexicon_v2_teacher_df_by_year.csv`），
     未出现在该表的词条 df=0（保留）；阈值 0.25 与 headline 同。**不改匹配器**（`--lexicon` 原路径）。
  4. 逐年目标的可达性（**两级 K 选择**）：
     一级（plan §2.2 处方）= `network_business_cluster_freeze.calibrate_K` 的二分 + 网格命中（`K_mode ∈ {exact, closest}`）。
     该二分假设「中位 peer 对 K 单调不增」，且**终局候选只有 h−l ≤ 2 个 K** → 目标落在阶梯跳点上时
     `closest` 会偏出容差（实测 8 个 (变体,年) 的 |dev| > 2）。
     二级（本脚本）= 当一级 |dev| > 2 时，在「freeze 选定 K ∪ 参照网络 K」±40 的**扩展阶梯**上重选 K
     （|dev| 最小优先、其次与 freeze K 最近；未命中则再扩窗），落地于 `--steps rebuild`
     （内存 patch `calibrate_K`，聚类算法与全部参数不变）。两级结果逐值留
     `3.2_k_selection.csv` / `3.2_k_ladder_{headline,per_year}.csv`。
     3.1 的 `3.1_k_ladder.csv` 窗口按池化目标 49 建（9 年里仅 2018 的目标 50 落在窗口内）→ 覆盖不到逐年
     目标，故本脚本另建扩展阶梯。

步骤（`--steps`，缺省 all = preflight,lexicons,networks,ladder,select,rebuild,compare,postflight）：
  preflight   9 个 pricing_gate 输入锚 + 额外只读输入 + headline 网络全文件 sha 台账（`3.2_anchor_pre.json`）
  lexicons    逐年词表（`phase3_shadow/lexicon_p3ydf_{y}.csv`）+ 计数/命中质量 + `3.2_df_removed_terms.csv`
  networks    4 变体 × 9 年：复用 `network_business_cluster_freeze.py` / `network_business_edges_build.py`（**不改磁盘脚本**，
              运行时注入 shadow `ver_prefix` / K 规则标签 / shadow 默认输出路径）；可增量、可续跑
              （已完成年跳过；`--force` 强制重建；`3.2_progress_{variant}.json` 逐落盘）
  ladder      逐年 K-中位 peer 扩展阶梯（headline 词表 + 逐年词表两套）+ 目标可达性 + **二级 K 选择**
              （一级 freeze 自校准 |dev| > 2 → 扩展阶梯重选；落 `3.2_k_selection.csv`）
  select      汇总两套词表的可达性 → 合并 `3.2_k_selection.csv`（从阶梯表复算，不重扫）
  rebuild     落地二级 K 选择（内存 patch `calibrate_K`，只重建 K 有变的年）
  compare     变体 × headline × p3asof 对比表（边数/Jaccard/密度/peer 中位与 p90/lift 两口径/楔子/K/
              isolated）+ 验收硬门 + 词表/目标效应分解；`3.2_comparison.csv` / `3.2_wedge.csv` /
              `3.2_freeze_summary.csv` / `3.2_output_bytes.json` / `3.2_counts.json`
  postflight  再次核验锚与 headline 网络 sha（`3.2_anchor_post.json`）；CHANGED → 停机

用法：
  python scripts/shadow_calibration.py --steps all
  python scripts/shadow_calibration.py --steps lexicons,compare
  python scripts/shadow_calibration.py --steps networks --variants p3yt --years 2014 2015"""

from __future__ import annotations

import sys
from pathlib import Path
from fileio import assert_path_allowed, atomic_write_text, json_default, write_csv_rows, write_json_atomic

import argparse
import importlib
import json
import os
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import shadow_asof_industry as p31  # noqa: E402  （as-of 映射 / lift / 锚台账：as-of 交接件，只读复用）
from pricing_lib import assert_pricing_inputs, require, file_size
from peermax_panel_build import (
    NETWORK_AVAILABLE_AT_RULE, load_versions, select_version,
)


ROOT = Path(__file__).resolve().parents[1]

SHADOW_DIR = ROOT / 'data/processed/phase3_shadow'

ART = ROOT / 'artifacts/phase3'

HEADLINE_NET = ROOT / 'data/processed/network_ma_teacher'

P3ASOF_NET = ROOT / 'data/processed/network_ma_teacher_p3asof'      # 3.1 产物（只读）

TEACHER_PANEL = ROOT / 'data/processed/teacher_mda_panel.csv'

TEACHER_LEXICON = ROOT / 'data/processed/product_lexicon_v2_teacher.csv'

DF_BY_YEAR = ROOT / 'data/processed/product_lexicon_v2_teacher_df_by_year.csv'

K_TARGETS_3_1 = ART / '3.1_k_targets.json'

LADDER_3_1 = ART / '3.1_k_ladder.csv'

SW2_MAP = ROOT / 'data/processed/sw2_industry_map.csv'

COUNTS_JSON = ART / '3.2_counts.json'

COMPARISON_CSV = ART / '3.2_comparison.csv'

WEDGE_CSV = ART / '3.2_wedge.csv'

FREEZE_SUMMARY_CSV = ART / '3.2_freeze_summary.csv'

LEXICON_BY_YEAR_CSV = ART / '3.2_lexicon_by_year.csv'

DF_REMOVED_CSV = ART / '3.2_df_removed_terms.csv'

OUTPUT_SHA = ART / '3.2_output_bytes.json'

ANCHOR_PRE = ART / '3.2_anchor_pre.json'

ANCHOR_POST = ART / '3.2_anchor_post.json'

DF_MAX = 0.25

YEARS = list(range(2014, 2023))                     # 文本年 → 网络年 y+1

HEADLINE_VERSION_PREFIX = 'pm_ma_ewcos_teacher_v1'

P3ASOF_VERSION_PREFIX = 'pm_ma_ewcos_teacher_p3asof_v1'

K_ASOF_PER_YEAR = {2014: 38, 2015: 40, 2016: 44, 2017: 49, 2018: 50,
                   2019: 52, 2020: 68, 2021: 57, 2022: 64}

K_LATEST_PER_YEAR = {2014: 26, 2015: 27, 2016: 30, 2017: 38, 2018: 37,
                     2019: 46, 2020: 52, 2021: 58, 2022: 62}

K_ASOF_POOLED = 49

K_HEADLINE_POOLED = 44

HEADLINE_K = {2014: 235, 2015: 238, 2016: 322, 2017: 291, 2018: 334,
              2019: 407, 2020: 414, 2021: 434, 2022: 463}

HEADLINE_EDGES = {2014: 157096, 2015: 156974, 2016: 152302, 2017: 214028,
                  2018: 204786, 2019: 238458, 2020: 246728, 2021: 249254,
                  2022: 293938}

HEADLINE_NODES = {2014: 2384, 2015: 2363, 2016: 2503, 2017: 2721, 2018: 2835,
                  2019: 3354, 2020: 3427, 2021: 3570, 2022: 3942}

EDGE_COLS = ['focal_id', 'source_id', 'similarity', 'weight']

REGISTERED_RESCUED_WITH_HITS = {2014: 560, 2015: 819}

REGISTERED_HIT_MASS_SHARE = {2014: 0.0726, 2015: 0.1208}

REGISTERED_HIT_MASS_TOL = 0.0005                    # ±0.05pp

LADDER_WINDOW = 40            # 扩展阶梯半窗（覆盖 freeze 选定 K ∪ 参照网络 K）

VARIANTS = {
    'p3yt': {
        'dir': ROOT / 'data/processed/network_ma_teacher_p3yt',
        'ver_prefix': 'pm_ma_ewcos_teacher_p3yt_v1',
        'lexicon': 'headline', 'target_kind': 'asof_per_year', 'l2_mapping': 'asof',
        'isolated_effect': 'L-2 K 目标（逐年 as-of vs headline 池化 44）',
        'lexicon_desc': 'headline 词表（9 年上确界 df≤0.25）',
        'target_desc': '逐年 as-of 目标 38/40/44/49/50/52/68/57/64（as-of 交接）',
    },
    'p3ydf': {
        'dir': ROOT / 'data/processed/network_ma_teacher_p3ydf',
        'ver_prefix': 'pm_ma_ewcos_teacher_p3ydf_v1',
        'lexicon': 'per_year', 'target_kind': 'pooled_44', 'l2_mapping': 'latest_static',
        'isolated_effect': 'L-6 词表 df（逐年 df≤0.25 vs 9 年上确界；K 目标固定 44）',
        'lexicon_desc': '逐年 df≤0.25（phase3_shadow/lexicon_p3ydf_{y}.csv）',
        'target_desc': '池化 44（= headline 目标，隔离词表效应）',
    },
    'p3yt_ydf': {
        'dir': ROOT / 'data/processed/network_ma_teacher_p3yt_ydf',
        'ver_prefix': 'pm_ma_ewcos_teacher_p3yt_ydf_v1',
        'lexicon': 'per_year', 'target_kind': 'asof_per_year', 'l2_mapping': 'asof',
        'isolated_effect': '组合：逐年 as-of 目标 + 逐年 df（两轴交互）',
        'lexicon_desc': '逐年 df≤0.25（phase3_shadow/lexicon_p3ydf_{y}.csv）',
        'target_desc': '逐年 as-of 目标 38/40/44/49/50/52/68/57/64',
    },
    'p3yt_latest': {
        'dir': ROOT / 'data/processed/network_ma_teacher_p3yt_latest',
        'ver_prefix': 'pm_ma_ewcos_teacher_p3yt_latest_v1',
        'lexicon': 'headline', 'target_kind': 'latest_per_year', 'l2_mapping': 'latest_static',
        'isolated_effect': '目标映射（逐年 latest vs 逐年 as-of；词表固定 headline）',
        'lexicon_desc': 'headline 词表（9 年上确界 df≤0.25）',
        'target_desc': '逐年 latest 目标 26/27/30/38/37/46/52/58/62（as-of 交接对照）',
    },
}

VARIANT_ORDER = ['p3yt', 'p3ydf', 'p3yt_ydf', 'p3yt_latest']

REF_NETS = ['headline', 'p3asof']                   # 只读参照网络（3 比较列）

EXTRA_INPUTS = {
    'sw2_industry_map': SW2_MAP,
    'teacher_mda_panel': TEACHER_PANEL,
    'product_lexicon_v2_teacher': TEACHER_LEXICON,
    'product_lexicon_v2_teacher_df_by_year': DF_BY_YEAR,
}

READONLY_INPUTS = (TEACHER_LEXICON, TEACHER_PANEL, DF_BY_YEAR, SW2_MAP)

ALLOWED_ROOTS = (SHADOW_DIR, ART) + tuple(v['dir'] for v in VARIANTS.values())

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

_PRISTINE: dict = {}

K_SELECTION_CSV = ART / '3.2_k_selection.csv'

LADDER_MAX_EXTEND = 2         # 目标未命中时向最优侧再扩几次（每次 +LADDER_WINDOW）

K_TOLERANCE = 2               # plan R1：|realized median peer − target| ≤ 2（超出须逐年解释）

REP_VARIANT_BY_MODE = {'headline': 'p3yt', 'per_year': 'p3ydf'}

__all__ = ['main', 'VARIANTS', 'VARIANT_ORDER', 'target_for', 'lexicon_for', 'net_dir', 'net_prefix',
           'assert_output_path_guard', 'DF_MAX', 'YEARS', 'K_ASOF_PER_YEAR', 'K_LATEST_PER_YEAR', 'dk']

def assert_output_path_guard(path) -> Path:
    """写出路径必须落在 shadow 允许根内；命中 headline/G0/configs/3.1 产物 → SystemExit(2)。"""
    p = assert_path_allowed(path, ALLOWED_ROOTS, (*BANNED_ROOTS, *READONLY_INPUTS))
    if p.parent == ART.resolve() and (p.name.startswith('3.1_') or p.name.startswith('3.0')):
        print(f'[HARD-GATE] 3.1 产物与 3.0 方案只读: {p}', file=sys.stderr, flush=True)
        raise SystemExit(2)
    return p

def atomic_write_text(path, text: str) -> None:
    p = assert_output_path_guard(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, p)

def write_json(path, obj) -> None:
    write_json_atomic(obj, assert_output_path_guard(path), indent=1, default=json_default,
                      trailing_newline=False)

def write_csv(path, rows, header) -> None:
    write_csv_rows(assert_output_path_guard(path), rows, header)

def write_text(path, text: str) -> None:
    atomic_write_text(assert_output_path_guard(path), text)

def log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

def anchors_snapshot(tag: str) -> dict:
    ledger = assert_pricing_inputs()
    extra = {k: {'file': str(Path(v).relative_to(ROOT)).replace('\\', '/'), 'bytes': file_size(v)}
             for k, v in EXTRA_INPUTS.items()}
    snap = {
        'tag': tag, 'run_at': datetime.now(timezone.utc).isoformat(),
        'pricing_inputs_9': {r['input']: r['rows'] for r in ledger},
        'pricing_inputs_ledger': ledger,
        'extra_inputs': extra,
        'headline_network_bytes': p31.sha_ledger_headline_net(),
        'shadow_net_dirs_sha_ledger_note': (
            'headline 网络目录（data/processed/network_ma_teacher/）全文件 sha；'
            '3.2 新增的 4 个 shadow 目录不在本台账内（各自 build_report 记录 network_version 校验）'),
    }
    snap['headline_network_n_files'] = len(snap['headline_network_bytes'])
    return snap

def load_k_targets_3_1() -> dict:
    """读 as-of 交接的 K 目标并逐值对账（容差 0）；不符 → 停机。"""
    require(K_TARGETS_3_1.is_file(), f'缺 as-of 交接件 {K_TARGETS_3_1}；先跑 3.1')
    d = json.loads(K_TARGETS_3_1.read_text(encoding='utf-8'))
    got_asof = {int(p['text_year']): int(round(p['median_peer'])) for p in d['asof']['per_year']}
    got_latest = {int(p['text_year']): int(round(p['median_peer'])) for p in d['latest']['per_year']}
    require(got_asof == K_ASOF_PER_YEAR, f'3.1 as-of 逐年目标 {got_asof} != 登记 {K_ASOF_PER_YEAR}；停机')
    require(got_latest == K_LATEST_PER_YEAR,
            f'3.1 latest 逐年目标 {got_latest} != 登记 {K_LATEST_PER_YEAR}；停机')
    require(d['asof']['median_peer_pooled_rounded'] == K_ASOF_POOLED,
            f'3.1 as-of 池化目标 {d["asof"]["median_peer_pooled_rounded"]} != {K_ASOF_POOLED}；停机')
    require(d['latest']['median_peer_pooled_rounded'] == K_HEADLINE_POOLED,
            f'3.1 latest 池化目标 {d["latest"]["median_peer_pooled_rounded"]} != {K_HEADLINE_POOLED}；停机')
    return d

def target_for(variant: str, y: int) -> int:
    spec = VARIANTS[variant]
    kind = spec['target_kind']
    if kind == 'asof_per_year':
        return K_ASOF_PER_YEAR[y]
    if kind == 'latest_per_year':
        return K_LATEST_PER_YEAR[y]
    if kind == 'pooled_44':
        return K_HEADLINE_POOLED
    raise ValueError(kind)

def lexicon_for(variant: str, y: int) -> Path:
    if VARIANTS[variant]['lexicon'] == 'headline':
        return TEACHER_LEXICON
    return SHADOW_DIR / f'lexicon_p3ydf_{y}.csv'

def net_dir(net: str) -> Path:
    if net == 'headline':
        return HEADLINE_NET
    if net == 'p3asof':
        return P3ASOF_NET
    return VARIANTS[net]['dir']

def net_prefix(net: str) -> str:
    if net == 'headline':
        return HEADLINE_VERSION_PREFIX
    if net == 'p3asof':
        return P3ASOF_VERSION_PREFIX
    return VARIANTS[net]['ver_prefix']

def build_yearly_lexicons(years, force: bool = False) -> dict:
    """`lexicon_p3ydf_{y}.csv`（term, df_ratio）：df_ratio = doc_count_{y} / n_docs_{y}。

    行序与 headline 词表**逐行相同**（含未命中词条 df=0）→ `terms_total_in_file` 与 headline 一致，
    变体唯一差别 = 保留集合（df 效应可辨识）。读回逐值复核 ≤1e-15。
    """
    t0 = time.time()
    lex = pd.read_csv(TEACHER_LEXICON, usecols=['term', 'df_ratio', 'doc_count', 'df_year'])
    dy = pd.read_csv(DF_BY_YEAR)
    require(('term' in dy.columns), 'df_by_year 缺 term 列')
    dmap = dy.set_index('term')
    lex_headline_keep = (lex['df_ratio'].fillna(0.0) <= DF_MAX).to_numpy()
    res = {'run_at': datetime.now(timezone.utc).isoformat(),
           'script': 'scripts/shadow_calibration.py',
           'df_max': DF_MAX, 'lexicon_source_rows': int(len(lex)),
           'df_by_year_rows': int(len(dy)),
           'definition': 'df_ratio_y = doc_count_{y} / n_docs_{y}；未出现在 df_by_year 的词条 df=0（保留）',
           'row_order': '与 headline product_lexicon_v2_teacher.csv 逐行相同（order-preserved）',
           'compare_to_headline_note': (
               'headline 词表的 df_ratio = 9 年上确界（max_y doc_count_y / n_docs_y，见 df_year 列）；'
               '逐年词表把上确界换成该文本年值 → 只可能「救回」被 max 规则多剔的词，'
               '不会再剔掉 headline 保留的词（extra_dropped 恒为 0，逐值核查）'),
           'per_year': {}, 'max_abs_diff_reload': 0.0, 'registered_gate': {}}
    removed_rows = []
    for y in years:
        c_dc, c_n = f'doc_count_{y}', f'n_docs_{y}'
        require(c_dc in dmap.columns and c_n in dmap.columns, f'df_by_year 缺 {c_dc}/{c_n}')
        n_docs = float(dmap[c_n].iloc[0])
        dc = pd.to_numeric(dmap[c_dc], errors='coerce')
        ratio = (lex['term'].map(dc) / n_docs)
        ratio = ratio.fillna(0.0).astype(float)
        out_f = SHADOW_DIR / f'lexicon_p3ydf_{y}.csv'
        existed = out_f.is_file()
        if force or not existed:
            frame = pd.DataFrame({'term': lex['term'], 'df_ratio': ratio})
            write_csv(out_f, frame.itertuples(index=False, name=None), ['term', 'df_ratio'])
        back = pd.read_csv(out_f)
        require(len(back) == len(lex), f'[{y}] 逐年词表行数 {len(back)} != headline {len(lex)}')
        require(bool((back['term'].astype(str) == lex['term'].astype(str)).all()),
                f'[{y}] 逐年词表 term 列与 headline 词表行序不一致；停机')
        diff = float(np.max(np.abs(back['df_ratio'].to_numpy(dtype=float) - ratio.to_numpy())))
        res['max_abs_diff_reload'] = max(res['max_abs_diff_reload'], diff)
        require(diff <= 1e-15, f'[{y}] 逐年词表读回与 df_by_year 重算不一致（max|Δ|={diff}）；停机')
        keep_y = (ratio <= DF_MAX).to_numpy()
        dc_arr = lex['term'].map(dc).fillna(0.0).to_numpy(dtype=float)
        nz = dc_arr > 0
        rescued = (~lex_headline_keep) & keep_y                 # max 规则多剔、逐年 df 保留
        rescued_hits = rescued & nz
        extra_dropped = lex_headline_keep & (~keep_y)            # 反向：应为 0
        total_hits = float(dc_arr.sum())
        hit_mass = float(dc_arr[rescued].sum() / total_hits) if total_hits else 0.0
        res['per_year'][str(y)] = {
            'n_docs': int(n_docs), 'terms_total_in_file': int(len(lex)),
            'terms_kept_headline': int(lex_headline_keep.sum()),
            'terms_kept_per_year_df': int(keep_y.sum()),
            'terms_added_vs_headline_all': int(rescued.sum()),
            'terms_added_vs_headline_with_hits': int(rescued_hits.sum()),
            'terms_extra_dropped_vs_headline': int(extra_dropped.sum()),
            'hit_terms_y': int(nz.sum()),
            'pct_of_hit_terms_rescued': round(100.0 * int(rescued_hits.sum()) / max(1, int(nz.sum())), 4),
            'hit_mass_share_rescued': round(hit_mass, 6),
            'headline_terms_dropped_by_max_rule': int((~lex_headline_keep).sum()),
            'file': str(out_f.relative_to(ROOT)).replace('\\', '/'),
            'file_bytes': int(out_f.stat().st_size), 'file_bytes': file_size(out_f),
            'file_existed_before_run': bool(existed),
            'n_terms_whose_headline_df_year_is_y': int(
                ((~lex_headline_keep) & (lex['df_year'].astype(str) == str(y))).sum()),
        }
        # R1 预登记对账（plan §2.2）：2014 = 560 词 / 7.26%；2015 = 819 词 / 12.08%（±0.05pp）
        if y in REGISTERED_RESCUED_WITH_HITS:
            exp_n = REGISTERED_RESCUED_WITH_HITS[y]
            exp_s = REGISTERED_HIT_MASS_SHARE[y]
            ok_n = int(rescued_hits.sum()) == exp_n
            ok_s = abs(hit_mass - exp_s) <= REGISTERED_HIT_MASS_TOL
            res['registered_gate'][str(y)] = {
                'terms_added_with_hits': int(rescued_hits.sum()), 'expected': exp_n, 'pass': bool(ok_n),
                'hit_mass_share': round(hit_mass, 6), 'expected_share': exp_s, 'share_pass': bool(ok_s),
                'tolerance_pp': round(REGISTERED_HIT_MASS_TOL * 100, 4)}
            require(ok_n and ok_s,
                    f'[{y}] 逐年词表与 plan R1 预登记不符（多剔词 {int(rescued_hits.sum())} vs {exp_n}；'
                    f'命中质量 {hit_mass:.6f} vs {exp_s}）；停机')
        if y in (2014, 2015):
            sub = pd.DataFrame({
                'text_year': y, 'term': lex['term'][rescued_hits].to_numpy(),
                'doc_count_y': dc_arr[rescued_hits], 'df_ratio_y': ratio.to_numpy()[rescued_hits],
                'headline_df_ratio_sup': lex['df_ratio'].to_numpy()[rescued_hits],
                'headline_df_year_argmax': lex['df_year'].to_numpy()[rescued_hits],
                'share_of_year_hit_mass': dc_arr[rescued_hits] / total_hits,
                'n_docs_y': int(n_docs),
            })
            sub = sub.sort_values(['doc_count_y', 'term'], ascending=[False, True])
            removed_rows.append(sub)
        log(f'lexicons[{y}] 保留 {int(keep_y.sum()):,}（headline {int(lex_headline_keep.sum()):,}）；'
            f'max 规则多剔 {int(rescued_hits.sum())} 词 / 命中质量 {hit_mass:.4%}；'
            f'反向多剔 {int(extra_dropped.sum())}')
    if removed_rows:
        allr = pd.concat(removed_rows, ignore_index=True)
        write_csv(DF_REMOVED_CSV, allr.itertuples(index=False, name=None), list(allr.columns))
        log(f'lexicons：多剔词明细（2014/2015）{len(allr)} 行 → {DF_REMOVED_CSV.name}')
    rows = []
    for ky in sorted(res['per_year'], key=int):
        d = res['per_year'][ky]
        rows.append({'text_year': int(ky), **d})
    write_csv(LEXICON_BY_YEAR_CSV, [[r.get(k) for k in rows[0].keys()] for r in rows],
              list(rows[0].keys()))
    res['lexicon_by_year_csv'] = str(LEXICON_BY_YEAR_CSV.relative_to(ROOT)).replace('\\', '/')
    res['df_removed_terms_csv'] = str(DF_REMOVED_CSV.relative_to(ROOT)).replace('\\', '/')
    res['elapsed_sec'] = round(time.time() - t0, 1)
    log(f'lexicons：9 年词表完成（max|Δ| 读回 = {res["max_abs_diff_reload"]:.2e}，'
        f'{res["elapsed_sec"]}s）')
    return res

def _variant_modules():
    """导入现脚本并记录**首次导入时的原始常量**（防「同名不同义」：patch 前必须等于磁盘值）。"""
    cg = importlib.import_module('network_business_cluster_grid')
    frz = importlib.import_module('network_business_cluster_freeze')
    be = importlib.import_module('network_business_edges_build')
    if not _PRISTINE:
        _PRISTINE.update({
            'cg_ver_prefix': cg.CORPORA['teacher']['ver_prefix'],
            'be_ver_prefix': be.CORPORA['teacher']['ver_prefix'],
            'be_k_rule': be.CORPORA['teacher']['k_rule'],
            'frz_out_dir': frz.SPECS['teacher']['out_dir'],
            'frz_report': frz.SPECS['teacher']['report'],
            'be_net_dir': be.CORPORA['teacher']['net_dir'],
        })
    return cg, frz, be

def patch_corpora(variant: str, y: int) -> dict:
    """运行时注入 shadow 版本前缀 / K 规则标签 / shadow 默认输出路径（**不改磁盘脚本**）。

    守卫：注入前 `CORPORA/SPECS` 的原始值必须是 headline 值（`_PRISTINE` = 首次导入磁盘值），
    且本变体前缀与 headline / 3.1 p3asof 均不同名。
    """
    spec = VARIANTS[variant]
    d = spec['dir']
    cg, frz, be = _variant_modules()
    require(_PRISTINE['be_ver_prefix'] == HEADLINE_VERSION_PREFIX,
            f'edges CORPORA[teacher].ver_prefix 磁盘值非 {HEADLINE_VERSION_PREFIX}；拒绝继续')
    require(_PRISTINE['cg_ver_prefix'] == HEADLINE_VERSION_PREFIX,
            f'cluster_grid CORPORA[teacher].ver_prefix 磁盘值非 {HEADLINE_VERSION_PREFIX}；拒绝继续')
    require(_PRISTINE['be_k_rule'] == 'per_year_target_peer_match(44)',
            f'edges CORPORA[teacher].k_rule 磁盘值意外：{_PRISTINE["be_k_rule"]!r}')
    require(spec['ver_prefix'] != HEADLINE_VERSION_PREFIX
            and not spec['ver_prefix'].startswith(P3ASOF_VERSION_PREFIX),
            f'变体 {variant} 版本前缀 {spec["ver_prefix"]} 与 headline/3.1 同名；停机')
    cg.CORPORA['teacher']['ver_prefix'] = spec['ver_prefix']
    be.CORPORA['teacher']['ver_prefix'] = spec['ver_prefix']
    target = target_for(variant, y)
    be.CORPORA['teacher']['k_rule'] = f'per_year_target_peer_match({target})'
    frz.SPECS['teacher']['out_dir'] = d / 'clusters'
    frz.SPECS['teacher']['report'] = ART / f'3.2_freeze_{variant}_default.json'
    be.CORPORA['teacher']['net_dir'] = d
    return {'ver_prefix': spec['ver_prefix'], 'k_rule': be.CORPORA['teacher']['k_rule'],
            'freeze_out_dir': str(frz.SPECS['teacher']['out_dir']),
            'pristine_disk_values': {'ver_prefix': _PRISTINE['be_ver_prefix'],
                                     'k_rule': _PRISTINE['be_k_rule']}}

def _call_main(mod, argv) -> None:
    old = sys.argv
    try:
        sys.argv = [Path(mod.__file__).name] + list(argv)
        mod.main()
    finally:
        sys.argv = old

def progress_path(variant: str) -> Path:
    """变体级 progress（**按变体分文件**：并行构建 4 变体时避免共享文件竞态）。"""
    return ART / f'3.2_progress_{variant}.json'

def year_built(variant: str, y: int, target: int, prog: dict) -> bool:
    """已完成判定：4 个产物齐 + 版本串/available_at 正确 + progress 记录的目标一致。"""
    d = VARIANTS[variant]['dir']
    files = [d / f'{y}_build_report.json', d / f'{y}_similarity.csv', d / f'{y}_support.csv',
             d / 'clusters' / f'{y}_clusters.csv']
    if not all(f.is_file() for f in files):
        return False
    try:
        rep = json.loads(files[0].read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    if rep.get('network_version') != f'{VARIANTS[variant]["ver_prefix"]}_{y}':
        return False
    if rep.get('available_at') != f'{y + 1}-04-30':
        return False
    prev = ((prog.get('variants') or {}).get(variant) or {}).get(str(y)) or {}
    return int(prev.get('target', -1)) == int(target) and prev.get('status') == 'done'

def build_variant_year(variant: str, y: int, force: bool = False,
                       k_override: int | None = None, k_mode: str = 'ladder_selected') -> dict:
    """单变体单年：freeze（K 校准 + 聚类标签）→ edges（边表/support/build_report）。

    `k_override`：两级 K 选择（见 `step_ladder`）的二级——freeze 自校准 |dev| > 2 时，用扩展阶梯
    选定的 K 覆盖 freeze 的 `calibrate_K`（**只在内存 patch，不改磁盘脚本**；团数/算法全同）。
    """
    spec = VARIANTS[variant]
    d, lex = spec['dir'], lexicon_for(variant, y)
    target = target_for(variant, y)
    require(lex.is_file(), f'[{variant}/{y}] 词表不存在：{lex}（先跑 --steps lexicons）')
    inj = patch_corpora(variant, y)
    t0 = time.time()
    frz_rep = ART / f'3.2_freeze_{variant}_{y}.json'
    log(f'[{variant}/{y}] freeze：target={target} lexicon={lex.name} '
        f'{"" if k_override is None else f"K_override={k_override} "}→ {d.name}/clusters')
    frz = _variant_modules()[1]
    restore = None
    if k_override is not None:
        _orig_ck = frz.calibrate_K

        def _fixed_ck(Z, l2_idx, n_l2, base, target, lo=20, hi=4000, iters=14, _k=int(k_override),
                      _m=str(k_mode)):
            return _k, _m
        frz.calibrate_K = _fixed_ck
        restore = lambda: setattr(frz, 'calibrate_K', _orig_ck)
    try:
        _call_main(frz, ['--corpus', 'teacher', '--years', str(y), '--target', str(target),
                         '--lexicon', str(lex), '--out-dir', str(d / 'clusters'),
                         '--out-report', str(frz_rep)])
    finally:
        if restore:
            restore()
    log(f'[{variant}/{y}] edges：ver_prefix={spec["ver_prefix"]} → {d.name}')
    _call_main(_variant_modules()[2], ['--corpus', 'teacher', '--years', str(y), '--net-dir', str(d),
                                       '--cluster-dir', str(d / 'clusters'), '--lexicon', str(lex)])
    rep = json.loads((d / f'{y}_build_report.json').read_text(encoding='utf-8'))
    expected_ver = f'{spec["ver_prefix"]}_{y}'
    require(rep['network_version'] == expected_ver,
            f'[{variant}/{y}] 边表 network_version={rep["network_version"]!r} != {expected_ver!r}')
    require(rep['available_at'] == f'{y + 1}-04-30',
            f'[{variant}/{y}] available_at={rep["available_at"]!r} != {(y + 1)}-04-30')
    require(rep['params']['K_rule'] == f'per_year_target_peer_match({target})',
            f'[{variant}/{y}] build_report K_rule={rep["params"]["K_rule"]!r} 与目标 {target} 不符')
    frz_doc = json.loads(frz_rep.read_text(encoding='utf-8'))
    fy = frz_doc['years'][str(y)]
    require(int(fy['K']) == int(rep['k_clusters']),
            f'[{variant}/{y}] freeze K={fy["K"]} != 边表 k_clusters={rep["k_clusters"]}')
    if k_override is not None:
        require(int(fy['K']) == int(k_override),
                f'[{variant}/{y}] K_override={k_override} 未生效（freeze 记 {fy["K"]}）')
    assert_output_path_guard(d / f'{y}_similarity.csv')
    return {'status': 'done', 'variant': variant, 'text_year': y, 'network_year': y + 1,
            'target': target, 'K': int(rep['k_clusters']), 'K_mode': fy['K_mode'],
            'K_override_used': (int(k_override) if k_override is not None else None),
            'peer_median': float(fy['peer_median']), 'n_edges': int(rep['n_edges']),
            'n_nodes': int(rep['n_nodes']), 'n_support': int(rep['n_support']),
            'n_no_support': int(rep['n_no_support']), 'lexicon': str(lex.relative_to(ROOT)).replace('\\', '/'),
            'network_version': rep['network_version'], 'available_at': rep['available_at'],
            'injected': inj, 'elapsed_sec': round(time.time() - t0, 1),
            'built_at': datetime.now(timezone.utc).isoformat()}

def step_networks(variants, years, force=False) -> dict:
    """4 变体 × 9 年构建；可增量（跳过已完成年）、逐年写 progress（超时/中断可续跑）。"""
    t0 = time.time()
    res = {'targets_used': {v: {str(y): target_for(v, y) for y in years} for v in variants},
           'variants': {}, 'skipped_years': [], 'rebuilt_years': [], 'force': bool(force)}
    for v in variants:
        prog = load_json_or(progress_path(v), {'variants': {}})
        res['variants'].setdefault(v, {'years': {}, 'ver_prefix': VARIANTS[v]['ver_prefix'],
                                       'dir': str(VARIANTS[v]['dir'].relative_to(ROOT)).replace('\\', '/'),
                                       'progress': str(progress_path(v).relative_to(ROOT)).replace('\\', '/')})
        for y in years:
            if not force and year_built(v, y, target_for(v, y), prog):
                prev = ((prog.get('variants') or {}).get(v) or {}).get(str(y)) or {}
                res['variants'][v]['years'][str(y)] = prev
                res['skipped_years'].append(f'{v}/{y}')
                log(f'[{v}/{y}] 已完成（target={prev.get("target")}）→ 跳过')
                continue
            out = build_variant_year(v, y, force=force)
            res['variants'][v]['years'][str(y)] = out
            res['rebuilt_years'].append(f'{v}/{y}')
            prog.setdefault('variants', {}).setdefault(v, {})[str(y)] = out
            prog['updated_at'] = datetime.now(timezone.utc).isoformat()
            write_json(progress_path(v), prog)
    # version_spec 注入验证（3.6 依赖：shadow 目录不登记 manifest）
    res['version_spec_check'] = version_spec_check(variants, years)
    res['elapsed_sec'] = round(time.time() - t0, 1)
    log(f'networks：{len(variants)} 变体 × {len(years)} 年完成（重建 {len(res["rebuilt_years"])}，'
        f'跳过 {len(res["skipped_years"])}，{res["elapsed_sec"]}s）')
    return res

def load_json_or(path: Path, default):
    if Path(path).is_file():
        try:
            return json.loads(Path(path).read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return default
    return default

def _best_choice(med: dict, target: int, ref_K: int | None):
    """在已扫描阶梯上选 K：先按 |dev| 最小，再按与参照 K（freeze 选定值）最近。"""
    if not med:
        return None, None
    best = min(med, key=lambda k: (abs(med[k] - target), abs(k - (ref_K if ref_K else k)), k))
    return int(best), float(abs(med[best] - target))

def step_ladder(years, lexicon_mode: str, window: int = LADDER_WINDOW,
                force_scan_all: bool = False) -> dict:
    """逐年 K→中位 peer 扩展阶梯 + 目标可达性 + **二级 K 选择**（freeze 粗网格失效时的兜底）。

    一级（plan §2.2 处方）= `network_business_cluster_freeze.calibrate_K` 的二分 + 网格命中（K_mode = exact/closest）；
    二级（本步）= 当一级 |dev| > 2 时，在 freeze 选定 K ∪ 参照网络 K 的 ±window 阶梯上重选 K
    （|dev| 最小优先，其次与 freeze K 最近；未命中则向最优侧再扩 max_extend 次）。
    `force_scan_all=True` 时对所有变体都出选择结果（用于留档），否则只标记 `needs_rebuild`。

    `lexicon_mode='headline'` → 服务 `p3yt` / `p3yt_latest`；`lexicon_mode='per_year'` → `p3ydf` / `p3yt_ydf`。
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.preprocessing import normalize

    from network_business_cluster_grid import build_vectors, load_refs
    from network_business_lexicon import build_automaton, load_lexicon

    t0 = time.time()
    uni = load_refs()
    progs = {v: load_json_or(progress_path(v), {'variants': {}}) for v in VARIANT_ORDER}
    rows, reach, selection = [], {}, []
    for y in years:
        lex = TEACHER_LEXICON if lexicon_mode == 'headline' else SHADOW_DIR / f'lexicon_p3ydf_{y}.csv'
        require(lex.is_file(), f'阶梯：词表不存在 {lex}')
        terms, lmeta = load_lexicon(lex)
        A = build_automaton(terms)
        firms, X, stats = build_vectors(y, A, uni, corpus='teacher')
        C = (normalize(X) @ normalize(X).T).toarray().astype(np.float32)
        np.fill_diagonal(C, 1.0)
        D = np.clip(1.0 - C, 0, 2)
        np.fill_diagonal(D, 0.0)
        Z = linkage(squareform(D, checks=False), method='average')

        def selected_K(v):
            prev = ((progs.get(v) or {}).get('variants') or {}).get(v) or {}
            prev = prev.get(str(y)) or {}
            if prev.get('K'):
                return int(prev['K'])
            f = ART / f'3.2_freeze_{v}_{y}.json'
            if f.is_file():
                return int(json.loads(f.read_text(encoding='utf-8'))['years'][str(y)]['K'])
            return None

        vs = [v for v in VARIANT_ORDER if VARIANTS[v]['lexicon'] == lexicon_mode]
        targets = {v: target_for(v, y) for v in vs}
        freeze_K = {v: selected_K(v) for v in vs}
        ref_Ks = []
        for ref in REF_NETS:
            f = net_dir(ref) / f'{y}_build_report.json'
            if f.is_file():
                ref_Ks.append(int(json.loads(f.read_text(encoding='utf-8'))['k_clusters']))
        anchors = [k for k in list(freeze_K.values()) + ref_Ks if k]
        require(bool(anchors), f'阶梯[{lexicon_mode}/{y}] 无 K 锚点（先跑 --steps networks）')
        med = {}

        def scan(lo, hi):
            hit = 0
            for K in range(max(20, int(lo)), min(4000, int(hi)) + 1):
                if K in med:
                    continue
                lab = fcluster(Z, K, criterion='maxclust')
                _, inv = np.unique(lab, return_inverse=True)
                sizes = np.bincount(inv)
                peers = sizes[inv] - 1
                m = float(np.median(peers))
                med[K] = m
                rows.append({'text_year': y, 'lexicon_mode': lexicon_mode, 'K': int(K), 'median_peer': m,
                             'mean_peer': round(float(peers.mean()), 2), 'size_max': int(sizes.max()),
                             'n_clusters': int(len(sizes)),
                             'is_freeze_K_of': ';'.join(sorted(f'{v}={freeze_K[v]}' for v in vs
                                                              if freeze_K.get(v) == K)),
                             'is_reference_K_of': ';'.join(sorted(f'{r}={k}' for r, k in
                                                                  zip(REF_NETS, ref_Ks) if k == K))})
                hit += 1
            return hit

        lo, hi = min(anchors) - window, max(anchors) + window
        n_scanned = scan(lo, hi)
        ext_log = []
        for v in vs:
            t = targets[v]
            K_sel, dev_sel = _best_choice(med, t, freeze_K.get(v))
            for _ in range(LADDER_MAX_EXTEND):
                if dev_sel == 0:
                    break
                # 最优 K 落在窗口边界 → 向该侧再扩一窗（目标可能在窗外）
                side = -1 if K_sel == min(med) else (1 if K_sel == max(med) else 0)
                if side == 0:
                    break
                n_ext = scan(min(med) - window, min(med) - 1) if side < 0 else scan(max(med) + 1, max(med) + window)
                ext_log.append({'variant': v, 'side': ('lower' if side < 0 else 'upper'), 'n_new_K': n_ext})
                K_sel, dev_sel = _best_choice(med, t, freeze_K.get(v))
        del C, D, Z
        log(f'ladder[{lexicon_mode}/{y}] n={len(firms)} K∈[{min(med)},{max(med)}]（{len(med)} 个 K，'
            f'首窗 {n_scanned}，扩 {len(ext_log)}）')
    write_csv(ART / f'3.2_k_ladder_{lexicon_mode}.csv', [list(r.values()) for r in rows], list(rows[0].keys()))
    sel = selection_from_ladder_csv(lexicon_mode)
    res = {'lexicon_mode': lexicon_mode, 'years': [int(y) for y in years], 'rows': len(rows),
           'window': window, 'max_extend': LADDER_MAX_EXTEND,
           'reachability': sel['reachability'], 'selection': sel['selection'],
           'csv': str((ART / f'3.2_k_ladder_{lexicon_mode}.csv').relative_to(ROOT)).replace('\\', '/'),
           'note': sel['note'],
           'elapsed_sec': round(time.time() - t0, 1)}
    log(f'ladder[{lexicon_mode}]：{len(rows)} 行 → {res["csv"]}（{res["elapsed_sec"]}s）')
    return res

def _freeze_K(v: str, y: int):
    """该 (变体, 年) 已落盘 freeze 报告的 K（一级结果；无则 None）。"""
    f = ART / f'3.2_freeze_{v}_{y}.json'
    if not f.is_file():
        return None
    d = json.loads(f.read_text(encoding='utf-8'))
    st = (d.get('years') or {}).get(str(y))
    return int(st['K']) if st else None

def selection_from_ladder_csv(lexicon_mode: str) -> dict:
    """由**已落盘的阶梯表**重算目标可达性 + 二级 K 选择（与 `step_ladder` 共用同一逻辑）。

    好处：阶梯扫描昂贵（≈35 s/年），选择/汇总可随时从 CSV 复算（成本 <1 s）；且落盘证据与报告
    逐值同源。`n_nodes` / `filter_stats` 取该词表模式代表变体的 freeze 报告（同词表 → 逐值相同）。
    """
    csv_p = ART / f'3.2_k_ladder_{lexicon_mode}.csv'
    require(csv_p.is_file(), f'阶梯表缺失：{csv_p}（先跑 --steps ladder）')
    d = pd.read_csv(csv_p)
    rep_v = REP_VARIANT_BY_MODE[lexicon_mode]
    vs = [v for v in VARIANT_ORDER if VARIANTS[v]['lexicon'] == lexicon_mode]
    out_rows, reach = [], {}
    for y in sorted(int(x) for x in d['text_year'].unique()):
        s = d[d['text_year'] == y].sort_values('K')
        med = dict(zip(s['K'].astype(int), s['median_peer'].astype(float)))
        require(bool(med), f'阶梯表 {csv_p.name} 缺 {y} 年')
        fz = json.loads((ART / f'3.2_freeze_{rep_v}_{y}.json').read_text(encoding='utf-8'))['years'][str(y)]
        lex_path = (TEACHER_LEXICON if lexicon_mode == 'headline'
                    else SHADOW_DIR / f'lexicon_p3ydf_{y}.csv')
        freeze_K = {v: _freeze_K(v, y) for v in vs}
        ref_Ks = []
        for ref in REF_NETS:
            f = net_dir(ref) / f'{y}_build_report.json'
            if f.is_file():
                ref_Ks.append(int(json.loads(f.read_text(encoding='utf-8'))['k_clusters']))
        eins = {}
        for v in vs:
            t = target_for(v, y)
            K_sel, dev_sel = _best_choice(med, t, freeze_K.get(v))
            dev_freeze = (abs(med[freeze_K[v]] - t) if freeze_K.get(v) in med else None)
            hits = sorted(k for k, val in med.items() if val == t)
            closest = sorted({val for val in med.values()}, key=lambda val: (abs(val - t), val))[:4]
            best_any = min(med, key=lambda k: (abs(med[k] - t), k))
            need_rb = bool(freeze_K.get(v) is not None and K_sel is not None
                           and int(K_sel) != int(freeze_K[v]) and (dev_freeze is None or dev_freeze > K_TOLERANCE))
            eins[v] = {'target': t, 'K_freeze': freeze_K.get(v), 'dev_freeze': dev_freeze,
                       'K_selected': K_sel, 'dev_selected': dev_sel,
                       'mode_selected': ('exact' if dev_sel == 0 else 'ladder_closest'),
                       'target_reachable_in_window': bool(hits), 'K_at_target_in_window': hits,
                       'closest_attainable_median_peer': closest,
                       'best_attainable_dev_in_window': float(abs(med[best_any] - t)),
                       'freeze_K_is_best_attainable': bool(int(best_any) == int(freeze_K.get(v) or -1)
                                                           or dev_sel == dev_freeze),
                       'needs_rebuild': need_rb,
                       'window_final': [min(med), max(med)], 'n_K_scanned': len(med)}
            out_rows.append({
                'variant': v, 'text_year': y, 'lexicon_mode': lexicon_mode, 'target': t,
                'K_freeze': freeze_K.get(v), 'dev_freeze': dev_freeze,
                'K_selected': K_sel, 'dev_selected': dev_sel, 'mode_selected': eins[v]['mode_selected'],
                'target_reachable_in_window': eins[v]['target_reachable_in_window'],
                'best_attainable_dev_in_window': eins[v]['best_attainable_dev_in_window'],
                'needs_rebuild': need_rb,
                'selection_rule': ('freeze_first（一级命中且 |dev| ≤ 2 → 保留 freeze K）；'
                                   'ladder_reselect（一级 |dev| > 2 → 扩展阶梯重选）'),
            })
        reach[str(y)] = {'lexicon_mode': lexicon_mode,
                         'lexicon': str(lex_path.relative_to(ROOT)).replace('\\', '/'),
                         'n_nodes': int(fz['n_firms']), 'filter_stats': fz['filter_stats'],
                         'freeze_Ks': freeze_K, 'reference_Ks': ref_Ks,
                         'window_final': [min(med), max(med)], 'n_K_scanned': len(med),
                         'jumps': [{'K': k, 'from': med[k - 1], 'to': med[k]}
                                   for k in sorted(med) if k - 1 in med and med[k] != med[k - 1]],
                         'median_peer_at_reference_K': {str(k): med.get(k) for k in ref_Ks},
                         'variants': eins}
    return {'lexicon_mode': lexicon_mode, 'rows': int(len(d)), 'reachability': reach,
            'selection': out_rows, 'csv': str(csv_p.relative_to(ROOT)).replace('\\', '/'),
            'note': ('两级 K 选择：一级 = freeze 二分+网格（plan §2.2 处方，K_mode ∈ {exact, closest}）；'
                     '二分假设「中位 peer 对 K 单调不增」，且**终局候选只有 h−l ≤ 2 个 K**。'
                     '二级 = 本扩展阶梯（[freeze K ∪ 参照网络 K] ±40，未命中且最优 K 在边界时再扩窗）'
                     '上重选 K（|dev| 最小优先，其次与 freeze K 最近）。3.1 的 `3.1_k_ladder.csv` 窗口'
                     '按池化目标 49 建（9 年里仅 2018 的目标 50 落在窗口内）→ 覆盖不到逐年目标，故另建。'
                     '`target_reachable_in_window=False` 表示**扫描窗内**目标不可达（阶梯跳点/平台），'
                     '不是全域不可达的数学证明；窗口取 freeze K ∪ 参照 K 的邻域。')}

def step_select(years, modes) -> dict:
    """汇总两套词表的可达性/选择 → 合并 `3.2_k_selection.csv`（不重算阶梯）。"""
    merged, all_sel = {}, []
    for m in modes:
        r = selection_from_ladder_csv(m)
        all_sel += r['selection']
        merged[m] = {'lexicon_mode': m, 'rows': r['rows'], 'csv': r['csv'], 'note': r['note'],
                     'window': LADDER_WINDOW, 'max_extend': LADDER_MAX_EXTEND,
                     'reachability': r['reachability']}
    all_sel = [x for x in all_sel if int(x['text_year']) in set(int(y) for y in years)]
    all_sel.sort(key=lambda x: (x['variant'], int(x['text_year'])))
    write_csv(K_SELECTION_CSV, [list(r.values()) for r in all_sel], list(all_sel[0].keys()))
    n_rb = sum(1 for r in all_sel if r['needs_rebuild'])
    log(f'select：K 选择表 {len(all_sel)} 行 → {K_SELECTION_CSV.name}（需二级重建 {n_rb}；'
        f'|dev| > 2 的年 {sum(1 for r in all_sel if r["dev_selected"] > K_TOLERANCE)}）')
    return {'k_ladder': merged, 'k_selection': all_sel,
            'n_needs_rebuild': n_rb, 'k_selection_csv': str(K_SELECTION_CSV.relative_to(ROOT)).replace('\\', '/')}

def step_rebuild(years, variants, selection_rows) -> dict:
    """二级 K 选择落地：对 `needs_rebuild` 的 (variant, year) 用选定 K 重建网络（内存 patch `calibrate_K`）。"""
    t0 = time.time()
    sel = {(r['variant'], int(r['text_year'])): r for r in selection_rows}
    res = {'rebuilt': [], 'kept': [], 'missing_selection': []}
    prog = {}
    for v in variants:
        for y in years:
            r = sel.get((v, y))
            if r is None:
                res['missing_selection'].append(f'{v}/{y}')
                continue
            if not r['needs_rebuild']:
                res['kept'].append({'variant': v, 'text_year': y, 'K': r['K_freeze'],
                                    'dev_freeze': r['dev_freeze'], 'reason': 'freeze_first'})
                continue
            rep = read_rep(v, y)
            if int(rep['k_clusters']) == int(r['K_selected']):
                res['kept'].append({'variant': v, 'text_year': y, 'K': int(rep['k_clusters']),
                                    'dev_freeze': r['dev_freeze'], 'reason': 'already_selected_K'})
                continue
            out = build_variant_year(v, y, force=True, k_override=int(r['K_selected']),
                                     k_mode=str(r['mode_selected']))
            p = load_json_or(progress_path(v), {'variants': {}})
            p.setdefault('variants', {}).setdefault(v, {})[str(y)] = out
            p['updated_at'] = datetime.now(timezone.utc).isoformat()
            write_json(progress_path(v), p)
            res['rebuilt'].append({'variant': v, 'text_year': y, 'K_old': int(rep['k_clusters']),
                                   'K_new': out['K'], 'dev_selected': r['dev_selected'],
                                   'mode_selected': r['mode_selected'], 'target': out['target'],
                                   'peer_median': out['peer_median'], 'n_edges': out['n_edges']})
    res['elapsed_sec'] = round(time.time() - t0, 1)
    log(f'rebuild：重建 {len(res["rebuilt"])} 个 (变体,年)；保留 {len(res["kept"])}；'
        f'{res["elapsed_sec"]}s')
    return res

def dk(v: str) -> bool:
    """变体是否使用逐年目标（报告用分类）。"""
    return VARIANTS[v]['target_kind'] in ('asof_per_year', 'latest_per_year')

def density(n_edges: int, n_nodes: int) -> float:
    return round(n_edges / max(1, n_nodes * (n_nodes - 1)), 8)

def edge_keys(df: pd.DataFrame, code_map: dict) -> np.ndarray:
    """(focal_id, source_id) → 单一 int64 键（用于集合运算；自环/重复单独记账）。"""
    n = len(code_map)
    f = df['focal_id'].map(code_map)
    s = df['source_id'].map(code_map)
    require(bool(f.notna().all() and s.notna().all()), '边表含未登记 firm_id（code_map 缺键）')
    return (f.to_numpy(dtype=np.int64) * n + s.to_numpy(dtype=np.int64))

def jaccard_from_keys(a: np.ndarray, b: np.ndarray) -> dict:
    ua, ub = np.unique(a), np.unique(b)
    inter = int(np.intersect1d(ua, ub, assume_unique=True).size)
    union = int(ua.size + ub.size - inter)
    return {'n_lhs': int(ua.size), 'n_rhs': int(ub.size), 'n_inter': inter, 'n_union': union,
            'jaccard': round(inter / union, 6) if union else None}

def build_code_map() -> dict:
    """firm_id → 连续整数编码（由 firm_id_map.csv 建，边表未知 id 追加）。"""
    m = pd.read_csv(ROOT / 'data/processed/firm_id_map.csv', dtype=str)
    ids = m['firm_id'].dropna().astype(str).tolist()
    cmap = {f: i for i, f in enumerate(dict.fromkeys(ids))}
    return cmap

def read_rep(net: str, y: int) -> dict:
    p = net_dir(net) / f'{y}_build_report.json'
    require(p.is_file(), f'[{net}/{y}] build_report 缺失：{p}')
    return json.loads(p.read_text(encoding='utf-8'))

def check_edge_contract(net: str, y: int, df: pd.DataFrame) -> dict:
    """边表合同：键唯一、无自环、行内权重和 = 1（≤1e-9）。"""
    dup = int(df.duplicated(['focal_id', 'source_id']).sum())
    required_self = int((df['focal_id'] == df['source_id']).sum())
    ws = df.groupby('focal_id', sort=False)['weight'].sum()
    worst = float(np.max(np.abs(ws.to_numpy() - 1.0))) if len(ws) else 0.0
    require(dup == 0, f'[{net}/{y}] 边表键重复 {dup} 行；停机')
    require(required_self == 0, f'[{net}/{y}] 边表含自环 {required_self} 行；停机')
    require(worst <= 1e-9, f'[{net}/{y}] 边表行内权重和最大偏差 {worst:.3e} > 1e-9；停机')
    return {'n_dup_keys': dup, 'n_self_loops': required_self,
            'max_abs_row_weight_sum_minus_1': worst, 'n_focal_with_edges': int(len(ws))}

def read_support_stats(net: str, y: int) -> dict:
    """support.csv → 节点/支持计数 + peer 分位（`n_edges` = 同团 peer 数，与 `cluster_stats` 同口径）。"""
    sup = pd.read_csv(net_dir(net) / f'{y}_support.csv', dtype={'focal_id': str})
    require(bool((sup['network_version'] == f'{net_prefix(net)}_{y}').all()),
            f'[{net}/{y}] support.csv network_version 与登记不符；停机')
    require(bool((sup['available_at'] == f'{y + 1}-04-30').all()),
            f'[{net}/{y}] support.csv available_at 与规则不符；停机')
    peers = sup['n_edges'].to_numpy(dtype=float)
    n_sup = int((sup['network_support_status'] == 'support').sum())
    stats = {
        'n_rows': int(len(sup)), 'n_support': n_sup, 'n_no_support': int(len(sup) - n_sup),
        'peer_median': float(np.median(peers)), 'peer_p25': float(np.quantile(peers, 0.25)),
        'peer_p90': float(np.quantile(peers, 0.90)), 'peer_mean': round(float(peers.mean()), 1),
        'isolated_share': round(float((peers == 0).mean()), 4),
        'weight_sum_median': round(float(sup['weight_sum'].median()), 8) if 'weight_sum' in sup else None,
    }
    return stats

def net_lift(net: str, y: int, latest_map: dict, asof_map: dict) -> dict:
    clu = pd.read_csv(net_dir(net) / 'clusters' / f'{y}_clusters.csv', dtype={'firm_id': str})
    out = {'lift_latest': p31.cluster_lift(clu, latest_map),
           'lift_asof': p31.cluster_lift(clu, asof_map)}
    return out

def compare_rows(years, ref_nets=REF_NETS, variants=None) -> tuple:
    """变体 × headline × p3asof：边数/Jaccard/密度/peer 中位与 p90/lift 两口径/楔子/K/isolated。"""
    t0 = time.time()
    variants = variants or VARIANT_ORDER
    nets = list(ref_nets) + list(variants)
    m31 = p31.load_sw2_map()
    latest_map = p31.headline_latest_l2_map()
    cmap = build_code_map()
    rows, keys, detail, flags = [], {}, {}, []
    for net in nets:
        keys[net], detail[net] = {}, {}
        for y in years:
            rep = read_rep(net, y)
            exp_ver = f'{net_prefix(net)}_{y}'
            require(rep['network_version'] == exp_ver,
                    f'[{net}/{y}] network_version={rep["network_version"]!r} != {exp_ver!r}；停机')
            require(rep['available_at'] == f'{y + 1}-04-30',
                    f'[{net}/{y}] available_at={rep["available_at"]!r}；停机')
            edf = pd.read_csv(net_dir(net) / f'{y}_similarity.csv', usecols=EDGE_COLS,
                              dtype={'focal_id': str, 'source_id': str})
            contract = check_edge_contract(net, y, edf)
            require(int(rep['n_edges']) == len(edf),
                    f'[{net}/{y}] build_report n_edges={rep["n_edges"]} != 边表行数 {len(edf)}；停机')
            keys[net][y] = edge_keys(edf, cmap)
            asof_map = p31.l2_map_for_key(m31, f'{y + 1}-04-30')
            lift = net_lift(net, y, latest_map, asof_map)
            sup = read_support_stats(net, y)
            require(sup['n_rows'] == int(rep['n_nodes']),
                    f'[{net}/{y}] support 行数 {sup["n_rows"]} != n_nodes {rep["n_nodes"]}；停机')
            # 两套 peer 读数必须一致：support['n_edges']（同团 peer 数）vs clusters 团规模−1
            require(abs(sup['peer_median'] - lift['lift_latest']['peer_median']) <= 1e-9,
                    f'[{net}/{y}] support peer 中位 {sup["peer_median"]} != clusters 口径 '
                    f'{lift["lift_latest"]["peer_median"]}；停机')
            require(sup['n_support'] == int(rep['n_support']) and sup['n_no_support'] == int(rep['n_no_support']),
                    f'[{net}/{y}] support 计数与 build_report 不符；停机')
            detail[net][y] = {'rep': rep, 'contract': contract, 'lift': lift, 'support': sup,
                              'asof_map_n': len(asof_map)}
            del edf
    # headline 登记读数复核（只读）
    for y in years:
        h = detail['headline'][y]['rep']
        require(int(h['n_edges']) == HEADLINE_EDGES[y] and int(h['k_clusters']) == HEADLINE_K[y]
                and int(h['n_nodes']) == HEADLINE_NODES[y],
                f'[{y}] headline 登记读数不符（edges {h["n_edges"]}/{HEADLINE_EDGES[y]}、'
                f'K {h["k_clusters"]}/{HEADLINE_K[y]}、nodes {h["n_nodes"]}/{HEADLINE_NODES[y]}）；停机')
    for net in nets:
        for y in years:
            rep = detail[net][y]['rep']
            sup = detail[net][y]['support']
            if net in VARIANTS:
                target = target_for(net, y)
                kd_kind = VARIANTS[net]['target_kind']
            elif net == 'headline':
                target, kd_kind = K_HEADLINE_POOLED, 'pooled_44_latest（headline 登记目标）'
            else:
                target, kd_kind = K_ASOF_POOLED, 'pooled_49_asof（3.1 N1）'
            ref = detail['headline'][y]['rep']
            jc_h = jaccard_from_keys(keys[net][y], keys['headline'][y])
            jc_a = jaccard_from_keys(keys[net][y], keys['p3asof'][y])
            ll, la = detail[net][y]['lift']['lift_latest'], detail[net][y]['lift']['lift_asof']
            peer = float(ll['peer_median'])
            spec = VARIANTS.get(net) or {}
            row = {
                'network': net, 'text_year': y, 'network_year': y + 1,
                'isolated_effect': spec.get('isolated_effect', '参照网络（只读，不属 3.2 变体）'),
                'l2_mapping': spec.get('l2_mapping', 'latest_static'),
                'lexicon': spec.get('lexicon_desc', 'headline 词表（只读参照）'),
                'k_target_kind': kd_kind,
                'k_target': target, 'network_version': rep['network_version'],
                'available_at': rep['available_at'],
                'K': int(rep['k_clusters']), 'n_nodes': int(rep['n_nodes']),
                'n_edges': int(rep['n_edges']), 'n_support': int(rep['n_support']),
                'n_no_support': int(rep['n_no_support']),
                'edge_delta_pct_vs_headline': round(100.0 * (int(rep['n_edges']) / int(ref['n_edges']) - 1), 3),
                'density': density(int(rep['n_edges']), int(rep['n_nodes'])),
                'peer_median': peer, 'peer_p25': sup['peer_p25'], 'peer_p90': sup['peer_p90'],
                'peer_mean': sup['peer_mean'],
                'isolated_share': float(ll['isolated_share']),
                'jaccard_vs_headline': jc_h['jaccard'], 'n_overlap_edges_headline': jc_h['n_inter'],
                'jaccard_vs_p3asof': jc_a['jaccard'], 'n_overlap_edges_p3asof': jc_a['n_inter'],
                'lift_latest': ll['sw2_lift'], 'lift_asof': la['sw2_lift'],
                'same_sw2_edge_share_latest': ll['same_sw2_edge_share'],
                'same_sw2_edge_share_asof': la['same_sw2_edge_share'],
                'base_same_sw2_rate_latest': ll['base_same_sw2_rate'],
                'base_same_sw2_rate_asof': la['base_same_sw2_rate'],
                'asof_ind_median_peer_y': K_ASOF_PER_YEAR[y],
                'latest_ind_median_peer_y': K_LATEST_PER_YEAR[y],
                'wedge_vs_asof_ind': round(peer / K_ASOF_PER_YEAR[y], 4),
                'wedge_vs_latest_ind': round(peer / K_LATEST_PER_YEAR[y], 4),
                'peer_minus_target': round(peer - target, 2),
                'n_clusters_lift': int(ll['n_clusters']), 'size_max_lift': int(ll['size_max']),
                'n_dup_keys': detail[net][y]['contract']['n_dup_keys'],
                'n_self_loops': detail[net][y]['contract']['n_self_loops'],
                'max_abs_row_weight_sum_minus_1': detail[net][y]['contract']['max_abs_row_weight_sum_minus_1'],
            }
            rows.append(row)
            # 验收旗标（软）与停机（硬）
            f = []
            if net in VARIANTS:
                if not (20 <= row['K'] <= 4000):
                    f.append('K_out_of_range_20_4000')
                if abs(peer - target) > 2:
                    f.append('median_peer_off_target_gt2')
                if abs(row['edge_delta_pct_vs_headline']) > 60:
                    f.append('edge_delta_over_60pct')
                if row['lift_asof'] <= 1 or row['lift_latest'] <= 1:
                    require(False, f'[{net}/{y}] lift ≤ 1（行业信号崩塌）；停机')
                if row['isolated_share'] > 0.15:
                    f.append('isolated_share_over_0.15')
                if row['n_dup_keys'] or row['n_self_loops'] or \
                        row['max_abs_row_weight_sum_minus_1'] > 1e-9:
                    require(False, f'[{net}/{y}] 边表合同失败；停机')
                # freeze 内部 lift（静态 latest 标签）与现算 lift_latest 一致性
                fz = ART / f'3.2_freeze_{net}_{y}.json'
                if fz.is_file():
                    fv = json.loads(fz.read_text(encoding='utf-8'))['years'][str(y)]
                    if abs(float(fv['sw2_lift']) - row['lift_latest']) > 0.01:
                        f.append('freeze_lift_vs_recomputed_mismatch')
            flags.append({'network': net, 'text_year': y, 'flags': f})
    log(f'compare：{len(rows)} 行（{len(nets)} 网络 × {len(years)} 年），{round(time.time() - t0, 1)}s')
    return rows, keys, detail, flags

def effect_decomposition(rows) -> dict:
    """词表 / 目标 / 目标映射效应分解（集合量不可加 → 报 Δ% 与 Jaccard，并给可加残差参考）。"""
    idx = {(r['network'], r['text_year']): r for r in rows}
    years = sorted({r['text_year'] for r in rows})

    def pair(lhs, rhs, label, axis):
        tot_l = sum(int(idx[(lhs, y)]['n_edges']) for y in years)
        tot_r = sum(int(idx[(rhs, y)]['n_edges']) for y in years)
        per_year = []
        for y in years:
            a, b = idx[(lhs, y)], idx[(rhs, y)]
            per_year.append({
                'text_year': y, 'K_lhs': a['K'], 'K_rhs': b['K'],
                'edges_lhs': a['n_edges'], 'edges_rhs': b['n_edges'],
                'edge_delta_pct': round(100.0 * (a['n_edges'] / b['n_edges'] - 1), 3),
                'delta_vs_headline_pct_lhs': a['edge_delta_pct_vs_headline'],
                'delta_vs_headline_pct_rhs': b['edge_delta_pct_vs_headline'],
                'peer_median_lhs': a['peer_median'], 'peer_median_rhs': b['peer_median'],
                'lift_asof_lhs': a['lift_asof'], 'lift_asof_rhs': b['lift_asof'],
                'lift_latest_lhs': a['lift_latest'], 'lift_latest_rhs': b['lift_latest'],
                'target_lhs': a['k_target'], 'target_rhs': b['k_target'],
            })
        return {'axis': axis, 'label': label, 'lhs': lhs, 'rhs': rhs,
                'edges_total_lhs': tot_l, 'edges_total_rhs': tot_r,
                'edge_delta_pct_total': round(100.0 * (tot_l / max(1, tot_r) - 1), 3),
                'per_year': per_year,
                'median_peer_median_lhs': float(np.median([p['peer_median_lhs'] for p in per_year])),
                'median_peer_median_rhs': float(np.median([p['peer_median_rhs'] for p in per_year]))}

    df_tot = pair('p3ydf', 'headline', 'L-6 词表 df（逐年 vs 9 年上确界；同目标 44）', 'lexicon_df')
    ds_tot = pair('p3yt', 'headline', 'L-2 K 目标（逐年 as-of vs 池化 44）', 'k_target_per_year_asof')
    lat_tot = pair('p3yt_latest', 'headline', 'K 目标（逐年 latest vs 池化 44）', 'k_target_per_year_latest')
    comb = pair('p3yt_ydf', 'headline', '组合（逐年 as-of 目标 + 逐年 df）', 'combined')
    lex_at_yt = pair('p3yt_ydf', 'p3yt', '词表效应（在逐年 as-of 目标下）', 'lexicon_df_at_per_year_target')
    tgt_at_ydf = pair('p3yt_ydf', 'p3ydf', '目标效应（在逐年 df 词表下）', 'k_target_effect_at_yearly_df')
    map_tot = pair('p3yt_latest', 'p3yt', '目标映射（latest vs as-of 逐年目标；同词表）', 'target_mapping')
    pooled = pair('p3yt', 'p3asof', '池化 vs 逐年（as-of：49 池化 vs 逐年 38…64）', 'pooled_vs_per_year')
    return {'note': ('效应为集合量（边集/邻域），**不可加**：组合 ≠ 词表 + 目标；'
                     '本节报 Δ% 与 Jaccard（`jaccard_cache`），附「可加残差」参考（单位 = 百分点）'),
            'pairs': {'lexicon_df': df_tot, 'k_target_per_year_asof': ds_tot,
                      'k_target_per_year_latest': lat_tot, 'combined': comb,
                      'lexicon_df_at_per_year_target': lex_at_yt, 'k_target_effect_at_yearly_df': tgt_at_ydf,
                      'target_mapping': map_tot, 'pooled_vs_per_year': pooled},
            'additivity_residual_pp': {
                'combined_minus_df_minus_target': round(comb['edge_delta_pct_total']
                                                        - df_tot['edge_delta_pct_total']
                                                        - ds_tot['edge_delta_pct_total'], 3),
                'interpretation': '交互项（词表 × 目标）；非零即证明两轴不可加'},
            'label_axis_note': ('L2 映射不进入网络构造（build_vectors/聚类/边表只吃 语料 × 词表 × K 目标）；'
                                '标签口径只影响 (a) K 目标的来源（as-of vs latest，见 target_mapping 项）、'
                                '(b) lift 分式的行业标签（本表 lift_asof / lift_latest 两列并报）。'
                                '因此不存在「标签进入网络」的构造通道：标签轴对网络的构造性影响 = 0。'),
            'wedge_note': ('逐年目标变体（p3yt / p3yt_ydf）的 realized median peer ≈ 该年 as-of 行业中位 peer '
                           '→ wedge_vs_asof_ind ≈ 1.0 **构造性成立**（不是经验发现）；'
                           'p3yt_latest 的 wedge 对照 latest 行业中位 peer 亦 ≈ 1.0。'
                           'headline 的 wedge（分子 = headline 网络）才是 1.69→0.71（latest）/ 1.16→0.69（as-of）'
                           '口径。')}

def variant_pair_jaccard(keys, years, pairs) -> dict:
    out = {}
    for lhs, rhs, label in pairs:
        per_year = []
        for y in years:
            jc = jaccard_from_keys(keys[lhs][y], keys[rhs][y])
            per_year.append({'text_year': y, **jc})
        out[f'{lhs}_vs_{rhs}'] = {'label': label, 'per_year': per_year,
                                  'median_jaccard': float(np.median([p['jaccard'] for p in per_year]))}
    return out

def freeze_summary_rows(years, variants) -> list:
    rows = []
    for v in variants:
        for y in years:
            f = ART / f'3.2_freeze_{v}_{y}.json'
            require(f.is_file(), f'[{v}/{y}] freeze 报告缺失：{f}')
            d = json.loads(f.read_text(encoding='utf-8'))
            st = d['years'][str(y)]
            fs = st['filter_stats']
            rows.append({
                'variant': v, 'text_year': y, 'target': d['target_peer'], 'K': st['K'],
                'K_mode': st['K_mode'], 'peer_median': st['peer_median'], 'peer_p25': st['peer_p25'],
                'peer_p90': st['peer_p90'], 'peer_mean': st['peer_mean'],
                'isolated_share': st['isolated_share'], 'n_clusters': st['n_clusters'],
                'size_median': st['size_median'], 'size_max': st['size_max'],
                'base_same_sw2_rate': st['base_same_sw2_rate'], 'sw2_lift_latest_internal': st['sw2_lift'],
                'docs': fs['docs'], 'drop_menlei': fs['drop_menlei'], 'drop_ipo': fs['drop_ipo'],
                'drop_min': fs['drop_min'], 'firms_kept': fs['firms_kept'],
                'active_vocab': fs['active_vocab'], 'mean_terms': fs['mean_terms'],
                'lexicon_terms_kept': d['lexicon']['terms_kept'],
                'lexicon_path': d['lexicon']['path'], 'sec': st['sec'],
            })
    return rows

def version_spec_check(variants, years) -> dict:
    """`load_versions(version_spec)` 注入装载验证（shadow 目录不登记 manifest，3.6 依赖此路径）。"""
    out = {}
    for v in variants:
        spec = {'version_prefix': VARIANTS[v]['ver_prefix'], 'years': [int(y) for y in years],
                'available_at_rule': NETWORK_AVAILABLE_AT_RULE}
        vers = load_versions(VARIANTS[v]['dir'], version_spec=spec)
        probe = ['2015-01', '2016-06', '2021-06', '2024-01']
        out[v] = {
            'n_years': len(vers), 'version_spec_used': spec, 'not_in_manifest': True,
            'selected_text_year_by_month': {m: (select_version(vers, m) or {}).get('text_year')
                                            for m in probe},
            'all_available_at': sorted({x['available_at'] for x in vers.values()}),
            'version_strings': sorted({f'{VARIANTS[v]["ver_prefix"]}_{y}' for y in vers}),
        }
        require(len(vers) == len(years), f'[{v}] load_versions(version_spec) 年份数 {len(vers)} != {len(years)}')
    return out

def networks_summary_from_disk(years, variants) -> dict:
    """从磁盘重建 networks 段（并行构建时 4 个进程各写 counts → 以磁盘为准，避免相互覆盖）。"""
    out = {'source': ('rebuilt_from_disk（*_build_report.json + 3.2_freeze_*.json）；'
                      '4 变体并行构建时各进程的 counts 段会互相覆盖，故此处以磁盘逐值重建'),
           'variants': {}}
    for v in variants:
        ys = {}
        for y in years:
            rep = read_rep(v, y)
            fz = json.loads((ART / f'3.2_freeze_{v}_{y}.json').read_text(encoding='utf-8'))
            fy = fz['years'][str(y)]
            ys[str(y)] = {
                'text_year': y, 'network_year': y + 1, 'target': int(fz['target_peer']),
                'K': int(rep['k_clusters']), 'K_mode': fy['K_mode'],
                'peer_median': float(fy['peer_median']), 'n_edges': int(rep['n_edges']),
                'n_nodes': int(rep['n_nodes']), 'n_support': int(rep['n_support']),
                'n_no_support': int(rep['n_no_support']),
                'network_version': rep['network_version'], 'available_at': rep['available_at'],
                'lexicon_terms_kept': int(rep['lexicon']['terms_kept']),
                'lexicon_path': rep['lexicon']['path'], 'elapsed_sec_edges': rep.get('elapsed_sec'),
                'freeze_report': str((ART / f'3.2_freeze_{v}_{y}.json').relative_to(ROOT)).replace('\\', '/'),
            }
        out['variants'][v] = {'ver_prefix': VARIANTS[v]['ver_prefix'], 'dir':
                              str(VARIANTS[v]['dir'].relative_to(ROOT)).replace('\\', '/'),
                              'years': ys, 'n_years': len(ys)}
    out['version_spec_check'] = version_spec_check(variants, years)
    return out

def step_compare(years, variants=None) -> dict:
    t0 = time.time()
    variants = variants or VARIANT_ORDER
    rows, keys, detail, flags = compare_rows(years, variants=variants)
    header = list(rows[0].keys())
    write_csv(COMPARISON_CSV, [[r[k] for k in header] for r in rows], header)
    # 楔子表（分子口径显式）
    wedge_rows = []
    for r in rows:
        wedge_rows.append({
            'network': r['network'], 'text_year': r['text_year'], 'network_year': r['network_year'],
            'network_peer_median': r['peer_median'], 'K': r['K'], 'k_target': r['k_target'],
            'asof_ind_median_peer_y': r['asof_ind_median_peer_y'],
            'latest_ind_median_peer_y': r['latest_ind_median_peer_y'],
            'wedge_vs_asof_ind': r['wedge_vs_asof_ind'], 'wedge_vs_latest_ind': r['wedge_vs_latest_ind'],
            'numerator_rule': '本网络同团成员法 realized median peer（团规模−1 的公司层面中位）',
            'denominator_rule': '同年申万二级同行业中位 peer（as-of 分类生效日口径 | 静态 latest 对照）',
            'note': ('逐年目标变体 wedge≈1.0 为构造性（目标即该口径行业中位 peer）；'
                     'headline/p3asof 的 wedge 才是密度楔子读数'),
        })
    write_csv(WEDGE_CSV, [list(r.values()) for r in wedge_rows], list(wedge_rows[0].keys()))
    fs_rows = freeze_summary_rows(years, variants)
    write_csv(FREEZE_SUMMARY_CSV, [list(r.values()) for r in fs_rows], list(fs_rows[0].keys()))
    eff = effect_decomposition(rows)
    eff['jaccard_cache'] = variant_pair_jaccard(keys, years, [
        ('p3yt', 'headline', 'K 目标逐年 as-of vs headline（同词表）'),
        ('p3ydf', 'headline', '词表 df 逐年 vs headline（同目标 44）'),
        ('p3yt_ydf', 'headline', '组合 vs headline'),
        ('p3yt_latest', 'headline', 'K 目标逐年 latest vs headline（同词表）'),
        ('p3yt_ydf', 'p3yt', '词表效应（逐年目标下）'),
        ('p3yt_ydf', 'p3ydf', '目标效应（逐年词表下）'),
        ('p3yt_latest', 'p3yt', '目标映射 latest vs as-of（同词表）'),
        ('p3yt', 'p3asof', '池化 vs 逐年（as-of）'),
    ])
    res = {'rows': rows, 'flags': flags, 'effect_decomposition': eff,
           'comparison_csv': str(COMPARISON_CSV.relative_to(ROOT)).replace('\\', '/'),
           'wedge_csv': str(WEDGE_CSV.relative_to(ROOT)).replace('\\', '/'),
           'freeze_summary_csv': str(FREEZE_SUMMARY_CSV.relative_to(ROOT)).replace('\\', '/'),
           'n_rows': len(rows), 'elapsed_sec': round(time.time() - t0, 1)}
    log(f'compare：对比表 {COMPARISON_CSV.name}（{len(rows)} 行）、楔子表 {WEDGE_CSV.name}、'
        f'freeze 汇总 {FREEZE_SUMMARY_CSV.name}（{res["elapsed_sec"]}s）')
    return res, keys, detail

def step_sha_ledger(years, variants) -> dict:
    """确定性产物 sha 台账（重跑逐字节一致证据；freeze/build_report 含 run_at 不参与）。"""
    files = []
    for v in variants:
        d = VARIANTS[v]['dir']
        files += sorted(d.glob('*_similarity.csv')) + sorted(d.glob('*_support.csv')) \
            + sorted((d / 'clusters').glob('*.csv'))
    files += sorted(SHADOW_DIR.glob('lexicon_p3ydf_*.csv'))
    files += [COMPARISON_CSV, WEDGE_CSV, FREEZE_SUMMARY_CSV, LEXICON_BY_YEAR_CSV, DF_REMOVED_CSV]
    files = [f for f in files if Path(f).is_file()]
    cur = {Path(f).relative_to(ROOT).as_posix(): file_size(f) for f in files}
    res = {'n_files': len(cur), 'sha': cur,
           'note': 'freeze 报告与 build_report 含 run_at/elapsed_sec，不参与逐字节比对'}
    if OUTPUT_SHA.is_file():
        prev = json.loads(OUTPUT_SHA.read_text(encoding='utf-8'))
        ps = prev.get('sha') or {}
        diff = sorted(k for k in set(ps) | set(cur) if ps.get(k) != cur.get(k))
        res['rerun_check'] = {'status': 'IDENTICAL' if not diff else 'DIFFERENT',
                              'n_compared': len(set(ps) & set(cur)), 'n_diff': len(diff),
                              'diff_keys': diff[:50]}
    write_json(OUTPUT_SHA, res)
    log(f'sha 台账：{res["n_files"]} 个确定性文件' + (
        f'；重跑比对 {res["rerun_check"]["status"]}（diff={res["rerun_check"]["n_diff"]}）'
        if 'rerun_check' in res else '（首次运行）'))
    return res

def step_preflight() -> dict:
    snap = anchors_snapshot('pre')
    write_json(ANCHOR_PRE, snap)
    log(f'preflight：9 锚 + {snap["headline_network_n_files"]} 个 headline 网络文件 sha 已记录')
    return snap

def step_postflight(counts) -> dict:
    post = anchors_snapshot('post')
    write_json(ANCHOR_POST, post)
    pre = load_json_or(ANCHOR_PRE, None)
    cmp_ = p31.compare_anchors(pre, post) if pre else {'status': 'NO_PRE', 'n_diff': None}
    log(f'postflight：锚对比 {cmp_["status"]}（diff={cmp_["n_diff"]}）')
    if cmp_['status'] != 'MATCH':
        print('[HARD-GATE] headline/输入锚发生变化；停机（禁止重算覆盖）', file=sys.stderr, flush=True)
        raise SystemExit(2)
    return {'anchor_compare': cmp_, 'headline_network_n_files': post['headline_network_n_files'],
            'anchor_pre': str(ANCHOR_PRE.relative_to(ROOT)).replace('\\', '/'),
            'anchor_post': str(ANCHOR_POST.relative_to(ROOT)).replace('\\', '/')}


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --- scripts/lib bootstrap ---
# --- end scripts/lib bootstrap ---


# ---------------------------------------------------------------- 路径与常量


# as-of 交接的 K 目标（`3.1_k_targets.json` 逐值复核；容差 0）
# headline 登记读数（只读对账；`3.0_plan.md` §1.1 / 3.1 §5）
# 逐年 df 词表落盘（plan §2.2 R1 登记值：2014 多剔 560 词 = 0.66% 命中词 / 7.26% 命中质量；
# 2015 = 819 词 / 0.99% / 12.08%）
# 3.1 扩展阶梯窗口（±K）

# 变体登记（唯一真源）

# 只读输入（sha 台账；headline 词表 / df_by_year / sw2 映射不得被写）


# ---------------------------------------------------------------- 输出路径守卫


# ---------------------------------------------------------------- 锚台账（复用 3.1 语义）


# ---------------------------------------------------------------- 变体登记 / 目标


# ---------------------------------------------------------------- 逐年词表（L-6）


# ---------------------------------------------------------------- 网络构建（复用现脚本）


# ---------------------------------------------------------------- 扩展 K 阶梯（目标可达性 + 二级 K 选择）


# ---------------------------------------------------------------- 读数 / 对比 / 验收


# ---------------------------------------------------------------- steps 入口


def main():
    ap = argparse.ArgumentParser(description='Phase 3.2 L-2/L-6 校准 shadow 变体（逐年 target / 逐年 df）')
    ap.add_argument('--steps', default='all',
                    help='all 或逗号分隔子集：preflight,lexicons,networks,ladder,rebuild,compare,postflight')
    ap.add_argument('--variants', nargs='*', default=None, choices=VARIANT_ORDER,
                    help='缺省全 4 个变体')
    ap.add_argument('--years', type=int, nargs='*', default=None, help='文本年（缺省全 9 年）')
    ap.add_argument('--ladder-lexicon', nargs='*', default=['headline', 'per_year'],
                    choices=['headline', 'per_year'], help='扩展阶梯所用词表（缺省两套）')
    ap.add_argument('--force', action='store_true', help='networks：强制重建已完成年')
    ap.add_argument('--force-lexicons', action='store_true', help='lexicons：强制重写词表文件')
    args = ap.parse_args()
    steps = (['preflight', 'lexicons', 'networks', 'ladder', 'select', 'rebuild', 'compare', 'postflight']
             if args.steps == 'all' else [s.strip() for s in args.steps.split(',') if s.strip()])
    years = args.years if args.years else YEARS
    variants = args.variants if args.variants else VARIANT_ORDER
    t_all = time.time()
    SHADOW_DIR.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    for v in VARIANT_ORDER:
        VARIANTS[v]['dir'].mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()
    counts = {'run_at': now, 'steps': steps, 'years': years, 'variants': variants,
              'shadow_version_tag': 'shadow_window_variants_v1',
              'run_history': []}
    prev = load_json_or(COUNTS_JSON, None)
    if isinstance(prev, dict):
        hist = prev.get('run_history') if isinstance(prev.get('run_history'), list) else []
        hist = hist + [{'run_at': prev.get('run_at'), 'steps': prev.get('steps'),
                        'elapsed_sec_total': prev.get('elapsed_sec_total')}]
        prev = {k: v for k, v in prev.items() if k != 'run_history'}
        prev.update({'run_at': now, 'steps': steps, 'years': years, 'variants': variants,
                     'run_history': hist})
        counts = prev
    counts['variant_registry'] = {v: {k: (str(x.relative_to(ROOT)).replace('\\', '/')
                                          if isinstance(x, Path) else x)
                                      for k, x in VARIANTS[v].items()} for v in VARIANT_ORDER}
    counts['k_targets_3_1'] = {'asof_per_year': K_ASOF_PER_YEAR, 'latest_per_year': K_LATEST_PER_YEAR,
                               'asof_pooled': K_ASOF_POOLED, 'latest_pooled_headline': K_HEADLINE_POOLED,
                               'source': str(K_TARGETS_3_1.relative_to(ROOT)).replace('\\', '/')}

    if 'preflight' in steps:
        counts['anchor_pre'] = step_preflight()
    if 'lexicons' in steps:
        counts['lexicons'] = build_yearly_lexicons(years, force=args.force_lexicons)
    if 'networks' in steps or 'compare' in steps:
        tgt3_1 = load_k_targets_3_1()
        counts['k_targets_3_1']['verified_against_3_1_json'] = True
        counts['k_targets_3_1']['3_1_targets'] = {
            'asof': {str(p['text_year']): p['median_peer'] for p in tgt3_1['asof']['per_year']},
            'latest': {str(p['text_year']): p['median_peer'] for p in tgt3_1['latest']['per_year']},
        }
    if 'networks' in steps:
        counts['networks'] = step_networks(variants, years, force=args.force)
    if 'ladder' in steps:
        lad = counts.get('k_ladder') if isinstance(counts.get('k_ladder'), dict) else {}
        for m in args.ladder_lexicon:
            lad[m] = step_ladder([int(y) for y in years], m)
        counts['k_ladder'] = lad
    if 'select' in steps:
        sel_res = step_select([int(y) for y in years], args.ladder_lexicon)
        counts['k_ladder'] = {**(counts.get('k_ladder') or {}), **sel_res['k_ladder']}
        counts['k_selection'] = sel_res['k_selection']
        counts['k_selection_summary'] = {
            'csv': sel_res['k_selection_csv'], 'n_pairs': len(sel_res['k_selection']),
            'n_needs_rebuild': sel_res['n_needs_rebuild'],
            'n_dev_over_tolerance': sum(1 for x in sel_res['k_selection']
                                        if float(x['dev_selected']) > K_TOLERANCE),
            'tolerance': K_TOLERANCE,
            'note': ('|dev| > tolerance 的 (变体,年) 均为「目标在扩展阶梯窗口内不可达」'
                     '（跳点/平台）；`target_reachable_in_window=False` 与 '
                     '`best_attainable_dev_in_window` 逐条留档'),
        }
    if 'rebuild' in steps:
        sel_rows = counts.get('k_selection') or []
        if not sel_rows and K_SELECTION_CSV.is_file():
            sel_rows = pd.read_csv(K_SELECTION_CSV).to_dict('records')
        require(bool(sel_rows), 'rebuild：缺 K 选择表（先跑 --steps select）')
        counts['rebuild'] = step_rebuild([int(y) for y in years], variants, sel_rows)
    if 'compare' in steps:
        res, keys, detail = step_compare([int(y) for y in years], variants=variants)
        counts['comparison_rows'] = res['rows']
        counts['networks_from_disk'] = networks_summary_from_disk([int(y) for y in years], variants)
        counts['acceptance_flags'] = res['flags']
        counts['effect_decomposition'] = res['effect_decomposition']
        counts['compare_meta'] = {'comparison_csv': res['comparison_csv'],
                                  'wedge_csv': res['wedge_csv'],
                                  'freeze_summary_csv': res['freeze_summary_csv'],
                                  'n_rows': res['n_rows'], 'elapsed_sec': res['elapsed_sec']}
        counts['flag_counts'] = {}
        for f in res['flags']:
            for x in f['flags']:
                counts['flag_counts'][x] = counts['flag_counts'].get(x, 0) + 1
        counts['output_sha_rerun'] = step_sha_ledger([int(y) for y in years], variants)
    if 'postflight' in steps:
        counts['postflight'] = step_postflight(counts)

    counts['elapsed_sec_total'] = round(time.time() - t_all, 1)
    counts['run_at'] = now
    write_json(COUNTS_JSON, counts)
    log(f'完成：总耗时 {counts["elapsed_sec_total"]}s → {COUNTS_JSON}')


if __name__ == '__main__':
    main()
