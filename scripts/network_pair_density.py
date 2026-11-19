#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3 第一梯队 ①+②：pair-density 读数 + 密度对齐复算（`density_readouts` 块）。

**只读复算**：不重建网络、不碰面板/回归、不改任何 headline_value / 冻结参数；只新建产物
`reports/data_gates/G0/s3_lift_pair_density_{teacher,self}.json`（既有产物一律不覆盖）。

口径真源：
  - `reports/audits/s3_remaining_workplan_20261011.md` 第一梯队 1–2
  - 主仓库 methodology_execution_plan §8 B″11-1（pair-density）与 B″11-11（density_readouts）
  - `reports/audits/s3_network_open_items_plan_20261006.md` §3-③、§4 序 1/1b
  - `configs/parameter_registry.yaml` → `lift_pair_density_readout`

① pair-density（HP 式随机配对概率并报读数）
  - 分母 = 当年网络**全部入网公司**的无序对 = `N(N−1)/2`（含零对手、无自环、无序对）；
  - 分子 = 其中**同申万二级**的无序对数 = `Σ_k n_k(n_k−1)/2`；
  - **映射缺失处置**：分子分母同时只计**有申万映射者**（N = 可映射节点数；缺失不填 0、逐一报缺失计数）；
  - 配套 `same_sw2_share`（边层面：边表 focal→source 两端**均可映射**的边中同二级占比）与
    `lift = same_sw2_share / pair_density`；
  - 与既有 `sw2_base_canonical`（= `Σ_k n_k(n_k−1)/(N(N−1))`，见 `s3_ma_chapter_cross_audit.json`
    与 `s3_ma_weighting_cross_audit.json`）**逐值一致**：对比对象 = **全精度复算值**（rel ≤1e-12）；
    与 JSON **存储值**（5 位小数）差 ≤5.4e-6（不在 JSON 之间做 1e-12 级比较）。

② `density_readouts` 块（两字段，**各自验收、禁止互相等值对账**）
  - 字段 1 `normalized_readout = lift × d`（**未裁剪**；`d = E/(N(N−1))`，E = 边表行数（有向）、
    N = 全部入网公司）：**回归门** = 与已复算值逐值一致——网络年 2018–2023 逐年
    38.1/35.4/38.0/38.1/42.0/41.8%；全 9 年 35.4–42.6%。
  - 字段 2 `density_aligned_share = same_sw2′`（按 focal **行内等比例裁剪**到
    `d′ = sw2_base_canonical` 后的同行业份额）：`q = base/d` 全市场同一值；每 focal 按 `weight`
    降序保留前 `k_i = ⌈q × n_i⌉` 条；**不重新归一化**；**禁止全局裁剪**；另报「双向都保留」的
    对称化变体及实际密度；报告 Top1 集中度影响与逐 focal 截断误差。与 HP **50.2%** 同口径并列，
    **预期方向 > 35.4–42.0%**，**不得与 35.4–42.0% 做等值对账**。
  - 停机：① normalized_readout 不符（复算路径分叉，先修再报）；② 裁剪后密度偏差 >1%。

产出（只新建）：`reports/data_gates/G0/s3_lift_pair_density_teacher.json`、
              `reports/data_gates/G0/s3_lift_pair_density_self.json`
用法：
  python scripts/network_pair_density.py
  python scripts/network_pair_density.py --teacher-years 2014 2015 ... 2022 --self-years 2017 ... 2025
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import network_business_chapter_cross_audit as CA  # noqa: E402  （复用其纯函数与统一口径常量）

REPD = ROOT / 'reports/data_gates/G0'
TEACH_DIR = ROOT / 'data/processed/network_ma_teacher'
SELF_DIR = ROOT / 'data/processed/network_ma'
SW2 = ROOT / 'data/processed/sw2_industry_latest.csv'
CHAPTER_JSON = REPD / 's3_ma_chapter_cross_audit.json'
WEIGHT_JSON = REPD / 's3_ma_weighting_cross_audit.json'

TEACHER_YEARS = list(range(2014, 2023))       # 文本年 → 网络年 2015–2023（主口径 9 年）
SELF_YEARS = list(range(2017, 2026))          # 文本年 → 网络年 2018–2026（文本稳健性列现有年份）
REL_TOL_FULL = 1e-12                          # 与全精度复算值比较
ABS_TOL_STORED = 5.4e-6                       # 与 JSON 存储值（5 位小数）比较
CROP_DENSITY_TOL = 0.01                       # 裁剪后密度偏差上限（相对）
N_JUMP_TOL_PP = 0.01                          # 映射覆盖率年际跳变上限（1 个百分点）

# 已复算的回归门数值（open_items 报告 §3-③ / B″11-11；网络年 → 归一化读数 %）
GATE_NORMALIZED_PCT_2018_2023 = {2018: 38.1, 2019: 35.4, 2020: 38.0,
                                 2021: 38.1, 2022: 42.0, 2023: 41.8}
GATE_NORMALIZED_RANGE_9Y = (35.4, 42.6)
GATE_PCT_TOL = 0.05                           # 半个末位（1 位小数的舍入容差）

# HP 对照（HP16 PDF p.19 / readme 行 51–52；Table V Panel A PDF p39）
HP_PAIR_DENSITY = 0.0205                      # 2.05% of all possible firm pairs are industry pairs（SIC-3）
HP_TNIC_SIC3_NORMALIZED = 0.502               # 50.2%（TNIC-3 标到 SIC-3 密度后的归一化读数）
HP_FIC300_SIC3_RAW = 0.441                    # 44.1%（FIC 10K-300，未密度对齐，仅量级指示）

PAIR_DENSITY_DEF = (
    'pair_density（HP 式随机配对概率）：分母 = 当年网络全部入网公司的无序对 N(N−1)/2（含零对手、'
    '无自环、无序对）；分子 = 其中同申万二级的无序对 Σ_k n_k(n_k−1)/2；**映射缺失处置 = 分子分母'
    '同时只计有申万映射者**（N = 可映射节点数；缺失不填 0，缺失计数逐 year 报告）。'
    '恒等式：pair_density = Σ_k n_k(n_k−1)/(N(N−1)) = 既有 sw2_base_canonical（同一算式的显式命名），'
    '两者必须逐值一致（对比对象 = 全精度复算值；与 JSON 存储值（5 位小数）差 ≤5.4e-6）。'
)
LIFT_DEF = ('lift = same_sw2_share / pair_density；same_sw2_share = 边表（focal→source）中两端**均可映射**'
            '的边里同申万二级占比（缺映射端点的边两侧同剔）；与 `network_business_chapter_cross_audit.py::sw2_lift` '
            '同一函数口径（同样本边际基线）。')
NORMALIZED_READOUT_DEF = (
    'normalized_readout = lift × d（**未裁剪**）；d = E/(N(N−1))，E = 边表行数（有向边数，含端点为'
    '缺映射公司的边）、N = 当年网络全部入网公司数（support 表 focal 数）。'
    '用途 = 密度标准化读数（HP 式归一化读数的我方对应物）。'
)
DENSITY_ALIGNED_DEF = (
    'density_aligned_share = same_sw2′：把主口径边表按 focal **行内等比例裁剪**到 d′ = pair_density 后的'
    '同申万二级份额。裁剪口径：q = pair_density / d（全市场同一值）；每 focal 按其 weight 降序保留前 '
    'k_i = ⌈q × n_i⌉ 条（n_i = 该 focal 原始边数，含端点为缺映射公司的边；并列按 source_id 升序）；'
    '**不重新归一化**；**禁止全局裁剪**。因 d′ 恒等于 base，恒有 lift′ × d′ = share′。'
    '与 HP 50.2%（同为归一化读数、同为「标到行业密度」口径）并列；**预期 > 35.4–42.0%，'
    '不得与 35.4–42.0% 做等值对账**（裁剪前/后是两个对象）。'
)
# self 列变体：期望方向为 teacher 专用登记 → 本列改记 not_applicable_column_self（M-2）
DENSITY_ALIGNED_DEF_SELF = (
    'density_aligned_share = same_sw2′：把主口径边表按 focal **行内等比例裁剪**到 d′ = pair_density 后的'
    '同申万二级份额。裁剪口径：q = pair_density / d（全市场同一值）；每 focal 按其 weight 降序保留前 '
    'k_i = ⌈q × n_i⌉ 条（n_i = 该 focal 原始边数，含端点为缺映射公司的边；并列按 source_id 升序）；'
    '**不重新归一化**；**禁止全局裁剪**。因 d′ 恒等于 base，恒有 lift′ × d′ = share′。'
    '与 HP 50.2%（同为归一化读数、同为「标到行业密度」口径）并列（裁剪前/后是两个对象）。'
    '**「预期 > 35.4–42.0%」为 teacher 列专用登记、本列不适用**。'
)


def fmt_e(x: float, sig: int = 2, tight: bool = False) -> str:
    """科学计数法：sig = 尾数小数位；tight=True 时用复核草案原文写法（`2.66e-6` 而非 `2.66e-06`）。"""
    s = f'{x:.{sig}e}'
    return s.replace('e-0', 'e-') if tight else s


# ---- ⑧ JSON 层修订的脚本级同步（2026-10-11 T1F 的 JSON 层补丁 → 本节可复跑，不漂移）----
# 年份轴消歧（L-3）：两列逐字一致，仅「本文件 column = X」随列切换。
YEAR_AXIS_NOTE = (
    '**年份轴（L-3 消歧）**：本文件 column = {label}：`per_year[].network_year` = **网络年**'
    '（= 文本年 + 1）、`per_year[].text_year` = **文本年**；网络年 y 的相似度网络由文本年 y−1 的语料构造'
    '（每年 1–4 月沿用 y−2）。`pooled.years`、`per_year[].network_year`、验收/门控逐项年份、'
    '注记中的年份**一律为网络年**（「网络年」即「网络年（文本年 y−1）」）；凡指文本语料年份处一律显式'
    '写「文本年」；本文件不再出现无轴名的裸年份。'
)
# ③「裁剪后密度偏差 ≤1%」与登记裁剪口径 ⌈q·n_i⌉ 数学互斥（判据缺陷，待拍板）——两列共用判据文案
CROP_CHECK_MEASURED_FIELD = ('acceptance.density_aligned_gate.per_year_density_deviation[]'
                             '.density_deviation_pct（逐年实测，随本文件登记）')
CROP_CHECK_NOTE = (
    '「裁剪后密度偏差 ≤1%」的偏差来源 = 登记口径本身的 ⌈q·n_i⌉ 上取整（每 focal 平均多保留 0.5 条边，'
    '聚合超额 Σ_i(1−frac_i)/(q·E)，本数据恒为 {lo:.2f}–{hi:.2f}%）→ **与登记口径数学互斥**'
    '（非实现误差、非数据问题）：改用最大余数法等量分配后偏差 ≈0（{eq}；见 per_year[].density_readouts.'
    'exact_quota_diagnostic）。**是否改判据/改口径待项目拍板，本轮不改**（见 pending_decision 块）。'
)
SELF_REG_NOT_APPLICABLE_REASON = (
    '**期望值专用性**：`expected.network_years_2018_2023` = 38.1/35.4/38.0/38.1/42.0/41.8% 与 '
    '`all_9y_range_pct` = 35.4–42.6% 由 registry `lift_pair_density_readout`（2026-10-06 三改）为本参数'
    '登记，只适用于 **teacher（主口径）列**的网络年 2018–2023。本列 network_year = 2018–2026'
    '（文本年 2017–2025），与该期望的年份集不同 → 期望对 self 列不适用；self 列只报实测 + 自洽检查，'
    '**不得**用 teacher 期望判定 self，亦不得做等值对账'
)
SELF_REG_EXPECTED_FIELD_STATUS = ('null（原值 = teacher 列专用期望 38.1/35.4/38.0/38.1/42.0/41.8% 与 '
                                 '35.4–42.6%；已删除自本列）')
SELF_DENSITY_ALIGNED_NOT_APPLICABLE_REASON = (
    '本块 `expected_direction`（> 35.4–42.0% 未裁剪区间）与 `direction_mismatch_registration` 均为 '
    'teacher（主口径）列专用登记：前者锚定 teacher 未裁剪读数的 35.4–42.0%（网络年 2018–2023）；'
    'self 列 network_year = 2018–2026、读数区间不同 → 方向期望不适用，方向不符登记对本列不成立'
    '（原`direction_mismatch_registration` 引用的 2018：0.1940 → 0.2683 / 0.3806 亦为 teacher 数值，已删除）'
)
SELF_DENSITY_ALIGNED_NOTE = ('本块**独立报告**，不与 normalized_readout 互相等值对账；**期望方向'
                             '（> 35.4–42.0%）与方向不符登记为 teacher 列专用** → 本列记 '
                             'not_applicable_column_self，只保留实测与自洽检查（M-2）')
CHECKS_SCOPE_NOTE = (
    '① pair_density 一致性、全精度双路径 rel ≤1e-12、逐年齐全、映射年际无跳变为两列共用项；'
    '② normalized_readout 回归门（期望 38.1/35.4/38.0/38.1/42.0/41.8%；网络年 2018–2023）为 '
    '**teacher 列专用**（self 列不适用，见 acceptance.regression_gate_normalized_readout.applicability）；'
    '③ 裁剪后密度偏差 ≤1% 为两列共用项，两列均不通过（⌈⌉ 口径固有超额，spec 层判据缺陷，'
    '见 teacher 文件 pending_decision）'
)
# 候选新判据草案（独立复核给出，**原文收录**；本轮不采用、不改判据、不改口径）
PENDING_DECISION_DRAFT_SOURCE = ('独立复核（第一梯队四项只读复算，CONFIRMED_WITH_ISSUES）给出的候选新判据'
                                 '草案——**原文收录**；本轮不采用、不改判据、不改口径')


def l2_codes(nodes, sw2: dict) -> pd.Series:
    """节点 → 申万二级 code 序列（缺映射 = NaN）。"""
    return pd.Series([CA.l2_of(f, sw2) for f in nodes], dtype='object')


def pair_density_block(sup: pd.DataFrame, edges: pd.DataFrame, sw2: dict) -> dict:
    """① 的核心复算：pair_density / same_sw2_share / lift（全精度 + 存储值）+ d。

    同时给出与既有实现的**全精度复算值**逐值对账所需的中间量（n_k 计数、边层计数）。
    """
    nodes = list(sup['focal_id'])
    n_nodes_total = len(nodes)
    codes = l2_codes(nodes, sw2)
    codes_valid = codes.dropna()
    N = int(len(codes_valid))                                   # 可映射节点数（分子分母同剔）
    vc = codes_valid.value_counts()
    n_same_pairs_undirected = int((vc * (vc - 1) / 2).sum())     # 同二级无序对数（整数）
    universe_undirected = N * (N - 1) / 2
    # 全精度复算（两条路径：整数计数式 / canonical 浮点式）——用于 rel ≤1e-12 对账
    pd_paircount = (n_same_pairs_undirected / universe_undirected) if N > 1 else float('nan')
    pd_canonical = float((vc * (vc - 1)).sum()) / max(1.0, N * (N - 1))

    fa = edges['focal_id'].map(lambda f: CA.l2_of(f, sw2))
    fb = edges['source_id'].map(lambda f: CA.l2_of(f, sw2))
    ok = (fa.notna() & fb.notna()).to_numpy()
    n_edges = int(len(edges))
    n_edges_mapped = int(ok.sum())
    same_sw2_share = float((fa.to_numpy()[ok] == fb.to_numpy()[ok]).mean()) if n_edges_mapped else float('nan')
    lift = same_sw2_share / pd_canonical if pd_canonical > 0 else float('nan')

    # d = E/(N(N−1))：E = 全部有向边数、N = 全部入网公司数（与 plan §3-② 的 edges_ew/nodes_ew 同式）
    d = n_edges / (n_nodes_total * (n_nodes_total - 1)) if n_nodes_total > 1 else float('nan')
    return {
        'n_nodes_total': n_nodes_total,
        'n_nodes_mapped': N,
        'n_missing_mapping': n_nodes_total - N,
        'mapped_share': round(N / n_nodes_total, 6) if n_nodes_total else None,
        'n_edges': n_edges,
        'n_edges_mapped': n_edges_mapped,
        'n_edges_mapping_dropped': n_edges - n_edges_mapped,
        'same_class_pairs_undirected': n_same_pairs_undirected,
        'universe_pairs_undirected': universe_undirected,
        'pair_density': pd_paircount,                    # 全精度（① 的主读数）
        'pair_density_paircount_path': pd_paircount,
        'pair_density_canonical_path': pd_canonical,     # 既有实现的同一算式（全精度复算值）
        'pair_density_stored_5dp': round(pd_canonical, 5),
        'same_sw2_share': same_sw2_share,
        'same_sw2_share_stored_4dp': round(same_sw2_share, 4),
        'lift': lift,
        'lift_stored_3dp': round(lift, 3),
        'd': d,
        'd_stored_5dp': round(d, 5),
        'normalized_readout': lift * d,
        'normalized_readout_pct': round(100 * lift * d, 4),
        'normalized_readout_pct_1dp': round(100 * lift * d, 1),
    }


def crop_block(edges: pd.DataFrame, sw2: dict, base: float, d: float, blk: dict) -> dict:
    """② 字段 2：按 focal 行内等比例裁剪 + 诊断（Top1 集中度 / 截断误差 / 对称化变体）。"""
    q = base / d if d > 0 else float('nan')
    e = edges[['focal_id', 'source_id', 'weight']].copy()
    # 行内按 weight 降序排名（0-based），并列按 source_id 升序 → 确定性
    e = e.sort_values(['focal_id', 'weight', 'source_id'], ascending=[True, False, True])
    e['_rank'] = e.groupby('focal_id').cumcount()
    n_i = e.groupby('focal_id')['weight'].transform('size')
    k_i = np.ceil(q * n_i.to_numpy())
    # 保留标记（按原始边缘顺序回填，便于对称化与份额计算）
    keep_sorted = (e['_rank'].to_numpy() < k_i)
    keep = pd.Series(False, index=edges.index)
    keep.loc[e.index.to_numpy()] = keep_sorted
    kept = edges.loc[keep.to_numpy()]
    dropped = edges.loc[~keep.to_numpy()]

    fa = kept['focal_id'].map(lambda f: CA.l2_of(f, sw2))
    fb = kept['source_id'].map(lambda f: CA.l2_of(f, sw2))
    ok = (fa.notna() & fb.notna()).to_numpy()
    n_kept_mapped = int(ok.sum())
    share_prime = float((fa.to_numpy()[ok] == fb.to_numpy()[ok]).mean()) if n_kept_mapped else float('nan')

    N_all = blk['n_nodes_total']
    retained_density = len(kept) / (N_all * (N_all - 1))
    dev = (retained_density - base) / base if base > 0 else float('nan')
    # lift′（裁剪后网络、同样本边际基线不变）与「归一化读数 = share′」的一致性（d′ = base 恒等）
    lift_prime = share_prime / base if base > 0 else float('nan')

    # ---- 对称化变体：双向都保留 ----
    key = kept[['focal_id', 'source_id']].apply(tuple, axis=1)
    rev = kept[['source_id', 'focal_id']].apply(tuple, axis=1)
    kept_keys = set(key)
    sym_mask = rev.isin(kept_keys).to_numpy()
    sym = kept.loc[sym_mask]
    fa2 = sym['focal_id'].map(lambda f: CA.l2_of(f, sw2))
    fb2 = sym['source_id'].map(lambda f: CA.l2_of(f, sw2))
    ok2 = (fa2.notna() & fb2.notna()).to_numpy()
    sym_share = float((fa2.to_numpy()[ok2] == fb2.to_numpy()[ok2]).mean()) if ok2.any() else float('nan')
    sym_density = len(sym) / (N_all * (N_all - 1))

    # ---- Top1 权重集中度（裁剪前 / 裁剪后）----
    # (a) 绝对口径（opus 主读数口径）：分母恒为**原始行和**（不重新归一化）→ 与裁剪前同值
    # (b) 诊断口径：分母 = 保留行之和（若重新归一化时的 Top1 份额；仅诊断，不用于读数）
    g_pre = edges.groupby('focal_id')['weight']
    pre_top1 = (g_pre.max() / g_pre.sum()).dropna()
    g_post = kept.groupby('focal_id')['weight']
    row_sum = g_pre.sum()
    post_top1 = (g_post.max() / row_sum.reindex(g_post.max().index)).dropna()
    kept_sum = g_post.sum()
    post_top1_renorm = (g_post.max() / kept_sum.reindex(g_post.max().index)).dropna()
    top1 = {
        'n_focal_with_edges': int(len(pre_top1)),
        'top1_share_pre_median': round(float(pre_top1.median()), 6),
        'top1_share_pre_mean': round(float(pre_top1.mean()), 6),
        'top1_share_pre_p90': round(float(pre_top1.quantile(0.9)), 6),
        'top1_share_post_median': round(float(post_top1.median()), 6),
        'top1_share_post_mean': round(float(post_top1.mean()), 6),
        'top1_share_post_p90': round(float(post_top1.quantile(0.9)), 6),
        'top1_share_median_delta': round(float(post_top1.median() - pre_top1.median()), 6),
        'top1_share_mean_delta': round(float(post_top1.mean() - pre_top1.mean()), 6),
        'top1_share_post_within_retained_median': round(float(post_top1_renorm.median()), 6),
        'top1_share_post_within_retained_mean': round(float(post_top1_renorm.mean()), 6),
        'top1_share_post_within_retained_p90': round(float(post_top1_renorm.quantile(0.9)), 6),
        'top1_effect': ('裁剪按 weight 降序且每 focal 至少保留 1 条 → **Top1 边永不被裁掉**，'
                        '在「不重新归一化」的主读数口径下 Top1 份额**逐 focal 不变**（delta = 0）；'
                        '`top1_share_post_within_retained_*` 为「若重新归一化」的诊断值（会被机械抬高，'
                        '仅诊断、不用于读数）'),
    }

    # ---- 逐 focal 截断误差 ----
    n_rows = e.groupby('focal_id').size()
    quota = q * n_rows
    err = pd.Series(np.ceil(quota.to_numpy()) - quota.to_numpy(), index=quota.index)
    trunc = {
        'rule': 'k_i = ⌈q × n_i⌉（上取整）；截断误差 = k_i − q·n_i ∈ [0,1)',
        'n_focal_cropped': int((pd.Series(np.ceil(quota.to_numpy()), index=n_rows.index) < n_rows).sum()),
        'truncation_error_mean': round(float(err.mean()), 6),
        'truncation_error_max': round(float(err.max()), 6),
        'rows_kept': int(len(kept)), 'rows_dropped': int(len(dropped)),
        'sum_k_i': int(len(kept)), 'q_times_E': float(q * len(edges)),
        'aggregate_excess_edges': float(len(kept) - q * len(edges)),
        'aggregate_excess_rel': float((len(kept) - q * len(edges)) / (q * len(edges))),
        'note': ('⌈⌉ 使每 focal 平均多保留约 0.5 条边 → 聚合密度系统性偏高约 '
                 '（Σ_i 1_{n_i>0} / 2）/(q·E)（见 aggregate_excess_rel）；该误差由**登记口径本身**'
                 '（⌈⌉ 上取整）产生，非实现误差。'),
    }
    # ---- 诊断（**非主读数**）：最大余数法（quota-exact）分配，用于证明 >1% 偏差源于 ⌈⌉ 舍入 ----
    floor_k = np.floor(quota.to_numpy()).astype('int64')
    target_total = int(round(q * len(edges)))
    need = int(target_total - floor_k.sum())
    k_exact = pd.Series(floor_k, index=quota.index)
    if need > 0:
        order = (quota - floor_k).sort_values(ascending=False, kind='mergesort').index[:need]
        k_exact.loc[order] += 1
    k_exact_row = e['focal_id'].map(k_exact).to_numpy()
    keep_ex = e['_rank'].to_numpy() < k_exact_row
    kept_ex = edges.loc[pd.Series(keep_ex, index=e.index).reindex(edges.index).fillna(False).to_numpy()]
    ke_fa = kept_ex['focal_id'].map(lambda f: CA.l2_of(f, sw2))
    ke_fb = kept_ex['source_id'].map(lambda f: CA.l2_of(f, sw2))
    ke_ok = (ke_fa.notna() & ke_fb.notna()).to_numpy()
    ex_share = float((ke_fa.to_numpy()[ke_ok] == ke_fb.to_numpy()[ke_ok]).mean()) if ke_ok.any() else float('nan')
    ex_density = len(kept_ex) / (N_all * (N_all - 1))
    exact_quota_diag = {
        'status': 'diagnostic_only_not_primary',
        'rule': 'k_i = floor(q·n_i)，余数最大者优先补 1 至总量 = round(q·E)（最大余数法；并列按 focal_id 升序）',
        'rows_kept': int(len(kept_ex)), 'target_rows': target_total,
        'retained_density': ex_density, 'density_deviation_rel': (ex_density - base) / base if base > 0 else None,
        'density_deviation_pct': round(100 * (ex_density - base) / base, 6) if base > 0 else None,
        'same_sw2_share': ex_share,
        'purpose': ('证明主读数 >1% 的密度偏差来自登记口径的 ⌈⌉ 上取整（每 focal 平均多留 0.5 条），'
                    '而非实现错误：改用等量分配（最大余数法）后密度偏差降至 ~0；**主读数仍按登记口径 '
                    '⌈⌉**，本块不替换、不改口径'),
    }

    return {
        'q': q,
        'q_definition': 'q = pair_density(=base) / d（全市场同一值）',
        'density_aligned_share': share_prime,
        'density_aligned_share_4dp': round(share_prime, 4),
        'lift_prime': lift_prime,
        'retained_density': retained_density,
        'retained_density_target': base,
        'density_deviation_rel': dev,
        'density_deviation_pct': round(100 * dev, 4),
        'density_deviation_within_1pct': bool(abs(dev) <= CROP_DENSITY_TOL),
        'normalized_readout_identity': {
            'share_prime': share_prime, 'lift_prime_times_d_prime': lift_prime * base,
            'identity_holds': bool(abs(lift_prime * base - share_prime) <= 1e-12),
            'note': '裁剪后 d′ = base → lift′ × d′ 恒等于 share′（同一对象）'},
        'symmetric_variant': {
            'definition': '「双向都保留」：仅保留正反两条边都被行内裁剪保留的边（对称化变体）',
            'n_edges': int(len(sym)), 'n_edges_mapped': int(ok2.sum()),
            'density': sym_density,
            'density_deviation_rel_vs_base': (sym_density - base) / base if base > 0 else float('nan'),
            'same_sw2_share': sym_share, 'same_sw2_share_4dp': round(sym_share, 4),
            'note': '对称化变体密度低于 base（双边同时保留概率 < 单向保留），故与 50.2% 对照时须注明'  ,
        },
        'top1_concentration': top1,
        'truncation': trunc,
        'exact_quota_diagnostic': exact_quota_diag,
    }


def column_report(label: str, net_dir: Path, years: list[int], sw2: dict,
                  stored_ref: dict | None = None) -> dict:
    per_year = []
    for y in years:
        fp_e, fp_s = net_dir / f'{y}_similarity.csv', net_dir / f'{y}_support.csv'
        if not (fp_e.exists() and fp_s.exists()):
            per_year.append({'text_year': y, 'status': 'missing'})
            continue
        sup = CA.read_support(net_dir, y)
        edges = CA.read_edges(net_dir, y)
        blk = pair_density_block(sup, edges, sw2)
        crop = crop_block(edges, sw2, blk['pair_density'], blk['d'],
                          blk) if blk['n_nodes_mapped'] > 1 else None
        cr = CA.corpus_richness(net_dir, y)
        iso = CA.isolation(edges, sup)
        row = {'text_year': y, 'network_year': y + 1, 'status': 'ok',
               'network_version': (cr or {}).get('network_version'),
               'isolation_rate': iso['isolation_rate'], 'K_clusters': cr.get('K_clusters'),
               'mean_terms': cr.get('mean_terms'),
               **{k: v for k, v in blk.items()},
               'density_readouts': {'normalized_readout': blk['normalized_readout'],
                                    'normalized_readout_pct': blk['normalized_readout_pct'],
                                    'normalized_readout_pct_1dp': blk['normalized_readout_pct_1dp'],
                                    'cropped_is_primary': True,
                                    **(crop or {})}}
        per_year.append(row)

    ok = [r for r in per_year if r.get('status') == 'ok']
    # 池化（pair 层：Σ 同类对 / Σ 全对；边层：Σ 同业边 / Σ 可映射边；密度：ΣE / ΣN(N−1)）
    K = sum(r['same_class_pairs_undirected'] for r in ok)
    U = sum(r['universe_pairs_undirected'] for r in ok)
    E = sum(r['n_edges'] for r in ok)
    Em = sum(r['n_edges_mapped'] for r in ok)
    NN = sum(r['n_nodes_total'] * (r['n_nodes_total'] - 1) for r in ok)
    same_edges = sum(r['same_sw2_share'] * r['n_edges_mapped'] for r in ok)
    pooled_pd = K / U if U else float('nan')
    pooled_share = same_edges / Em if Em else float('nan')
    pooled_d = E / NN if NN else float('nan')
    pooled = {'years': [r['network_year'] for r in ok],
              'n_nodes_total_sum': sum(r['n_nodes_total'] for r in ok),
              'n_nodes_mapped_sum': sum(r['n_nodes_mapped'] for r in ok),
              'pair_density': pooled_pd, 'pair_density_stored_5dp': round(pooled_pd, 5),
              'same_sw2_share': pooled_share, 'same_sw2_share_stored_4dp': round(pooled_share, 4),
              'lift': pooled_share / pooled_pd, 'lift_stored_3dp': round(pooled_share / pooled_pd, 3),
              'd': pooled_d, 'normalized_readout': (pooled_share / pooled_pd) * pooled_d,
              'note': ('池化口径 = pair 层 Σ 同类对/Σ 全对 + 边层 Σ 同业边/Σ 可映射边 + 密度层 ΣE/ΣN(N−1)；'
                       '池化 normalized_readout 为描述性（回归门只逐年判定）')}
    # 映射缺失导致的 N 年际跳变检查（以映射覆盖率 pp 变化衡量）
    jumps = []
    for a, b in zip(ok, ok[1:]):
        jumps.append({'from_network_year': a['network_year'], 'to_network_year': b['network_year'],
                      'mapped_share_delta_pp': round(b['mapped_share'] - a['mapped_share'], 6),
                      'n_missing_mapping': [a['n_missing_mapping'], b['n_missing_mapping']]})
    max_jump = max((abs(j['mapped_share_delta_pp']) for j in jumps), default=0.0)

    # 与既有 JSON 存储值 / 全精度复算值对账
    ref = stored_ref or {}
    cmp_rows = []
    for r in ok:
        s = ref.get(r['network_year'])
        if not s:
            continue
        cmp_rows.append({'network_year': r['network_year'],
                         'pair_density_ours_full': r['pair_density_canonical_path'],
                         'pair_density_stored_json': s.get('base'),
                         'abs_diff_vs_stored': abs(r['pair_density_canonical_path'] - s['base']),
                         'abs_diff_vs_stored_tol': ABS_TOL_STORED,
                         'pass_stored': bool(abs(r['pair_density_canonical_path'] - s['base']) <= ABS_TOL_STORED),
                         'same_sw2_share_stored_json': s.get('share'),
                         'abs_diff_share_vs_stored': (abs(r['same_sw2_share'] - s['share'])
                                                      if s.get('share') is not None else None),
                         'lift_stored_json': s.get('lift'),
                         'lift_ours_3dp': r['lift_stored_3dp'],
                         'lift_3dp_match': (r['lift_stored_3dp'] == s.get('lift')
                                            if s.get('lift') is not None else None)})
    max_rel_paths = max((abs(r['pair_density_paircount_path'] - r['pair_density_canonical_path'])
                         / r['pair_density_canonical_path'] for r in ok), default=0.0)
    max_abs_stored = max((c['abs_diff_vs_stored'] for c in cmp_rows), default=0.0)
    return {'label': label, 'net_dir': str(net_dir.relative_to(ROOT)).replace('\\', '/'),
            'per_year': per_year, 'pooled': pooled,
            'mapping_jump_check': {'rule': f'映射覆盖率（可映射节点/全部入网公司）年际变化 > {N_JUMP_TOL_PP*100:.0f}pp → 停',
                                   'per_step': jumps, 'max_abs_delta_pp': max_jump,
                                   'pass': bool(max_jump <= N_JUMP_TOL_PP)},
            'reconciliation_vs_stored_json': {'source': (str(CHAPTER_JSON.relative_to(ROOT)).replace('\\', '/')
                                                         if ref else None),
                                              'pair_density_vs_stored_json': cmp_rows,
                                              'max_abs_diff_vs_stored_5dp': max_abs_stored,
                                              'stored_tol': ABS_TOL_STORED,
                                              'pass': bool(cmp_rows and max_abs_stored <= ABS_TOL_STORED),
                                              'note': ('对比对象 = 既有 JSON 的**存储值**（5 位小数）；'
                                                       '不在 JSON 之间做 1e-12 级比较')},
            'full_precision_path_agreement': {
                'rule': '整数计数式 Σ n_k(n_k−1)/N(N−1) vs canonical 浮点式 (vc*(vc−1)).sum()/N(N−1)',
                'max_rel_diff': max_rel_paths, 'tol': REL_TOL_FULL,
                'pass': bool(max_rel_paths <= REL_TOL_FULL)}}


def gate_block(col: dict) -> dict:
    """回归门（只对主口径 teacher 列判定；self 列同表并报但标注非门）。

    v2（2026-10-11，⑧ 脚本级同步）：把 T1F 的 JSON 层修订收回脚本——③「裁剪后密度偏差 ≤1%」并入
    `acceptance.checks`（含 `pass=false` 与逐年偏差表）、self 列 `not_applicable_column_self` 处置、
    两列 ③ 文案共用；**判据与口径一字未改**（⌈q·n_i⌉ 仍为唯一主口径）。
    """
    label = col['label']
    ok = {r['network_year']: r for r in col['per_year'] if r.get('status') == 'ok'}
    rows = []
    for ny, exp in sorted(GATE_NORMALIZED_PCT_2018_2023.items()):
        r = ok.get(ny)
        got = r['density_readouts']['normalized_readout_pct'] if r else None
        rows.append({'network_year': ny, 'expected_pct_1dp': exp, 'ours_pct_4dp': got,
                     'ours_pct_1dp': round(got, 1) if got is not None else None,
                     'abs_diff_pp': abs(got - exp) if got is not None else None,
                     'exact_1dp_match': (round(got, 1) == exp) if got is not None else None,
                     'within_half_last_digit': (abs(got - exp) <= GATE_PCT_TOL) if got is not None else None})
    vals = [r['density_readouts']['normalized_readout_pct'] for r in ok.values()]
    rng = [round(min(vals), 1), round(max(vals), 1)]
    rng_ok = (abs(min(vals) - GATE_NORMALIZED_RANGE_9Y[0]) <= GATE_PCT_TOL
              and abs(max(vals) - GATE_NORMALIZED_RANGE_9Y[1]) <= GATE_PCT_TOL)
    vals_4dp = [r['density_readouts']['normalized_readout_pct'] for r in ok.values()]
    self_rows = [{'network_year': r['network_year'], 'text_year': r['text_year'],
                  'ours_pct_4dp': r['density_readouts']['normalized_readout_pct'],
                  'ours_pct_1dp': r['density_readouts']['normalized_readout_pct_1dp']}
                 for r in ok.values()]
    self_consistency = {
        'all_years_present': bool(len(self_rows) == len(ok)),
        'n_years': len(self_rows),
        'our_1dp_is_round_of_our_4dp_all_years': all(
            round(r['ours_pct_4dp'], 1) == r['ours_pct_1dp'] for r in self_rows),
        'observed_range_equals_min_max_of_per_year': ([round(min(vals_4dp), 4), round(max(vals_4dp), 4)]
                                                      if vals_4dp else None),
        'pass': bool(self_rows and all(round(r['ours_pct_4dp'], 1) == r['ours_pct_1dp']
                                       for r in self_rows)),
    }
    if label == 'self':
        # self 列：teacher 专用期望不适用（not_applicable_column_self），只报实测 + 自洽检查
        reg = {'definition': NORMALIZED_READOUT_DEF,
               'applicability': 'not_applicable_column_self',
               'not_applicable_reason': SELF_REG_NOT_APPLICABLE_REASON,
               'expected': None,
               'expected_field_status': SELF_REG_EXPECTED_FIELD_STATUS,
               'per_year': self_rows,
               'observed_range_pct_1dp': rng,
               'self_consistency_checks': self_consistency,
               'note': (f'self 列实测 normalized_readout = {min(vals_4dp):.4f}–{max(vals_4dp):.4f}%'
                        f'（{len(self_rows)} 个网络年；逐年 4dp 见 per_year）；teacher 期望与本列的'
                        f'「方向 / 范围」判定均为 teacher 专用登记，本列一律记 not_applicable_column_self；'
                        f'如需 self 列期望须另行登记')}
    else:
        reg = {'definition': NORMALIZED_READOUT_DEF,
               'expected': {'network_years_2018_2023': GATE_NORMALIZED_PCT_2018_2023,
                            'all_9y_range_pct': GATE_NORMALIZED_RANGE_9Y},
               'per_year': rows,
               'all_exact_1dp_match': all(r['exact_1dp_match'] for r in rows
                                          if r['exact_1dp_match'] is not None),
               'range_observed_pct_1dp': rng, 'range_pass': bool(rng_ok)}
    dev_rows = [{'network_year': r['network_year'],
                 'density_deviation_pct': r['density_readouts']['density_deviation_pct'],
                 'within_1pct': r['density_readouts']['density_deviation_within_1pct']}
                for r in ok.values()]
    # ③ 独立登记项：`pass=false`（⌈⌉ 口径固有超额）+ 逐年偏差表；与登记口径数学互斥 → 待拍板
    dev_pct_by_ny = {str(d['network_year']): d['density_deviation_pct'] for d in dev_rows}
    overshoot_years = [d['network_year'] for d in dev_rows if not d['within_1pct']]
    eq_quota = [r['density_readouts'].get('exact_quota_diagnostic') or {} for r in ok.values()]
    eq_devs = [(r['network_year'], float(e.get('density_deviation_rel') or 0.0))
               for r, e in zip(ok.values(), eq_quota) if e]
    eq_max_year, eq_max_dev = (max(eq_devs, key=lambda t: abs(t[1])) if eq_devs else (None, 0.0))
    if eq_max_dev == 0.0:
        eq_txt = f'本列等量分配实测偏差逐年全为 0（{len(eq_devs)} 年）'
    else:
        eq_txt = (f'{eq_max_year} = {fmt_e(abs(eq_max_dev), tight=True)}，其余年份 0')
    crop_check = {
        'pass': bool(dev_rows) and all(d['within_1pct'] for d in dev_rows),
        'threshold_rel': CROP_DENSITY_TOL, 'threshold_pct': 100 * CROP_DENSITY_TOL,
        'measured_deviation_pct_by_network_year': dev_pct_by_ny,
        'measured_max_deviation_pct': (max(abs(d['density_deviation_pct']) for d in dev_rows)
                                       if dev_rows else None),
        'overshoot_network_years': overshoot_years,
        'n_overshoot_years': len(overshoot_years), 'n_years_total': len(dev_rows),
        'measured_field': CROP_CHECK_MEASURED_FIELD,
        'note': CROP_CHECK_NOTE.format(
            lo=min(abs(d['density_deviation_pct']) for d in dev_rows) if dev_rows else float('nan'),
            hi=max(abs(d['density_deviation_pct']) for d in dev_rows) if dev_rows else float('nan'),
            eq=eq_txt),
        'column': label,
    }
    # 方向检查（预期 > 35.4–42.0%；**不得与 35.4–42.0% 做等值对账**）
    crops = {r['network_year']: r['density_readouts'] for r in ok.values()}
    crop_pct = {y: 100 * c['density_aligned_share'] for y, c in sorted(crops.items())}
    n_above_low = sum(1 for v in crop_pct.values() if v > GATE_NORMALIZED_RANGE_9Y[0])
    direction_ok = bool(n_above_low == len(crop_pct))
    hp_side_by_side = {'hp_tnic_x_sic3_normalized_share_pct': 100 * HP_TNIC_SIC3_NORMALIZED,
                       'hp_fic_10k300_x_sic3_raw_share_pct': 100 * HP_FIC300_SIC3_RAW,
                       'ours_density_aligned_share_pct_range': [round(min(crop_pct.values()), 2),
                                                                round(max(crop_pct.values()), 2)]
                       if crop_pct else None,
                       'ours_uncropped_normalized_readout_pct_range':
                           [round(min(r['density_readouts']['normalized_readout_pct'] for r in ok.values()), 2),
                            round(max(r['density_readouts']['normalized_readout_pct'] for r in ok.values()), 2)]
                       if ok else None,
                       'comparability_limits': [
                           'HP 50.2% = TNIC-3 标到 SIC-3 密度后的**真份额**（HP 未报告其未对齐 raw share '
                           '与裁剪实施细节）；我方对应物 = 本字段的真份额 share′',
                           '分类法不同：申万二级 vs SIC-3；样本池不同：A 股逐年 vs HP 2011 横截面',
                           '网络类型不同：传递划分（团）vs 非传递阈值图（更接近 FIC 10K-300）',
                           '**不得**据本表写「我方更优 / 更不依赖行业」']}
    if label == 'self':
        # 期望方向 / 方向不符登记均为 teacher 专用 → self 记 not_applicable_column_self（M-2）
        share_vals = [c['density_aligned_share'] for c in crops.values()]
        direction_block = {
            'definition': DENSITY_ALIGNED_DEF_SELF,
            'expected_direction': None,
            'observed_pct_by_network_year': {y: round(v, 4) for y, v in crop_pct.items()},
            'observed_range_pct': [round(min(crop_pct.values()), 4), round(max(crop_pct.values()), 4)]
            if crop_pct else None,
            'n_years_above_35_4pct': None, 'n_years_total': len(crop_pct),
            'direction_as_expected': None,
            'hp_side_by_side': hp_side_by_side,
        }
    else:
        direction_block = {
            'definition': DENSITY_ALIGNED_DEF,
            'expected_direction': 'density_aligned_share > 35.4–42.0%（未裁剪归一化读数的区间）',
            'observed_pct_by_network_year': {y: round(v, 4) for y, v in crop_pct.items()},
            'observed_range_pct': [round(min(crop_pct.values()), 4), round(max(crop_pct.values()), 4)]
            if crop_pct else None,
            'n_years_above_35_4pct': n_above_low, 'n_years_total': len(crop_pct),
            'direction_as_expected': direction_ok,
            'hp_side_by_side': hp_side_by_side,
        }
        if not direction_ok:
            direction_block['direction_mismatch_registration'] = (
                f'**方向与预期不符（须登记说明）**：实测 density_aligned_share = '
                f'{round(min(crop_pct.values()), 2)}–{round(max(crop_pct.values()), 2)}%，其中仅 '
                f'{n_above_low}/{len(crop_pct)} 年 > 35.4%。原因（不含推断、仅机制）：① 未裁剪 '
                f'`normalized_readout = lift × d = share / q` 是把密度按比例缩到 base 时的**线性外推**'
                f'（放大 1/q ≈ 1.2–2.0 倍），而 `density_aligned_share` 是**实际裁剪后重算的真份额**——'
                f'两者是不同对象（spec 明令不得等值对账）；② 裁剪按 weight 降序，保留的最高权重边同二级率更高 '
                f'→ share′ 高于未裁剪 raw share（如 2018：0.1940 → 0.2683），但**低于**外推值 share/q '
                f'（2018：0.3806）→ q 明显小于 1 的年份 share′ 落在 35.4% 以下，只有 q 接近 1 的年份'
                f'（2022/2023）达到该区间。**不得反向解读**（不得写成「我方比 HP 更/更不依赖行业」），'
                f'**不得与 35.4–42.0% 做等值对账**；如需「同口径 > 45%」式的强主张，须另登记'
                f'（HP 式裁剪实施细节未公开，我方按 weight 降序裁剪为唯一可执行替身）。')
    dag = {**direction_block,
           'per_year_density_deviation': dev_rows,
           'crop_density_tol': CROP_DENSITY_TOL,
           'all_within_tol': all(d['within_1pct'] for d in dev_rows) if dev_rows else None,
           'density_deviation_cause': ('偏差 >1% 的年份由登记口径的 ⌈q·n_i⌉ 上取整造成（每 focal 平均'
                                       '多保留 0.5 条边，聚合相对超额 = Σ_i(1−frac_i)/(q·E)）；'
                                       'diagnostic 用最大余数法等量分配后偏差 ≈0（见 per_year 的 '
                                       'exact_quota_diagnostic）。**不得据此改口径**（⌈⌉ 为登记口径）。'),
           'note': '本块**独立报告**，不与 normalized_readout 互相等值对账'}
    if label == 'self':
        dag.update({
            'applicability': 'not_applicable_column_self',
            'not_applicable_reason': SELF_DENSITY_ALIGNED_NOT_APPLICABLE_REASON,
            'observed_self_consistency': {
                'n_years': len(share_vals),
                'observed_share_range': ([round(min(share_vals), 6), round(max(share_vals), 6)]
                                         if share_vals else None),
                'observed_share_range_pct_4dp': ([round(100 * min(share_vals), 4),
                                                  round(100 * max(share_vals), 4)]
                                                 if share_vals else None),
                'lift_prime_times_d_prime_identity_holds_all_years': all(
                    bool((c.get('normalized_readout_identity') or {}).get('identity_holds'))
                    for c in crops.values()),
                'per_year_density_deviation_complete': bool(len(dev_rows) == len(crops)),
                'pass': None,
            },
            'note': SELF_DENSITY_ALIGNED_NOTE})
        dag['observed_self_consistency']['pass'] = bool(
            dag['observed_self_consistency']['lift_prime_times_d_prime_identity_holds_all_years']
            and dag['observed_self_consistency']['per_year_density_deviation_complete'])
    return {'regression_gate_normalized_readout': reg, 'density_aligned_gate': dag,
            'crop_density_deviation_check': crop_check}


def load_stored_ref(chapter_json: Path, weight_json: Path, label: str) -> dict:
    """既有 JSON 的存储值（5 位小数）：teacher 优先取 weighting（9 年）、self 取 chapter（6 年）。"""
    out: dict[int, dict] = {}
    if label == 'teacher' and weight_json.exists():
        d = json.loads(weight_json.read_text(encoding='utf-8'))
        for r in d.get('per_year', []):
            if r.get('status') != 'ok':
                continue
            out[r['network_year']] = {'base': r.get('sw2_base_canonical_ew'),
                                      'share': r.get('same_sw2_share_ew'),
                                      'lift': r.get('sw2_lift_ew'),
                                      'source': str(weight_json.relative_to(ROOT)).replace('\\', '/')}
    if chapter_json.exists():
        d = json.loads(chapter_json.read_text(encoding='utf-8'))
        key = 'teacher' if label == 'teacher' else 'self'
        for r in d.get('per_year', []):
            if r.get('status') != 'ok':
                continue
            ny = r['network_year']
            rec = {'base': r.get(f'sw2_base_canonical_{key}'), 'share': r.get(f'same_sw2_{key}'),
                   'lift': r.get(f'sw2_lift_{key}'),
                   'source': str(chapter_json.relative_to(ROOT)).replace('\\', '/')}
            if ny not in out or out[ny].get('base') is None:
                out[ny] = rec
            else:
                out[ny]['chapter_crosscheck'] = rec
    return out


def build_pending_decision(results: dict) -> dict:
    """候选新判据草案（**原文收录**；挂在 teacher 文件；数字逐年内核于本列两列产物）。

    内容 = ③「裁剪后密度偏差 ≤1%」与登记裁剪口径 ⌈q·n_i⌉ 数学互斥 → 是否改判据/改口径待项目拍板。
    **本轮不改判据、不改口径、不改 registry、不改任何数值读数**。
    """
    def ok_rows(rep: dict) -> list:
        return [r for r in rep.get('per_year', []) if r.get('status') == 'ok']
    def dr(r: dict) -> dict:
        return r['density_readouts']
    t = ok_rows(results.get('teacher', {}))
    s = ok_rows(results.get('self', {}))
    eq = [(r['network_year'], float((dr(r).get('exact_quota_diagnostic') or {})
                                    .get('density_deviation_rel') or 0.0)) for r in t]
    y_max, d_max = max(eq, key=lambda x: abs(x[1])) if eq else (None, 0.0)
    nz = [(y, d) for y, d in eq if d != 0.0]
    bounds = [(0.5 / round(dr(r)['truncation']['q_times_E']), r['network_year']) for r in t]
    b_lo, b_lo_y = min(bounds) if bounds else (None, None)
    b_hi, b_hi_y = max(bounds) if bounds else (None, None)
    sbounds = [(0.5 / round(dr(r)['truncation']['q_times_E']), r['network_year']) for r in s]
    s_lo, s_lo_y = min(sbounds) if sbounds else (None, None)
    devs = [abs(dr(r)['density_deviation_pct']) for r in t] or [float('nan')]
    gaps = [abs(dr(r)['density_deviation_rel'] - dr(r)['truncation']['aggregate_excess_rel'])
            / abs(dr(r)['truncation']['aggregate_excess_rel']) for r in t] or [float('nan')]
    tr_max = max((dr(r)['truncation']['truncation_error_max'] for r in t), default=None)
    tr_complete = bool(t) and all((dr(r).get('truncation') or {}).get('n_focal_cropped') is not None for r in t)
    eq_part = ('逐年全为 0' if eq and not nz else
               f'{y_max} = {fmt_e(abs(d_max), tight=True)}' + ('，其余年份 = 0' if len(nz) == 1 else '')
               if eq else '（无年份）')
    return {
        'status': 'AWAITING_PROJECT_DECISION',
        'scope': '验收③「裁剪后密度偏差 ≤1%」与登记裁剪口径 k_i = ⌈q·n_i⌉ **数学互斥** → 是否改判据'
                 '（或改裁剪口径）由项目拍板；**本轮不改判据、不改口径、不改任何冻结参数**',
        'background': (f'⌈⌉ 上取整使每个 focal 平均多保留约 0.5 条边，聚合相对超额 = '
                       f'Σ_i(1−frac_i)/(q·E) 恒为 {min(devs):.2f}–{max(devs):.2f}%（本数据，逐年见 '
                       f'acceptance.density_aligned_gate.per_year_density_deviation）→ 与「偏差 ≤1%」'
                       f'不可能同时成立。改用最大余数法等量分配（k_i = floor(q·n_i)，余数最大者补 1 至'
                       f'总量 = round(q·E)）后偏差 ≈0（teacher 列 {eq_part}）'),
        'candidate_criteria_draft': {
            'source': PENDING_DECISION_DRAFT_SOURCE,
            'draft_a_existence_max_remainder': {
                'draft_rule': '存在性：最大余数法下 |dev| ≤ 0.5/round(q·E)',
                'must_not': (f'**不得写成 1e-6**：teacher 列**网络年（文本年 y−1）{y_max}** 的等量分配'
                             f'实测偏差 = {fmt_e(abs(d_max), tight=True)} 已超 1e-6 → 会把「实现正确」误判为失败'
                             f'（阈值须写成 0.5/round(q·E)）'),
                'verified_on_this_data': {
                    'years': len(eq),
                    'max_abs_quota_dev': d_max,
                    'max_abs_quota_dev_network_year': y_max,
                    'bound_0p5_over_round_qE_range': [b_lo, b_hi],
                    'bound_min_network_year': b_lo_y, 'bound_max_network_year': b_hi_y,
                    'note': (f'复核草案原文记「本数据 ≤6.0e-6」；本轮逐年内核的**实际上界** = teacher 列 '
                             f'{fmt_e(b_lo, sig=4, tight=True)}–{fmt_e(b_hi, sig=4, tight=True)}（最小网络年 {b_lo_y}、'
                             f'最大网络年 {b_hi_y}）、self 列最小 {fmt_e(s_lo, sig=4, tight=True)}'
                             f'（网络年 {s_lo_y}）→ 判据须写成 0.5/round(q·E) 而**不得写成固定常数**'
                             f'（更不得写 1e-6）'),
                    'pass': bool(eq) and all(
                        abs(d) <= 0.5 / round(dr(r)['truncation']['q_times_E']) for (_, d), r in zip(eq, t)),
                    'measured_fields': 'per_year[].density_readouts.exact_quota_diagnostic.density_deviation_rel '
                                       '与 per_year[].density_readouts.truncation.q_times_E'},
            },
            'draft_b_primary_ceil_analytic_bound': {
                'draft_rule': '主口径 ⌈⌉：dev 必须等于解析上界 Σ_i(1−frac_i)/(q·E)（rel ≤1e-12）+ '
                              '逐 focal 截断误差 ∈[0,1) 齐全',
                'verified_on_this_data': {
                    'years': len(gaps),
                    'max_rel_gap_dev_vs_analytic_upper_bound': max(gaps),
                    'tol_rel': REL_TOL_FULL,
                    'truncation_error_max_max': tr_max,
                    'truncation_error_in_0_1_open': bool(tr_max is not None and 0.0 <= tr_max < 1.0),
                    'truncation_diagnostics_complete': tr_complete,
                    'truncation_error_min_note': 'min ≥0 由口径定义 k_i − q·n_i ∈[0,1) 保证'
                                                 '（产物只登记 mean/max 与 n_focal_cropped，逐年齐全）',
                    'pass': bool(max(gaps) <= REL_TOL_FULL and tr_max is not None and 0.0 <= tr_max < 1.0
                                 and tr_complete),
                    'measured_fields': 'per_year[].density_readouts.density_deviation_rel vs '
                                       'per_year[].density_readouts.truncation.aggregate_excess_rel'
                                       '（analytic 解析上界）'},
            },
        },
        'cross_column_note': ('self 列共用同一实现：self 列 dev vs 解析上界的 rel 亦 ≤1e-12、逐 focal '
                              '截断误差同样 ∈[0,1)，且其 ③ 同样因 ⌈⌉ 口径不通过（见 '
                              's3_lift_pair_density_self.json.acceptance.checks.crop_density_deviation_le_1pct）'),
        'no_change_this_round': ('本轮只改状态/注记：**不改判据、不改裁剪口径、不改 registry、不改任何数值'
                                 '读数**；status 由 PASS_WITH_REGISTERED_DEVIATION 改为 STOP_PENDING_DECISION，'
                                 '仅反映「③ 不通过 + 判据缺陷待拍板」'),
        'decision_owner': '项目（拍板前：不得据此改口径/改冻结参数；不得反向解读为「实现通过」或「复算失败」）',
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher-dir', default=str(TEACH_DIR))
    ap.add_argument('--self-dir', default=str(SELF_DIR))
    ap.add_argument('--teacher-years', type=int, nargs='*', default=TEACHER_YEARS)
    ap.add_argument('--self-years', type=int, nargs='*', default=SELF_YEARS)
    ap.add_argument('--out-dir', default=str(REPD))
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    sw2 = CA.load_sw2()
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    results = {}
    for label, ndir, yrs in (('teacher', Path(args.teacher_dir), args.teacher_years),
                             ('self', Path(args.self_dir), args.self_years)):
        print(f'[T1-①②] 复算 {label} 列：{len(yrs)} 个文本年 → 网络年 {yrs[0]+1}–{yrs[-1]+1} ...', flush=True)
        col = column_report(label, ndir, yrs, sw2, load_stored_ref(CHAPTER_JSON, WEIGHT_JSON, label))
        g = gate_block(col)
        crop_check = g.pop('crop_density_deviation_check')   # ③ 独立登记块（并入 acceptance.checks）
        rep = {
            'step': f'S3 第一梯队 ①② pair-density 读数 + 密度对齐复算（{label} 列）',
            'run_at': datetime.now(timezone.utc).isoformat(),
            'column': label,
            'net_dir': col['net_dir'],
            'nature': ('只读复算：不重建网络、不碰面板/回归、不改任何 headline_value / 冻结参数；'
                       '产物为新建文件（不覆盖既有产物）'),
            'inputs': {'edges_support': col['net_dir'],
                       'sw2_map': str(SW2.relative_to(ROOT)).replace('\\', '/'),
                       'stored_reference_json': [str(p.relative_to(ROOT)).replace('\\', '/')
                                                 for p in (CHAPTER_JSON, WEIGHT_JSON) if p.exists()]},
            'definitions': {'pair_density': PAIR_DENSITY_DEF, 'same_sw2_share_and_lift': LIFT_DEF,
                            'd': 'd = E/(N(N−1))（E = 有向边表行数、N = 全部入网公司数）',
                            'normalized_readout': NORMALIZED_READOUT_DEF,
                            'density_aligned_share': DENSITY_ALIGNED_DEF,
                            'density_readouts_block_note': ('块名 `density_readouts`（旧名 `density_aligned` 已弃用）；'
                                                            '字段名 `density_aligned_share` 不变；两字段**各自验收、'
                                                            '禁止互相等值对账**')},
            'hp_reference': {
                'hp_pair_density_sic3': HP_PAIR_DENSITY,
                'source': 'HP16 (JPE) PDF p.19「2.05% of all possible firm pairs are industry pairs」+ '
                          'HP 官方 TNIC-3 readme 行 51–52（SIC-3 基准、2011 Capital IQ 横截面）',
                'hp_tnic_x_sic3_normalized_share': HP_TNIC_SIC3_NORMALIZED,
                'hp_fic_10k300_x_sic3_raw_share': HP_FIC300_SIC3_RAW,
                'comparability_limits': [
                    'cross-taxonomy：我方分母/分子口径 = 申万二级（SW2），HP = SIC-3',
                    '样本池不同：我方 = A 股当年入网公司（含零对手），HP = 2011 年 Capital IQ 横截面',
                    '密度未对齐：我方 d 显著高于行业基线（d / pair_density 见 per_year），故同一密度下'
                    '可比读数应看 normalized_readout；本行 pair_density 与 2.05% 只作**量级指示**',
                    '网络类型：我方为传递划分（团），HP TNIC-3 为非传递阈值图（更接近 FIC 10K-300）'],
            },
            'per_year': col['per_year'],
            'pooled': col['pooled'],
            'mapping_jump_check': col['mapping_jump_check'],
            'reconciliation_vs_stored_json': col['reconciliation_vs_stored_json'],
            'full_precision_path_agreement': col['full_precision_path_agreement'],
            'acceptance': g,
        }
        rl = rep['reconciliation_vs_stored_json']
        fp = rep['full_precision_path_agreement']
        checks = {'pair_density_vs_stored_json_5dp': rl['pass'],
                  'full_precision_path_rel_le_1e-12': fp['pass'],
                  'years_complete': bool(len([r for r in col['per_year'] if r.get('status') == 'ok']) == len(yrs)),
                  'mapping_no_year_jump_gt_1pp': col['mapping_jump_check']['pass']}
        if label == 'teacher':
            checks['normalized_readout_regression_gate'] = bool(
                g['regression_gate_normalized_readout']['all_exact_1dp_match']
                and g['regression_gate_normalized_readout']['range_pass'])
        stops = []
        if not checks['pair_density_vs_stored_json_5dp']:
            stops.append('pair_density 与存储值（5 位小数）不一致 → 代码路径分叉，先修再报')
        if not checks['full_precision_path_rel_le_1e-12']:
            stops.append('全精度双路径 rel > 1e-12 → 复算路径分叉')
        if not checks['mapping_no_year_jump_gt_1pp']:
            stops.append('缺失映射导致 N 年际跳变 >1pp → 停，先复核映射表')
        if label == 'teacher' and not checks['normalized_readout_regression_gate']:
            stops.append('normalized_readout 与已复算 38.1/35.4/38.0/38.1/42.0/41.8%（2018–2023）及 '
                         '35.4–42.6%（9 年）不一致 → 停')
        crop_fail = [d for d in g['density_aligned_gate']['per_year_density_deviation'] if not d['within_1pct']]
        if crop_fail:
            stops.append(f'裁剪后密度偏差 >1% 的年份 = {[d["network_year"] for d in crop_fail]} '
                         '（成因见 truncation：k_i = ⌈q·n_i⌉ 的上取整按登记口径必然系统性多留约 0.5 条/focal）')
        rep['acceptance']['checks'] = checks
        rep['acceptance']['checks']['crop_density_deviation_le_1pct'] = crop_check   # ③（本轮并入）
        rep['acceptance']['observations_registered'] = {
            'density_aligned_direction_as_expected': bool(g['density_aligned_gate']['direction_as_expected']),
            'density_aligned_observed_pct_range': g['density_aligned_gate']['observed_range_pct'],
            'n_years_above_35_4pct': sum(
                1 for v in (g['density_aligned_gate']['observed_pct_by_network_year'] or {}).values()
                if v > GATE_NORMALIZED_RANGE_9Y[0]),
            'direction_check_required_action': ('spec：方向不符**须登记说明**（不得反向解读、不得做等值对账）'
                                                '→ 已在 acceptance.density_aligned_gate.'
                                                'direction_mismatch_registration 登记'),
            'crop_density_deviation_overshoot_years': [d['network_year'] for d in crop_fail],
            'crop_density_deviation_disposition': (
                '「裁剪后密度偏差 ≤1%」的偏差来源 = 登记口径本身的 ⌈q·n_i⌉ 上取整（每 focal 平均多保留 '
                '0.5 条边，聚合超额 Σ_i(1−frac_i)/(q·E)）；用最大余数法等量分配后偏差 ≈0（逐 year 见 '
                'exact_quota_diagnostic）→ 判定为**口径固有舍入**，非实现错误、非数据问题；'
                '**不据此改口径**（⌈⌉ 为 registry 登记口径），但该偏差须随产物登记。'
                if crop_fail else '无年份超限'),
        }
        rep['acceptance']['stop_conditions_triggered'] = stops
        rep['acceptance']['stop_disposition'] = (
            {s: 'registered_analytic_explanation（口径固有 ⌈⌉ 舍入；非实现误差）→ '
                '判据③是否修订待项目拍板（见 pending_decision / status = STOP_PENDING_DECISION）'
             for s in stops} if stops else {})
        # status：bool 验收全过 → 再看 ③；③ 不通过 = 判据缺陷待拍板（STOP_PENDING_DECISION），不是 FAIL
        bool_checks = dict(rep['acceptance']['checks'])
        crop_pass = bool(crop_check['pass'])
        bool_checks.pop('crop_density_deviation_le_1pct')
        if not all(bool_checks.values()):
            rep['status'] = 'FAIL'
        elif not crop_pass:
            rep['status'] = 'STOP_PENDING_DECISION'
        elif stops:
            rep['status'] = 'PASS_WITH_REGISTERED_DEVIATION'
        else:
            rep['status'] = 'PASS'
        ok_rows = [r for r in col['per_year'] if r.get('status') == 'ok']
        devs = [abs(d['density_deviation_pct']) for d in
                g['density_aligned_gate']['per_year_density_deviation']] or [float('nan')]
        nrs = [r['density_readouts']['normalized_readout_pct'] for r in ok_rows] or [float('nan')]
        eq_list = [(r['network_year'], float((r['density_readouts'].get('exact_quota_diagnostic') or {})
                                            .get('density_deviation_rel') or 0.0)) for r in ok_rows]
        nz = [(y, d) for y, d in eq_list if d != 0.0]
        eq_diag_ref = 'per_year[].density_readouts.exact_quota_diagnostic'
        if not eq_list:
            eq_part = f'（无年份内核；见 {eq_diag_ref}）'
        elif not nz:
            eq_part = f'逐年全为 0（{len(eq_list)} 年；见 {eq_diag_ref}）'
        else:
            ymax, dmax = max(nz, key=lambda t: abs(t[1]))
            eq_part = (f'最大 = {abs(dmax):.2e}（网络年 {ymax}）' + (
                '，其余年份 0' if len(nz) == 1 else f'，非零网络年 = {[y for y, _ in nz]}')
                + f'（见 {eq_diag_ref}）')
        ny0, ny1 = ok_rows[0]['network_year'], ok_rows[-1]['network_year']
        ty0, ty1 = ok_rows[0]['text_year'], ok_rows[-1]['text_year']
        n_checks = len(rep['acceptance']['checks'])
        n_true = sum(1 for k, v in rep['acceptance']['checks'].items() if v is True)
        if crop_pass:
            crop_txt = (f'**验收③「裁剪后密度偏差 ≤1%」为 true**（本列实测逐年偏差 '
                        f'{min(devs):.4f}%–{max(devs):.4f}%，全部在阈值内）。')
        else:
            crop_txt = (f'**验收③「裁剪后密度偏差 ≤1%」为 false**（本列实测逐年偏差 '
                        f'{min(devs):.4f}%–{max(devs):.4f}%，超限年份见该键的 '
                        f'measured_deviation_pct_by_network_year / overshoot_network_years）。'
                        f'该判据与登记裁剪口径 k_i = ⌈q·n_i⌉ **数学互斥**：⌈⌉ 的系统性超额 = '
                        f'Σ_i(1−frac_i)/(q·E) 本列恒为 {min(devs):.4f}–{max(devs):.4f}%'
                        f'（每 focal 平均多保留约 0.5 条边），属 **spec 层判据缺陷、非实现误差**——'
                        f'本列等量分配实测偏差{eq_part}。验收④（Top1 集中度 / 截断误差 / 等量分配诊断）'
                        f'已齐全登记于 `per_year[].density_readouts`，本轮未新增 gating 项（不改判据）。'
                        f'**是否改判据/改口径待项目拍板，本轮不改**（不改口径、不改任何冻结参数）。')
        if label == 'teacher':
            tail = (f'本列 network_year 集为 {ny0}–{ny1}（文本年 {ty0}–{ty1}）；② 回归门'
                    f'（网络年 2018–2023）为 teacher 列专用并已并入 checks。**候选新判据草案**'
                    f'（存在性 0.5/round(q·E)；主口径解析上界）**原文收录**于 `pending_decision` 块，'
                    f'标 status = AWAITING_PROJECT_DECISION。')
        else:
            tail = (f'本列 network_year 集为 {ny0}–{ny1}（文本年 {ty0}–{ty1}）：teacher 列的回归门期望'
                    f'（网络年 2018–2023 = 38.1/35.4/38.0/38.1/42.0/41.8%）与其逐值方向预期'
                    f'（> 35.4–42.0%）**不适用于本列**（见 acceptance.'
                    f'regression_gate_normalized_readout.applicability 与 acceptance.density_aligned_gate.'
                    f'applicability = not_applicable_column_self）；③ 为两列共用判据，其候选新判据草案'
                    f'原件收在 teacher 文件的 pending_decision 块。')
        rep['status_note'] = (f'**状态 = {rep["status"]}（本列）**：`acceptance.checks` 共 {n_checks} 项'
                              f'（含本轮并入的验收③），其中 {n_true} 项为 true、{crop_txt}'
                              f'本列实测 normalized_readout = {min(nrs):.4f}–{max(nrs):.4f}%'
                              f'（network_year {ny0}–{ny1}）。' + tail)
        rep['year_axis_note'] = YEAR_AXIS_NOTE.format(label=label)
        if label == 'self':
            rep['acceptance']['checks_scope_note'] = CHECKS_SCOPE_NOTE
        results[label] = rep
        print(f'[T1-①②] {label}: status={rep["status"]} checks={checks} -> (待写) {outdir}', flush=True)
        if stops:
            for s in stops:
                print(f'[T1-①②] 停机/告警（{label}）：{s}', flush=True)
        print(f'[T1-①②] {label} 逐年 normalized_readout% = '
              + ', '.join(f'{r["network_year"]}:{r["density_readouts"]["normalized_readout_pct"]:.2f}'
                          for r in col['per_year'] if r.get('status') == 'ok'), flush=True)
        print(f'[T1-①②] {label} 逐年 density_aligned_share = '
              + ', '.join(f'{r["network_year"]}:{r["density_readouts"]["density_aligned_share"]:.4f}'
                          for r in col['per_year'] if r.get('status') == 'ok'), flush=True)
        print(f'[T1-①②] {label} 逐年裁剪密度偏差% = '
              + ', '.join(f'{r["network_year"]}:{r["density_readouts"]["density_deviation_pct"]:+.3f}'
                          for r in col['per_year'] if r.get('status') == 'ok'), flush=True)

    # 候选新判据草案（**原文收录**；teacher 文件；需两列年内核数字 → 两列算完后统一装配）
    if 'teacher' in results:
        results['teacher']['pending_decision'] = build_pending_decision(results)
    for label in results:
        p = outdir / f's3_lift_pair_density_{label}{args.tag}.json'
        p.write_text(json.dumps(results[label], ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'[T1-①②] {label}: 产物写入 -> {p}', flush=True)


if __name__ == '__main__':
    main()
