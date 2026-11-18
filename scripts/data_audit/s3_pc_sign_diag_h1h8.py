#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3「PC → 预期收益」系数反号 · 深度诊断 H1–H8（2026-10-21，只读既有产物 + 新建诊断产物）。

背景（既定事实，不重复验证）：原文（刘亚辉等 2021）PC = −HHI(团内销售收入)，PC 越大 = 竞争越激烈，
PC → 预期收益 **为正**（表 4：+0.152/t=7.76、+0.130/t=5.40；表 3 H−L α3 +0.218/t=6.27）。
我方两套对表（MD&A 语料 / 年报全文语料）均反号（−0.507/t=−1.46、−0.456/t=−2.00 等）。

== 判据（先登记，先于运行写入本 docstring；不做事后调参、不做事后样本挑选）==
  · 每个 H 的 **成立** = 该 H 的口径/修正使 **主读数层**（登记样本 EW lag6 单变量）PC 系数 ≥ 0，
    且 4 子层（{EW, VW} × {五分位, 十分位} H−L、单变量/加控制 FM）中 ≥3 层同向转正；
  · **部分** = 向 0 移动 ≥50% 但仍为负；**排除** = 移动 <25% 或同号且 |t| 不降；**待定** = 数据不可得/证据冲突；
  · **H1 特有**（注入检验）：合成「PC 与下月收益正相关」面板（不改真数据，只在内存构造合成列）
    → 期望显著为正；**若输出为负 → 管线方向 bug 确诊，立即停止并写最小修复方案**；
  · **H8 特有**：PC 对团规模（log 团员数）回归取残差后 FM 系数 ≥0 → 机械/结构解释成立。

== 纪律 ==
  只新建脚本/产物；不覆盖任何既有产物、不改 headline_value/冻结值/主口径网络；
  全部新产物落 `artifacts/tmp/pc_diag/`（本脚本 + 各 H 的 JSON）；最终报告落
  `reports/audits/s3_pc_sign_diagnostics_20261021.{json,md}`（新建）。

== REV2 修订（2026-10-21，依独立复核意见 F1–F6；只改本诊断轮产物）==
  复核实测证据（采纳） = artifacts/tmp/review_diag/{rv_lib.py,A_injection.py,B_h2.py,C_h3h4.py,
  D_h5.py,E_h6.py,F_h8.py,F2_h8_identity.py} 及同名 JSON。修订采用**先登记后跑**（判据与新增检验在本
  docstring 记录后执行），并逐条：
  · F1｜H8 撤回「PC ≈ 团结构代理」结论（对处理变量的定义性结构分量回归取残差 = 过度控制/bad control，
    且结论对函数形式不稳健：log 形式翻正 +0.3221 但 1/N 形式仍负 −0.6302）→ 改判**待定/未被识别**；
    补做替代检验（本脚本独立重算）：(a) 恒等式分解 PC = −own_share² + (−同团他人集中度)（两分量 FM 各自符号/t）；
    (b) 机械控制换 1/N；(c) within-N 分箱 FM 与样本占比；(d) 分布匹配重加权（原文表 2 PC 分位密度 → 重要性
    重加权 + 分位匹配两种实现）；(e) 团标签置换 **1000 次**（报经验 p 与最小可检效应 MDE）；(f) 功效/MDE 分析。
  · F2｜H5 降级为**待定（无正面证据；2007–2016 窗口不可测）**；年度/子期间一律并报 (i) iid t、(ii) 月块
    bootstrap CI95、(iii) LOO 区间；显著性判据以 bootstrap CI 为准（不用 NW t 报"显著性"）。
  · F3｜H6 归因收窄：「团规模控制使 PC 转正」**仅限 log 形式且不显著**；补 1/N 线性控制；补控制项符号
    与原文一致计数（size/bm/rev 相反、mom/illiq 同号 = 3/5 相反）。
  · F4｜精度/强度表述修正：H2 max|Δhhi| ≤5e-9（写盘 round(x,8) 精度）→ 本轮以**全精度**复算并给出最大
    实测值；H3 「11/11 为负」→「无正号（多档 |t| ≤ 0.55，全部不显著）」；H3「高度吻合」限定为「中心四分位
    吻合（p25/中位/p75 vs 原文）、sd 0.2204 vs 0.337 尾部不一致」；H2 营收字段降级为「盘上 pkl = 营业收入
    （854.30 亿），原文附录表未核」。
  · F5｜H7 持有期 β 折算月均（t+2..6 / t+7..12 ÷ 窗长）并注明重叠窗 + NW(6) 未做重叠校正、t 不作推断依据。
  · F6｜治理登记（version_manifest / registry / spec 三角刷新 + 四审计重跑 + 三哈希闭环）。
  新增检验的**先登记判据**：
  · 分布匹配重加权的两种实现（重要性权重 WLS 与分位匹配变换）**只作诊断读数**；若任一实现使主读数转正
    且 bootstrap/置换支持 → 记为「分布口径待定线索」，否则保持「待定」；不据此改写 H8 撤回结论。
  · 置换检验：经验单侧 p = P(β_perm ≤ 观测 β)；p ≥ 0.05 → 「反号」不显著区别于零分布（表述只能用"未检出
    正号"）；MDE(80% 功效) = 2.8016 × se。
  · 功效：若 MDE(80%) > 原文效应 +0.152×2（或功效 < 20%）→ 报告必须声明「本窗口对原文量级功效不足」。

用法：
  python scripts/data_audit/s3_pc_sign_diag_h1h8.py --tests H1 H2
  python scripts/data_audit/s3_pc_sign_diag_h1h8.py --assemble
"""
from __future__ import annotations
# --- scripts/lib bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from lib.diag.consts import CLUSTER_DIR, CONTROLS, DIAG, FACTORS, NW, NW_PAPER, OUT_JSON, OUT_MD, PANEL, PAPER_CONTROLS, PAPER_T2, PAPER_TARGET_EFFECT, PC_DIR, PC_YEARS, PROGRESS, RETURNS, REV_PKL, ROOT, SEED, SW2_MAP, TARGETS, TIER_DIR
from lib.diag.util import _phi, dump, fmt_fm, pct, power_mde, psay, r4
from lib.diag.data import load_panel, load_pc, load_rev, merge_pc, sw2_asof
from lib.diag.stats import _ols_hac, ctrl_compare, fm, make_samples, monthly_factors, portfolio_key
from lib.diag.hypotheses import run_H1, run_H2, run_H3, run_H4, run_H5, run_H6, run_H7, run_H8
from lib.diag.report import assemble, make_md
# --- end scripts/lib bootstrap ---

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(ROOT / 'scripts/data_audit'))
from s3_m1_lib import (  # noqa: E402
    fm_regression, newey_west, param_value, rank_average_groups,
)

DIAG.mkdir(parents=True, exist_ok=True)
















# ---------------------------------------------------------------- 装载




















# ================================================================ H1


# ================================================================ H2




# ================================================================ H3


# ================================================================ H4


# ================================================================ H5


# ================================================================ H6




# ================================================================ H7


# ================================================================ H8


# ================================================================ 装配




def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tests', nargs='*', default=None)
    ap.add_argument('--assemble', action='store_true')
    ap.add_argument('--panel', default=str(PANEL))
    args = ap.parse_args()
    if args.assemble:
        assemble()
        return 0
    tests = [t.upper() for t in (args.tests or [])]
    panel = load_panel()
    pc = load_pc()
    m = merge_pc(panel, pc)
    S = make_samples(m)
    fmonth = monthly_factors()
    for t in tests:
        if t == 'H1':
            run_H1(m, S, fmonth)
        elif t == 'H2':
            run_H2(m, S, fmonth, panel)
        elif t == 'H3':
            run_H3(panel, fmonth)
        elif t == 'H4':
            run_H4(m, S, panel, fmonth)
        elif t == 'H5':
            run_H5(m, S, fmonth)
        elif t == 'H6':
            run_H6(m, S, fmonth)
        elif t == 'H7':
            run_H7(m, S, fmonth, panel)
        elif t == 'H8':
            run_H8(m, S, panel, fmonth)
        else:
            print(f'unknown test {t}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())