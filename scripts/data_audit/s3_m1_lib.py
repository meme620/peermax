#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M1（S5）月度定价共享库（Phase A / C2，2026-10-10）。

只提供纯计算与装载工具；不含任何 outcome 相关的口径选择（口径一律来自 registry：
alpha_m1 / K_Econ_M1 / K_Spread_M1 / K_Consist_M1 / nw_lag / winsorization_pct /
portfolio_breakpoints / random_seed / sample_start / sample_end / m1_sample_and_spec_registry /
m1_formula_ids）。

关键约定（先登记后跑，见 configs/parameter_registry.yaml）：
  - PIT：max{available_at < month}（面板既有映射，严格小于）
  - outcome：紧邻下一自然月 R，complete_month=1（n_valid_days>=15），不跨缺失月
  - winsor：winsorization_pct 月度截面（收益类变量；当前 registry [0.01,0.99]）
  - 样本窗口：sample_start/sample_end 截月（当前 registry 2015-01/2023-12）
  - FM：逐月 OLS/WLS -> 系数序列 -> 均值 + Newey-West(lag=6, Bartlett) SE
  - 单侧 p：H0 beta>=0 vs H1 beta<0（主方向 beta<0）时 p = Phi(t)
  - TOST：d = beta_EW - beta_VW，等价界 = +/- K_Consist_M1，alpha = alpha_m1
"""
from __future__ import annotations
# --- scripts/lib bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from lib.m1.paths import AMOUNT_CACHE_COLUMNS, AMOUNT_CACHE_DIR, AMOUNT_CACHE_META_NAME, AMOUNT_CACHE_META_REQUIRED, AMOUNT_CACHE_NAME, AMOUNT_PKL, ANALYST_DERIVED, CHARS, HARD_GATE_KEYS, ILLIQ_FORMULA_ID, IQR_POOL_REGISTERED, IQR_POOL_REGISTERED_SOURCE, IQR_POOL_RULE_ID, LEGACY_AMOUNT_CACHE_NAME, M1_INPUTS, M1_SAMPLE_VERSION, M1_SAMPLE_VERSION_V1, M1_SAMPLE_VERSION_V2, M1_SAMPLE_VERSION_V3, M1_SAMPLE_VERSION_V4, M1_SAMPLE_VERSION_V4_RENORM, MANIFEST, MDA_PANEL, ME_PANEL, NET_DIRS, OUT_DIR, PANELS, PANEL_COLS, PANEL_COLS_V4, PANEL_PATHS_V4, PROGRESS, REGISTRY, RETURNS_POSTADJ, RETURNS_V2, ROOT, ROWS_NOT_APPLICABLE, SAMPLE_DIR, SAMPLE_MONTHS, SPEC, SRC_MONTH, ST_PKL, SW2, TURNOVER_PKL, _M1_RETURN_COLS, _MONTH_RE, _SAMPLE_EXCLUSION_VALUE_COL, _SHA256_HEX
from lib.m1.contracts import _contract_entries, _is_zero_int, _manifest_input_anchor, _manifest_optional_anchor, _read_registry_doc, _registry_fail, _registry_month, _registry_param, anchor_sha, assert_anchored_file_sha, assert_m1_contract, assert_m1_inputs_sha, assert_named_m1_input_sha, contract_from_frame, load_manifest_entries, load_registry, load_spec_input_hashes, manifest_anchor_sha, manifest_input_key, param_value, params, require, sample_window_bounds
from lib.m1.stats import _ols, clip_by_month, fm_regression, joint_wald, newey_west, norm_cdf, nw_cov_matrix, one_sided_p, rank_average_groups, rcs_basis, tost_equivalence, treatment_only_mask, two_sided_p, vif_table, winsor_bounds, winsorize_bounds_by_month, winsorize_by_month
from lib.m1.data import _cache_contract_problems, _cache_fail, _check_amount_frame, _drive_monthly_wide, _read_amount_cache_meta, _rel_path, _write_json_atomic, _write_parquet_atomic, add_weight_rules, aggregate_edges, aggregate_stat, amount_cache_meta, code6, drive_code_to_firm_id, load_amount_100m, load_amount_100m_cached, load_analyst_derived, load_chars, load_edges, load_illiq, load_me, load_panel, load_postadj, load_returns_raw, load_returns_v2, load_src_month, load_st, load_support, load_sw2, load_turnover_drive, progress, sample_path, sha256_file, write_amount_cache, write_json
# --- end scripts/lib bootstrap ---


import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd


# W1 bundle（P1.2-01）：v4 面板路径（新文件名；不覆盖 v3）。默认行为仍走 PANELS。

# ---------------------------------------------------------------- I-14 amount 缓存契约
# 缓存名必须与内容一致：本缓存只存 firm_id/month/amount_100m（Drive 日成交额千元当月求和 / 1e5），
# **不含 illiq 列**。下游（s3_m1_1）事件月实际公式：illiq = |own_R_t| / amount_100m。
# 既有 registry/formula 登记名 ILLIQ_MONTHLY_AMIHUD_CNY100M_v1 保持不动（冻结登记），
# 本常量仅用于报告/缓存元数据澄清实际公式，不新增 Amihud 公式、不改数值。

# 样本版本（Phase A 复核修复批 2026-10-10）：
#   v1 = m1_s5_v1_2015_2023：未施加 microcap 筛除、ma/mb 未施加 sample 窗口过滤 →
#        status = superseded_registered_deviation（保留作「未施加 microcap 筛除」敏感性）
#   v2 = m1_s5_v2_2015_2023：microcap + 次新股筛除；2026-11-15 P1-01 起标 superseded
#        （旧 complete_month 用价格非空日数，非有效交易日）
#   v3 = m1_s5_v3_20261115：headline（P1-01：complete_month/n_valid_days = volume>0 交易日 >=15）
# W1 bundle 版本占位：v4 样本只在 W2 创建；W1 仅让 sample_path/CLI 可解析（默认行为不变）。

# ---------------------------------------------------------------- registry 参数绑定（Phase 1 I-5）
# 本块的注册表读取放在模块级常量之前：SAMPLE_WINDOW_* / winsor_bounds() 均以 registry 为准，
# 导入即 fail-closed（registry 缺失、参数缺失/重复、值域非法 → SystemExit(2)）。
# registry_path 仅作测试注入；生产调用一律用 configs/parameter_registry.yaml。














# sample 窗口（registry sample_start/sample_end，事件月口径；当前 2015-01 / 2023-12）
SAMPLE_WINDOW_START, SAMPLE_WINDOW_END = sample_window_bounds()

# I4：IQR_pool 登记值（treatment-only 宇宙、单一池化、全规格共用）。
# registry 只有规则 ID（m1_sample_and_spec_registry.iqr_scale_rule_id =
# IQR_POOL_TREATMENT_ONLY_v1）与规则正文，**没有数值字段**；故数值保持硬编码，
# 并在报告中登记来源，不猜参数名。outcome-blind 事前冻结，不按 v1/v2 结果调整。









# ---------------------------------------------------------------- 基础工具












# ---------------------------------------------------------------- M1 主输入锚（Phase 1 I-3）
# 9 个 M1 主输入（以 s3_m1_1 / s3_m1_lib 实际读取路径为准；market_equity_panel_full
# 为 market_equity_panel 的字节相同孪生件，按方案一并纳入锚账本；postadj 为 s3_m1_1
# 读取的后复权敏感性收益面板，R1 复核后加入）。
# 锚只读 configs/version_manifest.yaml 的 active + 64-hex sha256；缺条目/缺 sha/
# sha 不匹配一律 SystemExit(2)，禁止回退实测值或自动刷新 manifest。






















# ---------------------------------------------------------------- 输入契约硬门（Phase 1 I-2）
# 四条硬门（字段名 == s3_m1_sample_report.json:contracts.<column> 的实际键名）：
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

# W1 bundle（P1.2-01 A1 + D-08 B）v4 面板列集（43 + PeerMAXraw_renorm/formula_id_max_renorm）






















































# ---------------------------------------------------------------- 边表与权重变体








