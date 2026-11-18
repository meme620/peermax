"""Phase 3.4（N-1 / N-8 / N-11 推断层 shadow）：团聚类 SE / 有效对手数分层 / 双保持零分布 / PC 团-年推断。

范围（`artifacts/phase3/3.0_plan.md` §2.4；审计 N-1 / N-8 / N-11）
------------------------------------------------------------------
**只读 headline**（M1 样本 parquet、headline 边表/聚类/`{y}_pc.csv`、`source_month_argmax_panel.csv`、
`monthly_returns_clean.csv`、`monthly_pricing_panel_ma_teacher.csv`、registry 参数），
**只写** `artifacts/phase3/3.4_*`；不重跑 headline M1/网络/PC、不写 G0、不改 configs/reports/scripts。

三项子任务
----------
N-1（团规模 × 精度异质性）
  1. 有效对手数 `1/Σw²`（逐年 + 2019 分位；headline 登记值 p10..p90 = 2.00/13.67/44.58/90.16/168.31）；
  2. 分层 FM（主规格）：层基 = `n_sources`（审计口径 deg 1–2/3–10/11–50/51–150/151+）**与**
     有效对手数分箱两套；每套 n_rows 合计必须 = 181,178（FM 行数），否则停机；
  3. `PeerMAX` 方差分解（面板口径，与审计同定义）：组 = (month, 文本年, 团)，报组间/组内 SS 份额、
     按团规模分箱的组内 sd、100+ 家团行占比（面板 29.74%）与 FM `n_sources≥151` 行占比（16.16%）；
  4. 推断层 (a)/(b)/(c) 并报（点估计一律 = (a)，差 >1e-10 停机）：
     (a) headline：逐月横截面 OLS → β 序列 → NW(6)；
     (b) 月内**团聚类**稳健协方差（cluster = `cluster_id`）：月内抽样 SE + 与 (a) 同结构的
         hybrid（`sqrt(se_nw² + se_cluster² − se_iid_sampling²)`，即「跨月 NW × 月内聚类设计效应」）；
         另报团跨月的团聚类 SE（含同团跨月相关）；
     (c) 年块：按文本年 9 块做块聚类 SE + 块 bootstrap CI；
     (c′) 年块置换：把「文本年 → 月块」的映射随机置换（用 π(y) 年的网络算该块的 treatment），R 可配，
         报真网络 β 的经验 p 与分位（与 N-8 同引擎、不同零模型）。

N-8（双保持重连零分布）
  逐年对 headline 边集做 **industry-stratified swap-MCMC**：保持 (i) 每 focal 的 out-degree、
  (ii) 每 source 的 in-degree、(iii) 每 focal 的 **peer 行业多重集**（默认 as-of = 分类生效日口径）；
  交换 (i,j),(i',j') → (i,j'),(i',j)，仅当 `ind(j)=ind(j')`、`j'∉P_i`、`j∉P_{i'}`（提议对称 → 接受率 1）。
  权重口径 = `travel`（权重随边槽迁移，逐 focal 权重向量不变）；`resim` 为可选口径（需全相似度矩阵，默认不跑）。
  R 可配（主 200 / 增量 1000），每 rep 用同一 FM 规格（主规格 EW/VW，控制集不变）重算 β，
  报真网络 β 的零分布分位与经验 p、moved-edge share（信息量判据：中位 <10% → 停机改设计）。
  逐 rep 写盘 + `--append-reps` 续跑（rep 的 rng = f(seed, rep, year)，与执行顺序无关，可复现）。

N-11（PC 团-年常数）
  1. 逐年唯一 PC 值（登记值 198/217/273/271/235/258，合计 1,452）与「团-年内 PC 常数」逐团断言；
  2. PC 推断三层并报：(a) firm-month NW(6)（登记基准 −0.5068%/t=−1.4365）、(b) 月内团聚类、
     (c) 年块（6 块）聚类 SE + 块 bootstrap + 年块置换 p；另复算登记 T5B「团层等权 FM」
     (+0.2152% / t=+0.5815) 作交叉对账。

产物（全部在 `artifacts/phase3/`）
---------------------------------
  3.4_inference.csv            主推断长表（N-1 分层/方差/推断 + N-11 PC 三层）
  3.4_struct_layers.csv        N-1 分层 FM 读数（两套层基 × EW/VW）
  3.4_variance_decomposition.csv  N-1 方差分解 + 组内 sd 分箱 + 覆盖占比
  3.4_pc_inference.csv         N-11 PC 推断三层 + 逐年唯一值表
  3.4_placebo_null.csv         N-8 零分布逐 rep（含 identity rep 与 moved-edge share）
  3.4_null_summary.json        N-8 零分布汇总（分位/经验 p/信息量判据/子样本漂移）
  3.4_blockperm_reps.csv       N-1 (c′) 年块置换逐 rep
  3.4_counts.json              全部读数 + 对账 + 判据 + 锚台账
  3.4_anchor_pre.json / 3.4_anchor_post.json   9 锚 + headline 45 文件 sha 台账
  3.4_output_sha.json          本次确定性产物 sha
  3.4_run.log                  运行日志（stdout tee）

用法
----
  python scripts/shadow_inference.py --steps all
  python scripts/shadow_inference.py --steps n1,n11          # 快件
  python scripts/shadow_inference.py --steps n8 --reps 200   # 主零分布
  python scripts/shadow_inference.py --steps n8 --reps 1000 --append-reps  # 夜间增量
  python scripts/shadow_inference.py --steps n8perm --reps 200"""

from __future__ import annotations

import sys
from pathlib import Path
from fileio import json_default as _json_default

import argparse
import csv
import json
import math
import os
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from pricing_lib import (  # noqa: E402
    assert_m1_inputs_sha, fm_regression, newey_west, param_value, require, sha256_file,
    two_sided_p, winsor_bounds, _ols,
)
from peermax_panel_build import (  # noqa: E402
    ARGMAX, EXCLUDE_QUAL, RETURNS, load_versions, select_version,
)
from fileio import assert_path_allowed  # noqa: E402



ROOT = Path(__file__).resolve().parents[1]

ART = ROOT / 'artifacts/phase3'

HEADLINE_NET = ROOT / 'data/processed/network_ma_teacher'

SAMPLE_PARQUET = ROOT / 'data/processed/m1_common_sample_ma_teacher_v2.parquet'

PRICING_PANEL = ROOT / 'data/processed/monthly_pricing_panel_ma_teacher.csv'

ASOF_MAP = ROOT / 'data/processed/phase3_shadow/sw2_asof_map_p3shadow.parquet'   # 3.1 交接（只读）

SW2_LATEST = ROOT / 'data/processed/sw2_industry_latest.csv'

REGISTRY = ROOT / 'configs/parameter_registry.yaml'

INF_CSV = ART / '3.4_inference.csv'

STRUCT_CSV = ART / '3.4_struct_layers.csv'

VAR_CSV = ART / '3.4_variance_decomposition.csv'

PC_CSV = ART / '3.4_pc_inference.csv'

NULL_CSV = ART / '3.4_placebo_null.csv'

NULL_SUMMARY = ART / '3.4_null_summary.json'

BLOCKPERM_CSV = ART / '3.4_blockperm_reps.csv'

COUNTS_JSON = ART / '3.4_counts.json'

ANCHOR_PRE = ART / '3.4_anchor_pre.json'

ANCHOR_POST = ART / '3.4_anchor_post.json'

OUTPUT_SHA = ART / '3.4_output_sha.json'

RUN_LOG = ART / '3.4_run.log'

SHADOW_TAG = 'p3shadow_v1_inference'

YEARS = list(range(2014, 2023))          # 文本年（网络年 y+1）

PC_YEARS = list(range(2017, 2023))       # PC 可得文本年（y−1 年营收 ∈ 面板）

DEFAULT_SEED = 20261026                  # 3.0_plan §2.4 登记 seed

CLUSTER_SENTINEL = -1                    # 无团标签（不应出现；出现即报数）

CONTROLS = ['size', 'bm', 'mom', 'rev', 'turnover', 'illiq', 'AnalystCoverage', 'NetworkDegree']

MAIN_X = ['PeerMAXraw', 'OwnMAXraw', 'PeerReturn'] + CONTROLS

HEADLINE_EW = {'beta': 0.034273, 'se_nw': 0.072488, 't_nw': 0.473, 'tol': 1e-6}

HEADLINE_VW = {'beta': -0.109956, 'se_nw': 0.093089, 't_nw': -1.181, 'tol': 1e-5}

FM_ROWS = 181178

SAMPLE_ROWS = 307503

FM_MONTHS = 104

EFF_2019 = {'p10': 2.00, 'p25': 13.67, 'p50': 44.58, 'p75': 90.16, 'p90': 168.31}

EFF_TOP1 = {'gt_0.5': 0.1075, 'gt_0.8': 0.0758}

STRATA_DEG = [(1, 2), (3, 10), (11, 50), (51, 150), (151, 10 ** 9)]

STRATA_DEG_REG = {  # 审计已复算值（R2 对账，容差 ±0.001 / se ±0.0005）
    (1, 2): (+0.123506, 0.055649, +2.219), (3, 10): (-0.124150, 0.079347, -1.565),
    (11, 50): (+0.006517, 0.107414, +0.061), (51, 150): (+0.231120, 0.216836, +1.066),
    (151, 10 ** 9): (-0.400241, 1.539316, -0.260)}

STRATA_EFF = [(1, 2), (2, 10), (10, 50), (50, 150), (150, 10 ** 9)]   # 右开区间（1≤x<2 …）

VD_REG = {'between': 0.8948, 'within': 0.1052,
          'within_sd': {'2-5': 0.01577, '6-20': 0.00360, '21-50': 0.00147,
                        '51-100': 0.00111, '100+': 0.00070},
          'rows_100plus_share_panel': 0.2974, 'fm_nsources_151plus_share': 0.1616}

PC_UNIQUE_REG = {2017: 198, 2018: 217, 2019: 273, 2020: 271, 2021: 235, 2022: 258}

PC_UNIQUE_TOTAL_REG = 1452

PC_BASE_REG = {'beta_pct': -0.5068, 't_nw': -1.4365, 'label': '登记基准 EW lag6 单变量（对表读数）'}

PC_T5B_REG = {'beta_pct': +0.2152, 't_nw': +0.5815, 'label': 'T5/T5B 团层等权 FM 单变量'}

OLD_PLACEBO = {'beta': -0.061998, 't_nw': -2.1395, 'model': 'M1_RANDOM_DENSITY_NETWORK_v1（单次，只保 out-degree）'}

MOVED_SHARE_GATE = 0.10                  # 3.0_plan §2.4 停机条件：中位 moved-edge share < 10% → 停机

PCT_POINT_TOL = 1e-10                    # (b)/(c) 点估计 vs (a) 容差

NW_LAG = int(param_value('nw_lag'))      # registry headline_value = 6

M1_REG = param_value('m1_sample_and_spec_registry')

MIN_N = M1_REG['min_focal_count_per_month']

WINSOR = winsor_bounds()

REWIRE_REPS_REG = param_value('rewiring_reps')      # registry headline_value = 1000（登记值，本脚本不改）

REWIRE_SEED_REG = param_value('rewiring_seed')

ALLOWED_ROOTS = (ART,)

BANNED_ROOTS = (
    HEADLINE_NET, ROOT / 'data', ROOT / 'reports', ROOT / 'configs', ROOT / 'scripts',
    ROOT / 'tests', ROOT / 'docs', ROOT / 'deliverables',
    ROOT / 'data/processed/network_ma', ROOT / 'data/processed/network_mb',
    ROOT / 'data/processed/phase2_shadow',
)

_TEE = None

NULL_COLS = ['rep', 'identity', 'industry', 'weight_mode', 'swaps_per_edge', 'seed',
             'moved_edge_share', 'accept_rate_mean', 'beta_EW', 'se_EW', 't_EW',
             'n_rows_EW', 'n_months_EW', 'beta_VW', 'se_VW', 't_VW', 'n_rows_VW', 'n_months_VW',
             'elapsed_sec', 'moved_edge_share_by_year', 'accepted_swaps']

PERM_COLS = ['rep', 'seed', 'is_identity_perm', 'beta_EW', 'se_EW', 't_EW', 'n_rows_EW',
             'beta_VW', 'se_VW', 't_VW', 'n_rows_VW', 'elapsed_sec', 'perm']

def assert_output_path_guard(path) -> Path:
    """任何写出路径必须落在 artifacts/phase3/，且不命中 headline/G0/configs/data 等禁写集合。"""
    return assert_path_allowed(path, ALLOWED_ROOTS, BANNED_ROOTS)

def log(msg: str) -> None:
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    if _TEE is not None:
        _TEE.write(line + '\n')
    else:
        print(line, flush=True)

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

def write_json(path, obj) -> None:
    p = assert_output_path_guard(path)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=_json_default),
                   encoding='utf-8')
    os.replace(tmp, p)

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

def read_csv_rows(path, header):
    p = Path(path)
    if not p.is_file():
        return []
    out = []
    with open(p, encoding='utf-8', newline='') as fh:
        r = csv.reader(fh)
        got = next(r, None)
        if got != list(header):
            return []
        for row in r:
            out.append(row)
    return out

def r6(x):
    if x is None:
        return None
    v = float(x)
    return round(v, 6) if np.isfinite(v) else None

def sha_ledger_headline_net() -> dict:
    out = {}
    for p in sorted(HEADLINE_NET.rglob('*')):
        if p.is_file():
            out[p.relative_to(ROOT).as_posix()] = sha256_file(p)
    return out

def anchors_snapshot(tag: str) -> dict:
    ledger = assert_m1_inputs_sha()
    snap = {
        'tag': tag, 'run_at': datetime.now(timezone.utc).isoformat(),
        'script': 'scripts/shadow_inference.py',
        'm1_inputs_9': {r['input']: r['sha256'] for r in ledger},
        'm1_inputs_ledger': ledger,
        'headline_network_sha': sha_ledger_headline_net(),
    }
    snap['headline_network_n_files'] = len(snap['headline_network_sha'])
    return snap

def compare_anchors(pre: dict, post: dict) -> dict:
    diffs = []
    for sect in ('m1_inputs_9', 'headline_network_sha'):
        a, b = pre.get(sect) or {}, post.get(sect) or {}
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                diffs.append({'section': sect, 'key': k, 'pre': a.get(k), 'post': b.get(k)})
    return {'status': 'MATCH' if not diffs else 'CHANGED', 'n_diff': len(diffs), 'diffs': diffs,
            'n_m1_inputs': len(pre.get('m1_inputs_9') or {}),
            'n_headline_files': len(pre.get('headline_network_sha') or {})}

def fm_cluster_scores(df: pd.DataFrame, ycol: str, xcols, weight_col=None,
                      cluster_col='cid', month_col='month', min_n=None):
    """逐月横截面 OLS（与 `fm_regression` 同口径）→ β 序列 + 每团的得分 ψ_gm + 月度 iid 协方差。

    ψ_gm = (X_m'W_mX_m)^{-1} X_{g,m}'W_m e_{g,m}（得分函数），满足 Σ_g ψ_gm = β_m，
    因此 Σ_g ψ_gm 是 β_m 的团聚类稳健 meat 的"平方根"（月度抽样误差口径）。
    """
    min_n = MIN_N if min_n is None else min_n
    j = list(xcols).index('PeerMAXraw') if 'PeerMAXraw' in xcols else 0
    has_ty = 'ty' in df.columns
    per = []
    for m, sub in df.groupby(month_col, sort=True):
        sub = sub.dropna(subset=[ycol] + list(xcols))
        if weight_col:
            sub = sub[sub[weight_col].notna() & (sub[weight_col] > 0)]
        n = len(sub)
        if n < min_n:
            per.append({'month': m, 'n': n, 'used': False})
            continue
        X = np.column_stack([sub[c].to_numpy(dtype=float) for c in xcols] + [np.ones(n)])
        y = sub[ycol].to_numpy(dtype=float)
        w = sub[weight_col].to_numpy(dtype=float) if weight_col else None
        fit = _ols(X, y, w)
        if fit is None or not fit['finite']:
            per.append({'month': m, 'n': n, 'used': False})
            continue
        sw = np.sqrt(w) if w is not None else np.ones(n)
        Xw, yw = X * sw[:, None], y * sw
        resid = yw - Xw @ fit['beta']
        bread = np.linalg.pinv(Xw.T @ Xw)
        # 团键必须全局唯一：团 id 逐年重编号 → 用 (文本年 × 10⁶ + 团) 编码（否则跨月并成「伪团」）
        cid_v = sub[cluster_col].to_numpy(dtype=np.int64)
        keycode = (sub['ty'].to_numpy(dtype=np.int64) * 10 ** 6 + cid_v) if has_ty else cid_v
        g, uniq_keys = pd.factorize(keycode)
        psi = {}
        for gi in np.unique(g):
            sel = (g == gi)
            psi[int(uniq_keys[gi])] = bread @ (Xw[sel].T @ resid[sel])
        per.append({'month': m, 'n': n, 'used': True, 'beta': fit['beta'][j],
                    'beta_full': fit['beta'], 'cov_iid': fit['cov'], 'xidx': j,
                    'psi': {k: float(v[j]) for k, v in psi.items()},
                    'n_clusters': int(len(np.unique(g))), 'resid': None})
    return per

def _scale_rec(rec: dict, scale: float) -> dict:
    """把 β/se/CI/分位按 `scale` 换算（如小数收益 → %）；t 与比值不变。"""
    if scale == 1.0:
        return rec
    out = {}
    for k, v in rec.items():
        scale_me = (k in ('beta', 'beta_min', 'beta_max') or k.startswith('se')
                    or k.endswith('_se') or k.startswith('ci95')
                    or k.startswith('c_block_boot_ci'))
        if scale_me and isinstance(v, (int, float)) and v is not None:
            out[k] = r6(v * scale)
        else:
            out[k] = v
    out['scale_applied'] = scale
    return out

def inference_recipes(per, month_ty: dict, weight: str, scale: float = 1.0) -> dict:
    """(a) NW(6) / (b) 月内团聚类 / (b-hybrid) / (b2) 团跨月 / (c) 年块 五套读数（点估计同一）。"""
    used = [d for d in per if d['used']]
    require(used, 'FM 无可用月份（per-month 全部 used=False）')
    betas = np.array([d['beta'] for d in used])
    T = len(betas)
    beta_bar = float(betas.mean())
    nw = newey_west(betas, lag=NW_LAG)
    j = used[0]['xidx']
    se_samp_iid = math.sqrt(sum(float(d['cov_iid'][j, j]) for d in used)) / T
    s_cl = 0.0
    cross: dict = {}
    for d in used:
        for g, v in d['psi'].items():
            s_cl += v * v
            cross[g] = cross.get(g, 0.0) + v
    se_samp_cl = math.sqrt(s_cl) / T
    se_cross = math.sqrt(sum(v * v for v in cross.values())) / T
    hybrid = math.sqrt(max(nw['se_nw'] ** 2 + se_samp_cl ** 2 - se_samp_iid ** 2, 0.0))
    # 年块（文本年）块聚类 SE + 块 bootstrap
    blocks: dict = {}
    for d in used:
        blocks.setdefault(month_ty[d['month']], []).append(d['beta'])
    G = len(blocks)
    sb = sum((float(np.sum(np.asarray(v) - beta_bar)) ** 2) for v in blocks.values())
    se_block = math.sqrt(sb / T ** 2 * (G / (G - 1.0))) if G > 1 else None
    rng = np.random.default_rng(20261026)
    n_units = np.array([len(v) for v in blocks.values()], dtype=float)
    bmeans = np.array([float(np.mean(v)) for v in blocks.values()])
    draws = np.empty(2000)
    idx = rng.integers(0, G, size=(2000, G))
    for i in range(2000):
        wgt = n_units[idx[i]]
        draws[i] = float((bmeans[idx[i]] * wgt).sum() / wgt.sum())
    out = {
        'T': T, 'beta': beta_bar, 'n_clusters_total': len(cross), 'G_blocks': G,
        'beta_min': float(betas.min()), 'beta_max': float(betas.max()),
        'se_nw6': r6(nw['se_nw']), 't_nw6': r6(nw['t_nw']), 'se_ts_iid': r6(nw['se_iid']),
        'p_nw6_two_sided': r6(two_sided_p(nw['t_nw'])),
        'se_sampling_iid': r6(se_samp_iid),
        'se_sampling_cluster_within_month': r6(se_samp_cl),
        'deff_within_month': r6(se_samp_cl / se_samp_iid),
        'b_cluster_within_month_t': r6(beta_bar / se_samp_cl),
        'b_cluster_hybrid_se': r6(hybrid), 'b_cluster_hybrid_t': r6(beta_bar / hybrid),
        'b_cluster_hybrid_ratio_vs_a': r6(hybrid / nw['se_nw']),
        'b_cluster_cross_month_se': r6(se_cross), 'b_cluster_cross_month_t': r6(beta_bar / se_cross),
        'c_year_block_se': r6(se_block), 'c_year_block_t': r6(beta_bar / se_block if se_block else None),
        'c_year_block_ratio_vs_a': r6(se_block / nw['se_nw'] if se_block else None),
        'c_block_boot_se': r6(float(draws.std(ddof=1))),
        'c_block_boot_ci_low': r6(float(np.quantile(draws, 0.025))),
        'c_block_boot_ci_high': r6(float(np.quantile(draws, 0.975))),
        'weight': weight,
    }
    out['ci95_nw_low'] = r6(beta_bar - 1.96 * nw['se_nw'])
    out['ci95_nw_high'] = r6(beta_bar + 1.96 * nw['se_nw'])
    if se_block:
        out['ci95_year_block_low'] = r6(beta_bar - 1.96 * se_block)
        out['ci95_year_block_high'] = r6(beta_bar + 1.96 * se_block)
    return _scale_rec(out, scale)

def eff_opponents_from_edges(e: pd.DataFrame, nodes: list) -> tuple:
    """单年边表 → (有效对手数 eff=1/Σw², top1 权重, out-degree, weight_sum)。

    纯函数（供 N-1 与测试共用）；`nodes` 给出 focal 索引域（未出现的 focal 不返回）。
    """
    fi = pd.Categorical(e['focal_id'], categories=list(nodes)).codes.astype(np.int64)
    require(int(fi.min()) >= 0 if len(fi) else True, 'eff_opponents：边表含未知 focal')
    w = e['weight'].to_numpy(dtype=float)
    n = len(nodes)
    s2 = np.bincount(fi, weights=w * w, minlength=n)
    cnt = np.bincount(fi, minlength=n)
    wsum = np.bincount(fi, weights=w, minlength=n)
    top1 = np.zeros(n)
    np.maximum.at(top1, fi, w)
    sel = cnt > 0
    eff = pd.Series(1.0 / s2[sel], index=np.array(nodes, dtype=object)[sel])
    return eff, top1[sel], cnt[sel], wsum[sel]

def eff_opponents(inp: Inputs) -> tuple:
    """逐年 1/Σw²（= 有效对手数）、top1 权重占比；返回 (per_year, lookup:{y:{firm:eff}})。"""
    per_year, lookup = [], {}
    for y in YEARS:
        e = inp.edges[y]
        eff, top1, cnt, wsum = eff_opponents_from_edges(e, inp.nodes)
        lookup[y] = eff.to_dict()
        eff_arr = eff.to_numpy()
        q = eff.quantile([.1, .25, .5, .75, .9])
        per_year.append({
            'text_year': y, 'n_focal_with_edges': int(len(eff)),
            'eff_p10': r6(q.loc[.1]), 'eff_p25': r6(q.loc[.25]), 'eff_p50': r6(q.loc[.5]),
            'eff_p75': r6(q.loc[.75]), 'eff_p90': r6(q.loc[.9]),
            'eff_max': r6(float(eff_arr.max())),
            'eff_median_n_edges': r6(float(np.median(cnt))),
            'top1_gt_0.5_share': r6(float((top1 > 0.5).mean())),
            'top1_gt_0.8_share': r6(float((top1 > 0.8).mean())),
            'top1_eq_1_share': r6(float((top1 >= 1.0 - 1e-12).mean())),
            'weight_sum_dev_max': r6(float(np.abs(wsum - 1.0).max())),
        })
    return per_year, lookup

def strata_fm(fm: pd.DataFrame, basis: str, bins, weights, right_open: bool = False) -> tuple:
    """按 `basis` 分箱跑主规格 FM（点估计/SE/t）。返回 (rows, sum_check)。

    `right_open=True`（连续型层基，如有效对手数）：`lo ≤ x < hi`（分箱无缝隙、无重复）；
    `right_open=False`（整数层基，如 `n_sources`）：`lo ≤ x ≤ hi`（与审计口径逐值一致）。
    """
    rows, tot = [], 0
    for lo, hi in bins:
        if right_open:
            mask = (fm[basis] >= lo) & (fm[basis] < hi)
            lab = f'{lo}-{hi}' if hi < 10 ** 9 else f'{lo}+'
        else:
            mask = (fm[basis] >= lo) & (fm[basis] <= hi)
            lab = f'{lo}-{hi}' if hi < 10 ** 9 else f'{lo}+'
        sub = fm[mask]
        tot += len(sub)
        for weight, wcol in weights:
            d = {
                'basis': basis, 'layer': lab, 'weight': weight,
                'n_rows': int(len(sub)), 'beta': None, 'se': None, 't': None,
                'n_months': 0, 'p_two_sided': None, 'note': None,
            }
            if len(sub) == 0:
                d['note'] = 'EMPTY_STRATUM_NO_ROWS（不填 0、不外推）'
                rows.append(d)
                continue
            res, _ = fm_regression(sub, 'R_next', MAIN_X, weight_col=wcol, min_n=MIN_N, nw_lag=NW_LAG)
            c = res['coefs']['PeerMAXraw']
            d.update({'beta': r6(c['beta']), 'se': r6(c['se_nw']), 't': r6(c['t_nw']),
                      'n_months': int(res['n_months_used']), 'p_two_sided': r6(c['p_two_sided']),
                      'ci95_low': r6(c.get('ci95_low')), 'ci95_high': r6(c.get('ci95_high'))})
            rows.append(d)
    return rows, tot

def variance_decomposition(inp: Inputs) -> tuple:
    """面板口径方差分解（与审计同定义）：组 = (month, 文本年, 团)。"""
    p = inp.panel_frame()
    cid = np.full(len(p), CLUSTER_SENTINEL, dtype=np.int64)
    ty = p['ty'].to_numpy()
    foc = p['focal_id'].to_numpy()
    for y in YEARS:
        sel = np.flatnonzero(ty == y)
        if sel.size == 0:
            continue
        mapper = inp.cluster[y]
        cid[sel] = np.array([mapper.get(f, CLUSTER_SENTINEL) for f in foc[sel]], dtype=np.int64)
    p['cid'] = cid
    p['gkey'] = list(zip(p['month'], p['ty'], p['cid']))
    gm = p.groupby('gkey')['PeerMAXraw'].transform('mean')
    tot = float(((p['PeerMAXraw'] - p['PeerMAXraw'].mean()) ** 2).sum())
    wit = float(((p['PeerMAXraw'] - gm) ** 2).sum())
    gg = p.groupby('gkey').agg(sd=('PeerMAXraw', 'std'), n=('PeerMAXraw', 'size')).reset_index()
    csize = {}
    for y in YEARS:
        for k, v in inp.cluster_size[y].items():
            csize[(y, k)] = v
    gg['csize'] = [csize.get((t, c), np.nan) for _, t, c in gg['gkey']]
    rows = [{'block': 'variance_decomposition', 'item': 'ss_share_between', 'value': r6(1 - wit / tot),
             'reference': VD_REG['between'], 'n_rows': int(len(p)), 'n_groups': int(len(gg)),
             'note': '组 = (month, 文本年, 团)；SS 分解（审计 89.48/10.52）'},
            {'block': 'variance_decomposition', 'item': 'ss_share_within', 'value': r6(wit / tot),
             'reference': VD_REG['within'], 'n_rows': int(len(p)), 'n_groups': int(len(gg)),
             'note': '同上'}]
    for lo, hi, lab in [(2, 5, '2-5'), (6, 20, '6-20'), (21, 50, '21-50'),
                        (51, 100, '51-100'), (101, 10 ** 9, '100+')]:
        s = gg[(gg['csize'] >= lo) & (gg['csize'] <= hi)]
        rows.append({'block': 'within_sd_by_cluster_size', 'item': f'mean_within_sd_{lab}',
                     'value': r6(float(s['sd'].mean())), 'reference': VD_REG['within_sd'][lab],
                     'n_rows': int(len(s)), 'n_groups': int(len(s)),
                     'note': f'团规模 {lab} 的月×团组内 sd 未加权均值（组数 {len(s)}）'})
    rows_100 = int(gg.loc[gg['csize'] >= 101, 'n'].sum())
    share_panel = rows_100 / len(p)
    rows.append({'block': 'coverage', 'item': 'rows_in_100plus_cluster_share_panel',
                 'value': r6(share_panel), 'reference': VD_REG['rows_100plus_share_panel'],
                 'n_rows': int(len(p)), 'n_groups': None,
                 'note': '面板 PeerMAX 行中位于 100+ 家团的占比（审计 29.74%）'})
    fm_sh = float((inp.sample.loc[inp.fm_mask, 'n_sources'] >= 151).mean())
    rows.append({'block': 'coverage', 'item': 'fm_rows_nsources_151plus_share',
                 'value': r6(fm_sh), 'reference': VD_REG['fm_nsources_151plus_share'],
                 'n_rows': int(inp.fm_mask.sum()), 'n_groups': None,
                 'note': 'FM 样本 n_sources≥151 行占比（审计 16.16%）'})
    return rows, {'n_panel_rows': int(len(p)), 'n_groups': int(len(gg)),
                  'ss_between': r6(1 - wit / tot), 'ss_within': r6(wit / tot)}

def n1_inference(inp: Inputs, eff_lookup: dict) -> dict:
    fm = inp.fm_frame()
    fm['eff_opponents'] = [eff_lookup.get(int(y), {}).get(f, np.nan)
                           for f, y in zip(fm['focal_id'], fm['ty'])]
    corr = float(np.corrcoef(fm['eff_opponents'], fm['n_sources'])[0, 1])
    res = {'corr_eff_vs_nsources_in_fm': r6(corr),
           'n_fm_rows': int(len(fm)),
           'eff_quantiles_in_fm': {k: r6(float(v)) for k, v in
                                   fm['eff_opponents'].quantile([.1, .25, .5, .75, .9]).items()}}
    rows = []
    # 分层（两套层基 × EW/VW）
    weights = [('EW', None), ('VW', 'market_equity')]
    for basis in ('n_sources', 'eff_opponents'):
        if basis == 'eff_opponents':
            err = int(fm['eff_opponents'].isna().sum())
            require(err == 0, f'FM 样本有 {err} 行缺有效对手数；停机')
        sr, tot = strata_fm(fm, basis, STRATA_DEG if basis == 'n_sources' else STRATA_EFF,
                            weights, right_open=(basis == 'eff_opponents'))
        require(tot == FM_ROWS,
                f'分层（{basis}）n_rows 合计 {tot} != {FM_ROWS}；停机（3.0_plan §2.4 R1）')
        res[f'strata_sum_{basis}'] = tot
        for r in sr:
            rows.append({'block': f'struct_layer_{basis}', **r})
    # 审计口径对账（deg 层 EW，容差 ±0.001 / se ±0.0005）
    recon = {}
    for r in rows:
        if r['basis'] == 'n_sources' and r['weight'] == 'EW':
            key = None
            for lo, hi in STRATA_DEG:
                lab = f'{lo}-{hi}' if hi < 10 ** 9 else f'{lo}+'
                if r['layer'] == lab:
                    key = (lo, hi)
            if key is None or r['beta'] is None:
                continue
            reg = STRATA_DEG_REG[key]
            ok = (abs(r['beta'] - reg[0]) <= 0.001 and abs(r['se'] - reg[1]) <= 0.0005
                  and abs(r['t'] - reg[2]) <= 0.01)
            recon[f'{r["layer"]}'] = {'measured': [r['beta'], r['se'], r['t']],
                                      'registered': list(reg), 'pass': bool(ok)}
    require(all(v['pass'] for v in recon.values()),
            f'N-1 分层读数与审计登记不符：{ {k: v for k, v in recon.items() if not v["pass"]} }；停机')
    res['strata_reconcile_vs_audit'] = recon
    # 推断层 (a)/(b)/(c)
    inf = {}
    for weight, wcol in weights:
        per = fm_cluster_scores(fm, 'R_next', MAIN_X, weight_col=wcol, cluster_col='cid')
        rec = inference_recipes(per, inp.month_ty, weight)
        reg = HEADLINE_EW if weight == 'EW' else HEADLINE_VW
        require(abs(rec['beta'] - reg['beta']) <= 0.001,
                f'({weight}) 点估计 {rec["beta"]} 偏离 headline {reg["beta"]}；停机')
        a = newey_west([d['beta'] for d in per if d['used']], lag=NW_LAG)
        require(abs(float(a['mean']) - rec['beta']) <= PCT_POINT_TOL,
                f'({weight}) (b)/(c) 点估计 ≠ (a)：{float(a["mean"])} vs {rec["beta"]}；'
                '停机（3.0_plan §2.4 R1）')
        inf[weight] = rec
    res['inference'] = inf
    # headline 逐值对账
    res['headline_check'] = {
        'EW': {'beta': inf['EW']['beta'], 'se_nw6': inf['EW']['se_nw6'], 't_nw6': inf['EW']['t_nw6'],
               'registered': HEADLINE_EW,
               'pass': bool(abs(inf['EW']['beta'] - HEADLINE_EW['beta']) <= HEADLINE_EW['tol']
                            and abs(inf['EW']['se_nw6'] - HEADLINE_EW['se_nw']) <= 1e-5
                            and abs(inf['EW']['t_nw6'] - HEADLINE_EW['t_nw']) <= 1e-3)},
        'VW': {'beta': inf['VW']['beta'], 'se_nw6': inf['VW']['se_nw6'], 't_nw6': inf['VW']['t_nw6'],
               'registered': HEADLINE_VW,
               'pass': bool(abs(inf['VW']['beta'] - HEADLINE_VW['beta']) <= HEADLINE_VW['tol']
                            and abs(inf['VW']['se_nw6'] - HEADLINE_VW['se_nw']) <= 1e-5
                            and abs(inf['VW']['t_nw6'] - HEADLINE_VW['t_nw']) <= 1e-3)},
    }
    require(res['headline_check']['EW']['pass'] and res['headline_check']['VW']['pass'],
            f'headline 逐值复现失败：{res["headline_check"]}；停机')
    return {'rows': rows, 'res': res}

def pc_cluster_year_constancy(pc_df: pd.DataFrame, clusters_df: pd.DataFrame,
                              text_year=None) -> dict:
    """「PC 在团-年内常数」逐团断言（纯函数，供 N-11 与测试共用）。

    `pc_df` 需含 `focal_id`/`pc`；`clusters_df` 需含 `firm_id`/`cluster_id`。
    返回：团-年组数、含多个 PC 值的组数（>0 即违反）、单点团数、每组 PC 唯一值数分布。
    """
    d = pc_df.rename(columns={'focal_id': 'firm_id'})[['firm_id', 'pc']]
    d = d[d['pc'].notna()]
    j = d.merge(clusters_df[['firm_id', 'cluster_id']], on='firm_id', how='left')
    n_no_cluster = int(j['cluster_id'].isna().sum())
    gg = j.groupby('cluster_id')['pc'].nunique()
    sizes = clusters_df.groupby('cluster_id').size()
    return {
        'text_year': text_year,
        'n_cluster_years': int(len(gg)),
        'n_with_multiple_pc': int((gg > 1).sum()),
        'n_multi_violation_firms': int(j[j['cluster_id'].isin(gg[gg > 1].index)].shape[0]),
        'n_singleton_clusters': int((sizes == 1).sum()),
        'n_firms_without_cluster': n_no_cluster,
        'n_rows_with_pc': int(len(j)),
    }

def pc_readings(inp: Inputs) -> dict:
    rows = []
    frames, uniq = [], {}
    constancy = {'n_cluster_years': 0, 'n_with_multiple_pc': 0, 'n_singleton_no_pc': 0}
    per_year_constancy = {}
    for y in PC_YEARS:
        d = pd.read_csv(HEADLINE_NET / f'{y}_pc.csv', dtype={'focal_id': str})
        frames.append(d)
        u = int(d['pc'].nunique())
        uniq[y] = u
        rows.append({'block': 'pc_unique_values', 'item': f'unique_pc_{y}', 'value': u,
                     'reference': PC_UNIQUE_REG[y], 'n_rows': int(len(d)),
                     'n_groups': int(d['pc'].notna().sum()),
                     'note': f'文本年 {y}：{len(d)} 行 / pc 非空 {int(d["pc"].notna().sum())} 行'})
        cl = pd.read_csv(HEADLINE_NET / 'clusters' / f'{y}_clusters.csv', dtype={'firm_id': str})
        cy = pc_cluster_year_constancy(d, cl, text_year=y)
        per_year_constancy[y] = cy
        constancy['n_cluster_years'] += cy['n_cluster_years']
        constancy['n_with_multiple_pc'] += cy['n_with_multiple_pc']
        constancy['n_singleton_no_pc'] += cy['n_singleton_clusters']
        rows.append({'block': 'pc_unique_values', 'item': f'cluster_years_{y}',
                     'value': cy['n_cluster_years'], 'reference': None,
                     'n_rows': cy['n_rows_with_pc'], 'n_groups': cy['n_cluster_years'],
                     'note': f'团-年组数 = {cy["n_cluster_years"]}；含多值的组 = '
                             f'{cy["n_with_multiple_pc"]}；单点团 = {cy["n_singleton_clusters"]}'})
    pc_all = pd.concat(frames, ignore_index=True)
    tot_uniq = sum(uniq.values())
    rows.append({'block': 'pc_unique_values', 'item': 'unique_pc_total_2017_2022', 'value': tot_uniq,
                 'reference': PC_UNIQUE_TOTAL_REG, 'n_rows': int(len(pc_all)), 'n_groups': None,
                 'note': '登记值 1,452（= 6 年唯一值合计）'})
    require(uniq == PC_UNIQUE_REG and tot_uniq == PC_UNIQUE_TOTAL_REG,
            f'PC 逐年唯一值 {uniq} 合计 {tot_uniq} != 登记 {PC_UNIQUE_REG} / {PC_UNIQUE_TOTAL_REG}；停机')
    require(constancy['n_with_multiple_pc'] == 0,
            f'PC 团-年常数被违反：{constancy["n_with_multiple_pc"]} 个团-年存在多个 PC 值；停机')
    pc = pc_all[pc_all['pc'].notna()][['focal_id', 'text_year', 'pc', 'hhi', 'n_competitors',
                                      'revenue_coverage']].copy()
    pc['text_year'] = pc['text_year'].astype(int)
    d = inp.sample[['focal_id', 'month', 'ty', 'cid', 'R_next', 'm1_in_fm_sample',
                    'market_equity'] + CONTROLS].copy()
    d['ty'] = d['ty'].astype('Int64')
    m = d.merge(pc, left_on=['focal_id', 'ty'], right_on=['focal_id', 'text_year'], how='left')
    samp = m[m['m1_in_fm_sample'].astype(bool) & m['R_next'].notna() & m['pc'].notna()].copy()
    samp['PC'] = samp['pc'].astype(float)      # 原始单位（PC ∈ [−1,0]）；报告时 ×100 转 %/单位 PC
    inf = {}
    per = fm_cluster_scores(samp, 'R_next', ['PC'], cluster_col='cid')
    inf['a_firm_level'] = inference_recipes(per, inp.month_ty, 'EW-univariate', scale=100.0)
    per_c = fm_cluster_scores(samp, 'R_next', ['PC'] + CONTROLS, cluster_col='cid')
    inf['a_firm_level_controls'] = inference_recipes(per_c, inp.month_ty, 'EW-controls', scale=100.0)
    # T5B 交叉对账：团层等权 FM（每月团均值 R_next ~ 团 PC）
    cl = samp.groupby(['month', 'cid'], sort=False).agg(
        R=('R_next', 'mean'), PC=('PC', 'first'), n=('R_next', 'size'),
        ty=('ty', 'first')).reset_index()
    cl = cl[cl['n'] > 0]
    per_cl = fm_cluster_scores(cl, 'R', ['PC'], cluster_col='cid', min_n=10)
    inf['b_cluster_equal_weight_T5B'] = inference_recipes(per_cl, inp.month_ty,
                                                          'cluster-equal-weight', scale=100.0)
    checks = {
        'univariate_vs_registered': {
            'measured': [inf['a_firm_level']['beta'], inf['a_firm_level']['t_nw6']],
            'registered': [PC_BASE_REG['beta_pct'], PC_BASE_REG['t_nw']],
            'note': PC_BASE_REG['label'],
            'pass': bool(abs(inf['a_firm_level']['beta'] - PC_BASE_REG['beta_pct']) <= 0.02
                         and abs(inf['a_firm_level']['t_nw6'] - PC_BASE_REG['t_nw']) <= 0.1)},
        'T5B_vs_registered': {
            'measured': [inf['b_cluster_equal_weight_T5B']['beta'],
                         inf['b_cluster_equal_weight_T5B']['t_nw6']],
            'registered': [PC_T5B_REG['beta_pct'], PC_T5B_REG['t_nw']],
            'note': PC_T5B_REG['label'],
            'pass': bool(abs(inf['b_cluster_equal_weight_T5B']['beta'] - PC_T5B_REG['beta_pct']) <= 0.02
                         and abs(inf['b_cluster_equal_weight_T5B']['t_nw6'] - PC_T5B_REG['t_nw']) <= 0.1)},
    }
    require(checks['univariate_vs_registered']['pass'],
            f'PC 单变量 FM 未复现登记基准：{checks["univariate_vs_registered"]}；停机')
    for k, v in inf.items():
        rows.append({'block': 'pc_inference', 'item': k, 'value': v['beta'],
                     'reference': None, 'n_rows': v['T'], 'n_groups': v['n_clusters_total'],
                     'note': f'se_nw6={v["se_nw6"]} t={v["t_nw6"]}；团聚类 hybrid se={v["b_cluster_hybrid_se"]} '
                             f't={v["b_cluster_hybrid_t"]}；年块 se={v["c_year_block_se"]} t={v["c_year_block_t"]}；'
                             f'deff(sp1)={v["deff_within_month"]}'})
    return {'rows': rows, 'uniq': {str(k): v for k, v in uniq.items()},
            'uniq_total': tot_uniq, 'constancy': constancy,
            'per_year_constancy': {str(k): v for k, v in per_year_constancy.items()},
            'inference': inf, 'checks': checks,
            'n_matched_rows': int(len(samp)), 'n_months': int(samp['month'].nunique()),
            'pc_share_of_variation': r6(float(1 - samp.groupby(['month', 'cid'])['PC']
                                              .transform('mean').var() / samp['PC'].var()))}

def pc_block_permutation(inp: Inputs, reps: int, seed: int) -> dict:
    """年块置换：把 6 个 PC 年块的值向量随机重指派到 6 个月块，报经验 p 与分位。

    终检修正：主读数改为 **size-adjusted 分位**。旧（变量样本）口径下 per-rep 样本量随置换年缺失
    而变化（n 101,728–120,352），且 corr(n,β)=−0.35，使登记基准的左尾分位（4.5%）被放大。
    调整：β_adj = β − slope×(n − n_true)，slope 由 β~n OLS 得到；参照 = 登记单变量 β%，
    n_true = 真 PC 样本（120,352）。未调整口径留档。置换语义/seed 不变（rng=[seed, r, 1111]）。
    """
    pc = {}
    for y in PC_YEARS:
        d = pd.read_csv(HEADLINE_NET / f'{y}_pc.csv', dtype={'focal_id': str},
                        usecols=['focal_id', 'text_year', 'pc'])
        d = d[d['pc'].notna()]
        pc[y] = dict(zip(d['focal_id'], d['pc'].astype(float)))
    d = inp.sample[['focal_id', 'month', 'ty', 'cid', 'R_next', 'm1_in_fm_sample']].copy()
    d = d[d['m1_in_fm_sample'].astype(bool) & d['R_next'].notna() & d['ty'].isin(PC_YEARS)]
    d['ty'] = d['ty'].astype(int)
    firms = d['focal_id'].to_numpy()
    tyv = d['ty'].to_numpy()
    true_vals = np.array([pc[y].get(f, np.nan) for f, y in zip(firms, tyv)], dtype=float)
    n_true = float(np.isfinite(true_vals).sum())       # 真 PC 匹配样本（≈120,352）
    betas, ns = [], []
    for r in range(reps):
        rng = np.random.default_rng([seed, r, 1111])
        perm = {int(y): int(z) for y, z in zip(PC_YEARS, rng.permutation(PC_YEARS))}
        vals = np.array([pc[perm[y]].get(f, np.nan) for f, y in zip(firms, tyv)], dtype=float)
        s = d.assign(PC=vals).dropna(subset=['PC'])
        res, _ = fm_regression(s, 'R_next', ['PC'], min_n=MIN_N, nw_lag=NW_LAG)
        c = res['coefs'].get('PC') or {}
        b = c.get('beta')
        if b is None:
            continue
        betas.append(float(b))
        ns.append(int(len(s)))
    arr = np.array(betas)
    n_arr = np.array(ns, dtype=float)
    base = PC_BASE_REG['beta_pct'] / 100.0
    pct_ = float((arr < base).mean())
    slope, _icpt = np.polyfit(n_arr, arr, 1)
    corr = float(np.corrcoef(n_arr, arr)[0, 1])
    adj = arr - slope * (n_arr - n_true)
    pct_adj = float((adj < base).mean())
    out = {'reps': int(len(arr)), 'seed': int(seed), 'sample_design': 'variable_sample_pc6y',
           'n_true': n_true,
           'beta_pct_quantiles': {f'p{q}': r6(float(np.quantile(arr, q / 100.0)) * 100.0)
                                  for q in (1, 5, 50, 95, 99)},
           'beta_pct_mean': r6(float(arr.mean()) * 100.0),
           'beta_pct_sd': r6(float(arr.std(ddof=1)) * 100.0),
           'base_beta_pct': PC_BASE_REG['beta_pct'],
           'n_rows_by_rep': [int(x) for x in n_arr],
           'size_adjusted': {'n_true': n_true, 'slope_beta_per_row': slope, 'corr_n_beta': corr,
                             'n_rows_min': int(n_arr.min()), 'n_rows_max': int(n_arr.max()),
                             'n_rows_unique': int(len(np.unique(n_arr))),
                             'percentile': r6(100.0 * pct_adj),
                             'empirical_p_two_sided': r6(min(1.0, 2 * min(pct_adj, 1 - pct_adj))),
                             'empirical_p_left': r6(pct_adj), 'empirical_p_right': r6(1 - pct_adj),
                             'note': 'β_adj = β − slope×(n − n_true)（n_true = 真 PC 样本 120,352）；'
                                     '调整后对登记基准 β（−0.5068%）计算分位/经验 p'},
           'base_percentile_unadjusted': r6(100.0 * pct_),
           'empirical_p_two_sided_unadjusted': r6(min(1.0, 2 * min(pct_, 1 - pct_))),
           'empirical_p_left_unadjusted': r6(pct_),
           'base_percentile': r6(100.0 * pct_adj),        # 主读数 = size-adjusted
           'empirical_p_two_sided': r6(min(1.0, 2 * min(pct_adj, 1 - pct_adj))),
           'empirical_p_left': r6(pct_adj),
           'note': '年块置换（6 块）；主读数 = size-adjusted（n_true=120,352）；未调整口径留档；'
                   '基准 = 登记单变量 β（见 base_beta_pct）'}
    return out

def null_summary(recs, true_ew: float, true_vw: float, reps_target: int) -> dict:
    arr = np.array([r['beta_EW'] for r in recs if r.get('beta_EW') is not None], dtype=float)
    arr_vw = np.array([r['beta_VW'] for r in recs if r.get('beta_VW') is not None], dtype=float)
    moved = np.array([r['moved_edge_share'] for r in recs if r.get('moved_edge_share') is not None],
                     dtype=float)

    def pct(v, x):
        return float((v < x).mean())

    out = {
        'reps_target': int(reps_target), 'reps_done': int(len(arr)),
        'beta_EW_true': r6(true_ew), 'beta_VW_true': r6(true_vw),
        'null_beta_EW': {'mean': r6(float(arr.mean())), 'sd': r6(float(arr.std(ddof=1))),
                         'min': r6(float(arr.min())), 'max': r6(float(arr.max())),
                         'quantiles': {f'p{q}': r6(float(np.quantile(arr, q / 100.0)))
                                       for q in (1, 5, 50, 95, 99)}},
        'null_beta_VW': {'mean': r6(float(arr_vw.mean())), 'sd': r6(float(arr_vw.std(ddof=1))),
                         'quantiles': {f'p{q}': r6(float(np.quantile(arr_vw, q / 100.0)))
                                       for q in (1, 5, 50, 95, 99)}},
        'moved_edge_share': {'mean': r6(float(moved.mean())), 'median': r6(float(np.median(moved))),
                             'min': r6(float(moved.min())), 'max': r6(float(moved.max()))},
    }
    out['beta_EW_true_percentile'] = r6(100.0 * pct(arr, true_ew))
    out['empirical_p_two_sided_EW'] = r6(min(1.0, 2 * min(pct(arr, true_ew), 1 - pct(arr, true_ew))))
    out['empirical_p_right_EW'] = r6(1 - pct(arr, true_ew))       # H1: 真网络 β > 零分布
    out['empirical_p_left_EW'] = r6(pct(arr, true_ew))            # H1: 真网络 β < 零分布
    out['beta_VW_true_percentile'] = r6(100.0 * pct(arr_vw, true_vw))
    out['empirical_p_two_sided_VW'] = r6(min(1.0, 2 * min(pct(arr_vw, true_vw),
                                                          1 - pct(arr_vw, true_vw))))
    out['in_null_95_interval_EW'] = bool(np.quantile(arr, 0.025) <= true_ew <= np.quantile(arr, 0.975))
    out['old_single_placebo'] = OLD_PLACEBO
    out['old_placebo_in_null_95_interval_EW'] = bool(
        np.quantile(arr, 0.025) <= OLD_PLACEBO['beta'] <= np.quantile(arr, 0.975))
    # 子样本漂移（3.0_plan §2.4 R3：R=200→1000 分位漂移 ≤10%）
    drift = {}
    for tag, sub in (('first100', arr[:100]), ('first200', arr[:200]), ('all', arr)):
        if len(sub) == 0:
            continue
        drift[tag] = {f'p{q}': r6(float(np.quantile(sub, q / 100.0))) for q in (5, 50, 95)}
    if 'all' in drift and len(arr) >= 200:
        base = drift['first200'] if len(arr) > 250 else drift['all']
        comp = drift['all'] if len(arr) > 250 else drift['first100']
        dd = max(abs(comp[k] - base[k]) / max(abs(base[k]), 1e-9) for k in base if base[k])
        drift['max_rel_drift_200_vs_all'] = r6(dd)
    out['quantile_drift'] = drift
    out['moved_share_gate'] = {'threshold': MOVED_SHARE_GATE,
                               'status': 'PASS' if float(np.median(moved)) >= MOVED_SHARE_GATE else 'FAIL',
                               'note': '中位 moved-edge share < 10% → 零分布近乎原地，停机改设计（3.0_plan §2.4）'}
    out['null_has_information'] = bool(float(np.median(moved)) >= MOVED_SHARE_GATE
                                       and float(arr.std(ddof=1)) > 0)
    return out

def step_n1(inp: Inputs, counts: dict) -> dict:
    t0 = time.time()
    eff_year, eff_lookup = eff_opponents(inp)
    reg19 = eff_year[[d['text_year'] for d in eff_year].index(2019)]
    eff_check = {
        'p10..p90_2019': [reg19['eff_p10'], reg19['eff_p25'], reg19['eff_p50'],
                          reg19['eff_p75'], reg19['eff_p90']],
        'registered': EFF_2019,
        'top1': {'gt_0.5': reg19['top1_gt_0.5_share'], 'gt_0.8': reg19['top1_gt_0.8_share']},
        'registered_top1': EFF_TOP1,
        'pass': bool(all(abs(a - b) <= 0.01 for a, b in
                         zip([reg19['eff_p10'], reg19['eff_p25'], reg19['eff_p50'],
                              reg19['eff_p75'], reg19['eff_p90']],
                             EFF_2019.values()))
                     and abs(reg19['top1_gt_0.5_share'] - EFF_TOP1['gt_0.5']) <= 0.002
                     and abs(reg19['top1_gt_0.8_share'] - EFF_TOP1['gt_0.8']) <= 0.002),
    }
    require(eff_check['pass'], f'2019 有效对手数未复现登记值：{eff_check}；停机')
    var_rows, var_res = variance_decomposition(inp)
    n1 = n1_inference(inp, eff_lookup)
    rows = []
    for d in eff_year:
        rows.append({'block': 'eff_opponents_by_year', 'item': f'text_year_{d["text_year"]}',
                     'value': d['eff_p50'], 'reference': None, 'n_rows': d['n_focal_with_edges'],
                     'n_groups': None, 'note': json.dumps(d, ensure_ascii=False, default=_json_default)})
    rows += n1['rows']
    inf_rows = []
    for weight in ('EW', 'VW'):
        v = n1['res']['inference'][weight]
        inf_rows.append({'block': 'n1_inference', 'item': f'PeerMAX_{weight}', 'value': v['beta'],
                         'reference': None, 'n_rows': v['T'], 'n_groups': v['n_clusters_total'],
                         'note': json.dumps(v, ensure_ascii=False, default=_json_default)})
    res = {'eff_by_year': eff_year, 'eff_check_2019': eff_check,
           'variance_rows': var_rows, 'variance': var_res,
           'n1_rows': rows, 'inference_rows': inf_rows,
           'inference': n1['res'],
           'elapsed_sec': r6(time.time() - t0)}
    log(f'N-1 完成：eff 2019 p10-p90={eff_check["p10..p90_2019"]}；'
        f'between/within={var_res["ss_between"]}/{var_res["ss_within"]}；'
        f'EW β={n1["res"]["inference"]["EW"]["beta"]} '
        f'se_NW={n1["res"]["inference"]["EW"]["se_nw6"]} '
        f'se_cluster_hybrid={n1["res"]["inference"]["EW"]["b_cluster_hybrid_se"]} '
        f'se_year_block={n1["res"]["inference"]["EW"]["c_year_block_se"]}')
    return res

def step_n11(inp: Inputs, counts: dict, perm_reps: int, seed: int) -> dict:
    t0 = time.time()
    pc = pc_readings(inp)
    pc['block_perm'] = pc_block_permutation(inp, perm_reps, seed)
    log(f'N-11 完成：唯一值 {pc["uniq"]}（合计 {pc["uniq_total"]}）；'
        f'(a) β%={pc["inference"]["a_firm_level"]["beta"]} '
        f't={pc["inference"]["a_firm_level"]["t_nw6"]}；'
        f'T5B β%={pc["inference"]["b_cluster_equal_weight_T5B"]["beta"]} '
        f't={pc["inference"]["b_cluster_equal_weight_T5B"]["t_nw6"]}；'
        f'年块 se={pc["inference"]["a_firm_level"]["c_year_block_se"]}')
    pc['elapsed_sec'] = r6(time.time() - t0)
    return pc

def step_n8(inp: Inputs, counts: dict, reps: int, seed: int, industry: str,
            weight_mode: str, spe: float, append: bool) -> dict:
    t0 = time.time()
    eng = RewireEngine(inp, industry, weight_mode, spe)
    # identity rep（未交换网络）→ 必须逐值复现 headline（R1 引擎保真门）
    ident = eng.run_rep(-1, seed, identity=True)
    require(ident['beta_EW'] is not None and abs(ident['beta_EW'] - HEADLINE_EW['beta']) <= 1e-6,
            f'identity rep β={ident["beta_EW"]} != headline {HEADLINE_EW["beta"]}；'
            '零分布引擎不保真，停机')
    log(f'N-8 引擎保真门 PASS：identity rep β_EW={ident["beta_EW"]}（headline {HEADLINE_EW["beta"]}）')
    done = {}
    if append:
        for row in read_csv_rows(NULL_CSV, NULL_COLS):
            try:
                done[int(row[0])] = row
            except (TypeError, ValueError):
                continue
    rows = [dict(zip(NULL_COLS, r)) for r in done.values()]
    if -1 not in done:                                  # identity rep 一并留档（不参与零分布）
        rows.append({c: (json.dumps(ident[c], ensure_ascii=False, default=_json_default)
                         if isinstance(ident.get(c), dict) else ident.get(c)) for c in NULL_COLS})
    todo = [r for r in range(reps) if r not in done]
    log(f'N-8：目标 R={reps}（已存在 {len(done)} 个 rep，本次补跑 {len(todo)} 个；'
        f'seed={seed}；swaps/edge={spe}；industry={industry}；weight={weight_mode}）')
    for k, r in enumerate(todo, 1):
        rec = eng.run_rep(r, seed)
        row = {c: (json.dumps(rec[c], ensure_ascii=False, default=_json_default)
                   if isinstance(rec.get(c), dict) else rec.get(c)) for c in NULL_COLS}
        rows.append(row)
        write_csv(NULL_CSV, [[r.get(c) for c in NULL_COLS] for r in sorted(
            rows, key=lambda d: int(d['rep']))], NULL_COLS)   # 逐 rep 原子落盘（中断不丢已完成 rep）
        if k % 10 == 0 or k == len(todo):
            log(f'  进度 {k}/{len(todo)}：β_EW={rec["beta_EW"]} t={rec["t_EW"]} '
                f'moved={rec["moved_edge_share"]} ({rec["elapsed_sec"]}s/rep)')
    rows = sorted(rows, key=lambda d: int(d['rep']))
    write_csv(NULL_CSV, [[r.get(c) for c in NULL_COLS] for r in rows], NULL_COLS)
    recs = []
    for r in rows:
        d = dict(r)
        for c in ('rep', 'identity', 'seed'):
            d[c] = int(float(d[c])) if d.get(c) not in (None, '') else None
        for c in ('swaps_per_edge', 'moved_edge_share', 'accept_rate_mean', 'beta_EW', 'se_EW',
                  't_EW', 'n_rows_EW', 'n_months_EW', 'beta_VW', 'se_VW', 't_VW',
                  'n_rows_VW', 'n_months_VW', 'elapsed_sec'):
            d[c] = float(d[c]) if d.get(c) not in (None, '') else None
        d['moved_edge_share_by_year'] = json.loads(d['moved_edge_share_by_year']) \
            if d.get('moved_edge_share_by_year') else {}
        d['accepted_swaps'] = json.loads(d['accepted_swaps']) if d.get('accepted_swaps') else {}
        recs.append(d)
    real = [d for d in recs if d['rep'] >= 0]
    summary = null_summary(real, HEADLINE_EW['beta'], HEADLINE_VW['beta'], reps)
    summary['identity_rep'] = ident
    summary['n_reps_total'] = len(real)
    summary['swaps_per_edge'] = spe
    summary['industry_basis'] = industry
    summary['weight_mode'] = weight_mode
    summary['seed'] = int(seed)
    summary['registry_rewiring_reps'] = REWIRE_REPS_REG
    summary['registry_rewiring_seed'] = REWIRE_SEED_REG
    summary['registered_scope'] = ('registry `rewiring_reps` scope = degree- and industry-preserving '
                                   'network rewiring；本脚本按该口径实现（保 out-degree / in-degree / '
                                   'per-focal 行业多重集），权重口径见 weight_mode')
    summary['elapsed_sec'] = r6(time.time() - t0)
    write_json(NULL_SUMMARY, summary)
    log(f'N-8 完成：R={summary["reps_done"]}；null β_EW 中位 '
        f'{summary["null_beta_EW"]["quantiles"]["p50"]}；真网络分位 '
        f'{summary["beta_EW_true_percentile"]}%；经验 p(双侧)={summary["empirical_p_two_sided_EW"]}；'
        f'moved 中位={summary["moved_edge_share"]["median"]}')
    if summary['moved_share_gate']['status'] == 'FAIL':
        print('[HARD-GATE] moved-edge share 中位 < 10%：零分布近乎原地，按 3.0_plan §2.4 停机改设计',
              file=sys.stderr, flush=True)
        raise SystemExit(2)
    return summary

def step_n8perm(inp: Inputs, counts: dict, reps: int, seed: int, industry: str,
                append: bool) -> dict:
    """N-1 (c′) 年块（文本年）置换零分布：π 置换 9 个文本年 → treatment 取自 π(y) 年网络。

    终检修正：主读数改为 **size-adjusted 分位**。旧（变量样本）口径下 per-rep n_rows 与 β 正相关
    （n 144k–173k、corr=+0.24），真网络（n=181,178）分位被系统抬高（86.0 → 调整后 52.5）。
    调整：β_adj = β − slope×(n − n_true)，slope 由 β~n OLS 得到；分位/经验 p 以 β_adj 对真网络 β 计算。
    identity π 保真门仍用全样本（逐值复现 headline）。置换语义/seed 不变。
    """
    t0 = time.time()
    eng = RewireEngine(inp, industry, 'travel', 0.0)
    ident = eng.run_perm_rep(-1, seed)
    require(ident['beta_EW'] is not None and abs(ident['beta_EW'] - HEADLINE_EW['beta']) <= 1e-6,
            f'年块置换 identity π β={ident["beta_EW"]} != headline {HEADLINE_EW["beta"]}；停机')
    done = {}
    if append:
        for row in read_csv_rows(BLOCKPERM_CSV, PERM_COLS):
            try:
                done[int(row[0])] = row
            except (TypeError, ValueError):
                continue
    rows = [dict(zip(PERM_COLS, r)) for r in done.values()]
    if -1 not in done:                                  # identity rep 留档（全样本保真门，不入零分布）
        rows.append({c: (json.dumps(ident[c], ensure_ascii=False, default=_json_default)
                         if isinstance(ident.get(c), dict) else ident.get(c)) for c in PERM_COLS})
    todo = [r for r in range(reps) if r not in done]
    log(f'N-1 (c′) 年块置换：目标 R={reps}（已存在 {len(done)}，本次 {len(todo)}）')
    for k, r in enumerate(todo, 1):
        rec = eng.run_perm_rep(r, seed)
        row = {c: (json.dumps(rec[c], ensure_ascii=False, default=_json_default)
                   if isinstance(rec.get(c), dict) else rec.get(c)) for c in PERM_COLS}
        rows.append(row)
        write_csv(BLOCKPERM_CSV, [[r.get(c) for c in PERM_COLS] for r in sorted(
            rows, key=lambda d: int(d['rep']))], PERM_COLS)   # 逐 rep 原子落盘（中断不丢已完成 rep）
        if k % 10 == 0 or k == len(todo):
            log(f'  进度 {k}/{len(todo)}：β_EW={rec["beta_EW"]} t={rec["t_EW"]}')
    rows = sorted(rows, key=lambda d: int(d['rep']))
    write_csv(BLOCKPERM_CSV, [[r.get(c) for c in PERM_COLS] for r in rows], PERM_COLS)
    arr = np.array([float(r['beta_EW']) for r in rows
                    if r.get('beta_EW') not in (None, '') and int(r['rep']) >= 0], dtype=float)
    n_arr = np.array([float(r['n_rows_EW']) for r in rows
                      if r.get('n_rows_EW') not in (None, '') and int(r['rep']) >= 0], dtype=float)
    true = HEADLINE_EW['beta']
    pct = float((arr < true).mean())
    n_true = float(FM_ROWS)
    slope, _icpt = np.polyfit(n_arr, arr, 1)
    corr = float(np.corrcoef(n_arr, arr)[0, 1])
    adj = arr - slope * (n_arr - n_true)
    pct_adj = float((adj < true).mean())
    out = {'reps_done': int(len(arr)), 'reps_target': int(reps), 'seed': int(seed),
           'sample_design': 'variable_sample_full_fm',
           'beta_EW_true': true,
           'null_quantiles': {f'p{q}': r6(float(np.quantile(arr, q / 100.0))) for q in (1, 5, 50, 95, 99)},
           'null_mean': r6(float(arr.mean())), 'null_sd': r6(float(arr.std(ddof=1))),
           'true_percentile_unadjusted': r6(100.0 * pct),
           'empirical_p_two_sided_unadjusted': r6(min(1.0, 2 * min(pct, 1 - pct))),
           'true_percentile': r6(100.0 * pct_adj),          # 主读数 = size-adjusted
           'empirical_p_two_sided': r6(min(1.0, 2 * min(pct_adj, 1 - pct_adj))),
           'empirical_p_right': r6(1 - pct_adj), 'empirical_p_left': r6(pct_adj),
           'size_adjusted': {'n_true': n_true, 'slope_beta_per_row': slope, 'corr_n_beta': corr,
                             'n_rows_min': int(n_arr.min()), 'n_rows_max': int(n_arr.max()),
                             'n_rows_unique': int(len(np.unique(n_arr))),
                             'percentile': r6(100.0 * pct_adj),
                             'empirical_p_two_sided': r6(min(1.0, 2 * min(pct_adj, 1 - pct_adj))),
                             'empirical_p_left': r6(pct_adj), 'empirical_p_right': r6(1 - pct_adj),
                             'note': 'β_adj = β − slope×(n − n_true)（n_true = FM 行数 181,178）；'
                                     '调整后对真网络 β 计算分位/经验 p'},
           'identity_perm': ident, 'elapsed_sec': r6(time.time() - t0)}
    log(f'N-1 (c′) 完成：n_rows {int(n_arr.min()):,}–{int(n_arr.max()):,}，corr(n,β)={corr:.3f}；'
        f'真网络分位 未调整={out["true_percentile_unadjusted"]}% → size-adjusted={out["true_percentile"]}%；'
        f'经验 p(双侧)={out["empirical_p_two_sided"]}；null sd={out["null_sd"]}')
    return out

def step_sha(inp: Inputs) -> dict:
    files = [INF_CSV, STRUCT_CSV, VAR_CSV, PC_CSV, NULL_CSV, NULL_SUMMARY, BLOCKPERM_CSV]
    cur = {Path(p).relative_to(ROOT).as_posix(): sha256_file(p) for p in files if Path(p).is_file()}
    res = {'n_files': len(cur), 'sha': cur,
           'excluded_from_byte_compare': ['3.4_counts.json', '3.4_anchor_pre.json',
                                          '3.4_anchor_post.json', '3.4_run.log']}
    if OUTPUT_SHA.is_file():
        prev = json.loads(OUTPUT_SHA.read_text(encoding='utf-8'))
        ps = prev.get('sha') or {}
        diff = sorted(k for k in set(ps) | set(cur) if ps.get(k) != cur.get(k))
        res['rerun_check'] = {'status': 'IDENTICAL' if not diff else 'DIFFERENT',
                              'n_diff': len(diff), 'diff_keys': diff}
    write_json(OUTPUT_SHA, res)
    log(f'sha 台账：{res["n_files"]} 个确定性产物'
        + (f'；重跑比对 {res["rerun_check"]["status"]}（diff={res["rerun_check"]["n_diff"]}）'
           if 'rerun_check' in res else '（首次）'))
    return res

def _long_header():
    return ['block', 'basis', 'layer', 'weight', 'item', 'value', 'reference', 'se', 't',
            'ci95_low', 'ci95_high', 'p_two_sided', 'n_rows', 'n_months', 'n_groups', 'note']

def write_long_tables(counts: dict) -> None:
    """把各 step 的读数落成 CSV（幂等；缺哪些 step 就写哪些块）。"""
    rows = []

    def add(block, basis=None, layer=None, weight=None, item=None, value=None, reference=None,
            se=None, t=None, lo=None, hi=None, p=None, n_rows=None, n_months=None,
            n_groups=None, note=None):
        rows.append({'block': block, 'basis': basis, 'layer': layer, 'weight': weight,
                     'item': item, 'value': value, 'reference': reference, 'se': se, 't': t,
                     'ci95_low': lo, 'ci95_high': hi, 'p_two_sided': p, 'n_rows': n_rows,
                     'n_months': n_months, 'n_groups': n_groups, 'note': note})

    n1 = counts.get('n1') or {}
    for d in n1.get('eff_by_year') or []:
        add('eff_opponents_by_year', item=f'text_year_{d["text_year"]}', value=d['eff_p50'],
            n_rows=d['n_focal_with_edges'],
            note=f'p10={d["eff_p10"]} p25={d["eff_p25"]} p75={d["eff_p75"]} p90={d["eff_p90"]} '
                 f'max={d["eff_max"]} top1>0.5={d["top1_gt_0.5_share"]} top1>0.8={d["top1_gt_0.8_share"]}')
    for r in n1.get('n1_rows') or []:
        if r.get('block', '').startswith('struct_layer'):
            add(r['block'], basis=r.get('basis'), layer=r.get('layer'), weight=r.get('weight'),
                value=r.get('beta'), reference=json.dumps(r.get('reference') or ''),
                se=r.get('se'), t=r.get('t'), lo=r.get('ci95_low'), hi=r.get('ci95_high'),
                p=r.get('p_two_sided'), n_rows=r.get('n_rows'), n_months=r.get('n_months'),
                note=r.get('note'))
    var = n1.get('variance') or {}
    add('variance_summary', item='ss_between', value=var.get('ss_between'),
        reference=VD_REG['between'], n_rows=var.get('n_panel_rows'), n_groups=var.get('n_groups'),
        note='面板口径（month×文本年×团）')
    add('variance_summary', item='ss_within', value=var.get('ss_within'),
        reference=VD_REG['within'], n_rows=var.get('n_panel_rows'), n_groups=var.get('n_groups'))
    for r in n1.get('variance_rows') or []:
        add(r['block'], item=r['item'], value=r['value'], reference=r.get('reference'),
            n_rows=r.get('n_rows'), n_groups=r.get('n_groups'), note=r.get('note'))
    inf = ((n1.get('inference') or {}).get('inference') or {})
    for weight, v in inf.items():
        for est, se_key, t_key in (('a_nw6_headline', 'se_nw6', 't_nw6'),
                                   ('b_cluster_within_month_sampling', 'se_sampling_cluster_within_month', None),
                                   ('b_cluster_hybrid_vs_a', 'b_cluster_hybrid_se', 'b_cluster_hybrid_t'),
                                   ('b2_cluster_cross_month', 'b_cluster_cross_month_se', 'b_cluster_cross_month_t'),
                                   ('c_year_block', 'c_year_block_se', 'c_year_block_t'),
                                   ('c_year_block_boot', 'c_block_boot_se', None)):
            add('n1_inference', weight=weight, item=est, value=v['beta'], se=v.get(se_key),
                t=v.get(t_key), n_months=v.get('T'), n_groups=v.get('n_clusters_total'),
                note=f"G_blocks={v.get('G_blocks')} deff={v.get('deff_within_month')} "
                     f"ratio_vs_a={v.get('b_cluster_hybrid_ratio_vs_a') if 'hybrid' in est else v.get('c_year_block_ratio_vs_a')}")
        add('n1_inference', weight=weight, item='se_iid_ts', value=v['beta'], se=v.get('se_ts_iid'),
            n_months=v.get('T'), note='β 序列的时序 iid SE（fm_regression.se_iid）')
    bp = counts.get('n1_block_perm') or {}
    if bp:
        add('n1_year_block_permutation', item='beta_EW', value=bp.get('beta_EW_true'),
            se=bp.get('null_sd'), p=bp.get('empirical_p_two_sided'), n_rows=bp.get('reps_done'),
            note=f"design={bp.get('sample_design')}；size-adjusted 真网络分位={bp.get('true_percentile')}%"
                 f"（未调整 {bp.get('true_percentile_unadjusted')}%；corr(n,β)="
                 f"{(bp.get('size_adjusted') or {}).get('corr_n_beta')}）；"
                 f"null p5/p50/p95={bp['null_quantiles']['p5']}/{bp['null_quantiles']['p50']}/{bp['null_quantiles']['p95']}")
    n11 = counts.get('n11') or {}
    for r in n11.get('rows') or []:
        add(r['block'], item=r['item'], value=r['value'], reference=r.get('reference'),
            n_rows=r.get('n_rows'), n_groups=r.get('n_groups'), note=r.get('note'))
    for k, v in (n11.get('inference') or {}).items():
        for est, se_key, t_key in (('a_nw6_headline', 'se_nw6', 't_nw6'),
                                   ('b_cluster_hybrid_vs_a', 'b_cluster_hybrid_se', 'b_cluster_hybrid_t'),
                                   ('b2_cluster_cross_month', 'b_cluster_cross_month_se', 'b_cluster_cross_month_t'),
                                   ('c_year_block', 'c_year_block_se', 'c_year_block_t')):
            add('pc_inference_layers', basis='pct_per_unit_pc', item=f'{k}:{est}', value=v['beta'],
                se=v.get(se_key), t=v.get(t_key), n_months=v.get('T'),
                n_groups=v.get('n_clusters_total'),
                note=f"G_blocks={v.get('G_blocks')} deff={v.get('deff_within_month')}")
    n8 = counts.get('n8') or {}
    if n8:
        add('n8_null', item='true_beta_EW_percentile', value=n8.get('beta_EW_true_percentile'),
            p=n8.get('empirical_p_two_sided_EW'), n_rows=n8.get('reps_done'),
            note=f"null p1/p5/p50/p95/p99="
                 f"{n8['null_beta_EW']['quantiles']['p1']}/{n8['null_beta_EW']['quantiles']['p5']}/"
                 f"{n8['null_beta_EW']['quantiles']['p50']}/{n8['null_beta_EW']['quantiles']['p95']}/"
                 f"{n8['null_beta_EW']['quantiles']['p99']}；"
                 f"moved 中位={n8['moved_edge_share']['median']}；"
                 f"gate={n8['moved_share_gate']['status']}")
        add('n8_null', item='old_single_placebo_beta', value=OLD_PLACEBO['beta'],
            t=OLD_PLACEBO['t_nw'], note=OLD_PLACEBO['model'])
    write_csv(INF_CSV, [[r[c] for c in _long_header()] for r in rows], _long_header())
    # 专项表
    struct = [r for r in rows if r['block'].startswith('struct_layer')]
    if struct:
        write_csv(STRUCT_CSV, [[r[c] for c in _long_header()] for r in struct], _long_header())
    var_rows = [r for r in rows if r['block'] in ('variance_summary', 'within_sd_by_cluster_size',
                                                 'coverage')]
    if var_rows:
        write_csv(VAR_CSV, [[r[c] for c in _long_header()] for r in var_rows], _long_header())
    pc_rows = [r for r in rows if r['block'] in ('pc_unique_values', 'pc_inference_layers')]
    if pc_rows:
        write_csv(PC_CSV, [[r[c] for c in _long_header()] for r in pc_rows], _long_header())


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --- scripts/lib bootstrap ---
# --- end scripts/lib bootstrap ---






# ---------------------------------------------------------------- 路径与常量




# headline 登记读数（只读对账；`3.0_plan.md` §1.1 / 审计）





class _Tee:
    """stdout + `3.4_run.log` 双写（append）。"""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, 'a', encoding='utf-8', newline='\n')

    def write(self, s: str):
        sys.__stdout__.write(s)
        self.fh.write(s)
        self.fh.flush()

    def flush(self):
        sys.__stdout__.flush()
        self.fh.flush()
















# ---------------------------------------------------------------- 锚台账（9 锚 + headline 45 文件）






# ---------------------------------------------------------------- 只读输入装载
class Inputs:
    """headline 只读输入的一次性装载（样本 / 面板 / 边表 / 聚类 / 价值矩阵 / 行业标签）。"""

    def __init__(self, log_extra=True):
        self.vers = load_versions(HEADLINE_NET)
        require(sorted(self.vers) == YEARS, f'headline 网络年份 {sorted(self.vers)} != {YEARS}')
        self.avail = {y: self.vers[y]['available_at'] for y in YEARS}
        self.sample = pd.read_parquet(SAMPLE_PARQUET)
        require(len(self.sample) == SAMPLE_ROWS, f'样本行数 {len(self.sample)} != {SAMPLE_ROWS}')
        req_cols = ['focal_id', 'month', 'network_version', 'PeerMAXraw', 'OwnMAXraw', 'PeerReturn',
                    'R_next', 'm1_in_fm_sample', 'n_sources'] + CONTROLS
        miss = [c for c in req_cols if c not in self.sample.columns]
        require(not miss, f'样本缺列 {miss}')
        self.sample['ty'] = pd.to_numeric(self.sample['network_version'], errors='coerce')
        self.fm_mask = self.sample['m1_in_fm_sample'].astype(bool).to_numpy()
        require(int(self.fm_mask.sum()) == FM_ROWS, f'FM 行数 {int(self.fm_mask.sum())} != {FM_ROWS}')
        # month → 文本年（authoritative：available_at 规则；与面板 network_version 逐月对齐核验）
        months = sorted(self.sample.loc[self.sample['ty'].notna(), 'month'].unique())
        self.months = months
        self.month_ty = {m: int(select_version(self.vers, m)['text_year']) for m in months}
        chk = (self.sample.loc[self.sample['ty'].notna(), ['month', 'ty']].drop_duplicates()
               .groupby('month')['ty'].first())
        bad = [m for m in months if int(chk.get(m, -1)) != self.month_ty[m]]
        require(not bad, f'月度网络版本与 available_at 规则不一致：{bad[:5]}')
        # 团（headline 聚类）
        self.cluster = {}
        self.cluster_size = {}
        for y in YEARS:
            c = pd.read_csv(HEADLINE_NET / 'clusters' / f'{y}_clusters.csv', dtype={'firm_id': str})
            self.cluster[y] = dict(zip(c['firm_id'], c['cluster_id']))
            sizes = c.groupby('cluster_id').size()
            self.cluster_size[y] = {int(k): int(v) for k, v in sizes.items()}
        # 节点宇宙 + 边结构
        nodes = set(self.sample['focal_id'])
        self.edges = {}
        for y in YEARS:
            e = pd.read_csv(HEADLINE_NET / f'{y}_similarity.csv',
                            usecols=['focal_id', 'source_id', 'similarity', 'weight'],
                            dtype={'focal_id': str, 'source_id': str})
            self.edges[y] = e
            nodes |= set(e['source_id'])
        self.nodes = sorted(nodes)
        self.nidx = {f: i for i, f in enumerate(self.nodes)}
        self.n = len(self.nodes)
        # 价值矩阵（MAXraw / R）与资格掩码（与 s4_step2 面板口径逐项一致）
        maxraw, qual = {}, {}
        with open(ARGMAX, encoding='utf-8', newline='') as fh:
            for r in csv.DictReader(fh):
                maxraw.setdefault(r['month'], {})[r['source_id']] = float(r['MAXraw'])
                qual.setdefault(r['month'], {})[r['source_id']] = r['qualification_status']
        rmap = {}
        with open(RETURNS, encoding='utf-8', newline='') as fh:
            for r in csv.DictReader(fh):
                rmap.setdefault(r['month'], {})[r['firm_id']] = float(r['R'])
        n_m = len(months)
        self.Xm = np.full((self.n, n_m), np.nan)
        self.Xr = np.full((self.n, n_m), np.nan)
        for j, m in enumerate(months):
            mm, qq, rr = maxraw.get(m, {}), qual.get(m, {}), rmap.get(m, {})
            for f, i in self.nidx.items():
                if f in mm and qq.get(f) != EXCLUDE_QUAL:
                    self.Xm[i, j] = mm[f]
                if f in rr:
                    self.Xr[i, j] = rr[f]
        self.mpos = {m: j for j, m in enumerate(months)}
        # 样本行的 (月, 节点) 索引（加速逐 rep 组装）
        self.row_node = np.array([self.nidx[f] for f in self.sample['focal_id']], dtype=np.int64)
        self.month_rows = {m: np.flatnonzero(self.sample['month'].to_numpy() == m) for m in months}
        # 团 id 列（样本行 × 文本年）
        cid = np.full(len(self.sample), CLUSTER_SENTINEL, dtype=np.int64)
        ty = self.sample['ty'].to_numpy()
        foc = self.sample['focal_id'].to_numpy()
        for y in YEARS:
            sel = np.flatnonzero(ty == y)
            if sel.size == 0:
                continue
            mapper = self.cluster[y]
            cid[sel] = np.array([mapper.get(f, CLUSTER_SENTINEL) for f in foc[sel]], dtype=np.int64)
        self.sample['cid'] = cid
        # 行业标签（as-of 与 latest 两套）→ 逐文本年静态标签（键 = (y+1)-04-30）
        self.ind = self._industry_labels()
        if log_extra:
            log(f'输入装载：样本 {len(self.sample):,} 行 / FM {int(self.fm_mask.sum()):,} 行 / '
                f'{len(months)} 月 / 节点 {self.n:,} / 边 {sum(len(e) for e in self.edges.values()):,}')

    def _industry_labels(self) -> dict:
        """`{y: {firm_id: 行业码}}`：as-of（分类生效日口径）+ latest 两套。"""
        lat = pd.read_csv(SW2_LATEST, dtype={'firm_id': str, 'l2_code': str})
        lat = lat[lat['firm_id'].notna()]
        latest = dict(zip(lat['firm_id'], lat['l2_code']))
        asof = {y: {} for y in YEARS}
        src = 'asof_unavailable'
        if ASOF_MAP.is_file():
            m = pd.read_parquet(ASOF_MAP)
            m = m[['firm_id', 'month', 'effective_from', 'l2_code']].dropna(subset=['firm_id'])
            m['effective_from'] = pd.to_datetime(m['effective_from'])
            m = m.sort_values('effective_from', kind='stable')
            for y in YEARS:
                key = pd.Timestamp(f'{y + 1}-04-30')
                sub = m[m['effective_from'] <= key].drop_duplicates('firm_id', keep='last')
                asof[y] = dict(zip(sub['firm_id'], sub['l2_code']))
            src = 'phase3_shadow/sw2_asof_map_p3shadow.parquet（3.1 交接，只读）'
        out = {'asof': {}, 'latest': {}, 'source': src,
               'coverage': {}}
        for y in YEARS:
            a = {f: (asof[y].get(f) or latest.get(f) or 'NA') for f in self.nodes}
            l = {f: (latest.get(f) or 'NA') for f in self.nodes}
            out['asof'][y] = a
            out['latest'][y] = l
            out['coverage'][y] = {
                'asof_direct': int(sum(1 for f in self.nodes if f in asof[y])),
                'latest': int(sum(1 for f in self.nodes if f in latest)),
                'nodes': int(self.n)}
        return out

    def fm_frame(self) -> pd.DataFrame:
        cols = ['focal_id', 'month', 'ty', 'cid'] + MAIN_X + ['R_next', 'n_sources', 'market_equity']
        return self.sample.loc[self.fm_mask, cols].copy()

    def panel_frame(self) -> pd.DataFrame:
        d = pd.read_csv(PRICING_PANEL, dtype={'focal_id': str, 'month': str},
                        usecols=['focal_id', 'month', 'network_version', 'PeerMAXraw'])
        d = d[d['PeerMAXraw'].notna()].copy()
        d['ty'] = pd.to_numeric(d['network_version']).astype(int)
        return d


# ---------------------------------------------------------------- FM 推断机器






# ---------------------------------------------------------------- N-1 结构读数








# ---------------------------------------------------------------- N-1 推断


# ---------------------------------------------------------------- N-11 PC






# ---------------------------------------------------------------- N-8 双保持 swap-MCMC
class YearNet:
    """单文本年的二部图（focal × source）：swap-MCMC 所需的全部结构。"""

    def __init__(self, edges: pd.DataFrame, nodes: list, nidx: dict, ind: dict):
        self.n = len(nodes)
        self.fi = pd.Categorical(edges['focal_id'], categories=nodes).codes.astype(np.int64)
        self.si = pd.Categorical(edges['source_id'], categories=nodes).codes.astype(np.int64)
        require(int(self.fi.min()) >= 0 and int(self.si.min()) >= 0, '边表含未知节点')
        self.w = edges['weight'].to_numpy(dtype=float)
        self.peer = self.si.copy()
        self.adj = np.zeros((self.n, self.n), dtype=bool)
        self.adj[self.fi, self.si] = True
        self.orig_key = np.sort((self.fi << 20) | self.si)
        ind_code, _uniq = pd.factorize(np.array([ind[f] for f in nodes], dtype=object), sort=True)
        self.ind_code = ind_code.astype(np.int64)
        # 提议分布：按 peer 行业分组（组内均匀取两条边 → 提议对称）
        gi = pd.Categorical(self.ind_code[self.si]).codes.astype(np.int64)
        self.gi = gi
        self.G = int(gi.max()) + 1 if len(gi) else 0
        self.og = np.argsort(gi, kind='stable')
        cnt = np.bincount(gi[self.og], minlength=self.G)
        self.gcnt = cnt
        self.gstart = np.concatenate([[0], np.cumsum(cnt)])
        self.gp = cnt / cnt.sum()
        self.base_ind = self.ind_code[self.si].copy()

    def verify_invariants(self) -> dict:
        """三不变断言（逐 rep，向量化直方图口径）：
        out-degree（每 focal 边数）、in-degree（每 source 被选次数）、per-focal peer 行业多重集。"""
        n = self.n
        base_out = np.bincount(self.fi, minlength=n)
        cur_out = np.bincount(self.fi, minlength=n)          # 边数组长度不变 → 构造性保持
        base_in = np.bincount(self.si, minlength=n)
        cur_in = np.bincount(self.peer, minlength=n)
        n_ind = int(max(self.ind_code.max(), self.base_ind.max())) + 1
        b = np.bincount(self.fi * n_ind + self.base_ind, minlength=n * n_ind)
        c = np.bincount(self.fi * n_ind + self.ind_code[self.peer], minlength=n * n_ind)
        return {'out_degree_preserved': bool(np.array_equal(base_out, cur_out)),
                'in_degree_preserved': bool(np.array_equal(base_in, cur_in)),
                'per_focal_industry_multiset_preserved': bool(np.array_equal(b, c)),
                'n_focal_with_edges': int(base_out.sum()),
                'all_ok': bool(np.array_equal(base_out, cur_out)
                               and np.array_equal(base_in, cur_in) and np.array_equal(b, c))}

    def swap(self, target: int, rng: np.random.Generator, batch: int = 30000,
             progress_every: int = 0) -> dict:
        """接受 target 次合法交换（提议对称 → Metropolis 比 = 1）。"""
        acc = att = nb = 0
        n = self.n
        while acc < target:
            m = min(batch, target - acc + 10000)
            grp = rng.choice(self.G, size=m, p=self.gp)
            pos1 = self.gstart[grp] + (rng.random(m) * self.gcnt[grp]).astype(np.int64)
            pos2 = self.gstart[grp] + (rng.random(m) * self.gcnt[grp]).astype(np.int64)
            e1 = self.og[pos1]
            e2 = self.og[pos2]
            i1, i2 = self.fi[e1], self.fi[e2]
            j1, j2 = self.peer[e1], self.peer[e2]
            ok = (pos1 != pos2) & (i1 != i2) & (~self.adj[i1, j2]) & (~self.adj[i2, j1])
            ok &= (j2 != i1) & (j1 != i2)                  # 禁止自环（peer 不得为 focal 自身）
            idx = np.flatnonzero(ok)
            att += m
            need = target - acc
            if idx.size > need:            # 不超目标（逐 rep 接受数 = round(swaps_per_edge × 边数)）
                idx = idx[:need]
            used = np.zeros(n, dtype=bool)
            for k in idx:
                p, q = int(i1[k]), int(i2[k])
                if used[p] or used[q]:
                    continue
                used[p] = True
                used[q] = True
                a, b = int(j1[k]), int(j2[k])
                self.adj[p, a] = False
                self.adj[q, b] = False
                self.adj[p, b] = True
                self.adj[q, a] = True
                self.peer[e1[k]] = b
                self.peer[e2[k]] = a
                acc += 1
            nb += 1
            if progress_every and nb % progress_every == 0:
                log(f'    swap 进度 accepted={acc}/{target}（batches={nb}）')
        return {'accepted': int(acc), 'attempts': int(att), 'batches': int(nb),
                'accept_rate': r6(acc / att if att else None)}

    def moved_share(self) -> float:
        cur = np.sort((self.fi << 20) | self.peer)
        return float((~np.isin(cur, self.orig_key)).mean())

    def csr(self):
        import scipy.sparse as sp
        return sp.csr_matrix((self.w, (self.fi, self.peer)), shape=(self.n, self.n))


class RewireEngine:
    """零分布引擎：逐 rep 生成双保持网络 → 重算 treatment → 同一 FM 规格。"""

    def __init__(self, inp: Inputs, industry: str, weight_mode: str, swaps_per_edge: float):
        self.inp = inp
        self.industry = industry
        self.weight_mode = weight_mode
        self.spe = float(swaps_per_edge)
        self.fm_static = inp.sample.loc[inp.fm_mask,
                                        ['focal_id', 'month', 'ty', 'cid', 'R_next',
                                         'OwnMAXraw', 'PeerReturn', 'market_equity'] + CONTROLS].copy()
        self.fm_static['month_idx'] = np.array([inp.mpos[m] for m in self.fm_static['month']],
                                               dtype=np.int64)
        self.fm_rows_by_month = {m: np.flatnonzero(self.fm_static['month'].to_numpy() == m)
                                 for m in inp.months}
        self.n_fm = len(self.fm_static)
        self.true_pm = inp.sample.loc[inp.fm_mask, 'PeerMAXraw'].to_numpy(dtype=float)

    def new_nets(self) -> dict:
        out = {}
        for y in YEARS:
            ind = self.inp.ind[self.industry][y]
            out[y] = YearNet(self.inp.edges[y], self.inp.nodes, self.inp.nidx, ind)
        return out

    def treat_from_nets(self, nets: dict, perm: dict | None = None) -> tuple:
        """按（可选）文本年置换 π 重算 PeerMAXraw / PeerReturn（覆盖源重归一 + 月度 winsorize）。"""
        pm = np.full(len(self.inp.sample), np.nan)
        pr = np.full(len(self.inp.sample), np.nan)
        month_ty = self.inp.month_ty
        W_year = {y: nets[y].csr() for y in YEARS}      # 每年一张稀疏权重矩阵（rep 内复用）
        for m in self.inp.months:
            y = month_ty[m] if perm is None else perm[month_ty[m]]
            W = W_year[y]
            j = self.inp.mpos[m]
            x = self.inp.Xm[:, j]
            mk = np.isfinite(x)
            num = W @ np.where(mk, x, 0.0)
            den = W @ mk.astype(float)
            pmv = np.divide(num, den, out=np.full(self.inp.n, np.nan), where=den > 0)
            xr = self.inp.Xr[:, j]
            mr = np.isfinite(xr)
            numr = W @ np.where(mr, xr, 0.0)
            denr = W @ mr.astype(float)
            prv = np.divide(numr, denr, out=np.full(self.inp.n, np.nan), where=denr > 0)
            # 与面板构造同口径：覆盖源重归一后 round 8 位（`peermax_panel_build` 的 round(...,8)）
            pmv = np.round(pmv, 8)
            prv = np.round(prv, 8)
            rows = self.inp.month_rows[m]
            pm[rows] = pmv[self.inp.row_node[rows]]
            pr[rows] = prv[self.inp.row_node[rows]]
        return self._winsorize(pm), self._winsorize(pr)

    def _winsorize(self, v: np.ndarray) -> np.ndarray:
        """与样本构造同口径：按月的覆盖行（值非空）[0.01,0.99] clip。"""
        s = pd.Series(v)
        s = s.groupby(self.inp.sample['month'].to_numpy(), sort=False).transform(
            lambda x: x.clip(x.quantile(WINSOR[0]), x.quantile(WINSOR[1])))
        return s.to_numpy()

    def fm_on_treat(self, pm, pr, weight_col=None) -> dict:
        d = self.fm_static.copy()
        d['PeerMAXraw'] = self.true_pm if pm is None else pm[np.flatnonzero(self.inp.fm_mask)]
        if pr is not None:
            d['PeerReturn'] = pr[np.flatnonzero(self.inp.fm_mask)]
        res, _ = fm_regression(d, 'R_next', MAIN_X, weight_col=weight_col,
                               min_n=MIN_N, nw_lag=NW_LAG)
        c = res['coefs']['PeerMAXraw']
        return {'beta': r6(c['beta']), 'se': r6(c['se_nw']), 't': r6(c['t_nw']),
                'n_rows': int(len(d.dropna(subset=['R_next'] + MAIN_X))),
                'n_months': int(res['n_months_used'])}

    def run_rep(self, rep: int, seed: int, identity: bool = False) -> dict:
        t0 = time.time()
        nets = self.new_nets()
        swaps = {}
        if not identity:
            for y in YEARS:
                rng = np.random.default_rng([seed, rep, y])
                target = int(round(self.spe * len(self.inp.edges[y])))
                st = nets[y].swap(target, rng)
                swaps[int(y)] = st
        inv = {int(y): nets[y].verify_invariants() for y in YEARS}
        bad = {y: v for y, v in inv.items() if not v['all_ok']}
        if bad:
            write_json(ART / f'3.4_invariant_violation_rep{rep}.json',
                       {'rep': rep, 'violations': bad})
            print(f'[HARD-GATE] rep {rep} 三不变断言失败：{bad}；停机（3.0_plan §2.4 停机条件）',
                  file=sys.stderr, flush=True)
            raise SystemExit(2)
        moved = {int(y): nets[y].moved_share() for y in YEARS}
        pm, pr = self.treat_from_nets(nets)
        rec = {'rep': int(rep), 'seed': int(seed), 'identity': int(identity),
               'swaps_per_edge': self.spe, 'industry': self.industry,
               'weight_mode': self.weight_mode,
               'moved_edge_share': r6(float(np.mean(list(moved.values())))),
               'moved_edge_share_by_year': {str(k): r6(v) for k, v in moved.items()},
               'accepted_swaps': {str(k): v['accepted'] for k, v in swaps.items()},
               'accept_rate_mean': r6(float(np.mean([v['accept_rate'] for v in swaps.values()]))
                                      if swaps else None)}
        for weight, wcol in (('EW', None), ('VW', 'market_equity')):
            r = self.fm_on_treat(pm, pr, weight_col=wcol)
            rec[f'beta_{weight}'] = r['beta']
            rec[f'se_{weight}'] = r['se']
            rec[f't_{weight}'] = r['t']
            rec[f'n_rows_{weight}'] = r['n_rows']
            rec[f'n_months_{weight}'] = r['n_months']
        rec['elapsed_sec'] = r6(time.time() - t0)
        return rec

    def run_perm_rep(self, rep: int, seed: int) -> dict:
        """年块置换：π 随机置换 9 个文本年 → 每个月的 treatment 取自 π(y) 年的网络。"""
        t0 = time.time()
        nets = self.new_nets()
        if rep < 0:                                        # identity perm（保真门）：不消耗 rng
            perm = {int(y): int(y) for y in YEARS}
        else:
            rng = np.random.default_rng([seed, rep, 9999])
            perm = {int(y): int(z) for y, z in zip(YEARS, rng.permutation(YEARS))}
        pm, pr = self.treat_from_nets(nets, perm=perm)
        rec = {'rep': int(rep), 'seed': int(seed), 'perm': {str(k): v for k, v in perm.items()},
               'is_identity_perm': int(all(perm[y] == y for y in YEARS))}
        for weight, wcol in (('EW', None), ('VW', 'market_equity')):
            r = self.fm_on_treat(pm, pr, weight_col=wcol)
            rec[f'beta_{weight}'] = r['beta']
            rec[f'se_{weight}'] = r['se']
            rec[f't_{weight}'] = r['t']
            rec[f'n_rows_{weight}'] = r['n_rows']
        rec['elapsed_sec'] = r6(time.time() - t0)
        return rec






# ---------------------------------------------------------------- steps










def main():
    global _TEE
    ap = argparse.ArgumentParser(description='Phase 3.4 N-1/N-8/N-11 推断 shadow')
    ap.add_argument('--steps', default='all',
                    help='all 或逗号分隔子集：preflight,n1,n11,n8,n8perm,postflight,sha')
    ap.add_argument('--reps', type=int, default=200, help='N-8 零分布 rep 数（主 200；registry 1000）')
    ap.add_argument('--perm-reps', type=int, default=200, help='N-1 (c′) 年块置换 rep 数')
    ap.add_argument('--pc-perm-reps', type=int, default=200, help='N-11 年块置换 rep 数')
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--industry', default='asof', choices=['asof', 'latest'])
    ap.add_argument('--weight-mode', default='travel', choices=['travel'])
    ap.add_argument('--swaps-per-edge', type=float, default=1.0)
    ap.add_argument('--append-reps', action='store_true', help='N-8 逐 rep 续跑（跳过已存在 rep）')
    args = ap.parse_args()
    steps = (['preflight', 'n1', 'n11', 'n8', 'n8perm', 'postflight']
             if args.steps == 'all' else [s.strip() for s in args.steps.split(',') if s.strip()])
    ART.mkdir(parents=True, exist_ok=True)
    _TEE = _Tee(RUN_LOG)
    t_all = time.time()
    log(f'=== Phase 3.4 启动 steps={steps} reps={args.reps} seed={args.seed} '
        f'industry={args.industry} weight_mode={args.weight_mode} '
        f'swaps/edge={args.swaps_per_edge} registry_reps={REWIRE_REPS_REG} ===')

    now = datetime.now(timezone.utc).isoformat()
    counts = {'run_at': now, 'script': 'scripts/shadow_inference.py',
              'shadow_version_tag': SHADOW_TAG, 'steps': steps, 'args': vars(args),
              'run_history': []}
    if COUNTS_JSON.is_file():
        try:
            prev = json.loads(COUNTS_JSON.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            prev = None
        if isinstance(prev, dict):
            hist = prev.get('run_history') if isinstance(prev.get('run_history'), list) else []
            hist = hist + [{'run_at': prev.get('run_at'), 'steps': prev.get('steps')}]
            prev = {k: v for k, v in prev.items() if k != 'run_history'}
            prev.update({'run_at': now, 'steps': steps, 'args': vars(args),
                         'run_history': hist})
            counts = prev

    if 'preflight' in steps:
        counts['anchor_pre'] = anchors_snapshot('pre')
        write_json(ANCHOR_PRE, counts['anchor_pre'])
        log(f'preflight：9 锚 + {counts["anchor_pre"]["headline_network_n_files"]} 个 headline 网络文件 sha 已记录')

    inp = Inputs()
    counts['inputs'] = {
        'sample_rows': int(len(inp.sample)), 'fm_rows': int(inp.fm_mask.sum()),
        'months': len(inp.months), 'nodes': int(inp.n),
        'edges_total': int(sum(len(e) for e in inp.edges.values())),
        'industry_source': inp.ind['source'],
        'registry': {'nw_lag': NW_LAG, 'min_focal_count_per_month': MIN_N,
                     'winsorization_pct': list(WINSOR), 'rewiring_reps': REWIRE_REPS_REG,
                     'random_seed': param_value('random_seed'), 'registry_file': str(REGISTRY)},
    }

    if 'n1' in steps:
        counts['n1'] = step_n1(inp, counts)
    if 'n11' in steps:
        counts['n11'] = step_n11(inp, counts, args.pc_perm_reps, args.seed)
    if 'n8' in steps:
        counts['n8'] = step_n8(inp, counts, args.reps, args.seed, args.industry,
                               args.weight_mode, args.swaps_per_edge, args.append_reps)
    if 'n8perm' in steps:
        counts['n1_block_perm'] = step_n8perm(inp, counts, args.perm_reps, args.seed,
                                              args.industry, args.append_reps)

    # 汇总长表输出（依赖前序 step 的结果；增量运行时用 counts 里的既有读数）
    write_long_tables(counts)

    if 'postflight' in steps:
        post = anchors_snapshot('post')
        write_json(ANCHOR_POST, post)
        pre = json.loads(ANCHOR_PRE.read_text(encoding='utf-8')) if ANCHOR_PRE.is_file() else None
        cmp_ = compare_anchors(pre, post) if pre else {'status': 'NO_PRE'}
        counts['anchor_compare'] = cmp_
        log(f'postflight：锚对比 {cmp_["status"]}（9 锚 + {cmp_.get("n_headline_files")} headline 文件）')
        if cmp_['status'] != 'MATCH':
            write_json(COUNTS_JSON, counts)
            print('[HARD-GATE] headline/输入锚发生变化；停机（禁止重算覆盖）', file=sys.stderr, flush=True)
            raise SystemExit(2)
    if 'sha' in steps or 'postflight' in steps:
        counts['output_sha'] = step_sha(inp)

    counts['elapsed_sec_total'] = r6(time.time() - t_all)
    write_json(COUNTS_JSON, counts)
    log(f'=== 完成：总耗时 {counts["elapsed_sec_total"]}s → {COUNTS_JSON.name} ===')






if __name__ == '__main__':
    main()
