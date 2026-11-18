#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
  python scripts/data_audit/p3_4_inference_shadow.py --steps all
  python scripts/data_audit/p3_4_inference_shadow.py --steps n1,n11          # 快件
  python scripts/data_audit/p3_4_inference_shadow.py --steps n8 --reps 200   # 主零分布
  python scripts/data_audit/p3_4_inference_shadow.py --steps n8 --reps 1000 --append-reps  # 夜间增量
  python scripts/data_audit/p3_4_inference_shadow.py --steps n8perm --reps 200
"""
from __future__ import annotations
# --- scripts/lib bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from lib.shadow.p3_4_inference_shadow import ALLOWED_ROOTS, ANCHOR_POST, ANCHOR_PRE, ART, ASOF_MAP, BANNED_ROOTS, BLOCKPERM_CSV, CLUSTER_SENTINEL, CONTROLS, COUNTS_JSON, DEFAULT_SEED, EFF_2019, EFF_TOP1, FM_MONTHS, FM_ROWS, HEADLINE_EW, HEADLINE_NET, HEADLINE_VW, INF_CSV, M1_REG, MAIN_X, MIN_N, MOVED_SHARE_GATE, NULL_COLS, NULL_CSV, NULL_SUMMARY, NW_LAG, OLD_PLACEBO, OUTPUT_SHA, PCT_POINT_TOL, PC_BASE_REG, PC_CSV, PC_T5B_REG, PC_UNIQUE_REG, PC_UNIQUE_TOTAL_REG, PC_YEARS, PERM_COLS, PRICING_PANEL, REGISTRY, REWIRE_REPS_REG, REWIRE_SEED_REG, ROOT, RUN_LOG, SAMPLE_PARQUET, SAMPLE_ROWS, SHADOW_TAG, STRATA_DEG, STRATA_DEG_REG, STRATA_EFF, STRUCT_CSV, SW2_LATEST, VAR_CSV, VD_REG, WINSOR, YEARS, _TEE, _json_default, _long_header, _scale_rec, anchors_snapshot, assert_output_path_guard, compare_anchors, eff_opponents, eff_opponents_from_edges, fm_cluster_scores, inference_recipes, log, n1_inference, null_summary, pc_block_permutation, pc_cluster_year_constancy, pc_readings, r6, read_csv_rows, sha_ledger_headline_net, step_n1, step_n11, step_n8, step_n8perm, step_sha, strata_fm, variance_decomposition, write_csv, write_json, write_long_tables
# --- end scripts/lib bootstrap ---


import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(ROOT / 'scripts' / 'data_audit'))

from s3_m1_lib import (  # noqa: E402
    assert_m1_inputs_sha, fm_regression, newey_west, param_value, require, sha256_file,
    two_sided_p, _ols,
)
from s4_step2_peermax_formal import (  # noqa: E402
    ARGMAX, EXCLUDE_QUAL, RETURNS, load_versions, select_version,
)

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
            # 与面板构造同口径：覆盖源重归一后 round 8 位（`s4_step2_peermax_formal` 的 round(...,8)）
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
    counts = {'run_at': now, 'script': 'scripts/data_audit/p3_4_inference_shadow.py',
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
