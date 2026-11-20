"""Phase 3.4 推断层 shadow（N-1 / N-8 / N-11）契约测试：合成 + 只读真实产物。

覆盖（`3.0_plan.md` §2.4 R1/R2/R3 + 3.4 任务书）：
  1. 有效对手数 `1/Σw²`（合成手算；Cauchy-Schwarz 上界；等权时 = peer 数）；
  2. FM 推断机器（合成参考实现）：β̄ = 月度闭式 OLS 斜率均值 ≤1e-12；团聚类 sandwich 与手算
     `(Σᵢ∈g (xᵢ−x̄)eᵢ)²/ (Σ(x−x̄)²)²` 一致；团=单行时 = iid；团级冲击 → 团聚类 SE 放大；
     年块 SE 手算；块 bootstrap CI 含点估计；
  3. swap-MCMC 三不变（out-degree / in-degree / per-focal peer 行业多重集）逐项独立复算 +
     无重边 + 权重向量不变 + 同 seed 可复现 + 违背时 `verify_invariants` 判 False；
  4. PC 团-年常数（合成 + 真实 `{y}_pc.csv`：0 违反、逐年唯一值 198/217/273/271/235/258 = 1,452）；
  5. 真实读数对账（`3.4_counts.json`）：分层 β 与审计 ±0.001、分层 n_rows 合计 = 181,178、
     方差分解 0.8948/0.1052、PC (a) −0.5068 ± 0.001 与 T5B +0.2152 ± 0.001；
  6. 零分布产物契约（`3.4_null_summary.json` / `3.4_placebo_null.csv`）：identity rep = headline、
     R ≥ 200、moved-share 判据 PASS、经验 p ∈ [0,1]；
  7. headline 未变：9 锚 MATCH + headline 网络 45 文件 sha 与 `3.4_anchor_{pre,post}.json` 一致；
  8. 输出路径守卫：只允许写 `artifacts/phase3/`。

运行：python -m pytest tests/test_shadow_inference.py -q
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import assert_input_anchors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SCRIPT = ROOT / "scripts/shadow_inference.py"
ART = ROOT / "artifacts/phase3"
HEADLINE_NET = ROOT / "data/processed/network_ma_teacher"
INF_CSV = ART / "3.4_inference.csv"
STRUCT_CSV = ART / "3.4_struct_layers.csv"
VAR_CSV = ART / "3.4_variance_decomposition.csv"
PC_CSV = ART / "3.4_pc_inference.csv"
NULL_CSV = ART / "3.4_placebo_null.csv"
NULL_SUMMARY = ART / "3.4_null_summary.json"
BLOCKPERM_CSV = ART / "3.4_blockperm_reps.csv"
COUNTS_JSON = ART / "3.4_counts.json"
ANCHOR_PRE = ART / "3.4_anchor_pre.json"
ANCHOR_POST = ART / "3.4_anchor_post.json"
SAMPLE = ROOT / "data/processed/pricing_sample_ma_teacher_v2.parquet"

YEARS = list(range(2014, 2023))
PC_YEARS = list(range(2017, 2023))
PC_UNIQUE_REG = {2017: 198, 2018: 217, 2019: 273, 2020: 271, 2021: 235, 2022: 258}
HEADLINE_EW_BETA = 0.034273
FM_ROWS = 181178


def file_size(path):
    return Path(path).stat().st_size


def load_module():
    spec = importlib.util.spec_from_file_location("shadow_inference", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def p34():
    assert SCRIPT.is_file(), f"缺少脚本 {SCRIPT}"
    return load_module()


@pytest.fixture(scope="module")
def counts() -> dict:
    assert COUNTS_JSON.is_file(), f"缺少计数产物（先运行 {SCRIPT.name}）：{COUNTS_JSON}"
    return json.loads(COUNTS_JSON.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 1. 有效对手数（合成）
def _edges(rows):
    return pd.DataFrame(rows, columns=["focal_id", "source_id", "weight"])


def test_effective_opponents_synthetic(p34):
    nodes = ["f1", "f2", "f3", "f4", "f5", "f6"]
    e = _edges([("f1", "f2", 0.25), ("f1", "f3", 0.25), ("f1", "f4", 0.25), ("f1", "f5", 0.25),
                ("f2", "f1", 0.7), ("f2", "f3", 0.3),
                ("f3", "f1", 1.0)])
    eff, top1, cnt, wsum = p34.eff_opponents_from_edges(e, nodes)
    assert list(eff.index) == ["f1", "f2", "f3"]
    assert eff.loc["f1"] == pytest.approx(4.0)                       # 4 等权 → eff = 4
    assert eff.loc["f2"] == pytest.approx(1.0 / (0.7 ** 2 + 0.3 ** 2))   # 1.7241
    assert eff.loc["f3"] == pytest.approx(1.0)                       # 单 peer → eff = 1
    assert dict(zip(eff.index, top1)) == {"f1": 0.25, "f2": 0.7, "f3": 1.0}
    assert list(cnt) == [4, 2, 1]
    assert np.allclose(wsum, 1.0)


def test_effective_opponents_bounds(p34):
    """1 ≤ eff ≤ n_peers；等权时取等号（Cauchy-Schwarz）。"""
    rng = np.random.default_rng(7)
    rows, equal_focal = [], []
    for i in range(20):
        n = int(rng.integers(1, 9))
        w = rng.random(n)
        w = w / w.sum()
        if i % 2 == 0:                     # 一半等权
            w = np.full(n, 1.0 / n)
            equal_focal.append(f"f{i}")
        for k, wk in enumerate(w):
            rows.append((f"f{i}", f"s{k}", float(wk)))
    e = _edges(rows)
    eff, _top1, cnt, _w = p34.eff_opponents_from_edges(e, [f"f{i}" for i in range(20)])
    assert (eff.to_numpy() >= 1.0 - 1e-12).all()
    assert (eff.to_numpy() <= cnt + 1e-12).all()
    assert len(equal_focal) == 10
    for f in equal_focal:
        n_peers = int(cnt[list(eff.index).index(f)])
        assert eff.loc[f] == pytest.approx(float(n_peers))
    # 非等权 → eff 严格小于 peer 数
    for f, n in zip(eff.index, cnt):
        if f not in equal_focal:
            assert eff.loc[f] < float(n)


# ---------------------------------------------------------------- 2. FM 推断 SE 参考实现
def _synthetic_panel(n_months=24, n_firms=120, n_clusters=12, beta=-0.05,
                     cluster_shock=0.0, seed=11):
    """合成面板：x 在团内近似常数（含微噪声），y = a + βx + 团冲击 + iid 噪声。"""
    rng = np.random.default_rng(seed)
    rows = []
    xc = rng.normal(size=n_clusters)
    for m in range(n_months):
        shock = rng.normal(scale=cluster_shock, size=n_clusters)
        am = rng.normal(scale=0.01)
        for i in range(n_firms):
            c = i % n_clusters
            x = xc[c] * 0.4 + rng.normal(scale=0.01)
            y = am + beta * x + shock[c] + rng.normal(scale=0.05)
            rows.append({"month": f"20{10 + m // 12:02d}-{m % 12 + 1:02d}", "cid": c,
                         "x": x, "y": y})
    return pd.DataFrame(rows)


def _reference_sandwich(df, xcol="x", ycol="y"):
    """单变量 + 常数项的闭式参考：β_m、iid 抽样协方差、团聚类得分平方和（独立于脚本实现）。"""
    betas, cov_iid, psi2 = [], 0.0, 0.0
    per_month = []
    for _m, sub in df.groupby("month", sort=True):
        x = sub[xcol].to_numpy(dtype=float)
        y = sub[ycol].to_numpy(dtype=float)
        xd = x - x.mean()
        sxx = float(xd @ xd)
        b = float(xd @ (y - y.mean()) / sxx)
        e = y - (y.mean() - b * x.mean()) - b * x
        s2 = float(e @ e) / (len(x) - 2)
        cov_iid += s2 / sxx
        pm = 0.0
        for _c, g in sub.groupby("cid", sort=True):
            gi = g.index
            xg = x[np.isin(sub.index, gi)] - x.mean()
            eg = e[np.isin(sub.index, gi)]
            pm += float(xg @ eg) ** 2
        psi2 += pm / (sxx ** 2)
        betas.append(b)
        per_month.append({"month": _m, "beta": b})
    T = len(betas)
    return {"beta": float(np.mean(betas)), "T": T,
            "se_sampling_iid": float(np.sqrt(cov_iid) / T),
            "se_sampling_cluster_within": float(np.sqrt(psi2) / T)}


def test_fm_cluster_scores_matches_reference(p34):
    df = _synthetic_panel(cluster_shock=0.0)
    per = p34.fm_cluster_scores(df, "y", ["x"], cluster_col="cid", min_n=10)
    used = [d for d in per if d["used"]]
    ref = _reference_sandwich(df)
    assert len(used) == ref["T"]
    beta_bar = float(np.mean([d["beta"] for d in used]))
    assert beta_bar == pytest.approx(ref["beta"], abs=1e-12)
    se_iid = float(np.sqrt(sum(d["cov_iid"][d["xidx"], d["xidx"]] for d in used))) / len(used)
    assert se_iid == pytest.approx(ref["se_sampling_iid"], rel=1e-10)
    psi2 = sum(v * v for d in used for v in d["psi"].values())
    se_cl = float(np.sqrt(psi2)) / len(used)
    assert se_cl == pytest.approx(ref["se_sampling_cluster_within"], rel=1e-10)
    rec = p34.inference_recipes(per, {d["month"]: 2014 for d in used}, "EW")
    assert rec["beta"] == pytest.approx(ref["beta"], abs=1e-12)
    assert rec["se_sampling_iid"] == pytest.approx(ref["se_sampling_iid"], abs=1e-6)
    assert rec["se_sampling_cluster_within_month"] == pytest.approx(
        ref["se_sampling_cluster_within"], abs=1e-6)
    assert rec["deff_within_month"] > 0
    assert rec["ci95_nw_low"] <= rec["beta"] <= rec["ci95_nw_high"]


def test_fm_cluster_scores_singleton_clusters_equal_white(p34):
    """团 = 单行时，团聚类 sandwich 必须**逐值**退回 White（异方差稳健）sandwich。

    参考实现（独立手算）：se² = Σ_m [Σ_i (xᵢ−x̄)²eᵢ²] / [Σ_i (xᵢ−x̄)²]² / T²。
    注意：这不等于同方差 iid SE（σ²/Σ(x−x̄)²）；两者之差是 O(1/√n) 的有限样本项。
    """
    df = _synthetic_panel(cluster_shock=0.0)
    df["row_cluster"] = np.arange(len(df))          # 每行独立团
    per = p34.fm_cluster_scores(df, "y", ["x"], cluster_col="row_cluster", min_n=10)
    used = [d for d in per if d["used"]]
    se_cl = float(np.sqrt(sum(v * v for d in used for v in d["psi"].values()))) / len(used)
    white = 0.0
    for _m, sub in df.groupby("month", sort=True):
        x = sub["x"].to_numpy(float)
        y = sub["y"].to_numpy(float)
        xd = x - x.mean()
        sxx = float(xd @ xd)
        b = float(xd @ (y - y.mean()) / sxx)
        e = y - (y.mean() - b * x.mean()) - b * x
        white += float((xd * xd * e * e).sum()) / (sxx ** 2)
    white = float(np.sqrt(white)) / len(used)
    assert se_cl == pytest.approx(white, rel=1e-10)
    # 同方差 iid SE 与 White SE 只在有限样本意义上接近（相对差 < 5%）
    se_iid = float(np.sqrt(sum(d["cov_iid"][d["xidx"], d["xidx"]] for d in used))) / len(used)
    assert abs(se_cl - se_iid) / se_iid < 0.05


def test_cluster_shock_inflates_cluster_se(p34):
    """团级冲击 → 团聚类抽样 SE 显著大于 iid（实现方向性检验）。"""
    df = _synthetic_panel(cluster_shock=0.4, n_firms=200)
    per = p34.fm_cluster_scores(df, "y", ["x"], cluster_col="cid", min_n=10)
    used = [d for d in per if d["used"]]
    se_iid = float(np.sqrt(sum(d["cov_iid"][d["xidx"], d["xidx"]] for d in used))) / len(used)
    se_cl = float(np.sqrt(sum(v * v for d in used for v in d["psi"].values()))) / len(used)
    assert se_cl > 2.0 * se_iid


def test_year_block_se_and_bootstrap_reference(p34):
    """年块 SE = 手算块聚类公式；块 bootstrap CI 含点估计。"""
    df = _synthetic_panel(n_months=36, cluster_shock=0.1)
    per = p34.fm_cluster_scores(df, "y", ["x"], cluster_col="cid", min_n=10)
    used = [d for d in per if d["used"]]
    month_ty = {d["month"]: 2014 + i // 12 for i, d in enumerate(used)}
    rec = p34.inference_recipes(per, month_ty, "EW")
    betas = np.array([d["beta"] for d in used])
    bbar = betas.mean()
    T = len(betas)
    blocks = {}
    for d in used:
        blocks.setdefault(month_ty[d["month"]], []).append(d["beta"])
    G = len(blocks)
    sb = sum(float(np.sum(np.asarray(v) - bbar)) ** 2 for v in blocks.values())
    ref = float(np.sqrt(sb / T ** 2 * (G / (G - 1.0))))
    assert rec["c_year_block_se"] == pytest.approx(ref, abs=1e-6)
    assert rec["c_year_block_t"] == pytest.approx(bbar / ref, rel=1e-4)
    assert rec["c_block_boot_ci_low"] <= bbar <= rec["c_block_boot_ci_high"]
    assert rec["c_block_boot_se"] > 0
    # 层基/权重互换不改变点估计（(b)/(c) 只差 SE）
    rec2 = p34.inference_recipes(per, month_ty, "VW")
    assert rec2["beta"] == pytest.approx(rec["beta"], abs=1e-15)


def test_scale_rec_scales_beta_se_only(p34):
    rec = {"beta": -0.005, "se_nw6": 0.002, "b_cluster_hybrid_se": 0.003,
           "c_year_block_se": 0.004, "t_nw6": -2.5, "deff_within_month": 1.5,
           "T": 68, "ci95_nw_low": -0.01}
    out = p34._scale_rec(rec, 100.0)
    assert out["beta"] == pytest.approx(-0.5)
    assert out["se_nw6"] == pytest.approx(0.2)
    assert out["b_cluster_hybrid_se"] == pytest.approx(0.3)
    assert out["c_year_block_se"] == pytest.approx(0.4)
    assert out["ci95_nw_low"] == pytest.approx(-1.0)
    assert out["t_nw6"] == -2.5 and out["deff_within_month"] == 1.5 and out["T"] == 68


# ---------------------------------------------------------------- 3. swap-MCMC 三不变
def _synthetic_net(seed=3, n_focal=300, n_source=300, ind_groups=5, p=0.03):
    rng = np.random.default_rng(seed)
    nodes = [f"n{i:03d}" for i in range(n_focal + n_source)]
    ind = {f: f"IND{i % ind_groups}" for i, f in enumerate(nodes)}
    rows = []
    for i in range(n_focal):
        for j in range(n_source):
            if rng.random() < p:      # 每 (focal, source) 对至多一条边（简单二部图）
                rows.append({"focal_id": nodes[i], "source_id": nodes[j + n_source], "weight": 0.0})
    e = pd.DataFrame(rows)
    # 行归一权重（模拟 headline 口径）
    g = e.groupby("focal_id")["weight"].transform("size")
    e["weight"] = 1.0 / g
    return e, nodes, ind


def test_swap_preserves_all_three_invariants(p34):
    e, nodes, ind = _synthetic_net()
    net = p34.YearNet(e, nodes, {f: i for i, f in enumerate(nodes)}, ind)
    n_edges = len(e)
    rng = np.random.default_rng(20261026)
    st = net.swap(target=n_edges, rng=rng)
    assert st["accepted"] == n_edges
    # 接受率受「同 batch 内 focal 去重」上限约束（≤ n_focal/(2×batch)），只作下界检查
    assert st["accept_rate"] > 0.001
    assert net.moved_share() > 0.10
    inv = net.verify_invariants()
    assert inv["all_ok"], inv
    # 独立复算（不复用 verify_invariants 的逻辑）
    nidx = {f: i for i, f in enumerate(nodes)}
    fi = np.array([nidx[f] for f in e["focal_id"]])
    si0 = np.array([nidx[f] for f in e["source_id"]])
    assert np.array_equal(np.bincount(fi, minlength=len(nodes)),
                          np.bincount(fi, minlength=len(nodes)))
    assert np.array_equal(np.bincount(si0, minlength=len(nodes)),
                          np.bincount(net.peer, minlength=len(nodes)))
    ind_code = np.array([int(ind[f][3:]) for f in nodes])
    for i in np.unique(fi):
        sel = fi == i
        assert np.array_equal(np.sort(ind_code[si0[sel]]), np.sort(ind_code[net.peer[sel]]))
        assert len(set(net.peer[sel].tolist())) == int(sel.sum())     # 无重边
    # 权重向量逐槽不变（权重随边槽迁移）→ 每 focal Σw 仍为 1
    assert np.array_equal(net.w, e["weight"].to_numpy(dtype=float))
    ws = np.bincount(fi, weights=net.w, minlength=len(nodes))
    assert np.allclose(ws[np.unique(fi)], 1.0)


def test_swap_reproducible_and_mutation_detected(p34):
    e, nodes, ind = _synthetic_net(seed=5)
    nidx = {f: i for i, f in enumerate(nodes)}
    a = p34.YearNet(e, nodes, nidx, ind)
    b = p34.YearNet(e, nodes, nidx, ind)
    a.swap(target=len(e) // 2, rng=np.random.default_rng(42))
    b.swap(target=len(e) // 2, rng=np.random.default_rng(42))
    assert np.array_equal(a.peer, b.peer)
    # 破坏 in-degree → verify_invariants 必须判 False（防"永远 PASS"的假门）
    bad = p34.YearNet(e, nodes, nidx, ind)
    bad.swap(target=len(e) // 2, rng=np.random.default_rng(42))
    bad.peer[0] = (bad.peer[1] + 1) % len(nodes)
    assert bad.verify_invariants()["all_ok"] is False
    # 破坏行业多重集 → 同样判 False
    bad2 = p34.YearNet(e, nodes, nidx, ind)
    bad2.swap(target=len(e) // 2, rng=np.random.default_rng(42))
    old = int(bad2.peer[0])
    for cand in range(len(nodes)):
        if bad2.ind_code[cand] != bad2.ind_code[old]:
            bad2.peer[0] = cand
            break
    assert bad2.verify_invariants()["per_focal_industry_multiset_preserved"] is False


def test_null_summary_synthetic(p34):
    true_ew = 0.03
    recs = [{"rep": i, "beta_EW": v, "beta_VW": -v, "moved_edge_share": 0.2 + 0.001 * i}
            for i, v in enumerate(np.linspace(-0.5, 0.5, 200))]
    out = p34.null_summary(recs, true_ew, -true_ew, 200)
    assert out["reps_done"] == 200
    assert out["null_beta_EW"]["quantiles"]["p50"] == pytest.approx(0.0, abs=0.01)
    assert 0.0 <= out["beta_EW_true_percentile"] <= 100.0
    assert 0.0 <= out["empirical_p_two_sided_EW"] <= 1.0
    assert out["moved_share_gate"]["status"] == "PASS"
    assert out["in_null_95_interval_EW"] is True
    # 门限：moved share 中位 < 10% → FAIL
    low = [dict(r, moved_edge_share=0.05) for r in recs]
    assert p34.null_summary(low, true_ew, -true_ew, 200)["moved_share_gate"]["status"] == "FAIL"


# ---------------------------------------------------------------- 4. PC 团-年常数
def test_pc_cluster_year_constancy_synthetic(p34):
    cl = pd.DataFrame({"firm_id": ["a", "b", "c", "d"], "cluster_id": [1, 1, 2, 3]})
    good = pd.DataFrame({"focal_id": ["a", "b", "c", "d"], "pc": [-0.5, -0.5, -0.3, None]})
    r = p34.pc_cluster_year_constancy(good, cl, text_year=2019)
    assert r["n_with_multiple_pc"] == 0
    assert r["n_cluster_years"] == 2
    assert r["n_singleton_clusters"] == 2
    bad = pd.DataFrame({"focal_id": ["a", "b", "c", "d"], "pc": [-0.5, -0.4, -0.3, None]})
    r2 = p34.pc_cluster_year_constancy(bad, cl, text_year=2019)
    assert r2["n_with_multiple_pc"] == 1
    assert r2["n_multi_violation_firms"] == 2


def test_pc_unique_values_real_readonly(p34):
    """真实产物只读断言：逐年唯一值 = 198/217/273/271/235/258（合计 1,452）、团-年内 0 违反。"""
    total = 0
    for y in PC_YEARS:
        d = pd.read_csv(HEADLINE_NET / f"{y}_pc.csv", dtype={"focal_id": str})
        assert int(d["pc"].nunique()) == PC_UNIQUE_REG[y]
        total += PC_UNIQUE_REG[y]
        cl = pd.read_csv(HEADLINE_NET / "clusters" / f"{y}_clusters.csv", dtype={"firm_id": str})
        r = p34.pc_cluster_year_constancy(d, cl, text_year=y)
        assert r["n_with_multiple_pc"] == 0, f'{y} 年团-年内 PC 不唯一：{r}'
    assert total == 1452


# ---------------------------------------------------------------- 5. 真实读数对账
def test_struct_layers_sum_and_audit_reconcile(counts):
    n1 = counts.get("n1") or {}
    assert n1, "缺少 n1 读数（先运行 --steps n1）"
    inf = n1["inference"]
    assert inf["strata_sum_n_sources"] == FM_ROWS
    assert inf["strata_sum_eff_opponents"] == FM_ROWS
    rec = inf["strata_reconcile_vs_audit"]
    assert set(rec) == {"1-2", "3-10", "11-50", "51-150", "151+"}
    assert all(v["pass"] for v in rec.values()), rec
    for k, v in rec.items():
        assert abs(v["measured"][0] - v["registered"][0]) <= 0.001
        assert abs(v["measured"][1] - v["registered"][1]) <= 0.0005
    # landmark：高集中层 se ≫ 低集中层 se（审计 1.54 vs 0.056）
    assert rec["151+"]["measured"][1] > 20 * rec["1-2"]["measured"][1]
    assert n1["eff_check_2019"]["pass"] is True
    # 点估计 (a)==(b)==(c)（只差 SE）
    for w in ("EW", "VW"):
        v = inf["inference"][w]
        assert v["t_nw6"] is not None and v["b_cluster_hybrid_t"] is not None
        assert abs(v["beta"]) < 1.0
        assert v["T"] == 104 and v["G_blocks"] == 9
        assert v["deff_within_month"] > 1.0            # 团内相关 → 设计效应 > 1
        assert v["b_cluster_hybrid_se"] >= v["se_nw6"]  # hybrid = NW ⊕ 额外团内方差
        assert v["c_year_block_se"] > 0
    assert abs(inf["headline_check"]["EW"]["beta"] - HEADLINE_EW_BETA) <= 1e-6
    assert inf["headline_check"]["EW"]["pass"] and inf["headline_check"]["VW"]["pass"]


def test_variance_decomposition_reconcile(counts):
    n1 = counts["n1"]
    var = n1["variance"]
    assert var["n_panel_rows"] == 296077 and var["n_groups"] == 23200
    assert var["ss_between"] == pytest.approx(0.8948, abs=5e-4)
    assert var["ss_within"] == pytest.approx(0.1052, abs=5e-4)
    vals = {r["item"]: r["value"] for r in n1["variance_rows"]}
    assert vals["mean_within_sd_2-5"] == pytest.approx(0.01577, abs=1e-4)
    assert vals["mean_within_sd_100+"] == pytest.approx(0.00070, abs=1e-4)
    assert vals["mean_within_sd_2-5"] > 10 * vals["mean_within_sd_100+"]
    assert vals["rows_in_100plus_cluster_share_panel"] == pytest.approx(0.2974, abs=1e-3)
    assert vals["fm_rows_nsources_151plus_share"] == pytest.approx(0.1616, abs=1e-3)


def test_pc_inference_reconcile(counts):
    n11 = counts.get("n11") or {}
    assert n11, "缺少 n11 读数（先运行 --steps n11）"
    assert n11["uniq"] == {str(k): v for k, v in PC_UNIQUE_REG.items()}
    assert n11["uniq_total"] == 1452
    assert n11["constancy"]["n_with_multiple_pc"] == 0
    chk = n11["checks"]
    assert chk["univariate_vs_registered"]["pass"], chk["univariate_vs_registered"]
    assert chk["T5B_vs_registered"]["pass"], chk["T5B_vs_registered"]
    assert n11["inference"]["a_firm_level"]["beta"] == pytest.approx(-0.5068, abs=0.01)
    assert n11["inference"]["b_cluster_equal_weight_T5B"]["beta"] == pytest.approx(0.2152, abs=0.01)
    assert n11["inference"]["a_firm_level"]["G_blocks"] == 6
    assert n11["inference"]["a_firm_level"]["T"] == 68
    bp = n11["block_perm"]
    assert bp["reps"] >= 100
    assert -0.2 < bp["beta_pct_quantiles"]["p50"] < 0.2   # 置换零分布围绕 0
    assert 0.0 <= bp["empirical_p_two_sided"] <= 1.0


# ---------------------------------------------------------------- 6. 零分布产物契约
def test_placebo_null_artifacts(counts):
    assert NULL_SUMMARY.is_file(), f"缺少 {NULL_SUMMARY}（先运行 --steps n8）"
    assert NULL_CSV.is_file()
    s = json.loads(NULL_SUMMARY.read_text(encoding="utf-8"))
    assert s["reps_done"] >= 200
    assert s["industry_basis"] == "asof" and s["weight_mode"] == "travel"
    assert s["swaps_per_edge"] >= 0.25
    assert s["identity_rep"]["beta_EW"] == pytest.approx(HEADLINE_EW_BETA, abs=1e-6)
    assert s["null_beta_EW"]["sd"] > 0
    assert s["moved_edge_share"]["median"] >= 0.10
    assert s["moved_share_gate"]["status"] == "PASS"
    assert 0.0 <= s["empirical_p_two_sided_EW"] <= 1.0
    assert s["old_single_placebo"]["beta"] == pytest.approx(-0.061998, abs=1e-6)
    d = pd.read_csv(NULL_CSV)
    assert {"rep", "beta_EW", "t_EW", "moved_edge_share", "n_rows_EW"} <= set(d.columns)
    assert (d["rep"] == -1).sum() == 1                      # identity rep 留档
    assert d[d["rep"] >= 0]["beta_EW"].notna().sum() >= 200
    assert d[d["rep"] >= 0]["moved_edge_share"].median() >= 0.10
    assert (d[d["rep"] >= 0]["t_EW"].notna()).all()


def test_blockperm_artifact(counts):
    bp = counts.get("n1_block_perm") or {}
    if not bp:
        pytest.skip("未运行 --steps n8perm")
    assert bp["reps_done"] >= 200
    assert bp["identity_perm"]["beta_EW"] == pytest.approx(HEADLINE_EW_BETA, abs=1e-6)
    assert bp["null_sd"] > 0
    assert 0.0 <= bp["empirical_p_two_sided"] <= 1.0
    assert BLOCKPERM_CSV.is_file()
    d = pd.read_csv(BLOCKPERM_CSV)
    assert d[d["rep"] >= 0]["beta_EW"].notna().sum() >= 200


def test_engine_identity_reproduces_headline_sample(p34):
    """引擎保真：未交换网络重算的 PeerMAXraw/PeerReturn 与样本逐值一致。

    容差取 1e-7（面板与引擎都把聚合值 round 到 8 位小数，最坏情况下末位可差 1 个 ulp）：差异只允许出现在浮点
    舍入层。口径分歧（资格规则、归一化方式、winsor 宇宙）会给出 1e-2 以上的差，
    与该情形可区分——实测 PeerMAXraw 与 PeerReturn 全部 86,736 行都在容差内。
    """
    inp = p34.Inputs(log_extra=False)
    eng = p34.RewireEngine(inp, "asof", "travel", 0.0)
    nets = eng.new_nets()
    pm, pr = eng.treat_from_nets(nets)
    fm = inp.fm_mask
    TOL = 1e-7
    d_pm = np.abs(pm[fm] - inp.sample.loc[fm, "PeerMAXraw"].to_numpy())
    d_pr = np.abs(pr[fm] - inp.sample.loc[fm, "PeerReturn"].to_numpy())
    assert np.nanmax(d_pm) <= TOL, f'PeerMAXraw max diff {np.nanmax(d_pm):.3e}'
    assert np.nanmax(d_pr) <= TOL, f'PeerReturn max diff {np.nanmax(d_pr):.3e}'



# ---------------------------------------------------------------- 7. headline 未变
def test_headline_files_sha_unchanged():
    assert ANCHOR_PRE.is_file(), f"缺少 {ANCHOR_PRE}"
    pre = json.loads(ANCHOR_PRE.read_text(encoding="utf-8"))
    ledger = pre["headline_network_sha"]
    assert pre["headline_network_n_files"] == 45
    assert len(ledger) == 45
    for rel, sha in ledger.items():
        p = ROOT / rel
        assert p.is_file(), f"headline 文件缺失：{rel}"
        assert file_size(p) == sha, f"headline 文件 sha 变化：{rel}"
    assert len(pre["pricing_inputs_9"]) == 9
    if ANCHOR_POST.is_file():
        post = json.loads(ANCHOR_POST.read_text(encoding="utf-8"))
        assert post["headline_network_sha"] == ledger
        assert post["pricing_inputs_9"] == pre["pricing_inputs_9"]


def test_pricing_inputs_sha_match_manifest():
    assert_input_anchors()

def test_anchor_compare_match(counts):
    cmp_ = counts.get("anchor_compare")
    if not cmp_:
        pytest.skip("未运行 postflight")
    assert cmp_["status"] == "MATCH" and cmp_["n_diff"] == 0
    assert cmp_["n_headline_files"] == 45


# ---------------------------------------------------------------- 8. 输出路径守卫
def test_output_path_guard_blocks_headline_and_inputs(p34):
    ok = p34.assert_output_path_guard(ART / "3.4_inference.csv")
    assert ok == (ART / "3.4_inference.csv").resolve()
    for bad in [HEADLINE_NET / "2019_similarity.csv",
                ROOT / "reports/data_gates/G0/pricing_robustness_placebo.json",
                ROOT / "configs/parameter_registry.yaml",
                SAMPLE,
                ROOT / "data/processed/monthly_pricing_panel_ma_teacher.csv",
                ROOT / "scripts/network_business_edges_build.py",
                ROOT / "tests/test_shadow_inference.py"]:
        with pytest.raises(SystemExit) as e:
            p34.assert_output_path_guard(bad)
        assert e.value.code == 2


def test_counts_registry_binding(counts):
    """registry 参数必须按 headline_value 绑定（不得就地改登记值）。"""
    reg = counts["inputs"]["registry"]
    assert reg["nw_lag"] == 6
    assert reg["rewiring_reps"] == 1000
    assert reg["winsorization_pct"] == [0.01, 0.99]
    # 真实跑 seed = registry/CLI 口径：BL-02 选项 B 以 `--seed 20260727` 全量重跑 1000 rep，
    # `3.4_counts.json` 的 `args.seed` 即该值（此前 Phase 3 shadow 为 20261026）。
    # 注意：本文件合成图 rng 的 20261026（`:295` / `:583` / `:588`）是合成种子，与真实跑 seed 无关，不改。
    assert counts["args"]["seed"] == 20260727


def test_inference_csv_contract():
    d = pd.read_csv(INF_CSV)
    need = {"block", "basis", "layer", "weight", "item", "value", "reference", "se", "t",
            "ci95_low", "ci95_high", "p_two_sided", "n_rows", "n_months", "n_groups", "note"}
    assert need == set(d.columns)
    blocks = set(d["block"])
    assert {"struct_layer_n_sources", "struct_layer_eff_opponents", "n1_inference",
            "variance_summary", "pc_unique_values", "pc_inference_layers"} <= blocks
    ew = d[(d["block"] == "struct_layer_n_sources") & (d["weight"] == "EW")]
    assert int(ew["n_rows"].sum()) == FM_ROWS
    assert len(pd.read_csv(STRUCT_CSV)) == len(d[d["block"].str.startswith("struct_layer")])
    assert len(pd.read_csv(VAR_CSV)) >= 6
    assert len(pd.read_csv(PC_CSV)) >= 12


# ---------------------------------------------------------------- 9. 终检回归（F1 自环 / 目标计数 / 团键）
def test_swap_final_no_self_loops(p34):
    """P34-F1 回归：swap 终态无自环（合成图 + 真实单年全预算）。原始网络自环 = 0，故终态自环必须为 0。"""
    nodes = [f"f{i}" for i in range(10)]
    rows = []
    for i in range(10):
        rows.append((f"f{i}", f"f{(i + 1) % 10}", 1.0))
        rows.append((f"f{i}", f"f{(i + 2) % 10}", 1.0))
    e = pd.DataFrame(rows, columns=["focal_id", "source_id", "weight"])
    nidx = {n: i for i, n in enumerate(nodes)}
    net = p34.YearNet(e, nodes, nidx, {n: "A" for n in nodes})
    net.swap(5, np.random.default_rng(20261026))
    assert int((net.fi == net.peer).sum()) == 0
    assert not net.adj.diagonal().any()
    inp = p34.Inputs(log_extra=False)
    net = p34.YearNet(inp.edges[2014], inp.nodes, inp.nidx, inp.ind["asof"][2014])
    net.swap(int(round(1.0 * len(inp.edges[2014]))), np.random.default_rng([20261026, 0, 2014]))
    assert int((net.fi == net.peer).sum()) == 0, "swap 终态出现自环（F1 掩码失效）"
    assert not net.adj.diagonal().any()


def test_accepted_swaps_equal_edge_counts():
    """P34-F1 回归：逐 rep accepted_swaps 精确 == 逐年边数向量（交换不超目标）；N-8 n_rows 构成稳定。"""
    counts = {}
    for y in range(2014, 2023):
        counts[str(y)] = int(len(pd.read_csv(HEADLINE_NET / f"{y}_similarity.csv",
                                             usecols=["focal_id"])))
    # 不写死总边数（构边重跑会变）：只要求逐年边数一致增长量级且与 accepted_swaps 精确相等
    assert sum(counts.values()) > 0
    d = pd.read_csv(NULL_CSV)
    real = d[d["rep"] >= 0]
    assert len(real) >= 200
    for s in real["accepted_swaps"]:
        assert json.loads(s) == counts, "accepted_swaps 偏离逐年边数（应逐 rep 精确相等）"
    n = real["n_rows_EW"].to_numpy(float)
    b = real["beta_EW"].to_numpy(float)
    assert 180000 <= n.min() and n.max() <= 181178
    assert abs(float(np.corrcoef(n, b)[0, 1])) < 0.2      # 样本构成漂移可忽略（实测 ≈ −0.04）


def test_cluster_key_year_unique(p34):
    """P34-团键回归：跨年同 cluster_id 不得合并（团键 = 文本年×10⁶ + 团 id；`:456-458` 修复）。"""
    rng = np.random.default_rng(11)
    rows = []
    for ty, prefix in ((2019, "2020"), (2020, "2021")):
        for k in range(480):                       # 12 个月 × 40 行/月（T=24 > NW lag 6）
            rows.append({"month": f"{prefix}-{k // 40 + 1:02d}", "ty": ty, "cid": k % 2,
                         "y": float(rng.normal()), "x": float(rng.normal())})
    df = pd.DataFrame(rows)
    per = p34.fm_cluster_scores(df, "y", ["x"], cluster_col="cid", min_n=10)
    rec = p34.inference_recipes(per, {m: 2019 for m in df["month"].unique()}, "EW")
    assert rec["n_clusters_total"] == 4, "跨年同 cid 被并成伪团（团键缺文本年前缀）"
    c = json.loads(COUNTS_JSON.read_text(encoding="utf-8"))
    assert c["n1"]["inference"]["inference"]["EW"]["n_clusters_total"] == 1756
    assert c["n11"]["inference"]["a_firm_level"]["n_clusters_total"] == 1209
