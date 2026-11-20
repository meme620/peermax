"""边表内容级不变量（唯一 treatment 源的真实产物）。

只读真实边表：权重行和=1、无自环、无零/负权重、相似度对称、种子与 available_at。
此前 tests/ 对 network_ma_teacher 只有路径字面量断言，边表内容零覆盖——构边写出未归一
权重或把 singleton 混进边表时测试全绿。
"""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
NET_ROOT = ROOT / "data" / "processed"
COLS = ["focal_id", "source_id", "similarity", "weight", "available_at",
        "network_version", "network_support_status"]
TOL = 1e-6


def _years(name):
    d = NET_ROOT / name
    if not d.is_dir():
        return []
    return sorted(int(p.name.split("_")[0]) for p in d.glob("*_similarity.csv"))


@pytest.mark.parametrize("name", ["network_ma_teacher", "network_mb_teacher"])
def test_edge_tables_exist(name):
    years = _years(name)
    if not years:
        pytest.skip(f"缺 {name} 边表（先跑构边）")
    assert years == list(range(2014, 2023)), years


@pytest.mark.parametrize("name", ["network_ma_teacher", "network_mb_teacher"])
def test_weight_rows_sum_to_one(name):
    """行归一：Σ_j w_ij = 1（分母 = 被选同团集合）。"""
    years = _years(name)
    if not years:
        pytest.skip(f"缺 {name} 边表")
    for y in years:
        e = pd.read_csv(NET_ROOT / name / f"{y}_similarity.csv", usecols=["focal_id", "weight"])
        g = e.groupby("focal_id")["weight"].sum()
        assert float((g - 1.0).abs().max()) < TOL, f"{name} {y}: 权重行和偏离 1"


@pytest.mark.parametrize("name", ["network_ma_teacher", "network_mb_teacher"])
def test_no_self_loops_or_nonpositive_weight(name):
    years = _years(name)
    if not years:
        pytest.skip(f"缺 {name} 边表")
    for y in years:
        e = pd.read_csv(NET_ROOT / name / f"{y}_similarity.csv", usecols=COLS)
        assert int((e["focal_id"] == e["source_id"]).sum()) == 0, f"{name} {y}: 出现自环"
        assert int((e["weight"] <= 0).sum()) == 0, f"{name} {y}: 权重 <= 0"
        assert int((e["similarity"] <= 0).sum()) == 0, f"{name} {y}: 相似度 <= 0"
        assert e["focal_id"].notna().all() and e["source_id"].notna().all()


@pytest.mark.parametrize("name", ["network_ma_teacher", "network_mb_teacher"])
def test_available_at_and_version(name):
    """PIT：available_at 恒为 (文本年+1)-04-30；版本串可区分语料。"""
    years = _years(name)
    if not years:
        pytest.skip(f"缺 {name} 边表")
    for y in years:
        e = pd.read_csv(NET_ROOT / name / f"{y}_similarity.csv",
                        usecols=["available_at", "network_version"])
        assert set(e["available_at"].astype(str)) == {f"{y + 1}-04-30"}, (name, y)
        assert set(e["network_version"].astype(str)) ==             {f"{(  'pm_ma_ewcos_teacher_v1' if name == 'network_ma_teacher' else 'pm_mb_tfidf_teacher_v1')}_{y}"}


@pytest.mark.parametrize("name", ["network_ma_teacher", "network_mb_teacher"])
def test_similarity_is_symmetric_on_sampled_pairs(name):
    """相似度对称：抽样 300 条边，反向边相似度必须逐值相等（无向网络的必要条件）。"""
    years = _years(name)
    if not years:
        pytest.skip(f"缺 {name} 边表")
    y = years[len(years) // 2]
    e = pd.read_csv(NET_ROOT / name / f"{y}_similarity.csv",
                    usecols=["focal_id", "source_id", "similarity"])
    sample = e.sample(n=min(300, len(e)), random_state=20261120)
    rev = e.rename(columns={"focal_id": "source_id", "source_id": "focal_id"})
    m = sample.merge(rev, on=["focal_id", "source_id"], how="inner",
                     suffixes=("", "_rev"))
    if m.empty:
        pytest.skip("抽样对在反向表里无匹配（边集非对称，单列网络可能正常）")
    assert float((m["similarity"] - m["similarity_rev"]).abs().max()) < 1e-8, f"{name} {y}: 相似度不对称"


@pytest.mark.parametrize("name", ["network_ma_teacher", "network_mb_teacher"])
def test_support_status_vocabulary(name):
    """support.csv 的 status 取值域恰为 {support, no-network-support}。"""
    years = _years(name)
    if not years:
        pytest.skip(f"缺 {name} 边表")
    for y in years:
        p = NET_ROOT / name / f"{y}_support.csv"
        if not p.is_file():
            pytest.skip(f"缺 {name}/{y}_support.csv")
        s = pd.read_csv(p, usecols=["network_support_status"])
        vals = set(s["network_support_status"].astype(str))
        assert vals <= {"support", "no-network-support"}, f"{name} {y}: status 取值域异常 {vals}"
        assert "support" in vals


@pytest.mark.parametrize("name", ["network_ma_teacher", "network_mb_teacher"])
def test_singleton_focals_absent_from_edge_table(name):
    """singleton 团（no-network-support）不得出现在边表里（无对手 = 无行）。"""
    years = _years(name)
    if not years:
        pytest.skip(f"缺 {name} 边表")
    y = years[0]
    e = pd.read_csv(NET_ROOT / name / f"{y}_similarity.csv", usecols=["focal_id"])
    sp = NET_ROOT / name / f"{y}_support.csv"
    if not sp.is_file():
        pytest.skip(f"缺 {name}/{y}_support.csv")
    s = pd.read_csv(sp, usecols=["focal_id", "network_support_status"])
    ns = set(s.loc[s["network_support_status"] == "no-network-support", "focal_id"])
    assert not (set(e["focal_id"]) & ns), f"{name} {y}: singleton focal 出现在边表"
