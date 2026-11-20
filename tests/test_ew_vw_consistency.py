"""ew_vw_consistency 与 TOST 的执行面（此前 tests/ 零覆盖）。

TOST 是 ew_vw_consistency 的 AND 条件之一；此处验证判定逻辑本身与月份对齐，
不依赖任何产物。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pricing_gate_report as g  # noqa: E402
from pricing_lib import tost_equivalence  # noqa: E402


def _series(ew, vw, months_ew=None, months_vw=None):
    months_ew = months_ew or [f"2020-{i:02d}" for i in range(1, len(ew) + 1)]
    months_vw = months_vw or [f"2020-{i:02d}" for i in range(1, len(vw) + 1)]
    return {"beta_series_ew": ew, "months_ew": months_ew,
            "beta_series_vw": vw, "months_vw": months_vw}


def test_tost_equivalent_when_diff_tiny():
    """常数序列方差为 0 → NOT_COMPUTABLE 是正确行为；带噪声的小差异才可判等价。"""
    import numpy as np
    rng = np.random.default_rng(7)
    d = 0.001 + rng.normal(0, 0.02, 60)
    out = tost_equivalence(d.tolist(), bound=0.128, alpha=0.05, lag=1)
    assert out["status"] == "PASS" and out["equivalent"] is True


def test_tost_degenerate_constant_series_is_not_computable():
    out = tost_equivalence([0.001] * 40, bound=0.128, alpha=0.05, lag=1)
    assert out["status"] == "NOT_COMPUTABLE" and out["equivalent"] is False


def test_tost_fails_when_diff_exceeds_bound():
    out = tost_equivalence([0.5] * 40, bound=0.128, alpha=0.05, lag=1)
    assert out["equivalent"] is False


def test_tost_not_computable_on_degenerate_series():
    assert tost_equivalence([], bound=0.1, alpha=0.05)["status"] == "NOT_COMPUTABLE"
    assert tost_equivalence([0.01], bound=0.1, alpha=0.05)["status"] == "NOT_COMPUTABLE"


def test_aligned_beta_diff_uses_months_not_positions():
    """VW 缺一月时，按位置 zip 会给出虚假非零差；按月份对齐必须给 0。"""
    s = _series([0.01, 0.02, 0.03, 0.04],
                [0.01, 0.03, 0.04],
                months_ew=["2020-01", "2020-02", "2020-03", "2020-04"],
                months_vw=["2020-01", "2020-03", "2020-04"])
    d, info = g.aligned_beta_diff(s)
    assert d == [0.0, 0.0, 0.0], d
    assert info["n_common_months"] == 3
    assert info["ew_only"] == ["2020-02"]
    assert info["vw_only"] == []


def test_aligned_beta_diff_rejects_length_mismatch():
    bad = {"beta_series_ew": [0.01, 0.02], "months_ew": ["2020-01"],
           "beta_series_vw": [0.01], "months_vw": ["2020-01"]}
    with pytest.raises(SystemExit):
        g.aligned_beta_diff(bad)


def _block(b_ew, b_vw, p_ew=0.01, p_vw=0.01):
    def side(b, p):
        return {"n_months_used": 100,
                "coefs": {"PeerMAXraw": {"beta": b, "se_nw": 0.01, "t_nw": -2.0,
                                         "p_one_sided": p, "p_two_sided": p * 2,
                                         "n_months": 100}}}
    return {"ew": side(b_ew, p_ew), "vw": side(b_vw, p_vw),
            "n_months_used": 100, "n_rows": 1000, "per_month_n": {"min_used": 100}}


def _spread(v):
    return {"spread": {"ew": {"mean": v, "se_nw": 0.01, "t_nw": -2.0, "n_months": 100},
                       "vw": {"mean": v, "se_nw": 0.01, "t_nw": -2.0, "n_months": 100}}}


COLS_OK = {"all_months_full_rank": True, "all_months_residual_df_positive": True,
           "all_months_covariance_finite": True}


def test_ew_vw_consistency_passes_when_close_and_significant():
    import numpy as np
    rng = np.random.default_rng(11)
    s = _series((rng.normal(-0.05, 0.02, 60)).tolist(), (rng.normal(-0.052, 0.02, 60)).tolist())
    gates, _ = g.judge(_block(-0.05, -0.052), _spread(-0.01), s, COLS_OK)
    assert gates["ew_vw_consistency"]["pass"] is True


def test_ew_vw_consistency_fails_on_sign_flip():
    s = _series([-0.05] * 40, [0.05] * 40)
    gates, _ = g.judge(_block(-0.05, 0.05), _spread(-0.01), s, COLS_OK)
    assert gates["ew_vw_consistency"]["sign_same"] is False
    assert gates["ew_vw_consistency"]["pass"] is False


def test_ew_vw_consistency_fails_when_insignificant():
    s = _series([-0.05] * 40, [-0.05] * 40)
    gates, _ = g.judge(_block(-0.05, -0.05, p_ew=0.4, p_vw=0.4), _spread(-0.01), s, COLS_OK)
    assert gates["ew_vw_consistency"]["both_significant"] is False
    assert gates["ew_vw_consistency"]["pass"] is False


def test_fm_status_not_pass_when_no_month_used():
    """空集不得判 PASS（曾经 all(... if used) 在 0 个可用月时恒真）。"""
    s = _series([-0.05] * 5, [-0.05] * 5)
    _, summary = g.judge(_block(-0.05, -0.05), _spread(-0.01), s,
                         {"all_months_full_rank": False,
                          "all_months_residual_df_positive": False,
                          "all_months_covariance_finite": False})
    assert summary["FM_status"] == "FAIL"
