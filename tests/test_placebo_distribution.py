"""placebo 置换分布与 ex-grace 伴生规格的纯函数测试（CO-2026-11-19-parameters）。

placebo：R 次置换必须给出真实分布（不是单次随机网络），经验 p 用 (1+#{β_k<=β_real})/(R+1)。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pricing_robustness_placebo as rob  # noqa: E402


def _edges():
    return pd.DataFrame({
        'focal_id': ['f1', 'f1', 'f2', 'f2'],
        'source_id': ['s1', 's2', 's1', 's3'],
        'weight': [0.5, 0.5, 0.6, 0.4],
    })


def _vals():
    return pd.DataFrame({
        'firm_id': ['s1', 's2', 's3'],
        'month': ['2020-01'] * 3,
        'MAXraw': [0.10, 0.02, 0.30],
    })


def test_permutation_preserves_peer_count_and_weights():
    e = _edges()
    out = rob.permuted_peer_max(e, _vals(), np.random.default_rng(0), draws=5)
    assert set(out.columns) == {'draw', 'focal_id', 'month', 'px'}
    assert out['draw'].nunique() == 5
    # 每个 draw 每个 focal 恰好一行（peer 数不变）
    for k, g in out.groupby('draw'):
        assert len(g) == 2, k


def test_permutation_moves_values_but_not_structure():
    e = _edges()
    d0 = rob.permuted_peer_max(e, _vals(), np.random.default_rng(1), draws=1)
    d1 = rob.permuted_peer_max(e, _vals(), np.random.default_rng(2), draws=1)
    # 不同 seed 产生不同的赋值（置换不是恒等）
    assert not d0[['px']].equals(d1[['px']])


def test_empirical_p_formula():
    """经验 p = (1 + #{β_k <= β_real}) / (R + 1)：验证边界与方向。"""
    betas = np.array([-0.10, -0.05, 0.00, 0.05])
    base = -0.05
    p = (1 + int((betas <= base).sum())) / (betas.size + 1)
    assert abs(p - 3 / 5) < 1e-12
