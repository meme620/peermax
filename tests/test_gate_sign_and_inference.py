"""pricing_gate gate 的有符号判定与推断块（CO-2026-11-19-parameters）。

四个不变量：
1. 经济量级门与组合 spread 门用方向一致的判定：β*IQR <= -K_Econ、spread <= -K_SPREAD；abs_pass 只作审计；
2. inference_blocks 四条预注册条款各有一个非空执行面；
3. NW 敏感性在 lag=nw_lag 处与主表一致；
4. BH 的 m 与 design 登记的 19 不一致时必须显式警示。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pricing_gate_report as g  # noqa: E402


def _series(n=24, base=-0.02, noise=0.01):
    ew = [base + (noise if i % 2 == 0 else -noise) for i in range(n)]
    vw = [base + 0.001 + (noise if i % 3 == 0 else -noise) for i in range(n)]
    return {'beta_series_ew': ew, 'beta_series_vw': vw,
            'months_ew': [f'20{i//12:02d}-{i%12+1:02d}' for i in range(n)],
            'months_vw': [f'20{i//12:02d}-{i%12+1:02d}' for i in range(n)]}


def _block(beta_ew, beta_vw, p_ew=0.01, p_vw=0.01):
    return {'n_rows': 1000, 'fm_window': ['2015-05', '2023-12'],
            'per_month_n': {'min_used': 200, 'median_used': 400, 'max_used': 600,
                            'min_all_months': 150, 'months_below_min_n': 0},
            'ew': {'n_months_used': 24, 'coefs': {'PeerMAXraw': {
                'beta': beta_ew, 'se_nw': 0.01, 't_nw': beta_ew / 0.01,
                'p_one_sided': p_ew, 'p_two_sided': p_ew * 2}}},
            'vw': {'n_months_used': 24, 'coefs': {'PeerMAXraw': {
                'beta': beta_vw, 'se_nw': 0.012, 't_nw': beta_vw / 0.012,
                'p_one_sided': p_vw, 'p_two_sided': p_vw * 2}}},
            'all_months_full_rank': True, 'all_months_residual_df_positive': True,
            'all_months_covariance_finite': True}


def _spread(mean):
    return {'spread': {'ew': {'mean': mean, 'se_nw': 0.005, 't_nw': mean / 0.005,
                              'n_months': 24},
                       'vw': {'mean': mean * 0.8, 'se_nw': 0.006, 't_nw': mean / 0.006,
                              'n_months': 24}}}


def test_gate_economic_magnitude_and_spread_require_negative_direction():
    # K_Econ=0.003、IQR=0.0234 → β=-0.15 时 beta*IQR=-0.0035，刚好越过经济阈值
    gset, _ = g.judge(_block(-0.15, -0.148), _spread(-0.004), _series(), None)
    assert gset['economic_magnitude']['pass'] is True
    assert gset['portfolio_spread']['pass'] is True

    # 同量级正号：abs 过、有符号不过
    gpos, _ = g.judge(_block(0.15, 0.148), _spread(0.004), _series(), None)
    assert gpos['economic_magnitude']['abs_pass'] is True
    assert gpos['economic_magnitude']['pass'] is False
    assert gpos['portfolio_spread']['abs_pass'] is True
    assert gpos['portfolio_spread']['pass'] is False


def test_gate_small_negative_magnitude_fails():
    gset, _ = g.judge(_block(-0.02, -0.02), _spread(-0.0005), _series(), None)
    assert gset['economic_magnitude']['pass'] is False
    assert gset['portfolio_spread']['pass'] is False


def test_nw_lag_grid_matches_main_at_registered_lag():
    s = _series()
    # 主表读数须等于序列均值（真实产物满足此关系；这里显式对齐避免造不一致的夹具）
    m_ew = sum(s['beta_series_ew']) / len(s['beta_series_ew'])
    m_vw = sum(s['beta_series_vw']) / len(s['beta_series_vw'])
    price = {'per_month': {'ma_teacher': s},
             'main': {'ma_teacher': _block(m_ew, m_vw)}}
    blk = g._nw_lag_block(price)
    assert blk['grid'] == [0, 3, 6, 12]
    assert set(blk['rows']) == {'ew', 'vw'}
    for w in ('ew', 'vw'):
        assert blk['consistency_with_main'][w]['beta_equal'] is True


def test_vif_block_marks_vars_above_threshold():
    price = {'main': {'ma_teacher': {'vif': {'vif_max': {'peer': 4.0, 'own': 12.0},
                                              'condition_number_max': 6.1, 'n_months': 104}},
                      'mb_teacher': {'vif': {'vif_max': {'peer': 3.0}}}}}
    blk = g._vif_block(price)
    assert blk['threshold'] == 10.0
    assert blk['status'] == 'MARKED_NON_INTERPRETABLE'
    assert blk['columns']['ma_teacher']['exceed_vars'] == ['own']
    assert blk['columns']['ma_teacher']['vif_scope'] == 'per_month_max'


def test_vif_block_pass_when_below_threshold():
    price = {'main': {'ma_teacher': {'vif': {'vif_max': {'peer': 4.0}}}}}
    blk = g._vif_block(price)
    assert blk['status'] == 'PASS'
    assert blk['n_exceed_vars'] == 0


def test_bh_block_flags_mismatch_with_design_m():
    rob = {'specs': {f's{i}': {'coefs': {'PeerMAXraw': {'p_two_sided': 0.01 * (i + 1)}}}
                     for i in range(5)}}
    blk = g._bh_block(rob)
    assert blk['m'] == 5 and blk['m_registered'] == 19
    assert blk['m_matches_design'] is False
    assert 'warning' in blk
    assert blk['rows'][0]['rank'] == 1


def test_listingadj_block_absent_when_spec_missing():
    blk = g._listingadj_block({}, {'specs': {}}, {})
    assert blk['status'] == 'NOT_AVAILABLE'
    assert blk['gate_bearing'] is False
