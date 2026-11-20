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
    # K_Econ 与 IQR 都取自 registry（IQR 会随样本版本变，不写死）；
    # β 取 -1.5*K_Econ/IQR 的两倍，保证 beta*IQR 明确越过经济阈值。
    k_econ = g.K_ECON
    beta = -2.0 * k_econ / g.IQR_POOL_REGISTERED
    gset, _ = g.judge(_block(beta, beta * 0.99), _spread(-2.0 * g.K_SPREAD), _series(), None)
    assert gset['economic_magnitude']['pass'] is True
    assert gset['portfolio_spread']['pass'] is True

    # 同量级正号：abs 过、有符号不过
    gpos, _ = g.judge(_block(-beta, -beta * 0.99), _spread(2.0 * g.K_SPREAD), _series(), None)
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


def _rob_rec(p):
    """robustness 生产端的真实记录结构：treatment 系数在 coefs_peer_own_ind 下。"""
    return {'coefs_peer_own_ind': {'PeerMAXraw': {'beta': -0.01, 'p_two_sided': p}}}


def test_bh_block_flags_mismatch_with_design_m():
    rob = {'specs': {f's{i}': _rob_rec(0.01 * (i + 1)) for i in range(5)}}
    blk = g._bh_block(rob)
    assert blk['m'] == 5 and blk['m_registered'] == 19
    assert blk['m_matches_design'] is False
    assert 'warning' in blk
    assert blk['rows'][0]['rank'] == 1


def test_bh_block_reads_producer_key_and_yields_q_values():
    """BH 必须读生产端的 coefs_peer_own_ind（曾读 coefs 导致恒空表）。"""
    rob = {'specs': {f's{i}': _rob_rec(p) for i, p in enumerate(
        [0.001, 0.01, 0.02, 0.04, 0.5] + [0.2 + 0.01 * i for i in range(14)])}}
    blk = g._bh_block(rob)
    assert blk['m'] == 19 and blk['m_matches_design'] is True
    assert blk['n_specs_with_p'] == 19
    assert len(blk['rows']) == 19
    # 单调化：q_(i) = min_{j>=i} p_(j)*m/j → q 随 rank 递增，且 q <= p*m/rank（取前缀最小值）
    qs = [r['q'] for r in blk['rows']]
    assert qs == sorted(qs)
    for r in blk['rows']:
        assert r['q'] <= r['p_two_sided'] * blk['m'] / r['rank'] + 1e-12
    assert blk['n_reject_q'] >= 1


def test_bh_block_excludes_placebo_from_family():
    """安慰剂零模型不是替代规格，不得并入 BH 族（否则污染 q 值）。"""
    specs = {f's{i}': _rob_rec(0.01 * (i + 1)) for i in range(19)}
    specs['pricing_placebo_random'] = _rob_rec(0.9)
    specs['pricing_placebo_timeshift'] = _rob_rec(0.8)
    blk = g._bh_block({'specs': specs})
    assert blk['m'] == 19 and blk['m_matches_design'] is True
    assert set(blk['excluded_from_family']) == {'pricing_placebo_random', 'pricing_placebo_timeshift'}
    assert all(r['spec'].startswith('s') for r in blk['rows'])


def test_bh_block_reports_missing_p_without_silent_drop():
    specs = {f's{i}': _rob_rec(0.01 * (i + 1)) for i in range(18)}
    specs['s_no_p'] = {'spec_status': 'NOT_COMPUTABLE'}
    blk = g._bh_block({'specs': specs})
    assert blk['m'] == 19 and blk['n_specs_with_p'] == 18
    assert blk['p_missing'] == ['s_no_p']
    assert 'p_missing_note' in blk


def test_listingadj_block_absent_when_spec_missing():
    blk = g._listingadj_block({}, {'specs': {}}, {})
    assert blk['status'] == 'NOT_AVAILABLE'
    assert blk['gate_bearing'] is False
