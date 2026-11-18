"""Phase 3.2 合同测试：L-2 / L-6 校准 shadow 变体（逐年 target / 逐年 df / 组合 / latest 对照）。

覆盖（`artifacts/phase3/3.0_plan.md` §2.2 R1/R2/R3 + 3.2 任务书）：
  1. 变体命名：`pm_ma_ewcos_teacher_p3{yt,ydf,yt_ydf,yt_latest}_v1_{y}`，与 headline / 3.1 p3asof 不同名；
     9 年 `{similarity,support,build_report}` + `clusters/`，`available_at = (y+1)-04-30`；
  2. 逐年词表：`lexicon_p3ydf_{y}.csv` 与 `product_lexicon_v2_teacher_df_by_year.csv` 逐词一致、
     行序同 headline、只「救回」被 9 年上确界规则多剔的词（反向多剔 = 0）、
     2014/2015 预登记 560/819 词 + 7.26%/12.08% 命中质量（±0.05pp）；
  3. 逐年 target / ladder：K 目标来自 3.1 交接（容差 0）、K∈[20,4000]、
     |realized median peer − target| ≤ 2（2017 = 5 为阶梯不可达，留档）、2017 平台构造性同 K；
  4. 边表合同：键唯一 / 无自环 / 行内权重和 = 1 / support 与 clusters 两套 peer 读数一致；
  5. headline 未变：9 个 M1 输入锚 MATCH + headline 网络 45 文件 sha pre == post == 3.1 台账；
  6. 对比表 / 效应分解 / 楔子口径 / L2 标签不进入网络构造；
  7. 输出路径守卫：headline 网络 / G0 / configs / 3.1 产物 / headline 词表 → SystemExit(2)。
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/phase3'
SCRIPT = ROOT / 'scripts/shadow_calibration.py'
SHADOW_DIR = ROOT / 'data/processed/phase3_shadow'
HEADLINE_NET = ROOT / 'data/processed/network_ma_teacher'
P3ASOF_NET = ROOT / 'data/processed/network_ma_teacher_p3asof'
HEADLINE_LEXICON = ROOT / 'data/processed/product_lexicon_v2_teacher.csv'
DF_BY_YEAR = ROOT / 'data/processed/product_lexicon_v2_teacher_df_by_year.csv'
YEARS = list(range(2014, 2023))
VARIANTS = ('p3yt', 'p3ydf', 'p3yt_ydf', 'p3yt_latest')
PREFIX = {
    'p3yt': 'pm_ma_ewcos_teacher_p3yt_v1',
    'p3ydf': 'pm_ma_ewcos_teacher_p3ydf_v1',
    'p3yt_ydf': 'pm_ma_ewcos_teacher_p3yt_ydf_v1',
    'p3yt_latest': 'pm_ma_ewcos_teacher_p3yt_latest_v1',
}
HEADLINE_PREFIX = 'pm_ma_ewcos_teacher_v1'
P3ASOF_PREFIX = 'pm_ma_ewcos_teacher_p3asof_v1'
DF_MAX = 0.25
# 3.1 交接的 K 目标
K_ASOF = {2014: 38, 2015: 40, 2016: 44, 2017: 49, 2018: 50, 2019: 52, 2020: 68, 2021: 57, 2022: 64}
K_LATEST = {2014: 26, 2015: 27, 2016: 30, 2017: 38, 2018: 37, 2019: 46, 2020: 52, 2021: 58, 2022: 62}
REGISTERED_RESCUED = {2014: 560, 2015: 819}
REGISTERED_HIT_MASS = {2014: 0.0726, 2015: 0.1208}
HEADLINE_EDGES = {2014: 157096, 2015: 156974, 2016: 152302, 2017: 214028, 2018: 204786,
                  2019: 238458, 2020: 246728, 2021: 249254, 2022: 293938}
HEADLINE_K = {2014: 235, 2015: 238, 2016: 322, 2017: 291, 2018: 334,
              2019: 407, 2020: 414, 2021: 434, 2022: 463}
EDGE_COLS = ['focal_id', 'source_id', 'weight']
# 逐年目标下的已知不可达年（阶梯跳点）：target 49 在 2017 不可达（最近可达 = 44，偏差 5）
PLATEAU_2017 = {'target': 49, 'K': 291, 'realized': 44, 'deviation': 5}


def _load_mod():
    p = str(ROOT / 'scripts')
    if p not in sys.path:
        sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location('shadow_calibration', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def mod():
    return _load_mod()


@pytest.fixture(scope='module')
def counts():
    return json.loads((ART / '3.2_counts.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def comparison():
    return pd.read_csv(ART / '3.2_comparison.csv')


@pytest.fixture(scope='module')
def targets_3_1():
    return json.loads((ART / '3.1_k_targets.json').read_text(encoding='utf-8'))


def _net_dir(v):
    if v == 'headline':
        return HEADLINE_NET
    if v == 'p3asof':
        return P3ASOF_NET
    return ROOT / f'data/processed/network_ma_teacher_{v}'


def _sel_table():
    return pd.read_csv(ART / '3.2_k_selection.csv')


def _sel_row(v, y):
    t = _sel_table()
    r = t[(t['variant'] == v) & (t['text_year'] == y)]
    assert len(r) == 1, (v, y)
    return r.iloc[0]


def _freeze(v, y):
    return json.loads((ART / f'3.2_freeze_{v}_{y}.json').read_text(encoding='utf-8'))


def _report(v, y):
    return json.loads((_net_dir(v) / f'{y}_build_report.json').read_text(encoding='utf-8'))


# ---------------------------------------------------------------- 1 变体命名与目录
def test_variant_registry_naming_and_dirs(mod):
    assert tuple(mod.VARIANT_ORDER) == VARIANTS
    prefixes = set()
    for v in VARIANTS:
        spec = mod.VARIANTS[v]
        assert spec['dir'].name == f'network_ma_teacher_{v}'
        assert spec['dir'].is_dir(), spec['dir']
        assert spec['ver_prefix'] == PREFIX[v]
        # 不得与 headline / 3.1 p3asof 同名（N-9 型同名不同义）
        assert not spec['ver_prefix'].startswith(HEADLINE_PREFIX)
        assert not spec['ver_prefix'].startswith(P3ASOF_PREFIX)
        prefixes.add(spec['ver_prefix'])
    assert len(prefixes) == len(VARIANTS), '变体版本前缀必须互不相同'


def test_variant_matrix_axes(mod):
    """词表 / 目标两轴的分配（plan §2.2 矩阵）。"""
    assert mod.VARIANTS['p3yt']['lexicon'] == 'headline'
    assert mod.VARIANTS['p3yt']['target_kind'] == 'asof_per_year'
    assert mod.VARIANTS['p3ydf']['lexicon'] == 'per_year'
    assert mod.VARIANTS['p3ydf']['target_kind'] == 'pooled_44'
    assert mod.VARIANTS['p3yt_ydf']['lexicon'] == 'per_year'
    assert mod.VARIANTS['p3yt_ydf']['target_kind'] == 'asof_per_year'
    assert mod.VARIANTS['p3yt_latest']['lexicon'] == 'headline'
    assert mod.VARIANTS['p3yt_latest']['target_kind'] == 'latest_per_year'
    # 逐年目标向量与 3.1 交接值逐值一致
    assert mod.K_ASOF_PER_YEAR == K_ASOF and mod.K_LATEST_PER_YEAR == K_LATEST
    assert mod.target_for('p3yt', 2020) == 68 and mod.target_for('p3ydf', 2020) == 44
    assert mod.target_for('p3yt_latest', 2020) == 52
    # 逐年词表路径随年切换；headline 变体恒为 headline 词表
    assert mod.lexicon_for('p3ydf', 2014).name == 'lexicon_p3ydf_2014.csv'
    assert mod.lexicon_for('p3yt_ydf', 2022).name == 'lexicon_p3ydf_2022.csv'
    assert mod.lexicon_for('p3yt', 2014) == HEADLINE_LEXICON


@pytest.mark.parametrize('v', VARIANTS)
def test_shadow_network_files_and_versions(v):
    for y in YEARS:
        d = _net_dir(v)
        for f in (d / f'{y}_similarity.csv', d / f'{y}_support.csv',
                  d / f'{y}_build_report.json', d / 'clusters' / f'{y}_clusters.csv'):
            assert f.is_file(), f
        rep = _report(v, y)
        assert rep['network_version'] == f'{PREFIX[v]}_{y}'
        assert rep['network_version'] != f'{HEADLINE_PREFIX}_{y}'
        assert rep['available_at'] == f'{y + 1}-04-30'
        assert rep['corpus'] == 'teacher'
        assert rep['params']['competitor_set'] == 'same_cluster_members'
        assert rep['params']['K_rule'] == (
            f'per_year_target_peer_match({44 if v == "p3ydf" else (K_ASOF if v != "p3yt_latest" else K_LATEST)[y]})')
        clu = pd.read_csv(d / 'clusters' / f'{y}_clusters.csv', dtype={'firm_id': str})
        assert len(clu) == int(rep['n_nodes'])
        assert clu['cluster_id'].nunique() == int(rep['k_clusters'])


@pytest.mark.parametrize('v', VARIANTS)
def test_version_spec_load_recipe(mod, v):
    """shadow 目录不登记 manifest → 3.6 用注入 version_spec 装载。"""
    from peermax_panel_build import load_versions, select_version
    vers = load_versions(_net_dir(v), version_spec={
        'version_prefix': PREFIX[v], 'years': YEARS, 'available_at_rule': '(y+1)-04-30'})
    assert sorted(vers) == YEARS
    assert all(vers[y]['available_at'] == f'{y + 1}-04-30' for y in YEARS)
    assert select_version(vers, '2015-01') is None
    assert select_version(vers, '2015-05')['text_year'] == 2014
    assert select_version(vers, '2021-06')['text_year'] == 2020
    assert select_version(vers, '2024-01')['text_year'] == 2022


# ---------------------------------------------------------------- 2 逐年词表（L-6）
def test_per_year_lexicon_matches_df_by_year_bitwise():
    """逐年词表 df_ratio = doc_count_y / n_docs_y，且行序与 headline 词表逐行相同。"""
    dy = pd.read_csv(DF_BY_YEAR).set_index('term')
    lex = pd.read_csv(HEADLINE_LEXICON, usecols=['term'])
    for y in YEARS:
        f = SHADOW_DIR / f'lexicon_p3ydf_{y}.csv'
        assert f.is_file(), f
        d = pd.read_csv(f)
        assert list(d.columns) == ['term', 'df_ratio']
        assert len(d) == len(lex) == 1355237
        assert bool((d['term'].astype(str) == lex['term'].astype(str)).all()), f'[{y}] 行序不一致'
        n_docs = float(dy[f'n_docs_{y}'].iloc[0])
        want = (d['term'].map(pd.to_numeric(dy[f'doc_count_{y}'], errors='coerce'))
                .fillna(0.0) / n_docs).to_numpy()
        assert np.max(np.abs(want - d['df_ratio'].to_numpy())) <= 1e-15, f'[{y}] df 与 df_by_year 不符'


def test_per_year_lexicon_rescued_terms_and_hit_mass():
    """plan R1 预登记：2014 = 560 词 / 7.26%，2015 = 819 词 / 12.08%（±0.05pp）；反向多剔恒 0。"""
    lex = pd.read_csv(HEADLINE_LEXICON, usecols=['term', 'df_ratio'])
    dy = pd.read_csv(DF_BY_YEAR).set_index('term')
    keep_h = (lex['df_ratio'].fillna(0.0) <= DF_MAX).to_numpy()
    for y in YEARS:
        d = pd.read_csv(SHADOW_DIR / f'lexicon_p3ydf_{y}.csv')
        keep_y = (d['df_ratio'].to_numpy() <= DF_MAX)
        dc = lex['term'].map(pd.to_numeric(dy[f'doc_count_{y}'], errors='coerce')).fillna(0.0).to_numpy()
        assert int((keep_h & ~keep_y).sum()) == 0, f'[{y}] 逐年词表剔掉了 headline 保留的词'
        rescued_hits = int((~keep_h & keep_y & (dc > 0)).sum())
        mass = float(dc[~keep_h & keep_y].sum() / dc.sum())
        if y in REGISTERED_RESCUED:
            assert rescued_hits == REGISTERED_RESCUED[y], (y, rescued_hits)
            assert abs(mass - REGISTERED_HIT_MASS[y]) <= 0.0005, (y, mass)
    # 2016–2020 命中质量量级（plan §2.2 登记 8.8–10.8%）
    for y in (2016, 2017, 2018, 2019, 2020):
        d = pd.read_csv(SHADOW_DIR / f'lexicon_p3ydf_{y}.csv')
        keep_y = (d['df_ratio'].to_numpy() <= DF_MAX)
        dc = lex['term'].map(pd.to_numeric(dy[f'doc_count_{y}'], errors='coerce')).fillna(0.0).to_numpy()
        mass = float(dc[~keep_h & keep_y].sum() / dc.sum())
        assert 0.08 <= mass <= 0.12, (y, mass)


def test_df_removed_terms_csv_contents():
    p = ART / '3.2_df_removed_terms.csv'
    assert p.is_file()
    d = pd.read_csv(p)
    assert set(d['text_year'].unique()) == {2014, 2015}
    assert len(d) == 560 + 819
    for col in ('term', 'doc_count_y', 'df_ratio_y', 'headline_df_ratio_sup', 'share_of_year_hit_mass'):
        assert col in d.columns
    assert bool((d['df_ratio_y'] <= DF_MAX).all())
    assert bool((d['headline_df_ratio_sup'] > DF_MAX).all())
    assert abs(float(d.groupby('text_year')['share_of_year_hit_mass'].sum()[2014]) - 0.0726) < 5e-4


def test_lexicon_counts_in_counts_json(counts):
    lex = counts['lexicons']
    assert lex['df_max'] == DF_MAX and lex['lexicon_source_rows'] == 1355237
    assert lex['max_abs_diff_reload'] <= 1e-15
    for y in YEARS:
        d = lex['per_year'][str(y)]
        assert d['terms_extra_dropped_vs_headline'] == 0
        assert d['terms_added_vs_headline_all'] >= d['terms_added_vs_headline_with_hits'] > 0
        assert d['terms_kept_per_year_df'] > d['terms_kept_headline']
    g = lex['registered_gate']
    assert g['2014']['terms_added_with_hits'] == 560 and g['2014']['pass']
    assert g['2015']['terms_added_with_hits'] == 819 and g['2015']['pass']
    assert g['2014']['share_pass'] and g['2015']['share_pass']


def test_headline_lexicon_and_df_table_untouched():
    """只读输入未被改写：headline 词表 1,355,237 行 / df_by_year 194,417 行（sha 与 3.1 台账一致）。"""
    pre = json.loads((ART / '3.2_anchor_pre.json').read_text(encoding='utf-8'))
    post = json.loads((ART / '3.2_anchor_post.json').read_text(encoding='utf-8'))
    for k, name in (('product_lexicon_v2_teacher', 'product_lexicon_v2_teacher'),
                    ('product_lexicon_v2_teacher_df_by_year',
                     'product_lexicon_v2_teacher_df_by_year')):
        assert pre['extra_inputs'][k]['sha256'] == post['extra_inputs'][k]['sha256']
    anc31 = json.loads((ART / '3.1_anchor_post.json').read_text(encoding='utf-8'))
    assert (anc31['extra_inputs']['product_lexicon_v2_teacher']['sha256']
            == post['extra_inputs']['product_lexicon_v2_teacher']['sha256'])


# ---------------------------------------------------------------- 3 逐年 target / ladder
def test_k_targets_match_3_1_handoff(counts, targets_3_1):
    got = {int(p['text_year']): int(round(p['median_peer'])) for p in targets_3_1['asof']['per_year']}
    assert got == K_ASOF
    got_l = {int(p['text_year']): int(round(p['median_peer'])) for p in targets_3_1['latest']['per_year']}
    assert got_l == K_LATEST
    assert counts['k_targets_3_1']['verified_against_3_1_json'] is True
    assert counts['k_targets_3_1']['asof_pooled'] == 49


@pytest.mark.parametrize('v', VARIANTS)
def test_k_range_and_peer_target_deviation(v, counts):
    """K ∈ [20,4000]；|realized median peer − target| ≤ 2；超出容差的 8 个 (变体,年) 必须**逐条**满足
    「目标在扩展阶梯窗口内不可达（`target_reachable_in_window=False`）且所选 K 的偏差 =
    窗口内最小可达偏差」——即偏离是阶梯平台（跳点）造成的构造性结果，不是选 K 失败。"""
    for y in YEARS:
        fz = _freeze(v, y)
        st = fz['years'][str(y)]
        target = fz['target_peer']
        expect = 44 if v == 'p3ydf' else (K_ASOF if v != 'p3yt_latest' else K_LATEST)[y]
        assert target == expect, (v, y, target, expect)
        assert 20 <= int(st['K']) <= 4000
        dev = abs(float(st['peer_median']) - target)
        sel = _sel_row(v, y)
        assert abs(float(sel['dev_selected']) - dev) <= 1e-9, (v, y)     # freeze 实况 = 选择表留档
        assert int(sel['K_selected']) == int(st['K']) == int(_report(v, y)['k_clusters'])
        if dev <= 2:
            continue
        assert counts['flag_counts']['median_peer_off_target_gt2'] >= 1
        assert not bool(sel['target_reachable_in_window']), (v, y, '超容差必须逐条解释为不可达')
        assert abs(float(sel['best_attainable_dev_in_window']) - dev) <= 1e-9, (v, y)
        assert bool(sel['needs_rebuild']) is False, '无更优 K 可换 → 不得重建'
        if str(st['K_mode']).endswith('exact'):
            assert dev == 0


def test_k_deviation_explained_by_ladder_plateau():
    """8 个超容差 (变体,年) 的阶梯证据：跳点跨越目标（前值 > target > 后值 或反向）。"""
    bad = _sel_table()
    bad = bad[bad['dev_selected'] > 2]
    assert len(bad) == 8, sorted(zip(bad['variant'], bad['text_year']))
    assert set(zip(bad['variant'], bad['text_year'])) == {
        ('p3yt', 2017), ('p3yt', 2020), ('p3yt', 2022), ('p3ydf', 2020),
        ('p3yt_ydf', 2015), ('p3yt_ydf', 2022), ('p3yt_latest', 2016), ('p3yt_latest', 2017)}
    for _, r in bad.iterrows():
        K = int(r['K_selected'])
        mode = str(r['lexicon_mode'])
        d = pd.read_csv(ART / f'3.2_k_ladder_{mode}.csv')
        s = d[d['text_year'] == int(r['text_year'])].set_index('K')['median_peer']
        t = int(r['target'])
        assert t not in set(s.values), (r['variant'], r['text_year'], '窗口内不应有命中值')
        left = {k: v for k, v in s.items() if k <= K}
        right = {k: v for k, v in s.items() if k > K}
        # 跳点：所选 K 及其左侧与右侧分别位于目标两侧（构造性跨越）
        assert max(left.values()) > t > min(right.values()), (r['variant'], r['text_year'])


def test_extended_ladder_reachability(counts):
    lad = counts.get('k_ladder') or {}
    assert set(lad) == {'headline', 'per_year'}, sorted(lad)
    for mode, doc in lad.items():
        assert doc['rows'] > 0 and doc['window'] == 40
        for y in YEARS:
            r = doc['reachability'][str(y)]
            assert r['lexicon_mode'] == mode
            assert r['n_K_scanned'] > 0 and r['reference_Ks'], (mode, y)
            for v, e in r['variants'].items():
                assert e['K_selected'] is not None and e['K_freeze'] is not None
                assert e['mode_selected'] in ('exact', 'ladder_closest')
                assert isinstance(e['K_at_target_in_window'], list)
                assert isinstance(e['closest_attainable_median_peer'], list)
                assert e['window_final'][0] <= e['K_selected'] <= e['window_final'][1]
                if e['mode_selected'] == 'exact':
                    assert len(e['K_at_target_in_window']) > 0
                    assert e['K_selected'] in e['K_at_target_in_window']
                    assert e['dev_selected'] == 0
    # headline 词表阶梯：2017 的 as-of 目标 49 在扩展窗口内仍不可达（跳点 61→44）
    e2017 = lad['headline']['reachability']['2017']['variants']['p3yt']
    assert e2017['target'] == 49 and e2017['target_reachable_in_window'] is False
    assert e2017['dev_selected'] == 5 and e2017['needs_rebuild'] is False
    assert 44.0 in e2017['closest_attainable_median_peer']
    # 阶梯表逐行落盘
    for mode in ('headline', 'per_year'):
        p = ART / f'3.2_k_ladder_{mode}.csv'
        assert p.is_file()
        d = pd.read_csv(p)
        assert len(d) == lad[mode]['rows']
        assert set(d['lexicon_mode'].unique()) == {mode}
        assert {'text_year', 'K', 'median_peer', 'size_max', 'n_clusters'} <= set(d.columns)


def test_k_selection_two_tier_table():
    """两级 K 选择留档：一级 freeze 自校准；二级仅对「一级 |dev| > 2 且阶梯上有更优 K」的年重选。
    本仓库实况 = 0 个二级重建（freeze 的 K 在 36/36 对上已是窗口内最优）→ 该结论本身须留档。"""
    sel = _sel_table()
    assert len(sel) == len(VARIANTS) * len(YEARS)
    assert set(sel['variant']) == set(VARIANTS)
    assert set(sel['selection_rule'].unique()) == {
        'freeze_first（一级命中且 |dev| ≤ 2 → 保留 freeze K）；'
        'ladder_reselect（一级 |dev| > 2 → 扩展阶梯重选）'}
    rb = sel[sel['needs_rebuild'].astype(bool)]
    assert len(rb) == 0, '本实况：freeze K 已是窗口内最优 → 无需二级重建（若 >0 则应有实际重建记录）'
    for _, r in sel.iterrows():
        assert int(r['K_selected']) == int(r['K_freeze'])
        assert abs(float(r['dev_selected']) - float(r['dev_freeze'])) <= 1e-9
        # 所选 K = 窗口内 |dev| 最小的 K
        assert abs(float(r['dev_selected']) - float(r['best_attainable_dev_in_window'])) <= 1e-9
        assert int(r['K_selected']) > 0
        if float(r['dev_selected']) > 2:
            assert not bool(r['target_reachable_in_window'])


def test_2017_plateau_is_constructional_for_asof_target_49():
    """headline 词表下 as-of 目标 49 在 2017 阶梯不可达（K=291 处 61→44）→ p3yt 与 headline 同 K（边集
    逐值相同）；逐年 df 词表改变了阶梯 → p3yt_ydf 2017 达到 49（exact，K≠291）——平台是「词表 × 目标」性质。"""
    p3yt_2017 = _freeze('p3yt', 2017)['years']['2017']
    assert int(p3yt_2017['K']) == PLATEAU_2017['K'] == 291
    assert float(p3yt_2017['peer_median']) == PLATEAU_2017['realized'] == 44.0
    assert str(p3yt_2017['K_mode']) == 'closest'
    assert int(_report('p3yt', 2017)['k_clusters']) == 291
    assert int(_report('headline', 2017)['k_clusters']) == 291
    cmp_row = pd.read_csv(ART / '3.2_comparison.csv')
    row = cmp_row[(cmp_row['network'] == 'p3yt') & (cmp_row['text_year'] == 2017)].iloc[0]
    assert float(row['jaccard_vs_headline']) == 1.0 and float(row['peer_minus_target']) == -5.0
    yd = _freeze('p3yt_ydf', 2017)['years']['2017']
    assert float(yd['peer_median']) == 49.0 and str(yd['K_mode']).endswith('exact')
    assert int(yd['K']) != 291
    hits = [r for r in pd.read_csv(ART / '3.1_k_ladder.csv').to_dict('records')
            if r['text_year'] == 2017 and r['median_peer'] == 49]
    assert hits == [], '3.1 阶梯窗口内 2017 的 49 应不可达（留档）'


def test_per_year_lexicon_filter_stats_differ_only_via_terms(counts):
    """逐年词表 vs headline 词表：文档数/剔除判定同，活跃词表更大；入网公司数不减（仅可能因
    MIN_TERMS=10 门槛被多救回的词「捞回」个别公司，如 2018 年 +1 家）。"""
    nf = counts['networks_from_disk']['variants']
    fs_head, fs_y = {}, {}
    for y in YEARS:
        fs_head[y] = _freeze('p3yt', y)['years'][str(y)]['filter_stats']
        fs_y[y] = _freeze('p3yt_ydf', y)['years'][str(y)]['filter_stats']
    for y in YEARS:
        assert fs_head[y]['docs'] == fs_y[y]['docs']
        assert fs_head[y]['drop_menlei'] == fs_y[y]['drop_menlei']
        assert fs_head[y]['drop_ipo'] == fs_y[y]['drop_ipo']
        assert fs_y[y]['active_vocab'] > fs_head[y]['active_vocab']
        assert fs_y[y]['firms_kept'] >= fs_head[y]['firms_kept']
        assert fs_y[y]['firms_kept'] - fs_head[y]['firms_kept'] <= 5, (y, fs_y[y]['firms_kept'])
        assert fs_head[y]['drop_min'] >= fs_y[y]['drop_min']
    assert nf['p3yt']['years']['2014']['lexicon_terms_kept'] == 1353479
    assert nf['p3yt_ydf']['years']['2014']['lexicon_terms_kept'] == 1354043


# ---------------------------------------------------------------- 4 边表合同（抽样全 4 变体 × 2014/2022）
@pytest.mark.parametrize('v', VARIANTS)
@pytest.mark.parametrize('y', [2014, 2022])
def test_edge_table_contract(v, y):
    d = _net_dir(v)
    rep = _report(v, y)
    e = pd.read_csv(d / f'{y}_similarity.csv', usecols=EDGE_COLS, dtype={'focal_id': str, 'source_id': str})
    assert len(e) == int(rep['n_edges'])
    assert int(e.duplicated(['focal_id', 'source_id']).sum()) == 0
    assert int((e['focal_id'] == e['source_id']).sum()) == 0
    assert bool((e['weight'] > 0).all())
    ws = e.groupby('focal_id')['weight'].sum()
    assert float(np.max(np.abs(ws.to_numpy() - 1.0))) <= 1e-9
    sup = pd.read_csv(d / f'{y}_support.csv', dtype={'focal_id': str})
    assert len(sup) == int(rep['n_nodes'])
    assert int((sup['network_support_status'] == 'support').sum()) == int(rep['n_support'])
    assert int(sup['n_edges'].sum()) == len(e), 'support n_edges 之和应等于边数'
    if y in (2016, 2018, 2020, 2022):
        assert bool((sup['available_at'] == f'{y + 1}-04-30').all())


def test_edge_counts_match_comparison_table(comparison):
    for v in VARIANTS:
        sub = comparison[comparison['network'] == v]
        assert len(sub) == len(YEARS)
        for _, r in sub.iterrows():
            rep = _report(v, int(r['text_year']))
            assert int(r['n_edges']) == int(rep['n_edges'])
            assert int(r['K']) == int(rep['k_clusters'])
            assert 0.0 <= float(r['jaccard_vs_headline']) <= 1.0
            assert 0.0 <= float(r['jaccard_vs_p3asof']) <= 1.0
            assert float(r['lift_asof']) > 1.0 and float(r['lift_latest']) > 1.0
            assert float(r['isolated_share']) < 0.15


# ---------------------------------------------------------------- 5 headline 未变
def test_m1_anchors_match_pre_and_post():
    pre = json.loads((ART / '3.2_anchor_pre.json').read_text(encoding='utf-8'))
    post = json.loads((ART / '3.2_anchor_post.json').read_text(encoding='utf-8'))
    assert len(pre['m1_inputs_9']) == 9 and len(post['m1_inputs_9']) == 9
    assert all(r['match'] for r in pre['m1_inputs_ledger'])
    assert all(r['match'] for r in post['m1_inputs_ledger'])
    assert pre['m1_inputs_9'] == post['m1_inputs_9']
    anc31 = json.loads((ART / '3.1_anchor_post.json').read_text(encoding='utf-8'))
    assert pre['m1_inputs_9'] == anc31['m1_inputs_9']


def test_headline_network_45_files_sha_unchanged():
    pre = json.loads((ART / '3.2_anchor_pre.json').read_text(encoding='utf-8'))
    post = json.loads((ART / '3.2_anchor_post.json').read_text(encoding='utf-8'))
    assert pre['headline_network_n_files'] == post['headline_network_n_files'] == 45
    assert pre['headline_network_sha'] == post['headline_network_sha']
    anc31 = json.loads((ART / '3.1_anchor_post.json').read_text(encoding='utf-8'))
    assert post['headline_network_sha'] == anc31['headline_network_sha']
    assert set(post['headline_network_sha']) == {
        p.relative_to(ROOT).as_posix() for p in HEADLINE_NET.rglob('*') if p.is_file()}


def test_headline_reference_rows_in_comparison_unchanged(comparison):
    sub = comparison[comparison['network'] == 'headline']
    assert len(sub) == len(YEARS)
    for _, r in sub.iterrows():
        y = int(r['text_year'])
        assert int(r['n_edges']) == HEADLINE_EDGES[y]
        assert int(r['K']) == HEADLINE_K[y]
        assert float(r['jaccard_vs_headline']) == 1.0
        assert r['network_version'] == f'{HEADLINE_PREFIX}_{y}'


def test_p3asof_reference_rows_read_only(comparison):
    sub = comparison[comparison['network'] == 'p3asof']
    assert len(sub) == len(YEARS)
    for _, r in sub.iterrows():
        y = int(r['text_year'])
        assert r['network_version'] == f'{P3ASOF_PREFIX}_{y}'
        assert int(r['jaccard_vs_p3asof']) == 1.0


# ---------------------------------------------------------------- 6 对比 / 效应分解 / 楔子
def test_comparison_table_shape_and_columns(comparison):
    assert len(comparison) == 6 * len(YEARS)      # 2 参照 + 4 变体
    assert set(comparison['network']) == {'headline', 'p3asof'} | set(VARIANTS)
    for col in ('isolated_effect', 'l2_mapping', 'lexicon', 'k_target', 'K', 'n_edges', 'density',
                'peer_median', 'peer_p25', 'peer_p90', 'isolated_share', 'jaccard_vs_headline',
                'jaccard_vs_p3asof', 'lift_latest', 'lift_asof', 'wedge_vs_asof_ind',
                'wedge_vs_latest_ind', 'peer_minus_target'):
        assert col in comparison.columns, col
    for v in VARIANTS:
        eff = comparison[comparison['network'] == v]['isolated_effect'].unique()
        assert len(eff) == 1 and eff[0], v
    assert comparison[comparison['network'] == 'p3ydf']['k_target'].unique().tolist() == [44]


def test_wedge_csv_semantics():
    w = pd.read_csv(ART / '3.2_wedge.csv')
    assert len(w) == 6 * len(YEARS)
    assert {'network', 'text_year', 'network_peer_median', 'K', 'asof_ind_median_peer_y',
            'latest_ind_median_peer_y', 'wedge_vs_asof_ind', 'wedge_vs_latest_ind',
            'numerator_rule', 'denominator_rule'} <= set(w.columns)
    for _, r in w.iterrows():
        assert abs(float(r['wedge_vs_asof_ind'])
                   - float(r['network_peer_median']) / K_ASOF[int(r['text_year'])]) < 5e-4
        assert abs(float(r['wedge_vs_latest_ind'])
                   - float(r['network_peer_median']) / K_LATEST[int(r['text_year'])]) < 5e-4
    # 逐年目标变体：wedge 偏离 1 的量 = 自身 K 校准偏差 / 目标（构造性；偏差逐条取自 3.2_k_selection.csv）
    sel = _sel_table().set_index(['variant', 'text_year'])
    for net, denom, col in (('p3yt', K_ASOF, 'wedge_vs_asof_ind'),
                            ('p3yt_ydf', K_ASOF, 'wedge_vs_asof_ind'),
                            ('p3yt_latest', K_LATEST, 'wedge_vs_latest_ind')):
        sub = w[w['network'] == net]
        for _, r in sub.iterrows():
            y = int(r['text_year'])
            dev = float(sel.loc[(net, y), 'dev_selected'])
            assert abs(float(r[col]) - 1.0) <= dev / denom[y] + 1e-3, (net, y, r[col], dev)
    # 楔子读数：headline（分子 = headline 网络）复现 plan §2.2 预投影 1.69→0.71 / 1.16→0.69
    hl = w[w['network'] == 'headline'].sort_values('text_year')
    assert abs(hl.iloc[0]['wedge_vs_latest_ind'] - 1.6923) < 5e-4
    assert abs(hl.iloc[-1]['wedge_vs_latest_ind'] - 0.7097) < 5e-4
    assert abs(hl.iloc[0]['wedge_vs_asof_ind'] - 1.1579) < 5e-4
    assert abs(hl.iloc[-1]['wedge_vs_asof_ind'] - 0.6875) < 5e-4


def test_effect_decomposition_structure(counts):
    eff = counts['effect_decomposition']
    pairs = eff['pairs']
    assert set(pairs) == {'lexicon_df', 'k_target_per_year_asof', 'k_target_per_year_latest', 'combined',
                          'lexicon_df_at_per_year_target', 'k_target_effect_at_yearly_df',
                          'target_mapping', 'pooled_vs_per_year'}
    for k, p in pairs.items():
        assert len(p['per_year']) == len(YEARS)
        assert p['edges_total_lhs'] > 0 and p['edges_total_rhs'] > 0
        assert np.isfinite(p['edge_delta_pct_total'])
    assert 'additivity_residual_pp' in eff and 'combined_minus_df_minus_target' in eff['additivity_residual_pp']
    jc = eff['jaccard_cache']
    assert len(jc) == 8
    for k, d in jc.items():
        assert d['median_jaccard'] is not None and 0.0 <= d['median_jaccard'] <= 1.0
    assert 'label_axis_note' in eff and 'wedge_note' in eff
    # 词表效应（同目标 44）与目标效应（同词表）都应可辨识：Jaccard < 1
    assert jc['p3ydf_vs_headline']['median_jaccard'] < 1.0
    assert jc['p3yt_vs_headline']['median_jaccard'] < 1.0


def test_l2_labels_do_not_enter_network_construction(mod):
    """构造性证明（源码级）：网络构造路径不接收行业标签参数；标签只进 K 目标与 lift。

    证据 1：`build_vectors` 签名无标签参数；`calibrate_K` 的 l2 参数只用于 lift 基数。
    证据 2：同词表两变体（p3yt / p3yt_latest）节点集合与 filter_stats 逐值相同，
            仅 K（= 目标）不同 → 标签轴在网络层无可辨识通道。
    """
    import inspect
    from network_business_cluster_grid import build_vectors
    from network_business_cluster_freeze import calibrate_K
    sig = str(inspect.signature(build_vectors))
    assert 'l2' not in sig and 'industry' not in sig and 'sw2' not in sig
    assert 'l2' not in str(inspect.signature(calibrate_K)).replace('l2_idx', '').replace('n_l2', '')
    for y in (2014, 2018, 2022):
        a = pd.read_csv(_net_dir('p3yt') / 'clusters' / f'{y}_clusters.csv', dtype={'firm_id': str})
        b = pd.read_csv(_net_dir('p3yt_latest') / 'clusters' / f'{y}_clusters.csv', dtype={'firm_id': str})
        assert list(a['firm_id']) == list(b['firm_id']), f'[{y}] 同词表变体节点集合/顺序应逐值相同'
        fa = _freeze('p3yt', y)['years'][str(y)]['filter_stats']
        fb = _freeze('p3yt_latest', y)['years'][str(y)]['filter_stats']
        assert fa == fb, f'[{y}] 同词表变体 filter_stats 应逐值相同'
    # 两变体目标不同 → K 或团划分不同（至少一年不同）
    same = all(int(_freeze('p3yt', y)['years'][str(y)]['K'])
               == int(_freeze('p3yt_latest', y)['years'][str(y)]['K']) for y in YEARS)
    assert not same, '逐年 as-of 与 latest 目标应在至少一年给出不同 K'


def test_acceptance_flags_and_counts(counts):
    fl = counts['acceptance_flags']
    assert len(fl) == 6 * len(YEARS)
    for f in fl:
        for bad in ('K_out_of_range_20_4000', 'isolated_share_over_0.15'):
            assert bad not in f['flags'], (f, bad)
    nd = counts['networks_from_disk']
    assert set(nd['variants']) == set(VARIANTS)
    for v in VARIANTS:
        assert nd['variants'][v]['n_years'] == len(YEARS)
    assert set(nd['version_spec_check']) == set(VARIANTS)


def test_identical_lexicon_and_K_implies_identical_network(mod):
    """同 (词表, K) ⇒ 边集**逐值相同**：L2 标签 / 目标来源不进入网络构造。

    - `p3yt` 2021 ≡ `p3yt_latest` 2021（K=335）、2022（K=334）：**不同目标映射 → 同 K → 同网络**；
    - `p3yt_ydf` 2016 ≡ `p3ydf` 2016（K=383）：同词表 + 同目标 44；
    - `p3yt` 2016 ≡ `headline` 2016（K=322）、`p3yt` 2017 ≡ `headline` 2017（K=291）：
      该年逐年 as-of 目标与池化 44 落到同一 K（2017 是阶梯平台）。
    """
    cmap = mod.build_code_map()
    cache = {}

    def keys(net, y):
        if (net, y) not in cache:
            e = pd.read_csv(_net_dir(net) / f'{y}_similarity.csv', usecols=EDGE_COLS,
                            dtype={'focal_id': str, 'source_id': str})
            cache[(net, y)] = np.unique(mod.edge_keys(e, cmap))
        return cache[(net, y)]

    pairs = [(('p3yt', 2021), ('p3yt_latest', 2021)), (('p3yt', 2022), ('p3yt_latest', 2022)),
             (('p3yt_ydf', 2016), ('p3ydf', 2016)), (('p3yt', 2016), ('headline', 2016)),
             (('p3yt', 2017), ('headline', 2017))]
    for a, b in pairs:
        ka, kb = keys(*a), keys(*b)
        assert len(ka) == len(kb) > 0, (a, b, len(ka), len(kb))
        assert bool(np.array_equal(ka, kb)), (a, b, '同 (词表,K) 的边集应逐值相同')


# ---------------------------------------------------------------- 7 输出路径守卫
def test_output_path_guard_blocks_headline_g0_configs_and_31(mod, tmp_path):
    banned = [
        HEADLINE_NET / '2014_similarity.csv',
        ROOT / 'reports/data_gates/G0/s3_b3_ma_param_freeze_teacher.json',
        ROOT / 'configs/parameter_registry.yaml',
        ART / '3.1_counts.json',
        ART / '3.0_plan.md',
        HEADLINE_LEXICON,
        ROOT / 'data/processed/sw2_industry_map.csv',
        ROOT / 'tests/test_shadow_calibration.py',
        ROOT / 'docs/methodology/methodology.md',
    ]
    for p in banned:
        with pytest.raises(SystemExit) as ei:
            mod.assert_output_path_guard(p)
        assert ei.value.code == 2, p
    # 允许：4 个 shadow 目录 / phase3_shadow / artifacts/phase3 的 3.2_*
    for p in [SHADOW_DIR / 'lexicon_p3ydf_2014.csv', _net_dir('p3yt') / '2014_similarity.csv',
              _net_dir('p3yt_latest') / 'clusters' / '2014_clusters.csv',
              ART / '3.2_counts.json']:
        assert mod.assert_output_path_guard(p) == Path(p).resolve()


def test_sha_ledger_rerun_identical_if_present():
    p = ART / '3.2_output_sha.json'
    assert p.is_file()
    d = json.loads(p.read_text(encoding='utf-8'))
    assert d['n_files'] >= 4 * 9 * 3 + 9
    if 'rerun_check' in d:
        assert d['rerun_check']['status'] == 'IDENTICAL', d['rerun_check']['diff_keys'][:10]
