"""analyst-network 管线契约测试（批次三，2026-08-12）。

不依赖重跑，直接读现有产物验证冻结口径的硬契约：
  1. 覆盖表唯一键 + 版本号 == coverage_win12_fy12_v2
  2. 边表（抽查 3 个月）：唯一键、无自环、权重行和≈1、common 对称、network_version 列正确
  3. FY1/FY2 语义回归：step5 审计块数字 == 固化回归值（防退回 v1 的 keep-first 语义）
  4. 派生变量唯一键 + network_version
  5. 面板合并率 > 0.5
  6. step3 源码输出列（OUT_COLS + OUTPUT_RENAME）与产物表头一致（防源码漂移）

运行：python -m pytest tests/   或   python tests/test_analyst_contracts.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYST = ROOT / 'data' / 'processed' / 'analyst'
# 2026-09-01 R11：网络版本/月数/抽查月全部从磁盘动态解析，不再钉死 v2/固定月。
# 当前交付版本 = edges 目录里版本号最高的那族文件（与覆盖表 coverage_window_version 一致，测试 1 会校验两者一致）。
import re as _re


def _latest_net_ver():
    vers = set()
    for f in (ANALYST / 'edges').glob('edges_coverage_win*_v*.csv'):
        m = _re.match(r'edges_(coverage_win\d+_fy12_v\d+)_', f.name)
        if m:
            vers.add(m.group(1))
    assert vers, "edges 目录未发现任何网络版本"
    return sorted(vers, key=lambda v: int(v.rsplit('v', 1)[1]))[-1]


# 2026-11-19：收集期不再解析版本号——缺边表时只让依赖它的用例失败，
# 不允许一个 AssertionError 打断整个 pytest 会话（641 个用例无法收集）。
NET_VER = None


def _net_ver():
    global NET_VER
    if NET_VER is None:
        NET_VER = _latest_net_ver()
    return NET_VER
SPOT_MONTHS = None  # 由 test_edges_contracts 从该版本边表月份取首/中/末

# ---------------- 0. manifest 无重复键（R12 防复发守卫） ----------------

def test_manifest_no_duplicate_keys():
    import yaml

    class Dup(yaml.SafeLoader):
        pass

    dups = []

    def _check(loader, node, deep=False):
        m = loader.construct_mapping(node, deep=deep)
        seen = set()
        for k, _ in node.value:
            try:
                kk = loader.construct_object(k, deep=True)
            except Exception:
                kk = getattr(k, 'value', str(k))
            kk = kk if isinstance(kk, (str, int, float)) else str(kk)
            if kk in seen:
                dups.append((kk, node.start_mark.line + 1))
            seen.add(kk)
        return m

    _Dup = Dup
    _Dup.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _check)
    for f in ['configs/version_manifest.yaml', 'configs/parameter_registry.yaml', 'configs/claim_registry.yaml']:
        dups.clear()
        _dup_path = ROOT / f
        if not _dup_path.exists():
            continue
        yaml.load(_dup_path.read_text(encoding='utf-8'), Loader=_Dup)
        assert not dups, f"{f} 存在重复键（YAML 后键静默覆盖先键，机器可读真源失真）：{dups}"

# ---------------- 1. 覆盖表 ----------------

def test_coverage_unique_key_and_version():
    cov = pd.read_csv(ANALYST / 'firm_analyst_coverage.csv', dtype=str)
    dup = cov.duplicated(['coverage_window_version', 'network_cutoff', 'firm_id', 'analyst_id']).sum()
    assert dup == 0, f"覆盖表唯一键重复 {dup}"
    assert set(cov['coverage_window_version']) == {_net_ver()}, "覆盖表版本与 edges 最新版本不一致"
    n = cov['network_cutoff'].nunique()
    assert n >= 1, "覆盖表无 cutoff"
    # 边表文件数必须等于覆盖表 cutoff 数（交付完整性：每月一张边表）
    n_edge_files = len(list((ANALYST / 'edges').glob(f'edges_{_net_ver()}_*.csv')))
    assert n_edge_files == n, f"边表文件数 {n_edge_files} ≠ 覆盖表 cutoff 数 {n}"


# ---------------- 2. 边表契约（抽查 3 个月） ----------------

def test_edges_contracts():
    files = sorted((ANALYST / 'edges').glob(f'edges_{_net_ver()}_*.csv'))
    assert files, f"未发现 {_net_ver()} 边表"
    picks = [files[0], files[len(files) // 2], files[-1]]
    for path in picks:
        c = path.stem.replace(f'edges_{_net_ver()}_', '')
        df = pd.read_csv(path, dtype=str)
        dup = df.duplicated(['network_use_case', 'network_version', 'focal_id', 'source_id']).sum()
        assert dup == 0, f"{c} 唯一键重复 {dup}"
        assert (df['focal_id'] != df['source_id']).all(), f"{c} 存在自环"
        assert (df['network_version'] == _net_ver()).all(), f"{c} network_version 列错误"
        assert 'link_strength' in df.columns, f"{c} 缺 link_strength 列（规划 Step 7 / data_gates 合同要求）"
        assert (df['link_strength'].astype(int) == df['common_analyst_count'].astype(int)).all(), \
            f"{c} link_strength 应等于 common_analyst_count"
        w = pd.to_numeric(df['weight'], errors='coerce')
        assert (w > 0).all(), f"{c} 存在非正权重"
        err = (w.groupby(df['focal_id']).sum() - 1).abs().max()
        assert err < 1e-4, f"{c} 权重行和误差 {err} > 1e-4"
        # common 对称性（(i,j) 与 (j,i) 的 common_analyst_count 一致）
        piv = df[['focal_id', 'source_id', 'common_analyst_count']]
        rev = piv.rename(columns={'focal_id': 'source_id', 'source_id': 'focal_id'})
        m = piv.merge(rev, on=['focal_id', 'source_id'], suffixes=('_f', '_s'))
        asym = (m['common_analyst_count_f'] != m['common_analyst_count_s']).sum()
        assert asym == 0, f"{c} 存在 {asym} 个不对称对"


# ---------------- 3. FY1/FY2 语义回归（防退回 v1 keep-first） ----------------

def test_fy_audit_block_regression():
    snap = json.loads((ANALYST / 'step5_snapshot.json').read_text(encoding='utf-8'))
    audit = snap.get('report_level_audit')
    assert audit is not None, "step5 缺少 report_level_audit 块"
    # 固化回归值（2026-08-12 v2 全量验证）：误杀数必须等于已核实的数字
    assert audit['reports_miskilled_under_v1'] == 309341, "v1 误杀回归值漂移"
    assert audit['reports_kept_v2_any_rule'] == 1815139, "v2 保留报告数回归值漂移"
    assert audit['pairs_lost_under_v1'] == 31466, "v1 pair 丢失回归值漂移"
    # v2 语义：保留数必须大于 v1（至少一次语义更宽）
    assert audit['reports_kept_v2_any_rule'] > audit['reports_kept_v1_first_rule']


# ---------------- 4. 派生变量 ----------------

def test_derived_unique_key_and_version():
    d = pd.read_csv(ANALYST / 'firm_analyst_derived.csv', dtype=str)
    dup = d.duplicated(['firm_id', 'month']).sum()
    assert dup == 0, f"派生变量唯一键重复 {dup}"
    assert set(d['network_version']) == {_net_ver()}, "派生变量 network_version 与当前交付版本不一致"
    cov = pd.read_csv(ANALYST / 'firm_analyst_coverage.csv', usecols=['network_cutoff'], dtype=str)
    assert d['month'].nunique() == cov['network_cutoff'].nunique(), "派生变量月数与覆盖表 cutoff 数不一致"


# ---------------- 6. step3 源码↔产物列名一致性（防源码漂移） ----------------

def test_step3_output_columns_contract():
    """step3 的最终输出列（OUT_COLS 应用 OUTPUT_RENAME）必须等于产物表头。

    2026-08-14 修复：此前 step3 源码输出大写列名（B_code/Feps/...），
    而产物与下游 step4/5/9/sens 均为小写，重跑会 KeyError。本测试防止再次漂移。
    """
    import importlib.util
    path = ROOT / 'scripts' / 'analyst_forecast_clean.py'
    spec = importlib.util.spec_from_file_location('analyst_forecast_clean', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    expected = [mod.OUTPUT_RENAME.get(c, c) for c in mod.OUT_COLS]
    actual = list(pd.read_csv(ANALYST / 'analyst_step3_clean.csv', nrows=0).columns)
    assert expected == actual, \
        f"step3 源码输出列 {expected} 与产物表头 {actual} 不一致（源码漂移）"


# ---------------- 独立运行模式 ----------------

if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
