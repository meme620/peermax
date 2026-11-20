"""conftest 的"缺产物 → skip"判定自检。

这条判定决定哪些失败被降级为 skip：判宽了会掩盖真实缺陷，判窄了会把
"还没跑"混进真红。只认产物根下的路径，业务断言与 tests/ 自身 fixture 一律不跳。
"""
from pathlib import Path

import pytest

import conftest


def test_file_not_found_under_artifact_root():
    exc = FileNotFoundError(2, 'No such file', str(conftest.ROOT / 'data/processed/x.parquet'))
    p = conftest._missing_artifact_path(exc)
    assert conftest._under_artifact_root(p)


def test_relative_artifact_path():
    msg = '缺少 artifacts/phase2/2.6_counts.json（先跑 shadow 对比脚本）'
    p = conftest._missing_artifact_path(AssertionError(msg))
    assert conftest._under_artifact_root(p)


def test_posixpath_message():
    msg = "PosixPath('/Users/joe/Developer/peermax/data/processed/network_ma_teacher_p3yt')"
    p = conftest._missing_artifact_path(AssertionError(msg))
    assert conftest._under_artifact_root(p)


def test_missing_hint_without_path():
    msg = '缺少 shadow 产物（先运行 shadow_monthly_status.py）'
    p = conftest._missing_artifact_path(AssertionError(msg))
    assert conftest._under_artifact_root(p)


def test_business_assertion_not_skipped():
    """业务断言失败不得被当成缺产物。"""
    for msg in ('assert 0 == 15', 'beta 符号翻转', '行数不匹配（期望 10 实际 9）'):
        p = conftest._missing_artifact_path(AssertionError(msg))
        assert not conftest._under_artifact_root(p), msg


def test_fail_closed_rejection_not_skipped():
    """fail-closed 的业务拒绝（校验失败/不一致/重叠）不得降级成 skip。

    这三类措辞曾被并入 _MISSING_PAT，导致网络侧三条真实守卫被整体静音。
    """
    for msg in (
        '[STOP] 网络版本清单校验失败：net_dir=data/processed/network_ma reason=清单年份缺 2017',
        '校验失败：参数 registry 和 spec 不一致',
        '[STOP] available_at 与规则不符：net_dir=data/processed/network_ma',
        '权重行和误差 3.2e-2 > 1e-4 data/processed/network_ma_teacher/2019_similarity.csv',
    ):
        p = conftest._missing_artifact_path(AssertionError(msg))
        assert not conftest._under_artifact_root(p), f'被误判为缺产物: {msg}'


def test_genuine_missing_artifact_still_skipped():
    """真正的产物缺失仍按 skip 处理（不能被上一条修得过头）。"""
    for msg, want in (
        ('缺少产物 data/processed/x.csv', 'data/processed/x.csv'),
        ('缺 data/processed/pricing_sample_ma_teacher_v5.parquet',
         'data/processed/pricing_sample_ma_teacher_v5.parquet'),
    ):
        p = conftest._missing_artifact_path(AssertionError(msg))
        assert p == want, (msg, p)


def test_path_outside_artifact_root_not_skipped():
    msg = 'No such file or directory: /Users/joe/Developer/peermax/tests/fixture.csv'
    p = conftest._missing_artifact_path(AssertionError(msg))
    assert not conftest._under_artifact_root(p)


def test_require_artifacts_skips_on_missing(tmp_path: Path):
    missing = tmp_path / 'nope.csv'
    with pytest.raises(pytest.skip.Exception):
        conftest.require_artifacts(missing)


def test_require_artifacts_passes_when_present(tmp_path: Path):
    present = tmp_path / 'here.csv'
    present.write_text('a\n', encoding='utf-8')
    conftest.require_artifacts(present)
