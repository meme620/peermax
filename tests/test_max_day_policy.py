import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from peermax_max_build import resolve_max_day_policy, select_eligible_days  # noqa: E402

DAYS = [
    ('2020-01-02', 0.05, False, False),
    ('2020-01-03', 0.10, False, True),
    ('2020-01-06', -0.03, True, False),
    ('2020-01-07', None, False, False),
]

EXCLUDE = {'limit_hit': 'exclude', 'resume': 'exclude'}
INCLUDE = {'limit_hit': 'include', 'resume': 'include'}


def test_excludes_limit_and_resume_days() -> None:
    assert select_eligible_days(DAYS, EXCLUDE) == [('2020-01-02', 0.05)]


def test_include_keeps_flagged_days() -> None:
    got = select_eligible_days(DAYS, INCLUDE)
    assert [d for d, _ in got] == ['2020-01-02', '2020-01-03', '2020-01-06']


def test_none_return_dropped() -> None:
    assert all(r is not None for _, r in select_eligible_days(DAYS, INCLUDE))


def test_empty_input() -> None:
    assert select_eligible_days([], EXCLUDE) == []


def test_policy_from_registry() -> None:
    policy = resolve_max_day_policy()
    assert policy['limit_hit'] == 'exclude'
    assert policy['resume'] == 'exclude'
    assert isinstance(policy['listing_grace_days'], int) and policy['listing_grace_days'] >= 0


def test_listing_window_excluded_only_from_la_column() -> None:
    """上市宽限窗口只作用于 la 口径：MAXraw 不受影响，la 口径剔掉窗口内交易日。"""
    days = [('2020-01-02', 0.44, False, False), ('2020-01-03', 0.05, False, False)]
    base = select_eligible_days(days, EXCLUDE)
    la = select_eligible_days(days, EXCLUDE, {'2020-01-02'})
    assert base == [('2020-01-02', 0.44), ('2020-01-03', 0.05)]
    assert la == [('2020-01-03', 0.05)]
