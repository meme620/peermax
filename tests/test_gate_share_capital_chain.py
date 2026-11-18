"""数据门 share-capital-chain：送转展开链自洽。"""

from pathlib import Path

from peerlab import gates

HEADER = ['code', 'year', 'end_share', 'method', 'status', 'source']


def _bonus(root: Path, code: str, ratio: str, ex_date: str) -> None:
    path = root / 'data' / 'raw' / 'eastmoney' / 'sharebonus' / (code + '.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    body = ['BONUS_IT_RATIO,EX_DIVIDEND_DATE,PLAN_NOTICE_DATE', ratio + ',' + ex_date + ',2020-05-20']
    path.write_text('\n'.join(body) + '\n', encoding='utf-8')


def _shares(root: Path, rows) -> None:
    path = root / 'data' / 'processed' / 'share_capital_history_v2.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [','.join(HEADER)] + [','.join(str(v) for v in row) for row in rows]
    path.write_text('\n'.join(body) + '\n', encoding='utf-8')


def _rows(year_2020='110', method_2020='bonus_back', status_2020='ok', status_2021='ok'):
    return [['000001', '2019', '100', 'bonus_back', 'ok', 'anchor_A@2019'],
            ['000001', '2020', year_2020, method_2020, status_2020, 'anchor_A@2019'],
            ['000001', '2021', '110', 'bonus_back', status_2021, 'anchor_A@2019']]


def test_missing_inputs_blocked(tmp_path: Path) -> None:
    outcome = gates.check_share_capital_chain(tmp_path)
    assert outcome['blocked'] and outcome['problems'] == []


def test_valid_passes(tmp_path: Path) -> None:
    _bonus(tmp_path, '000001', '1', '2020-06-01')
    _shares(tmp_path, _rows())
    outcome = gates.check_share_capital_chain(tmp_path)
    assert outcome['problems'] == []
    assert outcome['checked'] == 3


def test_broken_expansion_fails(tmp_path: Path) -> None:
    _bonus(tmp_path, '000001', '1', '2020-06-01')
    _shares(tmp_path, _rows(year_2020='120'))
    problems = gates.check_share_capital_chain(tmp_path)['problems']
    assert any('展开应为' in p for p in problems)


def test_needs_review_row_skipped(tmp_path: Path) -> None:
    _bonus(tmp_path, '000001', '1', '2020-06-01')
    _shares(tmp_path, _rows(year_2020='120', method_2020='A', status_2020='needs_review'))
    problems = gates.check_share_capital_chain(tmp_path)['problems']
    assert not any('展开应为' in p for p in problems)


def test_unregistered_status_fails(tmp_path: Path) -> None:
    _bonus(tmp_path, '000001', '1', '2020-06-01')
    _shares(tmp_path, _rows(status_2021='manual_fix'))
    problems = gates.check_share_capital_chain(tmp_path)['problems']
    assert any('status' in p for p in problems)
