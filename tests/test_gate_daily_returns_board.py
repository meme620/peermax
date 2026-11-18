"""数据门 daily-returns-board：日收益表 board 列与代码前缀一致。"""

from pathlib import Path

from peerlab import gates


def _id_map(root: Path, rows) -> None:
    path = root / 'data' / 'processed' / 'firm_id_map.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    body = ['firm_id,code6,name,board,ipo_date,out_date']
    body += [','.join(r) for r in rows]
    path.write_text('\n'.join(body) + '\n', encoding='utf-8')


def _returns(root: Path, code6: str, header, values) -> None:
    path = root / 'data' / 'processed' / 'daily_returns_clean' / (code6 + '.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [','.join(header)]
    body += [','.join(str(v) for v in row) for row in values]
    path.write_text('\n'.join(body) + '\n', encoding='utf-8')


def test_missing_dir_blocked(tmp_path: Path) -> None:
    outcome = gates.check_daily_returns_board(tmp_path)
    assert outcome['blocked'] and outcome['problems'] == []


def test_valid_passes(tmp_path: Path) -> None:
    _id_map(tmp_path, [('sz000001', '000001', '平安银行', 'main', '', ''),
                       ('sh688001', '688001', '华兴源创', 'star', '', '')])
    _returns(tmp_path, '000001', ['date', 'ret', 'board'], [['2020-01-02', '0.01', 'main']])
    _returns(tmp_path, '688001', ['date', 'ret', 'board'], [['2020-01-02', '0.02', 'star']])
    outcome = gates.check_daily_returns_board(tmp_path)
    assert outcome['problems'] == []
    assert outcome['checked'] == 2


def test_wrong_board_value_fails(tmp_path: Path) -> None:
    _id_map(tmp_path, [('sz000001', '000001', '平安银行', 'main', '', '')])
    _returns(tmp_path, '000001', ['date', 'ret', 'board'], [['2020-01-02', '0.01', 'star']])
    problems = gates.check_daily_returns_board(tmp_path)['problems']
    assert any('board 值' in p for p in problems)


def test_duplicate_board_column_fails(tmp_path: Path) -> None:
    _id_map(tmp_path, [('sz000001', '000001', '平安银行', 'main', '', '')])
    _returns(tmp_path, '000001', ['date', 'ret', 'board', 'board'],
             [['2020-01-02', '0.01', 'main', 'main']])
    problems = gates.check_daily_returns_board(tmp_path)['problems']
    assert any('board 列 2 个' in p for p in problems)


def test_map_rule_mismatch_fails(tmp_path: Path) -> None:
    _id_map(tmp_path, [('sz000001', '000001', '平安银行', 'star', '', '')])
    _returns(tmp_path, '000001', ['date', 'ret', 'board'], [['2020-01-02', '0.01', 'star']])
    problems = gates.check_daily_returns_board(tmp_path)['problems']
    assert any('代码前缀规则' in p for p in problems)
