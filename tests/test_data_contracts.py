import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peerlab import data_contracts as dc

CODES = ['000001.SZ', '600000.SH']
DATES = pd.to_datetime(['2020-01-02', '2020-01-03', '2020-01-06'])


def _status(outcome: dict) -> str:
    if outcome.get('blocked'):
        return 'blocked'
    return 'fail' if outcome['problems'] else 'pass'


def _expect(outcome: dict, status: str, needle: str = '') -> dict:
    assert _status(outcome) == status, outcome
    if needle:
        texts = list(outcome['problems']) + ([outcome['blocked']] if outcome.get('blocked') else [])
        assert any(needle in p for p in texts), outcome
    return outcome


def _matrix(root: Path, rel: str, values, index=DATES, columns=CODES) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(values, index=index, columns=columns)
    with open(path, 'wb') as f:
        pickle.dump(df, f)
    return path


def _csv(root: Path, rel: str, header, rows) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [','.join(header)] + [','.join(str(c) for c in r) for r in rows]
    path.write_text('\n'.join(body) + '\n', encoding='utf-8')
    return path


def _spec(contract: dict) -> dict:
    return {'data_contracts_version': 'test', 'contracts': [contract]}


def _run(fn, root: Path, contract: dict) -> dict:
    return fn(root, _spec(contract))


def test_a1_schema_pass_and_fail(tmp_path: Path) -> None:
    rel = 'data/raw/drive/px.pkl'
    _matrix(tmp_path, rel, np.ones((3, 2)))
    contract = {'id': 'a1_schema', 'files': [{'path': rel, 'rows': 3, 'cols': 2,
                                              'start': '2020-01-02', 'end': '2020-01-06'}],
                'csvs': [], 'tolerance': {'rows': 0, 'cols': 0},
                'rules': {'dtype_kinds': ['f']}}
    _expect(_run(dc.check_a1_schema, tmp_path, contract), 'pass', '')['checked'] == 1
    contract['files'][0]['rows'] = 4
    _expect(_run(dc.check_a1_schema, tmp_path, contract), 'fail', 'rows=3 expected=4')


def test_a1_missing_file_is_blocked(tmp_path: Path) -> None:
    contract = {'id': 'a1_schema', 'files': [{'path': 'data/x.pkl', 'rows': 1, 'cols': 1}],
                'csvs': [], 'tolerance': {}, 'rules': {}}
    _expect(_run(dc.check_a1_schema, tmp_path, contract), 'blocked', '缺原始文件')


def test_a2_universe_pass_and_fail(tmp_path: Path) -> None:
    _matrix(tmp_path, 'data/raw/drive/px.pkl', np.ones((3, 2)))
    _csv(tmp_path, 'src_ok.csv', ['code'], [['000001'], ['600000']])
    contract = {'id': 'a2_universe', 'base': {'path': 'data/raw/drive/px.pkl'},
                'sources': [{'path': 'src_ok.csv', 'code_col': 'code', 'min_ratio': 0.95}]}
    _expect(_run(dc.check_a2_universe, tmp_path, contract), 'pass')
    _csv(tmp_path, 'src_bad.csv', ['code'], [['000001'], ['999999']])
    contract['sources'][0]['path'] = 'src_bad.csv'
    _expect(_run(dc.check_a2_universe, tmp_path, contract), 'fail', '覆盖率=0.5000')


def test_a3_pit_pass_and_fail(tmp_path: Path) -> None:
    _matrix(tmp_path, 'data/raw/drive/px.pkl', np.ones((3, 2)))
    contract = {'id': 'a3_pit', 'max_index_date': {'data/raw/drive/px.pkl': '2020-01-06'}}
    _expect(_run(dc.check_a3_pit, tmp_path, contract), 'pass')
    contract['max_index_date']['data/raw/drive/px.pkl'] = '2020-01-03'
    _expect(_run(dc.check_a3_pit, tmp_path, contract), 'fail', '超出快照')


def test_a4_units_pass_and_fail(tmp_path: Path) -> None:
    _matrix(tmp_path, 'data/raw/drive/turn.pkl', np.full((3, 2), 1.5))
    contract = {'id': 'a4_units', 'series': [{'path': 'data/raw/drive/turn.pkl',
                                              'median_min': 0.1, 'median_max': 5.0,
                                              'q999_max': 100.0}]}
    _expect(_run(dc.check_a4_units, tmp_path, contract), 'pass')
    _matrix(tmp_path, 'data/raw/drive/turn.pkl', np.full((3, 2), 250.0))
    _expect(_run(dc.check_a4_units, tmp_path, contract), 'fail', 'median=250')


def _write_monthly(root: Path, rel: str, rows) -> None:
    _csv(root, rel, ['firm_id', 'month', 'R'], rows)


def test_a5_cliffs_pass_and_fail(tmp_path: Path) -> None:
    idx = pd.to_datetime(['2020-01-31', '2020-02-28'])
    post = np.array([[10.0, 10.0], [11.0, 11.0]])
    _matrix(tmp_path, 'data/raw/drive/post.pkl', post, index=idx)
    _write_monthly(tmp_path, 'data/processed/monthly_returns_clean.csv',
                   [['sz000001', '2020-02', '0.1']])
    contract = {'id': 'a5_cliffs', 'cliff_threshold': -0.215,
                'post_adjusted': {'path': 'data/raw/drive/post.pkl', 'max_adjusted_only': 0},
                'monthly': {'path': 'data/processed/monthly_returns_clean.csv',
                            'tolerance': 1e-9}}
    _expect(_run(dc.check_a5_cliffs, tmp_path, contract), 'pass')
    _write_monthly(tmp_path, 'data/processed/monthly_returns_clean.csv',
                   [['sz000001', '2020-02', '0.5']])
    _expect(_run(dc.check_a5_cliffs, tmp_path, contract), 'fail', 'max|diff|')


def test_a5_cliffs_separate_adjustment_defect_from_genuine_crash(tmp_path: Path, monkeypatch) -> None:
    """真实行情不算缺陷；只有复权**引入**的断崖才计入（旧 max_cells 容差会把真实跌幅放行）。"""
    idx = pd.to_datetime(['2020-01-31', '2020-02-03', '2020-02-04'])
    # 第 1 列（600000.SH）：post 10 → 5（-50%），raw 同为 10 → 5 —— 真实行情
    # 第 2 列（000001.SZ）：post 10 → 5（-50%），raw 保持 10 → 10 —— 复权引入的缺陷
    post = np.array([[10.0, 10.0], [5.0, 5.0], [5.0, 5.0]])
    raw = np.array([[10.0, 10.0], [5.0, 10.0], [5.0, 10.0]])
    _matrix(tmp_path, 'data/raw/drive/post.pkl', post, index=idx)
    _matrix(tmp_path, 'data/raw/drive/raw.pkl', raw, index=idx)
    contract = {'id': 'a5_cliffs', 'cliff_threshold': -0.215,
                'post_adjusted': {'path': 'data/raw/drive/post.pkl', 'max_adjusted_only': 0},
                'raw': {'path': 'data/raw/drive/raw.pkl'}}
    # 复权引入 1 条 → 失败（旧口径下 2 条 <= 250 会 pass）
    _expect(_run(dc.check_a5_cliffs, tmp_path, contract), 'fail', '复权引入的断崖')
    # 容差放到 1 则放行复权缺陷，但真实行情仍不计入
    contract['post_adjusted']['max_adjusted_only'] = 1
    _expect(_run(dc.check_a5_cliffs, tmp_path, contract), 'pass')


def test_a5_cliffs_requires_raw_to_classify(tmp_path: Path) -> None:
    """缺未复权序列时无法分类，必须按缺陷处理而不是静默放行。"""
    idx = pd.to_datetime(['2020-01-31', '2020-02-03'])
    post = np.array([[10.0, 10.0], [5.0, 5.0]])
    _matrix(tmp_path, 'data/raw/drive/post.pkl', post, index=idx)
    contract = {'id': 'a5_cliffs', 'cliff_threshold': -0.215,
                'post_adjusted': {'path': 'data/raw/drive/post.pkl', 'max_adjusted_only': 0}}
    _expect(_run(dc.check_a5_cliffs, tmp_path, contract), 'fail', '无法区分')


def test_a6_limit_flags_pass_and_fail(tmp_path: Path) -> None:
    px = np.full((3, 2), 10.0)
    _matrix(tmp_path, 'data/raw/drive/px.pkl', px)
    _matrix(tmp_path, 'data/raw/drive/up.pkl', np.zeros((3, 2), dtype=bool))
    _matrix(tmp_path, 'data/raw/drive/st.pkl', np.zeros((3, 2)))
    contract = {'id': 'a6_limit_flags', 'price': 'data/raw/drive/px.pkl',
                'flag': 'data/raw/drive/up.pkl', 'st': 'data/raw/drive/st.pkl',
                'thresholds': {'main': 0.095, 'main_st': 0.045},
                'start': '2020-01-01', 'end': '2020-12-31', 'min_agreement': 0.999}
    _expect(_run(dc.check_a6_limit_flags, tmp_path, contract), 'pass')
    _matrix(tmp_path, 'data/raw/drive/up.pkl', np.ones((3, 2), dtype=bool))
    _expect(_run(dc.check_a6_limit_flags, tmp_path, contract), 'fail', '一致率=')


def test_a7_return_series_pass_and_fail(tmp_path: Path) -> None:
    _write_monthly(tmp_path, 'data/processed/monthly_returns_clean.csv',
                   [['sz000001', '2020-01', '0.10']])
    _csv(tmp_path, 'data/raw/drive/teacher.csv', ['month', 'firm_id', 'R_adj'],
         [['2020-01', '000001', '0.10']])
    contract = {'id': 'a7_return_series',
                'left': {'path': 'data/processed/monthly_returns_clean.csv',
                         'code_col': 'firm_id', 'value_col': 'R'},
                'right': {'path': 'data/raw/drive/teacher.csv',
                          'code_col': 'firm_id', 'value_col': 'R_adj'},
                'min_corr': 0.999, 'max_abs_diff': 1e-9}
    _expect(_run(dc.check_a7_return_series, tmp_path, contract), 'pass')
    _csv(tmp_path, 'data/raw/drive/teacher.csv', ['month', 'firm_id', 'R_adj'],
         [['2020-01', '000001', '0.20']])
    _expect(_run(dc.check_a7_return_series, tmp_path, contract), 'fail', 'max|diff|')


def test_a8_flag_coverage_pass_and_fail(tmp_path: Path) -> None:
    _matrix(tmp_path, 'data/raw/drive/flag.pkl', np.ones((3, 2)))
    contract = {'id': 'a8_flag_coverage',
                'flags': [{'path': 'data/raw/drive/flag.pkl', 'min_nonnull': 0.99}]}
    _expect(_run(dc.check_a8_flag_coverage, tmp_path, contract), 'pass')
    _matrix(tmp_path, 'data/raw/drive/flag.pkl', np.array([[1.0, np.nan]] * 3))
    _expect(_run(dc.check_a8_flag_coverage, tmp_path, contract), 'fail', '非空率=0.5000')


def test_run_contracts_writes_evidence(tmp_path: Path) -> None:
    _matrix(tmp_path, 'data/raw/drive/flag.pkl', np.ones((3, 2)))
    contract = {'id': 'a8_flag_coverage',
                'flags': [{'path': 'data/raw/drive/flag.pkl', 'min_nonnull': 0.99}]}
    results = dc.run_contracts(tmp_path, only=['a8_flag_coverage'], spec=_spec(contract))
    assert [r['contract'] for r in results] == ['a8_flag_coverage']
    assert results[0]['status'] == 'pass'
    assert (tmp_path / results[0]['evidence']).is_file()
