import hashlib
import json
import math

import numpy as np
import pytest

from lib.dates import next_month
from lib.fmt import fmt_float, fmt_int
from lib.io import (atomic_write_text, json_default, sha256_file,
                    write_csv_rows, write_json_atomic)
from lib.stats import nw_ols, nw_t


def test_next_month_crosses_year():
    assert next_month('2023-12') == '2024-01'
    assert next_month('2024-01') == '2024-02'


def test_write_json_atomic_creates_parents_and_trailing_newline(tmp_path):
    target = tmp_path / 'a/b/c.json'
    write_json_atomic({'k': [1, 2]}, target)
    assert json.loads(target.read_text(encoding='utf-8')) == {'k': [1, 2]}
    assert target.read_text(encoding='utf-8').endswith('\n')
    assert not (tmp_path / 'a/b/c.json.tmp').exists()


def test_atomic_write_text_replaces_without_leftover(tmp_path):
    target = tmp_path / 'x.txt'
    atomic_write_text(target, 'one')
    atomic_write_text(target, 'two')
    assert target.read_text(encoding='utf-8') == 'two'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['x.txt']


def test_sha256_file_matches_hashlib(tmp_path):
    target = tmp_path / 'blob.bin'
    target.write_bytes(b'peer max' * 1000)
    assert sha256_file(target) == hashlib.sha256(b'peer max' * 1000).hexdigest()


def test_write_csv_rows_header_and_rows(tmp_path):
    target = tmp_path / 't.csv'
    write_csv_rows(target, [[1, 'a'], [2, 'b']], ['x', 'y'])
    assert target.read_text(encoding='utf-8').splitlines() == ['x,y', '1,a', '2,b']


def test_json_default_handles_numpy_scalars_and_timestamps():
    import pandas as pd
    assert json_default(np.int64(3)) == 3
    assert json_default(np.float64(0.5)) == 0.5
    assert json_default(np.bool_(True)) is True
    assert json_default(pd.Timestamp('2024-01-02')) == '2024-01-02T00:00:00'
    assert json_default(object()) .startswith('<object')


def test_fmt_float_and_int_edges():
    assert fmt_float(0.5) == '0.5'
    assert fmt_float(None) == ''
    assert fmt_float(float('nan')) == ''
    assert fmt_int(3.6) == '4'
    assert fmt_int(None) == ''


def test_nw_t_matches_manual_computation():
    x = np.array([0.1, 0.2, 0.15, 0.3, 0.25])
    L = 1
    T = len(x)
    mu = x.mean()
    se2 = np.sum((x - mu) ** 2) / T / T
    gamma = np.sum((x[:-1] - mu) * (x[1:] - mu)) / T
    se2 += 2 * (1 - 1 / (L + 1)) * gamma / T
    assert nw_t(x, L) == pytest.approx(mu / math.sqrt(se2))


def test_nw_ols_recovers_linear_coefficients():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(200), rng.normal(size=200)])
    y = 2.0 + 3.0 * X[:, 1]
    beta, se, t = nw_ols(X, y, L=1)
    assert beta[0] == pytest.approx(2.0)
    assert beta[1] == pytest.approx(3.0)
    assert t[1] > 10
