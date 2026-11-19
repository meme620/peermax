"""Phase 1.11 / I-17：sample_version 从数据首末月计算（只读真实 CSV）。

覆盖：
- `compute_sample_version` / `annotate_sample_version`：合成 out 数据
  2000-02..2026-06、2018-01..2023-12 → 正确字符串；空输入 → SystemExit。
- `build_status`：合成价矩阵 + 空 panel 目录 → 写出 computed 标签与追溯列，
  列顺序为旧列 + I-1 新列 + I-17 新列。
- 当前 `monthly_returns_clean_v2.csv`：只读，856,771 行、已按 I-17 重写为 computed 标签
  （BL-01，2026-10-30；旧标签 mainline_2018_2025_fullperiod 退役）。
- 孪生文件：`monthly_returns_clean.csv` 与 `monthly_returns_full_2006_2025.csv`
  字节相同，后者 alias_of 前者。
- 不写入 data/、不跑 build_monthly_returns_status 主流程、不跑 pricing_gate/PC。
"""
import inspect
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import build_monthly_returns_status as build  # noqa: E402

from conftest import require_artifacts

from conftest import registered_rows

PROCESSED = ROOT / 'data' / 'processed'
V2_CSV = PROCESSED / 'monthly_returns_clean_v2.csv'
MAIN_CSV = PROCESSED / 'monthly_returns_clean.csv'
TWIN_CSV = PROCESSED / 'monthly_returns_full_2006_2025.csv'
LEGACY = 'mainline_2018_2025_fullperiod'
WINDOWS = [('2000-02', '2026-06'), ('2018-01', '2023-12')]


def _months(start: str, end: str) -> list:
    return [str(p) for p in pd.period_range(start, end, freq='M')]


# ---------------------------------------------------------------------------
# 合成 out 数据：标签由 min/max 计算
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('start,end', WINDOWS)
def test_compute_sample_version_from_minmax(start, end):
    months = _months(start, end)
    assert build.compute_sample_version(months) == f'{start}..{end}'

    # 顺序无关；缺失值被忽略
    shuffled = [None, months[-1]] + months[:-1][::-1]
    assert build.compute_sample_version(shuffled) == f'{start}..{end}'


@pytest.mark.parametrize('start,end', WINDOWS)
def test_annotate_sample_version_synthetic_out(start, end):
    out = pd.DataFrame({
        'firm_id': ['sh600000', 'sz000001'],
        'month': [end, start],          # 故意乱序
        'R': [0.1, -0.2],
    })
    got = build.annotate_sample_version(out)

    assert got['sample_version'].tolist() == [f'{start}..{end}'] * len(got)
    assert got['sample_version_source'].tolist() == ['computed_from_data'] * len(got)
    assert got['sample_version_legacy'].tolist() == [LEGACY] * len(got)

    # 不改入参，也不改其他列
    assert 'sample_version' not in out.columns
    assert got[['firm_id', 'month', 'R']].equals(out[['firm_id', 'month', 'R']])


def test_empty_or_all_missing_months_raise():
    with pytest.raises(SystemExit):
        build.compute_sample_version([])
    with pytest.raises(SystemExit):
        build.compute_sample_version([None, None])


def test_build_status_writes_computed_columns(tmp_path):
    dates = pd.to_datetime(['2000-02-01', '2000-02-02', '2026-06-01', '2026-06-02'])
    px = pd.DataFrame({'000001.SZ': [10.0, 10.1, 20.0, 20.2]}, index=dates)
    ship = pd.DataFrame({
        'firm_id': ['sz000001', 'sz000001'],
        'month': ['2000-02', '2026-06'],
        'R': [0.1, -0.2],
    })

    out = build.build_status(ship, px, panel_dir=tmp_path)

    # build_status 保留 I-1 契约列，并写入 computed sample_version
    assert list(out.columns) == build.OLD_COLS + build.NEW_COLS
    assert out['sample_version'].unique().tolist() == ['2000-02..2026-06']

    # 写出路径（main 在 to_csv 前调用）追加 I-17 追溯列
    wrote = build.annotate_sample_version(out)
    assert list(wrote.columns) == (build.OLD_COLS + build.NEW_COLS
                                   + build.SAMPLE_VERSION_COLS)
    assert wrote['sample_version'].unique().tolist() == ['2000-02..2026-06']
    assert wrote['sample_version_source'].unique().tolist() == ['computed_from_data']
    assert wrote['sample_version_legacy'].unique().tolist() == [LEGACY]


def test_no_hardcoded_sample_version_and_no_manual_override():
    src = (ROOT / 'scripts' / 'build_monthly_returns_status.py').read_text(
        encoding='utf-8')
    assert re.search(r'^\s*SAMPLE_VERSION\s*=', src, re.M) is None
    assert 'sample_version' not in inspect.signature(build.build_status).parameters
    assert build.SAMPLE_VERSION_SOURCE == 'computed_from_data'
    assert build.LEGACY_SAMPLE_VERSION == LEGACY


# ---------------------------------------------------------------------------
# 当前真实 CSV：只读，已按 I-17 重写为 computed 标签（BL-01）
# ---------------------------------------------------------------------------

def test_current_v2_csv_computed_label_read_only():
    assert V2_CSV.exists(), f'缺少 {V2_CSV}'
    df = pd.read_csv(V2_CSV, usecols=['month', 'sample_version'], dtype=str)

    assert len(df) == registered_rows('monthly_returns_clean_v2')
    assert df['sample_version'].unique().tolist() == [f"{df['month'].min()}..{df['month'].max()}"]
    # 覆盖面守卫：不写死具体月份（数据源变更会改起点），但必须覆盖因子与样本所需区间
    assert df['month'].min() <= '2006-01', f'月收益覆盖起点过晚: {df["month"].min()}'
    assert df['month'].max() >= '2023-12'

    computed = build.compute_sample_version(df['month'])
    assert computed == f"{df['month'].min()}..{df['month'].max()}"
    # Phase 4 BL-01（I-17）：已按 min(month)..max(month) 重写为 computed 标签（旧标签 LEGACY 已退役，
    # 仅在生成器追溯列 sample_version_legacy 中保留）
    assert df['sample_version'].iat[0] == computed
    assert LEGACY != computed


# ---------------------------------------------------------------------------
# 孪生文件：字节相同，alias_of 关系登记
# ---------------------------------------------------------------------------

def test_twin_files_byte_identical_alias_of():
    require_artifacts(MAIN_CSV, TWIN_CSV, hint='缺月收益产物')
    assert MAIN_CSV.exists() and TWIN_CSV.exists()
    assert MAIN_CSV.read_bytes() == TWIN_CSV.read_bytes()
    # 主文件被 analyst-network spec 锚定；full_2006_2025 是其别名，不改名/不删除
    assert MAIN_CSV.name == 'monthly_returns_clean.csv'
    assert TWIN_CSV.name == 'monthly_returns_full_2006_2025.csv'
