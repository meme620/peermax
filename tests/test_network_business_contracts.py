"""S3-MA 业务文本数据契约测试（2026-09-13，P0.3）。

读 data/processed/business_texts.csv（不重跑提取）验证硬契约：
  1. 唯一键 (code, year)
  2. 列完整（含审计列 src_identity/content_code/dup_n/year_signal/reason）
  3. status=ok：text 非空、length>0 且与 text 长度一致、无 CID 字形乱码
  4. status=bad_source：length == 0（文本置空）、reason 非空
  5. **去截断回归**：不得出现 900 字硬截断堆积（v6 缺陷：88.4% 文本 ==900）
  6. 安全上限：length <= 3e4
  7. 覆盖：与 annual_txt 语料一一对应（每个源文件恰有一行）
  8. 2017 覆盖不低于语料该年文件数（v6 缺 186 份）

运行：python tests/test_network_business_contracts.py
"""
import csv
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIZ = ROOT / 'data/processed/business_texts.csv'
RAW = ROOT / 'data/raw/cninfo/annual_txt'

COLS = ['code', 'year', 'text', 'anchor', 'length', 'status',
        'extract_format', 'src_identity', 'content_code', 'dup_n',
        'year_signal', 'reason']
SAFE_MAX = int(3e4)
CID_RE = re.compile(
    r'[\u0a00-\u0aff\u0b00-\u0b7f\u0c00-\u0c7f\u0d00-\u0d7f\u0e00-\u0e7f'
    r'\u1100-\u11ff\u1780-\u17ff\ua960-\ua97f\uac00-\ud7af\ue000-\uf8ff'
    r'\ufff0-\uffff\u2e80-\u2eff]')


def _load_meta():
    """只读元数据列（text 跳过）。"""
    csv.field_size_limit(10 ** 9)
    rows = []
    with open(BIZ, encoding='utf-8', newline='') as f:
        r = csv.reader(f)
        header = next(r)
        idx = {c: i for i, c in enumerate(header)}
        for row in r:
            rows.append({c: row[idx[c]] for c in COLS if c != 'text'})
    return header, rows


def _raw_pairs():
    pairs = set()
    for d in RAW.iterdir():
        if d.is_dir():
            for f in d.glob('*.txt'):
                pairs.add((d.name, f.stem))
    return pairs


def _to_panel(code6):
    if code6.startswith(('600', '601', '603', '605', '688', '689')):
        return 'sh' + code6
    if code6.startswith(('000', '001', '002', '003', '300', '301', '302')):
        return 'sz' + code6
    return code6


def test_biz_v7_exists_and_columns():
    assert BIZ.exists(), f'缺少 {BIZ}'
    header, rows = _load_meta()
    assert header == COLS, f'列不符: {header}'
    assert len(rows) == 39778, f'行数应为 39,778（=语料文件数），实际 {len(rows)}'


def test_unique_key_and_coverage():
    _, rows = _load_meta()
    keys = [(r['code'], r['year']) for r in rows]
    dup = [k for k, v in Counter(keys).items() if v > 1]
    assert not dup, f'唯一键重复: {dup[:5]}'
    raw = {( _to_panel(c), y) for c, y in _raw_pairs()}
    got = set(keys)
    missing = raw - got
    assert not missing, f'语料有而产物无: {len(missing)} 例 {list(missing)[:5]}'


def test_ok_rows_contract():
    csv.field_size_limit(10 ** 9)
    with open(BIZ, encoding='utf-8', newline='') as f:
        for d in csv.DictReader(f):
            if d['status'] != 'ok':
                continue
            assert d['text'], f"ok 行文本为空: {d['code']}/{d['year']}"
            assert int(d['length']) == len(d['text']), f"length 与文本不符: {d['code']}/{d['year']}"


def test_bad_source_rows_blank():
    _, rows = _load_meta()
    bad = [r for r in rows if r['status'] in ('bad_source', 'duplicate_text')]
    for r in bad:
        assert int(r['length']) == 0, f"{r['status']} 未置空: {r['code']}/{r['year']}"
        assert r['reason'], f"{r['status']} 缺 reason: {r['code']}/{r['year']}"


def test_no_900_truncation_regression():
    """v6 缺陷回归：88.4% 文本恰好 900 字。"""
    _, rows = _load_meta()
    ok = [int(r['length']) for r in rows if r['status'] == 'ok' and int(r['length']) > 0]
    n = len(ok)
    at_900 = sum(1 for x in ok if x == 900)
    share = at_900 / n
    assert share < 0.05, f'900 字堆积回归: {at_900}/{n} = {share:.1%}'
    assert max(ok) > 2000, f'最长仅 {max(ok)}，疑似仍被截断'


def test_safe_upper_bound():
    _, rows = _load_meta()
    over = [r for r in rows if int(r['length']) > SAFE_MAX]
    assert not over, f'超安全上限: {[(r["code"], r["year"]) for r in over[:5]]}'


def test_ok_rows_no_cid_glyph_corruption():
    csv.field_size_limit(10 ** 9)
    bad = []
    with open(BIZ, encoding='utf-8', newline='') as f:
        for d in csv.DictReader(f):
            if d['status'] != 'ok':
                continue
            t = d['text'] or ''
            if not t:
                continue
            rate = len(CID_RE.findall(t)) / len(t)
            if rate > 0.005:
                bad.append((d['code'], d['year'], round(rate, 4)))
    assert not bad, f'ok 行含 CID 字形乱码: {bad[:5]}'


def test_year_2017_coverage():
    """v6 该年仅 3,282 行（语料 3,468），缺口 186。"""
    _, rows = _load_meta()
    y17 = sum(1 for r in rows if r['year'] == '2017')
    raw17 = sum(1 for c, y in _raw_pairs() if y == '2017')
    assert y17 == raw17, f'2017 覆盖 {y17} != 语料 {raw17}'


if __name__ == '__main__':
    import sys
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'PASS {name}')
            except AssertionError as e:
                fails += 1
                print(f'FAIL {name}: {e}')
    sys.exit(1 if fails else 0)
