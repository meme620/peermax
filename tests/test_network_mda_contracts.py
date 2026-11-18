"""S3-MB MD&A 数据契约测试（2026-09-11）。

读 data/processed/mda_texts.csv（不重跑）验证硬契约：
  1. 唯一键 (code, year) 且与原始 annual_txt 文件一一对应
  2. 列完整（含审计列 src_identity/content_code/dup_n/extract_format/reason）
  3. status=ok：text 非空（length>0）、src_identity != mismatch、extract_format ∈ {new,old,weak}
  4. status=bad_source：length == 0（文本置空），reason 非空
  5. 审计清单一致性：ok 行不得与 mda_src_identity.csv 中 identity=mismatch 的 (code,year) 重叠；
     ok 行 year_signal 不得为 mismatch*（源文件年份错位，2026-09-12 新增维度）
  6. 去截断回归：不存在 length == 30000 的硬上限堆积（v1 缺陷；阈值检查为 >30000 占比 > 5%）
  7. 安全上限：length <= 3e5

运行：python -m pytest tests/   或   python tests/test_network_mda_contracts.py
"""
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MDA_V2 = ROOT / 'data' / 'processed' / 'mda_texts.csv'
IDENT = ROOT / 'artifacts' / 'review' / 'mda_src_identity.csv'
RAW = ROOT / 'data' / 'raw' / 'cninfo' / 'annual_txt'
MAX_LEN = 300000

COLS = ['code', 'year', 'text', 'anchor', 'length', 'status',
        'extract_format', 'src_identity', 'content_code', 'dup_n',
        'year_signal', 'reason']


def _load_meta():
    """只读元数据列（text 列跳过，避免测试读 3GB 文本）。"""
    csv.field_size_limit(10 ** 9)
    rows = []
    with open(MDA_V2, encoding='utf-8', newline='') as f:
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


def test_mda_v2_exists_and_columns():
    assert MDA_V2.exists(), f'缺少 {MDA_V2}（先运行 scripts/text_mda_extract.py）'
    header, _ = _load_meta()
    assert header == COLS, f'列漂移: {header}'


def test_unique_key_and_coverage():
    _, rows = _load_meta()
    keys = [(r['code'][2:] if r['code'][:2] in ('sz', 'sh') else r['code'], r['year']) for r in rows]
    dup = {k: v for k, v in Counter(keys).items() if v > 1}
    assert not dup, f'唯一键冲突: {list(dup)[:5]}'
    raw = _raw_pairs()
    assert set(keys) == raw, f'与原始语料不一一对应: 差 {len(set(keys) ^ raw)} 条'


def test_ok_rows_contract():
    _, rows = _load_meta()
    ok = [r for r in rows if r['status'] == 'ok']
    assert ok, '无 ok 行'
    bad = [r for r in ok if int(r['length']) <= 0]
    assert not bad, f'ok 行 text 为空: {len(bad)} 条'
    mis = [r for r in ok if r['src_identity'] == 'mismatch']
    assert not mis, f'ok 行含身份错位源: {len(mis)} 条'
    fmt = {r['extract_format'] for r in ok}
    assert fmt <= {'new', 'old', 'weak'}, f'extract_format 越界: {fmt}'
    assert {'new'} & fmt, '无 new 格式提取，疑似回归'


def test_bad_source_rows_blank():
    _, rows = _load_meta()
    bs = [r for r in rows if r['status'] == 'bad_source']
    assert bs, 'bad_source 行数为 0，身份列疑似失效'
    bad = [r for r in bs if int(r['length']) != 0 or not r['reason']]
    assert not bad, f'bad_source 行未置空/无 reason: {len(bad)} 条'


def test_ok_rows_exclude_identity_mismatch_list():
    if not IDENT.exists():
        pytest.skip(f'缺 {IDENT}；数据门 mda-source-identity 覆盖同一不变量')
    mis = set()
    with open(IDENT, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['identity'] == 'mismatch':
                mis.add((r['code6'], r['year']))
    _, rows = _load_meta()
    hit = [r for r in rows if r['status'] == 'ok'
           and (r['code'][2:] if r['code'][:2] in ('sz', 'sh') else r['code'], r['year']) in mis]
    assert not hit, f'审计清单错位行仍为 ok: {len(hit)} 条'


def test_ok_rows_year_signal_contract():
    _, rows = _load_meta()
    bad = [r for r in rows if r['status'] == 'ok' and str(r.get('year_signal', '')).startswith('mismatch')]
    assert not bad, f'ok 行含年份错位源: {len(bad)} 条'


def test_no_truncation_regression():
    _, rows = _load_meta()
    ok = [int(r['length']) for r in rows if r['status'] == 'ok']
    n = len(ok)
    assert max(ok) <= MAX_LEN, f'超安全上限: {max(ok)}'
    long_share = sum(1 for x in ok if x > 30000) / n
    assert long_share > 0.05, f'去截断回归：>30k 占比仅 {long_share:.3f}（v1 为 0）'


CID_GLYPH_RE = __import__('re').compile(
    r'[਀-૿଀-୿ఀ-౿ഀ-ൿ฀-๿'
    r'ᄀ-ᇿក-៿ꥠ-꥿가-힯-'
    r'￰-￿⺀-⻿]')


def test_ok_rows_no_cid_glyph_corruption():
    """CID 字体缺失 ToUnicode 导致的字形乱码不得留在 ok 行（2026-09-13 P0.1）。

    判据排除 U+2460-U+2469（①②③，中文年报正常序号），只统计真正的异域字形区。
    """
    csv.field_size_limit(10 ** 9)
    bad = []
    with open(MDA_V2, encoding='utf-8', newline='') as f:
        r = csv.DictReader(f)
        for d in r:
            if d['status'] != 'ok':
                continue
            t = d['text'] or ''
            if not t:
                continue
            rate = len(CID_GLYPH_RE.findall(t)) / len(t)
            if rate > 0.005:
                bad.append((d['code'], d['year'], round(rate, 4)))
    assert not bad, f'ok 行含 CID 字形乱码: {bad[:5]}'


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
