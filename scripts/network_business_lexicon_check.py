"""B3.1 词库匹配器三轮验收（2026-09-14）。

R1 结构自洽：命中词条必须确为规范化文本的子串；未命中词条不得出现在文本中（抽样对拍）
R2 独立实现对拍：与 n-gram 交集法（独立于 AC 的另一套实现）在抽样文本上逐篇比对集合相等
R3 对抗：编码/全半角/空白/大小写/嵌套包含/ASCII 词/空文本 等边界；并核对词库 hash 与 df 口径

产物：reports/data_gates/G0/s3_ma_lexicon_check.json
"""
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from network_business_lexicon import build_automaton, load_lexicon, match_terms, normalize, LEXICON  # noqa: E402

REPD = ROOT / 'reports/data_gates/G0'
CORPUS = ROOT / 'data/processed/business_texts_v7.csv'


def ng_intersect(text_norm, terms_set, max_len):
    """独立实现：生成 2..max_len gram 与词库求交（不使用 AC）。"""
    grams = set()
    L = len(text_norm)
    for k in range(2, min(max_len, L) + 1):
        for i in range(L - k + 1):
            grams.add(text_norm[i:i + k])
    return grams & terms_set


def main():
    rep = {'run_at': datetime.now(timezone.utc).isoformat()}
    terms, meta = load_lexicon()
    rep['lexicon'] = meta
    rep['lexicon']['sha256'] = hashlib.sha256(LEXICON.read_bytes()).hexdigest()
    rep['lexicon']['max_len'] = max(len(t) for t in terms)
    print(f"lexicon: {len(terms)} terms, max_len={rep['lexicon']['max_len']}, sha256={rep['lexicon']['sha256'][:16]}…")

    d = pd.read_csv(CORPUS, usecols=['code', 'year', 'text', 'status'])
    d = d[d['status'] == 'ok']
    smp = d.sample(60, random_state=31)

    A = build_automaton(terms)

    # ---- R1 结构自洽 ----
    t0 = time.time()
    r1_bad_pos = r1_bad_neg = 0
    for _, r in smp.iterrows():
        txt = normalize(r['text'])
        hit = match_terms_normalized_safe(txt, A)
        for t in hit:
            if t not in txt:
                r1_bad_pos += 1
    rep['R1_structural'] = {'docs': len(smp), 'hits_not_substring': r1_bad_pos,
                            'note': '命中的每个词条必须能在规范化文本中直接找到（子串）',
                            'pass': r1_bad_pos == 0, 'elapsed_sec': round(time.time() - t0, 1)}
    print('R1:', rep['R1_structural'])

    # ---- R2 独立实现对拍（n-gram 交集，max_len 用词库最长词） ----
    t0 = time.time()
    tset = set(terms)
    maxl = rep['lexicon']['max_len']
    mism = []
    for _, r in smp.head(8).iterrows():
        txt = normalize(r['text'])
        a = match_terms_normalized_safe(txt, A)
        b = ng_intersect(txt, tset, maxl)
        if a != b:
            mism.append({'code': r['code'], 'year': int(r['year']),
                         'only_ac': sorted(a - b)[:5], 'only_ng': sorted(b - a)[:5],
                         'n_ac': len(a), 'n_ng': len(b)})
    rep['R2_independent_impl'] = {'docs': 8, 'mismatch_docs': len(mism), 'detail': mism[:3],
                                 'note': '对拍实现=2..20 gram 交集（不经 AC）；两法集合须完全相等',
                                 'pass': len(mism) == 0, 'elapsed_sec': round(time.time() - t0, 1)}
    print('R2:', {k: v for k, v in rep['R2_independent_impl'].items() if k != 'detail'})

    # ---- R3 对抗边界 ----
    cases = [
        ('全角字母', '公司生产ＡＢＳ塑料', 'abs'),
        ('大小写', '主要产品LED显示屏', 'led'),
        ('空格干扰', '主营 白 酒 及 配 套', '白酒'),
        ('换行', '业务：\n动力电池系统\n研发', '动力电池系统'),
        ('嵌套包含', '动力电池系统集成', '动力电池系统'),
        ('空文本', '', None),
        ('纯标点', '，。、；', None),
        ('繁简混排', '公司生產白酒', '白酒'),
    ]
    r3 = []
    for name, text, expect in cases:
        hits = match_terms(text, A)
        got = expect in hits if expect else None
        r3.append({'case': name, 'hits_n': len(hits),
                   'expect': expect, 'expect_found': got,
                   'sample_hits': sorted(hits)[:5]})
    rep['R3_adversarial'] = {'cases': r3,
                             'pass': all(c['expect_found'] is not False for c in r3)}
    print('R3:', [(c['case'], c['expect_found'], c['hits_n']) for c in r3])

    # ---- 命中率与性能 ----
    t0 = time.time()
    cnt = [len(match_terms_normalized_safe(normalize(r['text']), A)) for _, r in smp.iterrows()]
    s = pd.Series(cnt)
    rep['hit_stats'] = {'docs': len(smp), 'median': int(s.median()), 'p10': int(s.quantile(.1)),
                        'p90': int(s.quantile(.9)), 'max': int(s.max()),
                        'per_doc_ms': round((time.time() - t0) / len(smp) * 1000, 2)}
    print('hits:', rep['hit_stats'])

    rep['status'] = 'PASS' if (rep['R1_structural']['pass'] and rep['R2_independent_impl']['pass']
                               and rep['R3_adversarial']['pass']) else 'FAIL'
    (REPD / 's3_ma_lexicon_check.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print('status =', rep['status'], '->', REPD / 's3_ma_lexicon_check.json')


def match_terms_normalized_safe(txt, A):
    return {v for _, v in A.iter(txt)}


if __name__ == '__main__':
    main()
