"""产品词库合并与规范化（M-A 输入之一，2026-09-14）。

来源（均为免费源，2026-09-14 抓取）：
  ① 搜狗细胞词库 data/processed/lexicon_sogou_words.csv（行业/科学类目 2092 库）
  ② 东财 F10 主营构成 data/processed/em_business_terms.csv（MAINOP_TYPE=2 按产品）

规范化规则（冻结，写入报告）：
  1. NFKC 归一（全角→半角、Ⅱ→II 等兼容形式）
  2. 去全部空白；ASCII 一律小写（匹配时同样小写化，大小写不敏感）
  3. 长度：2 ≤ len ≤ 20（字符）
  4. 必须含中日韩字符或 ASCII 字母数字；纯数字/纯标点丢弃
  5. 占位/会计口径词黑名单（"其他"类、"分部间抵销"类、"合计"类）——
     这些词在年报文本中高频出现但不含产品信息，进词库会造成全市场假匹配
  6. 去重（规范化后键唯一），保留来源标记与计数

产物：
  data/processed/product_lexicon_v1.csv      term + 来源/长度/计数
  reports/data_gates/G0/product_lexicon_report.json  统计/黑名单/样例

用法：
  python scripts/build_product_lexicon.py
"""
import csv
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOGOU = ROOT / 'data/processed/lexicon_sogou_words.csv'
EM = ROOT / 'data/processed/em_business_terms.csv'
OUT = ROOT / 'data/processed/product_lexicon_v1.csv'
REPD = ROOT / 'reports/data_gates/G0'

MIN_LEN, MAX_LEN = 2, 20
CJK = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff]')
ALNUM = re.compile(r'[0-9A-Za-z]')
ONLY_PUNCT = re.compile(r'^[\W_]+$', re.UNICODE)

# 占位/会计口径词（F10 主营构成常见非产品条目）；Sogou 侧偶有同名通用词，一并剔除
PLACEHOLDER_EXACT = {
    '其他', '其它', '其他业务', '其他主营业务', '其他收入', '其他产品', '其他(补充)', '其他（补充）',
    '其他业务收入', '其它业务', '其它业务收入', '其他产品收入', '其它产品',
    '分部间抵销', '内部抵销', '内部抵消', '抵销', '抵消', '分部间抵消', '内部交易抵销',
    '合计', '小计', '总计', '合计数', '主营', '主营业务', '主营业务收入', '营业收入', '营业总收入',
    '收入', '成本', '毛利', '毛利率', '减', '减:', '其中', '其中:', '合计：', '小计：',
    '补充', '调整', '抵消数', '抵销数', '分部抵消', '关联交易', '内部销售',
}
PLACEHOLDER_RE = re.compile(r'^(其他|其它|其中|合计|小计|总计|减[:：]?|合并|抵消|抵销)[(（]?.*$|^.*[(（](补充|抵消|抵销|其他|其它)[)）]$')


def norm(term: str) -> str:
    t = unicodedata.normalize('NFKC', str(term))
    t = re.sub(r'\s+', '', t)
    return t.lower()


def is_junk(t: str) -> bool:
    if not (MIN_LEN <= len(t) <= MAX_LEN):
        return True
    if ONLY_PUNCT.match(t):
        return True
    if not (CJK.search(t) or ALNUM.search(t)):
        return True
    if not re.search(r'[0-9A-Za-z\u3400-\u4dbf\u4e00-\u9fff]', t):
        return True
    if t.isdigit():
        return True
    if t in PLACEHOLDER_EXACT or PLACEHOLDER_RE.match(t):
        return True
    return False


def main():
    sogou = pd.read_csv(SOGOU, dtype=str).fillna('')
    em = pd.read_csv(EM, dtype=str)
    em = em[em['mainop_type'].astype(str) == '2'].copy()

    terms = {}
    n_sogou_raw, n_em_raw = len(sogou), len(em)
    drop = Counter()

    for _, r in sogou.iterrows():
        t = norm(r['word'])
        if is_junk(t):
            drop['sogou_junk'] += 1
            continue
        s = terms.setdefault(t, {'in_sogou': 0, 'sogou_dicts': set(), 'sogou_cates': set(),
                                 'in_em': 0, 'em_firms': set(), 'em_periods': 0})
        s['in_sogou'] += 1
        s['sogou_dicts'].add(r['dict_name'])
        s['sogou_cates'].add(r['cate_name'])

    for _, r in em.iterrows():
        t = norm(r['item_name'])
        if is_junk(t):
            drop['em_junk'] += 1
            continue
        s = terms.setdefault(t, {'in_sogou': 0, 'sogou_dicts': set(), 'sogou_cates': set(),
                                 'in_em': 0, 'em_firms': set(), 'em_periods': 0})
        s['in_em'] += 1
        s['em_firms'].add(r['firm_id'])
        s['em_periods'] += 1

    rows = []
    for t, s in sorted(terms.items()):
        rows.append({
            'term': t, 'char_len': len(t),
            'has_cjk': int(bool(CJK.search(t))),
            'in_sogou': int(s['in_sogou'] > 0),
            'sogou_dict_count': s['in_sogou'],
            'sogou_dict_sample': '/'.join(sorted(s['sogou_dicts'])[:3]),
            'sogou_cates': '/'.join(sorted(s['sogou_cates'])[:3]),
            'in_em': int(s['in_em'] > 0),
            'em_firm_count': len(s['em_firms']),
            'em_period_count': s['em_periods'],
        })
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    both = sum(1 for r in rows if r['in_sogou'] and r['in_em'])
    rep = {
        'run_at': datetime.now(timezone.utc).isoformat(),
        'inputs': {'sogou_rows': n_sogou_raw, 'em_product_rows': n_em_raw},
        'normalization': {
            'steps': ['NFKC', 'strip_all_whitespace', 'ascii_lowercase', f'len in [{MIN_LEN},{MAX_LEN}]',
                      'must_contain_cjk_or_alnum', 'placeholder_blacklist'],
            'placeholder_exact_n': len(PLACEHOLDER_EXACT),
            'matching_rule': '下游匹配须对文本与词条同时做 NFKC + 小写化 + 去空白后再做子串匹配',
        },
        'output': {
            'terms_total': len(rows),
            'in_sogou_only': sum(1 for r in rows if r['in_sogou'] and not r['in_em']),
            'in_em_only': sum(1 for r in rows if r['in_em'] and not r['in_sogou']),
            'in_both': both,
            'ascii_terms': sum(1 for r in rows if not r['has_cjk']),
            'len_p50': int(pd.Series([r['char_len'] for r in rows]).median()),
            'len_p90': int(pd.Series([r['char_len'] for r in rows]).quantile(0.9)),
        },
        'dropped': dict(drop),
        'top_em_terms_by_firm_count': sorted(rows, key=lambda r: -r['em_firm_count'])[:20],
        'placeholder_exact_list': sorted(PLACEHOLDER_EXACT),
    }
    REPD.mkdir(parents=True, exist_ok=True)
    (REPD / 'product_lexicon_report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: rep[k] for k in ['inputs', 'output', 'dropped']}, ensure_ascii=False, indent=1))
    print(f'wrote {OUT} and {REPD / "product_lexicon_report.json"}')


if __name__ == '__main__':
    main()
