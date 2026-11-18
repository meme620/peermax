"""M-A 词库匹配模块（B3.1 冻结口径，2026-09-14）。

口径（冻结）：
  - 词库：`data/processed/product_lexicon_v2.csv`，取 `keep_df25 == 1`（df ≤ 0.25 已过滤）
  - 规范化：NFKC → 去全部空白 → ASCII 小写（文本与词条同规则）
  - 匹配：**全子串匹配**（Aho-Corasick，pyahocorasick 2.3.1），返回命中的词条**集合**
    （0/1 向量语义，重复命中与嵌套命中均只计一次）
  - 依据：深度审计（free_source_deep_audit_20260914.md）实测朴素 jieba 分词下 96% 词条不命中，
    2–6 gram 子串口径下每篇命中中位 308 词；本模块为全子串（含长词）版本。

用法：
    from network_business_lexicon import load_lexicon, build_automaton, match_terms, normalize
    terms, meta = load_lexicon()
    A = build_automaton(terms)
    hit = match_terms(text, A)
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEXICON = ROOT / 'data/processed/product_lexicon_v2.csv'
DF_MAX_DEFAULT = 0.25

_WS = re.compile(r'\s+')


def normalize(text: str) -> str:
    """NFKC + 去空白 + ASCII 小写（文本/词条同规则）。"""
    s = unicodedata.normalize('NFKC', str(text))
    s = _WS.sub('', s)
    return s.lower()


def load_lexicon(path: Path | str = LEXICON, df_max: float = DF_MAX_DEFAULT):
    """返回 (terms: list[str], meta: dict)。df_max=None 时用 keep_df25 列原值。"""
    d = pd.read_csv(path, dtype=str)
    d['term'] = d['term'].astype(str)
    if 'df_ratio' in d.columns:
        d['df_ratio'] = pd.to_numeric(d['df_ratio'], errors='coerce')
        keep = d['df_ratio'].fillna(0.0) <= df_max
    else:
        keep = d.get('keep_df25', pd.Series(1, index=d.index)).astype(str).eq('1')
    terms = sorted(set(d.loc[keep, 'term']))
    meta = {
        'path': str(Path(path).relative_to(ROOT)).replace('\\', '/'),
        'df_max': df_max,
        'terms_total_in_file': int(len(d)),
        'terms_kept': len(terms),
        'normalize_rule': 'NFKC + strip_all_whitespace + ascii_lowercase',
        'match_rule': 'full_substring (Aho-Corasick, set semantics)',
    }
    return terms, meta


def build_automaton(terms):
    import ahocorasick
    A = ahocorasick.Automaton()
    for t in terms:
        A.add_word(t, t)
    A.make_automaton()
    return A


def match_terms(text: str, automaton) -> set:
    """返回命中词条集合（已规范化文本的匹配结果）。"""
    return {v for _, v in automaton.iter(normalize(text))}


def match_terms_normalized(norm_text: str, automaton) -> set:
    """输入须已是 normalize() 结果（批量场景避免重复规范化）。"""
    return {v for _, v in automaton.iter(norm_text)}
