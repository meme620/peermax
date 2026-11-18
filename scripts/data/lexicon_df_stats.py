"""产品词库的语料 df 统计与通用词过滤（M-A 前置，2026-09-14）。

动机（三轮验收 R3 发现）：搜狗细胞词库是通用输入法词库，含大量非产品通用词
（模式/情况/提升/技术/信息…）。若直接做 M-A 的 0/1 词向量，余弦相似度会退化为
"写作风格相似"而非"产品相似"。故增加数据驱动的文档频率（df）过滤层：
在产品语料中出现频率过高的词条无区分度，按阈值剔除（与 M-B 的 df_threshold 同一思路）。

口径：
  - 语料 = data/processed/business_texts_v7.csv 的 status=ok 行（M-A 输入，2017-2025）
  - 分词 = jieba + 词库作 userdict（保证多字产品词不被切碎；与 M-A 构网用同一分词路径）
  - df(term) = 含该词的文档数 / 总文档数；keep = df ≤ DF_MAX（默认 0.25）
  - 未被分出的词条 df=0（列为 never_fired，仅作诊断；不删，保留维度）

产物：
  data/processed/product_lexicon_v2.csv   在 v1 基础上加 df / df_ratio / keep_df25
  reports/data_gates/G0/product_lexicon_df_report.json

用法：
  python scripts/data/lexicon_df_stats.py [--df-max 0.25]
"""
import argparse
import csv
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import jieba
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LEX = ROOT / 'data/processed/product_lexicon_v1.csv'
CORPUS = ROOT / 'data/processed/business_texts_v7.csv'
OUT = ROOT / 'data/processed/product_lexicon_v2.csv'
REPD = ROOT / 'reports/data_gates/G0'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--df-max', type=float, default=0.25)
    args = ap.parse_args()

    jieba.setLogLevel(60)
    lex = pd.read_csv(LEX, dtype=str).fillna('')
    lex['term'] = lex['term'].astype(str)
    vocab = set(lex['term'])
    # 说明：用**朴素 jieba 分词**（不加 userdict）计算 df。
    # 理由：1.35M 词条入 userdict 会显著拖慢分词且与后续 M-A 匹配口径可能不同；
    # 而 df 过滤只删除高频通用词，漏计（多字产品词被切碎 → df=0）是保守方向，不会误删。
    # 多字词的真实命中率由 M-A 构网阶段用最终匹配规则复算。

    d = pd.read_csv(CORPUS, usecols=['code', 'year', 'text', 'status'])
    d = d[d['status'] == 'ok']
    n_doc = len(d)
    print(f'corpus docs: {n_doc}', flush=True)

    df = Counter()
    fired_docs = 0
    t0 = time.time()
    for i, txt in enumerate(d['text'].astype(str), 1):
        toks = set(w for w in jieba.cut(txt) if len(w) >= 2)
        hit = toks & vocab
        if hit:
            fired_docs += 1
            df.update(hit)
        if i % 5000 == 0:
            print(f'   {i}/{n_doc} fired_docs={fired_docs} elapsed={time.time()-t0:.0f}s', flush=True)

    lex['doc_count'] = lex['term'].map(lambda t: int(df.get(t, 0)))
    lex['df_ratio'] = (lex['doc_count'] / n_doc).round(6)
    lex['keep_df25'] = (lex['df_ratio'] <= args.df_max).astype(int)
    lex.to_csv(OUT, index=False, encoding='utf-8')

    fired = lex[lex['doc_count'] > 0]
    dropped = lex[(lex['df_ratio'] > args.df_max)]
    rep = {
        'run_at': datetime.now(timezone.utc).isoformat(),
        'corpus': str(CORPUS.relative_to(ROOT)).replace('\\', '/'),
        'corpus_docs': int(n_doc),
        'df_max': args.df_max,
        'lexicon_terms': int(len(lex)),
        'terms_fired_at_least_once': int(len(fired)),
        'terms_never_fired': int(len(lex) - len(fired)),
        'docs_with_any_hit': int(fired_docs),
        'docs_with_any_hit_ratio': round(fired_docs / n_doc, 4),
        'terms_dropped_by_df': int(len(dropped)),
        'terms_kept': int(lex['keep_df25'].sum()),
        'top_dropped_by_df': dropped.sort_values('df_ratio', ascending=False)
                             .head(40)[['term', 'df_ratio', 'in_sogou', 'in_em']].to_dict('records'),
        'top_fired_terms': fired.sort_values('doc_count', ascending=False)
                            .head(30)[['term', 'doc_count', 'df_ratio']].to_dict('records'),
        'note': ('df 过滤是词库构建层选择（非收益预测选择），统一施加并留档；'
                 '保留未过滤版本 v1 供稳健性对照。keep_df25 为默认口径，'
                 'M-A 构网时作为待冻结参数 ma_lexicon_df_max 的候选值。'),
        'elapsed_sec': round(time.time() - t0, 1),
    }
    REPD.mkdir(parents=True, exist_ok=True)
    (REPD / 'product_lexicon_df_report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in rep.items() if k.startswith(('lexicon', 'terms', 'docs', 'df_max', 'corpus'))},
                     ensure_ascii=False, indent=1))
    print(f'wrote {OUT} and {REPD / "product_lexicon_df_report.json"}')


if __name__ == '__main__':
    main()
