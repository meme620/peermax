"""S3 网络边级语义审计（P5.1/P5.2 原型实现）——v2（2026-10-13）。

对 M-A / M-B 正式边表做分层抽样，逐边收集"是否真竞争者"的可核验证据：
- 两家公司申万二级行业是否一致（同行业 = 弱阳性信号，非金标准）
- 共同命中产品词（M-A，按词库 df 升序）/ 共享高 idf token（M-B）
- 语料质量标记（**结构判据**越界 flag + **当前提取器 MAX_LEN** 截断 flag）
- 所在聚类/社团规模

**v2 变更（plan §8.10.4 修复要求，第二梯队 #10 路线 A 前置；2026-10-13）**：
  ① 语料越界 flag：旧 `notes_marker>=2` **关键词判据**（`NOTES_MARKERS` 含"外币/金融工具/
     递延所得税"等合法 MD&A 词，实测假阳性 38.2% vs 结构判据 0.33%）→ 改为 **§8.10.5 结构判据**：
     独占一行、去空白后为章节标记（第X节/第X章）或 `STOP_NAMES` 具名标题，且同一标题全文出现
     < 3 次（>= 3 次判为页眉，不计）。字段：`stop_heading_hits` / `out_of_scope`。
  ② 截断 flag：旧 `len >= 29900` 阈值来自**v1 提取器的 MAX_LEN=3e4**，
     v2 提取器已提到 3e5 → flag 失效。改为**读当前提取器的 MAX_LEN**：
       self 语料：M-A = `text_business_extract_v7.SAFE_MAX`（3e4）、M-B = `text_mda_extract_v2.MAX_LEN`（3e5）；
       teacher 语料：老师导出面板无本仓库提取器 → 实测存在 **Excel 单元格上限 32767**
       （107 篇长度恰好 = 32767），按数据源上限登记（`cell_max_hit`），**不等价于本仓库截断**。
     字段：`len` / `at_max_len` / `max_len_used` / `max_len_source`。
  ③ 新增 `--corpus {self,teacher}` / `--years` / `--net-dir` / `--lexicon` / `--out-prefix` /
     `--n-per-stratum`；默认输出改为 `*_v2` 文件名，**不覆盖 2026-09-15 旧产物**（其两列 flag 已失效）。
  ④ 语料按需分块读取（只保留抽样年份的 ok 行），避免整表载入；语料质量统计范围 = 抽样年份。

**角色限定**：本脚本产出的是**代理指标审计**（弱阳性信号 + 语料质量），
**不是 V5 人工判读样本**（后者见 `network_edge_precision_sample.py`）。本脚本只读，不修改任何冻结边集。
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

SEED = 20260915                 # 沿用 2026-09-15 旧审计的种子：新旧 flag 可在**同一批边**上逐条对比
N_PER_STRATUM = 60
YEARS = [2017, 2021, 2025]
OUT_DIR = ROOT / 'reports/data_gates/G0'

# 老师导出面板的 Excel 单元格上限（实测 107 篇长度恰好 = 32767）
TEACHER_CELL_MAX = 32767

CORPUS_SPEC = {
    ('self', 'ma'): {'file': 'data/processed/business_texts_v7.csv', 'id_col': 'code',
                     'lexicon': 'data/processed/product_lexicon_v2.csv',
                     'extractor': ('text_business_extract_v7', 'SAFE_MAX')},
    ('self', 'mb'): {'file': 'data/processed/mda_texts_v2.csv', 'id_col': 'code',
                     'lexicon': None, 'extractor': ('text_mda_extract_v2', 'MAX_LEN')},
    ('teacher', 'ma'): {'file': 'data/processed/teacher_mda_panel.csv', 'id_col': 'firm_id',
                        'lexicon': 'data/processed/product_lexicon_v2_teacher.csv',
                        'extractor': None, 'cell_max': TEACHER_CELL_MAX},
    ('teacher', 'mb'): {'file': 'data/processed/teacher_mda_panel.csv', 'id_col': 'firm_id',
                        'lexicon': None, 'extractor': None, 'cell_max': TEACHER_CELL_MAX},
}


# --------------------------------------------------------------------------- #
# 结构判据 / MAX_LEN 解析
# --------------------------------------------------------------------------- #
def _load_stop_names() -> tuple[set, str]:
    """§8.10.5 的 STOP_NAMES（独占行判据用）。优先从当前提取器模块读取。"""
    try:
        from text_mda_extract_v2 import STOP_NAMES
        return set(STOP_NAMES), 'text_mda_extract_v2.STOP_NAMES'
    except Exception as exc:      # pragma: no cover - 兜底，保持脚本可复跑
        fallback = {"公司治理", "重要事项", "监事会报告", "监事报告",
                    "董事、监事、高级管理人员和员工情况", "公司治理及内部控制",
                    "财务报告", "备查文件", "环境和社会责任", "社会责任报告",
                    "股份变动及股东情况", "审计报告", "独立董事述职报告",
                    "内部控制评价报告", "内部控制自我评价报告",
                    "募集资金存放与使用情况", "优先股相关情况", "债券相关情况",
                    "董事会报告"}
        return fallback, f'fallback({type(exc).__name__})'


STOP_NAMES, STOP_NAMES_SOURCE = _load_stop_names()
_CHAPTER_RE = re.compile(r'第[一二三四五六七八九十]+[章节]')
_HEAD_LINE_RE = re.compile(r'^[ \u3000]*([^\n]{2,12})[ \u3000]*$', re.M)


def stop_heading_hits(text: str) -> dict:
    """§8.10.5 结构判据：独占行章节标题 / STOP_NAMES 具名标题（排除全文 >=3 次的页眉）。

    返回 {标题: 出现次数}（已剔除页眉）；空 dict = 未见越界迹象。
    """
    counts: collections.Counter = collections.Counter()
    for m in _HEAD_LINE_RE.finditer(text):
        core = re.sub(r'\s+', '', m.group(1)).rstrip('0123456789')
        if _CHAPTER_RE.fullmatch(core) or core in STOP_NAMES:
            counts[core] += 1
    return {k: v for k, v in counts.items() if v < 3}


def resolve_max_len(corpus: str, net: str) -> tuple[int | None, str]:
    """返回 (max_len, 来源说明)。读**当前提取器**常量；teacher 语料登记 Excel 单元格上限。"""
    spec = CORPUS_SPEC[(corpus, net)]
    if spec.get('extractor'):
        mod_name, const = spec['extractor']
        try:
            mod = __import__(mod_name)
            val = int(getattr(mod, const))
            return val, f'{mod_name}.{const}'
        except Exception as exc:      # pragma: no cover - 兜底
            fallback = 30000 if net == 'ma' else 300000
            return fallback, f'fallback({type(exc).__name__}; documented {fallback})'
    if spec.get('cell_max'):
        return int(spec['cell_max']), 'teacher export Excel cell limit (empirical: 107 docs == 32767)'
    return None, 'n/a'


def text_flags(text: str, max_len: int | None) -> dict:
    hits = stop_heading_hits(text)
    return {
        'len': len(text),
        'max_len_used': max_len,
        'at_max_len': int(max_len is not None and len(text) >= max_len),
        'stop_heading_hits': int(sum(hits.values())),
        'stop_headings': '|'.join(list(hits)[:4]),
        'out_of_scope': int(len(hits) >= 1),
    }


# --------------------------------------------------------------------------- #
# 语料与词表
# --------------------------------------------------------------------------- #
def load_industry() -> dict:
    """firm_id → 申万二级名称（静态当期分类；不加 is_current_version 过滤，实测覆盖率 1.0000）。"""
    sw = pd.read_csv(ROOT / 'data/processed/sw2_industry_latest.csv',
                     usecols=['firm_id', 'l2_code'])
    l2n = pd.read_csv(ROOT / 'data/processed/sw2_l2_names.csv')
    nm = dict(zip(l2n['l2_code'].astype(str), l2n['l2_name']))
    out = {}
    for fid, code in zip(sw['firm_id'], sw['l2_code']):
        c = str(code)
        out[str(fid)] = nm.get(c[:4] + '00' if len(c) >= 4 else c, c)
    return out


def load_names() -> dict:
    fm = pd.read_csv(ROOT / 'data/processed/firm_id_map.csv', usecols=['firm_id', 'name'])
    return {str(k): v for k, v in zip(fm['firm_id'], fm['name'])}


def load_corpus(corpus: str, which: str, years: list[int], needed: set | None = None) -> dict:
    """分块读取语料，只保留抽样年份的 ok 行。返回 {(id, year): text}。

    needed 给定时只保留 (id, year) 命中的行（大幅降内存/耗时）；needed=None 时保留该年份全部 ok 行。
    """
    spec = CORPUS_SPEC[(corpus, which)]
    path = ROOT / spec['file']
    id_col = spec['id_col']
    out: dict = {}
    csv.field_size_limit(10 ** 9)
    years_s = {str(y) for y in years}
    with open(path, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('status') != 'ok':
                continue
            y = str(r.get('year', ''))
            if y not in years_s:
                continue
            key = (r[id_col], int(y))
            if needed is not None and key not in needed:
                continue
            out[key] = r['text']
    return out


def load_year_texts(corpus: str, which: str, year: int) -> list:
    """读某一年**全部** ok 行文本（M-B 年度 idf 需全年语料，与 network_mda_edges_build 同口径）。"""
    spec = CORPUS_SPEC[(corpus, which)]
    path = ROOT / spec['file']
    out = []
    csv.field_size_limit(10 ** 9)
    with open(path, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('status') != 'ok':
                continue
            if str(r.get('year', '')) != str(year):
                continue
            out.append(r['text'])
    return out


def load_lexicon_terms(path: Path, df_max: float = 0.25) -> list:
    """teacher/self 词库：取 df_ratio <= df_max 的词条，按 df_ratio 升序（稀有优先）。"""
    d = pd.read_csv(path, usecols=['term', 'df_ratio'])
    d['term'] = d['term'].astype(str)
    d['df_ratio'] = pd.to_numeric(d['df_ratio'], errors='coerce').fillna(0.0)
    d = d[d['df_ratio'] <= df_max]
    d = d.sort_values(['df_ratio', 'term'], ascending=[True, True])
    return d.drop_duplicates('term')[['term', 'df_ratio']].to_records(index=False).tolist()


def build_ma_automaton(terms) -> object:
    import ahocorasick
    A = ahocorasick.Automaton()
    for t, _df in terms:
        A.add_word(t, t)
    A.make_automaton()
    return A


def match_terms_norm(norm_text: str, automaton) -> set:
    return {v for _, v in automaton.iter(norm_text)}


def normalize(text: str) -> str:
    import unicodedata
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', str(text))).lower()


def shared_terms_ma(text_a: str, text_b: str, automaton, df_map: dict, top: int = 20) -> list:
    """共同命中产品词（交集），按词库 df 升序（稀有优先）取前 top。"""
    ha = match_terms_norm(normalize(text_a), automaton)
    hb = match_terms_norm(normalize(text_b), automaton)
    shared = sorted(ha & hb, key=lambda t: (df_map.get(t, 1.0), t))
    return shared[:top]


# --- M-B：与 network_mda_edges_build 同口径的 tokenizer + 年度 idf ------------------ #
def _mb_tokenize(text: str, jieba, stopwords, token_re) -> list:
    toks = []
    for w in jieba.cut(text):
        w = w.strip()
        if len(w) < 2 and not re.match(r'^[A-Za-z]{2,}$', w):
            continue
        if not token_re.match(w) or w in stopwords:
            continue
        toks.append(w)
    return toks


def mb_year_idf(texts: list) -> dict:
    """按 network_mda_edges_build.build_similarity 同口径算 {token: idf}（sklearn smooth idf）。

    输入必须是**该文本年全部 ok 行**（年度 idf 口径与构网一致，不用抽样池近似）。
    """
    import jieba
    jieba.setLogLevel(60)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from network_mda_edges_build import STOPWORDS, TOKEN_RE, DF_CUTOFF
    toks = [_mb_tokenize(t, jieba, STOPWORDS, TOKEN_RE) for t in texts]
    df = collections.Counter()
    for ts in toks:
        df.update(set(ts))
    n_doc = len(toks)
    kept = {w for w, c in df.items() if c / max(1, n_doc) <= DF_CUTOFF}
    toks2 = [[w for w in ts if w in kept] for ts in toks]
    vec = TfidfVectorizer(analyzer=lambda x: x, token_pattern=None, min_df=1)
    vec.fit_transform(toks2)
    return dict(zip(vec.get_feature_names_out(), vec.idf_))


def shared_tokens_mb(text_a: str, text_b: str, idf_map: dict, jieba, stopwords, token_re,
                     top: int = 20) -> list:
    ta = set(_mb_tokenize(text_a, jieba, stopwords, token_re))
    tb = set(_mb_tokenize(text_b, jieba, stopwords, token_re))
    shared = [w for w in (ta & tb) if w in idf_map]
    shared.sort(key=lambda w: (-idf_map[w], w))
    return shared[:top]


# --------------------------------------------------------------------------- #
# 抽样（代理指标审计用：top1/high/random/mid 四层，与 V5 人工判读设计无关）
# --------------------------------------------------------------------------- #
def sample_edges(net: str, year: int, net_dir: Path, n_per_stratum: int) -> pd.DataFrame:
    e = pd.read_csv(net_dir / f'{year}_similarity.csv',
                    usecols=['focal_id', 'source_id', 'similarity', 'weight'])
    e['focal_id'] = e['focal_id'].astype(str)
    e['source_id'] = e['source_id'].astype(str)
    rng = np.random.default_rng(SEED + year + (0 if net == 'ma' else 1))
    key = 'weight' if net == 'ma' else 'similarity'
    e['rank_pct'] = e.groupby('focal_id')[key].rank(pct=True)

    strata = {
        'high': e[e['rank_pct'] >= 0.95],
        'mid': e[(e['rank_pct'] >= 0.40) & (e['rank_pct'] <= 0.60)],
        'random': e,
        'top1': e[e['rank_pct'] == 1.0],
    }
    out = []
    for name, sub in strata.items():
        if len(sub) == 0:
            continue
        n = min(n_per_stratum, len(sub))
        idx = rng.choice(len(sub), size=n, replace=False)
        s = sub.iloc[idx].copy()
        s['stratum'] = name
        out.append(s)
    return pd.concat(out, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description='S3 边级语义审计 v2（代理指标；两处失效 flag 已修）')
    ap.add_argument('--corpus', choices=['self', 'teacher'], default='self',
                    help='self = 自建语料（business_texts_v7 / mda_texts_v2）；teacher = 老师导出 MD&A 面板')
    ap.add_argument('--years', default=','.join(str(y) for y in YEARS))
    ap.add_argument('--n-per-stratum', type=int, default=N_PER_STRATUM)
    ap.add_argument('--net-dir', default=None,
                    help='边表目录模板（默认 data/processed/network_{net}；teacher 列用 network_{net}_teacher）')
    ap.add_argument('--lexicon', default=None, help='M-A 词库（默认随 --corpus）')
    ap.add_argument('--out-dir', default=str(OUT_DIR))
    ap.add_argument('--out-prefix', default='s3_edge_precision_audit_v2')
    args = ap.parse_args()

    years = [int(x) for x in str(args.years).split(',') if x.strip()]
    net_dirs = {}
    for net in ('ma', 'mb'):
        if args.net_dir:
            net_dirs[net] = Path(args.net_dir)
        elif args.corpus == 'teacher':
            net_dirs[net] = ROOT / f'data/processed/network_{net}_teacher'
        else:
            net_dirs[net] = ROOT / f'data/processed/network_{net}'
    for net, d in net_dirs.items():
        missing = [y for y in years if not (d / f'{y}_similarity.csv').exists()]
        if missing:
            raise SystemExit(f'[STOP] {net} 边表缺失: {missing} in {d}')

    lex_path = Path(args.lexicon) if args.lexicon else (
        ROOT / CORPUS_SPEC[(args.corpus, 'ma')]['lexicon'])
    max_len = {net: resolve_max_len(args.corpus, net) for net in ('ma', 'mb')}

    ind = load_industry()
    nam = load_names()

    # 1) 先定抽样边，再按需读语料
    sampled = {net: {y: sample_edges(net, y, net_dirs[net], args.n_per_stratum) for y in years}
               for net in ('ma', 'mb')}
    needed = {net: {(str(r.focal_id), y) for y in years for r in sampled[net][y].itertuples()}
             | {(str(r.source_id), y) for y in years for r in sampled[net][y].itertuples()}
             for net in ('ma', 'mb')}

    cidx = {net: load_corpus(args.corpus, net, years, needed[net]) for net in ('ma', 'mb')}

    # 2) 证据来源：M-A 词库自动机（df 升序）；M-B 年度 idf
    terms = load_lexicon_terms(lex_path)
    df_map = {t: d for t, d in terms}
    automaton = build_ma_automaton(terms)
    idf_by_year = {}
    for y in years:
        docs = load_year_texts(args.corpus, 'mb', y)
        idf_by_year[y] = mb_year_idf(docs)
        print(f'[idf] {args.corpus} mb {y}: n_docs={len(docs)} '
              f'vocab={len(idf_by_year[y])}', flush=True)

    records = []
    for net in ('ma', 'mb'):
        for year in years:
            s = sampled[net][year]
            for r in s.itertuples():
                fid, sid = str(r.focal_id), str(r.source_id)
                ta = cidx[net].get((fid, year))
                tb = cidx[net].get((sid, year))
                rec = {
                    'corpus': args.corpus, 'network': net, 'year': year, 'stratum': r.stratum,
                    'focal_id': fid, 'source_id': sid,
                    'focal_name': nam.get(fid), 'source_name': nam.get(sid),
                    'similarity': r.similarity, 'weight': r.weight,
                    'focal_sw2': ind.get(fid), 'source_sw2': ind.get(sid),
                }
                rec['same_sw2'] = int(rec['focal_sw2'] == rec['source_sw2']
                                      and rec['focal_sw2'] is not None)
                if ta is None or tb is None:
                    rec.update({'text_a': 'MISSING', 'text_b': 'MISSING', 'shared_evidence': '',
                                'n_shared': 0, 'flag_a_out_of_scope': 0, 'flag_b_out_of_scope': 0,
                                'flag_a_at_max_len': 0, 'flag_b_at_max_len': 0,
                                'len_a': 0, 'len_b': 0})
                else:
                    fa = text_flags(ta, max_len[net][0])
                    fb = text_flags(tb, max_len[net][0])
                    if net == 'ma':
                        shared = shared_terms_ma(ta, tb, automaton, df_map)
                    else:
                        from network_mda_edges_build import STOPWORDS, TOKEN_RE
                        import jieba
                        jieba.setLogLevel(60)
                        shared = shared_tokens_mb(ta, tb, idf_by_year[year], jieba,
                                                  STOPWORDS, TOKEN_RE)
                    rec.update({
                        'text_a': ta[:180], 'text_b': tb[:180],
                        'shared_evidence': '|'.join(shared),
                        'n_shared': len(shared),
                        'flag_a_out_of_scope': fa['out_of_scope'],
                        'flag_b_out_of_scope': fb['out_of_scope'],
                        'flag_a_at_max_len': fa['at_max_len'], 'flag_b_at_max_len': fb['at_max_len'],
                        'len_a': fa['len'], 'len_b': fb['len'],
                    })
                records.append(rec)

    df = pd.DataFrame(records)
    outd = Path(args.out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    df.to_csv(outd / f'{args.out_prefix}_sample.csv', index=False)

    grp = df.groupby(['network', 'stratum']).agg(
        n=('same_sw2', 'size'),
        same_sw2_rate=('same_sw2', 'mean'),
        mean_shared=('n_shared', 'mean'),
        out_of_scope_docs=('flag_a_out_of_scope', 'sum'),
        at_max_len_docs=('flag_a_at_max_len', 'sum'),
    ).reset_index()
    grp['same_sw2_rate'] = grp['same_sw2_rate'].round(4)
    grp['mean_shared'] = grp['mean_shared'].round(2)

    # 语料质量（范围 = 抽样年份**全部** ok 行；flag 定义见模块 docstring ①②）
    corpus_q = {}
    sampled_q = {}
    for net in ('ma', 'mb'):
        full = []
        for y in years:
            full.extend(load_year_texts(args.corpus, net, y))
        recs = [text_flags(t, max_len[net][0]) for t in full]
        corpus_q[net] = {
            'n_ok_docs': len(recs),
            'max_len_used': max_len[net][0],
            'max_len_source': max_len[net][1],
            'share_at_max_len': round(float(np.mean([r['at_max_len'] for r in recs])), 6),
            'share_out_of_scope': round(float(np.mean([r['out_of_scope'] for r in recs])), 6),
            'flag_definition': ('out_of_scope = 独占行章节标题/STOP_NAMES 具名标题（排除 >=3 次页眉）'
                                ' >= 1；at_max_len = len >= 当前提取器 MAX_LEN（teacher 语料 = Excel 单元格上限）'),
        }
        srecs = [text_flags(t, max_len[net][0]) for t in cidx[net].values()]
        sampled_q[net] = {
            'n_docs_in_sample': len(srecs),
            'share_at_max_len': round(float(np.mean([r['at_max_len'] for r in srecs])), 6) if srecs else None,
            'share_out_of_scope': round(float(np.mean([r['out_of_scope'] for r in srecs])), 6) if srecs else None,
        }

    spurious_ma = df[(df.network == 'ma') & (df.n_shared <= 1)]
    payload = {
        'run_at': pd.Timestamp.utcnow().isoformat(),
        'role': 'proxy_indicator_audit_not_the_V5_judgement_sample',
        'seed': SEED, 'n_per_stratum': args.n_per_stratum, 'years': years,
        'corpus': args.corpus, 'net_dirs': {k: str(v) for k, v in net_dirs.items()},
        'lexicon': str(lex_path),
        'flag_fix_note': ('v2（2026-10-13）按 plan §8.10.4 修复要求替换两处失效 flag：'
                          '① 关键词越界判据 → §8.10.5 独占行结构判据；'
                          '② 固定 29900 截断阈值 → 当前提取器 MAX_LEN（teacher 语料为 Excel 上限）。'
                          '旧产物 s3_edge_precision_audit.json / _sample.csv 的对应两列已失效，不得引用。'),
        'stop_names_source': STOP_NAMES_SOURCE,
        'method_note': '同申万二级率仅作弱阳性代理，不作为人工金标准；本脚本只读，不修改冻结边集',
        'stratum_stats': grp.to_dict(orient='records'),
        'corpus_quality': corpus_q,
        'sampled_pool_quality': sampled_q,
        'ma_edges_with_le1_shared_term': {
            'n': int(len(spurious_ma)), 'total': int((df.network == 'ma').sum()),
            'share': round(len(spurious_ma) / max(1, int((df.network == 'ma').sum())), 4),
        },
        'by_network_year': df.groupby(['network', 'year']).agg(
            n=('same_sw2', 'size'), same_sw2_rate=('same_sw2', 'mean'),
            mean_shared=('n_shared', 'mean')).round(4).reset_index().to_dict(orient='records'),
    }
    (outd / f'{args.out_prefix}.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')

    print(grp.to_string(index=False))
    print()
    print(json.dumps(corpus_q, ensure_ascii=False, indent=1))
    print()
    print('M-A 共同词<=1 的抽样边:', payload['ma_edges_with_le1_shared_term'])
    print(f'\n-> {outd / (args.out_prefix + ".json")}')
    print(f'-> {outd / (args.out_prefix + "_sample.csv")}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
