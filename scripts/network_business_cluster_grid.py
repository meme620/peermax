"""S3-B3.2：M-A 聚类参数网格对比 v2（linkage × K 规则；不读收益）。

v3（2026-09-28，Phase B″2）：加 `--corpus {self,teacher}` / `--lexicon` / `--out-report` /
`--years` 参数化（语料路径、id 列名、年份默认、报告路径随语料切换）。**算法与阈值一字未改**：
默认 `--corpus self` 的向量、相似度与既有产物逐字节一致（B″2 验收 ①）。

v4（2026-10-05，Phase B″10）：加 `teacher_idf` 语料条目（**同语料 IDF 加权对照变体，不升主口径**）
——语料/词表/筛选/聚类算法与 `teacher` 完全相同，唯一差别是向量元素由等权 1.0 换成词条在该文本年的
**smooth IDF**：`idf = ln((1+N_y)/(1+df_y)) + 1`（与 M-B 侧 sklearn `TfidfVectorizer` 默认
`smooth_idf=True` 可比口径），`df_y` 取 `data/processed/product_lexicon_v2_teacher_df_by_year.csv`
的 `doc_count_{y}`（该年 ok 文档命中该词的文档数）、`N_y = n_docs_{y}`（该年 ok 文档总数）。
**命中集合与样本筛选与权重无关** → 变体逐年 `n_nodes` / `filter_stats` 与 `teacher` 主口径逐值相等
（只有相似度分布、K_y、边权可变化）。默认口径 `--corpus self / teacher` 仍走原 `np.ones` 分支，
逐字节不变（B″2/B″10 回归硬门）。

v5（2026-10-13，第二梯队 #8 路线 A / B″11-4）：加 `--sim-adjust {none,median}` 与语料条目
`teacher_medadj`（HP 式公司级中位数调整变体，**不升主口径**）。`sim_adj(i,j) = sim(i,j) − (m_i+m_j)/2`
（HP16 PDF p.19），`m_i` = 公司 i 当年对**全体公司**的相似度中位数（**全行含零相似对**，排除自身；
口径登记见 registry `ma_similarity_adjustment` 的 change_log 2026-10-13），**不裁剪负值**；
聚类距离 `1 − sim_adj`（sim_adj ∈ [−1,1] → 距离 ∈ [0,2] **构造上非负**，无需秩变换）。
缺省 `--sim-adjust none` 时向量、相似度、聚类与全部输出字段与 v4 逐字节一致（回归门）。

v6（2026-10-19，全文语料对表轮）：加语料条目 `ma_fulltext`（**中文法 × 年报全文**，对表语料敏感性列、
**不升主口径**；registry `ma_fulltext_text_corpus` F0 前置登记）。该条目文本读 parquet 面板
（`annual_fulltext_panel.parquet`），词条命中走 F1 命中缓存 → `build_vectors_from_hits`（新增函数，
**仅该条目使用**；等价性见 `network_business_fulltext_cache_check.py` 报告）。`--corpus` 缺省仍为 `self` /
`teacher` 既有分支，`build_vectors` 现场匹配路径与全部输出字段**逐字节不变**（回归门：默认路径不进入
`hits_dir` 分支）。
**v6-fix（同日，F3a part C 抓到）**：`build_vectors_from_hits` 的 df 掩码原误取 `hits_lexicon`
（`product_lexicon_v2.csv`，其 `df_ratio` = 老师 MD&A / 自建业务语料的 df）→ 保留 1,354,967 词条，
与 `load_lexicon(spec['lexicon'])`（全文语料重算 df，保留 1,350,179）建的自动机不一致 → 断言停机。
修正为**取本语料词表** `spec['lexicon']` 的 df 掩码，`hits_lexicon` 仅作行号参照（附行序逐行一致性断言）；
非 `ma_fulltext` 路径不经过该函数，逐字节不变。

v1 的教训：以"组内离差和"作肘点判据在 single-linkage + maxclust 下退化为平线
（K 增长只在树上切出巨团+单点），且未评估参数对**网络**的实际影响。
v2 在 K 网格上直接评估三类指标：
  A 聚类质量：团内平均相似度 within_sim
  B 网络密度：中位/均值 peer 数、孤立率
  C 行业一致性：边中同申万二级占比及其相对基准的 lift（基准=全对随机同行业率）
并给出两条候选肘点规则（kneedle / max_curvature）在 within 与 lift 曲线上的 K*。

判据（plan B3.2"团体内平均相似度最高且团体数稳定"的落地）：
  1. 稳定性：同一规则在 3 个参考年给出相近 K*；
  2. 质量：K* 下 within_sim 与 sw2_lift 越高越好。
冻结决定写入产物 JSON 的 `decision` 段。

用法：
  python scripts/network_business_cluster_grid.py --years 2019 2021 2023 --kmax 60
  python scripts/network_business_cluster_grid.py --corpus teacher --years 2016 2018 2020 2022 \
      --out-report reports/data_gates/G0/s3_b3_ma_cluster_grid_teacher.json
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
from network_business_lexicon import DF_MAX_DEFAULT, build_automaton, load_lexicon, normalize as norm_text  # noqa: E402

REPD = ROOT / 'reports/data_gates/G0'
# 语料真源 = 下面的 CORPORA（B″2 起）；此前的模块级单值 CORPUS=business_texts.csv 已并入
# CORPORA['self']['path']，避免第二份真源（无外部脚本 import 该常量）。
IDMAP = ROOT / 'data/processed/firm_id_map.csv'
IND = ROOT / 'data/raw/baostock/industry/all_current.csv'
SW2L = ROOT / 'data/processed/sw2_industry_latest.csv'
MIN_TERMS = 10
IPO_MIN_YEARS = 2
LINKAGES = ['single', 'complete', 'average']
# 相似度层调整（Phase B″11-4 / 第二梯队 #8 路线 A）：none = 主口径（默认）；median = HP 式中位数调整变体
SIM_ADJUST_CHOICES = ['none', 'median']

# 语料口径（Phase B″2 参数化）：路径 / id 列名 / 版本前缀 / 网络目录 / 年份默认值。
# 冻结阈值与筛选规则（MIN_TERMS / IPO_MIN_YEARS / J,S 门类 / linkage / df_max）不随语料变。
CORPORA = {
    'self': {
        'path': ROOT / 'data/processed/business_texts.csv',
        'id_col': 'code',
        'ver_prefix': 'pm_ma_ewcos_v1',
        'net_dir': ROOT / 'data/processed/network_ma',
        'src_version': 'business_texts+product_lexicon_v2',
        'lexicon': ROOT / 'data/processed/product_lexicon_v2.csv',
        'years': list(range(2017, 2026)),
        # grid_years 同时是该语料下 `--years` 的**默认值**（self 保持历史默认 [2019,2021,2023] 不变）
        'grid_years': [2019, 2021, 2023],
        'report': REPD / 's3_b3_ma_cluster_grid.json',
        'weighting': 'ew',
        'sim_adjust': 'none',
    },
    'teacher': {
        'path': ROOT / 'data/processed/teacher_mda_panel.csv',
        'id_col': 'firm_id',
        'ver_prefix': 'pm_ma_ewcos_teacher_v1',
        'net_dir': ROOT / 'data/processed/network_ma_teacher',
        'src_version': 'teacher_mda_panel+product_lexicon_v2_teacher',
        'lexicon': ROOT / 'data/processed/product_lexicon_v2_teacher.csv',
        'years': list(range(2014, 2023)),
        'grid_years': [2016, 2018, 2020, 2022],
        'report': REPD / 's3_b3_ma_cluster_grid_teacher.json',
        'weighting': 'ew',
        'sim_adjust': 'none',
    },
    # Phase B″10：**同语料（老师 MD&A）IDF 加权对照变体**，不升主口径。
    # 除向量元素权重外与 `teacher` 完全同构（同语料 / 同词表 / 同筛选 / 同算法 / 同 K 目标）。
    'teacher_idf': {
        'path': ROOT / 'data/processed/teacher_mda_panel.csv',
        'id_col': 'firm_id',
        'ver_prefix': 'pm_ma_idf_teacher_v1',
        'net_dir': ROOT / 'data/processed/network_ma_teacher_idf',
        'src_version': 'teacher_mda_panel+product_lexicon_v2_teacher',
        'lexicon': ROOT / 'data/processed/product_lexicon_v2_teacher.csv',
        'years': list(range(2014, 2023)),
        'grid_years': [2016, 2018, 2020, 2022],
        'report': REPD / 's3_b3_ma_cluster_grid_teacher_idf.json',
        'weighting': 'idf',
        'idf_by_year': ROOT / 'data/processed/product_lexicon_v2_teacher_df_by_year.csv',
        'sim_adjust': 'none',
    },
    # 第二梯队 #8 路线 A / B″11-4：**同语料 HP 式公司级中位数调整变体**，不升主口径。
    # 除相似度层多一步 sim_adj(i,j) = sim(i,j) − (m_i+m_j)/2 外与 `teacher` 完全同构
    # （同语料 / 同词表 / 同筛选 / 同算法 / 同 K 目标 44）。
    'teacher_medadj': {
        'path': ROOT / 'data/processed/teacher_mda_panel.csv',
        'id_col': 'firm_id',
        'ver_prefix': 'pm_ma_ewcos_teacher_medadj_v1',
        'net_dir': ROOT / 'data/processed/network_ma_teacher_medadj',
        'src_version': 'teacher_mda_panel+product_lexicon_v2_teacher',
        'lexicon': ROOT / 'data/processed/product_lexicon_v2_teacher.csv',
        'years': list(range(2014, 2023)),
        'grid_years': [2016, 2018, 2020, 2022],
        'report': REPD / 's3_b3_ma_cluster_grid_teacher_medadj.json',
        'weighting': 'ew',
        'sim_adjust': 'median',
    },
    # 2026-10-19 全文语料对表轮（F0 登记 `ma_fulltext_text_corpus`）：**中文法 × 年报全文**，
    # 对表语料敏感性列、**不升主口径**。与 `teacher` 同词表口径（df 阈值 0.25）/ 同筛选 / 同算法 /
    # 同 K 目标 44，唯一差别 = 语料（年报全文 vs 老师 MD&A 章节）→ 词表 df 在新语料上重算。
    # 文本读 parquet 面板；词条命中走 F1 缓存（全子串命中按词表行号存盘，免 18 GB 重复扫描），
    # 缓存与现场匹配的等价性由 scripts/network_business_fulltext_cache_check.py 逐篇核验。
    'ma_fulltext': {
        'path': ROOT / 'data/processed/annual_fulltext_panel.parquet',
        'id_col': 'firm_id',
        'format': 'parquet',
        'hits_dir': ROOT / 'data/processed/annual_fulltext_hits',
        'hits_lexicon': ROOT / 'data/processed/product_lexicon_v2.csv',
        'ver_prefix': 'pm_ma_ewcos_fulltext_v1',
        'net_dir': ROOT / 'data/processed/network_ma_fulltext',
        'src_version': 'annual_fulltext_panel+product_lexicon_v2_fulltext',
        'lexicon': ROOT / 'data/processed/product_lexicon_v2_fulltext.csv',
        'years': list(range(2017, 2023)),
        'grid_years': [2017, 2019, 2021, 2022],
        'report': REPD / 's3_b3_ma_cluster_grid_ma_fulltext.json',
        'weighting': 'ew',
        'sim_adjust': 'none',
    },
}

# ---- IDF 表（模块级缓存：9 年共用一份 15 MB CSV，避免逐年重复读）----
_IDF_TABLES: dict = {}


def load_idf_table(path):
    """读逐年 doc_count 表（一次），返回 (DataFrame, years, term->row 位置)。"""
    key = str(Path(path))
    if key not in _IDF_TABLES:
        d = pd.read_csv(key)
        years = sorted(int(c.rsplit('_', 1)[1]) for c in d.columns if c.startswith('doc_count_'))
        terms = d['term'].astype(str).to_numpy()
        pos = {t: i for i, t in enumerate(terms)}
        _IDF_TABLES[key] = (d, years, pos)
    return _IDF_TABLES[key]


def idf_vector(path, year):
    """该文本年的 (term->位置, idf float32 向量)：idf = ln((1+N_y)/(1+df_y)) + 1。

    口径与 M-B（sklearn TfidfVectorizer 默认 smooth_idf=True）可比；dtype 保持 float32
    （与主口径向量内部一致）。报告可另列纯式 ln(N_y/df_y) 作说明，实现只用 smooth 版。
    """
    d, years, pos = load_idf_table(path)
    if year not in years:
        raise KeyError(f'IDF 表 {path} 缺文本年 {year}（已有 {years}）')
    dc = d[f'doc_count_{year}'].to_numpy(dtype=np.float64)
    n = float(d[f'n_docs_{year}'].iloc[0])
    return pos, (np.log((1.0 + n) / (1.0 + dc)) + 1.0).astype(np.float32)


def row_median_excl_self(C):
    """m_i = 公司 i 当年对**全体公司**的相似度中位数（HP16 PDF p.19 的 `m_i`）。

    口径（registry `ma_similarity_adjustment` change_log 2026-10-13 登记）：
      - **主口径 = 全行含零相似对**：分母 = 当年全部入网公司 n_i − 1 个对手（排除自身），
        零相似对计入（余弦相似度为 0 的对是"最不像"的真实取值，不是缺失）；
      - 诊断变体 = 仅正相似（`sim > 0`），见 `row_median_positive_only`。
    实现：把对角（自身）置 NaN 后按行取中位数（`np.nanmedian`，float64）。
    """
    A = np.array(C, dtype=np.float64, copy=True)
    np.fill_diagonal(A, np.nan)
    return np.nanmedian(A, axis=1)


def row_median_positive_only(C):
    """诊断变体：仅对 `sim > 0` 的对手取中位数（零相似对不计入分母）。"""
    A = np.array(C, dtype=np.float64, copy=True)
    np.fill_diagonal(A, 0.0)
    A[A <= 0] = np.nan
    with np.errstate(invalid='ignore'):
        return np.nanmedian(A, axis=1)


def median_adjust(C, m=None):
    """`sim_adj(i,j) = sim(i,j) − (m_i + m_j)/2`（HP16 PDF p.19）；**不裁剪负值**。

    值域：sim ∈ [0,1]、m ∈ [0,1] → sim_adj ∈ [−1,1] → 聚类距离 `1 − sim_adj` ∈ [0,2]
    **构造上非负**，故不需要秩变换（plan §8 B″11-4 ③ 的备选口径不触发）。
    返回 `(C_adj float64, m float64)`；`m=None` 时按主口径（全行含零相似对）现算。
    """
    C64 = np.asarray(C, dtype=np.float64)
    if m is None:
        m = row_median_excl_self(C64)
    return C64 - (m[:, None] + m[None, :]) / 2.0, np.asarray(m, dtype=np.float64)


def adjust_stats(C, C_adj, m):
    """变体留痕：m_i 分布 + 负值占比（不裁剪的直接后果）。"""
    off = ~np.eye(len(m), dtype=bool)
    return {'sim_adjust': 'median', 'm_definition': 'median over all n−1 peers, zeros included, self excluded',
            'm_median': round(float(np.median(m)), 8), 'm_p10': round(float(np.quantile(m, .1)), 8),
            'm_p90': round(float(np.quantile(m, .9)), 8), 'm_min': round(float(m.min()), 8),
            'm_max': round(float(m.max()), 8),
            'share_pairs_sim_adj_le0': round(float((C_adj[off] <= 0).mean()), 6),
            'share_pairs_sim_le0_raw': round(float((np.asarray(C, dtype=np.float64)[off] <= 0).mean()), 6)}


def load_refs():
    m = pd.read_csv(IDMAP, dtype={'code6': str})
    m['ipo'] = pd.to_datetime(m['ipo_date'], errors='coerce')
    ind = pd.read_csv(IND, dtype=str)
    ind['code6'] = ind['code'].str.replace('.', '', regex=False).str[2:]
    ind['menlei'] = ind['industry'].astype(str).str[:1]
    m = m.merge(ind[['code6', 'menlei']], on='code6', how='left')
    lat = pd.read_csv(SW2L, dtype={'code6': str, 'l2_code': str})
    m = m.merge(lat[['code6', 'l2_code']], on='code6', how='left')
    return m.set_index('firm_id')


# ---- 命中缓存路径（2026-10-19，`ma_fulltext` 全文语料专用；主口径路径不经过此处）----
_LEX_KEEP_CACHE: dict = {}
_LEX_KEEP_VERIFIED: set = set()


def _lexicon_keep_mask(lex_path, df_max):
    """按 `network_business_lexicon.load_lexicon` 同规则返回 (terms, keep_mask)；按 (path, df_max) 缓存。"""
    key = (str(lex_path), float(df_max))
    if key not in _LEX_KEEP_CACHE:
        d = pd.read_csv(lex_path, dtype=str)
        terms = [str(t) for t in d['term']]
        if 'df_ratio' in d.columns:
            dr = pd.to_numeric(d['df_ratio'], errors='coerce')
            keep = (dr.fillna(0.0) <= float(df_max)).to_numpy()
        else:
            keep = d.get('keep_df25', pd.Series(1, index=d.index)).astype(str).eq('1').to_numpy()
        _LEX_KEEP_CACHE[key] = (terms, keep)
    return _LEX_KEEP_CACHE[key]


def build_vectors_from_hits(year, A, uni, corpus, df_max=None):
    """由命中缓存重建产品词向量（**与 build_vectors 现场匹配路径语义等价**）。

    缓存 = `{hits_dir}/{year}_hits.npz`（F1 产出）：`hits_idx` 为该行命中词条的**词表行号**
    （行号 = `hits_lexicon` 的行序；该文件仅作行号参照），`offsets` 为逐行切片，`firm_id` 与面板同年行序一致。
    **df 掩码取 `spec['lexicon']`**（本语料重算 df 的词表，v6-fix 2026-10-19）。
    向量构建与 `build_vectors` 逐步一致（筛选顺序、MIN_TERMS、等权 0/1、稀疏列=命中词条）；
    唯一差别 = 稀疏矩阵**列序**（等价于列置换，余弦矩阵不变，见缓存核验证据）。
    """
    spec = CORPORA[corpus]
    df_max = DF_MAX_DEFAULT if df_max is None else float(df_max)
    # v6-fix（2026-10-19，F3a part C 抓到）：**df 掩码必须取本语料词表** `spec['lexicon']`
    # （= 在全文语料上重算 df 的 product_lexicon_v2_fulltext.csv；registry `ma_fulltext_text_corpus` ③
    # 「词表 = 在本语料上重算 df」+ df 阈值 0.25 沿用）；`spec['hits_lexicon']` 只提供**行号参照**
    # （F1 命中缓存的 hits_idx = 该文件行号），两文件行序必须逐行一致（下断言，防行号错位）。
    # 修订前误用 `spec['hits_lexicon']` 的 df_ratio（= 老师 MD&A / 自建业务语料的 df）→ 保留 1,354,967
    # 词条 vs 正确 1,350,179（差 4,788）→ 与 `load_lexicon(spec['lexicon'])` 建的自动机不一致（断言停机）。
    terms, keep = _lexicon_keep_mask(spec['lexicon'], df_max)
    kept_terms = [t for t, k in zip(terms, keep) if k]
    ref_terms, _ = _lexicon_keep_mask(spec['hits_lexicon'], df_max)
    assert len(ref_terms) == len(terms) and ref_terms == terms, (
        f'{corpus}: 语料词表 {spec["lexicon"].name} 与命中缓存参照词表 {spec["hits_lexicon"].name} '
        '行序不一致（缓存行号会错位）')
    key = (str(spec['lexicon']), str(spec['hits_lexicon']), df_max)
    if key not in _LEX_KEEP_VERIFIED:
        # 不变量：自动机词集 == 词表按 df 阈值过滤后的词集（防 df 口径漂移：--df-max / 词表版本）
        assert set(A.keys()) == set(kept_terms), (
            f'{corpus}: 自动机词集（{len(set(A.keys()))}）与词表 df 过滤词集（{len(set(kept_terms))}）不一致')
        _LEX_KEEP_VERIFIED.add(key)
    col_of = np.full(len(terms), -1, dtype=np.int64)
    col_of[np.flatnonzero(keep)] = np.arange(int(keep.sum()), dtype=np.int64)

    pan = pd.read_parquet(spec['path'], columns=['firm_id', 'year', 'status'])
    pan = pan[pan['year'] == year]
    z = np.load(Path(spec['hits_dir']) / f'{year}_hits.npz')
    assert len(z['firm_id']) == len(pan) and bool((z['firm_id'] == pan['firm_id'].to_numpy()).all()), \
        f'{year}: 命中缓存与面板行序不一致（缓存需按现行面板重建）'
    idx, off = z['hits_idx'], z['offsets']
    ok = (pan['status'] == 'ok').to_numpy()

    firms, rows, cols, vocab = [], [], [], {}
    stats = {'docs': 0, 'drop_menlei': 0, 'drop_ipo': 0, 'drop_min': 0}
    cutoff = pd.Timestamp(f'{year}-12-31')
    for i, code in enumerate(pan['firm_id'].to_numpy()):
        if not ok[i]:
            continue
        stats['docs'] += 1
        info = uni.loc[code] if code in uni.index else None
        if info is not None:
            if info['menlei'] in ('J', 'S'):
                stats['drop_menlei'] += 1
                continue
            if pd.notna(info['ipo']) and info['ipo'] + pd.DateOffset(years=IPO_MIN_YEARS) > cutoff:
                stats['drop_ipo'] += 1
                continue
        h = col_of[idx[off[i]:off[i + 1]]]
        h = h[h >= 0]
        if len(h) < MIN_TERMS:
            stats['drop_min'] += 1
            continue
        ri = len(firms)
        firms.append(code)
        for c in h:
            rows.append(ri)
            cols.append(vocab.setdefault(int(c), len(vocab)))
    X = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                      shape=(len(firms), max(1, len(vocab))), dtype=np.float32)
    stats['firms_kept'] = len(firms)
    stats['active_vocab'] = len(vocab)
    stats['mean_terms'] = round(float(X.getnnz(axis=1).mean()), 1) if len(firms) else 0.0
    stats['build_vectors_path'] = 'hits_cache'
    return firms, X, stats


def build_vectors(year, A, uni, corpus='self', df_max=None):
    """逐年构产品词向量（稀疏 CSR）：等权 0/1（默认）或 IDF 加权（Phase B″10 对照变体）。

    corpus='self'（默认）→ business_texts.csv，id 列 `code`（历史口径，逐字节不变）；
    corpus='teacher'      → teacher_mda_panel.csv，id 列 `firm_id`（统一重命名为 `code`）。
    corpus='teacher_idf'  → 同上语料，但向量元素 = 该词文本年 smooth IDF（不升主口径）。
    corpus='ma_fulltext'  → annual_fulltext_panel.parquet（年报全文，**对表语料敏感性列**）+
      命中缓存路径（`build_vectors_from_hits`；语义等价、`df_max` 用于按词表 df 列复现过滤）。
    `uni`（load_refs() 以 firm_id 为索引）对两个语料同一映射，无需按语料切换。

    **权重不改变命中集合与样本筛选**：`weighting='ew'` 分支仍走 `np.ones`（与改动前逐字节等价），
    `weighting='idf'` 只替换 data 数组；行/列结构、筛选计数、active_vocab、mean_terms 完全一致。
    """
    spec = CORPORA[corpus]
    if spec.get('hits_dir') is not None:
        return build_vectors_from_hits(year, A, uni, corpus, df_max=df_max)
    weighting = spec.get('weighting', 'ew')
    pos = idf = None
    if weighting == 'idf':
        pos, idf = idf_vector(spec['idf_by_year'], year)
    d = pd.read_csv(spec['path'], usecols=[spec['id_col'], 'year', 'text', 'status'])
    if spec['id_col'] != 'code':
        d = d.rename(columns={spec['id_col']: 'code'})
    d = d[(d['status'] == 'ok') & (d['year'] == year)]
    firms, rows, cols, vocab, vals = [], [], [], {}, []
    stats = {'docs': 0, 'drop_menlei': 0, 'drop_ipo': 0, 'drop_min': 0}
    cutoff = pd.Timestamp(f'{year}-12-31')
    for code, text in zip(d['code'], d['text']):
        stats['docs'] += 1
        info = uni.loc[code] if code in uni.index else None
        if info is not None:
            if info['menlei'] in ('J', 'S'):
                stats['drop_menlei'] += 1
                continue
            if pd.notna(info['ipo']) and info['ipo'] + pd.DateOffset(years=IPO_MIN_YEARS) > cutoff:
                stats['drop_ipo'] += 1
                continue
        hits = {v for _, v in A.iter(norm_text(text))}
        if len(hits) < MIN_TERMS:
            stats['drop_min'] += 1
            continue
        ri = len(firms)
        firms.append(code)
        for t in hits:
            rows.append(ri)
            cols.append(vocab.setdefault(t, len(vocab)))
            if weighting == 'idf':
                # 命中词必在该年 doc_count 表中（doc_count ≥ 1 → 表内含 fired 词）；
                # 缺失即违反不变量 → KeyError 显式停机，不做静默丢弃
                vals.append(float(idf[pos[t]]))
    data = np.asarray(vals, dtype=np.float32) if weighting == 'idf' else np.ones(len(rows), dtype=np.float32)
    X = sp.csr_matrix((data, (rows, cols)),
                      shape=(len(firms), max(1, len(vocab))), dtype=np.float32)
    stats['firms_kept'] = len(firms)
    stats['active_vocab'] = len(vocab)
    stats['mean_terms'] = round(float(X.getnnz(axis=1).mean()), 1)
    return firms, X, stats


def curve_elbows(ks, vals):
    x = np.asarray(ks, dtype=float)
    y = np.asarray(vals, dtype=float)
    if len(x) < 3 or np.allclose(y, y[0]):
        return None, None
    xn = (x - x[0]) / (x[-1] - x[0])
    yn = (y - y[0]) / (y[-1] - y[0] + 1e-12)
    d = np.abs((yn[0] - yn[-1]) * xn - (xn[0] - xn[-1]) * yn + xn[-1] * yn[0] - yn[-1] * xn[0])
    k_knee = int(x[int(np.argmax(d))])
    g = np.gradient(np.gradient(yn, xn), xn)
    k_curv = int(x[int(np.argmax(np.abs(g)))])
    return k_knee, k_curv


def eval_K(C, labels, l2, base_rate):
    """由切分导出：团内最小相似度阈值 → 对手集合（sim > 阈值）→ 网络指标。"""
    n = C.shape[0]
    thr = np.empty(n, dtype=np.float32)
    sizes = []
    for k in np.unique(labels):
        idx = np.where(labels == k)[0]
        sizes.append(len(idx))
        if len(idx) > 1:
            sub = C[np.ix_(idx, idx)].copy()
            np.fill_diagonal(sub, 2.0)
            thr[idx] = float(sub.min())
        else:
            thr[idx] = 1.0
    mask = C > thr[:, None]
    np.fill_diagonal(mask, False)
    peers = mask.sum(axis=1)
    same = (l2[:, None] == l2[None, :])
    edges_tot = int(mask.sum())
    edges_same = int((mask & same).sum())
    sizes = np.array(sizes)
    return {
        'n_clusters': int(len(sizes)),
        'size_median': float(np.median(sizes)), 'size_max': int(sizes.max()),
        'singleton_share': round(float((sizes == 1).mean()), 4),
        'peer_median': float(np.median(peers)), 'peer_mean': round(float(peers.mean()), 1),
        'peer_p90': float(np.quantile(peers, .9)),
        'isolated_share': round(float((peers == 0).mean()), 4),
        'edges_per_firm_mean': round(edges_tot / n, 2),
        'same_sw2_edge_share': round(edges_same / max(1, edges_tot), 4),
        'sw2_lift': round((edges_same / max(1, edges_tot)) / max(1e-9, base_rate), 2),
    }


def run_year(year, A, uni, kmax, corpus='self', sim_adjust=None):
    t0 = time.time()
    firms, X, stats = build_vectors(year, A, uni, corpus=corpus)
    Xn = normalize(X)
    C = (Xn @ Xn.T).toarray().astype(np.float32)
    np.fill_diagonal(C, 1.0)
    adj = None
    if (sim_adjust or CORPORA[corpus].get('sim_adjust', 'none')) == 'median':
        C_adj, m = median_adjust(C)                 # sim_adj = sim − (m_i+m_j)/2（不裁剪负值）
        adj = adjust_stats(C, C_adj, m)
        C = C_adj.astype(np.float32)
        np.fill_diagonal(C, 1.0)
    D = np.clip(1.0 - C, 0, 2)
    np.fill_diagonal(D, 0.0)
    l2 = np.array([uni.loc[f, 'l2_code'] if (f in uni.index and isinstance(uni.loc[f, 'l2_code'], str))
                   else 'NA' for f in firms])
    iu = np.triu_indices(len(firms), k=1)
    base_rate = float((l2[iu[0]] == l2[iu[1]]).mean())
    cond = squareform(D, checks=False)
    out = {'year': year, 'n_firms': len(firms), 'filter_stats': stats,
           'base_same_sw2_rate': round(base_rate, 5), 'prep_sec': round(time.time() - t0, 1),
           'linkages': {}}
    if adj:
        out['sim_adjust_stats'] = adj          # 仅 median 变体留痕（none 路径不加键）
    ks = list(range(2, kmax + 1))
    for lk in LINKAGES:
        t1 = time.time()
        Z = linkage(cond, method=lk)
        rows = []
        for K in ks:
            lab = fcluster(Z, K, criterion='maxclust')
            vals = []
            for k in np.unique(lab):
                idx = np.where(lab == k)[0]
                if len(idx) > 1:
                    sub = C[np.ix_(idx, idx)]
                    iu2 = np.triu_indices(len(idx), k=1)
                    vals.append(float(sub[iu2].mean()))
            m = eval_K(C, lab, l2, base_rate)
            m['K'] = K
            m['within_sim'] = round(float(np.mean(vals)), 5) if vals else None
            rows.append(m)
        wk = [r['K'] for r in rows if r['within_sim']]
        wv = [r['within_sim'] for r in rows if r['within_sim']]
        lift_vals = [r['sw2_lift'] for r in rows]
        k_w_knee, k_w_curv = curve_elbows(wk, wv)
        k_l_knee, k_l_curv = curve_elbows(ks, lift_vals)
        pick = lambda K: next((r for r in rows if r['K'] == K), None)
        out['linkages'][lk] = {
            'sec': round(time.time() - t1, 1),
            'curve': rows,
            'elbow_within': {'kneedle': k_w_knee, 'curvature': k_w_curv},
            'elbow_lift': {'kneedle': k_l_knee, 'curvature': k_l_curv},
            'at_within_kneedle': pick(k_w_knee),
            'at_lift_kneedle': pick(k_l_knee),
            'at_lift_curvature': pick(k_l_curv),
        }
        print(f"  [{year}/{lk}] K_within_knee={k_w_knee} | K_lift_knee={k_l_knee} K_lift_curv={k_l_curv} "
              f"| at lift_knee: {json.dumps(out['linkages'][lk]['at_lift_kneedle'], ensure_ascii=False)}", flush=True)
    out['elapsed_sec'] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', choices=['self', 'teacher', 'teacher_idf', 'teacher_medadj', 'ma_fulltext'],
                    default='self',
                    help='文本语料：self=自建业务描述（文本稳健性列）；teacher=老师 MD&A（主口径）；'
                         'teacher_idf=同语料 IDF 加权对照变体（Phase B″10，不升主口径）；'
                         'teacher_medadj=同语料 HP 式中位数调整变体（第二梯队 #8 路线 A，不升主口径）；'
                         'ma_fulltext=自建**年报全文**面板（对表语料敏感性列，2026-10-19，不升主口径）')
    ap.add_argument('--sim-adjust', choices=SIM_ADJUST_CHOICES, default=None,
                    help='相似度层调整（Phase B″11-4）：none=主口径（默认）；median=sim_adj(i,j)='
                         'sim(i,j)−(m_i+m_j)/2（HP16 p.19，不裁剪负值）。缺省按语料（self/teacher/'
                         'teacher_idf=none，teacher_medadj=median）')
    ap.add_argument('--lexicon', default=None,
                    help='词库路径（缺省按语料取 CORPORA[corpus]["lexicon"]）')
    ap.add_argument('--years', type=int, nargs='*', default=None,
                    help='网格年份（缺省按语料取 grid_years；self 默认保持历史 [2019,2021,2023]）')
    ap.add_argument('--kmax', type=int, default=60)
    ap.add_argument('--out-report', default=None,
                    help='报告路径（缺省按语料取 CORPORA[corpus]["report"]；teacher 绝不覆盖历史报告）')
    args = ap.parse_args()

    spec = CORPORA[args.corpus]
    sim_adjust = args.sim_adjust if args.sim_adjust is not None else spec.get('sim_adjust', 'none')
    years = args.years if args.years else list(spec['grid_years'])
    lex_path = Path(args.lexicon) if args.lexicon else spec['lexicon']
    out_report = Path(args.out_report) if args.out_report else spec['report']
    terms, lmeta = load_lexicon(lex_path)
    A = build_automaton(terms)
    uni = load_refs()
    rep = {'run_at': datetime.now(timezone.utc).isoformat(), 'version': 2,
           # Phase 0.3（2026-10-27）：N-9 口径注记（append-only；不改任何数值/算法）
           'metric_scope': {
               'peer_rule': 'min_threshold_within_cluster（原文第四步阈值法）：peer_i = '
                            '#{j: sim(i,j) > min_{k,l in same cluster} sim(k,l)}；可含团外公司',
               'peer_fields': ['peer_median', 'peer_mean', 'peer_p90', 'isolated_share',
                               'edges_per_firm_mean'],
               'sw2_lift_rule': '同申万二级有向边占比 / 同样本边际基线'
                                '（same_sw2_edge_share / base_same_sw2_rate）',
               'sw2_lift_fields': ['same_sw2_edge_share', 'sw2_lift'],
               'not_comparable_to': 's3_b3_ma_param_freeze_teacher.json 的同名字段'
                                    '（同团成员法：peer = 团规模−1；sw2_lift = 同团同申万二级无序对数占比 / '
                                    '同样本边际基线）',
               'official_evidence_note': '本报告（kmax=60；文本年 2016/2018/2020/2022）不是 K 规则冻结的官方证据；'
                                         '官方证据 = s3_b3_ma_param_freeze_teacher.json 的 decision.elbow_rule + '
                                         's3_ma_K_explore.json（2021，K=400）',
               'phase0_3_note': '2026-10-27 Phase 0.3 口径注记；未改任何数值、未重跑网络',
           },
           'lexicon': lmeta,
           'corpus': args.corpus, 'corpus_path': str(spec['path'].relative_to(ROOT)).replace('\\', '/'),
           'weighting': spec.get('weighting', 'ew'),
           'idf_formula': ('idf = ln((1+N_y)/(1+df_y)) + 1（smooth，与 M-B sklearn 默认可比）；'
                           'df_y = product_lexicon_v2_teacher_df_by_year.csv 的 doc_count_{y}，'
                           'N_y = n_docs_{y}（该文本年 ok 文档总数）'
                           if spec.get('weighting') == 'idf' else None),
           'filter': {'exclude_menlei': ['J', 'S'], 'min_ipo_years': IPO_MIN_YEARS,
                      'min_product_terms': MIN_TERMS}, 'kmax': args.kmax, 'years': {}}
    if sim_adjust != 'none':
        # 仅非默认（median）变体加键，避免与既有 none 报告产生键差（同 --df-max 处置）
        rep['sim_adjust'] = sim_adjust
        rep['sim_adjust_formula'] = 'sim_adj(i,j) = sim(i,j) − (m_i+m_j)/2（HP16 PDF p.19；不裁剪负值）'
        rep['m_definition'] = 'm_i = median over all n−1 peers of the same year (zeros included, self excluded)'
    for y in years:
        print(f'=== year {y} ===', flush=True)
        rep['years'][str(y)] = run_year(y, A, uni, args.kmax, corpus=args.corpus, sim_adjust=sim_adjust)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"-> {out_report}")


if __name__ == '__main__':
    main()
