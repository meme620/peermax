"""S3-B3.3：M-A 正式构网（2026-09-14）。

输入（全部冻结）：
  - 词库（self：`data/processed/product_lexicon_v2.csv`；teacher：`product_lexicon_v2_teacher.csv`）
    + 匹配器 `network_business_lexicon.py`
  - 文本（self：`business_texts.csv`；teacher：`teacher_mda_panel.csv`，均取 ok 行）
  - 聚类标签 `{net_dir}/clusters/{year}_clusters.csv`（B3.2 产出，average 链 + 逐年 K）
  - 冻结参数：similarity_method_id=M-A / cluster_linkage=average / elbow_rule=per_year_target_peer_match /
    min_product_terms=10 / ma_sample_filter=剔 J,S + 上市不足 2 年

对手集合规则（B3.2 决策）：**同团成员**（团内最小相似度阈值法实测退化，已留档）。
权重：`w_ij = sim_ij / Σ_{k ∈ 对手集合} sim_ik`（与 B2 同构：分母=被选集合，非同团不入选）。
无同团成员（singleton）→ `network_support_status=no-network-support`，权重留空不填 0。

产物（与 M-B 同合同字段）：
  {net_dir}/{year}_similarity.csv / {year}_support.csv / {year}_build_report.json

v3（2026-09-28，Phase B″2/B″4）：加 `--corpus {self,teacher}` / `--net-dir` / `--cluster-dir` /
`--lexicon` 参数化。**算法与阈值一字未改**：默认 `--corpus self` 重跑任一历史年份与既有产物
逐字节一致（B″2 验收 ①）。teacher 侧版本串 `pm_ma_ewcos_teacher_v1_{y}`、输出目录
`data/processed/network_ma_teacher/`（不覆盖 `network_ma/`）。

用法：
  python scripts/network_business_edges_build.py [--years 2017 ... 2025]
  python scripts/network_business_edges_build.py --corpus teacher            # 2014-2022
  python scripts/network_business_edges_build.py --corpus self --years 2021 --net-dir <tmp>  # B″2 回归

v4（2026-09-28，Phase B″4 施工期加固）：`_build_report.json` 的 `output` 字段改用 `rel()`
（ROOT 外/临时目录回退绝对路径，原 `Path.relative_to(ROOT)` 会抛 ValueError）；**边表与
算法不变**。

v5（2026-10-05，Phase B″10）：加 `--corpus teacher_idf`（同语料 IDF 加权对照变体，**不升主口径**）。
向量权重由 `network_business_cluster_grid.build_vectors` 按 `CORPORA[corpus]['weighting']` 切换（TF 不改、
只 IDF 加权）；本脚本只改 `params.similarity_method` / 新增 `params.term_weighting` 标签。边权重公式
`w = sim / Σ_同团 sim` 与全部阈值一字未改；`self` / `teacher` 默认路径逐字节不变。
产物：`data/processed/network_ma_teacher_idf/{y}_{similarity,support,build_report}.*`
（主口径 `network_ma_teacher/` 不被触碰）。

v6（2026-10-11，第一梯队 ⑤ / B″11-8 df 阈值网格）：加 `--df-max`（词库按既有 `df_ratio` 列过滤，
`df_ratio ≤ df_max`；走 `network_business_lexicon.load_lexicon(path, df_max=...)`）。缺省不传时与 v5 逐字节一致
（默认 = `DF_MAX_DEFAULT=0.25`）；边权重公式、对手集合规则、12 位小数舍入与全部阈值一字未改。

v7（2026-10-13，第二梯队 #8 路线 A / B″11-4）：加 `--sim-adjust {none,median}` 与语料条目
`teacher_medadj`（HP 式公司级中位数调整变体，**不升主口径**）。median 路径：
  ① `m_i` = 公司 i 当年对**全体公司**的相似度中位数（**全行含零相似对**、排除自身；由 float64 全
     相似度矩阵现算——`{y}_similarity.csv` 是**边表**、只存同团正相似对，无法还原全行，故按冻结的
     `build_vectors` 路径重建矩阵）；
  ② `sim_adj(i,j) = sim(i,j) − (m_i+m_j)/2`（HP16 PDF p.19），**不裁剪负值**；
  ③ 边集规则 = **同团 ∧ sim_adj > 0**（与主口径「同团 ∧ sim > 0」同构；负/零 sim_adj 对保留在聚类层，
     仅在边集层按既有正相似规则剔除，且**不入权重分母**——分母 = 被选集合）；`similarity` 列写 `sim_adj`，
     权重 `w = sim_adj / Σ_{同团, sim_adj>0} sim_adj`（与主口径同一功能式，只是把 sim 换成 HP 调整后的分数）。
缺省 `--sim-adjust none` 时向量、聚类消费、边权重公式、12 位舍入与全部输出字段与 v6 逐字节一致（回归门）。
产物（teacher_medadj）：`data/processed/network_ma_teacher_medadj/{y}_{similarity,support,build_report}.*`
（主口径 `network_ma_teacher/` 不被触碰）。

v8（2026-10-19，全文语料对表轮）：加语料条目 `ma_fulltext`（**中文法 × 年报全文**，对表语料敏感性列、
**不升主口径**；registry `ma_fulltext_text_corpus`）：版本串 `pm_ma_ewcos_fulltext_v1_{y}`、
输出目录 `data/processed/network_ma_fulltext/`（不覆盖 `network_ma_teacher/`）；年份 FY2017–FY2022。
**边权重公式 `w = sim / Σ_同团 sim`、对手集合规则、正相似筛选、12 位小数舍入与全部阈值一字未改**
（该条目的向量走 `build_vectors` 命中缓存分支；等价性见 `network_business_fulltext_cache_check.py`）。
其余语料默认路径逐字节不变。
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from network_business_lexicon import build_automaton, load_lexicon  # noqa: E402
from network_business_cluster_grid import build_vectors, load_refs, row_median_excl_self, SIM_ADJUST_CHOICES  # noqa: E402

# 语料口径（Phase B″2 参数化）：与 network_business_cluster_grid.CORPORA 同构（此处独立一份，
# 因为本脚本还需 ver_prefix/net_dir/years/net_use_case；两处数值必须一致，回归测试覆盖）。
CORPORA = {
    'self': {
        'path': ROOT / 'data/processed/business_texts.csv',
        'id_col': 'code',
        'ver_prefix': 'pm_ma_ewcos_v1',
        'net_dir': ROOT / 'data/processed/network_ma',
        'src_version': 'business_texts+product_lexicon_v2',
        'lexicon': ROOT / 'data/processed/product_lexicon_v2.csv',
        'years': list(range(2017, 2026)),
        'k_rule': 'per_year_target_peer_match(56)',
        'weighting': 'ew',
    },
    'teacher': {
        'path': ROOT / 'data/processed/teacher_mda_panel.csv',
        'id_col': 'firm_id',
        'ver_prefix': 'pm_ma_ewcos_teacher_v1',
        'net_dir': ROOT / 'data/processed/network_ma_teacher',
        'src_version': 'teacher_mda_panel+product_lexicon_v2_teacher',
        'lexicon': ROOT / 'data/processed/product_lexicon_v2_teacher.csv',
        'years': list(range(2014, 2023)),
        'k_rule': 'per_year_target_peer_match(44)',
        'weighting': 'ew',
    },
    # Phase B″10：同语料（老师 MD&A）IDF 加权对照变体，不升主口径（与 network_business_cluster_grid.CORPORA 同构）
    'teacher_idf': {
        'path': ROOT / 'data/processed/teacher_mda_panel.csv',
        'id_col': 'firm_id',
        'ver_prefix': 'pm_ma_idf_teacher_v1',
        'net_dir': ROOT / 'data/processed/network_ma_teacher_idf',
        'src_version': 'teacher_mda_panel+product_lexicon_v2_teacher',
        'lexicon': ROOT / 'data/processed/product_lexicon_v2_teacher.csv',
        'years': list(range(2014, 2023)),
        'k_rule': 'per_year_target_peer_match(44)',
        'weighting': 'idf',
        'idf_by_year': ROOT / 'data/processed/product_lexicon_v2_teacher_df_by_year.csv',
        'sim_adjust': 'none',
    },
    # 第二梯队 #8 路线 A / B″11-4：**同语料 HP 式公司级中位数调整变体**，不升主口径。
    # 与 `teacher` 同语料/词表/筛选/聚类/K 目标，唯一差别 = 相似度层多一步
    # sim_adj(i,j) = sim(i,j) − (m_i+m_j)/2（不裁剪负值）。
    'teacher_medadj': {
        'path': ROOT / 'data/processed/teacher_mda_panel.csv',
        'id_col': 'firm_id',
        'ver_prefix': 'pm_ma_ewcos_teacher_medadj_v1',
        'net_dir': ROOT / 'data/processed/network_ma_teacher_medadj',
        'src_version': 'teacher_mda_panel+product_lexicon_v2_teacher',
        'lexicon': ROOT / 'data/processed/product_lexicon_v2_teacher.csv',
        'years': list(range(2014, 2023)),
        'k_rule': 'per_year_target_peer_match(44)',
        'weighting': 'ew',
        'sim_adjust': 'median',
    },
    # 2026-10-19 全文语料对表轮（registry `ma_fulltext_text_corpus`）：**中文法 × 年报全文**，
    # 对表语料敏感性列、**不升主口径**。与 `teacher` 同词表口径（df 0.25）/ 同筛选 / 同算法 /
    # 同 K 目标 44；唯一差别 = 语料（年报全文 vs MD&A 章节）→ 词表 df 在新语料上重算。
    # 文本读 parquet 面板；词条命中走 F1 缓存（见 network_business_fulltext_cache_check.py 等价性证据）。
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
        'k_rule': 'per_year_target_peer_match(44)',
        'weighting': 'ew',
    },
}
NET_USE_CASE = 'product_market_ma'          # 与 M-B 侧同构：use case 只标方法（ma/mb），语料由 network_version 区分


def rel(p):
    """相对 ROOT 的 POSIX 路径；ROOT 外或相对路径（回归用临时目录）回退为绝对路径。"""
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()
COLS = ['network_use_case', 'network_version', 'focal_id', 'source_id', 'similarity', 'weight',
        'available_at', 'source_version', 'coverage_flag', 'missing_reason', 'network_support_status']


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
                         'sim(i,j)−(m_i+m_j)/2（HP16 p.19，不裁剪负值；边集 = 同团 ∧ sim_adj>0）。'
                         '缺省按语料（self/teacher/teacher_idf=none，teacher_medadj=median）')
    ap.add_argument('--years', type=int, nargs='*', default=None, help='文本年（缺省按语料）')
    ap.add_argument('--net-dir', default=None, help='输出目录（缺省按语料；teacher 不覆盖 network_ma/）')
    ap.add_argument('--cluster-dir', default=None, help='聚类标签目录（缺省 = net_dir/clusters）')
    ap.add_argument('--lexicon', default=None, help='词库路径（缺省按语料）')
    ap.add_argument('--df-max', type=float, default=None,
                    help='词库 df 阈值（按既有 df_ratio 列过滤：df_ratio ≤ df_max；缺省 = load_lexicon '
                         '默认 0.25 = 现状/主口径）。用于第一梯队 ⑤ / B″11-8 的 10%%/25%%/100%% 网格')
    args = ap.parse_args()

    spec = CORPORA[args.corpus]
    sim_adjust = args.sim_adjust if args.sim_adjust is not None else spec.get('sim_adjust', 'none')
    years = args.years if args.years else list(spec['years'])
    outd = Path(args.net_dir) if args.net_dir else spec['net_dir']
    clud = Path(args.cluster_dir) if args.cluster_dir else outd / 'clusters'
    lex_path = Path(args.lexicon) if args.lexicon else spec['lexicon']
    terms, lmeta = (load_lexicon(lex_path) if args.df_max is None
                    else load_lexicon(lex_path, df_max=args.df_max))
    A = build_automaton(terms)
    uni = load_refs()
    outd.mkdir(parents=True, exist_ok=True)

    for y in years:
        t0 = time.time()
        clu = pd.read_csv(clud / f'{y}_clusters.csv', dtype={'firm_id': str})
        firms, X, stats = build_vectors(y, A, uni, corpus=args.corpus, df_max=args.df_max)
        # 与聚类标签对齐（build_vectors 是确定性的，但显式校验）
        c2k = dict(zip(clu['firm_id'], clu['cluster_id']))
        labels = np.array([c2k.get(f, 0) for f in firms])
        assert len(clu) == len(firms), f'{y}: 聚类标签数 {len(clu)} != 向量数 {len(firms)}'
        assert all(f in c2k for f in firms), f'{y}: 存在无标签公司'
        Xn = normalize(X)
        # HP 式中位数调整（median 路径）：m_i 需要**全行**（含零相似对）→ 由 float64 全矩阵现算
        # （{y}_similarity.csv 是边表、只存同团正相似对，无法还原全行；build_vectors 确定性 → 同一语料）。
        mvec = None
        if sim_adjust == 'median':
            Mfull = np.asarray((Xn @ Xn.T).todense(), dtype=np.float64)
            mvec = row_median_excl_self(Mfull)
            del Mfull
        avail = f'{y + 1}-04-30'
        ver = f'{spec["ver_prefix"]}_{y}'
        n_edges = 0
        n_pairs_pos = 0          # 同团对中 sim_adj > 0（入边集）
        n_pairs_nonpos = 0       # 同团对中 sim_adj ≤ 0（按既有正相似规则剔除；聚类层未裁剪）
        sup_rows = []
        fp = outd / f'{y}_similarity.csv'
        with open(fp, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(COLS)
            order = np.argsort(labels, kind='stable')
            lab_sorted = labels[order]
            bounds = np.flatnonzero(np.r_[True, lab_sorted[1:] != lab_sorted[:-1], True])
            for bi in range(len(bounds) - 1):
                idx = order[bounds[bi]:bounds[bi + 1]]
                m = len(idx)
                if m < 2:
                    sup_rows.append((firms[int(idx[0])], 0, 0.0, 'no-network-support'))
                    continue
                # float64 全程：float32 相似度矩阵 + float32 除法会让权重带 ~1e-7 相对误差
                # （验收里表现为 8e-8 级残差），与设计式 w = sim / Σsim 的精确实现不符
                S = np.asarray((Xn[idx] @ Xn[idx].T).todense(), dtype=np.float64)
                np.fill_diagonal(S, 0.0)
                if mvec is not None:
                    # sim_adj(i,j) = sim(i,j) − (m_i+m_j)/2（HP16 p.19）；**不裁剪负值**（聚类层口径）
                    S = S - (mvec[idx][:, None] + mvec[idx][None, :]) / 2.0
                    np.fill_diagonal(S, 0.0)
                    n_pairs_pos += int((S > 0).sum())
                    n_pairs_nonpos += int((S <= 0).sum())
                    # 边集层按既有「同团 ∧ 正相似」规则取对手集合：非正 sim_adj 不入边集，
                    # 也**不入权重分母**（分母 = 被选集合，与主口径同式）；聚类层仍不裁剪。
                    np.clip(S, 0.0, None, out=S)
                wsum = S.sum(axis=1, dtype=np.float64)
                for r, i in enumerate(idx):
                    if wsum[r] <= 0:
                        sup_rows.append((firms[int(i)], 0, 0.0, 'no-network-support'))
                        continue
                    src = idx[S[r] > 0]
                    sv = S[r][S[r] > 0]
                    for j, s_ in zip(src, sv):
                        w.writerow([NET_USE_CASE, ver, firms[int(i)], firms[int(j)],
                                    round(float(s_), 12), round(float(s_ / wsum[r]), 12),  # 12 位小数：M-A 行内 peer 数可达 1200，权重中位 ~8e-4，8 位小数的相对误差可达 5e-5
                                    avail, spec['src_version'], 'covered', '', 'support'])
                        n_edges += 1
                    sup_rows.append((firms[int(i)], int(len(src)), float(wsum[r]), 'support'))
        sp_ = outd / f'{y}_support.csv'
        with open(sp_, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['focal_id', 'n_edges', 'weight_sum', 'network_support_status',
                        'network_version', 'available_at'])
            for f_, ne, ws, st in sup_rows:
                w.writerow([f_, ne, round(ws, 8) if ws else '', st, ver, avail])
        n_sup = sum(1 for r in sup_rows if r[3] == 'support')
        n_ns = len(sup_rows) - n_sup
        meta = {'network_use_case': NET_USE_CASE, 'network_version': ver, 'text_year': y,
                'n_nodes': len(firms), 'n_nodes_total': len(firms),
                'n_support': n_sup, 'n_no_support': n_ns,
                'n_edges': n_edges, 'available_at': avail,
                'k_clusters': int(len(set(labels))), 'filter_stats': stats,
                'params': {'similarity_method': ('M-A 等权 0/1 产品词向量余弦 + HP 式公司级中位数调整'
                                                 '（sim_adj = sim − (m_i+m_j)/2）'
                                            if sim_adjust == 'median' else
                                            'M-A IDF 加权产品词向量余弦'
                                            if spec.get('weighting') == 'idf' else 'M-A 等权 0/1 产品词向量余弦'),
                           'term_weighting': ('idf_smooth: ln((1+N_y)/(1+df_y))+1'
                                              if spec.get('weighting') == 'idf' else 'ew_0/1'),
                           'cluster_linkage': 'average', 'K_rule': spec['k_rule'],
                           'K': int(len(set(labels))), 'competitor_set': 'same_cluster_members',
                           'min_product_terms': 10, 'sample_filter': 'exclude_CSRC_J_S_and_ipo<2y'},
                'lexicon': lmeta, 'run_at': datetime.now(timezone.utc).isoformat(),
                'corpus': args.corpus,
                'output': rel(fp),
                'elapsed_sec': round(time.time() - t0, 1)}
        if sim_adjust != 'none':
            # 仅 median 变体留痕（none 路径不加键 → 与既有 build_report 逐字节一致）
            meta['sim_adjust'] = sim_adjust
            meta['sim_adjust_formula'] = ('sim_adj(i,j) = sim(i,j) − (m_i+m_j)/2（HP16 PDF p.19；不裁剪负值）')
            meta['m_definition'] = ('m_i = median over all n−1 peers of the same year '
                                    '(zeros included, self excluded)')
            meta['sim_adjust_stats'] = {
                'm_median': round(float(np.median(mvec)), 8),
                'm_p10': round(float(np.quantile(mvec, .1)), 8),
                'm_p90': round(float(np.quantile(mvec, .9)), 8),
                'm_min': round(float(mvec.min()), 8), 'm_max': round(float(mvec.max()), 8),
                'same_cluster_pairs_sim_adj_gt0': n_pairs_pos,
                'same_cluster_pairs_sim_adj_le0': n_pairs_nonpos,
                'share_same_cluster_pairs_sim_adj_le0': round(
                    n_pairs_nonpos / max(1, n_pairs_pos + n_pairs_nonpos), 6),
                'edge_rule': 'same_cluster AND sim_adj > 0（负/零 sim_adj 对保留在聚类层、仅在边集层剔除）'}
        (outd / f'{y}_build_report.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f"[{y}] nodes={len(firms)} support={n_sup} ns={n_ns} edges={n_edges} "
              f"K={meta['k_clusters']} med_peer={np.median([r[1] for r in sup_rows]):.0f} [{meta['elapsed_sec']}s]",
              flush=True)


if __name__ == '__main__':
    main()
