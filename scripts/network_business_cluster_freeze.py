"""B3.2 定稿 v2：逐年校准 K（目标密度=56），向量化聚类统计（2026-09-14）。

v1（固定 K）实测：K=500 时逐年中位 peer 13→188（14 倍漂移）、同业 lift 10.8→2.1（崩塌）。
根因：K 固定而公司数翻倍 → 团规模被动变大，凝聚度下降。故改为**逐年校准 K_y**
（每使该年公司层面中位 peer 数 = 目标值），规则冻结、K 值逐年记录。
K 的选取沿用 τ 校准同一套"阶梯函数→网格命中区间"方法。

v3（2026-09-28，Phase B″2/B″3）：加 `--corpus {self,teacher}` / `--lexicon` / `--target` /
`--out-dir` / `--out-report` / `--years` 参数化。**算法、阈值与筛选一字未改**：
  - `--corpus self`（默认）= 历史口径：语料 business_texts.csv、目标 56、
    聚类落 `data/processed/network_ma/clusters/`、报告落 `s3_b3_ma_param_freeze.json`；
  - `--corpus teacher`（主口径，Phase B″3）= 语料 teacher_mda_panel.csv、目标 = registry
    `ma_k_target_teacher`（44）、聚类落 `data/processed/network_ma_teacher/clusters/`、
    报告落 `s3_b3_ma_param_freeze_teacher.json`（**不覆盖历史自建证据**）。

产物（self）：reports/data_gates/G0/s3_b3_ma_param_freeze.json
产物（teacher）：reports/data_gates/G0/s3_b3_ma_param_freeze_teacher.json

用法：
  python scripts/network_business_cluster_freeze.py                       # 历史 self 全 9 年
  python scripts/network_business_cluster_freeze.py --corpus teacher      # 主口径 2014-2022
  python scripts/network_business_cluster_freeze.py --corpus self --years 2021 --out-dir <tmp>   # B″2 回归

v4（2026-09-28，Phase B″3 施工期加固）：报告里 ROOT 外/相对临时目录改用 `rel()` 回退为
绝对路径（原 `Path.relative_to(ROOT)` 会抛 ValueError）；**输出字段与算法不变**。

v5（2026-10-05，Phase B″10）：加 `--corpus teacher_idf`（同语料 IDF 加权对照变体，**不升主口径**）：
后半段 SPECS / CLI / registry 目标读取（`corpus.startswith('teacher')` 共用 `ma_k_target_teacher=44`）
各加一个条目；报告新增 `weighting` / `idf_formula` 字段，`decision` 段的"等权 0/1"标签按 weighting
分支改写（IDF 产物不得自称等权）。`self` / `teacher` 的算法、阈值与输出字段一字未改。
产物（teacher_idf）：`reports/data_gates/G0/s3_b3_ma_param_freeze_teacher_idf.json` +
`data/processed/network_ma_teacher_idf/clusters/`（主口径产物不被触碰）。

v6（2026-10-11，第一梯队 ⑤ / B″11-8 df 阈值网格）：加 `--df-max`（词库按既有 `df_ratio` 列过滤，
`df_ratio ≤ df_max`；走 `network_business_lexicon.load_lexicon(path, df_max=...)`）——用于 10%/25%/100% 三档网格；
**不重跑词库 df 重算**（省 ~5,855 s）。缺省不传时行为与 v5 逐字节一致（默认仍 = `load_lexicon` 的
`DF_MAX_DEFAULT=0.25`）；**显式传 `--df-max` 时**报告新增 `df_max` 字段以便逐档留痕（默认路径不加该键，
避免与既有报告产生键差）。算法、阈值、筛选、K 校准与输出格式其余一字未改。

v7（2026-10-13，第二梯队 #8 路线 A / B″11-4）：加 `--sim-adjust {none,median}` 与语料条目
`teacher_medadj`（HP 式公司级中位数调整变体，**不升主口径**）：相似度层多一步
`sim_adj(i,j) = sim(i,j) − (m_i+m_j)/2`（HP16 PDF p.19；`m_i` = 全行含零相似对的相似度中位数，
排除自身；**不裁剪负值**），聚类距离 `1 − sim_adj`（构造上 ∈[0,2] 非负，无需秩变换）。缺省
`--sim-adjust none` 时向量、相似度、K 校准与全部输出字段与 v6 逐字节一致（回归门）。
产物（teacher_medadj）：`reports/data_gates/G0/s3_b3_ma_param_freeze_teacher_medadj.json` +
`data/processed/network_ma_teacher_medadj/clusters/`（主口径产物不被触碰）。

v8（2026-10-19，全文语料对表轮）：加语料条目 `ma_fulltext`（**中文法 × 年报全文**，对表语料敏感性列、
**不升主口径**；registry `ma_fulltext_text_corpus` F0 前置登记 2026-10-19）：年份 = 对表可用文本年
FY2017–FY2022；聚类落 `data/processed/network_ma_fulltext/clusters/`、报告落
`s3_b3_ma_param_freeze_ma_fulltext.json`；K 目标 = registry `ma_k_target_teacher`（44，**不重标定**）、
df 阈值 0.25、等权 0/1、average linkage、筛选规则**一字未改**。向量走 `build_vectors` 的命中缓存分支
（等价性见 `network_business_fulltext_cache_check.py`）。`self` / `teacher` / `teacher_idf` / `teacher_medadj`
路径的算法、阈值与输出字段**逐字节不变**（回归门：非 `ma_fulltext` 语料不进入 `hits_dir` 分支；
`--df-max` 缺省时该参数对现场匹配路径无作用）。
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from network_business_cluster_grid import (CORPORA, SIM_ADJUST_CHOICES, adjust_stats,  # noqa: E402
                                   build_vectors, load_refs, median_adjust)
from network_business_lexicon import build_automaton, load_lexicon  # noqa: E402

REPD = ROOT / 'reports/data_gates/G0'
TARGET = 56                                  # self（文本稳健性列）的历史目标，不改
K_TARGET_FALLBACK = {'self': 56, 'teacher': 44, 'teacher_idf': 44, 'teacher_medadj': 44,
                     'ma_fulltext': 44}
# 语料 → （默认年份、默认聚类目录、默认报告路径、默认目标来源标签）
SPECS = {
    'self': {'years': list(range(2017, 2026)),
             'out_dir': ROOT / 'data/processed/network_ma/clusters',
             'report': REPD / 's3_b3_ma_param_freeze.json',
             'target_source': 'network_business_cluster_freeze.TARGET=56（历史冻结；对应 registry elbow_rule 文本）'},
    'teacher': {'years': list(range(2014, 2023)),
                'out_dir': ROOT / 'data/processed/network_ma_teacher/clusters',
                'report': REPD / 's3_b3_ma_param_freeze_teacher.json',
                'target_source': 'registry ma_k_target_teacher（主口径 K 目标，Phase B″3）'},
    # Phase B″10：同语料 IDF 加权对照变体（不升主口径）——同语料/词表/筛选/算法，K 目标沿用 44
    'teacher_idf': {'years': list(range(2014, 2023)),
                    'out_dir': ROOT / 'data/processed/network_ma_teacher_idf/clusters',
                    'report': REPD / 's3_b3_ma_param_freeze_teacher_idf.json',
                    'target_source': 'registry ma_k_target_teacher（与主口径同目标 44；IDF 变体不改目标）'},
    # 第二梯队 #8 路线 A：同语料 HP 式中位数调整变体（不升主口径）——同语料/词表/筛选/算法，K 目标沿用 44
    'teacher_medadj': {'years': list(range(2014, 2023)),
                       'out_dir': ROOT / 'data/processed/network_ma_teacher_medadj/clusters',
                       'report': REPD / 's3_b3_ma_param_freeze_teacher_medadj.json',
                       'target_source': 'registry ma_k_target_teacher（与主口径同目标 44；中位数调整变体不改目标）'},
    # 2026-10-19 全文语料对表轮（registry `ma_fulltext_text_corpus`，不升主口径）：
    # 语料 = 自建年报全文面板；年份 = 对表可用文本年 FY2017–FY2022（PC 需 y−1 年营收）；
    # K 目标沿用 ma_k_target_teacher=44（**不因语料切换重标定**）、df 阈值沿用 0.25、等权 0/1、average。
    'ma_fulltext': {'years': list(range(2017, 2023)),
                    'out_dir': ROOT / 'data/processed/network_ma_fulltext/clusters',
                    'report': REPD / 's3_b3_ma_param_freeze_ma_fulltext.json',
                    'target_source': 'registry ma_k_target_teacher（沿用主口径目标 44；对表语料敏感性列不重标定）'},
}
REGISTRY = ROOT / 'configs/parameter_registry.yaml'


def rel(p):
    """相对 ROOT 的 POSIX 路径；ROOT 外或相对路径（回归用临时目录）回退为绝对路径。"""
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def registry_target(corpus, default):
    """从 registry 读语料对应的 K 目标（机器可读冻结值）；缺失/异常时回退 default 并记录。

    `teacher` / `teacher_idf`（Phase B″10 IDF 变体）/ `ma_fulltext`（2026-10-19 全文语料对表列）
    共用同一冻结 K 目标 `ma_k_target_teacher`（对表语料敏感性**不重标定 K**）。
    """
    param = 'ma_k_target_teacher' if corpus.startswith(('teacher', 'ma_fulltext')) else None
    if param is None:
        return default, 'constant'
    try:
        import yaml
        d = yaml.safe_load(REGISTRY.read_text(encoding='utf-8'))
        for p in d['parameters']:
            if p.get('name') == param:
                return int(p['headline_value']), f'registry:{param}'
    except Exception as e:                                    # pragma: no cover
        return default, f'fallback（registry 读取失败：{type(e).__name__}）'
    return default, f'fallback（registry 无 {param}）'


def cluster_stats(labels, l2_idx, n_l2, base):
    """向量化：团规模、peer 数、同申万二级对数占比。"""
    n = len(labels)
    ks, inv = np.unique(labels, return_inverse=True)
    sizes = np.bincount(inv)
    peers = sizes[inv] - 1
    comb = np.bincount(inv * n_l2 + l2_idx, minlength=len(ks) * n_l2).reshape(len(ks), n_l2)
    same = int((comb * (comb - 1) // 2).sum())
    tot = int((sizes * (sizes - 1) // 2).sum())
    return {'peer_median': float(np.median(peers)), 'peer_mean': round(float(peers.mean()), 1),
            'peer_p25': float(np.quantile(peers, .25)), 'peer_p90': float(np.quantile(peers, .9)),
            'isolated_share': round(float((peers == 0).mean()), 4),
            'n_clusters': int(len(ks)), 'size_median': float(np.median(sizes)),
            'size_max': int(sizes.max()), 'size_p90': float(np.quantile(sizes, .9)),
            'same_sw2_edge_share': round(same / max(1, tot), 4),
            'sw2_lift': round((same / max(1, tot)) / max(1e-9, base), 2)}


def calibrate_K(Z, l2_idx, n_l2, base, target, lo=20, hi=4000, iters=14):
    """求 K 使中位 peer 数 = target（阶梯函数：二分定位 + 网格命中区间中点）。"""
    def med(K):
        return float(np.median(np.bincount(fcluster(Z, K, criterion='maxclust'))[
            fcluster(Z, K, criterion='maxclust')] - 1))
    l, h = lo, hi
    for _ in range(iters):
        if h - l <= 2:
            break
        m = (l + h) // 2
        v = med(m)
        if v > target:
            l = m
        elif v < target:
            h = m
        else:
            h = m
    grid = np.unique(np.linspace(l, h, 25).astype(int))
    hits = [g for g in grid if med(int(g)) == target]
    if hits:
        return int(np.median(hits)), 'exact'
    best = min(grid, key=lambda g: abs(med(int(g)) - target))
    return int(best), 'closest'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', choices=['self', 'teacher', 'teacher_idf', 'teacher_medadj', 'ma_fulltext'],
                    default='self',
                    help='文本语料：self=自建业务描述（历史列）；teacher=老师 MD&A（主口径）；'
                         'teacher_idf=同语料 IDF 加权对照变体（Phase B″10，不升主口径）；'
                         'teacher_medadj=同语料 HP 式中位数调整变体（第二梯队 #8 路线 A，不升主口径）；'
                         'ma_fulltext=自建**年报全文**面板（对表语料敏感性列，2026-10-19，不升主口径）')
    ap.add_argument('--sim-adjust', choices=SIM_ADJUST_CHOICES, default=None,
                    help='相似度层调整（Phase B″11-4）：none=主口径（默认）；median=sim_adj(i,j)='
                         'sim(i,j)−(m_i+m_j)/2（HP16 p.19，不裁剪负值）。缺省按语料（self/teacher/'
                         'teacher_idf=none，teacher_medadj=median）')
    ap.add_argument('--linkage', default='average')
    ap.add_argument('--target', type=int, default=None,
                    help='K 校准目标（缺省：self=56；teacher/teacher_idf=registry ma_k_target_teacher）')
    ap.add_argument('--years', type=int, nargs='*', default=None, help='文本年（缺省按语料）')
    ap.add_argument('--lexicon', default=None, help='词库路径（缺省按语料取 CORPORA[corpus]["lexicon"]）')
    ap.add_argument('--out-dir', default=None, help='聚类标签目录（缺省按语料）')
    ap.add_argument('--out-report', default=None, help='报告路径（缺省按语料；teacher 不覆盖历史）')
    ap.add_argument('--df-max', type=float, default=None,
                    help='词库 df 阈值（按既有 df_ratio 列过滤：df_ratio ≤ df_max；缺省 = load_lexicon '
                         '默认 0.25 = 现状/主口径）。用于第一梯队 ⑤ / B″11-8 的 10%%/25%%/100%% 网格；'
                         '显式传入时报告新增 df_max 字段留痕')
    args = ap.parse_args()

    spec = SPECS[args.corpus]
    sim_adjust = args.sim_adjust if args.sim_adjust is not None else CORPORA[args.corpus].get('sim_adjust', 'none')
    years = args.years if args.years else list(spec['years'])
    outd = Path(args.out_dir) if args.out_dir else spec['out_dir']
    rp = Path(args.out_report) if args.out_report else spec['report']
    lex_path = Path(args.lexicon) if args.lexicon else CORPORA[args.corpus]['lexicon']
    if args.target is not None:
        target, target_src = int(args.target), 'CLI --target'
    else:
        target, target_src = registry_target(args.corpus, K_TARGET_FALLBACK[args.corpus])

    terms, lmeta = (load_lexicon(lex_path) if args.df_max is None
                    else load_lexicon(lex_path, df_max=args.df_max))
    A = build_automaton(terms)
    uni = load_refs()
    from scipy.cluster.hierarchy import linkage as hlink
    from sklearn.preprocessing import normalize
    rep = {'run_at': datetime.now(timezone.utc).isoformat(), 'linkage': args.linkage, 'target_peer': target,
           'rule': 'per_year_K_to_target_peer_density', 'lexicon': lmeta, 'years': {},
           'corpus': args.corpus,
           'corpus_path': rel(CORPORA[args.corpus]['path']),
           'target_source': target_src,
           'target_param': ('ma_k_target_teacher' if args.corpus.startswith(('teacher', 'ma_fulltext'))
                            else 'self_TARGET_constant'),
           'weighting': CORPORA[args.corpus].get('weighting', 'ew'),
           'idf_formula': ('idf = ln((1+N_y)/(1+df_y)) + 1（smooth；df_y = '
                           'product_lexicon_v2_teacher_df_by_year.csv 的 doc_count_{y}，N_y = n_docs_{y}）'
                           if CORPORA[args.corpus].get('weighting') == 'idf' else None),
           'out_dir': rel(outd)}
    # Phase 0.3（2026-10-27）：口径注记（append-only；不改任何数值/算法）
    # N-3：池化目标 44 与逐年实现是两个统计量；teacher 分支留档逐年参照目标/行业目标/实现/K。
    if args.corpus == 'teacher':
        rep['per_year_calibration'] = {
            'target_peer_pooled': 44,
            'target_definition': '公司层面申万二级同行业中位 peer（组内 N−1），2014–2022 九个文本年池化；'
                                 'K 校准把每年的团成员中位 peer 钉到 44',
            'text_years': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
            'network_years': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
            'target_median_peer_y': [32, 36, 36, 39, 43, 51, 49, 71, 54],
            'target_source': 's3_b1_tau_calibration_teacher.json → per_year_at_tau[].median_peer_at_tau'
                             '（单一 τ=0.0888212137 的阈值网络逐年中位 peer）',
            'sw2_median_peer_y': [26, 27, 30, 38, 37, 46, 52, 58, 62],
            'sw2_median_peer_source': 's3_b1_tau_calibration_teacher.json → sw2_target.per_year[].median_peer'
                                      '（逐年申万二级同行业中位 peer）',
            'realized_median_peer_y': [44, 46, 45, 44, 43, 46, 42, 45, 44],
            'K_y': [235, 238, 322, 291, 334, 407, 414, 434, 463],
            'K_mode': ['exact', 'closest', 'closest', 'exact', 'closest', 'closest', 'closest', 'closest', 'exact'],
            'note': 'target_median_peer_y 是 N-3 口径的逐年参照目标，不是 headline 44 的逐年分解；'
                    '44 是池化统计量。2016 年 K +35.29%（238→322，n_firms +5.92%）的机制未分解，'
                    '仅登记事实；未改目标、未重跑网络。',
        }
    # N-9：本报告 peer_median/sw2_lift 为同团成员法，与 grid/K_explore 的原文阈值法同名不同义。
    rep['metric_scope'] = {
        'peer_rule': 'same_cluster_members：peer_i = 同团成员数 = 团规模−1',
        'peer_fields': ['peer_median', 'peer_mean', 'peer_p25', 'peer_p90', 'isolated_share'],
        'sw2_lift_rule': '同团同申万二级无序对数 / 同团全部无序对数 / 同样本边际基线',
        'sw2_lift_fields': ['same_sw2_edge_share', 'sw2_lift'],
        'not_comparable_to': 's3_b3_ma_cluster_grid*.json / s3_ma_K_explore.json 的同名字段'
                             '（原文阈值法：sim > 团内最小相似度）',
        'phase0_3_note': '2026-10-27 Phase 0.3 口径注记；未改任何数值、未重跑网络',
    }
    if args.df_max is not None:
        rep['df_max'] = args.df_max            # 仅显式传参时留痕（默认路径不加键，避免与既有报告键差）
    if sim_adjust != 'none':
        # 仅非默认（median）变体加键，避免与既有 none 报告产生键差（同 --df-max 处置）
        rep['sim_adjust'] = sim_adjust
        rep['sim_adjust_formula'] = 'sim_adj(i,j) = sim(i,j) − (m_i+m_j)/2（HP16 PDF p.19；不裁剪负值）'
        rep['m_definition'] = 'm_i = median over all n−1 peers of the same year (zeros included, self excluded)'
        rep['distance_rule'] = 'D = 1 − sim_adj（sim_adj ∈ [−1,1] → D ∈ [0,2] 构造上非负；无需秩变换）'
        rep['decision_similarity_adjustment'] = (
            'company_median_adjust（HP16 PDF p.19）：sim_adj(i,j) = sim(i,j) − (m_i+m_j)/2；'
            'm_i = 全行含零相似对的行中位数（排除自身）；不裁剪负值；距离 = 1 − sim_adj（构造上非负）')
    t_all = time.time()
    for y in years:
        t0 = time.time()
        firms, X, stats = build_vectors(y, A, uni, corpus=args.corpus, df_max=args.df_max)
        C = (normalize(X) @ normalize(X).T).toarray().astype(np.float32)
        np.fill_diagonal(C, 1.0)
        adj = None
        if sim_adjust == 'median':
            C_adj, m = median_adjust(C)        # sim_adj = sim − (m_i+m_j)/2（不裁剪负值）
            adj = adjust_stats(C, C_adj, m)
            C = C_adj.astype(np.float32)
            np.fill_diagonal(C, 1.0)
        D = np.clip(1.0 - C, 0, 2); np.fill_diagonal(D, 0.0)
        l2s = [uni.loc[f, 'l2_code'] if (f in uni.index and isinstance(uni.loc[f, 'l2_code'], str)) else 'NA'
               for f in firms]
        u, l2_idx = np.unique(l2s, return_inverse=True)
        iu = np.triu_indices(len(firms), k=1)
        base = float((l2_idx[iu[0]] == l2_idx[iu[1]]).mean())
        Z = hlink(squareform(D, checks=False), method=args.linkage)
        K, mode = calibrate_K(Z, l2_idx, len(u), base, target)
        lab = fcluster(Z, K, criterion='maxclust')
        # 落盘聚类标签（供 B3.3/B4.x 复用，避免重复聚类）
        outd.mkdir(parents=True, exist_ok=True)
        with open(outd / f'{y}_clusters.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['firm_id', 'cluster_id'])
            for f_, c_ in zip(firms, lab):
                w.writerow([f_, int(c_)])
        st = cluster_stats(lab, l2_idx, len(u), base)
        st.update({'K': K, 'K_mode': mode, 'n_firms': len(firms), 'base_same_sw2_rate': round(base, 5),
                   'filter_stats': stats, 'sec': round(time.time() - t0, 1)})
        if adj:
            st['sim_adjust_stats'] = adj       # 仅 median 变体留痕
        rep['years'][str(y)] = st
        print(f"[{y}] n={len(firms):>4} K={K:>5}({mode}) 中位peer={st['peer_median']:.0f} "
              f"p90={st['peer_p90']:.0f} iso={st['isolated_share']:.3f} 团数={st['n_clusters']} "
              f"团规模中位={st['size_median']:.0f} lift={st['sw2_lift']} [{st['sec']}s]", flush=True)
    _w = CORPORA[args.corpus].get('weighting', 'ew')
    rep['decision'] = {
        'similarity_method_id': ('M-A（IDF 加权产品词向量 + 余弦；同语料对照变体）' if _w == 'idf'
                                 else 'M-A（等权 0/1 产品词向量 + 余弦）'),
        'vector_weighting': ('idf（smooth：ln((1+N_y)/(1+df_y)) + 1，df_y = 该年命中文档数）' if _w == 'idf'
                             else 'ew（等权 0/1；主口径）'),
        'cluster_linkage': args.linkage,
        'elbow_rule': f'per_year_target_peer_match（目标中位 peer={target}，K 逐年记录；'
                      '原拟肘点法实测不识别：K-lift 与 K-peer 曲线单调无拐点）',
        'competitor_set_rule': 'same_cluster_members（团内最小相似度阈值法实测退化，见 s3_ma_K_explore.json）',
        'min_product_terms': 10,
        'ma_sample_filter': '排除证监会 J（金融）/S（综合）门类 + 上市不足 2 年',
        'anti_drift_evidence': '固定 K=500 时逐年中位 peer 13→188、lift 10.84→2.08；逐年校准后见各年记录',
        'text_corpus': ('老师 MD&A 导出 teacher_mda_panel.csv（**同语料 IDF 加权对照变体，本次不升主口径**）'
                        if _w == 'idf' else
                        '老师 MD&A 导出 teacher_mda_panel.csv（**同语料 HP 式中位数调整变体，本次不升主口径**）'
                        if args.corpus == 'teacher_medadj' else
                        '自建年报全文面板 annual_fulltext_panel.parquet（**对表语料敏感性列，本次不升主口径**；'
                        'registry `ma_fulltext_text_corpus`，2026-10-19）' if args.corpus == 'ma_fulltext' else
                        '老师 MD&A 导出 teacher_mda_panel.csv（主口径 Phase B″3）' if args.corpus == 'teacher'
                        else '自建业务描述 business_texts.csv（文本稳健性列）'),
        'weighting_invariance': ('命中集合与样本筛选与权重无关：本变体逐年 n_nodes 与 filter_stats 应与'
                                 '主口径 teacher 逐值相等（只有相似度分布 / K_y / 边权可变化）'
                                 if _w == 'idf' else None),
    }
    rep['total_sec'] = round(time.time() - t_all, 1)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    ks = {y: rep['years'][y]['K'] for y in rep['years']}
    print('\nper-year K:', json.dumps(ks))
    print('per-year lift:', json.dumps({y: rep['years'][y]['sw2_lift'] for y in rep['years']}))
    print('->', rp)


if __name__ == '__main__':
    main()
