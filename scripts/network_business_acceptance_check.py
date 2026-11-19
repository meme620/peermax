"""S3-B3.4：M-A 边表机械化验收（2026-09-14）。

与 B2 验收同构，差异：
  - 期望节点数取自 build_report（M-A 含 J/S 与上市不足 2 年筛选，不能直接对语料 ok 数）
  - 追加聚类一致性检查：support+nns == n_nodes；每个 focal 的全部 source 必须与其同团
（聚类标签文件在 {clusters}/{year}_clusters.csv）

v2（2026-09-28，Phase B″5）：加 `--dir --clusters --years --ver-prefix --out` 参数化
（与 B″2 同风格）。**判据一条未改**：默认值仍指历史自建链，重跑 self 结果与原
`reports/data_gates/G0/s3_ma_acceptance.json` 逐字段一致（run_at 除外）。
九项：① 唯一键重复 0；② 自环 0；③ 权重行和偏差 0（≤1e-4）；④ 权重复算相对误差 <1e-5；
⑤ available_at 唯一且 = 次年 04-30；⑥ network_version 唯一且 = 期望版本串；
⑦ 边表 focal 集合 = support 表 support 节点集、零权重行 0、no-network-support 行无边；
⑧ 聚类一致性 = 1.000 且无节点缺标签；⑨ 节点数 = build_report.n_nodes。

v3（2026-09-29，Phase B″5 判据修订·口径统一）：`median_peer` 由「仅含有边 focal」
（`edges.groupby(focal_id).size().median()`）改为**当年网络全部公司（含零对手）**口径，
与 `network_business_cluster_freeze.cluster_stats` 的 `peers = sizes[inv] − 1`（覆盖全部入网公司，singleton
记 0）**逐值一致**，也与老师方法说明原文一致：
「We calculate the median number of peers **for all firms** … the median number of text-based
peers **across the pooled sample** equals N_SW2」（`ref/teacher-materials/Methodology Description.pdf` §2）。
实现：以当年 `{y}_support.csv` 的 focal 集合为分母（= 入网全部公司 = build_report.n_nodes），
peer 数 = 该 focal 的边数（无支持/零对手 = 0）。新增字段：`median_peer`（统一口径）、
`median_peer_edges_only`（旧口径，描述性）、`isolated_share`、`peers_universe`；
若给 `--freeze-report`，另报 `freeze_peer_median` 与 `peer_median_matches_freeze`（逐值对账，
进入 pass 条件）。**九项机械判据的合取式一条未改**（只新增一个可选对账项）。
旧产物（`s3_ma_acceptance.json` / `s3_ma_acceptance_teacher.json`）不被覆盖：v3 一律显式 `--out`。

产物：--out（缺省 reports/data_gates/G0/s3_ma_acceptance.json）
用法：
  python scripts/network_business_acceptance_check.py --out reports/data_gates/G0/s3_ma_acceptance_self_v3.json
  python scripts/network_business_acceptance_check.py --dir data/processed/network_ma_teacher \
      --clusters data/processed/network_ma_teacher/clusters --years 2014 2015 2016 2017 2018 2019 2020 2021 2022 \
      --ver-prefix pm_ma_ewcos_teacher_v1 --freeze-report reports/data_gates/G0/s3_b3_ma_param_freeze_teacher.json \
      --out reports/data_gates/G0/s3_ma_acceptance_teacher_v3.json
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPD = ROOT / 'reports/data_gates/G0'
TXTS = list(range(2017, 2026))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='data/processed/network_ma')
    ap.add_argument('--clusters', default='data/processed/network_ma/clusters')
    ap.add_argument('--years', type=int, nargs='*', default=TXTS, help='文本年（缺省 2017-2025）')
    ap.add_argument('--ver-prefix', default='pm_ma_ewcos_v1', help='期望版本串前缀（缺省自建链）')
    ap.add_argument('--freeze-report', default=None,
                    help='K 冻结报告（可选）：对账 median_peer 与 cluster_stats.peer_median 逐值一致')
    ap.add_argument('--out', default='reports/data_gates/G0/s3_ma_acceptance.json')
    args = ap.parse_args()
    netd = ROOT / args.dir
    clud = ROOT / args.clusters
    outp = ROOT / args.out
    freeze = None
    if args.freeze_report:
        fpz = ROOT / args.freeze_report
        freeze = json.loads(fpz.read_text(encoding='utf-8'))['years'] if fpz.exists() else None

    per_year, overall_ok = [], True
    for y in args.years:
        fp, sp, rp = netd / f'{y}_similarity.csv', netd / f'{y}_support.csv', netd / f'{y}_build_report.json'
        if not (fp.exists() and sp.exists() and rp.exists()):
            per_year.append({'text_year': y, 'pass': False, 'why': 'missing files'})
            overall_ok = False
            continue
        rep = json.loads(rp.read_text(encoding='utf-8'))
        edges = pd.read_csv(fp, dtype={'focal_id': str, 'source_id': str})
        sup = pd.read_csv(sp, dtype={'focal_id': str})
        clu = pd.read_csv(clud / f'{y}_clusters.csv', dtype={'firm_id': str})
        c2k = dict(zip(clu['firm_id'], clu['cluster_id']))

        dup = int(edges.duplicated(['network_version', 'focal_id', 'source_id']).sum())
        self_loop = int((edges['focal_id'] == edges['source_id']).sum())
        g = edges.groupby('focal_id')['weight'].sum()
        bad_sum = int(((g - 1).abs() > 1e-4).sum())
        focals = set(edges['focal_id'].drop_duplicates().sample(
            min(200, edges['focal_id'].nunique()), random_state=42))
        sub = edges[edges['focal_id'].isin(focals)]
        rr = sub.groupby('focal_id').apply(
            lambda d: (((d['similarity'] / d['similarity'].sum()) - d['weight']).abs()
                       / d['weight'].abs().clip(lower=1e-12)).max(), include_groups=False)
        max_rel = float(rr.max()) if len(rr) else None
        avail = sorted(set(edges['available_at'].astype(str)))
        versions = sorted(set(edges['network_version'].astype(str)))
        expect_avail, expect_ver = f'{y + 1}-04-30', f'{args.ver_prefix}_{y}'
        sup_nodes = set(sup.loc[sup['network_support_status'] == 'support', 'focal_id'])
        ns_nodes = set(sup.loc[sup['network_support_status'] == 'no-network-support', 'focal_id'])
        node_total = len(sup_nodes) + len(ns_nodes)
        # 聚类一致性：每条边的 focal 与 source 必须同团
        ek = edges[['focal_id', 'source_id']].drop_duplicates()
        same_cluster = sum(1 for a, b in zip(ek['focal_id'], ek['source_id'])
                           if c2k.get(a) == c2k.get(b))
        cluster_consistency = same_cluster / max(1, len(ek))
        zero_w = int((edges['weight'] == 0).sum())
        n_missing_label = int(sup['focal_id'].map(lambda f: f not in c2k).sum())
        # no-network-support 行不得带任何边（“权重留空不填 0”的强形式）
        ns_with_edges = len(set(edges['focal_id']) & ns_nodes)
        # --- median peer：v3 统一口径 = 当年网络全部公司（含零对手）---
        # 分母 = support 表 focal 集合（= 入网全部公司 = build_report.n_nodes）；peer 数 = 该 focal 边数，
        # 无支持/零对手 = 0。与 network_business_cluster_freeze.cluster_stats 的 peers = sizes[inv] − 1 逐值可比。
        edge_cnt = edges.groupby('focal_id').size()
        peers = sup['focal_id'].map(lambda f: int(edge_cnt.get(f, 0))).astype('int64')
        median_peer = float(peers.median())
        median_peer_edges_only = (float(edge_cnt.median()) if len(edge_cnt) else None)
        isolated_share = round(float((peers == 0).mean()), 4)
        fz = (freeze or {}).get(str(y))
        freeze_med = float(fz['peer_median']) if fz and fz.get('peer_median') is not None else None
        matches_freeze = (None if freeze_med is None else bool(median_peer == freeze_med))

        ok = (dup == 0 and self_loop == 0 and bad_sum == 0 and avail == [expect_avail]
              and versions == [expect_ver] and set(edges['focal_id']) == sup_nodes
              and zero_w == 0 and cluster_consistency == 1.0 and n_missing_label == 0
              and ns_with_edges == 0
              and node_total == rep['n_nodes'] and (max_rel is None or max_rel < 1e-5)
              and (matches_freeze is None or matches_freeze))
        overall_ok &= ok
        per_year.append({
            'text_year': y, 'network_year': y + 1, 'network_version': versions[0],
            'available_at': avail[0], 'nodes_total': node_total,
            'nodes_support': len(sup_nodes), 'nodes_no_support': len(ns_nodes),
            'n_edges': int(len(edges)), 'dup_keys': dup, 'self_loops': self_loop,
            'bad_row_sum': bad_sum, 'max_weight_recompute_rel_err': max_rel,
            'zero_weight_rows': zero_w, 'ns_nodes_with_edges': ns_with_edges,
            'cluster_consistency': round(cluster_consistency, 6),
            'median_peer': median_peer,
            'median_peer_edges_only': median_peer_edges_only,
            'peers_universe': int(len(peers)), 'isolated_share': isolated_share,
            'freeze_peer_median': freeze_med, 'peer_median_matches_freeze': matches_freeze,
            'K_clusters': rep.get('k_clusters'), 'pass': bool(ok)})
        print(f"  {y}->{y+1}: nodes={node_total}({len(sup_nodes)}+{len(ns_nodes)}ns) edges={len(edges)} "
              f"med_peer_all={median_peer:.0f}(edge-only={median_peer_edges_only:.0f}) "
              f"K={rep.get('k_clusters')} dup={dup} "
              f"loop={self_loop} badsum={bad_sum} clu_ok={cluster_consistency:.3f} pass={ok}")

    out = {'check': 'B3.4 M-A 边表验收', 'script_version': 'v3 (2026-09-29)',
           'median_peer_definition': ('当年网络全部公司（含零对手）：分母 = {y}_support.csv 的 focal 集合'
                                      '（= 入网公司 = build_report.n_nodes），peer 数 = 该 focal 边数，'
                                      '无支持 = 0；与 network_business_cluster_freeze.cluster_stats 的 peers = sizes[inv]-1 '
                                      '逐值可比；依据 = ref/teacher-materials/Methodology Description.pdf §2'
                                      '「median number of peers for all firms / across the pooled sample」'),
           'median_peer_edges_only_definition': '旧口径（描述性字段，不参与 pass）：edges.groupby(focal_id).size().median()',
           'dir': args.dir, 'clusters': args.clusters,
           'years': list(args.years), 'ver_prefix': args.ver_prefix,
           'freeze_report': args.freeze_report,
           'run_at': datetime.now(timezone.utc).isoformat(), 'per_year': per_year,
           'status': 'PASS' if overall_ok else 'FAIL'}
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('status =', out['status'], '->', outp)


if __name__ == '__main__':
    main()
