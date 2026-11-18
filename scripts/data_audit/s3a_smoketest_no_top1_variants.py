"""S3A smoketest：no_top1 剔除口径对比（2026-08-14）。

对抗审查 A5：sens_specs 的 no_top1 用跨月 distinct 覆盖公司数 >300（v2 下 6 人），标签却是 exclude_top1pct。
本脚本对比三种口径的逐月边数与 degree：
  full            : 不剔除
  gt300_cross     : 跨月 distinct 覆盖公司数 > 300（与 sens_specs no_top1 完全同口径；2026-08-14 统一）
  top1pct         : 每月按分析师覆盖公司数排序，剔除覆盖最广的前 1%（动态分位，~126 人量级）

用法: python scripts/data_audit/s3a_smoketest_no_top1_variants.py
输出: data/processed/analyst/smoketest_no_top1_variants.json
"""
import json
from pathlib import Path
from collections import defaultdict

COV = 'data/processed/analyst/firm_analyst_coverage.csv'
OUT = Path('data/processed/analyst/smoketest_no_top1_variants.json')


def main():
    by_cutoff = defaultdict(list)
    cross_firms = defaultdict(set)   # 跨月 distinct 覆盖公司数（sens_specs 同口径）
    with open(COV, encoding='utf-8') as f:
        header = f.readline().strip().split(',')
        idx = {name: i for i, name in enumerate(header)}
        for line in f:
            parts = line.rstrip('\n').split(',')
            fid, aid = parts[idx['firm_id']], parts[idx['analyst_id']]
            by_cutoff[parts[idx['network_cutoff']]].append((fid, aid))
            cross_firms[aid].add(fid)
    gt300_ids = {aid for aid, fs in cross_firms.items() if len(fs) > 300}
    print(f'gt300_cross（跨月 >300 家）: {len(gt300_ids)} 人', flush=True)

    def edge_stats(af):
        # edges = 唯一公司对（与 step67 口径一致）；mass = Σ common
        pairs = set()
        mass = 0
        for aid, firms in af.items():
            fl = sorted(firms)
            for a in range(len(fl)):
                for b in range(a + 1, len(fl)):
                    pairs.add((fl[a], fl[b]))
                    mass += 1
        deg = defaultdict(int)
        for i, j in pairs:
            deg[i] += 1
            deg[j] += 1
        d = sorted(deg.values())
        p50 = d[int((len(d) - 1) * 0.50)] if d else None
        return len(pairs), p50, mass

    months = []
    for cutoff in sorted(by_cutoff):
        pairs = by_cutoff[cutoff]
        af = defaultdict(set)
        for fid, aid in pairs:
            af[aid].add(fid)
        n_edges_full, d_full, m_full = edge_stats(af)
        # gt300_cross（跨月口径，与 sens_specs no_top1 一致）
        af_gt300 = {a: fs for a, fs in af.items() if a not in gt300_ids}
        n_gt300, d_gt300, m_gt300 = edge_stats(af_gt300)
        # top1% 动态：按覆盖公司数取前 1%
        ranked = sorted(af.items(), key=lambda kv: len(kv[1]), reverse=True)
        n_excl = max(1, round(0.01 * len(ranked)))
        excl_ids = {aid for aid, _ in ranked[:n_excl]}
        af_t1 = {a: fs for a, fs in af.items() if a not in excl_ids}
        n_t1, d_t1, m_t1 = edge_stats(af_t1)
        months.append({'cutoff': cutoff,
                       'analysts': len(af), 'top1pct_excluded': len(excl_ids),
                       'edges_full': n_edges_full, 'edges_gt300_cross': n_gt300, 'edges_top1pct': n_t1,
                       'mass_full': m_full, 'mass_gt300_cross': m_gt300, 'mass_top1pct': m_t1,
                       'degree_p50_full': d_full, 'degree_p50_gt300_cross': d_gt300, 'degree_p50_top1pct': d_t1})
        print(f'{cutoff}: full={n_edges_full} gt300_cross={n_gt300} top1pct={n_t1} (excl {len(excl_ids)})', flush=True)

    def mean(key):
        return round(sum(m[key] for m in months) / len(months), 1)

    summary = {
        'edges_monthly_mean_full': mean('edges_full'),
        'edges_monthly_mean_gt300_cross': mean('edges_gt300_cross'),
        'edges_monthly_mean_top1pct': mean('edges_top1pct'),
        'delta_pct_gt300_cross_vs_full': round((mean('edges_gt300_cross') - mean('edges_full')) / mean('edges_full'), 6),
        'delta_pct_top1pct_vs_full': round((mean('edges_top1pct') - mean('edges_full')) / mean('edges_full'), 6),
        'mass_monthly_mean_full': mean('mass_full'),
        'mass_monthly_mean_gt300_cross': mean('mass_gt300_cross'),
        'mass_monthly_mean_top1pct': mean('mass_top1pct'),
        'delta_mass_pct_top1pct_vs_full': round((mean('mass_top1pct') - mean('mass_full')) / mean('mass_full'), 6),
        'degree_p50_mean_full': mean('degree_p50_full'),
        'degree_p50_mean_gt300_cross': mean('degree_p50_gt300_cross'),
        'degree_p50_mean_top1pct': mean('degree_p50_top1pct'),
        'top1pct_excluded_mean': round(sum(m['top1pct_excluded'] for m in months) / len(months), 1),
    }
    OUT.write_text(json.dumps({'note': '三种剔除口径对比（v2 覆盖表）。top1pct=每月按覆盖广度前 1% 动态剔除；gt300_cross=跨月 distinct 覆盖公司数 >300（与 sens_specs no_top1 同口径，v2 下 6 人，2026-08-14 统一）。',
                               'summary': summary, 'months': months},
                              ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()