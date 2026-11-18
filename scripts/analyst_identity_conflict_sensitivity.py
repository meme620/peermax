"""S3A 身份冲突敏感性（版本自适应重跑，2026-08-14 首建；2026-09-01 修订）。

修复对抗审查 P0-1：旧 sensitivity_identity_conflict.json 基于 v1 语义（8/12 14:15 生成，
早于 v2 覆盖表 21:44），且无构造脚本。本脚本读当前交付覆盖表 + identity_conflict_ids.json，
按 step67 同逻辑重建逐月对比。
2026-09-01 修订（review R8）：覆盖表已是 v3（241 月）而产物/标签停留在 v2/97 月，
且原脚本无版本过滤（读什么版本就算什么版本，再硬写 v2 标签）。现从覆盖表自身解析
coverage_window_version，输出文件名与标签跟随实际版本。

用法: python scripts/analyst_identity_conflict_sensitivity.py
输出: data/processed/analyst/sensitivity_identity_conflict_<version>.json
"""
import json
import re
from pathlib import Path
from collections import defaultdict

COV = 'data/processed/analyst/firm_analyst_coverage.csv'
IDS = Path('data/processed/analyst/identity_conflict_ids.json')
OUT_TMPL = 'data/processed/analyst/sensitivity_identity_conflict_{ver}.json'


def main():
    conflict_ids = set(json.loads(IDS.read_text(encoding='utf-8'))['conflict_ids'])
    print(f'conflict ids: {len(conflict_ids)}', flush=True)

    # 按 cutoff 聚合 (firm, analyst)，避免重复读全表
    by_cutoff = defaultdict(list)  # cutoff -> list of (firm_id, analyst_id)
    with open(COV, encoding='utf-8') as f:
        header = f.readline().strip().split(',')
        # 找列位置（防表头顺序漂移）
        idx = {name: i for i, name in enumerate(header)}
        vers = set()
        for line in f:
            parts = line.rstrip('\n').split(',')
            vers.add(parts[idx['coverage_window_version']])
            by_cutoff[parts[idx['network_cutoff']]].append((parts[idx['firm_id']], parts[idx['analyst_id']]))
    assert len(vers) == 1, f'覆盖表混入多个版本: {vers}'
    net_ver = vers.pop()
    m_ver = re.search(r'v(\d+)$', net_ver)
    ver_short = f'v{m_ver.group(1)}' if m_ver else net_ver
    OUT = Path(OUT_TMPL.format(ver=ver_short))
    print(f'coverage version: {net_ver} -> OUT {OUT.name}', flush=True)

    months = []
    full_tot, excl_tot = 0, 0
    for cutoff in sorted(by_cutoff):
        pairs = by_cutoff[cutoff]
        # 主口径：analyst -> set(firms)
        af_full = defaultdict(set)
        for fid, aid in pairs:
            af_full[aid].add(fid)
        # 剔除冲突 ID
        af_excl = {aid: firms for aid, firms in af_full.items() if aid not in conflict_ids}

        def edge_count(af):
            # edges = 唯一公司对；mass = Σ common（共同分析师链接总和，权重分母）
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

        n_full, d_full, m_full = edge_count(af_full)
        n_excl, d_excl, m_excl = edge_count(af_excl)
        full_tot += n_full
        excl_tot += n_excl
        months.append({'cutoff': cutoff, 'edges_full': n_full, 'edges_excl': n_excl,
                       'delta_abs': n_excl - n_full,
                       'delta_pct': round((n_excl - n_full) / n_full, 6),
                       'mass_full': m_full, 'mass_excl': m_excl,
                       'mass_delta_pct': round((m_excl - m_full) / m_full, 6),
                       'degree_p50_full': d_full, 'degree_p50_excl': d_excl})
        print(f'{cutoff}: full={n_full} excl={n_excl} delta_pct={months[-1]["delta_pct"]:.4f}', flush=True)

    mean_full = full_tot / len(months)
    mean_excl = excl_tot / len(months)
    rep = {
        'note': f'版本自适应口径（{net_ver}）：读当前交付覆盖表 + identity_conflict_ids.json，与主口径同基线逐月对比。'
                'edges=唯一公司对（与 step67 口径一致）；mass=Σ common（共同分析师链接总和，权重分母）。'
                '2026-09-01 修订（review R8）：v2 时代产物（97 月）已过期，本文件按实际交付版本重建。',
        'network_version': net_ver,
        'conflict_ids': sorted(conflict_ids),
        'summary': {
            'months': len(months),
            'edges_monthly_mean_full': round(mean_full, 1),
            'edges_monthly_mean_excl': round(mean_excl, 1),
            'edges_total_full': full_tot,
            'edges_total_excl': excl_tot,
            'delta_edges_pct_mean_of_means': round((mean_excl - mean_full) / mean_full, 6),
            'delta_edges_pct_mean_of_monthly': round(
                sum(m['delta_pct'] for m in months) / len(months), 6),
            'mass_monthly_mean_full': round(sum(m['mass_full'] for m in months) / len(months), 1),
            'mass_monthly_mean_excl': round(sum(m['mass_excl'] for m in months) / len(months), 1),
            'delta_mass_pct_mean_of_means': round(
                (sum(m['mass_excl'] for m in months) - sum(m['mass_full'] for m in months))
                / sum(m['mass_full'] for m in months), 6),
            'degree_p50_mean_full': round(sum(m['degree_p50_full'] for m in months) / len(months), 2),
            'degree_p50_mean_excl': round(sum(m['degree_p50_excl'] for m in months) / len(months), 2),
        },
        'months': months,
    }
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(rep['summary'], ensure_ascii=False, indent=2), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
