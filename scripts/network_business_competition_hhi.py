"""network M-A Step：竞争强度 PC = −HHI（设计规格 required output，2026-09-14）。

设计真源（execution_plan network M-A 链条）：
    PC_{i,y} = −HHI(竞争对手集合内销售收入)
  口径：竞争对手集合 = M-A 聚类同团成员（含 focal 自身，标准 HHI 定义）；
        销售收入取 **y−1 年**（operating_revenue_annual.pkl，索引为年报期末日）；
        营收缺失的公司不计入 HHI 分母，但报告覆盖率；覆盖率 = 0 → PC missing（不填 0）。
  同时输出 `hhi_excl_self`（剔除自身的变体）以便审计与敏感性。

v2（2026-09-28，Phase B″2/B″4）：加 `--corpus {self,teacher}` / `--net-dir` / `--cluster-dir` /
`--years` / `--out-report` 参数化。**公式与口径一字未改**：默认 `--corpus self` 重跑任一历史年份
与既有 `network_ma/{y}_pc.csv` 逐字节一致。teacher 侧落 `network_ma_teacher/{y}_pc.csv` +
`network_business_pc_report_teacher.json`（不覆盖历史）。

产物：{net_dir}/{text_year}_pc.csv + reports/data_gates/G0/network_business_pc_report{,_teacher}.json
用法：
  python scripts/network_business_competition_hhi.py
  python scripts/network_business_competition_hhi.py --corpus teacher

v3（2026-09-28，Phase B″4 施工期加固）：报告里 `net_dir` / `cluster_dir` 改用 `rel()`
（ROOT 外/临时目录回退绝对路径）；**PC 口径与产物列不变**。

v4（2026-10-05，Phase B″10）：`SPECS` / CLI 加 `teacher_idf`（同语料 IDF 加权对照变体，
`ver_prefix=pm_ma_idf_teacher_v1`、网络目录 `data/processed/network_ma_teacher_idf/`、
报告 `network_business_pc_report_teacher_idf.json`）。**PC 公式、营收口径与列不变**；`self` / `teacher` 不变。

v5（2026-10-19，全文语料对表轮）：`SPECS` / CLI 加 `ma_fulltext`（**中文法 × 年报全文**，对表语料敏感性列、
不升主口径；registry `ma_fulltext_text_corpus`）——`ver_prefix=pm_ma_ewcos_fulltext_v1`、网络目录
`data/processed/network_ma_fulltext/`、报告 `network_business_pc_report_ma_fulltext.json`、文本年 FY2017–FY2022。
**PC 公式（PC = −HHI(同团成员 y−1 年销售收入)，含 focal 自身）、营收口径（`operating_revenue_annual.pkl`
的 y−1 年，缺营收不计入分母、覆盖 0 记 missing）与产物列一字未改**；其余语料默认路径不变。
"""
import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / 'data/raw/drive/operating_revenue_annual.pkl'
REPD = ROOT / 'reports/data_gates/G0'
SPECS = {
    'self': {'net_dir': ROOT / 'data/processed/network_ma',
             'years': list(range(2017, 2026)),
             'ver_prefix': 'pm_ma_ewcos_v1',
             'report': REPD / 'network_business_pc_report.json'},
    'teacher': {'net_dir': ROOT / 'data/processed/network_ma_teacher',
                'years': list(range(2014, 2023)),
                'ver_prefix': 'pm_ma_ewcos_teacher_v1',
                'report': REPD / 'network_business_pc_report_teacher.json'},
    # Phase B″10：同语料 IDF 加权对照变体（不升主口径）；PC 公式与营收口径一字未改
    'teacher_idf': {'net_dir': ROOT / 'data/processed/network_ma_teacher_idf',
                    'years': list(range(2014, 2023)),
                    'ver_prefix': 'pm_ma_idf_teacher_v1',
                    'report': REPD / 'network_business_pc_report_teacher_idf.json'},
    # 2026-10-19 全文语料对表轮（registry `ma_fulltext_text_corpus`，不升主口径）：
    # 文本年 FY2017–FY2022（= 对表可用年：PC_{i,y} 需 y−1 年营收，营收面板 FY2016–2024）。
    # **PC 公式、营收口径与列一字未改**（PC = −HHI(同团成员 y−1 年销售收入)，含 focal 自身）。
    'ma_fulltext': {'net_dir': ROOT / 'data/processed/network_ma_fulltext',
                    'years': list(range(2017, 2023)),
                    'ver_prefix': 'pm_ma_ewcos_fulltext_v1',
                    'report': REPD / 'network_business_pc_report_ma_fulltext.json'},
}
COLS = ['focal_id', 'text_year', 'network_version', 'available_at', 'revenue_year',
        'n_competitors', 'n_rev_covered', 'revenue_coverage', 'hhi', 'pc', 'hhi_excl_self',
        'network_support_status']


def rel(p):
    """相对 ROOT 的 POSIX 路径；ROOT 外或相对路径回退为绝对路径。"""
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def load_rev(path=REV):
    d = pd.read_pickle(path)
    d.index = [ts.strftime('%Y') if hasattr(ts, 'strftime') else str(ts)[:4] for ts in d.index]
    rev = {}
    for y, row in d.iterrows():
        rev[y] = {c.split('.')[0]: float(v) for c, v in row.items() if v == v and v > 0}
    return rev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', choices=['self', 'teacher', 'teacher_idf', 'ma_fulltext'], default='self',
                    help='文本语料（决定网络目录/版本前缀/年份/报告路径）；teacher_idf=同语料 IDF 对照变体；'
                         'ma_fulltext=自建**年报全文**面板（对表语料敏感性列，2026-10-19，不升主口径）')
    ap.add_argument('--net-dir', default=None, help='网络目录（缺省按语料；须含 {y}_clusters.csv 与 build_report）')
    ap.add_argument('--cluster-dir', default=None, help='聚类目录（缺省 = net_dir/clusters，回退 net_dir）')
    ap.add_argument('--years', type=int, nargs='*', default=None, help='文本年（缺省按语料）')
    ap.add_argument('--out-report', default=None, help='报告路径（缺省按语料）')
    ap.add_argument('--revenue', default=str(REV), help='营收面板 pkl')
    args = ap.parse_args()

    spec = SPECS[args.corpus]
    MA = Path(args.net_dir) if args.net_dir else spec['net_dir']
    CLU = Path(args.cluster_dir) if args.cluster_dir else MA / 'clusters'
    if not CLU.exists():
        CLU = MA
    years = args.years if args.years else list(spec['years'])
    rp = Path(args.out_report) if args.out_report else spec['report']
    nver_prefix = spec['ver_prefix']

    rev = load_rev(args.revenue)
    print('营收年:', sorted(rev), '| 2016 覆盖:', len(rev.get('2016', {})))
    rep = {'run_at': datetime.now(timezone.utc).isoformat(),
           'rule': 'PC = -HHI(同团成员 y-1 年销售收入份额)；含 focal 自身；缺营收不计入分母',
           'corpus': args.corpus,
           'net_dir': rel(MA),
           'cluster_dir': rel(CLU),
           'years': {}}
    for y in years:
        clu = pd.read_csv(CLU / f'{y}_clusters.csv', dtype={'firm_id': str})
        byc = defaultdict(list)
        for f, c in zip(clu['firm_id'], clu['cluster_id']):
            byc[c].append(f)
        ry = str(y - 1)
        rmap = rev.get(ry, {})
        ver = json.loads((MA / f'{y}_build_report.json').read_text(encoding='utf-8'))
        avail = ver['available_at']
        nver = f'{nver_prefix}_{y}'
        rows, stats = [], {'n': 0, 'cov0': 0, 'cov_lt1': 0, 'pc': [], 'cov': [],
                           'singleton': 0, 'no_comp_rev': 0, 'focal_rev_missing': 0}
        for c, members in byc.items():
            vals = {f: rmap.get(f[2:]) for f in members}
            covered = {f: v for f, v in vals.items() if v is not None}
            tot = sum(covered.values())
            for f in members:
                stats['n'] += 1
                if len(members) < 2 or tot <= 0 or f not in covered:
                    reason = 'singleton_cluster' if len(members) < 2 else (
                        'no_competitor_revenue' if tot <= 0 else 'focal_revenue_missing')
                    rows.append([f, y, nver, avail, ry, len(members) - 1, len(covered) - (1 if f in covered else 0),
                                 round(len(covered) / max(1, len(members)), 4) if members else '',
                                 '', '', '', 'no-network-support' if len(members) < 2 else 'support'])
                    if len(members) < 2:
                        stats['cov0'] += 1
                        stats['singleton'] += 1
                    elif tot <= 0:
                        stats['no_comp_rev'] += 1
                    else:
                        stats['focal_rev_missing'] += 1
                    continue
                shares = np.array([v / tot for v in covered.values()])
                hhi = float((shares ** 2).sum())
                excl = {k: v for k, v in covered.items() if k != f}
                if excl:
                    t2 = sum(excl.values())
                    hhi_x = float(((np.array(list(excl.values())) / t2) ** 2).sum()) if t2 > 0 else None
                else:
                    hhi_x = None
                cov = len(covered) / len(members)
                stats['pc'].append(-hhi); stats['cov'].append(cov)
                if cov < 1:
                    stats['cov_lt1'] += 1
                rows.append([f, y, nver, avail, ry, len(members) - 1, len(covered) - 1,
                             round(cov, 4), round(hhi, 8), round(-hhi, 8),
                             '' if hhi_x is None else round(-hhi_x, 8), 'support'])
        with open(MA / f'{y}_pc.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(COLS)
            w.writerows(rows)
        rep['years'][str(y)] = {
            'text_year': y, 'revenue_year': ry, 'n_firms': stats['n'],
            'singleton_no_support': stats['cov0'],
            'no_competitor_revenue_focals': stats['no_comp_rev'],
            'focal_revenue_missing_focals': stats['focal_rev_missing'],
            'revenue_coverage_lt1': stats['cov_lt1'],
            'pc_unique_values': len({round(float(x), 8) for x in stats['pc']}) if stats['pc'] else 0,
            'pc_median': round(float(np.median(stats['pc'])), 6) if stats['pc'] else None,
            'pc_p10': round(float(np.quantile(stats['pc'], .1)), 6) if stats['pc'] else None,
            'pc_p90': round(float(np.quantile(stats['pc'], .9)), 6) if stats['pc'] else None,
            'revenue_coverage_median': round(float(np.median(stats['cov'])), 4) if stats['cov'] else None,
        }
        print(f"[{y}] 营收年 {ry} n={stats['n']} 无支持={stats['cov0']} 覆盖<1={stats['cov_lt1']} "
              f"PC 中位={rep['years'][str(y)]['pc_median']} 覆盖中位={rep['years'][str(y)]['revenue_coverage_median']}",
              flush=True)
    # 已登记限制（Phase B″4）：营收输入 operating_revenue_annual.pkl 仅覆盖 FY2016–2024；
    # PC_{i,y} 需要 y−1 年营收 → 文本年 2014–2016（营收年 2013–2015）全缺（no_competitor_revenue）。
    gap_years = [y for y in years if not rev.get(str(y - 1))]
    # Phase 0.3（2026-10-27）：N-11 PC 识别层级注记（append-only；不改任何数值/算法）
    rep['pc_level_note'] = {
        'level': 'cluster-year',
        'definition': 'PC = −HHI(同团成员 y−1 年销售收入)；同一团全部成员共享同一 PC 值，因此 PC 是团-年常数',
        'per_year_unique_pc': {'2017': 198, '2018': 217, '2019': 273,
                               '2020': 271, '2021': 235, '2022': 258},
        'total_unique_cluster_year': 1452,
        'diagnostic_2019': '3,354 家公司 / 407 团 / 134 单点团；有 PC 的 3,218 行只有 273 个不同 PC 值'
                           '（= 407 − 134）',
        'full_pc_panel_rows_2017_2022': 19849,
        'pc_correspondence_merged_firm_year': 18934,
        'inference_recommendation': '横截面按团聚类（或团层 FM/团层等权），并叠加年块推断（团×年二维）；'
                                    '不得按 firm-month 独立处理，不得写「月度截面无重复」',
        'note': '2026-10-27 Phase 0.3 注记；未改 PC 数值、未重跑 PC',
    }
    rep['known_limitations'] = {
        'revenue_panel_years': sorted(rev),
        'pc_unavailable_years': gap_years,
        'rule': 'PC_{i,y} 需 y−1 年营收；y−1 不在营收面板年份内 → 该年全部 focal 标 '
                'no_competitor_revenue，pc 留空不填 0',
        'available_years': [y for y in years if y not in gap_years],
        'note': f'主口径 PC 仅 {len(years) - len(gap_years)}/{len(years)} 年可得'
                f'（文本年 {[y for y in years if y not in gap_years]}）；'
                '可选修复路径＝以 MRQ/累计营收重建 FY2013–2015 年度营收面板（本步不执行，登记待评估）。',
    }
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print('->', rp)


if __name__ == '__main__':
    main()
