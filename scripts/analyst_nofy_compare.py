# -*- coding: utf-8 -*-
"""analyst-network 9K 对照表生成（2026-09-01）：主口径 fy12(v3) vs nofy(v4) 网络层+结果层并排。
读取双方 JSON 产物，输出 reports/data_gates/G0/analyst_nofy_compare.{json,md}。
判读预案（execution plan Step 9·9K）：nofy 版 NPM EW 多空仍 ≥ Long 锚点 1.28 → "超锚点"
不依赖 FY 口径；回落到锚点之下 → FY 严宽是显式边界条件，报告须改口。
"""
import json
from pathlib import Path
from analyst_meta import attach_meta

A = Path('data/processed/analyst')
G = Path('reports/data_gates/G0')
LONG_ANCHOR_PCT, LONG_ANCHOR_T = 1.28, 5.20


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    v3d = load(A / 'analyst_network_diagnostics.json')['summary']
    v4d = load(A / 'step9k_nofy_diagnostics.json')['summary']
    rows = []

    def net(name, k3, k4, fmt='{}'):
        rows.append({'dim': name, 'fy12_v3': fmt.format(v3d.get(k3)), 'nofy_v4': fmt.format(v4d.get(k4))})

    rows.append({'dim': '无向边总数(241月)', 'fy12_v3': v3d.get('edges_total_undirected'), 'nofy_v4': v4d.get('edges_total_undirected')})
    rows.append({'dim': '边数月均', 'fy12_v3': v3d.get('edges_monthly_mean'), 'nofy_v4': v4d.get('edges_monthly_mean')})
    rows.append({'dim': 'degree p50(月均)', 'fy12_v3': v3d.get('degree_p50_mean'), 'nofy_v4': v4d.get('degree_p50_mean')})
    rows.append({'dim': 'common p50(月均)', 'fy12_v3': v3d.get('common_count_p50_mean'), 'nofy_v4': v4d.get('common_count_p50_mean')})
    rows.append({'dim': '弱边占比(common=1)', 'fy12_v3': v3d.get('weak_edge_ratio_mean'), 'nofy_v4': v4d.get('weak_edge_ratio_mean')})

    def res(name, fy12, nofy):
        mm3, ts3 = fy12['mean_monthly_ret_pct'], fy12['t_stats']
        mm4, ts4 = nofy['mean_monthly_ret_pct'], nofy['t_stats']
        return {'dim': name,
                'fy12_v3': f"{mm3['ew_ls']}%/t{ts3['ew_ls_iid']}",
                'nofy_v4': f"{mm4['ew_ls']}%/t{ts4['ew_ls_iid']}"}

    npm3 = load(A / 'npm_replication.json'); npm4 = load(A / 'npm_replication_nofy.json')
    npm3ln = load(A / 'npm_replication_ln.json'); npm4ln = load(A / 'npm_replication_ln_nofy.json')
    cf3 = load(A / 'cf_mom_replication.json'); cf4 = load(A / 'cf_mom_replication_nofy.json')
    al3 = load(A / 'cf_mom_alpha.json'); al4 = load(A / 'cf_mom_alpha_nofy.json')
    rows.append(res('NPM EW 多空(count)', npm3, npm4))
    rows.append(res('NPM EW 多空(ln)', npm3ln, npm4ln))
    rows.append(res('CF EW 多空(count)', cf3, cf4))
    rows.append({'dim': 'NPM VW 多空', 'fy12_v3': f"{npm3['mean_monthly_ret_pct']['vw_ls']}%/t{npm3['t_stats']['vw_ls_iid']}",
                 'nofy_v4': f"{npm4['mean_monthly_ret_pct']['vw_ls']}%/t{npm4['t_stats']['vw_ls_iid']}"})
    rows.append({'dim': 'CF ln+FF5 α EW', 'fy12_v3': f"{al3['ew_ls']['ff5']['alpha']['beta_pct']}%/t{al3['ew_ls']['ff5']['alpha']['t']}",
                 'nofy_v4': f"{al4['ew_ls']['ff5']['alpha']['beta_pct']}%/t{al4['ew_ls']['ff5']['alpha']['t']}"})

    npm4_ew = npm4['mean_monthly_ret_pct']['ew_ls']
    verdict = ('nofy 版 NPM EW 多空 %.2f%%/月 ≥ Long 锚点 1.28 → "超锚点"不依赖 FY 覆盖判定' % npm4_ew) if npm4_ew >= LONG_ANCHOR_PCT \
        else ('nofy 版 NPM EW 多空 %.2f%%/月 < Long 锚点 1.28 → FY 严宽是显式边界条件，报告须改口' % npm4_ew)
    out = {'note': '9K 对照变体（不进主 claim、不改冻结参数）；nofy=Long 口径=本数据源蒋口径等值（四预测字段全空=0 行）',
           'long_anchor': {'pct': LONG_ANCHOR_PCT, 't': LONG_ANCHOR_T},
           'table': rows, 'verdict': verdict,
           'network_versions': {'fy12_v3': 'coverage_win12_fy12_v3', 'nofy_v4': 'coverage_win12_nofy_v4'}}
    attach_meta(out, __file__, 'coverage_win12_fy12_v3 + coverage_win12_nofy_v4')
    (G / 'analyst_nofy_compare.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    md = ['# 9K 对照表：FY1/FY2 主口径(v3) vs 不限 FY Long 口径(v4)\n',
          f"- Long 锚点：NPM EW 多空 {LONG_ANCHOR_PCT}%/月（t={LONG_ANCHOR_T}，CSMAR 2004-2022）",
          f"- 判读：**{verdict}**\n",
          '| 维度 | fy12_v3（主口径） | nofy_v4（Long=蒋等值） |', '|---|---|---|']
    for r in rows:
        md.append(f"| {r['dim']} | {r['fy12_v3']} | {r['nofy_v4']} |")
    (G / 'analyst_nofy_compare.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(verdict)


if __name__ == '__main__':
    main()
