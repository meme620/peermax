"""B2 build_report 回填（2026-09-14）。

背景：network_mda_edges_build.py 在 `--out-dir` 为相对路径时 `fp.relative_to(ROOT)` 抛
ValueError，导致 9 个年份的边表/支持表已写出但 *_build_report.json 全部缺失。
本脚本从已产出的边表/支持表回填报告（不重算相似度），字段口径与主脚本一致：
  tau 取自 registry 冻结值；available_at = (文本年+1)-04-30；network_version = pm_mb_tfidf_v1_{y}
mass_retained_ratio 无法在不重算相似度的情况下恢复 → 置 null 并标注。
"""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
NETD = ROOT / 'data/processed/network_mb'
REG = ROOT / 'configs/parameter_registry.yaml'
TXTS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]


def main():
    reg = yaml.safe_load(REG.read_text(encoding='utf-8'))
    params = {p['name']: p['headline_value'] for p in reg['parameters']}
    tau = float(params['m_b_tau_fixed'])
    target = params['sw2_median_peer_target']
    print(f'registry tau={tau} target={target}')

    for y in TXTS:
        fp = NETD / f'{y}_similarity.csv'
        sp = NETD / f'{y}_support.csv'
        if not fp.exists():
            print(f'  {y}: missing edges, skip')
            continue
        n_edges = 0
        nodes, versions, avails = set(), set(), set()
        with open(fp, encoding='utf-8', newline='') as f:
            for r in csv.DictReader(f):
                n_edges += 1
                nodes.add(r['focal_id'])
                versions.add(r['network_version'])
                avails.add(r['available_at'])
        sup_status = {}
        with open(sp, encoding='utf-8', newline='') as f:
            for r in csv.DictReader(f):
                sup_status[r['network_support_status']] = sup_status.get(r['network_support_status'], 0) + 1
        meta = {
            'tau': tau, 'tau_mode': 'explicit', 'peer_target': target,
            'n_nodes': len(nodes), 'n_edges': n_edges,
            'mass_retained_ratio': None,
            'network_version': sorted(versions)[0] if len(versions) == 1 else sorted(versions),
            'available_at': sorted(avails)[0] if len(avails) == 1 else sorted(avails),
            'text_year': y,
            'run_at': datetime.now(timezone.utc).isoformat(),
            'output': f'data/processed/network_mb/{y}_similarity.csv',
            'support_status_counts': sup_status,
            'note': ('正式边表（B2）。由 network_report_backfill.py 回填元数据：'
                     'tau 取 registry m_b_tau_fixed（2026-09-14 冻结），'
                     'mass_retained_ratio 因未重算相似度置 null。'
                     '边选择阈值与权重归一化由 network_mda_edges_build.py --tau 执行。'),
        }
        (NETD / f'{y}_build_report.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                                     encoding='utf-8')
        print(f"  {y}: nodes={len(nodes)} edges={n_edges} support={sup_status} "
              f"ver={meta['network_version']} avail={meta['available_at']}")


if __name__ == '__main__':
    main()
