"""C1 验收：S4 v4 面板三轮检查（W1 bundle，P1.1-01 / P1.2-01 / P1.2-04 面板侧；2026-11-17）。

R1 结构/合同：
  - 唯一键 (focal_id, month, corpus_id, panel_version) / (focal_id, month, network_use_case, network_version) 重复 0
  - 版本选择 PIT：available_at < month（严格）；无版本月份标 `no_available_network_version`
  - 主列语义（D-08 A）：`support_status_* == complete` → 主列非空；partial / no_covered / not_applicable → 主列空；
    `missing_mass_* == 0` ↔ complete；`coverage_ratio_*` 与 `support_status_*` 一一对应
  - `data_status` 互斥且加总 = `rows_support`（报告侧）
  - 缺失不填 0：无支持行 / partial 行的主列必须为空
  - 权重和 support_mass ∈ [1-1e-4, 1+1e-4]（有边者）
R2 独立复算：
  - 抽样 `data_status=='ok'` 行，用原始边表 + source-month（complete_month）+ clean_v2（R/n_valid_days）
    **独立**重算 pm/pr（不除），与 `PeerMAXraw`/`PeerReturn` 逐值比对；断言 coverage_ratio==1、missing_mass==0
R3 对抗：
  - `limit_status_unknown` 只进 `n_max_excluded_limit_unknown` 与 `missing_mass_max`，不得出现在 complete 行
  - 交叉口径：同一 (focal, month) 的 OwnMAXraw 在不同 v4 面板必须完全一致（同一 source-month 底层）
  - 覆盖率分布：报告 p10/中位/mean

产物：reports/data_gates/G0/s4_c1_acceptance_{corpus}_v4.json
用法（单面板模式）：
  python scripts/peermax_panel_acceptance.py \
    --panel data/processed/monthly_pricing_panel_ma_teacher_v4.csv --use-case ma \
    --net-dir data/processed/network_ma_teacher \
    --ref-panel data/processed/monthly_pricing_panel_ma_v4.csv \
    --out reports/data_gates/G0/s4_c1_acceptance_ma_teacher_v4.json
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from pricing_lib import assert_m1_inputs_sha, sha256_file  # noqa: E402

REPD = ROOT / 'reports/data_gates/G0'
PANELS = {'ma': ROOT / 'data/processed/monthly_pricing_panel_ma.csv',
          'mb': ROOT / 'data/processed/monthly_pricing_panel_mb.csv'}
NETS = {'ma': ROOT / 'data/processed/network_ma', 'mb': ROOT / 'data/processed/network_mb'}
ARGMAX = ROOT / 'data/processed/source_month_argmax_panel.csv'
RETURNS = ROOT / 'data/processed/monthly_returns_clean_v2.csv'
EXCLUDE_QUAL = 'limit_status_unknown'
D03_STRICT = True
RNG = np.random.default_rng(11)


def ym_add(month, k):
    y, m = int(month[:4]), int(month[5:])
    m += k
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    return f'{y:04d}-{m:02d}'


def _num(v):
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else x


def _rel(path):
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', default=None,
                    help='单面板模式：待验收面板路径（本 bundle 一律使用单面板模式）')
    ap.add_argument('--use-case', default='ma', help='单面板模式的 use-case 标签（缺省 ma）')
    ap.add_argument('--net-dir', default=None, help='单面板模式的网络目录（缺省按 use-case 取 NETS）')
    ap.add_argument('--ref-panel', default=None,
                    help='单面板模式下 R3 OwnMAXraw 交叉核对的参照 v4 面板')
    ap.add_argument('--out', default='reports/data_gates/G0/s4_c1_acceptance_v4.json')
    args = ap.parse_args(argv)

    if not args.panel:
        raise SystemExit('[STOP] W1 acceptance 只允许单面板模式（--panel）；历史 ma+mb 双面板模式已停用')
    panels = {args.use_case: ROOT / args.panel}
    nets = {args.use_case: (ROOT / args.net_dir) if args.net_dir else NETS[args.use_case]}
    # 07 §8.8：报告登记 script/sha 与 input_anchors（9 个 M1 输入锚 + 被验收面板/参照面板）。
    m1_ledger = assert_m1_inputs_sha()
    input_anchors = [
        {'input': e['input'], 'file': e['file'], 'sha256': e['sha256'],
         'anchor_sha256': e['anchor_sha256'], 'anchor_source': e['anchor_source']}
        for e in m1_ledger
    ]
    input_anchors.append({'input': 'acceptance_panel',
                          'file': _rel(panels[args.use_case]),
                          'sha256': sha256_file(panels[args.use_case])})
    if args.ref_panel:
        input_anchors.append({'input': 'acceptance_ref_panel',
                              'file': _rel(ROOT / args.ref_panel),
                              'sha256': sha256_file(ROOT / args.ref_panel)})
    out = {'run_at': datetime.now(timezone.utc).isoformat(),
           'script': _rel(Path(__file__)),
           'script_sha256': sha256_file(Path(__file__)),
           'input_anchors': input_anchors,
           'panels': {}, 'mode': 'single:%s' % args.panel}
    # 载入底层数据（供 R2）
    maxraw, qual, lim, complete = defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict)
    with open(ARGMAX, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            m = r['month']
            sid = r['source_id']
            mv = _num(r.get('MAXraw'))
            if mv is not None:
                maxraw[m][sid] = mv
            q = (r.get('qualification_status') or '').strip()
            if q:
                qual[m][sid] = q
            lv = _num(r.get('limit_hit_at_argmax'))
            if lv is not None:
                lim[m][sid] = int(lv)
            complete[m][sid] = int(float(r['complete_month']))
    rmap, nvmap = defaultdict(dict), {}
    with open(RETURNS, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            m = r['month']
            fid = r['firm_id']
            rv = _num(r.get('R'))
            if rv is not None:
                rmap[m][fid] = rv
            nvmap[(fid, m)] = int(float(r['n_valid_days']))

    for uc, path in panels.items():
        d = pd.read_csv(path, dtype={'focal_id': str, 'month': str,
                                     'network_version': str, 'available_at': str})
        res = {'rows': int(len(d))}
        # R1
        key1 = ['focal_id', 'month', 'corpus_id', 'panel_version']
        key2 = ['focal_id', 'month', 'network_use_case', 'network_version']
        res['dup_keys'] = int(d.duplicated(key1).sum())
        res['dup_keys_nv'] = int(d.duplicated(key2).sum())
        has_ver = d['network_version'].notna() & (d['network_version'] != '')
        res['rows_with_version'] = int(has_ver.sum())
        res['rows_no_version'] = int((~has_ver).sum())
        pit_violation = 0
        for av, mo in zip(d.loc[has_ver, 'available_at'], d.loc[has_ver, 'month']):
            if not (av < mo):
                pit_violation += 1
        res['pit_violations_strict_lt'] = pit_violation
        nov = d[~has_ver]
        res['no_version_rows_ok'] = bool(
            (nov['network_support_status'] == 'no_available_network_version').all()
            and nov['data_status'].eq('no_available_network_version').all()
            and nov['PeerMAXraw'].isna().all()
            and nov['PeerReturn'].isna().all())
        # 主列语义：complete ↔ 非空；partial/no_covered/not_applicable ↔ 空
        ss_max = d['support_status_max'].fillna('')
        ss_ret = d['support_status_ret'].fillna('')
        miss_max = d['missing_mass_max'].fillna(0)
        miss_ret = d['missing_mass_ret'].fillna(0)
        zero = pd.Series(0.0, index=d.index)
        bad = 0
        bad += int(((ss_max == 'complete') & d['PeerMAXraw'].isna()).sum())
        bad += int(((ss_max != 'complete') & (ss_max != '') & d['PeerMAXraw'].notna()).sum())
        bad += int(((ss_ret == 'complete') & d['PeerReturn'].isna()).sum())
        bad += int(((ss_ret != 'complete') & (ss_ret != '') & d['PeerReturn'].notna()).sum())
        bad += int(((ss_max == 'complete') & (miss_max != 0)).sum())
        bad += int(((ss_ret == 'complete') & (miss_ret != 0)).sum())
        res['main_column_semantics_violations'] = bad
        # 覆盖率与 support_status 一一对应（只查有网络版本的行；no-version 行 status=not_applicable）
        dv = d[has_ver]
        cr_max = pd.to_numeric(dv['coverage_ratio_max'], errors='coerce')
        cr_ret = pd.to_numeric(dv['coverage_ratio_ret'], errors='coerce')
        sup = pd.to_numeric(dv['support_mass'], errors='coerce')
        cov_viol = 0
        for st_col, cr in (('support_status_max', cr_max), ('support_status_ret', cr_ret)):
            st = dv[st_col].fillna('')
            cov_viol += int(((st == 'complete') & ~((sup > 0) & (cr == 1))).sum())
            cov_viol += int(((st == 'partial') & ~((sup > 0) & (cr > 0) & (cr < 1))).sum())
            cov_viol += int(((st == 'no_covered_source_max') & ~((sup > 0) & (cr == 0))).sum())
            cov_viol += int(((st == 'no_covered_source_ret') & ~((sup > 0) & (cr == 0))).sum())
            cov_viol += int(((st == 'not_applicable') & ~(sup == 0)).sum())
        res['coverage_status_violations'] = cov_viol
        sm = pd.to_numeric(d.loc[has_ver, 'support_mass'], errors='coerce')
        res['support_mass_out_of_tolerance'] = int(((sm - 1).abs() > 1e-4).sum())
        cov = pd.to_numeric(d.loc[has_ver, 'coverage_ratio_max'], errors='coerce')
        res['coverage_max'] = {'p10': round(float(cov.quantile(.1)), 4),
                               'median': round(float(cov.median()), 4),
                               'mean': round(float(cov.mean()), 4),
                               'share_below_0.5': round(float((cov < 0.5).mean()), 4)}
        covr = pd.to_numeric(d.loc[has_ver, 'coverage_ratio_ret'], errors='coerce')
        res['coverage_ret'] = {'p10': round(float(covr.quantile(.1)), 4),
                               'median': round(float(covr.median()), 4)}
        # R3-a：limit_status_unknown 不得出现在 complete 行
        res['limit_unknown_in_complete_rows'] = int(
            ((d['n_max_excluded_limit_unknown'].fillna(0) > 0) & (ss_max == 'complete')).sum())
        res['rows_with_limit_excluded'] = int((d['n_max_excluded_limit_unknown'].fillna(0) > 0).sum())

        # R2 独立复算（data_status=='ok'；未归一化 pm/pr）
        cand = d[has_ver & (d['data_status'] == 'ok')].head(8, ).copy()
        cache = {}
        mismatches = []
        for _, row in cand.iterrows():
            ty = int(str(row['network_version']))
            key = (uc, ty)
            if key not in cache:
                edges = defaultdict(list)
                with open(nets[uc] / f'{ty}_similarity.csv', encoding='utf-8', newline='') as f:
                    for r in csv.DictReader(f):
                        edges[r['focal_id']].append((r['source_id'], float(r['weight'])))
                cache[key] = edges
            lst = cache[key].get(row['focal_id'], [])
            mm, qq = maxraw[row['month']], qual[row['month']]
            rrm = rmap[row['month']]
            cm = complete[row['month']]
            w_max = w_ret = 0.0
            pm = pr = 0.0
            miss_max = miss_ret = 0.0
            for src, w in lst:
                q = qq.get(src)
                if q is None:
                    miss_max += w
                elif q == EXCLUDE_QUAL:
                    miss_max += w
                elif cm.get(src, 0) != 1:
                    miss_max += w
                    if src in mm:
                        pm += w * mm[src]
                elif src not in mm:
                    miss_max += w
                else:
                    pm += w * mm[src]
                    w_max += w
                if src not in rrm:
                    miss_ret += w
                elif nvmap.get((src, row['month']), 0) == 0:
                    miss_ret += w
                elif D03_STRICT and cm.get(src, 0) != 1:
                    miss_ret += w
                else:
                    pr += w * rrm[src]
                    w_ret += w
            def close(a, b):
                if a is None or b != b:
                    return False
                return abs(a - b) < 1e-8 or abs(a - b) / max(1e-9, abs(a)) < 1e-6
            ok = (close(pm, row['PeerMAXraw']) and close(pr, row['PeerReturn'])
                  and abs(w_max - float(row['covered_mass_max'])) < 1e-8
                  and abs(w_ret - float(row['covered_mass_ret'])) < 1e-8
                  and float(row['coverage_ratio_max']) == 1.0
                  and float(row['coverage_ratio_ret']) == 1.0
                  and float(row['missing_mass_max']) == 0.0
                  and float(row['missing_mass_ret']) == 0.0)
            if not ok:
                mismatches.append({'focal': row['focal_id'], 'month': row['month'],
                                   'recomputed_pm': pm, 'panel_pm': row['PeerMAXraw'],
                                   'recomputed_pr': pr, 'panel_pr': row['PeerReturn']})
        res['R2_recompute'] = {'n_checked': int(len(cand)), 'mismatches': len(mismatches),
                               'detail': mismatches[:3],
                               'tol': 'abs<1e-8 或 rel<1e-6（面板 8 位小数落盘）',
                               'pass': len(mismatches) == 0}
        out['panels'][uc] = res

    # R3-b 交叉口径：同一 (focal, month) 的 OwnMAXraw 必须一致
    if args.ref_panel:
        a_path = panels[args.use_case]
        b_path = ROOT / args.ref_panel
        a = pd.read_csv(a_path, dtype={'focal_id': str, 'month': str},
                        usecols=['focal_id', 'month', 'OwnMAXraw'])
        b = pd.read_csv(b_path, dtype={'focal_id': str, 'month': str},
                        usecols=['focal_id', 'month', 'OwnMAXraw'])
        j = a.merge(b, on=['focal_id', 'month'], suffixes=('_a', '_b'))
        ja = pd.to_numeric(j['OwnMAXraw_a'], errors='coerce')
        jb = pd.to_numeric(j['OwnMAXraw_b'], errors='coerce')
        both = j[ja.notna() & jb.notna()]
        diff = (pd.to_numeric(both['OwnMAXraw_a'], errors='coerce')
                - pd.to_numeric(both['OwnMAXraw_b'], errors='coerce')).abs()
        out['R3_own_max_cross_panel'] = {
            'panel_a': a_path.resolve().relative_to(ROOT).as_posix(),
            'panel_b': b_path.resolve().relative_to(ROOT).as_posix(),
            'common_rows': int(len(both)),
            'max_abs_diff': float(diff.max()) if len(both) else None,
            'pass': bool(len(both) > 0 and diff.max() < 1e-8)}
    else:
        out['R3_own_max_cross_panel'] = {'panel_a': None, 'panel_b': None,
                                         'common_rows': 0, 'max_abs_diff': None,
                                         'pass': False,
                                         'note': '未提供 --ref-panel；W1 验收必须提供 v4 参照面板'}

    ok = all(v.get('dup_keys') == 0 and v.get('dup_keys_nv') == 0
             and v.get('pit_violations_strict_lt') == 0
             and v.get('main_column_semantics_violations') == 0
             and v.get('coverage_status_violations') == 0
             and v.get('no_version_rows_ok')
             and v.get('limit_unknown_in_complete_rows') == 0
             and v['R2_recompute']['pass']
             and v.get('support_mass_out_of_tolerance') == 0
             for v in out['panels'].values()) and out['R3_own_max_cross_panel']['pass']
    out['status'] = 'PASS' if ok else 'FAIL'
    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=1)[:3000])
    print('status =', out['status'], '->', outp)


if __name__ == '__main__':
    main()
