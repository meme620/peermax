"""申万二级分组独立核对 + L2 名称补全（同花顺 F10，2026-09-14）。

目的（B1 三轮验收的 R3 层）：
  官方 xls 只给代码不给名称，无法用成员样本以外的方式自证分组正确。
  同花顺 F10 公司页列出「所属申万行业：<一级> — <二级>」（SwClass2021 口径），
  是独立来源，可用于：
    ① 分组一致性核对：同一 L2 码下的股票，同花顺应给出同一二级名；
    ② 名称补全：为 l2_code 生成官方名称（众数 + 一致率），写入 sw2_l2_names.csv。

抽样：每个 L2 组按规模取 min(4, size) 只（大组多取用于组内一致性检验）。

产物：
  data/processed/sw2_l2_names.csv          l2_code, l1_name, l2_name, n_checked, agreement
  reports/data_gates/G0/sw2_ths_audit.json 抽样明细与一致率

用法：
  python scripts/data/audit_sw2_with_ths.py --per-group 1 --limit 20   # 小样调试
  python scripts/data/audit_sw2_with_ths.py                            # 全量抽样
"""
import argparse
import csv
import json
import re
import ssl
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LAT = ROOT / 'data/processed/sw2_industry_latest.csv'
CORPUS = ROOT / 'data/processed/mda_texts_v2.csv'
OUTN = ROOT / 'data/processed/sw2_l2_names.csv'
REPD = ROOT / 'reports/data_gates/G0'
UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36',
      'Referer': 'https://basic.10jqka.com.cn/'}
DELAY = 0.6


def _ctx():
    c = ssl.create_default_context()
    c.check_hostname = False
    c.verify_mode = ssl.CERT_NONE
    return c


def fetch_sw_industry(code6: str):
    """返回 (l1_name, l2_name) 或 None。"""
    url = f'https://basic.10jqka.com.cn/{code6}/company.html'
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=40, context=_ctx()) as r:
        html = r.read().decode('gbk', 'ignore')
    m = re.search(r'所属申万行业[：:]\s*</strong>\s*<span>([^<]{1,60})</span>', html)
    if not m:
        return None
    raw = m.group(1).strip()
    parts = [p.strip() for p in re.split(r'[—\-–]+', raw) if p.strip()]
    if not parts:
        return None
    l1 = parts[0]
    l2 = parts[1] if len(parts) > 1 else ''
    return l1, l2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-group', type=int, default=4, help='每组最多抽样数')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--groups', default='', help='只核对指定 L2 码（逗号分隔）')
    args = ap.parse_args()

    lat = pd.read_csv(LAT, dtype={'code6': str, 'l2_code': str, 'l1_code': str})
    lat['code6'] = lat['code6'].str.zfill(6)
    corpus = pd.read_csv(CORPUS, usecols=['code', 'status'])
    enrolled = {c[2:] for c in corpus.loc[corpus['status'] == 'ok', 'code'].unique()}
    lat = lat[lat['code6'].isin(enrolled) & lat['l2_code'].notna()].copy()

    if args.groups:
        want = {x.strip() for x in args.groups.split(',') if x.strip()}
        lat = lat[lat['l2_code'].isin(want)].copy()
    # 每组抽样（按 股票代码 排序保证确定性）
    sample = []
    for l2, g in lat.sort_values('code6').groupby('l2_code'):
        g = g[~g['firm_id'].isna()]
        sample.extend(g.head(args.per_group)['code6'].tolist())
    if args.limit:
        sample = sample[:args.limit]
    print(f'抽样 {len(sample)} 只（覆盖 {lat["l2_code"].nunique()} 个 L2 组）', flush=True)

    code2l2 = dict(zip(lat['code6'], lat['l2_code']))
    code2name = dict(zip(lat['code6'], lat['firm_id']))
    lat_cur = dict(zip(lat['code6'], lat['is_current_version']))
    records, fails = [], []
    for i, c in enumerate(sample, 1):
        got = None
        for attempt in (1, 2):
            try:
                got = fetch_sw_industry(c)
                break
            except Exception as e:
                err = f'{type(e).__name__}: {e}'
                time.sleep(0.8 * attempt)
        if got is None:
            fails.append({'code6': c, 'error': err if 'err' in dir() else 'no_match'})
            continue
        rec = {'code6': c, 'firm_id': code2name.get(c, ''), 'our_l2': code2l2.get(c, ''),
               'ths_l1': got[0], 'ths_l2': got[1]}
        cur = lat_cur.get(c, None)
        rec['is_current_version'] = bool(cur) if cur is not None else None
        records.append(rec)
        if i % 50 == 0:
            print(f'   {i}/{len(sample)} ok={len(records)} fail={len(fails)}', flush=True)
        time.sleep(DELAY)

    # 按我们的 L2 聚合：众数名称 + 一致率
    by_l2 = defaultdict(list)
    for r in records:
        by_l2[r['our_l2']].append(r)
    names, agree_rows = [], []
    for l2, rs in sorted(by_l2.items()):
        c_names = Counter(r['ths_l2'] for r in rs)
        c_l1 = Counter(r['ths_l1'] for r in rs)
        top_name, top_n = c_names.most_common(1)[0]
        names.append({'l2_code': l2, 'l1_name': c_l1.most_common(1)[0][0], 'l2_name': top_name,
                      'n_checked': len(rs), 'agreement': round(top_n / len(rs), 4)})
        agree_rows.append({'l2_code': l2, 'n': len(rs), 'agreement': top_n / len(rs),
                           'names': dict(c_names)})

    with open(OUTN, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['l2_code', 'l1_name', 'l2_name', 'n_checked', 'agreement'])
        w.writeheader()
        w.writerows(names)

    full_agree = sum(1 for a in agree_rows if a['agreement'] == 1.0 and a['n'] >= 2)
    multi = [a for a in agree_rows if a['n'] >= 2]
    # 逐股明细落盘（供定位分歧）
    with open(REPD / 'sw2_ths_audit_records.csv', 'w', newline='', encoding='utf-8') as f:
        w2 = csv.DictWriter(f, fieldnames=['code6', 'firm_id', 'our_l2', 'ths_l1', 'ths_l2', 'is_current_version'])
        w2.writeheader()
        for r in records:
            w2.writerow(r)
    # 分歧是否由老版代码解释：低一致率组内，不一致样本的老版代码占比
    legacy_disagree = legacy_total = 0
    for a in agree_rows:
        if a['n'] < 2 or a['agreement'] == 1.0:
            continue
        rs = by_l2[a['l2_code']]
        topname = Counter(r['ths_l2'] for r in rs).most_common(1)[0][0]
        for r in rs:
            if r['ths_l2'] != topname:
                legacy_total += 1
                if r['is_current_version'] is False:
                    legacy_disagree += 1
    rep = {
        'source': '同花顺 F10 公司页「所属申万行业」',
        'run_at': datetime.now(timezone.utc).isoformat(),
        'sampled': len(sample), 'fetched_ok': len(records), 'fetch_failed': len(fails),
        'l2_groups_covered': len(by_l2), 'l2_groups_total': int(lat['l2_code'].nunique()),
        'groups_with_multiple_checks': len(multi),
        'groups_fully_consistent': full_agree,
        'consistency_rate_strict': round(full_agree / len(multi), 4) if multi else None,
        'weighted_agreement': round(sum(a['agreement'] * a['n'] for a in agree_rows)
                                    / sum(a['n'] for a in agree_rows), 4) if agree_rows else None,
        'low_agreement_groups': [a for a in agree_rows if a['n'] >= 2 and a['agreement'] < 1.0],
        'disagreement_legacy_share': (round(legacy_disagree / legacy_total, 4) if legacy_total else None),
        'disagreement_n': legacy_total,
        'failures': fails[:30],
        'note': ('同花顺口径为 SwClass2021；若与我们 latest 映射不一致，优先怀疑老版代码残留'
                 '（官方 xls 中 26-50 只/年），明细见 low_agreement_groups。'),
    }
    REPD.mkdir(parents=True, exist_ok=True)
    (REPD / 'sw2_ths_audit.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in rep.items() if k not in ('failures', 'low_agreement_groups')},
                     ensure_ascii=False, indent=1))
    print(f'wrote {OUTN} and {REPD / "sw2_ths_audit.json"}')


if __name__ == '__main__':
    main()
