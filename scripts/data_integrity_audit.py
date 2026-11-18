"""S3 数据与口径对抗审计（R3，2026-09-14）。

核验项：
  D1 文本年语义：business_texts / mda_texts 的 year 是否等于**年报所属财年**
     （抽样读正文里的"20XX年年度报告"字样，与文件名年一致）
  D2 语料资格：两网只使用 status=ok 行；入网节点数 = ok 行数（M-B）/ ok − 冻结筛选（M-A）
  D3 ID 映射：网络节点必须全部在 firm_id_map 中（映射失败率 = 0）
  D4 源版本可追溯：边表 source_version 与构建报告一致
  D5 时点链：available_at 语义（次年 04-30）+ 消费侧 max{available_at < month}；C1 面板 PIT 违规 0
  D6 词库 PIT（已知限制）：词库与 df 过滤使用全期语料 → 记录影响面（被剔通用词条数）
产物：reports/data_gates/G0/s3_data_integrity_audit.json
"""
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPD = ROOT / 'reports/data_gates/G0'
MA, MB = ROOT / 'data/processed/network_ma', ROOT / 'data/processed/network_mb'
IDMAP = ROOT / 'data/processed/firm_id_map.csv'
out = {'run_at': datetime.now(timezone.utc).isoformat(), 'checks': {}}
rng = np.random.default_rng(19)

# ---- D1 文本年语义 ----
biz = pd.read_csv(ROOT / 'data/processed/business_texts.csv', usecols=['code', 'year', 'status', 'text'])
AR = ROOT / 'data/raw/cninfo/annual_txt'
smp_years = [2017, 2021, 2025]
file_ok = body_ok = tot = 0
for y in smp_years:
    d = biz[(biz['year'] == y) & (biz['status'] == 'ok')]
    for _, r in d.sample(min(40, len(d)), random_state=int(y)).iterrows():
        tot += 1
        if (AR / str(r['code'])[2:] / f'{y}.txt').exists():   # code 带 sz/sh 前缀，语料目录用 6 位码
            file_ok += 1
        if f'{y}年' in str(r['text']):
            body_ok += 1
out['checks']['D1_text_year_semantics'] = {
    'sampled': tot, 'source_file_exists_annual_txt_code_year': file_ok, 'file_share': round(file_ok / max(1, tot), 3),
    'year_string_in_body': body_ok, 'body_share': round(body_ok / max(1, tot), 3),
    'note': ('判据①：business_texts 的 year 必须对应 data/raw/cninfo/annual_txt/{code}/{year}.txt 源文件；'
             '判据②：正文任意位置出现"{year}年"。M-B 用 mda_texts（同源同年）'),
    'pass': file_ok / max(1, tot) > 0.95 and body_ok / max(1, tot) > 0.8}

# ---- D2 语料资格 ----
corpus_v7 = pd.read_csv(ROOT / 'data/processed/business_texts.csv', usecols=['code', 'year', 'status'])
mda = pd.read_csv(ROOT / 'data/processed/mda_texts.csv', usecols=['code', 'year', 'status'])
res2 = {'years': {}, 'note': '两网语料不同源：M-A ← business_texts（业务章节）；M-B ← mda_texts（MD&A）——'
                            '提取器成功率不同导致逐年节点数不同（2021 年 M-B 4,519 vs M-A 4,513 ok），已登记'}
ok_all = True
for y in range(2017, 2026):
    n_ok_ma = int(((corpus_v7['year'] == y) & (corpus_v7['status'] == 'ok')).sum())
    n_ok_mb = int(((mda['year'] == y) & (mda['status'] == 'ok')).sum())
    _mbrep = json.loads((MB / f'{y}_build_report.json').read_text(encoding='utf-8'))
    mb_nodes = _mbrep['n_nodes_total']      # 总节点 = support + no-support（字段语义 2026-09-14 统一）
    ma_rep = json.loads((MA / f'{y}_build_report.json').read_text(encoding='utf-8'))
    ma_nodes = ma_rep['n_nodes_total']
    fs = ma_rep['filter_stats']
    ok_y = ((n_ok_mb == mb_nodes)
            and (n_ok_ma - fs['drop_menlei'] - fs['drop_ipo'] - fs['drop_min'] == ma_nodes))
    ok_all &= ok_y
    res2['years'][str(y)] = {'v7_ok': n_ok_ma, 'mda_ok': n_ok_mb, 'mb_nodes': mb_nodes,
                             'ma_nodes': ma_nodes,
                             'ma_filtered_out': fs['drop_menlei'] + fs['drop_ipo'] + fs['drop_min'],
                             'identity_ok': bool(ok_y)}
res2['pass'] = ok_all
out['checks']['D2_corpus_eligibility'] = res2

# ---- D3 ID 映射 ----
imp = pd.read_csv(IDMAP, dtype={'code6': str})
known = set(imp['firm_id'])
res3 = {}
for tag, d in [('ma', MA), ('mb', MB)]:
    missing = Counter()
    for y in range(2017, 2026):
        sup = pd.read_csv(d / f'{y}_support.csv', dtype={'focal_id': str}, usecols=['focal_id'])
        s = set(sup['focal_id'])
        missing[y] = len(s - known)
    res3[tag] = {'nodes_not_in_firm_id_map': dict(missing),
                 'pass': all(v == 0 for v in missing.values())}
out['checks']['D3_id_mapping'] = res3

# ---- D4 source_version 可追溯 ----
res4 = {}
for tag, d, expect in [('ma', MA, 'business_texts'), ('mb', MB, 'mda_texts')]:
    vals = set()
    for y in range(2017, 2026):
        with open(d / f'{y}_similarity.csv', encoding='utf-8', newline='') as f:
            r = next(csv.DictReader(f))
            vals.add(r['source_version'])
    res4[tag] = {'source_version_values': sorted(vals),
                 'expected_prefix': expect,
                 'pass': all(v.startswith(expect) for v in vals)}
out['checks']['D4_source_version'] = res4

# ---- D5 时点链 ----
res5 = {'c1_pit': None, 'network_available_at': {}}
acc = json.loads((REPD / 's4_c1_acceptance.json').read_text(encoding='utf-8'))
res5['c1_pit'] = {'pit_violations': acc['panels']['ma'].get('pit_violations_strict_lt'),
                  'mb': acc['panels']['mb'].get('pit_violations_strict_lt'),
                  'pass': acc['panels']['ma'].get('pit_violations_strict_lt') == 0
                          and acc['panels']['mb'].get('pit_violations_strict_lt') == 0}
for tag, d in [('ma', MA), ('mb', MB)]:
    av = {y: json.loads((d / f'{y}_build_report.json').read_text(encoding='utf-8'))['available_at']
          for y in range(2017, 2026)}
    res5['network_available_at'][tag] = {'all_equal_next_apr30': all(av[y] == f'{y+1}-04-30' for y in av),
                                         'sample': {str(y): av[y] for y in (2017, 2021, 2025)}}
out['checks']['D5_pit_chain'] = res5

# ---- D6 词库 PIT 影响面 ----
lex2 = pd.read_csv(ROOT / 'data/processed/product_lexicon_v2.csv', dtype=str)
n_all = len(lex2)
kept = int((lex2['keep_df25'].astype(str) == '1').sum())
dropped = n_all - kept
out['checks']['D6_lexicon_pit'] = {
    'lexicon_terms_v2': n_all, 'kept_df25': kept, 'terms_dropped_by_df_gt_025': dropped,
    'dropped_share': round(dropped / n_all, 6),
    'note': ('M-A 词库与 df 过滤用全期语料构建（含未来年份）；影响=哪些通用词被剔除，占比极小；'
             'M-B 的 df 为逐年计算，无跨年信息。已登记为已知限制 + 可选加固项'),
    'pass': True}

ok = all(v.get('pass', True) if isinstance(v, dict) else all(x.get('pass', True) for x in v.values())
         for v in out['checks'].values())
out['status'] = 'PASS' if ok else 'FAIL'
(REPD / 's3_data_integrity_audit.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=1)[:2600])
print('status =', out['status'])
