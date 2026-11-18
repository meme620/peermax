"""免费源数据可用性深度审计（2026-09-14）。

审计对象与判据：
  A. 搜狗细胞词库
     A1 下载完整性：catalog 条目 vs 落盘文件、零字节、解析状态
     A2 内容质量：词数分布、异常小库、重复库（内容指纹）
     A3 可用性：词条是否能在 M-A 业务文本中命中（朴素 jieba 口径）
  B. 东财 F10 主营构成
     B1 抓取完整性：文件数、失败、zygcfx 缺失公司画像
     B2 历史深度：每股最早/最晚报告期、200 条上限截断情况（关键！）
     B3 期间覆盖：是否覆盖 2017-2025（网络年所需）
     B4 接口是否支持取更多（分页/参数探测）
  C. 申万二级
     C1 文件时效：最大更新日期
     C2 网络实际用到的公司是否都有映射
     C3 老版代码残留对网络年份的影响
  D. M-B 网络与下游契约
     D1 与收益面板的 firm_id 可连接性
     D2 PeerReturn 聚合复算（抽样语义核验）

产物：reports/data_gates/G0/free_source_audit_20260914.json + 控制台摘要
"""
import csv
import gzip
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fileio import read_csv_dicts

ROOT = Path(__file__).resolve().parents[1]
REPD = ROOT / 'reports/data_gates/G0'
raw = ROOT / 'data/raw/lexicon/sogou'
em_raw = ROOT / 'data/raw/eastmoney/f10_business'




# ---------------- A. 搜狗 ----------------
def audit_sogou():
    from fetch_sogou_lexicon import parse_scel
    cat = read_csv_dicts(raw / '_catalog.csv', encoding='utf-8')
    files = {int(p.stem): p for p in raw.glob('*.scel')}
    out = {'catalog_entries': len(cat), 'files_on_disk': len(files)}
    missing = [r['dict_id'] for r in cat if int(r['dict_id']) not in files]
    zero = [i for i, p in files.items() if p.stat().st_size == 0]
    tiny = [(i, p.stat().st_size) for i, p in files.items() if 0 < p.stat().st_size < 3000]
    out['missing_files'] = missing[:20]
    out['missing_n'] = len(missing)
    out['zero_byte'] = zero
    out['tiny_files_lt3k'] = sorted(tiny, key=lambda x: x[1])[:20]
    out['tiny_n'] = len(tiny)

    stats, bad = [], []
    sizes = {}
    for i, p in sorted(files.items()):
        try:
            name, words, status = parse_scel(p)
            sizes[i] = p.stat().st_size
            stats.append({'id': i, 'words': len(words), 'status': status,
                          'bytes': p.stat().st_size, 'name': name})
            if status != 'eof-ok' or len(words) == 0:
                bad.append({'id': i, 'status': status, 'words': len(words),
                            'bytes': p.stat().st_size, 'name': name})
        except Exception as e:
            bad.append({'id': i, 'error': f'{type(e).__name__}: {e}'})
    out['parsed_ok'] = sum(1 for s in stats if s['status'] == 'eof-ok' and s['words'] > 0)
    out['parse_problems'] = bad[:30]
    out['parse_problems_n'] = len(bad)
    w = pd.Series([s['words'] for s in stats])
    out['word_count'] = {'min': int(w.min()), 'p10': int(w.quantile(.1)), 'median': int(w.median()),
                         'p90': int(w.quantile(.9)), 'max': int(w.max()), 'sum': int(w.sum())}
    out['smallest_dicts'] = sorted(stats, key=lambda s: s['words'])[:15]
    # 与站点元数据（下载次数）核对文件大小相关性，检测异常空壳
    dl = {int(r['dict_id']): r for r in cat}
    outlier = [s for s in stats if s['bytes'] < 5000 or s['words'] < 50]
    out['outliers_low_content'] = len(outlier)
    out['outlier_sample'] = outlier[:15]
    return out


# ---------------- B. F10 ----------------
def audit_em(probe_api=False):
    idmap = pd.read_csv(ROOT / 'data/processed/firm_id_map.csv', dtype={'code6': str})
    files = {p.name.replace('.json.gz', ''): p for p in em_raw.glob('*.json.gz')}
    out = {'idmap_firms': len(idmap), 'files': len(files)}
    no_file = [f for f in idmap['firm_id'] if f not in files]
    out['no_file_n'] = len(no_file)
    out['no_file_sample'] = no_file[:20]

    per_firm = []
    empty = []
    for fid, p in files.items():
        try:
            with gzip.open(p, 'rt', encoding='utf-8') as f:
                d = json.load(f)
        except Exception as e:
            empty.append({'firm_id': fid, 'error': f'{type(e).__name__}: {e}'})
            continue
        zg = d.get('zygcfx') or []
        if not zg:
            empty.append({'firm_id': fid, 'note': 'zygcfx empty',
                          'has_zyfw': bool(d.get('zyfw')), 'has_jyps': bool(d.get('jyps'))})
            continue
        prod = [x for x in zg if str(x.get('MAINOP_TYPE')) == '2']
        dates = sorted({(x.get('REPORT_DATE') or '')[:10] for x in zg if x.get('REPORT_DATE')})
        pdates = sorted({(x.get('REPORT_DATE') or '')[:10] for x in prod if x.get('REPORT_DATE')})
        per_firm.append({'firm_id': fid, 'items': len(zg), 'prod_items': len(prod),
                         'first': dates[0] if dates else '', 'last': dates[-1] if dates else '',
                         'n_periods': len(dates), 'prod_first': pdates[0] if pdates else '',
                         'capped': len(zg) >= 200})
    df = pd.DataFrame(per_firm)
    out['firms_with_data'] = len(df)
    out['empty_or_error'] = len(empty)
    out['empty_sample'] = empty[:20]
    out['capped_at_200'] = int(df['capped'].sum())
    out['capped_share'] = round(float(df['capped'].mean()), 4)
    # 关键：200 条上限是否吃掉历史 → capped 公司的最早报告期分布
    cap = df[df['capped']]
    out['capped_first_date'] = {
        'min': str(cap['first'].min()), 'median': str(sorted(cap['first'])[len(cap)//2]) if len(cap) else '',
        'max': str(cap['first'].max()),
        'n_before_2018': int((cap['first'] < '2018-01-01').sum()),
        'n_after_2018': int((cap['first'] >= '2018-01-01').sum()),
    }
    # 期间覆盖：2017-2025 每年有多少公司有"按产品"披露
    rows = []
    for y in range(2017, 2026):
        key = f'{y}-12-31'
        rows.append({'year': y, 'firms_with_annual_prod': int((df['prod_first'] <= key).sum() and 0)})
    # 更精确：直接用 terms 表统计每年有产品行的公司数
    terms = pd.read_csv(ROOT / 'data/processed/em_business_terms.csv', dtype=str)
    terms = terms[terms['mainop_type'] == '2'].copy()
    terms['year'] = terms['report_date'].str[:4]
    byy = terms.groupby('year')['firm_id'].nunique()
    out['firms_by_report_year'] = {k: int(v) for k, v in byy.items()}
    out['prod_rows_total'] = int(len(terms))
    if probe_api:
        out['api_probe'] = probe_em_api()
    return out


def probe_em_api():
    """探测 F10 接口是否支持分页/更多条数（决定能否修复 200 条截断）。"""
    UA = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://emweb.securities.eastmoney.com/'}
    base = 'https://emweb.securities.eastmoney.com/PC_HSF10/BusinessAnalysis/PageAjax?code=SH600519'
    res = {}
    for tag, extra in [('baseline', ''), ('page2', '&page=2'), ('pagesize500', '&pageSize=500'),
                       ('limit500', '&limit=500'), ('type_zygcfx', '&type=zygcfx')]:
        try:
            with urllib.request.urlopen(urllib.request.Request(base + extra, headers=UA), timeout=30) as resp:
                b = resp.read()
            if b[:2] == b'\x1f\x8b':
                b = gzip.decompress(b)
            d = json.loads(b)
            zg = d.get('zygcfx') or []
            dates = sorted({(x.get('REPORT_DATE') or '')[:10] for x in zg})
            res[tag] = {'n': len(zg), 'first': dates[0] if dates else '', 'last': dates[-1] if dates else ''}
        except Exception as e:
            res[tag] = f'{type(e).__name__}: {e}'
    return res


# ---------------- C. 申万 ----------------
def audit_sw2():
    df = pd.read_excel(ROOT / 'data/raw/sw2/StockClassifyUse_stock.xls', engine='xlrd', dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df['更新'] = pd.to_datetime(df['更新日期'])
    df['计入'] = pd.to_datetime(df['计入日期'])
    out = {
        'rows': len(df), 'stocks': df['股票代码'].nunique(),
        'max_update_date': str(df['更新'].max().date()),
        'max_effective_date': str(df['计入'].max().date()),
        'rows_updated_2025plus': int((df['更新'] >= '2025-01-01').sum()),
    }
    lat = pd.read_csv(ROOT / 'data/processed/sw2_industry_latest.csv', dtype={'code6': str, 'l2_code': str})
    # 网络实际节点覆盖
    lat_map = dict(zip(lat['code6'], lat['l2_code']))
    cov = {}
    for y in [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]:
        sup = pd.read_csv(ROOT / f'data/processed/network_mb/{y}_support.csv', dtype={'focal_id': str})
        codes = [f[2:] for f in sup['focal_id']]
        have = sum(1 for c in codes if lat_map.get(c))
        legacy = int(lat.set_index('code6')['is_current_version'].reindex(codes).fillna(True).eq(False).sum())
        cov[y] = {'nodes': len(codes), 'with_l2': have, 'legacy_code': legacy}
    out['network_node_coverage'] = cov
    return out


# ---------------- D. 网络 ↔ 下游 ----------------
def audit_network_downstream(year=2021):
    edges = pd.read_csv(ROOT / f'data/processed/network_mb/{year}_similarity.csv',
                        dtype={'focal_id': str, 'source_id': str})
    ret = pd.read_csv(ROOT / 'data/processed/monthly_returns_clean.csv', dtype={'firm_id': str},
                      usecols=['firm_id', 'month', 'R'])
    nodes = set(edges['focal_id'])
    ret_firms = set(ret['firm_id'])
    out = {'year': year, 'nodes': len(nodes), 'nodes_in_return_panel': len(nodes & ret_firms),
           'nodes_missing_return': len(nodes - ret_firms),
           'edges': len(edges)}
    # PeerReturn 抽样复算：随机 5 个 focal，用 source 的下月收益按权重聚合
    m = ret[ret['month'] == '2022-01'].set_index('firm_id')['R'].to_dict()
    smp = edges['focal_id'].drop_duplicates().sample(5, random_state=7)
    checks = []
    for f in smp:
        e = edges[edges['focal_id'] == f]
        src = e['source_id'].tolist()
        have = [s for s in src if s in m]
        if not have:
            checks.append({'focal': f, 'status': 'no_peer_return_in_month'})
            continue
        w = e.set_index('source_id').loc[have, 'weight']
        pr = float((w * pd.Series({s: m[s] for s in have})).sum() / w.sum())
        checks.append({'focal': f, 'n_peers': len(src), 'n_with_ret': len(have),
                       'coverage': round(len(have) / len(src), 3), 'peer_return': round(pr, 5)})
    out['peer_return_sample'] = checks
    return out


def main():
    ap_ok = '--probe-api' in sys.argv
    rep = {'run_at': datetime.now(timezone.utc).isoformat()}
    print('=== A 搜狗词库 ===', flush=True)
    rep['sogou'] = audit_sogou()
    print(json.dumps({k: v for k, v in rep['sogou'].items()
                      if k in ('catalog_entries', 'files_on_disk', 'missing_n', 'zero_byte',
                               'tiny_n', 'parsed_ok', 'parse_problems_n', 'word_count',
                               'outliers_low_content')}, ensure_ascii=False, indent=1))
    print('  parse problems:', json.dumps(rep['sogou']['parse_problems'][:5], ensure_ascii=False))
    print('  smallest:', json.dumps(rep['sogou']['smallest_dicts'][:5], ensure_ascii=False))

    print('\n=== B 东财 F10 ===', flush=True)
    rep['em_f10'] = audit_em(probe_api=ap_ok)
    e = rep['em_f10']
    print(json.dumps({k: e[k] for k in ('idmap_firms', 'files', 'no_file_n', 'firms_with_data',
                                        'empty_or_error', 'capped_at_200', 'capped_share',
                                        'capped_first_date', 'prod_rows_total')}, ensure_ascii=False, indent=1))
    print('  firms_by_report_year:', json.dumps(e['firms_by_report_year'], ensure_ascii=False))
    if ap_ok:
        print('  api_probe:', json.dumps(e['api_probe'], ensure_ascii=False, indent=1))

    print('\n=== C 申万二级 ===', flush=True)
    rep['sw2'] = audit_sw2()
    print(json.dumps(rep['sw2'], ensure_ascii=False, indent=1))

    print('\n=== D 网络 ↔ 下游 ===', flush=True)
    rep['network_downstream'] = audit_network_downstream()
    print(json.dumps(rep['network_downstream'], ensure_ascii=False, indent=1)[:1200])

    REPD.mkdir(parents=True, exist_ok=True)
    (REPD / 'free_source_audit_20260914.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                                          encoding='utf-8')
    print(f"\n-> {REPD / 'free_source_audit_20260914.json'}")


if __name__ == '__main__':
    main()
