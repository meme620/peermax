"""东方财富 F10 主营构成抓取（免费源，2026-09-14）。

用途：M-A 产品词库来源之一 —— 刘亚辉等 2021 的"年报披露的业务构成产品词"。
F10 主营构成与年报"主营业务分产品情况"同源，`MAINOP_TYPE=2`（按产品）的 ITEM_NAME
即披露口径的产品名；`MAINOP_TYPE=1` 按行业、`3` 按地区，仅作辅助。

接口（免鉴权，实测 200）：
  https://emweb.securities.eastmoney.com/PC_HSF10/BusinessAnalysis/PageAjax?code=SH600519
  返回 {zyfw(主营范围), zygcfx(主营构成, ≤200 条多期), jyps(经营评述)}

产物：
  data/raw/eastmoney/f10_business/{sh600519}.json.gz   原始响应（逐股）
  data/processed/em_business_terms.csv                 解析后的产品词面板
  reports/data_gates/G0/em_business_report.json        抓取与解析统计

用法：
  python scripts/data/fetch_em_f10_business.py                # 抓取（可续传）+ 解析
  python scripts/data/fetch_em_f10_business.py --parse-only   # 只重解析
  python scripts/data/fetch_em_f10_business.py --limit 20     # 调试
"""
import argparse
import csv
import gzip
import json
import ssl
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IDMAP = ROOT / 'data/processed/firm_id_map.csv'
RAWD = ROOT / 'data/raw/eastmoney/f10_business'
TERMS = ROOT / 'data/processed/em_business_terms.csv'
REPD = ROOT / 'reports/data_gates/G0'
URL = 'https://emweb.securities.eastmoney.com/PC_HSF10/BusinessAnalysis/PageAjax?code={code}'
UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36',
      'Referer': 'https://emweb.securities.eastmoney.com/'}
DELAY = 0.25


def _ctx():
    c = ssl.create_default_context()
    c.check_hostname = False
    c.verify_mode = ssl.CERT_NONE
    return c


def em_code(firm_id: str) -> str:
    """sh600519 -> SH600519"""
    return firm_id[:2].upper() + firm_id[2:]


def fetch_one(code: str, timeout=45):
    req = urllib.request.Request(URL.format(code=code), headers=UA)
    with urllib.request.urlopen(req, timeout=timeout, context=_ctx()) as r:
        raw = r.read()
    # 服务端偶发返回 gzip 体（urllib 不自动解压）→ 按 magic 判断
    if raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)
    return raw


def load_json_gz(p: Path):
    with gzip.open(p, 'rt', encoding='utf-8') as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parse-only', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    idmap = list(csv.DictReader(open(IDMAP, encoding='utf-8')))
    RAWD.mkdir(parents=True, exist_ok=True)
    failures = []

    if not args.parse_only:
        print(f'=== 抓取 {len(idmap)} 只 ===', flush=True)
        ok = skip = fail = 0
        todo = idmap[:args.limit] if args.limit else idmap
        for i, r in enumerate(todo, 1):
            fid = r['firm_id']
            fp = RAWD / f'{fid}.json.gz'
            if fp.exists() and fp.stat().st_size > 0:
                skip += 1
                continue
            code = em_code(fid)
            blob = None
            for attempt in (1, 2):
                try:
                    raw = fetch_one(code)
                    d = json.loads(raw)
                    if not isinstance(d, dict):
                        raise ValueError('unexpected payload')
                    blob = json.dumps(d, ensure_ascii=False).encode('utf-8')
                    break
                except Exception as e:
                    err = f'{type(e).__name__}: {e}'
                    time.sleep(0.8 * attempt)
            if blob is None:
                fail += 1
                failures.append({'firm_id': fid, 'code': code, 'error': err})
            else:
                with gzip.open(fp, 'wb') as f:
                    f.write(blob)
                ok += 1
            if i % 100 == 0:
                print(f'   {i}/{len(todo)} ok={ok} skip={skip} fail={fail}', flush=True)
            time.sleep(DELAY)
        print(f'抓取完成: ok={ok} skip={skip} fail={fail}')
        if failures:
            (RAWD / '_failures.json').write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'  失败 -> {RAWD / "_failures.json"}')

    print('\n=== 解析 ===', flush=True)
    rows = []
    per_firm = []
    codes_with_data = 0
    type_counts = {}
    for r in idmap:
        fid = r['firm_id']
        fp = RAWD / f'{fid}.json.gz'
        if not fp.exists():
            continue
        try:
            d = load_json_gz(fp)
        except Exception:
            continue
        zg = d.get('zygcfx') or []
        if zg:
            codes_with_data += 1
        n_items = 0
        for x in zg:
            t = str(x.get('MAINOP_TYPE') or '')
            type_counts[t] = type_counts.get(t, 0) + 1
            rd = (x.get('REPORT_DATE') or '')[:10]
            rows.append((fid, r['name'], rd, t, x.get('ITEM_NAME') or '',
                         x.get('MAIN_BUSINESS_INCOME'), x.get('MBI_RATIO'), x.get('RANK')))
            n_items += 1
        per_firm.append({'firm_id': fid, 'items': n_items,
                         'has_zyfw': bool(d.get('zyfw')), 'has_jyps': bool(d.get('jyps'))})

    with open(TERMS, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['firm_id', 'name', 'report_date', 'mainop_type', 'item_name',
                    'income', 'ratio', 'rank'])
        w.writerows(rows)

    by_prod = [r for r in rows if r[3] == '2']
    uniq_names = len({r[4] for r in by_prod if r[4]})
    rep = {
        'source': 'eastmoney F10 BusinessAnalysis PageAjax',
        'url_template': URL,
        'fetched_at': datetime.now(timezone.utc).isoformat(),
        'firms_in_idmap': len(idmap),
        'firms_with_file': len(per_firm),
        'firms_with_zygcfx': codes_with_data,
        'rows_total': len(rows),
        'rows_by_type': type_counts,
        'rows_by_product': len(by_prod),
        'unique_product_names': uniq_names,
        'product_rows_period_range': [min((r[2] for r in by_prod), default=''),
                                      max((r[2] for r in by_prod), default='')],
        'note': ('接口每次最多返回 200 条 zygcfx（含多期），故早期年份可能被截断；'
                 '解析保留 MAINOP_TYPE=2（按产品）作产品词主源，1/3 为辅助。'),
    }
    REPD.mkdir(parents=True, exist_ok=True)
    (REPD / 'em_business_report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    print(f'wrote {TERMS} and {REPD / "em_business_report.json"}')


if __name__ == '__main__':
    main()
