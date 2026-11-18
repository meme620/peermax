"""申万二级行业分类获取（免费官方源，2026-09-14）。

来源：申万宏源证券研究所官网 `swindex/pdf/SwClass2021/StockClassifyUse_stock.xls`
（SwClass2021 = 2021 版分类；AKShare `stock_industry_clf_hist_sw()` 同源）。

用法：
    python scripts/fetch_sw2_industry.py            # 下载 + 落盘 + 校验
    python scripts/fetch_sw2_industry.py --inspect  # 仅解析本地文件并打印结构

产物：
    data/raw/sw2/StockClassifyUse_stock.xls
    data/raw/sw2/fetch_meta.json   （url/http/bytes/sha256/fetched_at + 解析诊断）

纪律：
    - 原子落盘（.part → rename），失败不清空既有文件
    - 站点 HEAD 返 502，必须 GET；curl 在本机 TLS 不通，用 urllib（关证书校验）
    - 只下载不加工：解析产物另由 s3_* 脚本负责
"""
import argparse
import hashlib
import json
import ssl
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTD = ROOT / 'data' / 'raw' / 'sw2'
XLS = OUTD / 'StockClassifyUse_stock.xls'
META = OUTD / 'fetch_meta.json'
URL = 'https://www.swsresearch.com/swindex/pdf/SwClass2021/StockClassifyUse_stock.xls'


def _ctx():
    c = ssl.create_default_context()
    c.check_hostname = False
    c.verify_mode = ssl.CERT_NONE
    return c


def download() -> dict:
    OUTD.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Referer': 'https://www.swsresearch.com/',
    })
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120, context=_ctx()) as r:
        status = r.status
        ctype = r.headers.get('Content-Type')
        blob = r.read()
    part = XLS.with_suffix('.xls.part')
    part.write_bytes(blob)
    part.replace(XLS)
    sha = hashlib.sha256(blob).hexdigest()
    rec = {
        'source_name': '申万宏源证券研究所 SwClass2021 股票行业分类',
        'url': URL,
        'access_method': 'free_http_get',
        'http_status': status,
        'content_type': ctype,
        'bytes': len(blob),
        'sha256': sha,
        'magic': blob[:8].hex(),
        'fetched_at': datetime.now(timezone.utc).isoformat(),
        'elapsed_sec': round(time.time() - t0, 2),
        'local_path': str(XLS.relative_to(ROOT)).replace('\\', '/'),
    }
    print(f"downloaded {len(blob)} bytes -> {XLS} (sha256 {sha[:16]}…)")
    return rec


def inspect() -> dict:
    import pandas as pd
    xl = pd.ExcelFile(XLS, engine='xlrd')
    diag = {'sheets': xl.sheet_names, 'sheets_detail': {}}
    for sh in xl.sheet_names:
        df = xl.parse(sh, dtype=str)
        df.columns = [str(c).strip() for c in df.columns]
        d = {
            'shape': list(df.shape),
            'columns': list(df.columns),
            'head': df.head(3).to_dict('records'),
            'nunique': {c: int(df[c].nunique()) for c in df.columns[:6]},
            'null_counts': {c: int(df[c].isna().sum()) for c in df.columns},
        }
        # 行业代码层级分布（按前 4 位分组）
        code_col = next((c for c in df.columns if '行业' in c and '代码' in c), None)
        if code_col is not None:
            s = df[code_col].dropna().astype(str).str.strip()
            d['industry_code_len_counts'] = {int(k): int(v) for k, v in s.str.len().value_counts().items()}
            d['industry_code_prefix3_top'] = {str(k): int(v) for k, v in s.str[:3].value_counts().head(10).items()}
        diag['sheets_detail'][sh] = d
        print(f"[sheet {sh}] shape={d['shape']} cols={d['columns']}")
        for c in df.columns:
            print(f"   - {c}: nunique={df[c].nunique()} null={df[c].isna().sum()} sample={df[c].dropna().head(2).tolist()}")
        if code_col:
            print(f"   industry code len counts: {d.get('industry_code_len_counts')}")
            print(f"   industry code prefix3 top: {d.get('industry_code_prefix3_top')}")
    return diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inspect', action='store_true', help='只解析本地文件')
    ap.add_argument('--skip-download', action='store_true')
    args = ap.parse_args()

    rec = None
    if not args.inspect and not args.skip_download:
        rec = download()
    if not XLS.exists():
        sys.exit(f"missing {XLS}; run without --inspect first")

    diag = inspect()
    meta = {}
    if META.exists():
        meta = json.loads(META.read_text(encoding='utf-8'))
    if rec:
        meta.update(rec)
    if not meta:
        blob = XLS.read_bytes()
        meta = {
            'source_name': '申万宏源证券研究所 SwClass2021 股票行业分类',
            'url': URL,
            'local_path': str(XLS.relative_to(ROOT)).replace('\\', '/'),
            'bytes': len(blob),
            'sha256': hashlib.sha256(blob).hexdigest(),
            'note': 'meta 由 --inspect 路径补写（未重新下载）',
        }
    meta['parse_diagnostics'] = diag
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\nmeta -> {META}")


if __name__ == '__main__':
    main()
