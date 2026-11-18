"""批次 1 下载：Drive → data/raw/drive/（走 socks5h://127.0.0.1:7993 代理）。
大文件用 drive.usercontent.google.com 端点（免 confirm token）。
"""
import json, subprocess, time
from pathlib import Path
from fileio import read_json

OUT = Path('data/raw/drive')
OUT.mkdir(parents=True, exist_ok=True)
IDS = read_json('artifacts/review/drive_file_ids.json')
PROXY = 'socks5h://127.0.0.1:7993'

BATCH1 = ['daily_limit_up_flag_20260630.pkl', 'daily_limit_down_flag_20260630.pkl',
          'ind_code.pkl', 'daily_adj_close_post_20260630.pkl',
          'stock_st_flag_20260630.pkl', 'daily_close_none_20260630.pkl']

def dl(fid, dest):
    url = f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"
    r = subprocess.run(['curl', '-sL', '--proxy', PROXY, '--max-time', '1800',
                        '--retry', '2', '-o', str(dest), '-w', '%{http_code} %{size_download}',
                        url], capture_output=True, text=True)
    return r.stdout.strip()

for name in BATCH1:
    info = IDS.get(name)
    if not info:
        print(f"SKIP {name} (no id)")
        continue
    dest = OUT / name
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"SKIP {name} (exists {dest.stat().st_size/1e6:.1f}MB)")
        continue
    t0 = time.time()
    res = dl(info['id'], dest)
    code_size = res.split(' ')
    ok = dest.exists() and dest.stat().st_size > 1000
    head = b''
    if ok:
        head = dest.read_bytes()[:4]
    print(f"{'OK ' if ok else 'FAIL'} {name}: http={code_size[0] if code_size else '?'} "
          f"size={dest.stat().st_size/1e6:.1f}MB head={head[:4]} {time.time()-t0:.0f}s", flush=True)
print("BATCH1 DONE")
