"""Drive 批量下载（socks5h 代理 + confirm token 处理）。
用法: python scripts/drive_download.py --ids artifacts/review/drive_file_ids.json --names "name1" "name2" ...
"""
import argparse, json, re, subprocess
from pathlib import Path

PROXY = "socks5h://127.0.0.1:7993"
OUT = Path("data/raw/drive")
OUT.mkdir(parents=True, exist_ok=True)

def curl(url, out_path, timeout=600):
    r = subprocess.run(["curl", "-sL", "--max-time", str(timeout), "-x", PROXY, url, "-o", str(out_path)],
                       capture_output=True)
    return r.returncode

def download(fname, fid):
    out0 = OUT / f"{fname}.tmp"
    final = OUT / fname
    # 1) 首次请求（小文件直接返回；大文件返回 confirm 页）
    curl(f"https://drive.google.com/uc?export=download&id={fid}", out0)
    b = out0.read_bytes()
    if b[:2] == b"PK" or b[:4] == b"\x80\x04" or b[:2] == b"\x80":
        out0.rename(final)
        return final.stat().st_size
    # 2) confirm token 提取
    html = out0.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'name="confirm" value="([^"]+)"', html)
    uuid = re.search(r'name="uuid" value="([^"]+)"', html)
    confirm = m.group(1) if m else "t"
    u = uuid.group(1) if uuid else ""
    url2 = (f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm={confirm}"
            + (f"&uuid={u}" if u else ""))
    curl(url2, out0)
    if out0.stat().st_size < 1000:
        return -out0.stat().st_size
    out0.rename(final)
    return final.stat().st_size

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="artifacts/review/drive_file_ids.json")
    ap.add_argument("--names", nargs="+", required=True)
    args = ap.parse_args()
    ids = json.loads(Path(args.ids).read_text(encoding="utf-8"))
    for name in args.names:
        if name not in ids:
            print(f"SKIP 未知文件: {name}", flush=True)
            continue
        fid = ids[name]["id"]
        target = OUT / name
        if target.exists():
            print(f"SKIP 已存在: {name} ({target.stat().st_size/1e6:.0f} MB)", flush=True)
            continue
        try:
            sz = download(name, fid)
            if sz > 0:
                print(f"OK   {name}: {sz/1e6:.1f} MB", flush=True)
            else:
                print(f"FAIL {name}: 下载过小({-sz}B)", flush=True)
        except Exception as e:
            print(f"FAIL {name}: {e}", flush=True)

if __name__ == "__main__":
    main()
