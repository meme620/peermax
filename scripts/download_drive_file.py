"""Drive 文件下载（socks5h 代理 + confirm 处理）。用法: python download_drive_file.py <file_id> <out_path>"""
import sys, subprocess, tempfile, re, os, time
from pathlib import Path

PROXY = "socks5h://127.0.0.1:7993"
def curl(args, timeout=300, retry=3):
    for i in range(retry):
        r = subprocess.run(["curl", "--proxy", PROXY, "-sL", "--max-time", str(timeout)] + args,
                           capture_output=True, timeout=timeout + 30)
        if r.returncode == 0:
            return r
        time.sleep(3)
    return r

def main():
    fid, out = sys.argv[1], Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    # 1) 先试 usercontent 直接下载
    url = f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"
    tmp = out.with_suffix(".part")
    r = curl(["-o", str(tmp), url])
    if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
        head = tmp.read_bytes()[:200]
        if head[:4] != b"%PDF" and b"<!DOCTYPE" not in head and b"<html" not in head:
            tmp.replace(out)
            print(f"OK {out.name} ({tmp.stat().st_size/1e6:.1f} MB)")
            return
    # 2) confirm 页处理：uc?export=download 拿 confirm token
    r = curl(["-o", str(tmp), f"https://drive.google.com/uc?export=download&id={fid}"])
    if r.returncode != 0 or not tmp.exists():
        print(f"FAIL {out.name}: curl err={r.returncode}"); return
    body = tmp.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'name="confirm" value="([\w-]+)"', body) or re.search(r'confirm=([\w-]+)', body)
    if m:
        tok = m.group(1) if '/' not in m.group(1) else m.group(1).split('/')[0]
        r2 = curl(["-L", "-o", str(tmp),
                   f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm={tok}"])
        if r2.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(out)
            print(f"OK {out.name} ({tmp.stat().st_size/1e6:.1f} MB, confirm)")
            return
    print(f"FAIL {out.name}: size={tmp.stat().st_size if tmp.exists() else 0}, head={tmp.read_bytes()[:60] if tmp.exists() else ''}")

if __name__ == "__main__":
    main()
