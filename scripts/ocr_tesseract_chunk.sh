#!/usr/bin/env bash
# OCR 逐页 tesseract（供 scripts/network_prospectus_anchor.py 的 OCR 回退通道调用；
# registry 1.5.13 登记通道 = WSL Ubuntu tesseract 5.3.4, chi_sim, --psm 3, 200 DPI）。
# 用法: ocr_tesseract_chunk.sh <页目录(dir)> <start> <end> [lang] [psm]
#   dir 内页图命名 p%05d.png（Windows 侧 pymupdf 渲染）；逐页产出 p%05d.txt（同目录）。
# 关键：限制 tesseract 的 OpenMP 线程数。默认 tesseract 5 每进程起 nproc 个线程；
# 多文件并发 OCR 时（6 文件 × 2 chunk）线程数迅速压垮 vCPU（实测 load 95/32 → 0.12 页/s）。
export OMP_THREAD_LIMIT=2
export OMP_NUM_THREADS=2

dir="$1"; s="$2"; e="$3"; lang="${4:-chi_sim}"; psm="${5:-3}"
cd "$dir" || exit 3
for i in $(seq "$s" "$e"); do
  f=$(printf 'p%05d.png' "$i")
  [ -f "$f" ] && tesseract "$f" "${f%.png}" -l "$lang" --psm "$psm" >/dev/null 2>&1
done
exit 0
