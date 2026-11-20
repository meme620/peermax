"""analyst-network Step 1：分片文件级验收 + 合并（零依赖，流式）。

输入: 预测情况(1).7z（外层 ZIP，内含 6 个 xlsx + 字段说明）
输出:
  data/raw/analyst/forecast_raw_all.csv   合并后的原始数据（保留两行表头语义，数据行 13 列）
  data/raw/analyst/step1_acceptance.json  验收报告
  data/raw/analyst/预测情况_字段说明.txt   字段说明解出落盘
"""
import zipfile, io, json, csv
from pathlib import Path

from analyst_xlsx_read import load_shared_strings, iter_rows

OUT = Path('data/raw/analyst')
OUT.mkdir(parents=True, exist_ok=True)

ZIP = 'data/raw/analyst/预测情况(1).7z'
SHARDS = [f'预测情况_{i}.xlsx' for i in range(1, 7)]
EXPECTED_HEADER = ['Scode','Anndats','Fpenddt','A_name','A_code','Star',
                   'B_name','B_code','Feps','Fpe','Fnetpro','Foperev','Anndats_act']

def main():
    report = {'shards': []}
    total_rows = 0
    total_bad = 0
    headers_ok = True
    with zipfile.ZipFile(ZIP) as outer:
        # 解出字段说明
        outer.extract('预测情况_字段说明.txt', OUT)
        with open(OUT / 'forecast_raw_all.csv', 'w', newline='', encoding='utf-8') as fout:
            w = csv.writer(fout)
            w.writerow(EXPECTED_HEADER)
            for shard in SHARDS:
                data = outer.read(shard)
                zf = zipfile.ZipFile(io.BytesIO(data))
                shared = load_shared_strings(zf)
                rows = 0; bad = 0; header_line = 0; hdr_seen = None; hdr2 = None
                with zf.open('xl/worksheets/sheet1.xml') as f:
                    for r in iter_rows(zf, shared):
                        header_line += 1
                        if header_line == 1:
                            hdr_seen = list(r)
                            if hdr_seen != EXPECTED_HEADER:
                                headers_ok = False
                            continue
                        if header_line == 2:
                            hdr2 = list(r)
                            # 表头断言（2026-08-12 偏差 #1）：第 2 行必须是中文说明行，
                            # 若与英文表头相同则说明重复表头（数据格式变化），标记异常
                            if hdr2 == EXPECTED_HEADER:
                                headers_ok = False
                            continue  # 中文说明行
                        if len(r) != 13:
                            bad += 1
                            continue
                        rows += 1   # data_rows 只计好行（坏行单独计 bad_rows）
                        w.writerow(r)
                total_rows += rows; total_bad += bad
                report['shards'].append({
                    'shard': shard, 'data_rows': rows, 'bad_rows': bad,
                    'header_matches': hdr_seen == EXPECTED_HEADER,
                    'second_row_is_dup_header': hdr2 == EXPECTED_HEADER,   # 重复表头检测（#1）
                })
                print(f"{shard}: data_rows={rows} bad={bad} header_ok={hdr_seen == EXPECTED_HEADER}", flush=True)
    report['total_data_rows'] = total_rows
    report['total_bad_rows'] = total_bad
    report['all_headers_ok'] = headers_ok
    (OUT / 'step1_acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"DONE total={total_rows} bad={total_bad} headers_ok={headers_ok}")

if __name__ == '__main__':
    main()
