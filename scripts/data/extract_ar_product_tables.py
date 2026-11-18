"""年报「主营业务分产品情况」产品词提取（原型，2026-09-14）。

动机：东财 F10 主营构成接口硬性返回 ≤200 行（最早报告期≈2017-03，导致 2017-2019
覆盖仅 58-79%），而年报原文含完整「主营业务分行业、分产品、分地区」表，且正是
刘亚辉等 2021 的产品词原始出处。本脚本从 data/raw/cninfo/annual_txt/{code}/{year}.txt
提取分产品明细（名称/收入/占比）。

提取逻辑（面向"表格被打平为连续文本"的语料）：
  1. 定位章节：优先 '主营业务分产品情况' / '分产品' / '营业收入构成' 且其后 3000 字内
     出现 '分产品' 或 '营业收入' 表头；
  2. 在章节窗口内按"数值尾随"切行：一行 = 文本名 + 连续 ≥2 个数值%组；
  3. 过滤表头词（分产品/营业收入/营业成本/毛利率/单位/币种/适用…）与过短名字。

用法：
  python scripts/data/extract_ar_product_tables.py --sample 40     # 原型抽样评估
  python scripts/data/extract_ar_product_tables.py --full          # 全量
"""
import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AR = ROOT / 'data/raw/cninfo/annual_txt'
OUT = ROOT / 'data/processed/ar_product_tables.csv'
REPD = ROOT / 'reports/data_gates/G0'

NUM = r'(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)'
PCT = NUM + r'%'
HEADER_WORDS = set('''分产品 分行业 分地区 分销售模式 营业收入 营业成本 毛利率 单位 元 币种 人民币
适用 不适用 情况 我国 报告期 营业收入比上年增减 营业成本比上年增减 毛利率比上年增减 主营业务分产品情况
主营业务分行业情况 主营业务分地区情况 合计 小计 其中 其他业务 主营业务 产品 行业 地区 销售模式 项目 金额 占比
本公司 公司 数据 数值 说明 备注'''.split())
NAME_OK = re.compile(r'^[\u4e00-\u9fffA-Za-z0-9（）()·\-—/、\s%]{2,40}$')


def find_section(t: str):
    """定位「分产品」子表窗口：起点=分产品表头，终点=下一个子表/大标题。"""
    # 优先精确锚点
    starts = [m.start() for m in re.finditer(r'主营业务分产品情况|分产品情况|分产品', t)]
    for i in starts:
        win = t[i:i + 3000]
        if not re.search(NUM, win):
            continue
        nxt = re.search(r'(分地区情况|分行业情况|分销售模式|主营业务分地区|主营业务分行业|'
                        r'前五名客户|前五名供应商|研发投入情况|现金流量表|'
                        r'主营业务分行业、分产品、分地区、分销售模式情况)', t[i + 3:i + 3000])
        end = i + 3 + (nxt.start() if nxt else 1500)
        return i, min(end, len(t))
    return None


PREFIX_RE = re.compile(r'^(分产品情况|主营业务分产品情况|分产品|分行业情况|主营业务分行业情况|分行业|'
                       r'分地区情况|主营业务分地区情况|分地区|分销售模式)\s*[:：]?\s*')
DROP_NAMES = {'年', '第', '页', '单位', '币种', '人民币', '元', '续', '合计'}


FOOTER_RE = re.compile(r'(\d{4}\s*年年度报告全文|年度报告全文|第\s*\d+\s*页|单位\s*[:：]|币种\s*[:：]|'
                       r'[（(]续[)）]|^[一-鿿]{0,6}股份有限公司$)')
BAD_TAIL = re.compile(r'(销售量|生产量|库存量|产量|销量|金额|占比|比例|本期|上期)$')


def clean_name(name: str) -> str:
    n = re.sub(r"[\s\n\r]+", "", name)
    n = PREFIX_RE.sub("", n)
    n = FOOTER_RE.sub("", n)
    n = re.sub(r"^[、,，。:：]+", "", n)
    return n.strip()


def extract_rows(win: str):
    """在窗口内按数值尾随切行。"""
    # 找出所有数值组的位置
    num_iter = list(re.finditer(NUM + r'%?', win))
    rows = []
    if not num_iter:
        return rows
    # 行边界：一个"名字段"后紧跟 ≥2 个数值（收入/成本/毛利率 或 收入/占比）
    idx = 0
    while idx < len(num_iter):
        # 候选名字 = 上一个数值结束到当前数值开始之间的文本
        prev_end = num_iter[idx - 1].end() if idx > 0 else 0
        name = win[prev_end:num_iter[idx].start()].strip()
        name = re.sub(r'^[\s、,，。:：]+', '', name)
        # 统计紧随的连续数值个数
        k = idx
        while k + 1 < len(num_iter) and (num_iter[k + 1].start() - num_iter[k].end()) <= 2:
            k += 1
        n_contig = k - idx + 1
        name = clean_name(name)
        if (2 <= len(name) <= 30 and n_contig >= 2 and NAME_OK.match(name)
                and name not in HEADER_WORDS and not BAD_TAIL.search(name)
                and name not in DROP_NAMES
                and not re.fullmatch(r'[\d\s,，.%\-—]+', name)):
            revenue = None
            try:
                revenue = float(num_iter[idx].group(0).rstrip('%').replace(',', ''))
            except Exception:
                pass
            pct = None
            for j in range(idx, min(k + 1, idx + 3)):
                s = num_iter[j].group(0)
                if s.endswith('%'):
                    try:
                        pct = float(s.rstrip('%'))
                    except Exception:
                        pass
            rows.append({'item_name': name, 'revenue': revenue, 'pct': pct})
            idx = k + 1
        else:
            idx += 1
    return rows


def process(path: Path, code: str, year: str):
    t = path.read_text(encoding='utf-8', errors='ignore')
    sec = find_section(t)
    if not sec:
        return None, 'no_section'
    win = t[sec[0]:sec[1]]
    rows = extract_rows(win)
    if not rows:
        return None, 'no_rows'
    for r in rows:
        r.update({'code6': code, 'year': year})
    return rows, 'ok'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=0)
    ap.add_argument('--full', action='store_true')
    args = ap.parse_args()

    files = sorted(AR.rglob('*.txt'))
    if args.sample:
        step = max(1, len(files) // args.sample)
        files = files[::step][:args.sample]
    print(f'reports to process: {len(files)}', flush=True)

    all_rows, stats, examples = [], {'ok': 0, 'no_section': 0, 'no_rows': 0}, []
    for i, f in enumerate(files, 1):
        code, year = f.parent.name, f.stem
        rows, why = process(f, code, year)
        stats[why] = stats.get(why, 0) + 1
        if rows:
            all_rows.extend(rows)
            if len(examples) < 8:
                examples.append({'file': f'{code}/{year}', 'n_items': len(rows),
                                 'sample': [r['item_name'] for r in rows[:6]]})
        if i % 2000 == 0:
            print(f'  {i}/{len(files)} ok={stats["ok"]} no_section={stats["no_section"]} no_rows={stats["no_rows"]}', flush=True)

    if not args.sample or len(all_rows) > 100:
        with open(OUT, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=['code6', 'year', 'item_name', 'revenue', 'pct'])
            w.writeheader()
            w.writerows(all_rows)
    rep = {'run_at': datetime.now(timezone.utc).isoformat(), 'files': len(files),
           'stats': stats, 'rows': len(all_rows),
           'ok_rate': round(stats['ok'] / len(files), 4),
           'unique_items': len({r['item_name'] for r in all_rows}),
           'examples': examples,
           'output': str(OUT.relative_to(ROOT)).replace('\\', '/')}
    (REPD / 'ar_product_tables_report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in rep.items() if k != 'examples'}, ensure_ascii=False, indent=1))
    print('examples:')
    for e in examples[:6]:
        print('  ', e)


if __name__ == '__main__':
    main()
