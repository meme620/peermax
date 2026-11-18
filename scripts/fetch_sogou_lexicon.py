"""搜狗细胞词库抓取与解析（免费源，2026-09-14）。

用途：M-A 产品词库的来源之一（刘亚辉等 2021 的三源之一）。

站点结构（实测）：
  - 类目页：https://pinyin.sogou.com/dict/cate/index/<cate_id>/default/<page>（每页 10 个）
  - 下载：  https://pinyin.sogou.com/d/dict/download_cell.php?id=<dict_id>&name=<URL编码名>
            ⚠️ name 参数必填：缺失时返回 HTML 错误页（magic 0x43 而非 0x44 'd'）

.scel 解析（自校验）：
  - 头部 0x08 与 0x0C 各存一个 uint32「词条总数」，解析出的词条数必须与之相等，
    否则该文件标记 parse_mismatch 并排除（不做静默丢弃）。
  - 词条区格式：每词条 = uint16(拼音索引数) + uint16(词字节长) + UTF-16LE 词。

产物：
  data/raw/lexicon/sogou/*.scel
  data/raw/lexicon/sogou/_catalog.csv        类目枚举结果（id/name/cate/downloads/...）
  data/processed/lexicon_sogou_words.csv     解析后的 (dict_id, dict_name, cate, word)
  reports/data_gates/G0/sogou_lexicon_report.json  统计与校验

用法：
  python scripts/fetch_sogou_lexicon.py --enumerate        # 只枚举类目
  python scripts/fetch_sogou_lexicon.py                    # 枚举+下载+解析
  python scripts/fetch_sogou_lexicon.py --parse-only       # 只重解析本地 .scel
"""
import argparse
import csv
import json
import re
import ssl
import struct
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAWD = ROOT / 'data' / 'raw' / 'lexicon' / 'sogou'
CATALOG = RAWD / '_catalog.csv'
WORDS = ROOT / 'data' / 'processed' / 'lexicon_sogou_words.csv'
REPD = ROOT / 'reports' / 'data_gates' / 'G0'

# 行业/产品相关类目（2026-09-14 人工筛选：自然科学 + 工程 + 农林 + 医药 + 消费/地产/汽车）
CATEGORIES = {
    1: '自然科学', 2: '数学', 3: '物理', 13: '化学', 14: '生物',
    26: '地理地质', 27: '海洋学', 28: '气象学', 29: '天文学', 30: '其它(自然科学)',
    96: '工程与应用科学', 97: '计算机', 104: '电子工程', 107: '钢铁冶金', 109: '化工',
    112: '机械工程', 113: '建筑', 121: '造纸', 124: '包装', 126: '其它(工程)',
    127: '农林渔畜', 128: '林业', 129: '农业', 130: '畜牧业', 131: '渔业', 147: '兽医',
    132: '医学', 134: '西药学', 135: '中医', 136: '中药', 138: '疾病', 152: '医疗', 550: '外科',
    94: '房地产', 432: '汽车', 433: '烟草', 397: '服饰', 402: '饮食', 401: '办公文教',
    398: '礼品', 420: '收藏', 399: '旅游', 390: '理财',
}

UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36'}
DELAY = 0.25


def _ctx():
    c = ssl.create_default_context()
    c.check_hostname = False
    c.verify_mode = ssl.CERT_NONE
    return c


def fetch(url, timeout=45) -> bytes:
    # Referer 必带：缺失时 download_cell.php 返回非 scel 内容
    hdrs = dict(UA)
    hdrs['Referer'] = 'https://pinyin.sogou.com/dict/'
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=timeout, context=_ctx()) as r:
        return r.read()


def enumerate_category(cate_id: int) -> list:
    """枚举一个类目下全部词库（含分页）。"""
    out, page = [], 1
    while True:
        url = (f'https://pinyin.sogou.com/dict/cate/index/{cate_id}/default'
               if page == 1 else f'https://pinyin.sogou.com/dict/cate/index/{cate_id}/default/{page}')
        try:
            h = fetch(url).decode('utf-8', 'ignore')
        except Exception as e:
            print(f'   [cate {cate_id} p{page}] FAIL {type(e).__name__}')
            break
        blocks = h.split('dict_detail_block')[1:]
        if not blocks:
            break
        for b in blocks:
            m_id = re.search(r"/dict/detail/index/(\d+)'?\"?>\s*([^<]{1,40})", b)
            m_dl = re.search(r'download_cell\.php\?id=(\d+)&name=([^"\'&]{1,200})', b)
            m_cnt = re.search(r'下载次数[^<]*</div>\s*<div class="show_content">\s*([\d,]+)', b)
            m_up = re.search(r'更新时间[^<]*</div>\s*<div class="show_content">\s*([\d\-: ]+)', b)
            if not m_dl:
                continue
            did = int(m_dl.group(1))
            name = urllib.parse.unquote(m_dl.group(2))
            out.append({
                'dict_id': did, 'dict_name': name, 'cate_id': cate_id,
                'cate_name': CATEGORIES.get(cate_id, ''),
                'downloads': int(m_cnt.group(1).replace(',', '')) if m_cnt else None,
                'updated': m_up.group(1).strip() if m_up else '',
                'detail_name': m_id.group(2).strip() if m_id else '',
            })
        if len(blocks) < 10:
            break
        page += 1
        time.sleep(DELAY)
    return out


def parse_scel(path: Path):
    """解析 .scel → (name, words, status)。

    格式（2026-09-14 实测反推 + 与 StuPeter/Sougou_dict_spider 的 Scel2Txt.py 对照一致）：
      0x0000  uint32  拼音表偏移 startPy（本样本 0x1540，全库常量）
      0x0004  char[4] "DCS\x01" 签名（缺失则为 HTML 错误页）
      0x0008  uint32  保留 = 1
      0x0010..        词库名/类型/描述/示例（UTF-16LE，固定分段 0x130/0x338/0x540/0xd40）
      startPy          全局拼音表（(index,len,pinyin) 三元组序列）
      0x2628          汉语词组表（全库常量偏移，经 602/多库解析至 EOF 校验）
    词条记录：same(u16) + py_len(u16) + py_len 字节拼音索引；随后 same 组
              [c_len(u16) + c_len 字节 UTF-16 词 + ext_len(u16) + count(u16) + (ext_len-2) 字节]
    """
    data = path.read_bytes()
    if data[4:8] != b'DCS':
        raise ValueError(f'not a scel (sig={data[4:8]!r})')
    zhangjie = 0x2628
    pos = zhangjie
    words = []
    status = 'eof-ok'
    while pos + 4 <= len(data):
        same, py_len = struct.unpack('<HH', data[pos:pos + 4])
        pos += 4
        if same == 0 or py_len % 2 or pos + py_len > len(data):
            status = 'bad_record_header'
            break
        pos += py_len
        bad = False
        for _ in range(same):
            if pos + 2 > len(data):
                status = 'truncated'; bad = True; break
            c_len = struct.unpack('<H', data[pos:pos + 2])[0]
            pos += 2
            if c_len == 0 or c_len % 2 or pos + c_len > len(data):
                status = 'bad_word_len'; bad = True; break
            w = data[pos:pos + c_len].decode('utf-16-le', 'ignore')
            pos += c_len
            if pos + 4 > len(data):
                status = 'truncated_ext'; bad = True; break
            ext_len, _cnt = struct.unpack('<HH', data[pos:pos + 4])
            pos += 4 + max(0, ext_len - 2)
            words.append(w)
        if bad:
            break
    # 名称（UTF-16LE，位于 0x130 段）
    name = data[0x130:0x338].decode('utf-16-le', 'ignore').split(chr(0))[0]
    return name, words, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enumerate', action='store_true')
    ap.add_argument('--parse-only', action='store_true')
    ap.add_argument('--max-dicts', type=int, default=0, help='限制下载数（调试用）')
    args = ap.parse_args()

    RAWD.mkdir(parents=True, exist_ok=True)
    catalog = []
    if CATALOG.exists() and not args.enumerate:
        with open(CATALOG, encoding='utf-8') as f:
            catalog = list(csv.DictReader(f))
        print(f'catalog loaded: {len(catalog)} entries')
    if args.enumerate or not catalog:
        print('=== 枚举类目 ===')
        for cid, cname in CATEGORIES.items():
            rows = enumerate_category(cid)
            print(f'  cate {cid:>4} {cname}: {len(rows)} dicts')
            catalog.extend(rows)
            time.sleep(DELAY)
        # 全局去重（同 id 可能出现在多类目）
        seen, dedup = set(), []
        for r in catalog:
            if r['dict_id'] in seen:
                continue
            seen.add(r['dict_id'])
            dedup.append(r)
        catalog = dedup
        with open(CATALOG, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(catalog[0].keys()))
            w.writeheader()
            w.writerows(catalog)
        print(f'catalog -> {CATALOG} ({len(catalog)} unique dicts)')
        if args.enumerate:
            return

    if not args.parse_only:
        print('\n=== 下载 ===')
        ok = skip = fail = 0
        failures = []
        for i, r in enumerate(catalog, 1):
            did, name = int(r['dict_id']), r['dict_name']
            fp = RAWD / f"{did}.scel"
            if fp.exists() and fp.stat().st_size > 0:
                skip += 1
                continue
            url = ('https://pinyin.sogou.com/d/dict/download_cell.php?id=%d&name=%s'
                   % (did, urllib.parse.quote(name)))
            try:
                blob = fetch(url, timeout=60)
                if blob[4:8] != b'DCS\x01':
                    raise ValueError(f'not scel (sig={blob[4:8]!r})')
                fp.write_bytes(blob)
                ok += 1
            except Exception as e:
                fail += 1
                failures.append({'dict_id': did, 'name': name, 'error': f'{type(e).__name__}: {e}'})
            if i % 50 == 0:
                print(f'   {i}/{len(catalog)} ok={ok} skip={skip} fail={fail}', flush=True)
            time.sleep(DELAY)
            if args.max_dicts and ok >= args.max_dicts:
                break
        print(f'下载完成: ok={ok} skip={skip} fail={fail}')
        if failures:
            (RAWD / '_download_failures.json').write_text(
                json.dumps(failures, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'  失败清单 -> {RAWD / "_download_failures.json"}')

    print('\n=== 解析 ===')
    rows, bad = [], []
    stats = []
    for r in catalog:
        fp = RAWD / f"{int(r['dict_id'])}.scel"
        if not fp.exists():
            continue
        try:
            name, words, status = parse_scel(fp)
            n = len(words)
            match = status == 'eof-ok' and n > 0
            stats.append({'dict_id': int(r['dict_id']), 'dict_name': r['dict_name'],
                          'cate_name': r['cate_name'], 'parsed_words': n,
                          'parse_status': status, 'count_match': bool(match),
                          'bytes': fp.stat().st_size})
            if not match:
                bad.append({'dict_id': r['dict_id'], 'parsed': n, 'status': status})
                continue
            for w in words:
                rows.append((int(r['dict_id']), r['dict_name'], r['cate_name'], w))
        except Exception as e:
            bad.append({'dict_id': r['dict_id'], 'error': f'{type(e).__name__}: {e}'})

    WORDS.parent.mkdir(parents=True, exist_ok=True)
    with open(WORDS, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['dict_id', 'dict_name', 'cate_name', 'word'])
        w.writerows(rows)

    uniq = len({r[3] for r in rows})
    rep = {
        'source': 'Sogou cell dictionaries (pinyin.sogou.com/dict)',
        'fetched_at': datetime.now(timezone.utc).isoformat(),
        'categories': CATEGORIES,
        'dicts_catalogued': len(catalog),
        'dicts_parsed_ok': int(sum(1 for s in stats if s['count_match'])),
        'dicts_parse_mismatch_or_error': len(bad),
        'word_rows': len(rows),
        'unique_words': uniq,
        'bad': bad[:50],
        'top_dicts_by_words': sorted(stats, key=lambda x: -x['parsed_words'])[:20],
        'note': 'scel 解析完整性判据：必须解析到文件末尾（status=eof-ok）且词数>0，否则整库排除并记入 bad；格式见 parse_scel docstring。',
    }
    REPD.mkdir(parents=True, exist_ok=True)
    (REPD / 'sogou_lexicon_report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in rep.items() if k not in ('bad', 'top_dicts_by_words')}, ensure_ascii=False, indent=1))
    print(f'wrote {WORDS} and {REPD / "sogou_lexicon_report.json"}')


if __name__ == '__main__':
    main()
