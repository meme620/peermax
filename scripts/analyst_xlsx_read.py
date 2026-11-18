"""S3A Step 1 辅助：标准库流式解析 xlsx（零依赖）。
支持直接读外层 zip 内的 xlsx 成员。
用法:
  python scripts/analyst_xlsx_read.py <zip路径> --inner 预测情况_1.xlsx [--max-rows N]
"""
import sys, zipfile, io, re
import xml.etree.ElementTree as ET

M = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
TAG_ROW = M + 'row'
TAG_C = M + 'c'
TAG_V = M + 'v'
TAG_T = M + 't'
TAG_SI = M + 'si'

def col_index(ref):
    m = re.match(r'([A-Z]+)', ref)
    n = 0
    for ch in m.group(1):
        n = n * 26 + (ord(ch) - 64)
    return n - 1

def load_shared_strings(zf):
    strings = []
    try:
        with zf.open('xl/sharedStrings.xml') as f:
            ctx = ET.iterparse(f, events=('end',))
            cur = None
            for ev, el in ctx:
                if el.tag == TAG_SI:
                    strings.append(''.join(t.text or '' for t in el.iter(TAG_T)))
                    el.clear()
                elif el.tag == TAG_T:
                    pass
    except KeyError:
        pass
    return strings

def iter_rows(zf, shared):
    with zf.open('xl/worksheets/sheet1.xml') as f:
        ctx = ET.iterparse(f, events=('end',))
        row = []
        for ev, el in ctx:
            if el.tag == TAG_C:
                ref = el.get('r', 'A')
                t = el.get('t')
                v = None
                for child in el.iter():
                    if child.tag in (TAG_V, TAG_T) and child.text is not None:
                        v = child.text
                        break
                if t == 's' and v is not None:
                    try:
                        v = shared[int(v)]
                    except (ValueError, IndexError):
                        v = None
                row.append((col_index(ref), v))
            elif el.tag == TAG_ROW:
                row.sort()
                yield [v for _, v in row]
                row = []
                el.clear()

def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    path = args[0]
    inner = None
    if '--inner' in sys.argv:
        inner = sys.argv[sys.argv.index('--inner') + 1]
    max_rows = None
    for a in sys.argv:
        if a.startswith('--max-rows='):
            max_rows = int(a.split('=')[1])

    if inner:
        with zipfile.ZipFile(path) as outer:
            data = outer.read(inner)
        zf = zipfile.ZipFile(io.BytesIO(data))
    else:
        zf = zipfile.ZipFile(path)

    with zf:
        shared = load_shared_strings(zf)
        print(f"[{inner or path}] sharedStrings: {len(shared)}", file=sys.stderr)
        n = 0
        for r in iter_rows(zf, shared):
            if n < 3 or (max_rows and n < max_rows):
                print(r)
            n += 1
            if max_rows and n >= max_rows:
                break
        print(f"[{inner or path}] rows read: {n}", file=sys.stderr)

if __name__ == '__main__':
    main()
