"""S3 边级 precision 人工判读：抽样 / 工作表 / 统计（第二梯队 #10 路线 A，B″11-9）。

**冻结口径真源**：`configs/parameter_registry.yaml` → `edge_precision_manual_annotation` 与
`reports/audits/s3_edge_precision_protocol_20261013.md`（**先冻结后抽样**：脚本启动时校验协议冻结时间戳
早于当前时间，否则 [STOP] 退出）。

模式：
  --sample（默认）  按冻结口径分层抽样 → 样本 CSV + 人读工作表 HTML + result.json 空壳 + 反例库空表头
  --compute        读**已判读的**样本 CSV → 分层 precision + Wilson CI95 + 加权池化 + 反例库
                   （2026-10-14 扩展：支持 **3 个 AI 判读代理**——`--labels A,B,C` 先把三位代理标签
                    合并回填样本 CSV（`label_1/2/3` + `final_label` + `n_agree` + `disagreement`，
                    既有 31 列列名与列序不动、`label_3` / `n_agree` 追加于末尾），再出结果；
                    一致性统计 = 逐对一致率 + Cohen's kappa、三方 Fleiss' kappa、`n_agree` 分布、
                    按层/列/年的不一致率热力表。原双人（label_1/label_2）路径保持兼容）

抽样（冻结）：框 = teacher 两列正式边表全集；文本年 2016/2019/2022（网络年 2017/2020/2023）；
5 层（high/mid/random/same_sw2/cross_sw2）× 40 条 × 2 列 × 3 年 = 1,200 条；
seed = registry random_seed 20260727（派生式见协议 §5）；层间优先级 high→mid→same_sw2→cross_sw2→random
（拒绝采样，剔除已抽项及其镜像对）。

用法：
  python scripts/network_edge_precision_sample.py                       # 抽样 + 工作表 + 模板
  python scripts/network_edge_precision_sample.py --compute \
      --annotated reports/data_gates/G0/s3_edge_precision_sample_20261013.csv  # 判读后统计
"""
from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

from network_edge_precision_audit import (          # noqa: E402  (同目录 v2 模块，复用同口径实现)
    build_ma_automaton, load_corpus, load_industry, load_lexicon_terms, load_names,
    match_terms_norm, mb_year_idf, normalize, _mb_tokenize,
)

# ---- 冻结常量（须与 registry `edge_precision_manual_annotation` 逐字一致） ---- #
SEED = 20260727                     # registry random_seed
TEXT_YEARS = [2016, 2019, 2022]
NETWORK_YEARS = {2016: 2017, 2019: 2020, 2022: 2023}
N_PER_STRATUM = 40
STRATUM_ORDER = ['high', 'mid', 'same_sw2', 'cross_sw2', 'random']
LABELS = ['direct_competitor', 'related_not_direct', 'spurious']
LABEL_CONSERVATISM = {l: i for i, l in enumerate(LABELS)}   # 保守度递增
THRESHOLDS = {'acceptable': 0.60, 'recheck': 0.40}
EXCERPT_TARGET = 300
EXCERPT_HALF = 150
MA_DF_MAX = 0.25
Z95 = 1.959964
CORPUS = 'teacher'
PROTOCOL = ROOT / 'reports/audits/s3_edge_precision_protocol_20261013.md'
NET_DIRS = {'ma': ROOT / 'data/processed/network_ma_teacher',
            'mb': ROOT / 'data/processed/network_mb_teacher'}
NET_PREFIX = {'ma': 'pm_ma_ewcos_teacher_v1', 'mb': 'pm_mb_tfidf_teacher_v1'}
LEXICON = ROOT / 'data/processed/product_lexicon_v2_teacher.csv'

CSV_COLUMNS = [
    'spec_id', 'network_year', 'text_year', 'network_version', 'stratum',
    'focal_code', 'focal_name', 'source_code', 'source_name',
    'focal_sw2', 'source_sw2', 'same_sw2', 'similarity', 'weight',
    'rank_in_focal_pct', 'n_shared', 'shared_evidence',
    'excerpt_a', 'excerpt_b', 'excerpt_a_anchor', 'excerpt_b_anchor',
    'corpus_a_status', 'corpus_b_status',
    'annotator_1', 'label_1', 'annotator_2', 'label_2', 'final_label',
    'disagreement', 'note_1', 'note_2',
]
# 「合并时扩列」约定（registry `edge_precision_manual_annotation` change_log 2026-10-13 修订）：
# 既有 31 列列名与列序不动，`label_3` / `n_agree` 由合并脚本追加于**末尾**。
MERGE_EXTRA_COLUMNS = ['label_3', 'n_agree']

# 判读证据层级（registry 1.5.9 / 协议修订节：如实降级，不得冒充人工金标准）
EVIDENCE_LEVEL = 'AI 独立判读（三代理，非人工）'
EVIDENCE_LEVEL_DISCLOSURE = ('3 个独立上下文 AI 判读代理（模型 deepseek-v4.1-flash，ZCode 执行场）各独立判读全部 '
                            '1,200 条（每条 3 个独立判读）；**非人工判读**，不得表述为「人工判读 / 双人独立判读 / 人工金标准」')
ANNOTATOR_IDS = ['A', 'B', 'C']
DEFAULT_LABEL_DIR = ROOT / 'artifacts/tmp/tier2_v5'
DEFAULT_LABEL_FILES = [DEFAULT_LABEL_DIR / f'labels_annotator_{a}.csv' for a in ANNOTATOR_IDS]

# 已知限制（M6；如实登记，判读后追加、不改任何冻结口径）
KNOWN_LIMITATIONS = [
    {
        'id': 'LIM1_stratum_industry_snapshot',
        'title': '抽样层的 focal_sw2 / source_sw2 为现行（2021 版）快照分类',
        'detail': ('样本 CSV 的 `focal_sw2` / `source_sw2` 取自 `data/processed/sw2_industry_latest.csv` 的'
                   '**静态当期（2021 版）申万二级**快照（冻结口径：不加 is_current_version 过滤、不按 '
                   'effective_from 做 PIT），与历史摘录年份（文本年 2016/2019/2022）的业务可能不符。'
                   'B、C 两代理均报告过此类行；接手轮自查确认**非样本错位**（行-代码-公司名与样本 CSV 逐行'
                   '对齐一致），系公司重组 / 更名（如房地产类历史业务）所致。')
    },
    {
        'id': 'LIM2_annotator_C_provenance',
        'title': 'C 代理标签文件的 provenance（终稿丢失的运行产出）',
        'detail': ('`labels_annotator_C.csv` 由一次**终稿丢失的运行**产出（provider 拒绝返回终稿文本）；'
                   '接手轮**未逐行复判**，只做了完整性校验（1,200 行、row_id 1..1200 各一次、标签合法、'
                   'reason 非空且 ≤15 字）与约 17 行抽检（含稀有三类样本，标签与摘录相符；含快照行业与'
                   '历史年份不符的行，按文本判读）。**C 的标签未经第二次逐行复核**，本结果按其在盘内容'
                   '原样采用并如实登记。')
    },
    {
        'id': 'LIM3_annotator_A_reason_len',
        'title': 'A 代理理由字段长度超出登记的 ≤15 字上限（69 条 = 16 字）',
        'detail': ('冻结修订登记要求每条判读理由 ≤15 字；A 的标签文件有 **69 条 = 16 字**（B / C 均为 ≤15 字）。'
                   '理由仍为简短判读依据、**标签不受影响**；合并脚本**未改写任何理由**（改写等同二次判读），'
                   '仅作偏离登记。')
    },
]
PROVENANCE_NOTES = [
    '判读轮次与模型：3 个独立上下文 AI 判读代理（A/B/C，模型 deepseek-v4.1-flash），各独立判读全部 1,200 条'
    '（每条 3 个独立判读、无子样本），互不可见对方标签。',
    '登记层级：按 registry 1.5.9 / 协议修订节如实登记为「AI 独立判读（三代理，非人工）」，**非人工金标准**。',
    '合并脚本：`scripts/network_edge_precision_sample.py --compute --labels A,B,C`（合并回填 → 一致性统计 →'
    '分层统计 → 判定 → 反例库）；`label_3` / `n_agree` 按「合并时扩列」约定追加于样本 CSV 末尾（既有 31 列列名'
    '与列序未动）。',
    'C 的判读理由（`note_3`）只落反例库（登记列集合不含 note_3）；A / B 的理由回填 `note_1` / `note_2`。',
    '门槛判定为**机械判定**（冻结阈值），证据层级随登记一并限定；解释性上下文见 `verdict_context`。',
]


# --------------------------------------------------------------------------- #
# 冻结前置校验
# --------------------------------------------------------------------------- #
def check_freeze_precedes_now() -> str:
    """校验协议冻结时间戳早于当前时间（V6 停机条件③）。返回冻结时间戳字符串。"""
    if not PROTOCOL.exists():
        raise SystemExit(f'[STOP] 协议文件缺失：{PROTOCOL}（门槛/层定义未先冻结就抽样 → 停机）')
    txt = PROTOCOL.read_text(encoding='utf-8')
    m = re.search(r'冻结时间戳[^0-9]*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)', txt)
    if not m:
        raise SystemExit('[STOP] 协议中未找到冻结时间戳（格式 YYYY-MM-DDTHH:MM:SSZ）→ 停机')
    frozen = m.group(1)
    fz = datetime.strptime(frozen, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    if not fz < now:
        raise SystemExit(f'[STOP] 冻结时间戳 {frozen} 不早于当前时间 {now.isoformat()} → 停机')
    print(f'[V3] 冻结前置校验 PASS：协议冻结 {frozen} < 抽样时刻 {now.strftime("%Y-%m-%dT%H:%M:%SZ")}')
    return frozen


def check_frames() -> None:
    """V6 停机条件①：teacher 两列边表齐备。"""
    missing = []
    for net, d in NET_DIRS.items():
        for y in TEXT_YEARS:
            f = d / f'{y}_similarity.csv'
            if not f.exists():
                missing.append(str(f))
    if missing:
        raise SystemExit(f'[STOP] teacher 两列边表缺失（V6 条件①）：{missing}')
    print(f'[V3] 边表齐备校验 PASS：{len(NET_DIRS)} 列 × {len(TEXT_YEARS)} 年 = '
          f'{len(NET_DIRS) * len(TEXT_YEARS)} 个边表')


# --------------------------------------------------------------------------- #
# 抽样
# --------------------------------------------------------------------------- #
def load_edges(net: str, year: int) -> pd.DataFrame:
    e = pd.read_csv(NET_DIRS[net] / f'{year}_similarity.csv',
                    usecols=['focal_id', 'source_id', 'similarity', 'weight'])
    e['focal_id'] = e['focal_id'].astype(str)
    e['source_id'] = e['source_id'].astype(str)
    return e


def build_frames(e: pd.DataFrame, ind: dict, net: str) -> dict:
    """五层抽样框（冻结定义）。返回 {层名: (frame_df, 判据说明)}。"""
    key = 'weight' if net == 'ma' else 'similarity'
    e = e.copy()
    e['pct'] = e.groupby('focal_id')[key].rank(pct=True)
    f_sw2 = e['focal_id'].map(ind)
    s_sw2 = e['source_id'].map(ind)
    known = f_sw2.notna() & s_sw2.notna()
    if net == 'ma':
        high = e[e['pct'] >= 0.95]
        mid = e[(e['pct'] >= 0.40) & (e['pct'] <= 0.60)]
        high_rule = 'same-focal weight rank(pct) >= 0.95'
        mid_rule = 'same-focal weight pct in [0.40, 0.60]'
    else:
        med = float(e['similarity'].median())
        s_pct = e['similarity'].rank(pct=True)
        high = e[e['similarity'] >= med * 1.5]
        mid = e[(s_pct >= 0.40) & (s_pct <= 0.60)]
        high_rule = f'network-level similarity >= median x 1.5 (median={med:.8f}, thr={med * 1.5:.8f})'
        mid_rule = 'network-level similarity pct in [0.40, 0.60]'
    return {
        'high': (high, high_rule),
        'mid': (mid, mid_rule),
        'same_sw2': (e[known & (f_sw2 == s_sw2)], 'both endpoints mapped and same l2_code'),
        'cross_sw2': (e[known & (f_sw2 != s_sw2)], 'both endpoints mapped and different l2_code'),
        'random': (e, 'uniform over full frame (all directed edges)'),
    }


def draw(e: pd.DataFrame, ind: dict, net: str, year: int, n_target: int) -> tuple:
    """按冻结优先级 + 拒绝采样抽 (层, 边)。返回 (rows, frame_counts, shortfall, rules)。"""
    rng = np.random.default_rng(SEED + 1000 * year + (0 if net == 'ma' else 1))
    frames = build_frames(e, ind, net)
    frame_counts = {k: int(len(v[0])) for k, v in frames.items()}
    rules = {k: v[1] for k, v in frames.items()}
    taken_ids, taken_pairs, rows, shortfall = set(), set(), [], {}
    for st in STRATUM_ORDER:
        frame, rule = frames[st]
        picked = []
        if len(frame):
            for i in rng.permutation(len(frame)):
                r = frame.iloc[int(i)]
                fid, sid = str(r['focal_id']), str(r['source_id'])
                pair = tuple(sorted((fid, sid)))
                if (fid, sid) in taken_ids or pair in taken_pairs:
                    continue
                picked.append((fid, sid, float(r['similarity']), float(r['weight']),
                               float(r['pct'])))
                taken_ids.add((fid, sid))
                taken_pairs.add(pair)
                if len(picked) >= n_target:
                    break
        if len(picked) < n_target:
            shortfall[st] = {'target': n_target, 'drawn': len(picked),
                             'frame_size': int(len(frame))}
        for fid, sid, sim, w, pct in picked:
            rows.append({'spec_id': net, 'stratum': st, 'focal_code': fid, 'source_code': sid,
                         'similarity': sim, 'weight': w, 'rank_in_focal_pct': round(pct, 6),
                         'text_year': year, 'network_year': NETWORK_YEARS[year],
                         'network_version': f'{NET_PREFIX[net]}_{year}',
                         'stratum_rule': rule})
    return rows, frame_counts, shortfall, rules


# --------------------------------------------------------------------------- #
# 证据词与摘录
# --------------------------------------------------------------------------- #
def find_span_raw(text: str, term: str):
    """在原始文本中定位词条（字符间允许空白、忽略大小写；匹配口径为 NFKC+去空白+小写）。"""
    pat = r'\s*'.join(re.escape(c) for c in term)
    m = re.search(pat, text, re.IGNORECASE)
    return (m.start(), m.end()) if m else None


def make_excerpt(text: str, anchors: list, finder) -> tuple:
    """按冻结摘录规则取 ~300 字窗口，返回 (摘录, 锚点说明)。"""
    for a in anchors:
        span = finder(text, a)
        if span is None:
            continue
        center = (span[0] + span[1]) // 2
        start = max(0, center - EXCERPT_HALF)
        end = min(len(text), start + EXCERPT_TARGET)
        start = max(0, end - EXCERPT_TARGET)
        return text[start:end], f'{a}@{span[0]}'
    return text[:EXCERPT_TARGET], 'head_300'


def evidence_and_excerpt(net: str, ta: str, tb: str, automaton, df_map: dict,
                         idf_map: dict, jieba, stopwords, token_re) -> dict:
    if net == 'ma':
        ha = match_terms_norm(normalize(ta), automaton)
        hb = match_terms_norm(normalize(tb), automaton)
        shared = sorted(ha & hb, key=lambda t: (df_map.get(t, 1.0), t))[:20]
        anchors_a = shared if shared else sorted(ha, key=lambda t: (df_map.get(t, 1.0), t))[:20]
        anchors_b = shared if shared else sorted(hb, key=lambda t: (df_map.get(t, 1.0), t))[:20]
        ea, aa = make_excerpt(ta, anchors_a, find_span_raw)
        eb, ab = make_excerpt(tb, anchors_b, find_span_raw)
    else:
        sa = set(_mb_tokenize(ta, jieba, stopwords, token_re))
        sb = set(_mb_tokenize(tb, jieba, stopwords, token_re))
        shared = [w for w in (sa & sb) if w in idf_map]
        shared.sort(key=lambda w: (-idf_map[w], w))
        shared = shared[:20]
        ea, aa = make_excerpt(ta, shared, lambda t, w: (
            (t.find(w), t.find(w) + len(w)) if t.find(w) >= 0 else None))
        eb, ab = make_excerpt(tb, shared, lambda t, w: (
            (t.find(w), t.find(w) + len(w)) if t.find(w) >= 0 else None))
    return {'n_shared': len(shared), 'shared_evidence': '|'.join(shared),
            'excerpt_a': ea, 'excerpt_b': eb,
            'excerpt_a_anchor': aa, 'excerpt_b_anchor': ab}


# --------------------------------------------------------------------------- #
# 工作表（HTML，人读版；不做任何 AI 预标注）
# --------------------------------------------------------------------------- #
def build_worksheet(df: pd.DataFrame, path: Path, meta: dict) -> None:
    parts = ["""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>S3 边级判读工作表（V5 / B″11-9 路线 A）</title>
<style>
body{font-family:"Microsoft YaHei",system-ui,sans-serif;margin:24px;background:#fafafa;color:#222}
h1{font-size:20px}h2{font-size:17px;margin-top:28px;border-bottom:2px solid #ccd;padding-bottom:4px}
h3{font-size:15px;margin-top:20px;color:#345}
.card{background:#fff;border:1px solid #ddd;border-radius:6px;padding:12px;margin:10px 0;page-break-inside:avoid}
.hdr{font-weight:600;font-size:14px;margin-bottom:6px}
.meta{font-size:12px;color:#555;margin-bottom:6px}
.ex{white-space:pre-wrap;font-size:13px;line-height:1.6;background:#f7f9fb;border-left:3px solid #9bc;padding:8px;margin:6px 0;max-height:260px;overflow:auto}
.mark{background:#ffe9a8}
.ev{font-size:12px;color:#345;word-break:break-all}
.lab{font-size:12px;color:#444;background:#f0f4f8;padding:6px;border-radius:4px;margin-top:6px}
.note{font-size:12px;color:#666}
table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccc;padding:3px 6px}
</style></head><body>"""]
    parts.append('<h1>S3 边级 precision 人工判读工作表（V5 / B″11-9 路线 A）</h1>')
    parts.append('<p class="note">本工作表由 <code>network_edge_precision_sample.py</code> 自动生成，'
                 '内容只有「公司/行业/权重/证据词/两端 MD&A 摘录」——<b>不含任何 AI 预标注、不含任何建议标签</b>。'
                 '判读标签定义、门槛与裁定规则见协议 '
                 '<code>reports/audits/s3_edge_precision_protocol_20261013.md</code>。</p>')
    parts.append('<div class="lab"><b>三标签勾选说明</b>：'
                 '<b>direct_competitor</b> = 同一产品/服务市场可相互替代（有明确共同产品线或共同客户群）；'
                 '<b>related_not_direct</b> = 同一产业链上下游 / 同集团生态 / 仅共享行业通用词但无直接替代；'
                 '<b>spurious</b> = 明显误连（只共享「公司/发展/经营」类通用词，或文本误提、口径错位）。'
                 '两人各自独立判读、不互看；不一致时按<b>保守取较低标签</b>裁定。'
                 '判读结果回填到样本 CSV 的 <code>label_1</code> / <code>label_2</code> 列（见协议 §10）。</div>')
    parts.append(f'<p class="meta">样本文件：<code>{meta["sample_csv"]}</code>　'
                 f'生成时间（UTC）：{meta["generated_at"]}　协议冻结：{meta["frozen_at"]}　'
                 f'seed={SEED}　总条数：{len(df)}</p>')
    for (spec, ty), g in df.groupby(['spec_id', 'text_year'], sort=True):
        ny = int(g['network_year'].iloc[0])
        parts.append(f'<h2>{spec.upper()}　文本年 {ty}（网络年 {ny}，'
                     f'{g["network_version"].iloc[0]}）　共 {len(g)} 条</h2>')
        for st in STRATUM_ORDER:
            sub = g[g['stratum'] == st]
            if sub.empty:
                continue
            parts.append(f'<h3>层 {st}（{len(sub)} 条）</h3>')
            for k, r in enumerate(sub.itertuples(), 1):
                ev = [w for w in str(r.shared_evidence).split('|') if w]
                ea = html.escape(str(r.excerpt_a))
                eb = html.escape(str(r.excerpt_b))
                for w in sorted(ev, key=len, reverse=True):
                    for variant in {w, html.escape(w)}:
                        ea = ea.replace(variant, f'<span class="mark">{variant}</span>')
                        eb = eb.replace(variant, f'<span class="mark">{variant}</span>')
                parts.append(
                    f'<div class="card"><div class="hdr">[{spec} {ty} {st} #{k}] '
                    f'{r.focal_name}（{r.focal_code}） ↔ {r.source_name}（{r.source_code}）</div>'
                    f'<div class="meta">申万二级：{r.focal_sw2} ↔ {r.source_sw2}'
                    f'（same_sw2={r.same_sw2}）　similarity={r.similarity:.6f}　weight={r.weight:.6f}'
                    f'　focal 内分位={r.rank_in_focal_pct}　证据词数={r.n_shared}</div>'
                    f'<div class="ev"><b>证据词</b>：{html.escape(str(r.shared_evidence))}</div>'
                    f'<div class="ex"><b>{r.focal_name}（{r.focal_code}）摘录</b> '
                    f'<span class="note">[{r.excerpt_a_anchor}]</span>\n{ea}</div>'
                    f'<div class="ex"><b>{r.source_name}（{r.source_code}）摘录</b> '
                    f'<span class="note">[{r.excerpt_b_anchor}]</span>\n{eb}</div>'
                    f'<div class="lab">判读：A ☐ direct_competitor ☐ related_not_direct ☐ spurious　　'
                    f'B ☐ direct_competitor ☐ related_not_direct ☐ spurious　　'
                    f'（spurious 请写原因）</div></div>')
    parts.append('</body></html>')
    path.write_text(''.join(parts), encoding='utf-8')


# --------------------------------------------------------------------------- #
# 三代理标签合并 + 一致性统计（M1–M3）
# --------------------------------------------------------------------------- #
def read_label_file(path: Path, n_expect: int) -> tuple:
    """读单个判读代理标签文件（row_id,label,reason）并做完整性校验（M1）。

    校验：行数 == n_expect、row_id 恰为 1..n 各一次、label ∈ 三标签、reason 非空。
    任一不符 → [STOP]。返回 (labels, reasons, report)。
    """
    if not Path(path).exists():
        raise SystemExit(f'[STOP] 标签文件缺失：{path}')
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in ('row_id', 'label', 'reason'):
        if c not in df.columns:
            raise SystemExit(f'[STOP] 标签文件缺列 {c}：{path}')
    ids = df['row_id'].str.strip().astype(int)
    lab = df['label'].str.strip()
    rsn = df['reason'].str.strip()
    checks = {
        'n_rows': int(len(df)),
        'row_id_exact_1..n': sorted(ids.tolist()) == list(range(1, n_expect + 1)),
        'row_id_duplicated': int(ids.duplicated().sum()),
        'labels_valid': sorted(set(lab.unique())) == sorted(LABELS),
        'unexpected_labels': sorted(set(lab.unique()) - set(LABELS)),
        'reason_all_nonempty': bool((rsn != '').all()),
        'reason_max_len': int(rsn.str.len().max()),
    }
    fail = (checks['n_rows'] != n_expect or not checks['row_id_exact_1..n']
            or checks['row_id_duplicated'] or checks['unexpected_labels']
            or not checks['reason_all_nonempty'])
    print(f'[M1] {Path(path).name}: rows={checks["n_rows"]}/{n_expect} '
          f'row_id_1..n={checks["row_id_exact_1..n"]} dup={checks["row_id_duplicated"]} '
          f'labels_ok={checks["labels_valid"]} reason_nonempty={checks["reason_all_nonempty"]} '
          f'reason_max_len={checks["reason_max_len"]} -> {"PASS" if not fail else "FAIL"}')
    if fail:
        raise SystemExit(f'[STOP] 标签文件完整性校验失败（M1）：{path} -> {checks}')
    report = {
        'file': str(Path(path).as_posix()),
        'sha256': sha256(Path(path).read_bytes()).hexdigest(),
        'distribution': {k: int(v) for k, v in lab.value_counts().items()},
        'reason_max_len': checks['reason_max_len'],
        'checks': checks,
    }
    return (dict(zip(ids.tolist(), lab.tolist())),
            dict(zip(ids.tolist(), rsn.tolist())), report)


def decide_final(labels: list) -> tuple:
    """冻结裁定规则（registry change_log 2026-10-13 修订 2）→ (final_label, n_agree)。"""
    if any(x == '' for x in labels):
        return 'NOT_ANNOTATED', 0
    cnt = Counter(labels)
    top = cnt.most_common(1)[0]
    if top[1] == len(labels):                       # 三人一致
        return labels[0], 3
    if top[1] == len(labels) - 1:                   # 两人一致 → 多数
        return top[0], 2
    # 1:1:1 无多数 → 保守取较低标签
    return max(labels, key=lambda x: LABEL_CONSERVATISM[x]), 1


def merge_labels(sample_csv: Path, label_files: list) -> dict:
    """把 N 位代理标签合并回填样本 CSV（M3）：label_1/2/3 + note_1/2 + final_label + n_agree
    + disagreement。既有列名与列序不动，`label_3` / `n_agree` 追加于末尾（登记约定）。"""
    df = pd.read_csv(sample_csv, dtype=str, keep_default_na=False)
    n = len(df)
    maps, notes, reports = [], [], []
    for p in label_files:
        m, r, rep = read_label_file(Path(p), n)
        maps.append(m)
        notes.append(r)
        reports.append(rep)
    labels_by_row = [[maps[k][i] for k in range(len(maps))] for i in range(1, n + 1)]
    fin = [decide_final(ls) for ls in labels_by_row]
    df['label_1'] = [labels_by_row[i][0] for i in range(n)]
    df['label_2'] = [labels_by_row[i][1] for i in range(n)]
    df['label_3'] = [labels_by_row[i][2] for i in range(n)]
    df['note_1'] = [notes[0][i] for i in range(1, n + 1)]
    df['note_2'] = [notes[1][i] for i in range(1, n + 1)]
    df['annotator_1'] = ANNOTATOR_IDS[0]
    df['annotator_2'] = ANNOTATOR_IDS[1]
    df['final_label'] = [f for f, _ in fin]
    df['n_agree'] = [str(k) for _, k in fin]
    df['disagreement'] = [('0' if k == 3 else '1') for _, k in fin]
    df = df.reindex(columns=CSV_COLUMNS + MERGE_EXTRA_COLUMNS)
    df.to_csv(sample_csv, index=False, encoding='utf-8-sig')
    nag = Counter(k for _, k in fin)
    mode_share = Counter(f for f, _ in fin)
    print(f'[M3] 合并回填完成：{sample_csv.name} -> {len(df)} 行 × {len(df.columns)} 列'
          f'（+{",".join(MERGE_EXTRA_COLUMNS)}）')
    print(f'[M3] n_agree 分布：3={nag[3]} 2={nag[2]} 1={nag[1]}；'
          f'final_label 分布：{dict(mode_share)}')
    return {'label_files': reports, 'n_rows': n, 'n_columns_after_merge': int(len(df.columns)),
            'n_agree_distribution': {str(k): int(nag[k]) for k in (3, 2, 1)},
            'final_label_distribution': {k: int(v) for k, v in mode_share.items()},
            'merged_csv_sha256': sha256(sample_csv.read_bytes()).hexdigest(),
            '_reasons': notes}


def cohen_kappa(a: list, b: list) -> dict:
    """两位判读者一致率 + Cohen's kappa（3 类别）。"""
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((sum(1 for x in a if x == c) / n) * (sum(1 for y in b if y == c) / n) for c in LABELS)
    k = (po - pe) / (1 - pe) if pe < 1 else None
    return {'n': n, 'agreement_rate': round(po, 6), 'p_expected': round(pe, 6),
            'kappa': round(k, 6) if k is not None else None}


def fleiss_kappa(rows_labels: list) -> dict:
    """三方（固定 3 判读者）Fleiss' kappa（3 类别）。"""
    n_items = len(rows_labels)
    rats = len(rows_labels[0])
    counts = [[sum(1 for x in r if x == lab) for lab in LABELS] for r in rows_labels]
    p_i = [sum(c * c for c in cc) - rats for cc in counts]
    p_bar = sum(p_i) / (n_items * rats * (rats - 1))
    p_j = [sum(cc[j] for cc in counts) / (n_items * rats) for j in range(len(LABELS))]
    pe = sum(p * p for p in p_j)
    k = (p_bar - pe) / (1 - pe) if pe < 1 else None
    return {'n_items': n_items, 'n_raters': rats, 'observed_agreement_p_bar': round(p_bar, 6),
            'expected_agreement_pe': round(pe, 6),
            'fleiss_kappa': round(k, 6) if k is not None else None,
            'category_marginals': {lab: round(p, 6) for lab, p in zip(LABELS, p_j)}}


def disagreement_tables(df: pd.DataFrame) -> dict:
    """不一致率热力表（摘要）：按层 / 列 / 年 及其交叉。"""
    def cell(sub: pd.DataFrame) -> dict:
        n = len(sub)
        d = int((sub['disagreement'] == '1').sum())
        return {'n': int(n), 'n_disagree': d, 'rate': round(d / n, 6) if n else None}

    def table(keys: list) -> dict:
        out = {}
        for kv, sub in df.groupby(keys, sort=True):
            kv = kv if isinstance(kv, tuple) else (kv,)
            out['|'.join(str(x) for x in kv)] = cell(sub)
        return out
    return {'by_stratum': table(['stratum']), 'by_spec': table(['spec_id']),
            'by_text_year': table(['text_year']), 'by_spec_x_text_year': table(['spec_id', 'text_year']),
            'by_stratum_x_spec': table(['stratum', 'spec_id']),
            'by_stratum_x_text_year': table(['stratum', 'text_year'])}


# --------------------------------------------------------------------------- #
# 统计（--compute）
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int) -> tuple:
    if n == 0:
        return (None, None, None)
    p = k / n
    denom = 1 + Z95 ** 2 / n
    center = (p + Z95 ** 2 / (2 * n)) / denom
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95 ** 2 / (4 * n * n)) / denom
    return (round(p, 6), round(center - half, 6), round(center + half, 6))


def pooled_estimate(rec: pd.DataFrame, n_same: int, n_cross: int,
                    category: str = 'direct_competitor') -> dict:
    """按 plan §5.7A P5.2 公式：层 = same_sw2 / cross_sw2 二分加权池化。"""
    cells, num, var = [], 0.0, 0.0
    N = n_same + n_cross
    for st, Nh in (('same_sw2', n_same), ('cross_sw2', n_cross)):
        sub = rec[rec['stratum'] == st]
        nh = len(sub)
        ph = float((sub['final_label'] == category).mean()) if nh else None
        cells.append({'stratum': st, 'N_h': Nh, 'n_h': nh, 'p_h': ph})
        if nh and Nh:
            num += Nh * ph
            var += (Nh / N) ** 2 * ph * (1 - ph) / max(1, nh - 1)
    if not N:
        return {'precision': None, 'se': None, 'ci95': [None, None], 'cells': cells}
    prec = num / N
    se = math.sqrt(var)
    return {'precision': round(prec, 6), 'se': round(se, 6),
            'ci95': [round(prec - 1.96 * se, 6), round(prec + 1.96 * se, 6)], 'cells': cells}


def verdict(p: float | None) -> str:
    if p is None:
        return 'NOT_COMPUTABLE'
    if p >= THRESHOLDS['acceptable']:
        return 'ACCEPTABLE'
    if p < THRESHOLDS['recheck']:
        return 'TRIGGER_NETWORK_RECHECK'
    return 'CONDITIONAL'


def compute(annotated: Path, out_dir: Path, out_prefix: str, label_files=None) -> int:
    merge_report = None
    if label_files:
        print(f'[M3] 合并三位代理标签 → {annotated}')
        merge_report = merge_labels(Path(annotated), label_files)
    df = pd.read_csv(annotated, dtype=str, keep_default_na=False).fillna('')

    label_cols = [c for c in ('label_1', 'label_2', 'label_3') if c in df.columns]
    for col in label_cols:
        df[col] = df[col].str.strip()
        bad = df[col][(df[col] != '') & (~df[col].isin(LABELS))]
        if len(bad):
            raise SystemExit(f'[STOP] {col} 含非法标签（只允许 {LABELS}）：{sorted(set(bad))[:5]}')
    triple = ('label_3' in df.columns) and bool((df['label_3'] != '').all())
    mode = 'triple_ai_annotation' if triple else 'double_annotation_legacy'
    unfilled = df[(df['label_1'] == '') | (df['label_2'] == '')]
    if len(unfilled):
        print(f'[WARN] 有 {len(unfilled)}/{len(df)} 条未填满 → final_label 记为 NOT_ANNOTATED，不计入 p_h',
              flush=True)

    final, dis, nagree = [], [], []
    for i in range(len(df)):
        ls = [df['label_1'].iloc[i], df['label_2'].iloc[i]] + \
             ([df['label_3'].iloc[i]] if triple else [])
        f, k = decide_final(ls)
        final.append(f)
        nagree.append(k)
        dis.append('' if f == 'NOT_ANNOTATED' else ('0' if k == len(ls) else '1'))
    df['final_label'] = final
    df['disagreement'] = dis
    if 'n_agree' in df.columns:
        df['n_agree'] = [str(k) for k in nagree]
    scored = df[df['final_label'].isin(LABELS)].copy()
    print(f'[M3] 裁定完成（{mode}）：scored={len(scored)}/{len(df)}，'
          f'disagreement={int((df["disagreement"] == "1").sum())}，'
          f'n_agree 分布={dict(Counter(nagree))}')

    # ---- M2 一致性统计（三代理） ---- #
    consistency = None
    if triple:
        A, B, C = df['label_1'].tolist(), df['label_2'].tolist(), df['label_3'].tolist()
        pair = {'A_B': cohen_kappa(A, B), 'A_C': cohen_kappa(A, C), 'B_C': cohen_kappa(B, C)}
        fk = fleiss_kappa([[a, b, c] for a, b, c in zip(A, B, C)])
        nag = Counter(nagree)
        consistency = {
            'mode': mode, 'n_items': int(len(df)), 'n_raters': 3,
            'pairwise': pair, 'fleiss_kappa': fk,
            'n_agree_distribution': {str(k): int(nag[k]) for k in (3, 2, 1)},
            'unanimous_rate': round(nag[3] / len(df), 6),
            'majority_rate': round(nag[2] / len(df), 6),
            'no_majority_rate': round(nag[1] / len(df), 6),
            'disagreement_tables': disagreement_tables(df),
            'annotator_distributions': {
                'A': {k: int(v) for k, v in Counter(A).items()},
                'B': {k: int(v) for k, v in Counter(B).items()},
                'C': {k: int(v) for k, v in Counter(C).items()}},
        }
        print(f'[M2] 逐对一致率/kappa：'
              + '；'.join(f'{k}={v["agreement_rate"]:.4f}/{v["kappa"]:.4f}' for k, v in pair.items()))
        print(f'[M2] Fleiss kappa={fk["fleiss_kappa"]:.4f}（P_bar={fk["observed_agreement_p_bar"]:.4f}）'
              f'　n_agree 3/2/1 = {nag[3]}/{nag[2]}/{nag[1]}')

    # ---- 框计数（用于 N_h） ---- #
    ind = load_industry()
    frame_counts = {}
    for spec in ('ma', 'mb'):
        for ty in TEXT_YEARS:
            e = load_edges(spec, int(ty))
            f_sw2 = e['focal_id'].map(ind)
            s_sw2 = e['source_id'].map(ind)
            known = f_sw2.notna() & s_sw2.notna()
            frame_counts[(spec, int(ty))] = {
                'same_sw2': int((known & (f_sw2 == s_sw2)).sum()),
                'cross_sw2': int((known & (f_sw2 != s_sw2)).sum()),
                'full': int(len(e)),
            }
            print(f'[V5] frame {spec} {ty}: N_same={frame_counts[(spec, int(ty))]["same_sw2"]} '
                  f'N_cross={frame_counts[(spec, int(ty))]["cross_sw2"]}', flush=True)

    def layer_block(sub: pd.DataFrame) -> list:
        out = []
        for st in STRATUM_ORDER:
            s = sub[sub['stratum'] == st]
            if not len(s):
                continue
            n = len(s)
            k = int((s['final_label'] == 'direct_competitor').sum())
            n_rel = int((s['final_label'] == 'related_not_direct').sum())
            n_sp = int((s['final_label'] == 'spurious').sum())
            p, lo, hi = wilson(k, n)
            out.append({'stratum': st, 'n': n, 'k_direct': k, 'p_direct': p,
                        'wilson_ci95': [lo, hi],
                        'n_related_not_direct': n_rel, 'n_spurious': n_sp,
                        'n_not_annotated': int((s['final_label'] == 'NOT_ANNOTATED').sum()),
                        'n_disagreement': int((s['disagreement'] == '1').sum()),
                        'n_agree_3': int((s['n_agree'] == '3').sum()) if 'n_agree' in s.columns else None,
                        # 次要读数（并列报告，不改判据）
                        'p_related_or_direct': round((k + n_rel) / n, 6),
                        'p_spurious': round(n_sp / n, 6)})
        return out

    per_year, per_spec = {}, {}
    for spec in ('ma', 'mb'):
        sub_spec = scored[scored['spec_id'] == spec]
        n_same = sum(frame_counts[(spec, y)]['same_sw2'] for y in TEXT_YEARS)
        n_cross = sum(frame_counts[(spec, y)]['cross_sw2'] for y in TEXT_YEARS)
        pooled = pooled_estimate(sub_spec, n_same, n_cross)
        per_spec[spec] = {
            'role': 'headline_primary (M-A 中文法 × 老师 MD&A)' if spec == 'ma' else 'method_robustness (M-B TF-IDF)',
            'layers': layer_block(sub_spec),
            'pooled_weighted': pooled,
            'pooled_weighted_secondary': {
                'p_related_or_direct': pooled_estimate(sub_spec, n_same, n_cross, 'related_not_direct')['precision'],
                'p_spurious': pooled_estimate(sub_spec, n_same, n_cross, 'spurious')['precision']},
        }
        rnd = sub_spec[sub_spec['stratum'] == 'random']
        k = int((rnd['final_label'] == 'direct_competitor').sum())
        per_spec[spec]['random_layer_unbiased_overall'] = dict(
            zip(['p_direct', 'wilson_lo', 'wilson_hi'], wilson(k, len(rnd))), n=int(len(rnd)))
        hi = sub_spec[sub_spec['stratum'] == 'high']
        p, lo, hi_ci = wilson(int((hi['final_label'] == 'direct_competitor').sum()), len(hi))
        per_spec[spec]['high_layer_overall'] = {'n': int(len(hi)), 'p_direct': p,
                                                'wilson_ci95': [lo, hi_ci]}
        per_spec[spec]['verdict_weighted'] = verdict(pooled['precision'])
        per_spec[spec]['verdict_random_layer'] = verdict(
            per_spec[spec]['random_layer_unbiased_overall']['p_direct'])
        for ty in TEXT_YEARS:
            sub_y = sub_spec[sub_spec['text_year'] == str(ty)]
            pl = pooled_estimate(sub_y, frame_counts[(spec, ty)]['same_sw2'],
                                 frame_counts[(spec, ty)]['cross_sw2'])
            per_year.setdefault(spec, {})[str(ty)] = {
                'layers': layer_block(sub_y), 'pooled_weighted': pl,
                'verdict_weighted': verdict(pl['precision']),
                'pooled_weighted_secondary': {
                    'p_related_or_direct': pooled_estimate(
                        sub_y, frame_counts[(spec, ty)]['same_sw2'],
                        frame_counts[(spec, ty)]['cross_sw2'], 'related_not_direct')['precision'],
                    'p_spurious': pooled_estimate(
                        sub_y, frame_counts[(spec, ty)]['same_sw2'],
                        frame_counts[(spec, ty)]['cross_sw2'], 'spurious')['precision']}}

    # ---- 门槛判定（M5；登记口径 = 按列 same/cross 二分加权池化 p_direct） ---- #
    gate_verdict = {
        'rule': 'p_direct >= 0.60 ACCEPTABLE / < 0.40 TRIGGER_NETWORK_RECHECK / else CONDITIONAL '
                '（registry `edge_precision_manual_annotation`.thresholds，抽前冻结，未改）',
        'headline_readout': 'M-A（主口径）same_sw2/cross_sw2 二分加权池化 p_direct',
        'headline_verdict': per_spec['ma']['verdict_weighted'],
        'per_spec': {s: {'readout': 'same_sw2/cross_sw2 加权池化 p_direct',
                         'p_direct': per_spec[s]['pooled_weighted']['precision'],
                         'ci95': per_spec[s]['pooled_weighted']['ci95'],
                         'verdict': per_spec[s]['verdict_weighted'],
                         'random_layer_control': per_spec[s]['random_layer_unbiased_overall'],
                         'verdict_random_layer': per_spec[s]['verdict_random_layer']}
                     for s in ('ma', 'mb')},
        'per_year': {s: {y: {'p_direct': per_year[s][y]['pooled_weighted']['precision'],
                             'verdict': per_year[s][y]['verdict_weighted']}
                         for y in per_year[s]} for s in ('ma', 'mb')},
    }
    ctx = ('解释性上下文（不改判据、不放水）：本网络是**产品相似度网络（TNIC 式，非行业分类网络）**——'
           '边 = 两端 MD&A 文本产品词/共享 token 相似度超阈；`direct_competitor` 是**严格标签**'
           '（须同一产品/服务市场可相互替代、有明确共同产品线或共同客户群）；`related_not_direct`'
           '（产业链上下游 / 同生态 / 仅共享行业通用词）**亦属真实的产品市场关联**（对下游 peer 溢出'
           '仍有经济含义），只有 `spurious` 才是噪声。因此 `p_direct` 是**语义强度的下界式读数**：'
           '它把「真实但不直接替代」的关联计入非 direct，判据口径偏保守（设计张力登记：被冻结的阈值'
           '以 direct 为唯一判据，而网络的经济用途不限于直接替代）。此上下文**不改变**冻结判定的机械结果；'
           '报告引用该读数时必须同时披露证据层级 = 「AI 独立判读（三代理，非人工）」。')

    shell = {}
    res_path = out_dir / f'{out_prefix}_result.json'
    if res_path.exists():
        try:
            shell = json.loads(res_path.read_text(encoding='utf-8'))
        except Exception:
            shell = {}
    keep = ['protocol_file', 'protocol_freeze_timestamp', 'sample_generated_at',
            'freeze_precedes_sample', 'registry_parameter', 'seed', 'seed_rule',
            'rotation_years_text_year', 'rotation_years_network_year', 'n_per_stratum',
            'stratum_counts', 'shortfall', 'stratum_rules', 'label_order']
    result = {k: shell[k] for k in keep if k in shell}
    result.update({
        'artifact': 's3_edge_precision_result',
        'status': 'COMPUTED_AWAITING_REVIEW',
        'previous_status': shell.get('status', 'AWAITING_HUMAN_ANNOTATION'),
        'status_note': ('判读已完成（3 个独立 AI 判读代理 × 1,200 条）；结果为分层 precision + Wilson CI95 + '
                        'same/cross 二分加权池化 + random 层无偏对照 + 门槛判定。**证据层级 = '
                        '「AI 独立判读（三代理，非人工）」**；「pre-validated / 已验证」仍禁用（判读完成 ≠ '
                        'pre-validated；外部锚仍未满足）。'),
        'computed_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'annotated_file': str(annotated.as_posix()),
        'annotated_file_sha256': sha256(Path(annotated).read_bytes()).hexdigest(),
        'n_rows_total': int(len(df)), 'n_rows_scored': int(len(scored)),
        'n_disagreement': int((df['disagreement'] == '1').sum()),
        'agreement_rate': round(float((df['disagreement'] == '0').mean()), 6),
        'annotation_mode': mode,
        'annotation_evidence_level': EVIDENCE_LEVEL,
        'annotation_evidence_level_disclosure': EVIDENCE_LEVEL_DISCLOSURE,
        'annotation_sources': merge_report['label_files'] if merge_report else None,
        'merge': ({k: v for k, v in merge_report.items() if not k.startswith('_') and k != 'label_files'}
                  if merge_report else None),
        'thresholds': THRESHOLDS, 'labels': LABELS,
        'decision_rule': ('triple-annotator conservative rule: unanimous -> label; 2/3 -> majority; '
                          '1:1:1 -> conservative lowest (direct_competitor > related_not_direct > spurious); '
                          'register n_agree (1/2/3) + disagreement (1/0)'),
        'estimation_rule': ('per-layer Wilson CI95; pooled weighted = same_sw2/cross_sw2 partition '
                            '(plan 5.7A P5.2); random layer = unbiased overall; no cross-column pooling; '
                            'p_related_or_direct / p_spurious = secondary readouts only'),
        'frame_counts': {f'{k[0]}_{k[1]}': v for k, v in frame_counts.items()},
        'consistency': consistency,
        'per_spec': per_spec, 'per_spec_year': per_year,
        'gate_verdict': gate_verdict,
        'verdict_context': ctx,
        # Phase 0.3（2026-10-27）：N-15 κ 语义并排注记（append-only；不改任何数值/算法）
        'semantics': {
            'fleiss_kappa': 0.809146,
            'fleiss_kappa_meaning': 'inter-rater agreement among three independent AI annotators using '
                                    'the same model (deepseek-v4.1-flash), 1,200 edges; '
                                    'NOT accuracy and NOT edge precision',
            'pairwise_cohen_kappa': {'A_B': 0.8274, 'A_C': 0.808621, 'B_C': 0.791338},
            'accuracy_readout': {
                'p_direct_ma_pooled_weighted': 0.170798,
                'ci95': [0.12974, 0.211856],
                'random_layer_control': 0.2,
                'mb_p_direct': 0.239184,
            },
            'kappa_to_p_direct_ratio': 4.738,
            'disclosure': 'three independent-context AI annotators; same model; non-human; '
                          'LIM2 (annotator C provenance) applies; pre-validated remains blocked',
            'citation_rule': '任何 κ 出现处必须并排写 p_direct = 0.170798（CI95 0.129740–0.211856）与「三代理同模型、非人工」；'
                             '不得写成「边精度 0.81 / precision = 0.81」',
        },
        'known_limitations': KNOWN_LIMITATIONS,
        'provenance_notes': PROVENANCE_NOTES,
        'wording_constraints': {
            'evidence_level': EVIDENCE_LEVEL,
            'forbidden': ['人工判读', '双人独立判读', '人工金标准', 'human double-annotation',
                          'human gold standard'],
            'allowed': ['三轮独立判读（AI 代理，独立上下文）+ 一致性检验'],
            'must_disclose': '判读者为 AI 代理及其模型 deepseek-v4.1-flash',
            'pre_validated_blocked': True,
            'pre_validated_reason': ('判读完成 ≠ pre-validated：外部锚（全量招股书自报锚 / 分析师共同覆盖）'
                                     '仍未满足 → 论文仍禁用「网络已验证 / pre-validated / precision = x」措辞'),
            # Phase 0.3（2026-10-27）：N-15——κ 是一致性，不是准确度
            'kappa_semantics': 'Fleiss κ=0.809146 = agreement among three same-model AI annotators, '
                               'not accuracy; accuracy readout = p_direct 0.170798 '
                               '(CI95 0.129740–0.211856); κ/p_direct = 4.738',
        },
    })
    res_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')

    # ---- 反例库（登记口径 = spurious + related_not_direct 全量；含三方标签与理由） ---- #
    ce = df[df['final_label'].isin(['spurious', 'related_not_direct'])].copy()
    ce_cols = ['spec_id', 'text_year', 'network_year', 'stratum', 'focal_code', 'focal_name',
               'source_code', 'source_name', 'focal_sw2', 'source_sw2', 'same_sw2', 'similarity',
               'weight', 'n_shared', 'shared_evidence', 'final_label', 'n_agree',
               'label_1', 'label_2', 'label_3', 'disagreement',
               'note_1', 'note_2', 'excerpt_a', 'excerpt_b']
    if triple and merge_report:
        note3 = pd.Series([merge_report['_reasons'][2][i] for i in range(1, len(df) + 1)],
                          index=df.index)
        ce['note_3'] = note3.loc[ce.index]
        ce_cols.append('note_3')
    ce = ce.reindex(columns=ce_cols)
    ce.to_csv(out_dir / f'{out_prefix}_counterexamples.csv', index=False)
    n_sp = int((ce['final_label'] == 'spurious').sum())
    print(f'[M4] counterexamples（登记口径 = spurious + related_not_direct 全量）: {len(ce)} 条'
          f'（spurious {n_sp} / related_not_direct {len(ce) - n_sp}）')
    # 任务附加交付：spurious-only 子集（final_label == spurious，含三方标签与理由）
    sp = ce[ce['final_label'] == 'spurious'].copy()
    sp.to_csv(out_dir / f'{out_prefix}_counterexamples_spurious.csv', index=False)
    print(f'[M4] counterexamples_spurious（子集）: {len(sp)} 条')
    for spec in ('ma', 'mb'):
        print(f'[M5] {spec} pooled p_direct = {per_spec[spec]["pooled_weighted"]["precision"]} '
              f'CI95={per_spec[spec]["pooled_weighted"]["ci95"]} '
              f'verdict={per_spec[spec]["verdict_weighted"]}（random 层对照 '
              f'{per_spec[spec]["random_layer_unbiased_overall"]["p_direct"]} / '
              f'{per_spec[spec]["verdict_random_layer"]}）')
    print(f'[M5] headline（M-A 加权池化）判定 = {gate_verdict["headline_verdict"]}')
    print(f'\n-> {res_path}')
    print(f'-> {out_dir / (out_prefix + "_counterexamples.csv")}')
    return 0


# --------------------------------------------------------------------------- #
# 抽样主流程
# --------------------------------------------------------------------------- #
def sample(args) -> int:
    frozen = check_freeze_precedes_now()
    check_frames()
    ind, nam = load_industry(), load_names()

    print(f'[V3] 载入 teacher 语料（文本年 {TEXT_YEARS}）...', flush=True)
    corpus = load_corpus(CORPUS, 'ma', TEXT_YEARS, None)     # teacher 两列同语料
    print(f'[V3] 语料 ok 行 = {len(corpus)}', flush=True)

    terms = load_lexicon_terms(LEXICON, MA_DF_MAX)
    df_map = {t: d for t, d in terms}
    automaton = build_ma_automaton(terms)
    print(f'[V3] 词库保留词条 = {len(terms)}（df_ratio <= {MA_DF_MAX}）', flush=True)

    from network_mda_edges_build import STOPWORDS, TOKEN_RE
    import jieba
    jieba.setLogLevel(60)
    idf_by_year = {}
    for y in TEXT_YEARS:
        docs = [t for (cid, yy), t in corpus.items() if yy == y]
        idf_by_year[y] = mb_year_idf(docs)
        print(f'[V3] M-B idf {y}: n_docs={len(docs)} vocab={len(idf_by_year[y])}', flush=True)

    all_rows, frames_all, shortfall_all, rules_all = [], {}, {}, {}
    for net in ('ma', 'mb'):
        for y in TEXT_YEARS:
            e = load_edges(net, y)
            rows, fc, sf, rules = draw(e, ind, net, y, args.n_per_stratum)
            frames_all[f'{net}_{y}'] = fc
            rules_all[f'{net}_{y}'] = rules
            if sf:
                shortfall_all[f'{net}_{y}'] = sf
                print(f'[V3][SHORTFALL] {net} {y}: {sf}（不得静默降层，已登记）', flush=True)
            print(f'[V3] {net} {y}: 抽出 {len(rows)} 条；框计数 {fc}', flush=True)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    # 证据词 + 摘录 + 行业 + 名称
    for i, r in enumerate(df.itertuples()):
        fid, sid, y, net = r.focal_code, r.source_code, int(r.text_year), r.spec_id
        ta = corpus.get((fid, y)); tb = corpus.get((sid, y))
        df.at[i, 'focal_name'] = nam.get(fid); df.at[i, 'source_name'] = nam.get(sid)
        df.at[i, 'focal_sw2'] = ind.get(fid); df.at[i, 'source_sw2'] = ind.get(sid)
        df.at[i, 'same_sw2'] = int(ind.get(fid) == ind.get(sid) and ind.get(fid) is not None)
        if ta is None or tb is None:
            df.at[i, 'corpus_a_status'] = 'MISSING' if ta is None else 'ok'
            df.at[i, 'corpus_b_status'] = 'MISSING' if tb is None else 'ok'
            df.at[i, 'n_shared'] = 0; df.at[i, 'shared_evidence'] = ''
            df.at[i, 'excerpt_a'] = ''; df.at[i, 'excerpt_b'] = ''
            df.at[i, 'excerpt_a_anchor'] = 'no_text'; df.at[i, 'excerpt_b_anchor'] = 'no_text'
            continue
        df.at[i, 'corpus_a_status'] = 'ok'; df.at[i, 'corpus_b_status'] = 'ok'
        ev = evidence_and_excerpt(net, ta, tb, automaton, df_map, idf_by_year[y],
                                  jieba, STOPWORDS, TOKEN_RE)
        for k, v in ev.items():
            df.at[i, k] = v
    for c in ('annotator_1', 'label_1', 'annotator_2', 'label_2', 'final_label',
              'disagreement', 'note_1', 'note_2'):
        df[c] = ''
    df = df.reindex(columns=CSV_COLUMNS)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sample_csv = out_dir / f'{args.out_prefix}_sample_{args.date}.csv'
    df.to_csv(sample_csv, index=False, encoding='utf-8-sig')
    ws = out_dir / f'{args.out_prefix}_worksheet_{args.date}.html'
    generated_at = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    build_worksheet(df, ws, {'sample_csv': sample_csv.name, 'generated_at': generated_at,
                             'frozen_at': frozen})

    # result.json 空壳（字段结构 + AWAITING_HUMAN_ANNOTATION + 门槛/裁定规则引用）
    shell = {
        'artifact': 's3_edge_precision_result',
        'status': 'AWAITING_HUMAN_ANNOTATION',
        'note': ('空壳模板：人工双人判读完成后用 `network_edge_precision_sample.py --compute '
                 '--annotated <填好的样本 CSV>` 覆盖本文件为填好的结果'),
        'protocol_file': str(PROTOCOL.relative_to(ROOT)).replace('\\', '/'),
        'protocol_freeze_timestamp': frozen,
        'sample_generated_at': generated_at,
        'freeze_precedes_sample': True,
        'registry_parameter': 'edge_precision_manual_annotation',
        'seed': SEED, 'seed_rule': 'default_rng(20260727 + 1000*text_year + (0 if ma else 1))',
        'rotation_years_text_year': TEXT_YEARS,
        'rotation_years_network_year': [NETWORK_YEARS[y] for y in TEXT_YEARS],
        'n_per_stratum': args.n_per_stratum,
        'n_rows': int(len(df)),
        'stratum_counts': df.groupby(['spec_id', 'text_year', 'stratum']).size()
        .reset_index(name='n').to_dict(orient='records'),
        'shortfall': shortfall_all or 'none',
        'frame_counts': frames_all,
        'stratum_rules': rules_all,
        'thresholds': THRESHOLDS,
        'threshold_semantics': {'p_direct >= 0.60': 'ACCEPTABLE',
                                'p_direct < 0.40': 'TRIGGER_NETWORK_RECHECK',
                                '0.40 <= p_direct < 0.60': 'CONDITIONAL'},
        'labels': LABELS,
        'label_order': 'direct_competitor > related_not_direct > spurious (conservatism increasing)',
        'decision_rule': 'conservative_lower_label (no third-annotator channel)',
        'estimation_rule': ('per-layer Wilson CI95; pooled weighted = same_sw2/cross_sw2 partition '
                            '(plan 5.7A P5.2); random layer = unbiased overall; no cross-column pooling'),
        'per_spec': {}, 'per_spec_year': {},
        'outputs_expected_after_compute': {
            'result_json_filled': f'{args.out_prefix}_result.json',
            'counterexamples': f'{args.out_prefix}_counterexamples.csv'},
    }
    (out_dir / f'{args.out_prefix}_result.json').write_text(
        json.dumps(shell, ensure_ascii=False, indent=1), encoding='utf-8')

    ce_cols = ['spec_id', 'text_year', 'network_year', 'stratum', 'focal_code', 'focal_name',
               'source_code', 'source_name', 'focal_sw2', 'source_sw2', 'similarity', 'weight',
               'n_shared', 'shared_evidence', 'final_label', 'label_1', 'label_2',
               'disagreement', 'note_1', 'note_2', 'excerpt_a', 'excerpt_b']
    (out_dir / f'{args.out_prefix}_counterexamples.csv').write_text(
        ','.join(ce_cols) + '\n', encoding='utf-8-sig')

    print(f'[V3] 样本 CSV：{sample_csv}（{len(df)} 行）')
    print(f'[V4] 工作表 HTML：{ws}')
    print(f'[V5] result 空壳：{out_dir / (args.out_prefix + "_result.json")}')
    print(f'[V5] 反例库空表头：{out_dir / (args.out_prefix + "_counterexamples.csv")}')
    cnt = df.groupby(['spec_id', 'text_year', 'stratum']).size().unstack(fill_value=0)
    print(cnt.to_string())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description='S3 边级 precision 判读：抽样 / 工作表 / 统计')
    ap.add_argument('--compute', action='store_true', help='判读后统计模式')
    ap.add_argument('--annotated', default=None, help='--compute 用：已填好的样本 CSV')
    ap.add_argument('--labels', default=None,
                    help='--compute 用：三代理标签文件（逗号分隔，顺序 A,B,C）；给出则先合并回填 '
                         '--annotated 的样本 CSV（label_1/2/3 + final_label + n_agree + disagreement）')
    ap.add_argument('--out-dir', default=str(ROOT / 'reports/data_gates/G0'))
    ap.add_argument('--out-prefix', default='s3_edge_precision')
    ap.add_argument('--date', default='20261013')
    ap.add_argument('--n-per-stratum', type=int, default=N_PER_STRATUM)
    args = ap.parse_args()
    if args.compute:
        if not args.annotated:
            raise SystemExit('[STOP] --compute 需 --annotated <填好的样本 CSV>')
        label_files = [Path(p.strip()) for p in args.labels.split(',') if p.strip()] if args.labels else None
        return compute(Path(args.annotated), Path(args.out_dir), args.out_prefix, label_files)
    return sample(args)


if __name__ == '__main__':
    raise SystemExit(main())
