"""S4 Step 2 正式版：PeerMAXraw / OwnMAXraw / PeerReturn（P1.2-01 + P1.2-04 面板侧，2026-11-17）。

W1 bundle 口径（D-03=A / D-08=A+B / T-01）：
  1. **主列未归一化**（D-08 A）：`PeerMAXraw = Σ_j w_ij · MAXraw_jt`、`PeerReturn = Σ_j w_ij · R_jt`；
     仅当 `support_mass>0` 且对应 `missing_mass==0` 时写值；否则主列 missing，
     `PeerMAXraw_partial` / `PeerReturn_partial` 只进 coverage/敏感性诊断。
  2. **再归一化独立登记**（D-08 B）：`PeerMAXraw_renorm = pm / w_max`（pm = Σ_observed w·x），
     `formula_id_max_renorm=PEERMAX_RAW_RENORM_v1`，`allowed_use=robustness_only`、`gate_bearing=false`，
     不得覆盖写回主列。D-03=A 下**不产出** `PeerReturn_renorm` / `formula_id_ret_renorm`。
  3. **peer R 质量（P1.2-04 面板侧）**：收益源切到 `monthly_returns_clean_v2.csv`；`nv==0` 的 peer R
     不进 `pr`（计 `n_ret_zero_trading_days` / `missing_mass_ret`）；D-03=A 下 `0<nv<15` 的 peer 也排除
     （计 `n_ret_incomplete` / `incomplete_mass_ret` / `missing_mass_ret`）。
  4. **source-month 完整月 gate（P1.1-01）**：读 `complete_month`（缺列 → SystemExit(2)）；
     MAX 支持集 = `qualification_status` 已知且非 `limit_status_unknown` 且 `complete_month==1`；
     不完整 source 的观测 MAX 只进 partial；`OwnMAXraw` 在 focal `complete_month!=1` 时留空。
  5. **版本身份（T-01）**：`network_version` = 文本年（下游 `pd.to_numeric` 依赖）；
     `network_version_label` = `{year}_build_report.json:network_version`（完整标签；缺失/非字符串 → 停）。
  6. **聚合唯一实现** = `pricing_lib.aggregate_stat`（P1.2-01 owner；P1.4-05/M2 复用，禁止第二套）。

输入：
  data/processed/source_month_argmax_panel.csv（MAXraw / qualification / complete_month）
  data/processed/monthly_returns_clean_v2.csv（R / n_valid_days / complete_month / return_status）
  data/processed/network_{ma_teacher,mb_teacher,ma,mb}/{text_year}_similarity.csv（冻结边表）
输出：
  data/processed/monthly_pricing_panel_{corpus}_v4.csv
  reports/data_gates/G0/s4_step2_{corpus}_v4_report.json
用法：
  python scripts/peermax_panel_build.py --use-case ma \
    --net-dir data/processed/network_ma_teacher --corpus-id ma_teacher \
    --panel-version s4_panel_ma_teacher_2015_01_2023_12_v1 \
    --out data/processed/monthly_pricing_panel_ma_teacher_v4.csv \
    --report reports/data_gates/G0/s4_step2_ma_teacher_v4_report.json \
    --start 2015-01 --end 2023-12
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from pricing_lib import aggregate_stat, assert_m1_inputs_sha, assert_named_m1_input_sha, param_value, sha256_file

ARGMAX = ROOT / 'data/processed/source_month_argmax_panel.csv'
RETURNS = ROOT / 'data/processed/monthly_returns_clean_v2.csv'
REGISTRY = ROOT / 'configs/parameter_registry.yaml'
MANIFEST = ROOT / 'configs/version_manifest.yaml'
NETS = {'ma': ROOT / 'data/processed/network_ma', 'mb': ROOT / 'data/processed/network_mb'}
# 语料变体网络目录（--net-dir 指定时覆盖 NETS；输出文件名随之加后缀，避免覆盖主口径）
NET_DIR_SUFFIX = {'network_mb_teacher': '_teacher', 'network_ma_teacher': '_teacher',
                  'network_ma_teacher_idf': '_teacher_idf',
                  'network_ma_teacher_medadj': '_teacher_medadj',
                  'network_ma_fulltext': '_fulltext'}
KNOWN_NET_DIRS = set(NET_DIR_SUFFIX) | {'network_ma', 'network_mb'}
OUTD = ROOT / 'data/processed'
REPD = ROOT / 'reports/data_gates/G0'
SAMPLE_DIR = ROOT / 'data/processed'
SAMPLE_START, SAMPLE_END = '2018-01', '2025-12'
EXCLUDE_QUAL = 'limit_status_unknown'
D03_STRICT = True   # D-03=A：complete_month=1 才计入 PeerReturn
AVAILABLE_AT_RULE = '{text_year+1}-04-30; missing=STOP'
_AVAILABLE_AT_RE = re.compile(r'\d{4}-\d{2}-\d{2}\Z')
NETWORK_VERSIONS_KEY = 'network_versions'
NETWORK_AVAILABLE_AT_RULE = '(y+1)-04-30'
_NETWORK_PREFIX_RE = re.compile(r'[a-z][a-z0-9_]*_v\d+\Z')
_SIMILARITY_NAME_RE = re.compile(r'(\d{4})_similarity\.csv\Z')

# P1.2-01 A1（43 列）+ D-08 B（PeerMAXraw_renorm / formula_id_max_renorm，追加表尾）
COLS = ['focal_id', 'month', 'network_use_case', 'network_version',
        'network_version_label', 'corpus_id', 'panel_version', 'available_at',
        'PeerMAXraw', 'OwnMAXraw', 'PeerReturn', 'PeerMAXraw_partial',
        'PeerReturn_partial', 'n_sources', 'n_max_sources', 'n_ret_sources',
        'n_max_incomplete', 'n_ret_incomplete', 'n_max_missing_statistic',
        'n_max_excluded_limit_unknown', 'n_ret_missing_statistic',
        'n_ret_zero_trading_days', 'support_mass', 'covered_mass_max',
        'covered_mass_ret', 'incomplete_mass_max', 'incomplete_mass_ret',
        'missing_mass_max', 'missing_mass_ret', 'coverage_ratio_max',
        'coverage_ratio_ret', 'limit_hit_weight_share', 'support_status_max',
        'support_status_ret', 'missing_reason_max', 'missing_reason_ret',
        'network_support_status', 'data_status', 'coverage_flag',
        'missing_reason', 'formula_id_max', 'formula_id_ret',
        'source_version',
        'PeerMAXraw_renorm', 'formula_id_max_renorm']
SOURCE_VERSION = 'MAXRAW_v1|monthly_returns_clean_v2'
FORMULA_ID_MAX = 'PEERMAX_RAW_v1'
FORMULA_ID_RET = 'PEER_RETURN_v1'
FORMULA_ID_MAX_RENORM = 'PEERMAX_RAW_RENORM_v1'
WEIGHT_POLICY = 'raw_unnormalized_complete_only'


def rel(p):
    """相对 ROOT 的 POSIX 路径；ROOT 外或相对路径回退为绝对路径。"""
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def ym_add(month, k):
    y, m = int(month[:4]), int(month[5:])
    m += k
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    return f'{y:04d}-{m:02d}'


def resolve_outputs(args, netd):
    """解析输出路径；未登记 net_dir 必须显式给 --out 与 --report，禁止缺省名。"""
    suffix = NET_DIR_SUFFIX.get(netd.name, '')
    if netd.name not in KNOWN_NET_DIRS:
        if not (args.out and args.report):
            raise SystemExit(
                f'[STOP] 未登记的网络目录 {netd.name!r}：必须同时显式给出 --out 与 --report；'
                '禁止使用缺省输出名（会覆盖主口径/文本稳健性列面板）。'
                f'已登记目录: {sorted(KNOWN_NET_DIRS)}')
        print(f'[WARN] 未登记网络目录 {netd.name!r}；按显式 --out/--report 写出', file=sys.stderr)
    out = (ROOT / args.out) if args.out else OUTD / (
        f'monthly_pricing_panel_{args.use_case}{suffix}.csv')
    rp = (ROOT / args.report) if args.report else REPD / (
        f's4_step2_{args.use_case}{suffix}_report.json')
    return out, rp


def load_registry_panel_versions(registry_path=None):
    """registry 顶层 `panel_versions.headline_value` -> dict；缺失/非映射 → SystemExit(2)。"""
    p = Path(registry_path) if registry_path is not None else REGISTRY
    if not p.is_file():
        raise SystemExit(f'[STOP] registry 不存在: {rel(p)}')
    doc = yaml.safe_load(p.read_text(encoding='utf-8'))
    if not isinstance(doc, dict) or not isinstance(doc.get('parameters'), list):
        raise SystemExit(f'[STOP] registry 结构非法（缺 parameters 列表）: {rel(p)}')
    hits = [x for x in doc['parameters']
            if isinstance(x, dict) and x.get('name') == 'panel_versions']
    if len(hits) != 1:
        raise SystemExit(f'[STOP] registry panel_versions 命中 {len(hits)} 条 != 1')
    hv = hits[0].get('headline_value')
    if not isinstance(hv, dict) or not hv:
        raise SystemExit('[STOP] registry panel_versions.headline_value 必须是非空映射')
    return {str(k): str(v) for k, v in hv.items()}


def _expected_available_at(text_year):
    """冻结规则：文本年 y 的网络在 (y+1)-04-30 起可用。"""
    return f'{text_year + 1}-04-30'


def _stop_available_at(text_year, report_path, reason):
    """available_at 硬门失败：stderr 打印 text_year/文件/原因并退出码 2。"""
    print(f'[STOP] available_at 校验失败：text_year={text_year} '
          f'file={rel(report_path)} reason={reason}；'
          f'规则={AVAILABLE_AT_RULE}；禁止回退、推算或伪造日期',
          file=sys.stderr, flush=True)
    raise SystemExit(2)


def _stop_network_manifest(net_dir, reason):
    """I-9 清单硬门失败：stderr 打印目录/清单/原因并退出码 2。"""
    print(f'[STOP] 网络版本清单校验失败：net_dir={rel(net_dir)} '
          f'manifest={rel(MANIFEST)} reason={reason}；'
          f'规则：`{NETWORK_VERSIONS_KEY}[{net_dir.name}].years` 必须与目录内 '
          '`{year}_similarity.csv` 完全一致；未登记目录/年份/多余文件一律拒绝',
          file=sys.stderr, flush=True)
    raise SystemExit(2)


def _manifest_network_versions():
    """读取 manifest 顶层 `network_versions`；缺失或不可读 → SystemExit(2)。"""
    try:
        doc = yaml.safe_load(MANIFEST.read_text(encoding='utf-8'))
    except (OSError, yaml.YAMLError) as exc:
        print(f'[STOP] 网络版本清单不可读：manifest={rel(MANIFEST)} '
              f'reason={exc}', file=sys.stderr, flush=True)
        raise SystemExit(2)
    if not isinstance(doc, dict):
        print(f'[STOP] 网络版本清单顶层不是对象：manifest={rel(MANIFEST)}',
              file=sys.stderr, flush=True)
        raise SystemExit(2)
    specs = doc.get(NETWORK_VERSIONS_KEY)
    if not isinstance(specs, dict):
        print(f'[STOP] manifest 缺少顶层 {NETWORK_VERSIONS_KEY} 映射：'
              f'manifest={rel(MANIFEST)}', file=sys.stderr, flush=True)
        raise SystemExit(2)
    return specs


def _validate_version_spec(net_dir, spec):
    """校验清单条目格式；返回 (version_prefix, 年份集合)。"""
    if not isinstance(spec, dict):
        _stop_network_manifest(net_dir, f'清单条目不是映射: {type(spec).__name__}')
    prefix = spec.get('version_prefix')
    if not isinstance(prefix, str) or not _NETWORK_PREFIX_RE.fullmatch(prefix):
        _stop_network_manifest(
            net_dir, f'version_prefix 非法（须形如 pm_*_v<数字>）: {prefix!r}')
    years = spec.get('years')
    if not isinstance(years, list) or not years:
        _stop_network_manifest(net_dir, f'years 必须是非空整数列表: {years!r}')
    seen = set()
    for y in years:
        if isinstance(y, bool) or not isinstance(y, int) or not 1000 <= y <= 9999:
            _stop_network_manifest(net_dir, f'years 含非法年份（须 4 位整数）: {y!r}')
        if y in seen:
            _stop_network_manifest(net_dir, f'years 含重复年份: {y}')
        seen.add(y)
    rule = spec.get('available_at_rule')
    if rule != NETWORK_AVAILABLE_AT_RULE:
        _stop_network_manifest(
            net_dir, f'available_at_rule 非 {NETWORK_AVAILABLE_AT_RULE!r}: {rule!r}')
    return prefix, seen


def load_versions(net_dir, version_spec=None):
    """按 manifest 白名单装载网络版本：text_year -> (available_at, label, path)。

    年份清单唯一来源 = `configs/version_manifest.yaml` 顶层
    `network_versions[net_dir.name]`（`version_spec` 仅供测试/显式调用注入）。
    每年 `available_at` 必须来自同目录 `{text_year}_build_report.json` 且严格等于
    `(text_year+1)-04-30`（I-7）；T-01：`network_version_label` 必须来自 build_report
    的 `network_version`（字符串、非空，且 build_report.text_year 与目录年份一致）。
    """
    if version_spec is None:
        specs = _manifest_network_versions()
        if net_dir.name not in specs:
            _stop_network_manifest(
                net_dir, f'未登记的网络目录（不在 {NETWORK_VERSIONS_KEY}；'
                f'已登记={sorted(specs)}）')
        version_spec = specs[net_dir.name]
    _prefix, years = _validate_version_spec(net_dir, version_spec)

    actual = {}
    for p in sorted(net_dir.glob('*similarity*.csv')):
        m = _SIMILARITY_NAME_RE.fullmatch(p.name)
        if m is None:
            _stop_network_manifest(
                net_dir, f'目录含非 {{year}}_similarity.csv 文件: {p.name!r}')
        ty = int(m.group(1))
        if ty not in years:
            _stop_network_manifest(
                net_dir, f'未登记年份 {ty}（文件 {p.name}）；清单年份={sorted(years)}')
        actual[ty] = p
    missing = sorted(years - set(actual))
    if missing:
        _stop_network_manifest(
            net_dir, f'清单年份缺 {{year}}_similarity.csv: {missing}')

    vers = {}
    for ty in sorted(actual):
        p = actual[ty]
        rp = net_dir / f'{ty}_build_report.json'
        if not rp.is_file():
            _stop_available_at(ty, rp, 'build_report 缺失')
        try:
            doc = json.loads(rp.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            _stop_available_at(ty, rp, f'build_report 不可读或非法 JSON: {exc}')
        if not isinstance(doc, dict):
            _stop_available_at(ty, rp, f'build_report 顶层不是对象: {type(doc).__name__}')
        if 'available_at' not in doc:
            _stop_available_at(ty, rp, 'available_at 键缺失')
        av = doc['available_at']
        if av is None:
            _stop_available_at(ty, rp, 'available_at 为 null')
        if not isinstance(av, str):
            _stop_available_at(ty, rp, f'available_at 非字符串: {av!r}')
        if not av.strip():
            _stop_available_at(ty, rp, f'available_at 为空串/空白: {av!r}')
        if not _AVAILABLE_AT_RE.fullmatch(av):
            _stop_available_at(ty, rp, f'available_at 格式非 YYYY-MM-DD: {av!r}')
        try:
            datetime.strptime(av, '%Y-%m-%d')
        except ValueError:
            _stop_available_at(ty, rp, f'available_at 非合法日期: {av!r}')
        expected = _expected_available_at(ty)
        if av != expected:
            _stop_available_at(ty, rp, f'available_at 与规则不符: {av!r} != {expected!r}')
        # T-01：完整标签只从 build_report 读取，不用文件名猜测
        if 'network_version' not in doc:
            _stop_available_at(ty, rp, 'network_version 键缺失（T-01 标签来源）')
        label = doc['network_version']
        if not isinstance(label, str) or not label.strip():
            _stop_available_at(ty, rp, f'network_version 非非空字符串: {label!r}')
        if 'text_year' in doc:
            by = doc['text_year']
            if isinstance(by, bool) or not isinstance(by, int) or by != ty:
                _stop_available_at(
                    ty, rp, f'build_report.text_year={by!r} 与目录年份 {ty} 不一致')
        vers[ty] = {'text_year': ty, 'available_at': av, 'label': label, 'path': p}
    return vers


def select_version(vers, month):
    cand = [(ty, v) for ty, v in vers.items() if v['available_at'] < month]
    if not cand:
        return None
    return max(cand, key=lambda x: x[1]['available_at'])[1]


def load_net(path, cache, key):
    if key in cache:
        return cache[key]
    edges = defaultdict(list)
    with open(path, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            edges[r['focal_id']].append((r['source_id'], float(r['weight'])))
    cache[key] = edges
    return edges


def _num(v):
    """dict.get 值 -> float or None（空串/None/NaN → None）。"""
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else x


def _no_version_row(focal, month, use_case, corpus_id, panel_version):
    """A7：无可用网络版本行（全套字段；数值列留空，不填 0）。"""
    row = {c: '' for c in COLS}
    row.update({
        'focal_id': focal, 'month': month, 'network_use_case': use_case,
        'network_version': '', 'network_version_label': '',
        'corpus_id': corpus_id, 'panel_version': panel_version,
        'available_at': '',
        'support_status_max': 'not_applicable', 'support_status_ret': 'not_applicable',
        'network_support_status': 'no_available_network_version',
        'data_status': 'no_available_network_version',
        'coverage_flag': 'not_covered',
        'missing_reason': 'no_available_network_version',
    })
    return row


def _write_reason(ds, mrmax, mrret):
    """首 token = data_status（仅非 ok）；ok 行写空串（R2/R3 阻断修复，2026-11-17）。

    规则：
      - `data_status=='ok'` → `missing_reason=''`（不得写 'ok'；ok 不是缺失原因）；
      - 非 ok → 首 token = data_status，后接 missing_reason_max/ret 细节 token（去重）。
    """
    ds_arr = np.asarray(ds, dtype=object)
    out = np.array(['' if str(s) == 'ok' else str(s) for s in ds_arr], dtype=object)
    for detail in (mrmax, mrret):
        d = np.asarray(detail, dtype=object)
        mask = np.array([bool(x) if isinstance(x, str) else False for x in d])
        out = np.where(mask & (out != '') & (out != d), out + ';' + d, out)
    # ok 行即使残留细节 token 也一律清空（ok 行不应有 missing_reason）
    return np.where(ds_arr == 'ok', '', out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--use-case', choices=['ma', 'mb'], required=True)
    ap.add_argument('--net-dir', default=None,
                    help='显式网络目录（语料变体用，如 data/processed/network_mb_teacher）；缺省按 use-case 取 NETS')
    ap.add_argument('--corpus-id', default=None,
                    help='语料身份（缺省 = net_dir.name，如 ma_teacher）')
    ap.add_argument('--panel-version', default=None,
                    help='panel_version（缺省读 registry panel_versions[corpus_id]）')
    ap.add_argument('--out', default=None,
                    help='显式输出 CSV 路径（缺省按 use-case + net_dir suffix 命名）')
    ap.add_argument('--report', default=None,
                    help='缺省 reports/data_gates/G0/s4_step2_{use_case}{suffix}_report.json；'
                         '未登记 net_dir 必须显式给出')
    ap.add_argument('--start', default=SAMPLE_START)
    ap.add_argument('--end', default=SAMPLE_END)
    ap.add_argument('--registry', default=None,
                    help='registry YAML（缺省 configs/parameter_registry.yaml；仅测试注入）')
    args = ap.parse_args(argv)

    # 读取 M1 输入之前先核验锚（source-month 与 clean_v2 均在 manifest 登记）。
    # 报告登记（07 §8.8）：input_anchors 记录本次实际核验的 9 个 M1 输入锚。
    m1_ledger = assert_m1_inputs_sha()
    anchor_argmax = assert_named_m1_input_sha('source_month_argmax_panel', ARGMAX)
    anchor_returns = assert_named_m1_input_sha('monthly_returns_clean_v2', RETURNS)
    input_anchors = [
        {'input': e['input'], 'file': e['file'], 'sha256': e['sha256'],
         'anchor_sha256': e['anchor_sha256'], 'anchor_source': e['anchor_source']}
        for e in m1_ledger
    ]

    netd = (ROOT / args.net_dir) if args.net_dir else NETS[args.use_case]
    corpus_id = args.corpus_id or (netd.name if args.net_dir else args.use_case)
    panel_versions = load_registry_panel_versions(args.registry)
    if args.panel_version:
        panel_version = args.panel_version
    else:
        if corpus_id not in panel_versions:
            raise SystemExit(
                f'[STOP] registry panel_versions 缺 corpus {corpus_id!r}；'
                f'已登记={sorted(panel_versions)}')
        panel_version = panel_versions[corpus_id]
    out, rp = resolve_outputs(args, netd)
    vers = load_versions(netd)
    print(f'{args.use_case}/{corpus_id}: 网络版本 {sorted(vers)}; panel_version={panel_version}')

    # source-month：MAXraw / 资格 / 完整月
    header = pd.read_csv(ARGMAX, nrows=0).columns.tolist()
    required = {'source_id', 'month', 'MAXraw', 'qualification_status',
                'limit_hit_at_argmax', 'complete_month', 'n_valid_days', 'max_status'}
    missing_cols = sorted(required - set(header))
    if missing_cols:
        raise SystemExit(
            f'[STOP] source_month_argmax_panel 缺 P1.1-01 字段 {missing_cols}；'
            '不得默认 complete_month=1（先执行 P1.1-01）')
    maxraw, qual, lim, complete = defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict)
    with open(ARGMAX, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            m = r['month']
            mid = r['source_id']
            mv = _num(r.get('MAXraw'))
            if mv is not None:
                maxraw[m][mid] = mv
            q = (r.get('qualification_status') or '').strip()
            if q:
                qual[m][mid] = q
            lv = _num(r.get('limit_hit_at_argmax'))
            if lv is not None:
                lim[m][mid] = int(lv)
            complete[m][mid] = int(float(r['complete_month']))

    # 月收益（v2；P1.2-04 切换）
    rmap, nvmap, ret_complete = defaultdict(dict), {}, {}
    with open(RETURNS, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            m = r['month']
            fid = r['firm_id']
            rv = _num(r.get('R'))
            if rv is not None:
                rmap[m][fid] = rv
            nvmap[(fid, m)] = int(float(r['n_valid_days']))
            ret_complete[(fid, m)] = int(float(r['complete_month']))

    months = []
    m = args.start
    while m <= args.end:
        months.append(m)
        m = ym_add(m, 1)
    print(f'月份 {months[0]}..{months[-1]} ({len(months)})')

    cache = {}
    rows = []
    st = defaultdict(float)
    status_counts = defaultdict(int)
    year_mass_zero = defaultdict(float)
    year_mass_support = defaultdict(float)
    n_ret_zero_total = 0
    zero_by_row = {}
    sup_by_row = {}
    gate = {'min_valid_days': int(param_value('min_valid_days', args.registry)),
            'included_incomplete_sources': 0,
            'rows_with_any_incomplete_peer': 0, 'rows_all_peer_incomplete': 0,
            'rows_own_incomplete': 0,
            'definition': 'peer incomplete = source-month complete_month != 1（MAX 侧；R 侧按 n_ret_incomplete 并报）'}
    for month in months:
        v = select_version(vers, month)
        if v is None:
            for f_ in sorted(set(list(maxraw[month]) + list(rmap[month]))):
                rows.append(_no_version_row(f_, month, args.use_case, corpus_id, panel_version))
                st['rows_no_network'] += 1
                # R2 修复（2026-11-17）：no-version 行必须计入 data_status 汇总，
                # 使 rows_by_data_status 之和 = rows_total；不得留误导性 0。
                status_counts['no_available_network_version'] += 1
            continue
        edges = load_net(v['path'], cache, (args.use_case, v['text_year']))
        records = [(focal, src, w) for focal, lst in edges.items() for src, w in lst]
        edf = pd.DataFrame(records, columns=['focal_id', 'source_id', 'weight'])
        edf['month'] = month
        sids = edf['source_id']
        mm = maxraw[month]
        qq = qual[month]
        ll = lim[month]
        cm = complete[month]
        rrm = rmap[month]
        src = pd.DataFrame({
            'source_id': sids,
            'MAXraw': sids.map(lambda s: mm.get(s)),
            'R': sids.map(lambda s: rrm.get(s)),
            'q': sids.map(lambda s: qq.get(s)),
            'lim': sids.map(lambda s: ll.get(s)),
            'cm': sids.map(lambda s: cm.get(s, 0)),
            'has_ret': sids.map(lambda s: s in rrm),
            'nv': [nvmap.get((s, month), 0) for s in sids],
        })
        q_known = src['q'].notna()
        is_excl = q_known & src['q'].eq(EXCLUDE_QUAL)
        cm_ok = src['cm'].fillna(0).astype(int).eq(1)
        elig_max = q_known & ~is_excl & cm_ok
        inc_max = q_known & ~is_excl & ~cm_ok
        nv = pd.to_numeric(src['nv'], errors='coerce').fillna(0).astype(int)
        has_ret = src['has_ret'].astype(bool)
        zero_ret = has_ret & nv.eq(0)
        inc_ret = has_ret & nv.gt(0) & ~cm_ok
        elig_ret = has_ret & cm_ok
        extra_w = np.where(src['lim'].fillna(0).astype(float).eq(1), edf['weight'], 0.0)

        a_max = aggregate_stat(
            edf, src, 'MAXraw', elig_max, 'max', excluded_mask=is_excl,
            incomplete_mask=inc_max, extra_weight=extra_w)
        a_ret = aggregate_stat(
            edf, src, 'R', elig_ret, 'ret', zero_trading_days_mask=zero_ret,
            incomplete_mask=inc_ret)
        agg = a_max.merge(a_ret, on=['month', 'focal_id'], how='outer')
        agg = agg.sort_values(['month', 'focal_id'], kind='stable').reset_index(drop=True)

        sup = agg['support_mass_max'].to_numpy(float)
        wmax = agg['covered_mass_max'].to_numpy(float)
        wret = agg['covered_mass_ret'].to_numpy(float)
        miss_max = agg['missing_mass_max'].to_numpy(float)
        miss_ret = agg['missing_mass_ret'].to_numpy(float)
        v_max = agg['weighted_value_max'].to_numpy(float)
        v_ret = agg['weighted_value_ret'].to_numpy(float)
        p_max = agg['partial_value_max'].to_numpy(float)
        p_ret = agg['partial_value_ret'].to_numpy(float)
        ds = np.select(
            [sup == 0, wmax == 0, miss_max > 0, wret == 0, miss_ret > 0],
            ['no_network_support', 'no_covered_max_source', 'partial_max_coverage',
             'no_covered_ret_source', 'partial_ret_coverage'],
            default='ok')
        cov_flag = np.where(ds == 'ok', 'covered',
                            np.where([str(x).startswith('partial') for x in ds],
                                     'partial', 'not_covered'))
        reason = _write_reason(ds, agg['missing_reason_max'], agg['missing_reason_ret'])

        # D-08 B：pm/w_max（pm = Σ_observed w·x = covered + incomplete observed）
        pm_all = np.where(np.isnan(v_max), p_max, v_max)
        renorm = np.where(wmax > 0, pm_all / np.where(wmax > 0, wmax, 1.0), np.nan)
        peer_max = np.where(np.isnan(v_max), '', np.round(v_max, 8))
        peer_ret = np.where(np.isnan(v_ret), '', np.round(v_ret, 8))
        peer_max_partial = np.where(np.isnan(v_max) & (wmax > 0), np.round(p_max, 8), '')
        peer_ret_partial = np.where(np.isnan(v_ret) & (wret > 0), np.round(p_ret, 8), '')
        renorm_out = np.where(wmax > 0, np.round(renorm, 8), '')
        fid_arr = agg['focal_id'].to_numpy()
        own = np.array([
            (mm[f] if (f in mm and f in cm and cm.get(f, 0) == 1) else np.nan)
            for f in fid_arr
        ], dtype=float)
        own_out = np.where(np.isnan(own), '', np.round(own, 8))
        own_inc = np.array([int(f in cm and cm.get(f, 0) != 1) for f in fid_arr])

        for i, fid in enumerate(fid_arr):
            zero_by_row[(month, fid)] = float(agg['zero_trading_days_mass_ret'].iloc[i])
            sup_by_row[(month, fid)] = float(sup[i])
            rows.append({
                'focal_id': fid, 'month': month, 'network_use_case': args.use_case,
                'network_version': str(v['text_year']),
                'network_version_label': v['label'],
                'corpus_id': corpus_id, 'panel_version': panel_version,
                'available_at': v['available_at'],
                'PeerMAXraw': peer_max[i], 'OwnMAXraw': own_out[i],
                'PeerReturn': peer_ret[i],
                'PeerMAXraw_partial': peer_max_partial[i],
                'PeerReturn_partial': peer_ret_partial[i],
                'n_sources': int(agg['n_sources_max'].iloc[i]),
                'n_max_sources': int(agg['n_covered_max'].iloc[i]),
                'n_ret_sources': int(agg['n_covered_ret'].iloc[i]),
                'n_max_incomplete': int(agg['n_incomplete_max'].iloc[i]),
                'n_ret_incomplete': int(agg['n_incomplete_ret'].iloc[i]),
                'n_max_missing_statistic': int(agg['n_missing_statistic_max'].iloc[i]),
                'n_max_excluded_limit_unknown': int(agg['n_excluded_max'].iloc[i]),
                'n_ret_missing_statistic': int(agg['n_missing_statistic_ret'].iloc[i]),
                'n_ret_zero_trading_days': int(agg['n_zero_trading_days_ret'].iloc[i]),
                'support_mass': round(sup[i], 8),
                'covered_mass_max': round(wmax[i], 8),
                'covered_mass_ret': round(wret[i], 8),
                'incomplete_mass_max': round(float(agg['incomplete_mass_max'].iloc[i]), 8),
                'incomplete_mass_ret': round(float(agg['incomplete_mass_ret'].iloc[i]), 8),
                'missing_mass_max': round(miss_max[i], 8),
                'missing_mass_ret': round(miss_ret[i], 8),
                'coverage_ratio_max': round(wmax[i] / sup[i], 8) if sup[i] > 0 else '',
                'coverage_ratio_ret': round(wret[i] / sup[i], 8) if sup[i] > 0 else '',
                'limit_hit_weight_share': (
                    round(float(agg['limit_hit_weight_share_max'].iloc[i]), 6)
                    if (ds[i] == 'ok') else ''),
                'support_status_max': agg['support_status_max'].iloc[i],
                'support_status_ret': agg['support_status_ret'].iloc[i],
                'missing_reason_max': agg['missing_reason_max'].iloc[i],
                'missing_reason_ret': agg['missing_reason_ret'].iloc[i],
                'network_support_status': 'support',
                'data_status': ds[i], 'coverage_flag': cov_flag[i],
                'missing_reason': reason[i],
                'formula_id_max': FORMULA_ID_MAX, 'formula_id_ret': FORMULA_ID_RET,
                'source_version': SOURCE_VERSION,
                'PeerMAXraw_renorm': renorm_out[i],
                'formula_id_max_renorm': FORMULA_ID_MAX_RENORM if renorm_out[i] != '' else '',
            })
            st['rows_support'] += 1
            st['mass_support'] += sup[i]
            st['mass_missing_max'] += miss_max[i]
            st['mass_missing_ret'] += miss_ret[i]
            status_counts[ds[i]] += 1
            if agg['support_status_max'].iloc[i] == 'complete':
                st['rows_complete_max'] += 1
            if agg['support_status_ret'].iloc[i] == 'complete':
                st['rows_complete_ret'] += 1
            if agg['support_status_max'].iloc[i] == 'partial':
                st['rows_partial_max'] += 1
            if agg['support_status_ret'].iloc[i] == 'partial':
                st['rows_partial_ret'] += 1
            if agg['n_incomplete_max'].iloc[i] > 0 or agg['n_incomplete_ret'].iloc[i] > 0:
                gate['rows_with_any_incomplete_peer'] += 1
            if int(agg['n_incomplete_max'].iloc[i]) == int(agg['n_sources_max'].iloc[i]):
                gate['rows_all_peer_incomplete'] += 1
            if own_inc[i]:
                gate['rows_own_incomplete'] += 1

        # 零交易日 R 权重（面板级 + FM 子集诊断；FM 子集在报告阶段计算）
        zero_mass = float(edf.loc[zero_ret.to_numpy(), 'weight'].sum())
        st['mass_ret_zero_td'] += zero_mass
        n_ret_zero_total += int(zero_ret.sum())

    # support rows 的 status 计数和 must equal rows_support（互斥）
    with open(out, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    mass_support = st['mass_support']
    # nv0_ret_weight_share：06 证据口径 = FM 行（v3 m1_in_fm_sample=1，只读诊断；W1 不产 v4 样本）
    fm_keys = None
    fm_universe = 'panel_all_support_rows'
    v3p = SAMPLE_DIR / f'm1_common_sample_{corpus_id}_v3.parquet'
    if v3p.is_file():
        try:
            v3 = pd.read_parquet(v3p, columns=['focal_id', 'month', 'm1_in_fm_sample'])
            _fm = v3.loc[v3['m1_in_fm_sample'].fillna(False).astype(bool)]
            fm_keys = set(zip(_fm['month'], _fm['focal_id']))
            fm_universe = 'v3_fm_rows_readonly_diagnostic'
        except Exception as exc:  # 诊断口径不可用时回退面板级并在报告标注
            fm_universe = f'panel_all_support_rows(v3_fm_unavailable:{exc})'
    fm_zero = sum(zero_by_row.get(k, 0.0) for k in fm_keys) if fm_keys is not None else st['mass_ret_zero_td']
    fm_sup = sum(sup_by_row.get(k, 0.0) for k in fm_keys) if fm_keys is not None else mass_support
    year_zero, year_sup = defaultdict(float), defaultdict(float)
    keys_iter = fm_keys if fm_keys is not None else sup_by_row.keys()
    for k in keys_iter:
        year_zero[k[1][:4]] += zero_by_row.get(k, 0.0)
        year_sup[k[1][:4]] += sup_by_row.get(k, 0.0)
    # R2 修复（2026-11-17）：data_status 汇总计入 no-version 行（sum=rows_total），
    # 另列 support-only 汇总（sum=rows_support）+ rows_no_available_network_version。
    ds_keys = ('ok', 'partial_max_coverage', 'partial_ret_coverage',
               'no_covered_max_source', 'no_covered_ret_source',
               'no_network_support', 'not_covered_by_corpus',
               'no_available_network_version')
    rows_by_data_status = {k: int(status_counts.get(k, 0)) for k in ds_keys}
    rows_by_data_status_support = {
        k: v for k, v in rows_by_data_status.items()
        if k != 'no_available_network_version'}
    status_checks = {
        'rows_by_data_status_sum_equals_rows_total':
            sum(rows_by_data_status.values()) == len(rows),
        'rows_by_data_status_support_sum_equals_rows_support':
            sum(rows_by_data_status_support.values()) == int(st['rows_support']),
        'rows_support_plus_no_version_equals_rows_total':
            int(st['rows_support']) + int(st['rows_no_network']) == len(rows),
        'included_incomplete_sources_zero':
            int(gate['included_incomplete_sources']) == 0,
        'ok_rows_missing_reason_empty':
            all(r['missing_reason'] == '' for r in rows if r['data_status'] == 'ok'),
        'non_ok_rows_missing_reason_nonempty':
            all(r['missing_reason'] != '' for r in rows if r['data_status'] != 'ok'),
    }
    status_rows_contract = (
        'PASS' if (status_checks['rows_by_data_status_sum_equals_rows_total']
                   and status_checks['rows_by_data_status_support_sum_equals_rows_support']
                   and status_checks['rows_support_plus_no_version_equals_rows_total'])
        else 'FAIL')
    status_complete_month_gate = (
        'PASS' if status_checks['included_incomplete_sources_zero'] else 'FAIL')
    status_missing_reason = (
        'PASS' if (status_checks['ok_rows_missing_reason_empty']
                   and status_checks['non_ok_rows_missing_reason_nonempty'])
        else 'FAIL')
    rep = {
        'report_id': f's4_step2_peermax_{args.use_case}',
        'use_case': args.use_case, 'corpus_id': corpus_id,
        'panel_version': panel_version,
        'run_at': datetime.now(timezone.utc).isoformat(),
        'script': rel(Path(__file__)),
        'script_sha256': sha256_file(Path(__file__)),
        'input_anchors': input_anchors,
        'input_anchor_argmax': anchor_argmax['sha256'],
        'input_anchor_returns': anchor_returns['sha256'],
        'status': 'PASS' if all(status_checks.values()) else 'FAIL',
        'status_checks': status_checks,
        'status_rows_contract': status_rows_contract,
        'status_complete_month_gate': status_complete_month_gate,
        'status_missing_reason': status_missing_reason,
        'months': len(months), 'month_range': [months[0], months[-1]],
        'network_versions_used': sorted({r['network_version'] for r in rows if r['network_version']}),
        'network_version_labels': sorted({r['network_version_label'] for r in rows
                                          if r['network_version_label']}),
        'versions_without_report': 0,
        'excluded_versions': [],
        'available_at_rule': AVAILABLE_AT_RULE,
        'rows_total': len(rows), 'rows_support': int(st['rows_support']),
        'rows_no_network': int(st['rows_no_network']),
        'rows_complete_max': int(st['rows_complete_max']),
        'rows_complete_ret': int(st['rows_complete_ret']),
        'rows_partial_max': int(st['rows_partial_max']),
        'rows_partial_ret': int(st['rows_partial_ret']),
        'rows_by_data_status': rows_by_data_status,
        'rows_by_data_status_support': rows_by_data_status_support,
        'rows_no_available_network_version': int(st['rows_no_network']),
        'rows_by_data_status_sum': int(sum(rows_by_data_status.values())),
        'mass_support': round(mass_support, 4),
        'mass_missing_max': round(st['mass_missing_max'], 4),
        'mass_missing_ret': round(st['mass_missing_ret'], 4),
        'mass_ret_zero_trading_days': round(st['mass_ret_zero_td'], 8),
        'mass_ret_zero_trading_days_fm': round(fm_zero, 8),
        'n_ret_zero_trading_days_total': int(n_ret_zero_total),
        'nv0_ret_weight_share': (round(fm_zero / fm_sup, 8) if fm_sup > 0 else None),
        'nv0_ret_weight_share_universe': fm_universe,
        'nv0_ret_weight_share_panel': (round(st['mass_ret_zero_td'] / mass_support, 8)
                                       if mass_support > 0 else None),
        'nv0_ret_weight_share_by_year': {
            y: round(year_zero[y] / year_sup[y], 8)
            for y in sorted(year_sup) if year_sup[y] > 0},
        'coverage_ratio_max_overall': (round(sum(float(r['covered_mass_max']) for r in rows
                                                 if r.get('covered_mass_max') != '') / mass_support, 6)
                                       if mass_support > 0 else None),
        'coverage_ratio_ret_overall': (round(sum(float(r['covered_mass_ret']) for r in rows
                                                 if r.get('covered_mass_ret') != '') / mass_support, 6)
                                       if mass_support > 0 else None),
        'weight_policy': WEIGHT_POLICY,
        'formula_ids': {'max': FORMULA_ID_MAX, 'ret': FORMULA_ID_RET,
                        'max_renorm': FORMULA_ID_MAX_RENORM},
        'renorm_policy': {'column': 'PeerMAXraw_renorm',
                          'formula': 'sum_observed w*x / covered_mass_max',
                          'allowed_use': 'robustness_only', 'gate_bearing': False,
                          'peer_return_renorm': 'not_produced_under_D03_A'},
        'complete_month_gate': gate,
        'output': rel(out),
        'rules': {'version_selection': 'max{available_at < month}',
                  'aggregation': 'raw unnormalized complete-only weighted sum (Σ w·x); partial diagnostics separate',
                  'missing': 'no 0-fill; empty + data_status; missing_mass accumulated directly; partial never headline',
                  'source_qualification': f'exclude {EXCLUDE_QUAL} from MAX support set',
                  'return_measure': 'monthly_returns_clean_v2 R (raw close, frozen primary); '
                                    'zero-trading-day and incomplete-month peers excluded (D-03=A)',
                  'source_month_gate': 'MAX/R complete_month==1 required; incomplete peers counted separately'},
    }
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in rep.items() if k not in ('rules',)},
                     ensure_ascii=False, indent=1)[:1800])
    print('->', out)
    print('-> report:', rp)


if __name__ == '__main__':
    main()
