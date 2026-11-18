"""python -m peerlab：status / smoke / doctor / lint / gate / run / run-all / report。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import collect, doctor, gates, imports, lint, registry, runner

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / 'result'


def _fmt_missing(missing) -> str:
    return ', '.join(missing) if missing else '-'


def cmd_status(args) -> int:
    rows = []
    for exp in registry.EXPERIMENTS:
        check = runner.readiness(exp)
        rows.append({'id': exp.id, 'title': exp.title, 'ready': check['ready'],
                     'missing': check['missing'], 'scope': exp.scope,
                     'seeded': exp.accepts_seed, 'runtime': exp.runtime_hint})
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    ready_count = sum(1 for r in rows if r['ready'])
    print('实验数 ' + str(len(rows)) + '，就绪 ' + str(ready_count) + '，受阻 ' + str(len(rows) - ready_count))
    print('')
    for row in rows:
        flag = 'ready  ' if row['ready'] else 'blocked'
        seeded = 'seed' if row['seeded'] else '    '
        print('  ' + flag + '  ' + seeded + '  ' + row['id'].ljust(22) + row['title'])
        if not row['ready']:
            print('            缺：' + _fmt_missing(row['missing']))
    return 0


def cmd_run(args) -> int:
    exp = registry.get(args.exp)
    record = runner.run_experiment(exp, args.seed, dry_run=args.dry_run, smoke=args.smoke)
    status = record['status']
    if status == 'blocked':
        print('受阻：' + exp.id + ' 缺 ' + _fmt_missing(record['missing']))
        for command in record['commands']:
            print('  ' + ' '.join(command))
        return 2
    if status == 'dry-run':
        print('就绪，命令如下（未执行）：')
        for command in record['commands']:
            print('  ' + ' '.join(command))
        return 0
    print(status + '  ' + record['run_id'] + '  产物 ' + str(len(record['outputs'])) + ' 个  用时 ' + str(record['seconds']) + 's')
    print('  目录：' + record['run_dir'])
    return 0 if status == 'ok' else 1


def cmd_run_all(args) -> int:
    only = [x.strip() for x in args.only.split(',')] if args.only else None
    seeded = args.seeds or list(registry.SEEDS)
    experiments = [e for e in registry.EXPERIMENTS if only is None or e.id in only]
    if not experiments:
        print('没有匹配的实验：' + str(args.only))
        return 2
    outcomes = []
    for exp in experiments:
        if exp.scope == 'prepare':
            record = runner.run_experiment(exp, seeded[0], dry_run=args.dry_run)
            outcomes.append(record)
            print('prepare ' + exp.id + ' -> ' + record['status'])
            if record['status'] in ('blocked', 'failed') and not args.continue_on_error:
                break
            continue
        for seed in seeded:
            record = runner.run_experiment(exp, seed, dry_run=args.dry_run)
            outcomes.append(record)
            print(exp.id + ' seed=' + str(seed) + ' -> ' + record['status'])
            if record['status'] == 'failed' and not args.continue_on_error:
                break
        if outcomes and outcomes[-1]['status'] == 'failed' and not args.continue_on_error:
            break
    blocked = sum(1 for o in outcomes if o['status'] == 'blocked')
    failed = sum(1 for o in outcomes if o['status'] == 'failed')
    ok = sum(1 for o in outcomes if o['status'] in ('ok', 'dry-run'))
    print('')
    print('汇总：ok/dry-run ' + str(ok) + '，blocked ' + str(blocked) + '，failed ' + str(failed))
    if failed:
        return 1
    if blocked:
        return 2
    return 0


def cmd_smoke(args) -> int:
    if args.imports:
        return imports.run(ROOT, clean=args.clean)
    targets = [registry.get(args.exp)] if args.exp else list(registry.EXPERIMENTS)
    report = {'configs_missing': [], 'experiments': []}
    for name in registry.CONFIG_FILES:
        if not (ROOT / name).is_file():
            report['configs_missing'].append(name)
    blocked = 0
    for exp in targets:
        check = runner.readiness(exp)
        scripts = [step.script for step in exp.steps]
        missing_scripts = [s for s in scripts if not (ROOT / s).is_file()]
        ready = check['ready'] and not missing_scripts
        if not ready:
            blocked += 1
        report['experiments'].append({'id': exp.id, 'ready': ready,
                                      'missing_inputs': check['missing'],
                                      'missing_scripts': missing_scripts,
                                      'commands': [runner.build_command(exp, step, 0, smoke=True)
                                                   for step in exp.steps]})
    smoke_dir = RESULT_DIR / 'smoke'
    smoke_dir.mkdir(parents=True, exist_ok=True)
    (smoke_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding='utf-8')
    lines = ['# 冒烟报告（L0 就绪检查）', '',
             '配置缺失 ' + str(len(report['configs_missing'])) + ' 个：' + _fmt_missing(report['configs_missing']), '']
    for item in report['experiments']:
        state = 'ready' if item['ready'] else 'blocked'
        lines.append('- ' + item['id'].ljust(22) + state)
        if item['missing_inputs']:
            lines.append('    - 缺输入：' + _fmt_missing(item['missing_inputs']))
        if item['missing_scripts']:
            lines.append('    - 缺脚本：' + _fmt_missing(item['missing_scripts']))
    (smoke_dir / 'report.md').write_text(chr(10).join(lines) + chr(10), encoding='utf-8')
    print('冒烟：' + str(len(targets)) + ' 个实验，就绪 ' + str(len(targets) - blocked) + '，受阻 ' + str(blocked))
    print('报告：result/smoke/report.md')
    return 3 if blocked else 0


def cmd_doctor(args) -> int:
    levels = ['env', 'configs', 'data'] if args.level == 'all' else [args.level]
    ok, text = doctor.run(levels, ROOT)
    print(text)
    print('' + chr(10) + '结论:', '通过' if ok else '未通过')
    return 0 if ok else 1


def cmd_lint(args) -> int:
    code, problems = lint.run(ROOT)
    if not problems:
        print('规范检查通过')
        return code
    print('规范检查发现 ' + str(len(problems)) + ' 处问题：')
    for item in problems:
        print('  - ' + item)
    return code


def cmd_gate(args) -> int:
    only = [x.strip() for x in args.only.split(',')] if args.only else None
    if only:
        known = {gate.id for gate in gates.GATES}
        unknown = [x for x in only if x not in known]
        if unknown:
            print('未知门：' + ', '.join(unknown) + '；可选：' + ', '.join(sorted(known)))
            return 2
    results = gates.run_gates(ROOT, only=only)
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        counts = {s: sum(1 for r in results if r['status'] == s) for s in ('pass', 'fail', 'blocked')}
        print('数据门 ' + str(len(results)) + ' 个：pass ' + str(counts['pass']) +
              '，fail ' + str(counts['fail']) + '，blocked ' + str(counts['blocked']))
        for item in results:
            print('  ' + item['status'].ljust(8) + item['gate'].ljust(24) + item['title'])
            if item['blocked']:
                print('            ' + item['blocked'])
            for problem in item['problems']:
                print('            ' + problem)
    statuses = {r['status'] for r in results}
    if 'fail' in statuses:
        return 1
    if 'blocked' in statuses:
        return 2
    return 0


def cmd_report(args) -> int:
    written = collect.aggregate(RESULT_DIR)
    if not written:
        print('没有可聚合的运行')
        return 0
    for path in written:
        print('写入 ' + str(path.relative_to(ROOT)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='peerlab', description='PeerMAX 实验执行与结果聚合')
    sub = parser.add_subparsers(dest='command', required=True)

    p_status = sub.add_parser('status', help='各实验就绪状态')
    p_status.add_argument('--json', action='store_true')
    p_status.set_defaults(func=cmd_status)

    p_smoke = sub.add_parser('smoke', aliases=['-smoke'], help='冒烟：L0 就绪检查或 --imports 逐脚本导入')
    p_smoke.add_argument('--exp', default=None)
    p_smoke.add_argument('--imports', action='store_true', help='逐个导入 scripts/ 下脚本')
    p_smoke.add_argument('--clean', action='store_true', help='删除导入过程写出的文件')
    p_smoke.set_defaults(func=cmd_smoke)

    p_doctor = sub.add_parser('doctor', help='解释器 / 依赖 / configs / 数据自检')
    p_doctor.add_argument('--level', default='all', choices=['env', 'configs', 'data', 'all'])
    p_doctor.set_defaults(func=cmd_doctor)

    p_lint = sub.add_parser('lint', help='命名与实验文档规范检查')
    p_lint.set_defaults(func=cmd_lint)

    p_run = sub.add_parser('run', help='跑一个实验的一次 seed')
    p_run.add_argument('--exp', required=True)
    p_run.add_argument('-seed', '--seed', dest='seed', type=int, default=0)
    p_run.add_argument('-smoke', '--smoke', dest='smoke', action='store_true')
    p_run.add_argument('--dry-run', action='store_true')
    p_run.set_defaults(func=cmd_run)

    p_all = sub.add_parser('run-all', help='全实验 × 多种子')
    p_all.add_argument('--only', default=None, help='逗号分隔的实验 id')
    p_all.add_argument('-seeds', '--seeds', dest='seeds', type=int, nargs='+', default=None)
    p_all.add_argument('--continue-on-error', action='store_true')
    p_all.add_argument('--dry-run', action='store_true')
    p_all.set_defaults(func=cmd_run_all)

    p_gate = sub.add_parser('gate', help='数据不变量门（pass / fail / blocked）')
    p_gate.add_argument('--only', default=None, help='逗号分隔的门 id')
    p_gate.add_argument('--json', action='store_true')
    p_gate.set_defaults(func=cmd_gate)

    p_report = sub.add_parser('report', help='重算 result/tables 与 summary')
    p_report.set_defaults(func=cmd_report)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
