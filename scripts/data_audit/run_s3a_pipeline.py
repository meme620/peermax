"""S3A 管线 runner（批次三，2026-08-12）：按依赖顺序调用各步并汇总关键数字。

用法：
  python scripts/data_audit/run_s3a_pipeline.py             # 全流程（步骤 1 起）
  python scripts/data_audit/run_s3a_pipeline.py 5           # 从步骤 5 起（snapshot 之后）

步骤号：1=ingest 2=audit 3=clean 4=dedup 5=snapshot 6=edges 7=diag
        8=derived 9=panel 10=sens 11=cf_mom 12=alpha 13=identity

全流程会重跑 step1-4（约 10-20 分钟，读取原始 7z/xlsx 并重新拆分），
日常增量建议从 5 开始（覆盖表已在冻结口径 v2 下生成）。
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'scripts' / 'data_audit'
ANALYST = ROOT / 'data' / 'processed' / 'analyst'

STEPS = [
    ('1_ingest', 's3a_step1_ingest.py', []),
    ('2_audit', 's3a_step2_audit.py', []),
    ('3_clean', 's3a_step3_clean.py', []),
    ('4_dedup', 's3a_step4_dedup.py', []),
    ('5_snapshot', 's3a_step5_snapshot.py', []),
    ('6_edges', 's3a_step67_edges.py', []),
    ('7_diag', 's3a_step8_diag.py', []),
    ('8_derived', 's3a_step9_derived.py', []),
    ('9_panel', 's3a_step10_panel_merge.py', []),
    ('10_sens', 's3a_sens_specs.py', []),
    ('11_cf_mom', 's3a_cf_mom_replication.py', []),   # 实际参数见 CF_MOM_ARGS（4 变体）
    ('12_alpha', 's3a_cf_mom_alpha.py', []),
    ('13_identity', 's3a_identity_audit.py', []),
]

# cf_mom 参数映射（4 个变体：count/ln × h1/h12）
CF_MOM_ARGS = [['count'], ['ln'], ['count', '12'], ['ln', '12']]


def run(cmd, label):
    print(f"\n===== [{label}] {' '.join(str(c) for c in cmd)} =====", flush=True)
    r = subprocess.run([sys.executable, str(cmd[0])] + [str(c) for c in cmd[1:]],
                       cwd=ROOT, capture_output=True, text=True)
    print(r.stdout[-2000:], flush=True)
    if r.returncode != 0:
        print(f"!! [{label}] FAILED\n{r.stderr[-1500:]}", flush=True)
        sys.exit(1)
    print(f"===== [{label}] OK =====", flush=True)


def summarize():
    print("\n===== 汇总（关键数字） =====", flush=True)
    snap = json.loads((ANALYST / 'step5_snapshot.json').read_text(encoding='utf-8'))
    print(f"覆盖: rows={snap['rows']} 误杀(v1口径)={snap['report_level_audit']['reports_miskilled_under_v1']}")
    diag = json.loads((ANALYST / 'analyst_network_diagnostics.json').read_text(encoding='utf-8'))['summary']
    print(f"网络: 月均边={diag['edges_monthly_mean']} deg中位={diag['degree_p50_mean']} 弱边={diag['weak_edge_ratio_mean']}")
    d9 = json.loads((ANALYST / 'step9_derived.json').read_text(encoding='utf-8'))
    print(f"派生: rows={d9['rows']} AnalystCoverage均值={d9['coverage_means']['AnalystCoverage']}")
    alpha = json.loads((ANALYST / 'cf_mom_alpha.json').read_text(encoding='utf-8'))
    # cf_mom_alpha.json 结构为 ew_ls.ff5.alpha / ew_ls.ff4.alpha（FF5 含 REV 为主口径）
    print(f"CF MOM α (FF5): {alpha['ew_ls']['ff5']['alpha']['beta_pct']}%/月 t={alpha['ew_ls']['ff5']['alpha']['t']}")


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    for idx, (label, script, _) in enumerate(STEPS, start=1):
        if idx < start:
            continue
        if label == '11_cf_mom':
            for args in CF_MOM_ARGS:
                run([SCRIPTS / 's3a_cf_mom_replication.py'] + args, f'{label}:{args}')
        else:
            run([SCRIPTS / script], label)
    summarize()
    print("\n===== 全流程完成 =====", flush=True)


if __name__ == '__main__':
    main()
