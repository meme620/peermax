#!/usr/bin/env python3
"""B5 / S-7: A9 全 20 行规格表 + gate MD 覆盖对照 + registry 登记核对.

约束（本脚本硬性遵守）:
  - 只读输入: reports/data_gates/G0/s3_m1_robustness_placebo.json,
    reports/data_gates/G0/s3_m1_gate_report_v2.md, configs/parameter_registry.yaml
  - 不重跑 M1 / PC / 网络; 不重估任何系数; 所有读数逐字取自 JSON.
  - 不改 headline / 系数 / gate / configs / G0 / reports.
  - 仅写 artifacts/phase3b/3b_s7_{a9_full.csv,counts.json,execution.md}.

stdout 仅输出机器可解析 JSON summary; 过程日志走 stderr.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

ROBUSTNESS_JSON = ROOT / "reports" / "data_gates" / "G0" / "s3_m1_robustness_placebo.json"
GATE_MD = ROOT / "reports" / "data_gates" / "G0" / "s3_m1_gate_report_v2.md"
REGISTRY_YAML = ROOT / "configs" / "parameter_registry.yaml"

OUT_DIR = ROOT / "artifacts" / "phase3b"
OUT_CSV = OUT_DIR / "3b_s7_a9_full.csv"
OUT_JSON = OUT_DIR / "3b_s7_counts.json"
OUT_MD = OUT_DIR / "3b_s7_execution.md"

FAMILY_PREFIXES = (
    ("m1_rob_limit_", "m1_rob_limit"),
    ("m1_rob_size_", "m1_rob_size"),
    ("m1_rob_liq_", "m1_rob_liq"),
)

COLUMN_LABELS = ("ma_teacher", "mb_teacher", "ma", "mb")

STRATUM_LABELS = {
    "m1_rob_raw": "基准（raw 主口径）",
    "m1_rob_postadj": "outcome 变体（后复权）",
    "m1_rob_st": "ST 分层",
    "m1_rob_non_st": "ST 分层",
    "m1_rob_limit_all_limit": "limit-hit 分层",
    "m1_rob_limit_all_non_limit": "limit-hit 分层",
    "m1_rob_limit_mixed": "limit-hit 分层",
    "m1_rob_size_low": "size 三分位分层",
    "m1_rob_size_high": "size 三分位分层",
    "m1_rob_liq_low": "liq 三分位分层",
    "m1_rob_liq_high": "liq 三分位分层",
    "m1_rob_col_ma_teacher": "四列并报",
    "m1_rob_col_mb_teacher": "四列并报",
    "m1_rob_col_ma": "四列并报",
    "m1_rob_col_mb": "四列并报",
    "m1_weight_equal": "权重规则",
    "m1_weight_linear": "权重规则",
    "m1_weight_logsim": "权重规则",
    "m1_placebo_random": "placebo 网络",
    "m1_placebo_timeshift": "placebo 网络",
}

STRATA_FOCUS = (
    "m1_rob_liq_low",
    "m1_rob_liq_high",
    "m1_rob_size_low",
    "m1_rob_size_high",
    "m1_rob_limit_all_limit",
    "m1_rob_limit_all_non_limit",
    "m1_rob_limit_mixed",
    "m1_rob_non_st",
    "m1_rob_st",
)


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def file_size(path):
    return path.stat().st_size


def as_float(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def registry_block(reg: dict, name: str) -> dict:
    for param in reg.get("parameters", []):
        if param.get("name") == name:
            return param
    raise KeyError(f"registry parameter not found: {name}")


def parse_registry(reg: dict) -> dict:
    spec_block = registry_block(reg, "m1_sample_and_spec_registry")
    hv = spec_block["headline_value"]
    spec_ids = [s.strip() for s in str(hv["specification_ids"]).split("/") if s.strip()]
    k_econ = as_float(registry_block(reg, "K_Econ_M1").get("headline_value"))
    return {
        "specification_ids": spec_ids,
        "min_focal_count_per_month": int(hv["min_focal_count_per_month"]),
        "min_effective_month_count": int(hv["min_effective_month_count"]),
        "nw_lag": int(hv["nw_lag"]),
        "iqr_scale_rule_id": hv.get("iqr_scale_rule_id"),
        "K_Econ_M1": k_econ,
        "registry_version": reg.get("registry_version"),
    }


def family_of(spec_id: str) -> str | None:
    for prefix, family in FAMILY_PREFIXES:
        if spec_id.startswith(prefix):
            return family
    return None


def parse_gate_md(text: str) -> dict:
    lines = text.splitlines()
    a9 = []
    for lineno, line in enumerate(lines, 1):
        m = re.match(r"^- A9 (m1_[A-Za-z0-9_]+):", line)
        if m:
            a9.append({"spec_id": m.group(1), "line": lineno, "text": line})
    four_col = {}
    for lineno, line in enumerate(lines, 1):
        for label in COLUMN_LABELS:
            if line.startswith(f"| {label} |"):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if len(cells) >= 6:
                    four_col[label] = {
                        "line": lineno,
                        "n_rows": int(cells[2].replace(",", "")),
                        "T": int(cells[3]),
                        "beta_ew": as_float(cells[5]),
                    }
    return {"lines": lines, "a9": a9, "four_column": four_col}


def literal_lines(spec_id: str, lines: list[str]) -> list[int]:
    return [i for i, line in enumerate(lines, 1) if spec_id in line]


def beta_in_md(a9_line: str):
    m = re.search(r"beta=(-?\d+\.\d+)", a9_line)
    return float(m.group(1)) if m else None


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    robustness = json.loads(ROBUSTNESS_JSON.read_text(encoding="utf-8"))
    registry = yaml.safe_load(REGISTRY_YAML.read_text(encoding="utf-8"))
    gate_text = GATE_MD.read_text(encoding="utf-8")
    gate = parse_gate_md(gate_text)
    reg_info = parse_registry(registry)

    specs = robustness["specs"]
    min_months = reg_info["min_effective_month_count"]
    min_focal = reg_info["min_focal_count_per_month"]
    nw_lag = reg_info["nw_lag"]
    k_econ = reg_info["K_Econ_M1"]
    iqr_pool = as_float(robustness.get("iqr_pool_registered"))

    rows = []
    for spec_id, spec in specs.items():
        status = spec.get("spec_status")
        beta = as_float(spec.get("spec_beta"))
        t_nw = as_float(spec.get("t_nw"))
        months = spec.get("n_months_used")
        n_rows = spec.get("n_rows")
        p_one = as_float(spec.get("p_one_sided"))
        econ = beta * iqr_pool if (beta is not None and iqr_pool is not None) else None

        literal = literal_lines(spec_id, gate["lines"])
        four_label = None
        if spec_id.startswith("m1_rob_col_"):
            candidate = spec_id[len("m1_rob_col_"):]
            if candidate in gate["four_column"]:
                four_label = candidate

        if literal:
            coverage = "a9_spec_line"
        elif four_label:
            coverage = "four_column_table"
        elif spec_id == "m1_rob_raw":
            coverage = "headline_reference"
        else:
            coverage = "none"

        if spec_id in reg_info["specification_ids"]:
            reg_registration = "exact"
            reg_id = spec_id
            reg_note = ""
        else:
            fam = family_of(spec_id)
            if fam and fam in reg_info["specification_ids"]:
                reg_registration = "family"
                reg_id = fam
                reg_note = f"仅族级登记 {fam}，未登记层 id"
            else:
                reg_registration = "none"
                reg_id = ""
                if spec_id == "m1_rob_non_st":
                    reg_note = "未登记：ST 族仅登记 m1_rob_st，无 non_st"
                elif spec_id.startswith("m1_rob_col_"):
                    reg_note = "未登记：registry 有「四列并报」设计说明，specification_ids 无 m1_rob_col_*"
                else:
                    reg_note = "未登记"

        if status != "COMPUTED":
            coverage_flag = "NOT_COMPUTABLE"
        elif isinstance(months, int) and months < min_months:
            coverage_flag = "LOW_COVERAGE"
        else:
            coverage_flag = "OK"

        low_warning = spec.get("low_coverage_warning", "")
        status_reason = spec.get("status_reason", "")
        if status == "NOT_COMPUTABLE":
            coverage_detail = status_reason
        elif coverage_flag == "LOW_COVERAGE":
            coverage_detail = (
                f"{low_warning}；T={months} < min_effective_month_count={min_months}"
                if low_warning
                else f"T={months} < min_effective_month_count={min_months}"
            )
        else:
            coverage_detail = f"T={months} >= min_effective_month_count={min_months}，无 warning"

        degenerate = ""
        if status != "COMPUTED" or (isinstance(months, int) and months < min_months):
            if months == 0:
                degenerate = "T=0：不可估计（NOT_COMPUTABLE）"
            elif isinstance(months, int) and months <= nw_lag:
                degenerate = (
                    f"T={months}：低于 min_effective_month_count={min_months} 且 "
                    f"T <= NW lag={nw_lag}，NW 协方差退化，t 不可解释"
                )
            else:
                degenerate = f"T={months}：低于 min_effective_month_count={min_months}"

        sign = "NA" if beta is None else ("NEG" if beta < 0 else ("POS" if beta > 0 else "ZERO"))
        sign_vs_hyp = "NA" if beta is None else ("same" if beta < 0 else "opposite")

        rows.append(
            {
                "spec_id": spec_id,
                "spec_status": status,
                "beta": beta,
                "t_nw": t_nw,
                "T_months": months,
                "n_rows": n_rows,
                "p_one_sided": p_one,
                "economic_magnitude_beta_x_iqr": econ,
                "sign": sign,
                "sign_vs_registered_direction_beta_lt_0": sign_vs_hyp,
                "coverage_flag": coverage_flag,
                "low_coverage_warning": low_warning,
                "status_reason": status_reason,
                "coverage_detail": coverage_detail,
                "in_gate_md_literal": bool(literal),
                "gate_md_lines": ";".join(str(x) for x in literal),
                "gate_md_coverage": coverage,
                "gate_md_coverage_detail": (
                    f"A9 第 {literal[0]} 行" if literal
                    else (f"四列并报表 {gate['four_column'][four_label]['line']} 行（列名 {four_label}）"
                          if four_label else
                          ("headline / A9 placebo 对照读数（未以 spec_id 形式列出）"
                           if spec_id == "m1_rob_raw" else "无"))
                ),
                "registry_registration": reg_registration,
                "registry_id_or_family": reg_id,
                "registry_note": reg_note,
                "stratum": STRATUM_LABELS.get(spec_id, ""),
                "degenerate_T": degenerate,
                "spec_note": spec.get("note", ""),
            }
        )

    # ---------- counts ----------
    in_md = [r["spec_id"] for r in rows if r["in_gate_md_literal"]]
    not_in_md_literal = [r["spec_id"] for r in rows if not r["in_gate_md_literal"]]
    four_col_covered = [r["spec_id"] for r in rows if r["gate_md_coverage"] == "four_column_table"]
    headline_ref = [r["spec_id"] for r in rows if r["gate_md_coverage"] == "headline_reference"]
    true_gap = [r["spec_id"] for r in rows if r["gate_md_coverage"] == "none"]
    a9_ids = [a["spec_id"] for a in gate["a9"]]
    a9_computed = [sid for sid in a9_ids if specs.get(sid, {}).get("spec_status") == "COMPUTED"]
    a9_not_computable = [sid for sid in a9_ids if specs.get(sid, {}).get("spec_status") != "COMPUTED"]

    exact = [r["spec_id"] for r in rows if r["registry_registration"] == "exact"]
    family = [r["spec_id"] for r in rows if r["registry_registration"] == "family"]
    unregistered = [r["spec_id"] for r in rows if r["registry_registration"] == "none"]

    degenerate_specs = [r["spec_id"] for r in rows if r["degenerate_T"]]
    low_coverage = [r["spec_id"] for r in rows if r["coverage_flag"] == "LOW_COVERAGE"]
    not_computable = [r["spec_id"] for r in rows if r["coverage_flag"] == "NOT_COMPUTABLE"]

    # ---------- consistency checks ----------
    checks = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    check("spec_count_20", len(specs) == 20, f"JSON specs = {len(specs)}")
    check("csv_row_count_20", len(rows) == 20, f"rows = {len(rows)}")
    check(
        "gate_md_a9_spec_lines",
        len(a9_ids) == 7 and len(set(a9_ids)) == 7,
        f"A9 spec 行 = {len(a9_ids)}（COMPUTED {len(a9_computed)} + NOT_COMPUTABLE {len(a9_not_computable)}）",
    )
    check(
        "gate_md_literal_equals_a9_ids",
        set(in_md) == set(a9_ids),
        f"literal-in-MD {sorted(in_md)} vs A9 ids {sorted(a9_ids)}",
    )

    md_beta_pairs = []
    for item in gate["a9"]:
        sid = item["spec_id"]
        spec_beta = as_float(specs.get(sid, {}).get("spec_beta"))
        md_beta = beta_in_md(item["text"])
        if spec_beta is None and md_beta is None:
            continue
        md_beta_pairs.append(
            {
                "spec_id": sid,
                "json_beta": spec_beta,
                "md_beta": md_beta,
                "abs_diff": None if (spec_beta is None or md_beta is None) else abs(spec_beta - md_beta),
            }
        )
    beta_mismatch = [p for p in md_beta_pairs if p["abs_diff"] is None or p["abs_diff"] >= 5e-5]
    check(
        "gate_md_a9_beta_matches_json",
        len(md_beta_pairs) == 6 and not beta_mismatch,
        f"可比 {len(md_beta_pairs)} 条；不匹配 {len(beta_mismatch)} 条",
    )

    four_pairs = []
    for r in rows:
        if not r["spec_id"].startswith("m1_rob_col_"):
            continue
        label = r["spec_id"][len("m1_rob_col_"):]
        md_row = gate["four_column"].get(label)
        four_pairs.append(
            {
                "spec_id": r["spec_id"],
                "json_beta": r["beta"],
                "md_beta": None if md_row is None else md_row["beta_ew"],
                "md_line": None if md_row is None else md_row["line"],
            }
        )
    four_mismatch = [
        p for p in four_pairs
        if p["md_beta"] is None or abs(p["json_beta"] - p["md_beta"]) >= 5e-5
    ]
    check(
        "four_column_beta_matches_json",
        len(four_pairs) == 4 and not four_mismatch,
        f"四列 {len(four_pairs)} 条；不匹配 {len(four_mismatch)} 条",
    )

    check(
        "registry_spec_id_count_27",
        len(reg_info["specification_ids"]) == 27,
        f"registry specification_ids = {len(reg_info['specification_ids'])}",
    )
    iqr_values = {s.get("iqr_pool") for s in specs.values()}
    check(
        "iqr_pool_single_registered_value",
        iqr_values == {iqr_pool},
        f"spec iqr_pool 值集 = {sorted(iqr_values)}；registry/JSON 登记值 = {iqr_pool}",
    )
    check(
        "coverage_buckets_reconcile",
        len(in_md) + len(not_in_md_literal) == 20
        and len(four_col_covered) + len(headline_ref) + len(true_gap) + len(in_md) == 20,
        f"in_md {len(in_md)} + not_in_md {len(not_in_md_literal)} = 20；"
        f"a9 {len(in_md)} + four_col {len(four_col_covered)} + headline {len(headline_ref)} + none {len(true_gap)} = "
        f"{len(in_md) + len(four_col_covered) + len(headline_ref) + len(true_gap)}",
    )
    check(
        "registry_buckets_reconcile",
        len(exact) + len(family) + len(unregistered) == 20,
        f"exact {len(exact)} + family {len(family)} + none {len(unregistered)} = "
        f"{len(exact) + len(family) + len(unregistered)}",
    )

    # placebos / columns / strata key readings
    def reading(sid: str) -> dict:
        r = next(x for x in rows if x["spec_id"] == sid)
        return {
            "beta": r["beta"],
            "t_nw": r["t_nw"],
            "T": r["T_months"],
            "n_rows": r["n_rows"],
            "p_one_sided": r["p_one_sided"],
            "economic_magnitude": r["economic_magnitude_beta_x_iqr"],
            "sign": r["sign"],
            "sign_vs_registered_direction": r["sign_vs_registered_direction_beta_lt_0"],
            "coverage_flag": r["coverage_flag"],
            "degenerate_T": r["degenerate_T"],
            "gate_md_coverage": r["gate_md_coverage"],
            "registry_registration": r["registry_registration"],
        }

    max_abs_econ = max(
        (r for r in rows if r["economic_magnitude_beta_x_iqr"] is not None),
        key=lambda r: abs(r["economic_magnitude_beta_x_iqr"]),
    )
    max_abs_econ_usable = max(
        (
            r
            for r in rows
            if r["economic_magnitude_beta_x_iqr"] is not None and r["coverage_flag"] == "OK"
        ),
        key=lambda r: abs(r["economic_magnitude_beta_x_iqr"]),
    )

    counts = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
        "inputs": {
            "robustness_json": {
                "path": str(ROBUSTNESS_JSON.relative_to(ROOT)).replace("\\", "/"),
                "bytes": file_size(ROBUSTNESS_JSON),
            },
            "gate_md": {
                "path": str(GATE_MD.relative_to(ROOT)).replace("\\", "/"),
                "bytes": file_size(GATE_MD),
            },
            "registry_yaml": {
                "path": str(REGISTRY_YAML.relative_to(ROOT)).replace("\\", "/"),
                "bytes": file_size(REGISTRY_YAML),
            },
        },
        "source_metadata": {
            "spec_id": robustness.get("spec_id"),
            "sample_version": robustness.get("sample_version"),
            "sample_version_v1_superseded": robustness.get("sample_version_v1_superseded"),
            "random_seed": robustness.get("random_seed"),
            "iqr_pool_registered": iqr_pool,
            "main_column": robustness.get("main_column"),
            "outcome": robustness.get("outcome"),
            "four_column_reporting": robustness.get("four_column_reporting"),
            "registry_version": reg_info["registry_version"],
        },
        "thresholds": {
            "min_effective_month_count": min_months,
            "min_focal_count_per_month": min_focal,
            "nw_lag": nw_lag,
            "K_Econ_M1": k_econ,
            "iqr_pool_registered": iqr_pool,
        },
        "total_specs": len(rows),
        "status_counts": {
            "COMPUTED": sum(1 for r in rows if r["spec_status"] == "COMPUTED"),
            "NOT_COMPUTABLE": sum(1 for r in rows if r["spec_status"] == "NOT_COMPUTABLE"),
        },
        "coverage": {
            "low_coverage_below_min_months": low_coverage,
            "not_computable": not_computable,
            "degenerate_T": degenerate_specs,
            "degenerate_detail": {r["spec_id"]: r["degenerate_T"] for r in rows if r["degenerate_T"]},
            "below_104_months": [
                {"spec_id": r["spec_id"], "T": r["T_months"]}
                for r in rows
                if isinstance(r["T_months"], int) and r["T_months"] < 104
            ],
        },
        "gate_md": {
            "a9_spec_lines_total": len(a9_ids),
            "a9_computed_picks": a9_computed,
            "a9_not_computable": a9_not_computable,
            "a9_line_numbers": {a["spec_id"]: a["line"] for a in gate["a9"]},
            "literal_in_md": in_md,
            "literal_not_in_md": not_in_md_literal,
            "literal_not_in_md_count": len(not_in_md_literal),
            "covered_by_four_column_table": four_col_covered,
            "covered_as_headline_reference": headline_ref,
            "true_coverage_gap": true_gap,
            "true_coverage_gap_count": len(true_gap),
            "beta_check": {
                "pairs": md_beta_pairs,
                "mismatches": beta_mismatch,
            },
            "four_column_beta_check": four_pairs,
        },
        "registry": {
            "specification_ids_total": len(reg_info["specification_ids"]),
            "exact": exact,
            "family_only": family,
            "family_mapping": {sid: family_of(sid) for sid in family},
            "unregistered": unregistered,
            "note": (
                "registry 的 m1_rob_limit / m1_rob_size / m1_rob_liq 为族级 id，"
                "各覆盖 3/2/2 个层规格；m1_rob_non_st 与 m1_rob_col_* 未出现在 specification_ids。"
            ),
        },
        "strata_readings": {sid: reading(sid) for sid in STRATA_FOCUS},
        "max_abs_economic_magnitude": {
            "spec_id": max_abs_econ["spec_id"],
            "beta": max_abs_econ["beta"],
            "t_nw": max_abs_econ["t_nw"],
            "economic_magnitude": max_abs_econ["economic_magnitude_beta_x_iqr"],
            "K_Econ_M1": k_econ,
            "abs_ge_K_Econ_M1": abs(max_abs_econ["economic_magnitude_beta_x_iqr"]) >= k_econ
            if k_econ is not None else None,
            "sign": max_abs_econ["sign"],
            "coverage_flag": max_abs_econ["coverage_flag"],
        },
        "max_abs_economic_magnitude_usable": {
            "spec_id": max_abs_econ_usable["spec_id"],
            "beta": max_abs_econ_usable["beta"],
            "t_nw": max_abs_econ_usable["t_nw"],
            "economic_magnitude": max_abs_econ_usable["economic_magnitude_beta_x_iqr"],
            "K_Econ_M1": k_econ,
            "abs_ge_K_Econ_M1": abs(max_abs_econ_usable["economic_magnitude_beta_x_iqr"]) >= k_econ
            if k_econ is not None else None,
            "sign": max_abs_econ_usable["sign"],
            "coverage_flag": max_abs_econ_usable["coverage_flag"],
        },
        "sign_counts": {
            "POS": sum(1 for r in rows if r["sign"] == "POS"),
            "NEG": sum(1 for r in rows if r["sign"] == "NEG"),
            "NA": sum(1 for r in rows if r["sign"] == "NA"),
        },
        "consistency_checks": checks,
        "all_checks_pass": all(c["pass"] for c in checks),
        "outputs": {
            "csv": str(OUT_CSV.relative_to(ROOT)).replace("\\", "/"),
            "counts_json": str(OUT_JSON.relative_to(ROOT)).replace("\\", "/"),
            "execution_md": str(OUT_MD.relative_to(ROOT)).replace("\\", "/"),
        },
        "constraints": [
            "未重跑 M1 / PC / 网络",
            "未重估任何系数或标准误",
            "未改 headline / 系数 / gate / configs / G0 / reports",
            "所有读数逐字取自 s3_m1_robustness_placebo.json",
        ],
    }

    # ---------- CSV ----------
    csv_fields = [
        "spec_id",
        "spec_status",
        "beta",
        "t_nw",
        "T_months",
        "n_rows",
        "p_one_sided",
        "economic_magnitude_beta_x_iqr",
        "sign",
        "sign_vs_registered_direction_beta_lt_0",
        "coverage_flag",
        "low_coverage_warning",
        "status_reason",
        "coverage_detail",
        "in_gate_md_literal",
        "gate_md_lines",
        "gate_md_coverage",
        "gate_md_coverage_detail",
        "registry_registration",
        "registry_id_or_family",
        "registry_note",
        "stratum",
        "degenerate_T",
        "spec_note",
    ]
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in csv_fields})

    OUT_JSON.write_text(json.dumps(counts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---------- execution MD ----------
    def f5(x):
        return "" if x is None else f"{x:.5f}"

    def f2(x):
        return "" if x is None else f"{x:.2f}"

    def yi(x):
        return "" if x is None else f"{x:,}"

    md = []
    md.append("# B5 / S-7：A9 全 20 行规格表与 gate 报告覆盖对照")
    md.append("")
    md.append(f"- 生成脚本：`{counts['script']}`")
    md.append(f"- 生成时间（UTC）：{counts['generated_at_utc']}")
    md.append("- 输入（只读）：")
    for key, item in counts["inputs"].items():
        md.append(f"  - `{item['path']}`（bytes `{item['bytes'][:16]}…`）")
    md.append(
        f"- 样本版本：`{robustness.get('sample_version')}`；"
        f"IQR_pool 登记值 = `{iqr_pool}`；registry 版本 = `{reg_info['registry_version']}`"
    )
    md.append(
        f"- 门槛：min_effective_month_count = {min_months}；min_focal_count_per_month = {min_focal}；"
        f"NW lag = {nw_lag}；K_Econ_M1 = {k_econ}"
    )
    md.append(
        "- 约束：**未重跑 M1 / PC / 网络，未重估任何系数**；"
        "所有 β / t / T / n_rows 逐字取自 `s3_m1_robustness_placebo.json`；"
        "未改 headline / 系数 / gate / configs / G0 / reports。"
    )
    md.append("")
    md.append("> T = `n_months_used`（FM 有效月份数，不是日历月）；n_rows = 该规格 FM 行集。")
    md.append(
        "> A9 各规格的 `spec_beta` 为 PeerMAXraw 的 EW 系数（`m1_rob_raw` = 0.03427 与 headline EW 一致）；"
        "该 JSON 未包含各 A9 规格的 VW 对应读数，VW 并报见 gate MD 的 M1-1 / M1-4 / 四列表。"
    )
    md.append("")
    md.append("## 1. 全 20 行规格表")
    md.append("")
    md.append(
        "| # | spec_id | spec_status | β | t(NW) | T | n_rows | β·IQR | sign | vs 登记方向 β<0 | coverage | gate MD | registry |"
    )
    md.append("|---:|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        gate_cell = f"是（L{r['gate_md_lines']}）" if r["in_gate_md_literal"] else "否"
        reg_cell = {
            "exact": "exact",
            "family": f"family:{r['registry_id_or_family']}",
            "none": "未登记",
        }[r["registry_registration"]]
        md.append(
            f"| {i} | `{r['spec_id']}` | {r['spec_status']} | {f5(r['beta'])} | {f2(r['t_nw'])} | "
            f"{r['T_months']} | {yi(r['n_rows'])} | {f5(r['economic_magnitude_beta_x_iqr'])} | "
            f"{r['sign']} | {r['sign_vs_registered_direction_beta_lt_0']} | {r['coverage_flag']} | "
            f"{gate_cell} | {reg_cell} |"
        )
    md.append("")
    md.append(
        f"状态合计：COMPUTED = {counts['status_counts']['COMPUTED']}，"
        f"NOT_COMPUTABLE = {counts['status_counts']['NOT_COMPUTABLE']}；"
        f"sign：POS = {counts['sign_counts']['POS']}，NEG = {counts['sign_counts']['NEG']}，"
        f"NA = {counts['sign_counts']['NA']}。"
    )
    md.append("")

    md.append("## 2. 重点分层读数（liq / size / limit / non_st）")
    md.append("")
    md.append("| spec_id | 层 | β | t(NW) | T | n_rows | β·IQR | sign | 单侧 p | 读法 |")
    md.append("|---|---|---:|---:|---:|---:|---:|---|---:|---|")
    readings_text = {
        "m1_rob_liq_low": "负号、t=-1.26、单侧 p=0.1035；T=89（<104，仍 ≥60，无 low_coverage_warning）。",
        "m1_rob_liq_high": (
            "唯一可用月份 ≥60 且 abs(β·IQR) ≥ K_Econ_M1 的规格（0.00303 vs 0.003；"
            "all_limit 的 0.01687 因 T=2 退化不计），"
            "但符号为**正**（与登记方向 β<0 相反）、t=1.58、单侧 p=0.9426"
            "→ 只作异质性描述，**不得**当作 M1-3 通过或用于补救 M1。"
        ),
        "m1_rob_size_low": "正向、t=1.02；小市值层点估计更大但未显著。",
        "m1_rob_size_high": "正向、t=0.49。",
        "m1_rob_limit_all_limit": (
            "**T=2 退化**：T < min_effective_month_count=60 且 T ≤ NW lag=6，"
            "β=0.72073 / t=36.50 由退化协方差给出，不可作推断依据；"
            "JSON 已带 low_coverage_warning。"
        ),
        "m1_rob_limit_all_non_limit": "T=96；可用月份 ≥60，是 all_non_limit 层的唯一读数，β=0.01934。",
        "m1_rob_limit_mixed": "T=104；β=0.02592，与 raw 同号同量级。",
        "m1_rob_non_st": (
            "β=0.03457 vs raw 0.03427（Δ=+0.00030）；n=180,108 = raw 181,178 的 99.41%；"
            "ST 剔除对 headline 无实质影响。"
        ),
        "m1_rob_st": (
            "**NOT_COMPUTABLE**：T=0，n_rows=1,070（层内月样本中位 9 < min_focal_count_per_month=100）；"
            "status_reason=ALL_MONTHS_BELOW_MIN_FOCAL_COUNT。"
        ),
    }
    for sid in STRATA_FOCUS:
        r = next(x for x in rows if x["spec_id"] == sid)
        md.append(
            f"| `{sid}` | {r['stratum']} | {f5(r['beta'])} | {f2(r['t_nw'])} | {r['T_months']} | "
            f"{yi(r['n_rows'])} | {f5(r['economic_magnitude_beta_x_iqr'])} | {r['sign']} | "
            f"{f5(r['p_one_sided'])} | {readings_text[sid]} |"
        )
    md.append("")
    md.append(
        "符号口径：登记方向为 β<0（M1-1 要求 EW 与 VW 同为负）。`sign` 只记读数正负，"
        "`vs 登记方向` 的 same/opposite 是符号台账，不是新的统计判定。"
    )
    md.append("")

    md.append("## 3. T 退化规格")
    md.append("")
    md.append(
        f"- `m1_rob_st`：T=0，spec_status=NOT_COMPUTABLE，n_rows=1,070。"
        f"原因见 status_reason：ALL_MONTHS_BELOW_MIN_FOCAL_COUNT"
        f"（min_focal_count_per_month={min_focal}；层内月样本中位 9；n_usable_months=0）。"
    )
    md.append(
        f"- `m1_rob_limit_all_limit`：T=2，**低于 min_effective_month_count={min_months}，"
        f"也低于 NW lag={nw_lag}**。JSON 的 t=36.50 是退化 NW 协方差的产物，不是可信推断；"
        "该规格只能作为样本覆盖率记录。"
    )
    md.append(
        "- 其余规格 T 均 ≥60：liq_low 89、limit_all_non_limit 96、placebo_timeshift 92、四列 ma/mb 68；"
        "低于 104 但未触发 low_coverage_warning。"
    )
    md.append("")

    md.append("## 4. 与 gate MD 的覆盖对照")
    md.append("")
    md.append(
        f"**实际 A9 行 = {len(a9_ids)} 条（{len(a9_computed)} 条 COMPUTED + "
        f"{len(a9_not_computable)} 条 NOT_COMPUTABLE），不是 6 条。**"
        f"gate MD 第 {min(a['line'] for a in gate['a9'])}–{max(a['line'] for a in gate['a9'])} 行逐条列出："
        + "、".join(f"`{sid}`(L{counts['gate_md']['a9_line_numbers'][sid]})" for sid in a9_ids)
        + "。6 个「picks」是其中的 COMPUTED 子集，NOT_COMPUTABLE 的 `m1_rob_st` 也占一行。"
    )
    md.append("")
    md.append(
        f"按 spec_id 逐字 grep：**{len(in_md)} 条出现、{len(not_in_md_literal)} 条未出现**。"
        f"未出现的 {len(not_in_md_literal)} 条中："
        f"{len(four_col_covered)} 条四列规格由 MD 的四列并报表（L59–62）以列名覆盖且 β 对得上；"
        f"{len(headline_ref)} 条 `m1_rob_raw` 的 β 作为 headline / placebo 对照读数出现（但未以 spec_id 形式列出）；"
        f"剩余 **{len(true_gap)} 条为真实报告覆盖缺口**（无 spec id 行，也无对应表格行）。"
    )
    md.append("")
    md.append("### 4.1 未进 MD 的 8 条（真实缺口）")
    md.append("")
    md.append("| spec_id | 层 | β | t(NW) | T | n_rows | sign | β·IQR |")
    md.append("|---|---|---:|---:|---:|---:|---|---:|")
    for sid in true_gap:
        r = next(x for x in rows if x["spec_id"] == sid)
        md.append(
            f"| `{sid}` | {r['stratum']} | {f5(r['beta'])} | {f2(r['t_nw'])} | {r['T_months']} | "
            f"{yi(r['n_rows'])} | {r['sign']} | {f5(r['economic_magnitude_beta_x_iqr'])} |"
        )
    md.append("")
    md.append("### 4.2 13 条按 spec_id 未出现者的归类")
    md.append("")
    md.append("| spec_id | gate MD 覆盖方式 |")
    md.append("|---|---|")
    for sid in not_in_md_literal:
        r = next(x for x in rows if x["spec_id"] == sid)
        md.append(f"| `{sid}` | {r['gate_md_coverage_detail']} |")
    md.append("")
    md.append(
        "**这是报告覆盖问题，不是重算问题。** 上述 13 条的读数本来就写在 "
        "`s3_m1_robustness_placebo.json` 里；gate MD 只是没有逐条列出。"
        "本表不改变任何 gate 判定、headline 或 S5_status；也不把未列出的规格读作「通过」。"
    )
    md.append("")

    md.append("## 5. Registry 登记核对")
    md.append("")
    md.append(
        f"`m1_sample_and_spec_registry.headline_value.specification_ids` 共 "
        f"{len(reg_info['specification_ids'])} 个 id。20 个 A9 规格的登记状态："
    )
    md.append("")
    md.append(f"- **exact（{len(exact)}）**：" + "、".join(f"`{x}`" for x in exact))
    md.append(
        f"- **仅族级（{len(family)}）**："
        + "、".join(f"`{x}`→`{family_of(x)}`" for x in family)
    )
    md.append(f"- **未登记（{len(unregistered)}）**：" + "、".join(f"`{x}`" for x in unregistered))
    md.append("")
    md.append(
        "- 族级问题：`m1_rob_limit` / `m1_rob_size` / `m1_rob_liq` 各只登记一个族 id，"
        "但实际产物有 3 / 2 / 2 个层规格；层 id 无法从 `specification_ids` 直接追溯到产物。"
    )
    md.append(
        "- ST 族只登记 `m1_rob_st`，没有 `m1_rob_non_st`；"
        "四列规格（`m1_rob_col_*`）在 registry 文本中有「四列并报」设计说明，但 `specification_ids` 无对应 id。"
    )
    md.append(
        "- 本核对只读 registry，未改任何登记；上述缺口属登记粒度/收录问题，需另行裁定，不在本任务改动范围。"
    )
    md.append("")

    md.append("## 6. 最值得补报的 3 条")
    md.append("")
    md.append(
        "1. **`m1_rob_liq_low` / `m1_rob_liq_high`（符号分裂 + 量级阈值边缘）**："
        "liq_low β=-0.09948（t=-1.26，T=89），liq_high β=+0.12940（t=1.58，T=104）；"
        "liq_high 的 abs(β·IQR)=0.00303 是可用月份 ≥60 的规格中唯一 ≥ K_Econ_M1=0.003 的读数"
        "（all_limit 的 0.01687 因 T=2 退化不计），"
        "但符号为正、单侧 p=0.9426。补报时必须同时写符号与「不得当作 M1-3 通过/补救」，"
        "否则容易被 abs() 规则误读。"
    )
    md.append(
        "2. **`m1_rob_limit_all_limit`（T=2 退化）**：β=0.72073、t=36.50、T=2、n_rows=2,303。"
        "T 既低于 min_effective_month_count=60，也低于 NW lag=6；"
        "当前 MD 未列它，等于默认不引用——安全但无记录。补一行标注「退化、不可推断」，"
        "可防止后续有人从 JSON 里单独捞出这个 t=36.50 当作强证据。"
    )
    md.append(
        "3. **`m1_rob_non_st`（ST 剔除的无关性检查）**：β=0.03457 vs raw 0.03427"
        "（Δ=+0.00030），T=104，n_rows=180,108（raw 的 99.41%）。"
        "该读数直接回答「ST 分层处理是否驱动 headline」，且与 raw 几乎相同，"
        "适合作为一行 coverage check 补进 MD。"
    )
    md.append("")
    md.append(
        f"补充参考：全部规格中 abs(β·IQR) 最大者为 "
        f"`{counts['max_abs_economic_magnitude']['spec_id']}`"
        f"（{f5(counts['max_abs_economic_magnitude']['economic_magnitude'])}，"
        f"sign={counts['max_abs_economic_magnitude']['sign']}，"
        f"coverage={counts['max_abs_economic_magnitude']['coverage_flag']}）；"
        f"限定可用月份 ≥60 后最大者为 "
        f"`{counts['max_abs_economic_magnitude_usable']['spec_id']}`"
        f"（{f5(counts['max_abs_economic_magnitude_usable']['economic_magnitude'])}，"
        f"sign={counts['max_abs_economic_magnitude_usable']['sign']}，"
        f"abs ≥ K_Econ_M1 = {counts['max_abs_economic_magnitude_usable']['abs_ge_K_Econ_M1']}）。"
    )
    md.append("")

    md.append("## 7. 一致性与校验")
    md.append("")
    md.append("| check | pass | detail |")
    md.append("|---|---|---|")
    for c in checks:
        md.append(f"| {c['check']} | {c['pass']} | {c['detail']} |")
    md.append("")
    md.append(f"全部校验通过：**{counts['all_checks_pass']}**。")
    md.append("")
    md.append("## 8. 输出")
    md.append("")
    md.append(f"- `{counts['outputs']['csv']}`（20 行 + 表头，UTF-8-SIG）")
    md.append(f"- `{counts['outputs']['counts_json']}`")
    md.append(f"- `{counts['outputs']['execution_md']}`（本件）")
    md.append("")
    md.append(
        "复现：`python scripts/shadow_robustness_table.py`（只读三份输入，重写上述三件产物）。"
    )
    md.append("")

    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    summary = {
        "outputs": counts["outputs"],
        "total_specs": counts["total_specs"],
        "status_counts": counts["status_counts"],
        "gate_md_a9_lines": len(a9_ids),
        "literal_not_in_md_count": len(not_in_md_literal),
        "true_coverage_gap_count": len(true_gap),
        "registry": {
            "exact": len(exact),
            "family_only": len(family),
            "unregistered": len(unregistered),
        },
        "degenerate_T": degenerate_specs,
        "all_checks_pass": counts["all_checks_pass"],
    }
    print(json.dumps(summary, ensure_ascii=False))
    if not counts["all_checks_pass"]:
        log("WARNING: consistency checks failed; see counts JSON")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
