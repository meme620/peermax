#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B4 / S-6：A9 稳健性族 BH / Bonferroni 诊断（只读输入，只写 3b_s6_*）。

输入（只读）
------------
reports/data_gates/G0/pricing_robustness_placebo.json
  20 个 spec：19 COMPUTED（含 t_nw / spec_beta / n_months_used）+ 1 NOT_COMPUTABLE（pricing_rob_st）。

输出（仅 artifacts/phase3b/）
-----------------------------
3b_s6_bh.csv        逐 spec：beta / t / T / 双侧 p / BH rank / q（主族 m=19 与敏感性 m=18）
3b_s6_counts.json   族计数、最小 p/q、存活规格、阈值、依赖性与近似交叉核对

方法
----
- 双侧 p：正态近似 p = erfc(|t_nw| / sqrt(2)) = 2 * (1 - Phi(|t_nw|))。
  输入 JSON 的 p_one_sided 为左尾 Phi(t)（对应登记方向 beta<0），本脚本不使用该列。
- BH：升序排序，q_(i) = min_{j>=i} p_(j) * m / j（末位起单调化），q <= alpha 拒绝；
  step-up 的拒绝集 = 满足 p_(i) <= i*alpha/m 的最大 i 及其之前全部。
- Bonferroni：alpha / m。
- 主族 m = 19（全部 COMPUTED）；敏感性族 m = 18（剔除 T = n_months_used = 2 的退化规格
  pricing_rob_limit_all_limit；阈值 60 = registry min_effective_month_count）。
- NOT_COMPUTABLE 规格不进入任何族（不按 p=1 计入）。
- 输入 bytes 必须等于 version_manifest 登记值，且与既有 3b_s6_counts.json 记录一致，
  否则 SystemExit（防止输入漂移后静默产出不可比的 q 值）。

不改 headline / 系数 / gate / configs / G0 / reports；不重跑 pricing_gate/PC/网络。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from fileio import file_size

BASE = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BASE / "reports" / "data_gates" / "G0" / "pricing_robustness_placebo.json"
DEFAULT_OUT_DIR = BASE / "artifacts" / "phase3b"
DEFAULT_ALPHA = 0.05
MIN_EFFECTIVE_MONTHS = 60  # registry min_effective_month_count
OUT_PREFIX = "3b_s6_"
# configs/version_manifest.yaml 登记值（file: reports/data_gates/G0/pricing_robustness_placebo.json）
EXPECTED_INPUT_bytes = "06d440aad63644e7b79e7f32512cb075bee5d09ffbd36d08afa03a91c359869b"

PLACEBO_SPECS = {"pricing_placebo_random", "pricing_placebo_timeshift"}


def two_sided_p_normal(t: float) -> float:
    """双侧正态近似 p。"""
    return math.erfc(abs(t) / math.sqrt(2.0))


def two_sided_p_t(t: float, df: int) -> float | None:
    """可选交叉核对：t(df) 双侧 p；无 scipy 时返回 None。"""
    if df <= 0:
        return None
    try:
        from scipy import stats  # noqa: PLC0415

        return float(2.0 * stats.t.sf(abs(t), df))
    except Exception:  # pragma: no cover - 无 scipy 环境
        if df == 1:  # 闭式：P(|T_1| > t) = 1 - 2*arctan(t)/pi
            return float(1.0 - 2.0 * math.atan(abs(t)) / math.pi)
        return None


def bh_adjust(pvals: dict[str, float], alpha: float) -> dict:
    """BH step-up + 单调化 q。返回排序、rank、q、拒绝集。"""
    m = len(pvals)
    order = sorted(pvals.items(), key=lambda kv: (kv[1], kv[0]))
    q: dict[str, float] = {}
    prev = 1.0
    for i in range(m, 0, -1):
        spec, p = order[i - 1]
        prev = min(prev, p * m / i)
        q[spec] = prev
    ranks = {spec: i for i, (spec, _) in enumerate(order, 1)}
    k = 0
    for i, (_, p) in enumerate(order, 1):
        if p <= i * alpha / m:
            k = i
    rejected = [spec for spec, _ in order[:k]]
    return {
        "m": m,
        "alpha": alpha,
        "bonferroni_threshold": alpha / m,
        "order": order,
        "ranks": ranks,
        "q": q,
        "rejected": rejected,
        "bh_reject_cutoff_p": order[k - 1][1] if k else None,
    }


def summarize(adj: dict, degenerate: set[str]) -> dict:
    order = adj["order"]
    q = adj["q"]
    min_p_spec, min_p = order[0]
    min_q_spec = min(q, key=lambda s: (q[s], s))
    p_lt = [(s, p) for s, p in order if p < adj["alpha"]]
    q_le = [(s, q[s]) for s, _ in order if q[s] <= adj["alpha"]]
    bonf = [(s, p) for s, p in order if p <= adj["bonferroni_threshold"]]
    nondeg = [(s, p) for s, p in order if s not in degenerate]
    min_p_nd_spec, min_p_nd = nondeg[0]
    min_q_nd_spec = min(
        (s for s, _ in nondeg), key=lambda s: (q[s], s)
    )
    return {
        "m": adj["m"],
        "alpha": adj["alpha"],
        "bonferroni_threshold": adj["bonferroni_threshold"],
        "min_p": min_p,
        "min_p_spec": min_p_spec,
        "min_q": q[min_q_spec],
        "min_q_spec": min_q_spec,
        "min_p_nondegenerate": min_p_nd,
        "min_p_nondegenerate_spec": min_p_nd_spec,
        "min_q_nondegenerate": q[min_q_nd_spec],
        "min_q_nondegenerate_spec": min_q_nd_spec,
        "n_unadjusted_p_lt_alpha": len(p_lt),
        "unadjusted_p_lt_alpha": [{"spec": s, "p": p} for s, p in p_lt],
        "n_q_le_alpha": len(q_le),
        "q_le_alpha_specs": [{"spec": s, "q": v} for s, v in q_le],
        "n_bonferroni_reject": len(bonf),
        "bonferroni_reject_specs": [{"spec": s, "p": p} for s, p in bonf],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="A9 稳健性族 BH/Bonferroni 诊断（只读 G0）")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = ap.parse_args()

    src = args.input.resolve()
    out_dir = args.out_dir.resolve()
    allowed_root = (BASE / "artifacts" / "phase3b").resolve()
    if not out_dir.is_relative_to(allowed_root):
        raise SystemExit(f"拒绝写出：out-dir 必须位于 {allowed_root} 之下（得到 {out_dir}）")
    if not src.is_file():
        raise SystemExit(f"输入不存在：{src}")

    raw = src.read_bytes()
    actual_bytes = file_size(raw)
    if actual_bytes != EXPECTED_INPUT_bytes:
        raise SystemExit(
            f"输入 bytes 与登记值不一致：{actual_bytes} != {EXPECTED_INPUT_bytes}（{src}）"
        )
    counts_path = out_dir / f"{OUT_PREFIX}counts.json"
    if counts_path.is_file():
        try:
            prior = json.loads(counts_path.read_text(encoding="utf-8"))
            prior_bytes = (prior.get("input") or {}).get("bytes")
        except (json.JSONDecodeError, OSError):
            prior_bytes = None
        if prior_bytes and prior_bytes != actual_bytes:
            raise SystemExit(
                f"输入 bytes 与既有 {counts_path.name} 记录不一致："
                f"{actual_bytes} != {prior_bytes}（{counts_path}）"
            )
    data = json.loads(raw.decode("utf-8"))
    specs = data["specs"]
    if len(specs) != 20:
        raise SystemExit(f"spec 数异常：{len(specs)}（预期 20）")

    computed, not_computable = {}, []
    for spec, rec in specs.items():
        status = rec.get("spec_status")
        if status == "COMPUTED":
            t = rec.get("t_nw")
            if t is None:
                raise SystemExit(f"COMPUTED 规格缺 t_nw：{spec}")
            computed[spec] = rec
        else:
            not_computable.append(
                {
                    "spec": spec,
                    "spec_status": status,
                    "status_reason": rec.get("status_reason"),
                    "n_months_used": rec.get("n_months_used"),
                }
            )

    if len(computed) != 19:
        raise SystemExit(f"COMPUTED 数异常：{len(computed)}（预期 19）")

    t_by_spec = {s: float(r["t_nw"]) for s, r in computed.items()}
    p_by_spec = {s: two_sided_p_normal(t) for s, t in t_by_spec.items()}
    degenerate = {s for s, r in computed.items() if int(r["n_months_used"]) < MIN_EFFECTIVE_MONTHS}
    if degenerate != {"pricing_rob_limit_all_limit"}:
        raise SystemExit(f"退化集合异常：{sorted(degenerate)}")

    adj19 = bh_adjust(p_by_spec, args.alpha)
    p18 = {s: p for s, p in p_by_spec.items() if s not in degenerate}
    adj18 = bh_adjust(p18, args.alpha)
    sum19 = summarize(adj19, degenerate)
    sum18 = summarize(adj18, degenerate)

    # 可选交叉核对：t(T-1) 双侧 p（仅诊断，不替代主 p）
    t_cross = {}
    for s, r in computed.items():
        df = int(r["n_months_used"]) - 1
        t_cross[s] = two_sided_p_t(t_by_spec[s], df)
    nd_diffs = [
        abs(t_cross[s] - p_by_spec[s])
        for s in computed
        if s not in degenerate and t_cross.get(s) is not None
    ]

    # 写出 CSV（按 m=19 BH rank 排序）
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{OUT_PREFIX}bh.csv"
    rows = []
    for spec, p in adj19["order"]:
        r = computed[spec]
        rows.append(
            {
                "spec": spec,
                "family_role": "placebo" if spec in PLACEBO_SPECS else "same_hypothesis",
                "spec_status": r["spec_status"],
                "beta": r["spec_beta"],
                "t_nw": t_by_spec[spec],
                "n_months": int(r["n_months_used"]),
                "p_two_sided": p,
                "bh_rank_m19": adj19["ranks"][spec],
                "bh_q_m19": adj19["q"][spec],
                "bh_rank_m18": adj18["ranks"].get(spec, ""),
                "bh_q_m18": adj18["q"].get(spec, ""),
                "degenerate_T_lt_60": spec in degenerate,
                "note": r.get("note", ""),
            }
        )
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(
                {
                    k: (
                        f"{v:.12g}"
                        if isinstance(v, float)
                        else ("" if v is None else v)
                    )
                    for k, v in row.items()
                }
            )

    signs = {
        "beta_negative": [s for s, r in computed.items() if float(r["spec_beta"]) < 0],
        "beta_positive": [s for s, r in computed.items() if float(r["spec_beta"]) > 0],
    }
    same_hyp = [s for s in computed if s not in PLACEBO_SPECS]
    same_hyp_nondeg = [s for s in same_hyp if s not in degenerate]
    n_rows = {s: int(r["n_rows"]) for s, r in computed.items()}
    limit_rows = {
        "pricing_rob_limit_all_limit": n_rows["pricing_rob_limit_all_limit"],
        "pricing_rob_limit_mixed": n_rows["pricing_rob_limit_mixed"],
        "pricing_rob_limit_all_non_limit": n_rows["pricing_rob_limit_all_non_limit"],
    }
    limit_sum = sum(limit_rows.values())
    full_sample = n_rows["pricing_rob_raw"]
    if limit_sum != full_sample:
        raise SystemExit(
            f"limit 三层 n_rows 之和 {limit_sum} != 主样本 {full_sample}"
        )
    notes = {
        "duplicate_pair": {
            "specs": ["pricing_rob_raw", "pricing_rob_col_ma_teacher"],
            "evidence": "beta/t/T/n_rows 逐值相同（同一主规格在主表与四列并报中各出现一次）",
            "consequence": "族内重复计入同一检验；m 多计 1，BH 略保守",
        },
        "sample_partition": {
            "limit_strata": {
                **limit_rows,
                "sum": limit_sum,
                "full_sample": full_sample,
                "note": "三规格构成对主样本的完全划分（互斥），p 值强负相关/互补",
            },
            "size_strata": {
                "pricing_rob_size_low": n_rows["pricing_rob_size_low"],
                "pricing_rob_size_high": n_rows["pricing_rob_size_high"],
                "note": "仅低/高三分位，中三分位未入族",
            },
            "liq_strata": {
                "pricing_rob_liq_low": n_rows["pricing_rob_liq_low"],
                "pricing_rob_liq_high": n_rows["pricing_rob_liq_high"],
                "note": "仅低/高三分位，中三分位未入族",
            },
            "st_strata": {
                "pricing_rob_st": "NOT_COMPUTABLE",
                "pricing_rob_non_st": n_rows["pricing_rob_non_st"],
                "note": "ST 层 n_usable_months=0，non_st 约等于全样本",
            },
        },
        "nested": "全部规格共享同一 FM 引擎、同一 outcome/controls/NW(6)，仅替换一个维度；各规格 t 序列高度相关（无法从汇总结果估计相关阵）",
        "placebo_null": "pricing_placebo_random / pricing_placebo_timeshift 的零假设与 treatment 系数族不同（网络被打乱/错位），严格意义上不属于同一检验族",
        "family_not_preregistered": "registry multiple_comparison_method 的 scope 仅覆盖定价 gate 与经济量级门与 cutoff C-1~C-4 族；A9 规格族为本次诊断按可用 spec 列表定义，非预注册族",
        "t_approximation": {
            "primary": "normal",
            "note": "normal 与 t(T-1) 的 |Δp| 在非退化规格上很小；退化规格 T=2 时 NW(6) 本身不可识别（lag>T-1），p 无推断意义",
            "max_abs_p_diff_nondegenerate": max(nd_diffs) if nd_diffs else None,
            "degenerate_p_t_df1": t_cross.get("pricing_rob_limit_all_limit"),
        },
    }

    counts = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/shadow_multiple_testing.py",
        "input": {
            "path": str(src.relative_to(BASE)),
            "bytes": file_size(raw),
            "spec_id": data.get("spec_id"),
            "sample_version": data.get("sample_version"),
        },
        "method": {
            "p_two_sided": "erfc(|t_nw|/sqrt(2)) = 2*(1-Phi(|t_nw|))（正态近似）",
            "bh": "q_(i)=min_{j>=i} p_(j)*m/j；q<=alpha 拒绝；step-up 同判",
            "bonferroni": "alpha/m",
            "alpha": args.alpha,
            "degenerate_rule": f"n_months_used < {MIN_EFFECTIVE_MONTHS}（min_effective_month_count）",
        },
        "family": {
            "specs_total": len(specs),
            "n_computed": len(computed),
            "n_not_computable": len(not_computable),
            "not_computable": not_computable,
            "n_same_hypothesis": len(same_hyp),
            "n_placebo": len(PLACEBO_SPECS),
            "degenerate_specs": sorted(degenerate),
            "m_primary": adj19["m"],
            "m_sensitivity": adj18["m"],
        },
        "primary_m19": sum19,
        "sensitivity_m18": sum18,
        "signs": {
            "n_beta_negative": len(signs["beta_negative"]),
            "beta_negative": signs["beta_negative"],
            "n_beta_positive": len(signs["beta_positive"]),
            "registered_direction": "beta<0（sign/one_sided_significance 单侧左尾）",
            "negative_same_hypothesis_nondegenerate_specs": [
                s for s in signs["beta_negative"] if s in same_hyp_nondeg
            ],
            "n_same_hypothesis_nondegenerate": len(same_hyp_nondeg),
        },
        "dependence_and_limits": notes,
        "per_spec": [
            {
                "spec": spec,
                "beta": computed[spec]["spec_beta"],
                "t_nw": t_by_spec[spec],
                "n_months": int(computed[spec]["n_months_used"]),
                "p_two_sided": p_by_spec[spec],
                "bh_rank_m19": adj19["ranks"][spec],
                "bh_q_m19": adj19["q"][spec],
                "bh_rank_m18": adj18["ranks"].get(spec),
                "bh_q_m18": adj18["q"].get(spec),
                "degenerate": spec in degenerate,
            }
            for spec, _ in adj19["order"]
        ],
    }

    json_path = out_dir / f"{OUT_PREFIX}counts.json"
    json_path.write_text(json.dumps(counts, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print(f"[3b_s6] wrote {csv_path.relative_to(BASE)}", file=sys.stderr)
    print(f"[3b_s6] wrote {json_path.relative_to(BASE)}", file=sys.stderr)
    print(
        f"[3b_s6] m19 min_q={sum19['min_q']:.6g} ({sum19['min_q_spec']}); "
        f"m18 min_q={sum18['min_q']:.6g} ({sum18['min_q_spec']})",
        file=sys.stderr,
    )
    print(
        f"[3b_s6] m19 reject={sum19['q_le_alpha_specs']}; "
        f"m18 reject={sum18['q_le_alpha_specs']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
