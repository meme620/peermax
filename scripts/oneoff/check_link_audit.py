#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S0 registry_link_audit 校验脚本（纯标准库，不依赖 yaml 包）。

读 scope_card.yaml + claim_registry.yaml + parameter_registry.yaml（name 列表），检查：
  1. 每个 claim 的 7 类连接齐全（treatment/outcome/unit/horizon/formula/input_field/required_gate）；
  2. formula_id 非空且无重复主版本（candidate 状态除外）；
  3. scope_card 与 claim_registry 一致性（M1_headline 的 treatment/outcome 必须等于 scope_card 的 primary_treatment/primary_outcome）；
  4. required_gates 均在 G1-G8 范围内；
  5. input_tables/input_fields 非空。
输出 JSON 报告；exit code 0=PASS（无 FAIL），1=FAIL。

用法: python scripts/data_audit/check_link_audit.py [--out reports/execution/stage0/registry_link_audit_report.json]
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCOPE_CARD = ROOT / "configs" / "scope_card.yaml"
CLAIM_REGISTRY = ROOT / "configs" / "claim_registry.yaml"
PARAM_REGISTRY = ROOT / "configs" / "parameter_registry.yaml"
VALID_GATES = {f"G{i}" for i in range(1, 9)}
LINK_TYPES = ["treatment", "outcome", "unit", "horizon", "formula", "input_field", "required_gate"]


def read_yaml_flat(path):
    """轻量解析受控结构 yaml：返回 {key: value}；嵌套列表按顶层 key 分组原样保留文本。"""
    text = path.read_text(encoding="utf-8")
    result = {}
    current_key = None
    current_block = []
    for line in text.splitlines():
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*:", line):
            if current_key is not None:
                result[current_key] = "\n".join(current_block).strip()
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)", line)
            current_key, rest = m.group(1), m.group(2)
            current_block = [rest] if rest else []
        elif line.startswith("  ") and current_key is not None:
            current_block.append(line.strip())
        elif current_key is not None and not line.strip():
            current_block.append("")
    if current_key is not None:
        result[current_key] = "\n".join(current_block).strip()
    return result


def parse_claims(text):
    """从 claim_registry.yaml 提取 claim 块列表：按 '- claim_id:' 切块，每块转 dict。"""
    blocks = re.split(r"(?m)^  - claim_id:", text)
    claims = []
    for b in blocks[1:]:
        lines = b.splitlines()
        d = {}
        for i, line in enumerate(lines):
            m = re.match(r"^    ([a-z_]+):\s*(.*)", line)
            if m:
                key, val = m.group(1), m.group(2).strip()
                if key == "formula_id":
                    # 收集后续列表项
                    items = [val] if val else []
                    for nxt in lines[i + 1:]:
                        nm = re.match(r"^      - \"?(.+?)\"?$", nxt)
                        if nm:
                            items.append(nm.group(1).strip())
                        else:
                            break
                    d[key] = items
                elif key in ("input_tables", "input_fields", "required_gates", "blocked_claims"):
                    # 支持行内数组 ["A", "B", ...] 与块状列表两种写法
                    if val.startswith("[") and val.endswith("]"):
                        items = [x.strip().strip('"').strip("'") for x in val[1:-1].split(",") if x.strip()]
                    else:
                        items = [val] if val else []
                        for nxt in lines[i + 1:]:
                            nm = re.match(r"^      - \"?(.+?)\"?$", nxt)
                            if nm:
                                items.append(nm.group(1).strip())
                            else:
                                break
                    d[key] = items
                else:
                    d[key] = val
        if d:
            claims.append(d)
    return claims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "reports/execution/stage0/registry_link_audit_report.json"))
    args = ap.parse_args()

    scope = read_yaml_flat(SCOPE_CARD)
    claims_text = CLAIM_REGISTRY.read_text(encoding="utf-8")
    claims = parse_claims(claims_text)
    param_text = PARAM_REGISTRY.read_text(encoding="utf-8")
    registry_names = set(re.findall(r"^  - name: (\S+)", param_text, re.M))

    links = []
    fails = []

    def add(claim_id, link_type, status, detail=""):
        links.append({"claim_id": claim_id, "link_type": link_type,
                      "link_status": status, "detail": detail})

    if not claims:
        fails.append("claim_registry: 未解析到任何 claim 块")
    for c in claims:
        cid = c.get("claim_id", "?")
        is_candidate = c.get("status", "").strip() == "candidate"
        # 1) 七类连接齐全
        for t in LINK_TYPES:
            if t == "formula":
                ok = bool(c.get("formula_id"))
                detail = ";".join(c.get("formula_id") or [])
            elif t == "input_field":
                ok = bool(c.get("input_fields"))
                detail = str(len(c.get("input_fields") or [])) + " fields"
            elif t == "required_gate":
                ok = bool(c.get("required_gates"))
                detail = ",".join(c.get("required_gates") or [])
            else:
                ok = bool(c.get(t))
                detail = str(c.get(t, ""))[:60]
            if not ok:
                fails.append(f"{cid}: 缺 {t} 连接")
            add(cid, t, "PASS" if ok else "FAIL", detail)
        # 2) formula 主版本唯一（active 状态要求唯一；candidate 允许多个候选）
        formulas = c.get("formula_id") or []
        if not is_candidate and len(set(formulas)) != len(formulas):
            fails.append(f"{cid}: formula_id 重复")
        # 3) gates 范围
        gates = c.get("required_gates") or []
        bad = [g for g in gates if g not in VALID_GATES]
        if bad:
            fails.append(f"{cid}: 非法 gate {bad}")
        # 4) input_tables 非空
        if not c.get("input_tables"):
            fails.append(f"{cid}: input_tables 为空")

    # 5) scope_card 一致性（M1_headline ↔ primary_treatment/primary_outcome）
    m1 = next((c for c in claims if c.get("claim_id") == "M1_headline"), None)
    if m1:
        if m1.get("treatment", "").strip() != scope.get("primary_treatment", "").strip():
            fails.append("M1_headline.treatment 与 scope_card.primary_treatment 不一致")
        if m1.get("outcome", "").strip() != scope.get("primary_outcome", "").strip():
            fails.append("M1_headline.outcome 与 scope_card.primary_outcome 不一致")
        add("M1_headline", "scope_card_link", "PASS", "treatment/outcome 一致")
    # 6) registry 可读
    if not registry_names:
        fails.append("parameter_registry: 未解析到参数名")

    status = "FAIL" if fails else "PASS"
    report = {
        "report_id": "registry_link_audit",
        "status": status,
        "scope_card_id": scope.get("scope_card_id", ""),
        "claim_registry_version": read_yaml_flat(CLAIM_REGISTRY).get("claim_registry_version", ""),
        "parameter_registry_version": scope.get("parameter_registry_version", ""),
        "n_claims": len(claims),
        "n_links": len(links),
        "link_details": links,
        "failures": fails,
        "next_step_allowed": "S1" if status == "PASS" else "S0（修复后重跑）",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "link_details"},
                     ensure_ascii=False, indent=2))
    print(f"\n输出: {out}")
    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
