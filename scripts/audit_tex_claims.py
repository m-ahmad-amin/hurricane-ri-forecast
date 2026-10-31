from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tex = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
c = json.loads((ROOT / "outputs/metrics/cautionary_metrics.json").read_text(encoding="utf-8"))
issues: list[str] = []

for f in re.findall(r"includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
    if not (ROOT / "paper/figures" / f).exists():
        issues.append(f"MISSING FIG {f}")

bib = (ROOT / "paper/references.bib").read_text(encoding="utf-8")
for block in re.findall(r"\\cite[a-z]?\{([^}]+)\}", tex):
    for k in block.split(","):
        k = k.strip()
        if not re.search(r"\{" + re.escape(k) + r"\s*,", bib):
            issues.append(f"MISSING BIB {k}")

for b in [
    "ensemble-fair",
    "Earlier draft",
    "untuned GBM",
    "closes the gap",
    "closes that gap",
    "Fusion primary",
    "$--$",
    "near-zero / positive",
    "competent weighted imagery/fusion",
]:
    if b in tex:
        issues.append(f"BAD PHRASE: {b}")

if "Murphy decomposition is reported" in tex:
    issues.append("Promises Murphy numbers in Methods but Results omit them")
if "PR-AUC" in tex and "PR-AUC" not in tex.split("\\section{Results}")[-1]:
    # mentioned only before results
    if "PR-AUC" in tex.split("\\section{Results}")[0]:
        issues.append("PR-AUC in Methods/preamble but no Results numbers")

# cross-table consistency: fus-mlp
if "+0.03$ $[-0.01,0.07]$" not in tex.replace(" ", ""):
    # allow spaced version
    if not re.search(r"\+0\.03\s*\$\[-0\.01,\s*0\.07\]\$", tex):
        issues.append("Missing consistent fus-mlp +0.03 [-0.01,0.07]")

# ladder full MLP must match main mlp_u_iso wording
if "+0.20$ $[0.09,0.27]$" not in tex and "+0.20$ $[0.09,0.27]" not in tex:
    if tex.count("[0.09,0.27]") < 2:
        issues.append("Ladder/main MLP iso CI [0.09,0.27] should appear >=2 times")

# abstract must mention fusion -0.13 or not claim fusion positive raw
if "fusion $-0.13$" not in tex and "fusion $-0.13$" not in tex:
    if "versus near-zero / positive" in tex:
        issues.append("Abstract still overstates weighted fusion raw")

abs_ = c["abs"]
round_checks = [
    ("mlp_w_raw", -0.43),
    ("fus_w_raw", -0.13),
    ("mlp_u_iso", 0.20),
    ("gbm_default_avg_iso", 0.11),
    ("gbm_reg_posthoc_iso", 0.21),
    ("ir_only_iso", 0.08),
]
for name, expected in round_checks:
    got = round(abs_[name]["bss"], 2)
    if got != expected and abs(abs_[name]["bss"] - expected) > 0.006:
        issues.append(f"ABS mismatch {name}: json~{got} paper expects {expected}")

print("ISSUES:")
for i in issues:
    print(" -", i)
print("n_issues", len(issues))
print("figs_ok", all((ROOT / "paper/figures" / f).exists() for f in re.findall(r"includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)))
print("abstract_has_fus_neg", "fusion $-0.13$" in tex or "$-0.13$" in tex.split("abstract")[1][:800] if "abstract" in tex else False)
