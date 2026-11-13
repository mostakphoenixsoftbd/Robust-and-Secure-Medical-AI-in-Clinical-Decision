"""Print the value for every CXR [[R: ...]] placeholder (see manuscript/build/cxr_fill_plan.md).

    python cxr_fill_values.py                                    # ../pipeline_share/cxr_results_cxr.json
    python cxr_fill_values.py ../pipeline_share/ecg_results_ecg.json   # same schema: logic test on ECG

Reads aggregate files in pipeline_share/ only (results JSON, cohort summary, cross_task_results.json).
Formats follow the matching ECG/ICU numbers in the same manuscript sentence (AUROC/accuracy 3 dp,
flips per 1,000 0 dp, percentages 1 dp, Kendall's W 2 dp). Every key path is printed next to its value, so
each filled number can be traced; check_numbers.py must still PASS after filling.
Exit code 1 if any key path does not resolve.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import config as C

MISSING: list[str] = []


def get(obj, path: str):
    """'attacks/D0/PGD/<p>/auroc_adv' -> value, or None (recorded in MISSING)."""
    cur = obj
    for k in path.split("/"):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            MISSING.append(path)
            return None
    return cur


def f3(x):
    return "MISSING" if x is None else ("<10" if x == "<10" else f"{x:.3f}")


def f2(x):
    return "MISSING" if x is None else f"{x:.2f}"


def f0(x):
    return "MISSING" if x is None else ("<10" if x == "<10" else f"{x:,.0f}")


def pct(x):
    return "MISSING" if x is None else f"{100 * x:.1f}%"


def ci0(r, path):
    v, c = get(r, path), get(r, path + "_ci")
    return f"{f0(v)} (95% CI {f0(c[0])}-{f0(c[1])})" if v is not None and c else "MISSING"


def ci3(r, path):
    v, c = get(r, path), get(r, path.rsplit("/", 1)[0] + "/ci")
    return f"{f3(v)} (95% CI {f3(c[0])} to {f3(c[1])})" if v is not None and c else "MISSING"


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else C.SHARE_DIR / "cxr_results_cxr.json"
    if not path.exists():
        print(f"not found: {path} (run share_results.py after the CXR run finished)")
        return 1
    r = json.loads(path.read_text(encoding="utf-8"))
    task = r.get("task", "cxr")
    if task == "cxr" and "finished" not in r:
        print("WARNING: results file has no 'finished' key: partial run, values are not final")
    p = f"{C.EPS_PRIMARY[task]:.6g}"                                   # CXR: 0.00784314 (2/255)
    pgd0 = get(r, "attacks/D0/PGD") or {}
    small = min(pgd0, key=float) if pgd0 else "?"                        # CXR: 0.00196078 (0.5/255)
    unit = {"cxr": lambda e: f"{round(float(e) * 255, 3):g}/255", "ecg": lambda e: f"{float(e) * 1000:g} µV",
            "ehr": lambda e: f"{float(e) * 100:g}%"}[task]
    A = f"attacks/{{d}}/{{a}}/{p}"
    a = lambda d, att, k: f"{A.format(d=d, a=att)}/{k}"
    coh_p = C.SHARE_DIR / f"{task}_cohort_summary.json"
    coh = json.loads(coh_p.read_text(encoding="utf-8")) if coh_p.exists() else {}
    cross_p = C.SHARE_DIR / "cross_task_results.json"
    cross = json.loads(cross_p.read_text(encoding="utf-8")) if cross_p.exists() else {}

    seeds = [get(r, f"clean_d0_seeds/{s}/auroc") for s in (r.get("clean_d0_seeds") or {})]
    seeds = [s for s in seeds if s is not None]
    cert = r.get("certification") or {}
    ca = cert.get("certified_accuracy") or {}
    radii = sorted(ca, key=float)
    r1 = radii[1] if len(radii) > 1 else None
    rmax = max((k for k in radii if ca[k] > 0), key=float, default=None)
    rzero = next((k for k in radii if float(k) > float(rmax or 0) and ca[k] == 0), None)

    print(f"# values for task '{task}' from {path.name}; primary budget key {p} ({unit(p)}); "
          f"smallest budget key {small} ({unit(small)})\n")
    rows = [
        # id, location (file:line on 4 Nov 23:30), value text, source paths
        ("M1", "04_methods.md:122 attack-set positives",
         f"{f0(get(r, 'attack_subset/positives'))} of {f0(get(r, 'attack_subset/n'))}",
         "attack_subset/positives, attack_subset/n"),
        ("R1", "05_results.md:12 D0 clean AUROC (Table 2)", f3(get(r, "clean/D0/auroc")), "clean/D0/auroc"),
        ("R2", "05_results.md:14 D0 seed range",
         f"{f3(min(seeds))} to {f3(max(seeds))}" if seeds else "MISSING", "min/max clean_d0_seeds/seed{0,1,2}/auroc"),
        ("R3", "05_results.md:18 D1, D2 clean AUROC",
         f"D1 {f3(get(r, 'clean/D1/auroc'))}, D2 {f3(get(r, 'clean/D2/auroc'))}", "clean/D1/auroc, clean/D2/auroc"),
        ("R3x", "05_results.md:19-22 (optional, same paragraph) D3/D4/D6 clean AUROC; D5 clean rejection",
         f"D3 {f3(get(r, 'clean/D3/auroc'))}, D4 {f3(get(r, 'clean/D4/auroc'))}, D6 {f3(get(r, 'clean/D6/auroc'))}; "
         f"D5 rejects {pct(get(r, 'detector/clean_false_rejection_subset'))}",
         "clean/D{3,4,6}/auroc, detector/clean_false_rejection_subset"),
        ("R4", "05_results.md:34 D0 AUROC under PGD (+ success rate, same sentence)",
         f"AUROC {f3(get(r, a('D0', 'PGD', 'auroc_adv')))}; success {pct(get(r, a('D0', 'PGD', 'attack_success_rate')))}",
         a("D0", "PGD", "auroc_adv") + ", .../attack_success_rate"),
        ("R5", "05_results.md:37 D0 PGD flips/1000 with CI (also D2 06_discussion.md:12)",
         ci0(r, a("D0", "PGD", "flips_per_1000")), a("D0", "PGD", "flips_per_1000") + "(_ci)"),
        ("R6", "05_results.md:39 induced FP vs FN per 1000 (also D5 06_discussion.md:38)",
         f"{f0(get(r, a('D0', 'PGD', 'induced_fp_per_1000')))} induced FP against "
         f"{f0(get(r, a('D0', 'PGD', 'induced_fn_per_1000')))} induced FN",
         a("D0", "PGD", "induced_fp_per_1000") + ", ..._fn_per_1000"),
        ("R6x", "05_results.md:41-42 (optional) D0 APGD flips; FN vs attack-set positives",
         f"APGD {f0(get(r, a('D0', 'APGD', 'flips_per_1000')))} flips; positives {f0(get(r, 'attack_subset/positives'))}",
         a("D0", "APGD", "flips_per_1000")),
        ("R7", f"05_results.md:46 D0 PGD at smallest budget {unit(small)}",
         f"{f0(get(r, f'attacks/D0/PGD/{small}/flips_per_1000'))} flips per 1,000 "
         f"(AUROC {f3(get(r, f'attacks/D0/PGD/{small}/auroc_adv'))})",
         f"attacks/D0/PGD/{small}/flips_per_1000, .../auroc_adv"),
        ("R8", "05_results.md:55 D1 PGD flips with CI, AUROC (also D3/D6 06_discussion.md:16,71)",
         f"{ci0(r, a('D1', 'PGD', 'flips_per_1000'))}, AUROC under attack {f3(get(r, a('D1', 'PGD', 'auroc_adv')))}",
         a("D1", "PGD", "flips_per_1000") + "(_ci), .../auroc_adv"),
        ("R8x", "05_results.md:56-60 (optional) D1 APGD flips; robust-acc gain D1, D2 vs D0; D2 flips/AUROC",
         f"D1 APGD {f0(get(r, a('D1', 'APGD', 'flips_per_1000')))}; D1 gain "
         f"{ci3(r, 'paired_vs_D0_pgd_primary/D1/diff')} p_holm {f3(get(r, 'paired_vs_D0_pgd_primary/D1/p_holm'))}; "
         f"D2 {ci0(r, a('D2', 'PGD', 'flips_per_1000'))} AUROC {f3(get(r, a('D2', 'PGD', 'auroc_adv')))} gain "
         f"{f3(get(r, 'paired_vs_D0_pgd_primary/D2/diff'))}",
         "attacks/D1/APGD, paired_vs_D0_pgd_primary/D{1,2}, attacks/D2/PGD"),
        ("R9", "05_results.md:71 D5 detection rate; accuracy with rejection, detector-aware PGD",
         f"flagged {pct(get(r, a('D5', 'PGD', 'detection_rate')))}; detector-aware "
         f"{f3(get(r, a('D5', 'PGD_adaptive_detector', 'accuracy_with_rejection')))}",
         a("D5", "PGD", "detection_rate") + ", " + a("D5", "PGD_adaptive_detector", "accuracy_with_rejection")),
        ("R9x", "05_results.md:65-72 (optional) D3 flips + gain; D4 BPDA vs non-adaptive flips; D6 flips",
         f"D3 {f0(get(r, a('D3', 'PGD', 'flips_per_1000')))} (gain {ci3(r, 'paired_vs_D0_pgd_primary/D3/diff')}); "
         f"D4 BPDA {f0(get(r, a('D4', 'PGD', 'flips_per_1000')))} vs non-adaptive "
         f"{f0(get(r, a('D4', 'PGD_nonadaptive', 'flips_per_1000')))}; D6 {f0(get(r, a('D6', 'PGD', 'flips_per_1000')))}",
         "attacks/D{3,4,6}/PGD, attacks/D4/PGD_nonadaptive, paired_vs_D0_pgd_primary/D3"),
        ("R10", "05_results.md:78 D0 Square AUROC and flips (Square runs on the first SQUARE_SUBSET inputs: CXR 100)",
         f"AUROC {f3(get(r, a('D0', 'Square', 'auroc_adv')))}, {f0(get(r, a('D0', 'Square', 'flips_per_1000')))} flips "
         f"per 1,000 (n = {f0(get(r, a('D0', 'Square', 'n')))})", a("D0", "Square", "auroc_adv") + ", ..flips, ..n"),
        ("R10x", "05_results.md:79-83 (optional) transfer to D0; D1 Square/transfer; D4, D6 transfer",
         f"D0 transfer {f0(get(r, a('D0', 'Transfer', 'flips_per_1000')))} flips AUROC "
         f"{f3(get(r, a('D0', 'Transfer', 'auroc_adv')))}; D1 Square {f0(get(r, a('D1', 'Square', 'flips_per_1000')))}, "
         f"transfer {f0(get(r, a('D1', 'Transfer', 'flips_per_1000')))}; D4 transfer "
         f"{f0(get(r, a('D4', 'Transfer', 'flips_per_1000')))}, D6 transfer {f0(get(r, a('D6', 'Transfer', 'flips_per_1000')))}",
         "attacks/D{0,1,4,6}/{Square,Transfer}"),
        ("R11", "05_results.md:93 certification (L2 radius in input units; CXR pixels in [0,1], sigma "
                f"{cert.get('sigma')})",
         (f"n {f0(get(r, 'certification/n'))}; abstained {pct(get(r, 'certification/abstain_rate'))}; certified acc "
          f"{f3(ca.get('0'))} at radius 0, {f3(ca.get(r1))} at {r1}; largest radius with certified acc > 0: {rmax} "
          f"({f3(ca.get(rmax))}); none certified at {rzero}; median radius (correct) "
          f"{f3(get(r, 'certification/median_radius_correct'))}") if cert else "MISSING",
         "certification/{n,abstain_rate,certified_accuracy/<r>,median_radius_correct}"),
        ("R12", "05_results.md:104 Kendall's W across all tasks (cross_task_results.json)",
         (f"W = {f2(cross.get('kendall_w'))} across {len(cross.get('tasks', []))} tasks {cross.get('tasks')}; "
          f"ranks {task}: {cross.get('ranks', {}).get(task)}") if cross else "MISSING",
         "cross_task_results.json kendall_w, tasks, ranks/<task>"),
        ("D4", "06_discussion.md:32 D5 accuracy with rejection: non-adaptive vs detector-aware PGD",
         f"{f3(get(r, a('D5', 'PGD', 'accuracy_with_rejection')))} vs "
         f"{f3(get(r, a('D5', 'PGD_adaptive_detector', 'accuracy_with_rejection')))}",
         a("D5", "PGD", "accuracy_with_rejection") + ", " + a("D5", "PGD_adaptive_detector", "accuracy_with_rejection")),
    ]
    if task == "cxr" and cross and "cxr" not in cross.get("tasks", []):
        print("WARNING: cross_task_results.json does not include cxr yet: run make_tables.py\n")
    for pid, loc, val, src in rows:
        print(f"{pid:5s} {loc}\n      VALUE: {val}\n      SOURCE: {path.name if 'cross_task' not in src else ''} {src}\n")
    print("Reuse: D2 = R5 flips; D3 = R8 flips + R3 D1 clean AUROC (vs R1 D0); D5 = R6; D6 = R8 flips.")
    if coh:
        s = coh.get("splits", {})
        if task == "cxr":
            print(f"A1 cohort (abstract): {f0(coh.get('studies'))} studies / {f0(coh.get('patients'))} patients; "
                  f"test {f0(s.get('test', {}).get('studies'))} "
                  f"({f0(s.get('test', {}).get('labels', {}).get('pleural_effusion', {}).get('positives'))} positive)")
    if MISSING:
        print("\nUNRESOLVED key paths:\n  " + "\n  ".join(sorted(set(MISSING))))
        return 1
    print("\nall key paths resolved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
