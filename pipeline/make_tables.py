"""Manuscript tables (Markdown) built only from the aggregate files in SHARE_DIR (filled by share_results.py).

    python make_tables.py          # writes ../manuscript/tables/table*.md

Every number is copied from <task>_results_<task>.json / <task>_cohort_summary.json with fixed rounding,
so check_numbers.py can trace it. Tasks whose results are not shared yet are left out (noted in the file).
"""
from __future__ import annotations

import json
from pathlib import Path

import config as C

OUT = Path(__file__).resolve().parents[1] / "manuscript" / "tables"
TASK_NAME = {"cxr": "Chest radiograph (pleural effusion)", "ecg": "12-lead ECG (AF/flutter)",
             "ehr": "ICU time series (deterioration)"}
DEF_NAME = {"D0": "D0 standard", "D1": "D1 PGD-AT", "D2": "D2 TRADES", "D3": "D3 noise + smoothing",
            "D4": "D4 squeezing", "D5": "D5 detector", "D6": "D6 ensemble"}
DEFS = tuple(DEF_NAME)


def load(task, name):
    p = C.SHARE_DIR / f"{task}_{name}"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def f0(x):
    return "-" if x is None else f"{x:.0f}"


def ci(v, key, fmt):
    c = v.get(f"{key}_ci")
    return f"{fmt(v[key])} ({fmt(c[0])}-{fmt(c[1])})" if c else fmt(v.get(key))


def eps_label(task, eps_key):
    e = float(eps_key)
    if task == "cxr":
        return f"{e * 255:g}/255"
    if task == "ecg":
        return f"{e * 1000:g} µV"
    return f"{e * 100:g}%"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {t: load(t, f"results_{t}.json") for t in C.TASKS}
    coh = {t: load(t, "cohort_summary.json") for t in C.TASKS}
    have = [t for t in ("cxr", "ecg", "ehr") if res[t]]
    missing = [t for t in ("cxr", "ecg", "ehr") if not res[t]]
    note = f"\n\n_Not yet available: {', '.join(missing)}._\n" if missing else "\n"

    # Table 1: cohorts and splits
    rows = []
    for t in ("cxr", "ecg", "ehr"):
        c = coh[t]
        if not c:
            continue
        s = c["splits"]
        if t == "cxr":
            n = [s[k]["studies"] for k in ("train", "val", "test")]
            pos = [s[k]["labels"]["pleural_effusion"]["positives"] for k in ("train", "val", "test")]
        elif t == "ecg":
            n = [s[k]["n"] for k in ("train", "val", "test")]
            pos = [s[k]["af_positives"] for k in ("train", "val", "test")]
        else:
            n = [s[k]["n"] for k in ("train", "val", "test")]
            pos = [s[k]["positives"] for k in ("train", "val", "test")]
        rows.append([TASK_NAME[t]] + [f"{a:,} ({b:,})" for a, b in zip(n, pos)])
    (OUT / "table1_cohorts.md").write_text(
        "# Table 1. Cohorts: inputs (positives) per split\n\n" +
        table(["Task", "Train", "Validation", "Test"], rows) + "\n", encoding="utf-8")

    # Table 2: clean test performance
    rows = []
    for t in have:
        for d in DEFS:
            m = res[t]["clean"].get(d)
            if m:
                rows.append([TASK_NAME[t], DEF_NAME[d], f3(m["auroc"]), f3(m["auprc"]), f3(m["sensitivity"]),
                             f3(m["specificity"]), f3(m["brier"]), f3(m["ece"])])
    (OUT / "table2_clean.md").write_text(
        "# Table 2. Clean test performance (threshold at validation sensitivity 0.90)\n\n" +
        table(["Task", "Defence", "AUROC", "AUPRC", "Sensitivity", "Specificity", "Brier", "ECE"], rows) + note,
        encoding="utf-8")

    # Table 3: primary budget, strongest attacks
    rows = []
    for t in have:
        eps = f"{C.EPS_PRIMARY[t]:.6g}"
        for d in DEFS:
            a = res[t]["attacks"].get(d, {})
            for att in ("PGD", "APGD", "Square", "Transfer"):
                v = a.get(att, {}).get(eps)
                if v:
                    rows.append([TASK_NAME[t], DEF_NAME[d], att, f3(v["auroc_adv"]), f3(v["robust_accuracy"]),
                                 ci(v, "flips_per_1000", f0), f0(v["induced_fn_per_1000"]),
                                 f0(v["induced_fp_per_1000"])])
    (OUT / "table3_attacks_primary.md").write_text(
        "# Table 3. Attacks at the primary budget (CXR 2/255, ECG 50 µV, ICU 5%)\n\n" +
        table(["Task", "Defence", "Attack", "AUROC under attack", "Robust accuracy",
               "Flips/1000 (95% CI)", "Induced FN/1000", "Induced FP/1000"], rows) + note, encoding="utf-8")

    # Table S1: PGD budget sweep
    rows = []
    for t in have:
        for d in DEFS:
            pgd = res[t]["attacks"].get(d, {}).get("PGD", {})
            for e in sorted(pgd, key=float):
                v = pgd[e]
                rows.append([TASK_NAME[t], DEF_NAME[d], eps_label(t, e), f3(v["auroc_adv"]),
                             f3(v["robust_accuracy"]), f0(v["flips_per_1000"])])
    (OUT / "tableS1_pgd_sweep.md").write_text(
        "# Table S1. PGD over the budget sweep\n\n" +
        table(["Task", "Defence", "Budget", "AUROC under attack", "Robust accuracy", "Flips/1000"], rows) + note,
        encoding="utf-8")

    # Table 4: adaptive checks, modality-specific threat models, certification, paired tests
    rows = []
    for t in have:
        eps = f"{C.EPS_PRIMARY[t]:.6g}"
        A = res[t]["attacks"]
        na, ad = A.get("D4", {}).get("PGD_nonadaptive", {}).get(eps), A.get("D4", {}).get("PGD", {}).get(eps)
        if na and ad:
            rows.append([TASK_NAME[t], "D4 squeezing: non-adaptive vs BPDA PGD", "flips/1000",
                         f"{f0(na['flips_per_1000'])} vs {f0(ad['flips_per_1000'])}"])
        det = A.get("D5", {})
        if det.get("PGD", {}).get(eps):
            rows.append([TASK_NAME[t], "D5 detector: detection rate, PGD", "fraction",
                         f3(det["PGD"][eps].get("detection_rate"))])
        if det.get("PGD_adaptive_detector", {}).get(eps):
            rows.append([TASK_NAME[t], "D5 detector: accuracy with rejection, detector-aware PGD", "fraction",
                         f3(det["PGD_adaptive_detector"][eps]["accuracy_with_rejection"])])
        for d in DEFS:
            for att in ("PGD_smooth", f"PGD_sparse{C.EHR_SPARSE_FEATURES}"):
                v = A.get(d, {}).get(att, {}).get(eps)
                if v:
                    rows.append([TASK_NAME[t], f"{DEF_NAME[d]}: {att}", "AUROC / flips/1000",
                                 f"{f3(v['auroc_adv'])} / {f0(v['flips_per_1000'])}"])
        cert = res[t].get("certification")
        if cert:
            rows.append([TASK_NAME[t], f"D3 certified: abstain rate (n = {cert['n']})", "fraction",
                         f3(cert["abstain_rate"])])
            rows.append([TASK_NAME[t], "D3 certified: median radius (correct)", "radius",
                         f3(cert["median_radius_correct"])])
        g = res[t].get("gbm")
        if g:
            rows.append([TASK_NAME[t], "LightGBM clean", "AUROC", f3(g["clean"]["auroc"])])
            for k in ("transfer_from_GRU_D0", "transfer_from_surrogate"):
                rows.append([TASK_NAME[t], f"LightGBM {k.replace('_', ' ')}", "AUROC / flips/1000",
                             f"{f3(g[k]['auroc_adv'])} / {f0(g[k]['flips_per_1000'])}"])
    (OUT / "table4_adaptive_specific.md").write_text(
        "# Table 4. Adaptive checks, modality-specific threat models, certification and the GBM target\n\n" +
        table(["Task", "Analysis", "Measure", "Value"], rows) + note, encoding="utf-8")

    rows = []
    for t in have:
        for d, c in res[t].get("paired_vs_D0_pgd_primary", {}).items():
            rows.append([TASK_NAME[t], DEF_NAME[d], f3(c["diff"]), f"{f3(c['ci'][0])} to {f3(c['ci'][1])}",
                         f3(c["p_holm"])])
    (OUT / "tableS2_paired.md").write_text(
        "# Table S2. Robust accuracy under PGD (primary budget): defence minus D0, paired bootstrap, Holm\n\n" +
        table(["Task", "Defence", "Difference", "95% CI", "p (Holm)"], rows) + note, encoding="utf-8")
    # cross-task consistency: rank defences by robust accuracy under PGD at the primary budget, Kendall's W
    if len(have) >= 2:
        import numpy as np
        from scipy.stats import rankdata
        R = np.array([rankdata([-res[t]["attacks"][d]["PGD"][f"{C.EPS_PRIMARY[t]:.6g}"]["robust_accuracy"]
                                for d in DEFS]) for t in have])          # m tasks x n defences, 1 = best
        m, n = R.shape
        S = ((R.sum(0) - m * (n + 1) / 2) ** 2).sum()
        ties = sum(((c ** 3) - c).sum() for row in R for c in [np.unique(row, return_counts=True)[1]])
        W = float(12 * S / (m ** 2 * (n ** 3 - n) - m * ties))           # tie-corrected (D0 and D5 share a classifier)
        cross = {"tasks": have, "measure": "robust_accuracy_PGD_primary", "kendall_w": W,
                 "mean_rank": {d: float(r) for d, r in zip(DEFS, R.mean(0))},
                 "ranks": {t: {d: float(r) for d, r in zip(DEFS, R[i])} for i, t in enumerate(have)}}
        (C.SHARE_DIR / "cross_task_results.json").write_text(json.dumps(cross, indent=2), encoding="utf-8")
        rows = [[DEF_NAME[d]] + [f"{cross['ranks'][t][d]:g}" for t in have] + [f"{cross['mean_rank'][d]:.2f}"]
                for d in DEFS]
        (OUT / "table5_consistency.md").write_text(
            "# Table 5. Rank of each defence by robust accuracy under PGD at the primary budget (1 = best)\n\n" +
            table(["Defence"] + [TASK_NAME[t] for t in have] + ["Mean rank"], rows) +
            f"\n\nKendall's W across {len(have)} tasks = {W:.2f}.\n", encoding="utf-8")
    print(f"tables written to {OUT} (tasks with results: {', '.join(have) or 'none'})")


if __name__ == "__main__":
    main()
