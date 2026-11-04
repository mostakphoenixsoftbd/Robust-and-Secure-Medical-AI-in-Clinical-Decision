"""Figures from the aggregate results JSONs only (no row-level data).

    python make_figures.py

Writes to <OUTPUT_DIR>/figures/ (copied to pipeline_share by share_results.py):
    fig_robustness_curves.png/.svg   robust accuracy vs budget, PGD (adaptive), one panel per task
    fig_decision_flips.png/.svg      induced false negatives / false positives per 1,000 at the primary budget
    fig_certified.png/.svg           certified accuracy vs L2 radius (randomized smoothing)
    fig_tradeoff.png/.svg            clean AUROC vs robust AUROC (PGD, primary budget) per defence
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config as C  # noqa: E402

NAMES = {"D0": "Standard", "D1": "PGD-AT", "D2": "TRADES", "D3": "Rand. smoothing", "D4": "Squeezing (BPDA)",
         "D5": "Detector", "D6": "Ensemble"}
COLORS = dict(zip(NAMES, ["#4c4c4c", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#d62728"]))
TITLES = {"cxr": "Chest X-ray (pleural effusion)", "ecg": "12-lead ECG (atrial fibrillation)",
          "ehr": "ICU time series (deterioration)"}
UNITS = {"cxr": lambda e: e * 255, "ecg": lambda e: e * 1000, "ehr": lambda e: e * 100}
XLAB = {"cxr": "L∞ budget (×1/255)", "ecg": "L∞ budget (µV)", "ehr": "relative budget (%)"}


def load():
    out = {}
    for t in C.TASKS:
        p = C.OUTPUT_DIR / t / f"results_{t}.json"
        if p.exists():
            r = json.loads(p.read_text(encoding="utf-8"))
            if "attacks" in r:
                out[t] = r
    return out


def save(fig, name):
    d = C.OUTPUT_DIR / "figures"
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(d / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def robustness(res):
    fig, axes = plt.subplots(1, len(res), figsize=(4.2 * len(res), 3.4), squeeze=False)
    for ax, (t, r) in zip(axes[0], res.items()):
        for d in NAMES:
            pgd = r["attacks"].get(d, {}).get("PGD", {})
            if not pgd:
                continue
            e = sorted(pgd, key=float)
            x = [UNITS[t](float(k)) for k in e]
            y = [pgd[k]["robust_accuracy"] for k in e]
            clean = r["attacks"][d]["PGD"][e[0]]["accuracy_clean"]
            ax.plot([0] + x, [clean] + y, marker="o", ms=3, lw=1.4, color=COLORS[d], label=NAMES[d])
        ax.set_title(TITLES[t], fontsize=9)
        ax.set_xlabel(XLAB[t], fontsize=8)
        ax.set_ylabel("accuracy at operating threshold", fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    axes[0][0].legend(fontsize=7, frameon=False)
    save(fig, "fig_robustness_curves")


def flips(res):
    fig, axes = plt.subplots(1, len(res), figsize=(4.2 * len(res), 3.2), squeeze=False)
    for ax, (t, r) in zip(axes[0], res.items()):
        key = f"{C.EPS_PRIMARY[t]:.6g}"
        ds = [d for d in NAMES if key in r["attacks"].get(d, {}).get("PGD", {})]
        fn = [r["attacks"][d]["PGD"][key]["induced_fn_per_1000"] for d in ds]
        fp = [r["attacks"][d]["PGD"][key]["induced_fp_per_1000"] for d in ds]
        xi = np.arange(len(ds))
        ax.bar(xi, fn, color="#d62728", label="induced missed cases (FN)")
        ax.bar(xi, fp, bottom=fn, color="#1f77b4", label="induced false alarms (FP)")
        ax.set_xticks(xi, [NAMES[d] for d in ds], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("flipped decisions per 1,000", fontsize=8)
        ax.set_title(TITLES[t], fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0][0].legend(fontsize=7, frameon=False)
    save(fig, "fig_decision_flips")


def certified(res):
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for t, r in res.items():
        c = r.get("certification", {}).get("certified_accuracy")
        if not c:
            continue
        ks = sorted(c, key=float)
        rad = [float(k) / r["certification"]["sigma"] for k in ks]
        ax.plot(rad, [c[k] for k in ks], marker="o", ms=3, label=TITLES[t])
    ax.set_xlabel("certified L2 radius (in units of σ)", fontsize=8)
    ax.set_ylabel("certified accuracy", fontsize=8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, frameon=False)
    save(fig, "fig_certified")


def tradeoff(res):
    fig, axes = plt.subplots(1, len(res), figsize=(4.2 * len(res), 3.2), squeeze=False)
    for ax, (t, r) in zip(axes[0], res.items()):
        key = f"{C.EPS_PRIMARY[t]:.6g}"
        for d in NAMES:
            a = r["attacks"].get(d, {}).get("PGD", {}).get(key)
            if a:
                ax.scatter(r["clean"][d]["auroc"], a["auroc_adv"], color=COLORS[d], s=25, label=NAMES[d])
        ax.set_xlabel("clean AUROC (test)", fontsize=8)
        ax.set_ylabel("AUROC under PGD", fontsize=8)
        ax.set_title(TITLES[t], fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    axes[0][0].legend(fontsize=6, frameon=False)
    save(fig, "fig_tradeoff")


def main():
    res = load()
    if not res:
        print("no results yet")
        return
    robustness(res); flips(res); certified(res); tradeoff(res)
    print("figures for:", ", ".join(res))


if __name__ == "__main__":
    main()
