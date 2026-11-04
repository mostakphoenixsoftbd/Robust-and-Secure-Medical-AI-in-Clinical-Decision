"""Metrics (PROTOCOL sections 6-7): clean and robust discrimination, decision flips, bootstrap CIs."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import config as C


def threshold_at_sensitivity(y, p, target=C.TARGET_SENSITIVITY) -> float:
    """Largest threshold whose sensitivity on (validation) data is >= target."""
    pos = np.sort(p[y == 1])
    k = int(np.floor((1 - target) * len(pos)))
    return float(pos[k]) if len(pos) else 0.5


def ece(y, p, bins=10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def clean_metrics(y, p, thr) -> dict:
    pred = p >= thr
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "sensitivity": float(pred[y == 1].mean()), "specificity": float((~pred[y == 0]).mean()),
            "brier": float(brier_score_loss(y, p)), "ece": ece(y, p), "threshold": float(thr), "n": int(len(y))}


def _point(y, pc, pa, thr, detected=None):
    dc, da = pc >= thr, pa >= thr
    yb = y.astype(bool)
    correct_c, correct_a = dc == yb, da == yb
    out = {
        "auroc_adv": float(roc_auc_score(y, pa)) if 0 < y.mean() < 1 else float("nan"),
        "robust_accuracy": float(correct_a.mean()),
        "flips_per_1000": float(1000 * (dc != da).mean()),
        "induced_fn_per_1000": float(1000 * (yb & dc & ~da).mean()),
        "induced_fp_per_1000": float(1000 * (~yb & ~dc & da).mean()),
        "attack_success_rate": float((~correct_a[correct_c]).mean()) if correct_c.any() else float("nan"),
    }
    if detected is not None:
        out["accuracy_with_rejection"] = float((correct_a | detected).mean())
    return out


def attack_metrics(y, pc, pa, thr, detected=None, n_boot=C.BOOTSTRAP, seed=0) -> dict:
    """Point estimates + 95 % bootstrap CIs (resampling patients; one row per patient in the subset)."""
    pt = _point(y, pc, pa, thr, detected)
    pt.update({"auroc_clean": float(roc_auc_score(y, pc)), "accuracy_clean": float(((pc >= thr) == y.astype(bool)).mean()),
               "n": int(len(y))})
    rng = np.random.default_rng(seed)
    keys = ["auroc_adv", "robust_accuracy", "flips_per_1000", "induced_fn_per_1000", "induced_fp_per_1000"]
    boots = {k: [] for k in keys}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        b = _point(y[i], pc[i], pa[i], thr, None if detected is None else detected[i])
        for k in keys:
            boots[k].append(b[k])
    for k in keys:
        if boots[k]:
            pt[k + "_ci"] = [float(np.percentile(boots[k], 2.5)), float(np.percentile(boots[k], 97.5))]
    return pt


def paired_diff_ci(y, pa1, pa2, thr1, thr2, n_boot=C.BOOTSTRAP, seed=0) -> dict:
    """Robust-accuracy difference (defence - D0) on the same attack subset, paired bootstrap; two-sided p."""
    c1, c2 = (pa1 >= thr1) == y.astype(bool), (pa2 >= thr2) == y.astype(bool)
    d = float(c1.mean() - c2.mean())
    rng = np.random.default_rng(seed)
    bs = np.array([(c1[i].mean() - c2[i].mean()) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))])
    p = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
    return {"diff": d, "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "p": p}


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out
