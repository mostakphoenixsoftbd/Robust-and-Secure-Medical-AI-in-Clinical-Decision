"""Train all models and run the full attack x defence grid for one task (PROTOCOL sections 3-7).

    python run_task.py ecg            # or ehr / cxr
    python run_task.py ecg --stage train     # only train (weights cached; re-runs skip finished models)
    python run_task.py cxr --limit 50        # smoke test on a tiny attack subset

Writes <OUTPUT_DIR>/<task>/:
    weights/*.pt                       model weights (local only)
    results_<task>.json                aggregate metrics only (shared by share_results.py)
"""
from __future__ import annotations

import argparse
import os
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import attacks as A
import common as U
import config as C
import defences as D
import metrics as M
import models as Mo
from data import TaskData, cat_x, slice_x

BS = {"cxr": 16, "ecg": 250, "ehr": 1000}
# Gradient attacks on CXR run 8 images per batch (6 Nov 2026): APGD on the smoothed D3 model (EOT x4) ran out of
# memory at 16 on the shared 16 GB machine. Per-image attacks are independent of batch size apart from the seeded
# random start; attacks already cached at 16 are kept.
ATTACK_BS = {"cxr": 8}


# --------------------------------------------------------------------------- models
def new_model(task, data, kind="target"):
    mean, std = data.norm_stats() if task != "cxr" else (None, None)
    if task == "cxr":
        return Mo.CXRNet()
    if task == "ecg":
        return Mo.ECGNet(mean, std) if kind == "target" else Mo.ECGSurrogate(mean, std)
    return Mo.EHRNet(mean, std) if kind == "target" else Mo.EHRSurrogate(mean, std)


def get_model(task, data, name, mode, tr, va, log, init=None, kind="target", seed=0, **kw):
    wdir = U.task_dir(task) / "weights"
    wdir.mkdir(exist_ok=True)
    path = wdir / f"{name}.pt"
    m = new_model(task, data, kind)
    if path.exists():
        m.load_state_dict(torch.load(path))
        m.eval()
        return m
    if init is not None:
        m.load_state_dict(init.state_dict())
    t0 = time.time()
    ckpt = wdir / f"{name}.ckpt"
    m, hist = D.train_model(task, m, data, tr, va, mode=mode, seed=seed, ckpt=ckpt, **kw)
    torch.save(m.state_dict(), path)
    ckpt.unlink(missing_ok=True)
    log.setdefault("training", {})[name] = {"history": hist, "seconds": round(time.time() - t0, 1)}
    U.save_json(log, U.task_dir(task) / "training_log.json")     # after every model (runs can be interrupted)
    return m


def train_all(task, data, log):
    tr, va = data.idx("train"), data.idx("val")
    cfg = C.TRAIN[task]
    if task == "cxr":
        data.cxr_cache(Mo.CXRNet())        # frozen part is identical for every CXR model
    d0 = [get_model(task, data, f"D0_seed{s}", "std", tr, va, log, seed=s) for s in C.SEEDS]
    half = tr[np.random.default_rng(7).random(len(tr)) < 0.5]
    sur = get_model(task, data, "surrogate", "std", half, va, log, kind="surrogate", seed=11)
    ft_tr = tr
    if task in C.FT_TRAIN_SUBSET:
        ft_tr = np.random.default_rng(3).choice(tr, size=min(C.FT_TRAIN_SUBSET[task], len(tr)), replace=False)
    ft = dict(epochs=cfg["ft_epochs"], batch=cfg["ft_batch"], lr=cfg["lr"] / (1 if task == "cxr" else 2))
    d1 = get_model(task, data, "D1_pgd_at", "pgd", ft_tr, va, log, init=d0[0], select_robust=task != "cxr", **ft)
    d2 = get_model(task, data, "D2_trades", "trades", ft_tr, va, log, init=d0[0], select_robust=task != "cxr", **ft)
    d3 = get_model(task, data, "D3_noise", "noise", ft_tr, va, log, init=d0[0], **ft)
    return d0, sur, d1, d2, d3


# --------------------------------------------------------------------------- defence wrappers
class Ensemble(torch.nn.Module):
    def __init__(self, ms):
        super().__init__()
        self.ms = torch.nn.ModuleList(ms)

    def forward(self, x):
        return torch.stack([m(x) for m in self.ms]).mean(0)


def build_defences(task, data, d0, d1, d2, d3, log):
    """name -> dict(score(x)->prob, loss(y)->loss_fn, model for threshold/features)."""
    base = d0[0]
    ens = Ensemble(d0).eval()
    n_pred = 8 if task == "cxr" else C.SMOOTH_N_PREDICT
    sig = lambda m: (lambda x: torch.sigmoid(m(x)))
    defs = {
        "D0": {"score": sig(base), "loss": lambda y: A.bce_loss(base, y)},
        "D1": {"score": sig(d1), "loss": lambda y: A.bce_loss(d1, y)},
        "D2": {"score": sig(d2), "loss": lambda y: A.bce_loss(d2, y)},
        "D3": {"score": lambda x: D.smoothed_prob(d3, task, x, n=n_pred),
               "loss": lambda y: D.eot_loss(d3, task, y)},
        "D4": {"score": lambda x: torch.sigmoid(base(D.purify(task, x))),
               "loss": lambda y: A.bce_loss(lambda inp: base(D.bpda(task, inp)), y)},
        "D5": {"score": sig(base), "loss": lambda y: A.bce_loss(base, y)},
        "D6": {"score": sig(ens), "loss": lambda y: A.bce_loss(ens, y)},
    }
    # D5 detector: Mahalanobis on D0 penultimate features (train subsample), threshold on clean validation
    tr = data.idx("train", primary_only=True)
    tr = np.random.default_rng(5).choice(tr, size=min(len(tr), 4000 if task == "cxr" else 20000), replace=False)
    feats = batched(lambda x: base.features(x), data, tr, task)
    det = D.Mahalanobis().fit(feats, data.y[tr])
    va = data.idx("val", primary_only=True)
    sv = det.score(torch.from_numpy(batched(lambda x: base.features(x), data, va, task))).numpy()
    det.thr = float(np.quantile(sv, C.DETECT_TPR))
    det.scale = float(np.median(sv))
    defs["D5"]["detector"] = det
    return defs, ens


@torch.no_grad()
def batched(fn, data, idx, task):
    out = []
    for b in range(0, len(idx), BS[task] * (4 if task != "cxr" else 2)):
        out.append(fn(data.x(idx[b: b + BS[task] * (4 if task != "cxr" else 2)])).float())
    return torch.cat(out).numpy()


def scores_on(fn, x_full, task):
    out = []
    n = x_full[0].shape[0] if isinstance(x_full, tuple) else x_full.shape[0]
    with torch.no_grad():
        for b in range(0, n, BS[task] * 2):
            out.append(fn(slice_x(x_full, slice(b, b + BS[task] * 2))).float())
    return torch.cat(out).numpy()


def run_attack(attack_fn, x_full, y_full, task):
    """Apply an attack batch-wise; attack_fn(x_batch, y_batch) -> adversarial batch."""
    n = len(y_full)
    parts = []
    bs = ATTACK_BS.get(task, BS[task])
    for b in range(0, n, bs):
        sl = slice(b, b + bs)
        parts.append(attack_fn(slice_x(x_full, sl), y_full[sl]))
    return cat_x(parts)


# --------------------------------------------------------------------------- main experiment
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=C.TASKS)
    ap.add_argument("--stage", choices=["train", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: attack subset size")
    ap.add_argument("--one-eps", action="store_true", help="smoke test: only the primary budget in the grid")
    args = ap.parse_args(argv)
    task = args.task
    U.set_threads()
    torch.manual_seed(0)
    out = U.task_dir(task)
    res_path = out / f"results_{task}{'_smoke' if args.limit else ''}.json"
    # PIPELINE_HELPER=1: a second worker that pre-fills adv_cache for Square (except D3) and Transfer while the
    # main run is busy; it writes its own results file and stops after Transfer (main run loads these from cache)
    helper = os.environ.get("PIPELINE_HELPER") == "1"
    if helper:
        res_path = out / f"results_{task}_helper.json"
    tlog_path = out / "training_log.json"
    tlog = json.loads(tlog_path.read_text(encoding="utf-8")) if tlog_path.exists() else {}   # keep earlier runs' histories
    tlog.setdefault("training", {})
    data = TaskData(task)
    d0, sur, d1, d2, d3 = train_all(task, data, tlog)
    U.save_json(tlog, tlog_path)
    if args.stage == "train":
        return 0
    log: dict = {"task": task, "started": time.strftime("%Y-%m-%d %H:%M")}

    defs, ens = build_defences(task, data, d0, d1, d2, d3, log)
    va = data.idx("val", primary_only=True)
    te = data.idx("test", primary_only=True)

    # resume: clean scores per (model, index set) are cached locally (row-level, never shared) so a restart
    # after an out-of-memory crash skips the hours of clean scoring; keyed by the index set and the weights'
    # timestamps, so retrained weights or a different split invalidate the cache (scores are deterministic:
    # smoothing noise is seeded)
    import hashlib
    wdir = out / "weights"
    wstamp = max((f.stat().st_mtime_ns for f in wdir.rglob("*") if f.is_file()), default=0) if wdir.exists() else 0
    ccache = out / ("clean_cache_smoke" if args.limit else "clean_cache")
    ccache.mkdir(exist_ok=True)

    def cbatched(key, fn, idx):
        h = hashlib.sha1(np.ascontiguousarray(idx).tobytes() + str(wstamp).encode()).hexdigest()[:16]
        f = ccache / f"{key}_{h}.npy"
        if f.exists():
            return np.load(f)
        p = batched(fn, data, idx, task)
        tmp = f.with_name(f.stem + ".tmp.npy")
        np.save(tmp, p)
        tmp.replace(f)
        return p

    # thresholds on validation, clean metrics on the full test set
    log["clean"], thr = {}, {}
    xva_scores = {}
    for name, d in defs.items():
        pv = cbatched(f"{name}_val", d["score"], va)
        thr[name] = M.threshold_at_sensitivity(data.y[va], pv)
        pt = cbatched(f"{name}_test", d["score"], te)
        log["clean"][name] = M.clean_metrics(data.y[te], pt, thr[name])
        U.log(f"clean {name}: AUROC {log['clean'][name]['auroc']:.3f}")
    log["clean_d0_seeds"] = {}
    for s, m in zip(C.SEEDS, d0):
        p = cbatched(f"D0seed{s}_test", lambda x: torch.sigmoid(m(x)), te)
        log["clean_d0_seeds"][f"seed{s}"] = M.clean_metrics(data.y[te], p, M.threshold_at_sensitivity(
            data.y[va], cbatched(f"D0seed{s}_val", lambda x: torch.sigmoid(m(x)), va)))
    p_sur = cbatched("surrogate_test", lambda x: torch.sigmoid(sur(x)), te)
    log["clean"]["surrogate"] = M.clean_metrics(data.y[te], p_sur, 0.5)

    # attack subset (one row per patient, label-stratified)
    n_sub = args.limit or C.ATTACK_SUBSET[task]
    sub = te[U.stratified_subset(te, data.y, data.subjects, n_sub, seed=1)]
    xs, ys, _ = data.batch(sub)
    y_np = ys.numpy()
    log["attack_subset"] = {"n": int(len(sub)), "positives": int(y_np.sum())}
    clean_p = {k: scores_on(d["score"], xs, task) for k, d in defs.items()}
    det = defs["D5"]["detector"]
    feat_scores = lambda x: det.score(torch.from_numpy(scores_on(lambda z: d0[0].features(z), x, task))).numpy()
    log["detector"] = {"threshold_quantile": C.DETECT_TPR,
                       "clean_false_rejection_subset": float((feat_scores(xs) > det.thr).mean())}

    grid = log["attacks"] = {}

    # resume: adversarial scores per (defence, attack, eps) are cached locally (row-level, never shared),
    # so an interrupted run (e.g. out of memory on the shared machine) restarts where it stopped
    cache_dir = out / ("adv_cache_smoke" if args.limit else "adv_cache")
    cache_dir.mkdir(exist_ok=True)
    sub_file = cache_dir / "subset.npy"
    if sub_file.exists() and not np.array_equal(np.load(sub_file), sub):
        for f in cache_dir.glob("*.npz"):
            f.unlink()                     # different attack subset: the cache is stale
    np.save(sub_file, sub)

    def cached(key, make):
        f = cache_dir / f"{key}.npz"
        if f.exists():
            with np.load(f) as z:
                return {k: z[k] for k in z.files}
        arrs = make()
        tmp = f.with_name(f.stem + ".tmp.npz")
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrs)
        tmp.replace(f)
        return arrs

    def record(defn, attack, eps, make_xa):
        def make():
            t0 = time.time()
            xa = make_xa()
            a = {"pa": scores_on(defs[defn]["score"], xa, task), "seconds": np.array(time.time() - t0)}
            if defn == "D5":
                a["det"] = feat_scores(xa) > det.thr
            return a
        a = cached(f"{defn}_{attack}_{eps:.6g}", make)
        pa, detected = a["pa"], a.get("det")
        r = M.attack_metrics(y_np, clean_p[defn], pa, thr[defn], detected)
        if detected is not None:
            r["detection_rate"] = float(detected.mean())
        r["seconds"] = round(float(a["seconds"]), 1)
        grid.setdefault(defn, {}).setdefault(attack, {})[f"{eps:.6g}"] = r
        U.log(f"  {defn} {attack} eps={eps:.4g}: robust acc {r['robust_accuracy']:.3f}  AUROC {r['auroc_adv']:.3f}  "
              f"flips/1000 {r['flips_per_1000']:.0f}")
        U.save_json(log, res_path)
        return pa

    adv_pa = {}
    t_start = time.time()
    for eps in ((C.EPS_PRIMARY[task],) if args.one_eps else C.EPS[task]):
        for defn, d in defs.items():
            tm = A.ThreatModel(task, eps)
            record(defn, "FGSM", eps, lambda: run_attack(lambda xb, yb: A.fgsm(d["loss"](yb), tm, xb), xs, ys, task))
            adv_pa[(defn, eps)] = record(defn, "PGD", eps, lambda: run_attack(
                lambda xb, yb: A.pgd(d["loss"](yb), tm, xb, C.PGD_STEPS[task], C.PGD_RESTARTS[task]), xs, ys, task))
            if defn == "D4":     # non-adaptive: craft on the undefended model, then purify (gradient-masking check)
                record(defn, "PGD_nonadaptive", eps, lambda: run_attack(
                    lambda xb, yb: A.pgd(A.bce_loss(d0[0], yb), tm, xb, C.PGD_STEPS[task], 1), xs, ys, task))
            if defn == "D5":     # detector-aware adaptive attack: minimise detection score too
                def adaptive():
                    best_ok = np.zeros(len(y_np), bool)
                    for lam in C.DETECT_ADAPTIVE_LAMBDA:
                        lf = lambda yb, lam=lam: (lambda inp: F.binary_cross_entropy_with_logits(d0[0](inp), yb, reduction="none")
                                                  - lam * det.score(d0[0].features(inp)) / det.scale)
                        xa = run_attack(lambda xb, yb: A.pgd(lf(yb), tm, xb, C.PGD_STEPS[task], 1), xs, ys, task)
                        pa = scores_on(defs["D5"]["score"], xa, task)
                        evaded = ~(feat_scores(xa) > det.thr)
                        wrong = (pa >= thr["D5"]) != y_np.astype(bool)
                        best_ok |= evaded & wrong
                    return {"best_ok": best_ok}
                best_ok = cached(f"D5_PGD_adaptive_detector_{eps:.6g}", adaptive)["best_ok"]
                grid["D5"].setdefault("PGD_adaptive_detector", {})[f"{eps:.6g}"] = {
                    "accuracy_with_rejection": float(1 - best_ok.mean()), "n": int(len(y_np))}
                U.save_json(log, res_path)
        if task == "ecg":
            for defn in ("D0", "D1", "D2"):
                tm = A.ThreatModel(task, eps, smooth=True)
                record(defn, "PGD_smooth", eps, lambda: run_attack(
                    lambda xb, yb: A.pgd(defs[defn]["loss"](yb), tm, xb, C.PGD_STEPS[task], 1), xs, ys, task))
        if task == "ehr" and eps in (C.EPS_PRIMARY[task], max(C.EPS[task])):
            for defn in ("D0", "D1", "D2", "D4", "D6"):
                tm = A.ThreatModel(task, eps, sparse_k=C.EHR_SPARSE_FEATURES)
                record(defn, f"PGD_sparse{C.EHR_SPARSE_FEATURES}", eps, lambda: run_attack(
                    lambda xb, yb: A.pgd(defs[defn]["loss"](yb), tm, xb, C.PGD_STEPS[task], 1), xs, ys, task))
    log["seconds_grid_this_run"] = round(time.time() - t_start, 1)    # per-attack compute time: "seconds" entries

    # primary budget: APGD, Square (score-based), transfer from the surrogate
    eps = C.EPS_PRIMARY[task]
    tm = A.ThreatModel(task, eps)
    for defn, d in defs.items():
        record(defn, "APGD", eps, lambda: run_attack(
            lambda xb, yb: A.apgd(d["loss"](yb), tm, xb, C.APGD_STEPS[task]), xs, ys, task))
    n_sq = min(len(y_np), C.SQUARE_SUBSET[task])
    xs_sq, ys_sq = slice_x(xs, slice(0, n_sq)), ys[:n_sq]
    for defn, d in defs.items():
        if helper and defn == "D3":
            continue
        # queries to the smoothed classifier use 8 noise draws (final evaluation uses the full count)
        q_score = (lambda x: D.smoothed_prob(d3, task, x, n=8)) if defn == "D3" else d["score"]
        sq_loss = lambda yb, q=q_score: (lambda inp: F.binary_cross_entropy(q(inp).clamp(1e-6, 1 - 1e-6), yb,
                                                                            reduction="none"))

        def square():
            t0 = time.time()
            xa = run_attack(lambda xb, yb: A.square(sq_loss(yb), tm, xb, C.SQUARE_QUERIES[task]), xs_sq, ys_sq, task)
            return {"pa": scores_on(d["score"], xa, task), "seconds": np.array(time.time() - t0)}
        a = cached(f"{defn}_Square_{eps:.6g}", square)
        r = M.attack_metrics(y_np[:n_sq], clean_p[defn][:n_sq], a["pa"], thr[defn])
        r["seconds"] = round(float(a["seconds"]), 1)
        grid[defn].setdefault("Square", {})[f"{eps:.6g}"] = r
        U.log(f"  {defn} Square: robust acc {r['robust_accuracy']:.3f}")
        U.save_json(log, res_path)
    memo = {}

    def transfer_x():                      # crafted once on the surrogate, reused for every defence
        if "x" not in memo:
            memo["x"] = run_attack(lambda xb, yb: A.pgd(A.bce_loss(sur, yb), tm, xb, C.PGD_STEPS[task], 1), xs, ys, task)
        return memo["x"]
    for defn in defs:
        record(defn, "Transfer", eps, transfer_x)
    if helper:
        U.log("helper done (Square except D3, Transfer cached)")
        return 0

    # GBM (EHR only): black-box target, transfer from the GRU (D0) and from the surrogate
    if task == "ehr":
        log["gbm"] = gbm_block(data, xs, ys, transfer_x(), defs, tm, va, te)

    # paired comparison of each defence with D0 under PGD at the primary budget (Holm within task)
    comp, pv = {}, {}
    for defn in defs:
        if defn == "D0":
            continue
        c = M.paired_diff_ci(y_np, adv_pa[(defn, eps)], adv_pa[("D0", eps)], thr[defn], thr["D0"])
        comp[defn], pv[defn] = c, c["p"]
    for k, p in M.holm(pv).items():
        comp[k]["p_holm"] = p
    log["paired_vs_D0_pgd_primary"] = comp

    # D3 certification (randomized smoothing) on the certification subset
    n_c = min(len(sub), args.limit or C.CERT_SUBSET[task])
    xc = slice_x(xs, slice(0, n_c))
    thr_logit = float(np.log(thr["D3"] / (1 - thr["D3"]))) if 0 < thr["D3"] < 1 else 0.0
    preds, radii = [], []
    for b in range(0, n_c, BS[task]):
        p_, r_ = D.certify(d3, task, slice_x(xc, slice(b, b + BS[task])), thr_logit, C.SMOOTH_N_CERT[task])
        preds.append(p_); radii.append(r_)
    preds, radii = np.concatenate(preds), np.concatenate(radii)
    ok = preds == y_np[:n_c]
    radius_grid = np.linspace(0, 2 * C.NOISE_SIGMA[task] * 2, 9)
    log["certification"] = {"n": int(n_c), "sigma": C.NOISE_SIGMA[task], "n_samples": C.SMOOTH_N_CERT[task],
                            "abstain_rate": float((preds == -1).mean()),
                            "certified_accuracy": {f"{r:.4g}": float((ok & (radii >= r)).mean()) for r in radius_grid},
                            "median_radius_correct": float(np.median(radii[ok])) if ok.any() else 0.0}
    log["finished"] = time.strftime("%Y-%m-%d %H:%M")
    U.save_json(log, res_path)
    U.log(f"done -> {res_path}")
    return 0


def gbm_block(data, xs, ys, xa_sur, defs, tm, va, te):
    import lightgbm as lgb
    tr = data.idx("train")
    feat = lambda idx: Mo.ehr_summary_features(data.vals[data.rows[idx]], data.mask[data.rows[idx]], data.static[idx])
    clf = lgb.LGBMClassifier(**C.GBM_PARAMS, random_state=0).fit(feat(tr), data.y[tr])
    pv = clf.predict_proba(feat(va))[:, 1]
    thr = M.threshold_at_sensitivity(data.y[va], pv)
    out = {"clean": M.clean_metrics(data.y[te], clf.predict_proba(feat(te))[:, 1], thr)}
    to_np = lambda x: Mo.ehr_summary_features(x[0].numpy(), x[1].numpy(), x[2].numpy())
    pc = clf.predict_proba(to_np(xs))[:, 1]
    xa_gru = run_attack(lambda xb, yb: A.pgd(defs["D0"]["loss"](yb), tm, xb, C.PGD_STEPS["ehr"], 1), xs, ys, "ehr")
    for name, xa in (("transfer_from_GRU_D0", xa_gru), ("transfer_from_surrogate", xa_sur)):
        out[name] = M.attack_metrics(ys.numpy(), pc, clf.predict_proba(to_np(xa))[:, 1], thr)
    return out


if __name__ == "__main__":
    sys.exit(main())
