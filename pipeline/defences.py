"""Defences (PROTOCOL section 5): training procedures, purification, randomized smoothing, detection.

Training modes (train_model):
    std     D0 / surrogates: clean training
    pgd     D1: PGD adversarial training (Madry et al., 2018); CXR: one-step FGSM with random start
            ("fast AT", Wong et al., 2020) on a training subset because of the CPU budget
    trades  D2: TRADES (Zhang et al., 2019), KL between clean and adversarial predictions, beta = 6
    noise   D3: Gaussian noise augmentation, the base classifier for randomized smoothing (Cohen et al., 2019)
D1-D3 start from the seed-0 D0 weights (fine-tuning).
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import beta as beta_dist, norm
from sklearn.metrics import roc_auc_score

import attacks as A
import common as U
import config as C
from data import TaskData, slice_x


# --------------------------------------------------------------------------- losses
def label_loss(task, logits_all, ymat):
    """Masked multi-label BCE (CXR uncertain = -1 ignored); single-label BCE otherwise. Per sample."""
    if task == "cxr":
        m = (ymat >= 0).float()
        l = F.binary_cross_entropy_with_logits(logits_all, ymat.clamp(min=0), reduction="none") * m
        return l.sum(1) / m.sum(1).clamp(min=1)
    return F.binary_cross_entropy_with_logits(logits_all.view(-1), ymat.view(-1), reduction="none")


def all_logits(model, task, x):
    return model.all_logits(x) if task == "cxr" else model(x).unsqueeze(1)


def add_noise(task, x, sigma, gen=None):
    """Gaussian noise in the threat model's space (EHR: relative, observed mutable values only)."""
    if task == "ehr":
        vals, mask, st = x
        mut = torch.tensor([v not in C.EHR_IMMUTABLE for v in C.EHR_VARIABLES], dtype=torch.float32)
        z = torch.randn(vals.shape, generator=gen)
        return (vals * (1 + sigma * z * mask * mut), mask, st)
    z = torch.randn(x.shape, generator=gen)
    return x + sigma * z


# --------------------------------------------------------------------------- training
def train_model(task, model, data: TaskData, tr_idx, va_idx, mode="std", epochs=None, seed=0,
                lr=None, batch=None, select_robust=False, ckpt=None):
    """ckpt: path of an end-of-epoch checkpoint; an interrupted run resumes from it (RNG states included)."""
    cfg = C.TRAIN[task]
    epochs = epochs or cfg["epochs"]
    lr, batch = lr or cfg["lr"], batch or cfg["batch"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=C.WEIGHT_DECAY)
    tm = A.ThreatModel(task, C.EPS_TRAIN[task])
    best, best_state, bad, hist = -1.0, copy.deepcopy(model.state_dict()), 0, []
    start = 0
    if ckpt is not None and ckpt.exists():
        s = torch.load(ckpt, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        best, best_state, bad, hist, start = s["best"], s["best_state"], s["bad"], s["hist"], s["epoch"]
        rng.bit_generator.state = s["rng"]; gen.set_state(s["gen"]); torch.set_rng_state(s["torch_rng"])
        U.log(f"  [{task}/{mode}] resumed after epoch {start}")
        if bad >= cfg["patience"]:
            start = epochs
    for ep in range(start, epochs):
        model.train()
        _freeze_bn_if_cxr(model, task)
        order = rng.permutation(tr_idx)
        t0, tot = time.time(), 0.0
        for b in range(0, len(order), batch):
            ib = order[b: b + batch]
            if mode == "std" and task == "cxr":          # cached frozen activations (same numerics)
                ymat = torch.from_numpy(data.ymat[ib])
                loss = label_loss(task, model.head(model.trainable_part(data.cached(ib))), ymat).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.detach()) * len(ib)
                continue
            x, y, ymat = data.batch(ib)
            if mode == "std":
                loss = label_loss(task, all_logits(model, task, x), ymat).mean()
            elif mode == "noise":
                loss = label_loss(task, all_logits(model, task, add_noise(task, x, C.NOISE_SIGMA[task], gen)), ymat).mean()
            else:
                model.eval()
                lf = lambda inp: label_loss(task, all_logits(model, task, inp), ymat)
                if mode == "pgd":
                    xa = A.pgd(lf, tm, x, steps=C.AT_STEPS[task], alpha_frac=1.25 if C.AT_STEPS[task] == 1 else 2.5,
                               seed=int(rng.integers(1 << 30)))
                    model.train(); _freeze_bn_if_cxr(model, task)
                    loss = label_loss(task, all_logits(model, task, xa), ymat).mean()
                else:  # trades
                    with torch.no_grad():
                        p_clean = torch.sigmoid(all_logits(model, task, x))
                    kl = lambda inp: _bin_kl(p_clean, torch.sigmoid(all_logits(model, task, inp)), ymat, task)
                    xa = A.pgd(kl, tm, x, steps=C.AT_STEPS[task], alpha_frac=1.25 if C.AT_STEPS[task] == 1 else 2.5,
                               seed=int(rng.integers(1 << 30)))
                    model.train(); _freeze_bn_if_cxr(model, task)
                    lc = all_logits(model, task, x)
                    la = all_logits(model, task, xa)
                    loss = (label_loss(task, lc, ymat) +
                            C.TRADES_BETA * _bin_kl(torch.sigmoid(lc), torch.sigmoid(la), ymat, task)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(ib)
        score = (robust_val_auc(task, model, data, va_idx) if select_robust else val_auc(task, model, data, va_idx))
        hist.append({"epoch": ep + 1, "train_loss": round(tot / len(order), 5), "val_score": round(score, 5),
                     "seconds": round(time.time() - t0, 1)})
        U.log(f"  [{task}/{mode}] epoch {ep + 1}: loss {tot / len(order):.4f} val {score:.4f} ({time.time() - t0:.0f}s)")
        if score > best:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if ckpt is not None:
            tmp = ckpt.with_name(ckpt.name + ".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "best": best, "best_state": best_state,
                        "bad": bad, "hist": hist, "epoch": ep + 1, "rng": rng.bit_generator.state,
                        "gen": gen.get_state(), "torch_rng": torch.get_rng_state()}, tmp)
            tmp.replace(ckpt)
        if bad >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, hist


def _freeze_bn_if_cxr(model, task):
    # CXR: only denseblock4/norm5 train; frozen layers keep their CheXpert batch-norm statistics
    if task == "cxr":
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.BatchNorm2d) and not (".denseblock4" in "." + name or name.endswith("norm5")):
                m.eval()


def _bin_kl(p, q, ymat, task):
    eps = 1e-6
    p, q = p.clamp(eps, 1 - eps), q.clamp(eps, 1 - eps)
    kl = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
    if task == "cxr":
        return kl.mean(1)
    return kl.view(-1)


@torch.no_grad()
def predict(model, task, data, idx, bs=256, fn=None):
    out = []
    bs = 32 if task == "cxr" else bs
    for b in range(0, len(idx), bs):
        x = data.x(idx[b: b + bs])
        out.append((fn or model)(x).float())
    return torch.cat(out).numpy()


def val_auc(task, model, data, va_idx):
    model.eval()
    va = va_idx[data.valid[va_idx]]
    if task == "cxr" and getattr(data, "cache", None) is not None:
        with torch.no_grad():
            s = torch.cat([model.head(model.trainable_part(data.cached(va[b: b + 256])))[:, model.primary]
                           for b in range(0, len(va), 256)]).numpy()
        return float(roc_auc_score(data.y[va], s))
    return float(roc_auc_score(data.y[va], predict(model, task, data, va)))


def robust_val_auc(task, model, data, va_idx, n=500):
    """Model selection for adversarial training: AUROC under PGD-5 at the training budget (val subset)."""
    model.eval()
    va = va_idx[data.valid[va_idx]][:n]
    x, y, _ = data.batch(va)
    tm = A.ThreatModel(task, C.EPS_TRAIN[task])
    xa = A.pgd(A.bce_loss(model, y), tm, x, steps=5)
    with torch.no_grad():
        s = model(xa).numpy()
    return float(roc_auc_score(y.numpy(), s))


# --------------------------------------------------------------------------- D4 purification (feature squeezing)
_PRECISION = {"temperature": 0.1, "potassium": 0.1, "creatinine": 0.1, "lactate": 0.1, "inr": 0.1,
              "bilirubin": 0.1, "hemoglobin": 0.1, "wbc": 0.1}


def purify(task, x):
    if task == "cxr":                                       # 3x3 median filter + 5-bit depth
        p = F.pad(x, (1, 1, 1, 1), mode="reflect")
        med = p.unfold(2, 3, 1).unfold(3, 3, 1).reshape(*x.shape, 9).median(-1).values
        return torch.round(med * 31) / 31
    if task == "ecg":                                       # 50 ms moving average + 5 uV quantisation
        k = torch.ones(1, 1, 5) / 5
        b, l, t = x.shape
        sm = F.conv1d(F.pad(x.reshape(b * l, 1, t), (2, 2), mode="replicate"), k).reshape(b, l, t)
        return torch.round(sm / 0.005) * 0.005
    vals, mask, st = x                                      # EHR: round to reporting precision
    prec = torch.tensor([_PRECISION.get(v, 1.0) for v in C.EHR_VARIABLES])
    return (torch.round(vals / prec) * prec * mask + vals * (1 - mask), mask, st)


def bpda(task, x):
    """Purifier in the forward pass, identity gradient (Athalye et al., 2018)."""
    if task == "ehr":
        pv = purify(task, x)[0]
        return (x[0] + (pv - x[0]).detach(), x[1], x[2])
    return x + (purify(task, x) - x).detach()


# --------------------------------------------------------------------------- D3 randomized smoothing
def smoothed_prob(model, task, x, n=C.SMOOTH_N_PREDICT, seed=0):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        return torch.stack([torch.sigmoid(model(add_noise(task, x, C.NOISE_SIGMA[task], gen))) for _ in range(n)]).mean(0)


def eot_loss(model, task, y, k=C.EOT_SAMPLES):
    """Expectation over Gaussian noise of the BCE (adaptive attack on the smoothed classifier)."""
    def lf(inp):
        return torch.stack([F.binary_cross_entropy_with_logits(model(add_noise(task, inp, C.NOISE_SIGMA[task])), y,
                                                               reduction="none") for _ in range(k)]).mean(0)
    return lf


def certify(model, task, x, thr_logit, n, alpha=C.SMOOTH_ALPHA, n0=100, seed=0, chunk=100):
    """Cohen et al. (2019) CERTIFY for a binary base classifier g(x) = 1[logit > thr].
    Returns (prediction or -1 for abstain, certified L2 radius in the noise space)."""
    gen = torch.Generator().manual_seed(seed)

    def counts(m):
        c = torch.zeros(_bs(x))
        with torch.no_grad():
            for s in range(0, m, chunk):
                k = min(chunk, m - s)
                for _ in range(k):
                    c += (model(add_noise(task, x, C.NOISE_SIGMA[task], gen)) > thr_logit).float()
        return c.numpy()
    c0 = counts(n0)
    guess = (c0 > n0 / 2).astype(int)
    c1 = counts(n)
    k = np.where(guess == 1, c1, n - c1)
    p_low = beta_dist.ppf(alpha, k, n - k + 1)
    p_low = np.nan_to_num(p_low, nan=0.0)
    pred = np.where(p_low > 0.5, guess, -1)
    radius = np.where(p_low > 0.5, C.NOISE_SIGMA[task] * norm.ppf(np.clip(p_low, 1e-12, 1 - 1e-12)), 0.0)
    return pred, radius


def _bs(x):
    return x[0].shape[0] if isinstance(x, tuple) else x.shape[0]


# --------------------------------------------------------------------------- D5 Mahalanobis detector
class Mahalanobis:
    """Class-conditional Gaussians with shared covariance on penultimate features (Lee et al., 2018)."""

    def fit(self, feats: np.ndarray, y: np.ndarray):
        self.mu = np.stack([feats[y == c].mean(0) for c in (0, 1)])
        cen = feats - self.mu[y.astype(int)]
        cov = cen.T @ cen / len(feats) + 1e-3 * np.eye(feats.shape[1])
        self.prec = torch.from_numpy(np.linalg.inv(cov).astype(np.float32))
        self.mu_t = torch.from_numpy(self.mu.astype(np.float32))
        return self

    def score(self, f: torch.Tensor) -> torch.Tensor:
        """Minimum Mahalanobis distance over classes (higher = more anomalous). Differentiable."""
        d = torch.stack([(((f - self.mu_t[c]) @ self.prec) * (f - self.mu_t[c])).sum(1) for c in (0, 1)])
        return d.min(0).values
