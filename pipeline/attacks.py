"""Evasion attacks (PROTOCOL section 4) under one threat-model abstraction.

A ThreatModel knows, for one task and budget eps, which part of the input may change and how much:
    CXR  every pixel, |delta| <= eps, pixels stay in [0, 1]                         (L-inf)
    ECG  every sample, |delta| <= eps mV; optional Gaussian-smoothed perturbation    (L-inf)
    EHR  only observed values (mask = 1) of mutable variables, |delta| <= eps * |x|,
         values stay inside the variable's physiological range; optional sparsity
         (at most k variables changed per patient, chosen by gradient magnitude)

Attacks maximise a per-sample loss `loss_fn(model_input) -> (B,)` (default: BCE against the true label,
i.e. untargeted). Adaptive variants (BPDA, EOT, detector-aware) are built by the caller through
`loss_fn`, so every attack works unchanged against every defence.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

import config as C


class ThreatModel:
    def __init__(self, task: str, eps: float, sparse_k: int | None = None, smooth: bool = False):
        self.task, self.eps, self.sparse_k, self.smooth = task, float(eps), sparse_k, smooth
        if task == "ehr":
            lo = torch.tensor([C.EHR_RANGES[v][0] for v in C.EHR_VARIABLES], dtype=torch.float32)
            hi = torch.tensor([C.EHR_RANGES[v][1] for v in C.EHR_VARIABLES], dtype=torch.float32)
            self.lo, self.hi = lo + 1e-4, hi
            self.mutable = torch.tensor([v not in C.EHR_IMMUTABLE for v in C.EHR_VARIABLES], dtype=torch.float32)
        if smooth:
            s = C.ECG_SMOOTH_SIGMA
            r = int(3 * s)
            k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=torch.float32) / s) ** 2)
            self.kernel = (k / k.sum()).view(1, 1, -1)

    # -- which tensor is perturbed, and how a perturbed tensor is put back into the model input
    def get(self, x):
        return x[0] if self.task == "ehr" else x

    def put(self, x, p):
        return (p, x[1], x[2]) if self.task == "ehr" else p

    def radius(self, x, allowed=None) -> torch.Tensor:
        p0 = self.get(x)
        if self.task == "ehr":
            a = x[1] * self.mutable if allowed is None else allowed
            return self.eps * p0.abs() * a
        return torch.full_like(p0, self.eps)

    def sparse_allowed(self, x, grad: torch.Tensor) -> torch.Tensor:
        """EHR sparse attack: keep the k variables with the largest summed |gradient| per patient."""
        a = x[1] * self.mutable
        score = (grad.abs() * a).sum(1)                       # (B, F)
        top = score.topk(self.sparse_k, dim=1).indices
        keep = torch.zeros_like(score).scatter_(1, top, 1.0)
        return a * keep[:, None, :]

    def delta_from(self, eta):
        """Smooth ECG attack: delta = Gaussian-smoothed eta (a convex combination, so |delta| <= eps)."""
        if not self.smooth:
            return eta
        b, l, t = eta.shape
        pad = self.kernel.shape[-1] // 2
        return F.conv1d(F.pad(eta.reshape(b * l, 1, t), (pad, pad), mode="replicate"), self.kernel).reshape(b, l, t)

    def project(self, p, p0, rad):
        p = torch.max(torch.min(p, p0 + rad), p0 - rad)
        if self.task == "cxr":
            p = p.clamp(0.0, 1.0)
        elif self.task == "ehr":
            boxed = torch.max(torch.min(p, self.hi), self.lo)
            p = torch.where(rad > 0, boxed, p0)
        return p


def bce_loss(f, y):
    return lambda inp: F.binary_cross_entropy_with_logits(f(inp), y, reduction="none")


def _grad(loss_fn, tm, x, p):
    p = p.detach().requires_grad_(True)
    loss = loss_fn(tm.put(x, p))
    g, = torch.autograd.grad(loss.sum(), p)
    return loss.detach(), g


def _setup(loss_fn, tm, x):
    p0 = tm.get(x).detach()
    allowed = None
    if tm.sparse_k:
        _, g = _grad(loss_fn, tm, x, p0)
        allowed = tm.sparse_allowed(x, g)
    return p0, tm.radius(x, allowed)


def fgsm(loss_fn, tm: ThreatModel, x):
    p0, rad = _setup(loss_fn, tm, x)
    _, g = _grad(loss_fn, tm, x, p0)
    return tm.put(x, tm.project(p0 + rad * g.sign(), p0, rad).detach())


def pgd(loss_fn, tm: ThreatModel, x, steps: int, restarts: int = 1, alpha_frac: float = 2.5, seed: int = 0):
    """L-inf PGD with random start; per sample keeps the highest-loss iterate over steps and restarts."""
    gen = torch.Generator().manual_seed(seed)
    p0, rad = _setup(loss_fn, tm, x)
    best_p, best_l = p0.clone(), loss_fn(tm.put(x, p0)).detach()
    alpha = alpha_frac * rad / max(steps, 1)
    for _ in range(restarts):
        if tm.smooth:
            eta = (torch.rand(p0.shape, generator=gen) * 2 - 1) * rad
            for _ in range(steps):
                eta = eta.detach().requires_grad_(True)
                loss = loss_fn(tm.put(x, p0 + tm.delta_from(eta)))
                g, = torch.autograd.grad(loss.sum(), eta)
                _keep(best_p, best_l, p0 + tm.delta_from(eta).detach(), loss.detach())
                eta = torch.max(torch.min(eta.detach() + alpha * g.sign(), rad), -rad)
            p = p0 + tm.delta_from(eta)
        else:
            p = tm.project(p0 + (torch.rand(p0.shape, generator=gen) * 2 - 1) * rad, p0, rad)
            for _ in range(steps):
                loss, g = _grad(loss_fn, tm, x, p)
                _keep(best_p, best_l, p, loss)
                p = tm.project(p + alpha * g.sign(), p0, rad)
        with torch.no_grad():
            _keep(best_p, best_l, p.detach(), loss_fn(tm.put(x, p)).detach())
    return tm.put(x, best_p.detach())


def _keep(best_p, best_l, p, loss):
    better = loss > best_l
    if better.any():
        idx = better.nonzero().squeeze(1)
        best_p[idx] = p.detach()[idx]
        best_l[idx] = loss[idx]


def apgd(loss_fn, tm: ThreatModel, x, steps: int, rho: float = 0.75, seed: int = 0):
    """APGD-CE (Croce & Hein, 2020): momentum, step-size halving at checkpoints, restart from best."""
    gen = torch.Generator().manual_seed(seed)
    p0, rad = _setup(loss_fn, tm, x)
    b = p0.shape[0]
    view = (b,) + (1,) * (p0.dim() - 1)
    pj = [0, 0.22]
    while pj[-1] < 1:
        pj.append(pj[-1] + max(pj[-1] - pj[-2] - 0.03, 0.06))
    checkpoints = sorted({int(math.ceil(q * steps)) for q in pj if q <= 1})
    eta = torch.full(view, 2.0)                                # step size in units of the radius
    p = tm.project(p0 + (torch.rand(p0.shape, generator=gen) * 2 - 1) * rad, p0, rad)
    loss, g = _grad(loss_fn, tm, x, p)
    best_p, best_l = p.clone(), loss.clone()
    p_prev = p.clone()
    p_new = tm.project(p + eta * rad * g.sign(), p0, rad)
    count = torch.zeros(b)
    last_ck, l_at_ck, eta_at_ck = 0, best_l.clone(), eta.clone()
    for k in range(1, steps):
        loss_new, g_new = _grad(loss_fn, tm, x, p_new)
        count += (loss_new > loss).float()
        _keep(best_p, best_l, p_new, loss_new)
        z = tm.project(p_new + eta * rad * g_new.sign(), p0, rad)
        mom = rho if k > 1 else 1.0
        cand = tm.project(p_new + mom * (z - p_new) + (1 - mom) * (p_new - p_prev), p0, rad)
        p_prev, p, loss = p_new, p_new, loss_new
        p_new = cand
        if k in checkpoints:
            h = k - last_ck
            cond1 = count < 0.75 * h
            cond2 = (eta.view(b) == eta_at_ck.view(b)) & (best_l <= l_at_ck)
            red = cond1 | cond2
            eta_at_ck = eta.clone()
            eta[red] /= 2
            if red.any():
                p_new[red] = best_p[red]
            count.zero_()
            last_ck, l_at_ck = k, best_l.clone()
    with torch.no_grad():
        _keep(best_p, best_l, p_new, loss_fn(tm.put(x, p_new)))
    return tm.put(x, best_p.detach())


def square(loss_fn, tm: ThreatModel, x, queries: int, p_init: float = 0.1, seed: int = 0):
    """Score-based random search in the L-inf ball (Andriushchenko et al., 2020), adapted per modality:
    square patches (CXR), single-lead windows (ECG), one variable over a time window (EHR)."""
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        p0, rad = tm.get(x).detach(), tm.radius(x)
        sign = torch.from_numpy(rng.choice([-1.0, 1.0], size=p0.shape).astype(np.float32))
        p = tm.project(p0 + rad * sign, p0, rad)
        best = loss_fn(tm.put(x, p))
        for q in range(queries):
            frac = p_init * (0.5 ** int(8 * q / max(queries, 1)))      # schedule: halve 8 times
            new = p.clone()
            for i in range(p0.shape[0]):
                sl = _patch(p0.shape[1:], frac, rng)
                s = float(rng.choice([-1.0, 1.0]))
                new[i][sl] = p0[i][sl] + s * rad[i][sl]
            new = tm.project(new, p0, rad)
            l_new = loss_fn(tm.put(x, new))
            better = l_new > best
            p[better] = new[better]
            best = torch.where(better, l_new, best)
    return tm.put(x, p)


def _patch(shape, frac, rng):
    if len(shape) == 3:                                   # CXR (1, H, W)
        h, w = shape[1:]
        s = max(1, int(round(math.sqrt(frac * h * w))))
        r, c = rng.integers(0, h - s + 1), rng.integers(0, w - s + 1)
        return (slice(None), slice(r, r + s), slice(c, c + s))
    if shape[0] == 12:                                    # ECG (12, T)
        t = shape[1]
        s = max(1, int(round(frac * t * 12)))
        s = min(s, t)
        st = rng.integers(0, t - s + 1)
        return (int(rng.integers(0, 12)), slice(st, st + s))
    t, f = shape                                          # EHR (T, F)
    s = max(1, int(round(frac * t * f)))
    s = min(s, t)
    st = rng.integers(0, t - s + 1)
    return (slice(st, st + s), int(rng.integers(0, f)))
