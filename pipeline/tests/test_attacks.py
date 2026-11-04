"""Synthetic-data tests: every attack respects its threat model and actually increases the loss."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import attacks as A  # noqa: E402
import config as C  # noqa: E402
import defences as D  # noqa: E402
import metrics as M  # noqa: E402
import models as Mo  # noqa: E402

torch.manual_seed(0)
F_ = len(C.EHR_VARIABLES)


def ecg_batch(n=8):
    return torch.randn(n, 12, C.ECG_SAMPLES) * 0.3, torch.randint(0, 2, (n,)).float()


def ehr_batch(n=8):
    lo = torch.tensor([C.EHR_RANGES[v][0] for v in C.EHR_VARIABLES])
    hi = torch.tensor([C.EHR_RANGES[v][1] for v in C.EHR_VARIABLES])
    mid = (lo + torch.clamp(hi, max=lo + 200)) / 2 + 1
    mask = (torch.rand(n, C.N_BINS, F_) < 0.5).float()
    vals = mid * (1 + 0.1 * torch.randn(n, C.N_BINS, F_)) * mask
    static = torch.randn(n, 2)
    return (vals, mask, static), torch.randint(0, 2, (n,)).float()


@pytest.fixture(scope="module")
def ecg_model():
    return Mo.ECGNet().eval()


@pytest.fixture(scope="module")
def ehr_model():
    return Mo.EHRNet().eval()


@pytest.mark.parametrize("attack", ["fgsm", "pgd", "apgd", "square"])
def test_ecg_linf_bound_and_loss(ecg_model, attack):
    x, y = ecg_batch()
    tm = A.ThreatModel("ecg", 0.05)
    lf = A.bce_loss(ecg_model, y)
    xa = {"fgsm": lambda: A.fgsm(lf, tm, x), "pgd": lambda: A.pgd(lf, tm, x, 10, 1),
          "apgd": lambda: A.apgd(lf, tm, x, 20), "square": lambda: A.square(lf, tm, x, 50)}[attack]()
    assert (xa - x).abs().max() <= 0.05 + 1e-6
    with torch.no_grad():
        assert lf(xa).mean() >= lf(x).mean() - 1e-6


def test_ecg_smooth_attack_bounded(ecg_model):
    x, y = ecg_batch()
    tm = A.ThreatModel("ecg", 0.05, smooth=True)
    xa = A.pgd(A.bce_loss(ecg_model, y), tm, x, 5, 1)
    d = xa - x
    assert d.abs().max() <= 0.05 + 1e-6
    # smoothed: sample-to-sample changes are small relative to eps
    assert (d[..., 1:] - d[..., :-1]).abs().max() < 0.05


def test_cxr_box_and_bound():
    lin = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(16 * 16, 1))
    f = lambda x: lin(x).squeeze(-1)
    x, y = torch.rand(4, 1, 16, 16), torch.tensor([0., 1., 0., 1.])
    tm = A.ThreatModel("cxr", 8 / 255)
    xa = A.pgd(A.bce_loss(f, y), tm, x, 5, 1)
    assert xa.min() >= 0 and xa.max() <= 1
    assert (xa - x).abs().max() <= 8 / 255 + 1e-6


@pytest.mark.parametrize("sparse", [None, 3])
def test_ehr_constraints(ehr_model, sparse):
    x, y = ehr_batch()
    eps = 0.05
    tm = A.ThreatModel("ehr", eps, sparse_k=sparse)
    xa = A.pgd(A.bce_loss(ehr_model, y), tm, x, 10, 1)
    vals, mask, static = x
    va = xa[0]
    changed = (va - vals).abs() > 1e-7
    assert torch.equal(xa[1], mask) and torch.equal(xa[2], static)
    assert not (changed & (mask == 0)).any(), "unobserved entries changed"
    imm = [C.EHR_VARIABLES.index(v) for v in C.EHR_IMMUTABLE]
    assert not changed[..., imm].any(), "immutable variables changed"
    assert ((va - vals).abs() <= eps * vals.abs() + 1e-4).all()
    if sparse:
        assert (changed.any(1).sum(1) <= sparse).all()


def test_purify_and_bpda_shapes():
    x, _ = ecg_batch(2)
    assert D.purify("ecg", x).shape == x.shape
    xr = x.clone().requires_grad_(True)
    D.bpda("ecg", xr).sum().backward()
    assert torch.allclose(xr.grad, torch.ones_like(x))
    img = torch.rand(2, 1, 32, 32)
    assert D.purify("cxr", img).shape == img.shape
    e, _ = ehr_batch(2)
    assert D.purify("ehr", e)[0].shape == e[0].shape


def test_certify_constant_classifier():
    f = lambda x: torch.full((x.shape[0],), 5.0)      # always class 1 -> large radius
    p, r = D.certify(f, "ecg", torch.zeros(3, 12, C.ECG_SAMPLES), 0.0, n=200, n0=20)
    assert (p == 1).all() and (r > 0).all()


def test_mahalanobis_and_metrics():
    rng = np.random.default_rng(0)
    feats, y = rng.normal(size=(200, 5)), rng.integers(0, 2, 200)
    det = D.Mahalanobis().fit(feats, y)
    s = det.score(torch.from_numpy(feats.astype(np.float32)))
    far = det.score(torch.full((1, 5), 50.0))
    assert far.item() > s.max().item()
    p = rng.random(200)
    thr = M.threshold_at_sensitivity(y, p, 0.9)
    assert (p[y == 1] >= thr).mean() >= 0.9
    r = M.attack_metrics(y, p, np.clip(p + 0.2 * (1 - 2 * y), 0, 1), thr, n_boot=50)
    assert r["robust_accuracy"] <= r["accuracy_clean"] + 1e-9
    assert M.holm({"a": 0.01, "b": 0.04})["b"] == pytest.approx(0.04)
