"""Models for the three tasks. Every model maps the task's *raw input space* (the space attacks act in)
to one logit for the primary decision label (CXR: 5 logits, primary first used for decisions).

    CXR  x in [0, 1], (B, 1, 224, 224)       -> DenseNet-121 (CheXpert weights), last block trainable
    ECG  x in mV, (B, 12, 1000)               -> 1-D ResNet (target) / plain CNN (surrogate)
    EHR  x = raw values (B, 24, F) + mask     -> GRU with in-model forward fill (target) / 1-D CNN (surrogate)

`features(x)` returns the penultimate representation (Mahalanobis detector, D5).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


# --------------------------------------------------------------------------- CXR
class CXRNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchxrayvision as xrv
        base = xrv.models.DenseNet(weights=C.CXR_WEIGHTS) if pretrained else xrv.models.DenseNet(num_classes=18)
        self.features_net = base.features
        self.head = nn.Linear(1024, len(C.CXR_LABELS))
        self.primary = C.CXR_LABELS.index(C.CXR_PRIMARY)
        for name, p in self.features_net.named_parameters():
            p.requires_grad = name.startswith("denseblock4") or name.startswith("norm5")

    def frozen_part(self, x01: torch.Tensor) -> torch.Tensor:
        h = (2 * x01 - 1) * 1024                      # torchxrayvision intensity range
        for name, mod in self.features_net.named_children():
            if name == "denseblock4":
                break
            h = mod(h)
        return h

    def trainable_part(self, h: torch.Tensor) -> torch.Tensor:
        h = self.features_net.denseblock4(h)
        h = self.features_net.norm5(h)
        return F.adaptive_avg_pool2d(F.relu(h), 1).flatten(1)

    def features(self, x01):
        return self.trainable_part(self.frozen_part(x01))

    def all_logits(self, x01):
        return self.head(self.features(x01))

    def forward(self, x01):
        return self.all_logits(x01)[:, self.primary]


# --------------------------------------------------------------------------- ECG
class _Res1d(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 9, stride, 4, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 9, 1, 4, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.skip = (nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))
                     if (cin != cout or stride != 1) else nn.Identity())

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(h)) + self.skip(x))


class ECGNet(nn.Module):
    """Small 1-D ResNet. Per-lead normalisation (train mean/std) happens inside the model."""

    def __init__(self, mean=None, std=None, width=32):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, 12, 1) if mean is None else torch.as_tensor(mean).view(1, 12, 1))
        self.register_buffer("std", torch.ones(1, 12, 1) if std is None else torch.as_tensor(std).view(1, 12, 1))
        self.stem = nn.Sequential(nn.Conv1d(12, width, 15, 2, 7, bias=False), nn.BatchNorm1d(width), nn.ReLU())
        self.blocks = nn.Sequential(_Res1d(width, width, 1), _Res1d(width, 2 * width, 2),
                                    _Res1d(2 * width, 4 * width, 2), _Res1d(4 * width, 4 * width, 2))
        self.head = nn.Linear(4 * width, 1)

    def features(self, x):
        h = self.blocks(self.stem((x - self.mean) / self.std))
        return h.mean(-1)

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


class ECGSurrogate(nn.Module):
    """Plain CNN (different architecture) for transfer attacks."""

    def __init__(self, mean=None, std=None):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, 12, 1) if mean is None else torch.as_tensor(mean).view(1, 12, 1))
        self.register_buffer("std", torch.ones(1, 12, 1) if std is None else torch.as_tensor(std).view(1, 12, 1))
        layers, c = [], 12
        for cout in (32, 64, 64, 128):
            layers += [nn.Conv1d(c, cout, 7, 2, 3), nn.BatchNorm1d(cout), nn.ReLU()]
            c = cout
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(128, 1)

    def features(self, x):
        return self.net((x - self.mean) / self.std).amax(-1)

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


# --------------------------------------------------------------------------- EHR
def forward_fill(z: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable last-observation-carried-forward over time (dim 1); start value 0 (= train mean
    after normalisation). Also returns hours since last observation, capped and scaled to [0, 1]."""
    t = torch.arange(z.shape[1]).view(1, -1, 1).expand_as(z)
    idx = torch.where(m > 0, t, torch.full_like(t, -1)).cummax(dim=1).values      # last observed bin
    seen = idx >= 0
    zf = torch.gather(z, 1, idx.clamp(min=0)) * seen
    delta = torch.where(seen, ((t - idx).float() / C.DELTA_CAP_HOURS).clamp(max=1.0), torch.ones_like(z))
    return zf, delta


class _EHRBase(nn.Module):
    def __init__(self, mean=None, std=None, n_static: int = 2):
        super().__init__()
        f = len(C.EHR_VARIABLES)
        self.register_buffer("mean", torch.zeros(f) if mean is None else torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.ones(f) if std is None else torch.as_tensor(std, dtype=torch.float32))
        self.n_static = n_static

    def encode_inputs(self, vals, mask):
        z = (vals - self.mean) / self.std * mask
        zf, delta = forward_fill(z, mask)
        return torch.cat([zf, mask, delta], -1)


class EHRNet(_EHRBase):
    """GRU over [forward-filled normalised values, mask, time since last value] + static (age, sex)."""

    def __init__(self, mean=None, std=None, n_static: int = 2):
        super().__init__(mean, std, n_static)
        f = len(C.EHR_VARIABLES)
        self.gru = nn.GRU(3 * f, C.GRU_HIDDEN, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(C.GRU_HIDDEN + n_static, 32), nn.ReLU(), nn.Linear(32, 1))

    def features(self, x):
        vals, mask, static = x
        h, _ = self.gru(self.encode_inputs(vals, mask))
        return torch.cat([h[:, -1], static], -1)

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


class EHRSurrogate(_EHRBase):
    """1-D CNN over time (different architecture) for transfer attacks."""

    def __init__(self, mean=None, std=None, n_static: int = 2):
        super().__init__(mean, std, n_static)
        f = len(C.EHR_VARIABLES)
        self.conv = nn.Sequential(nn.Conv1d(3 * f, 64, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(64, 64, 3, padding=1), nn.ReLU())
        self.head = nn.Linear(64 + n_static, 1)

    def features(self, x):
        vals, mask, static = x
        h = self.conv(self.encode_inputs(vals, mask).transpose(1, 2)).amax(-1)
        return torch.cat([h, static], -1)

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


def ehr_summary_features(vals: np.ndarray, mask: np.ndarray, static: np.ndarray) -> np.ndarray:
    """Last / min / max / mean / count per variable + static, for the gradient-boosting model."""
    m = mask.astype(bool)
    big = np.where(m, vals, np.nan)
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        mn, mx, mean = np.nanmin(big, 1), np.nanmax(big, 1), np.nanmean(big, 1)
    idx = np.where(m, np.arange(m.shape[1])[None, :, None], -1).max(1)
    last = np.take_along_axis(vals, np.clip(idx, 0, None)[:, None, :], 1)[:, 0]
    last = np.where(idx >= 0, last, np.nan)
    return np.concatenate([last, mn, mx, mean, m.sum(1), static], 1).astype(np.float32)
