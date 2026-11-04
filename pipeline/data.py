"""Task data in one interface: indices per split, batches as (model input, primary label, label matrix)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import common as U
import config as C


class TaskData:
    def __init__(self, task: str):
        self.task = task
        d = U.task_dir(task)
        self.df = pd.read_parquet(d / "cohort.parquet")
        if task == "cxr":
            self.arr = np.load(d / "images.npy", mmap_mode="r")
            lab = self.df[list(C.CXR_LABELS)].to_numpy()
            self.ymat = lab.astype(np.float32)                 # 1 / 0 / -1 (uncertain -> masked)
            self.y = (self.df[C.CXR_PRIMARY] == 1).astype(np.float32).to_numpy()
            self.valid = (self.df[C.CXR_PRIMARY] != -1).to_numpy()   # primary-label rows for decisions
        elif task == "ecg":
            self.arr = np.load(d / "waveforms.npy", mmap_mode="r")
            self.y = self.df[C.ECG_PRIMARY].astype(np.float32).to_numpy()
            self.ymat = self.y[:, None]
            self.valid = np.ones(len(self.df), bool)
        else:
            z = np.load(d / "arrays.npz")
            self.vals, self.mask = z["vals"], z["mask"].astype(np.float32)
            self.static = np.stack([(self.df["age"].to_numpy() - 65.0) / 15.0,
                                    self.df["female"].to_numpy()], 1).astype(np.float32)
            self.y = self.df["y"].astype(np.float32).to_numpy()
            self.ymat = self.y[:, None]
            self.valid = np.ones(len(self.df), bool)
        self.rows = self.df["row"].to_numpy()
        self.subjects = self.df["subject_id"].to_numpy()

    def idx(self, split: str, primary_only: bool = False) -> np.ndarray:
        m = (self.df["split"] == split).to_numpy().copy()     # pandas 3 may return a read-only view
        if primary_only:
            m &= self.valid
        return np.nonzero(m)[0]

    def x(self, idx):
        idx = np.asarray(idx)
        if self.task == "cxr":
            order = np.argsort(self.rows[idx])                 # sorted reads from the memmap
            a = np.empty((len(idx), 1, C.CXR_SIZE, C.CXR_SIZE), np.float32)
            a[order, 0] = self.arr[self.rows[idx][order]] / 255.0
            return torch.from_numpy(a)
        if self.task == "ecg":
            order = np.argsort(self.rows[idx])
            a = np.empty((len(idx), 12, C.ECG_SAMPLES), np.float32)
            a[order] = self.arr[self.rows[idx][order]]
            return torch.from_numpy(a)
        r = self.rows[idx]
        return (torch.from_numpy(self.vals[r]), torch.from_numpy(self.mask[r]), torch.from_numpy(self.static[idx]))

    def batch(self, idx):
        return self.x(idx), torch.from_numpy(self.y[idx]), torch.from_numpy(self.ymat[idx])

    # -- CXR: frozen-part activations, computed once, for clean training only (attacks use full images)
    def cxr_cache(self, model):
        path = U.task_dir("cxr") / "frozen_cache.npy"
        n = len(self.arr)
        if not path.exists():
            tmp = path.with_name("frozen_cache.tmp.npy")
            done_file = path.with_name("frozen_cache.progress")      # rows finished so far (resume after a crash)
            start = int(done_file.read_text()) if tmp.exists() and done_file.exists() else 0
            mm = (np.lib.format.open_memmap(tmp, mode="r+") if start else
                  np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, 512, 7, 7)))
            if start:
                U.log(f"  cxr cache resumed at {start}/{n}")
            model.eval()
            with torch.no_grad():
                for b in range(start, n, 64):
                    rows = np.arange(b, min(n, b + 64))
                    x = torch.from_numpy(self.arr[rows][:, None].astype(np.float32) / 255.0)
                    mm[rows] = model.frozen_part(x).numpy().astype(np.float16)
                    if (b // 64) % 20 == 19:
                        mm.flush()
                        done_file.write_text(str(rows[-1] + 1))
                    if b % 6400 == 0:
                        U.log(f"  cxr cache {b}/{n}")
            mm.flush()
            del mm
            import os
            os.replace(tmp, path)
            done_file.unlink(missing_ok=True)
        self.cache = np.load(path, mmap_mode="r")
        return self.cache

    def cached(self, idx):
        r = self.rows[np.asarray(idx)]
        order = np.argsort(r)
        a = np.empty((len(r), 512, 7, 7), np.float32)
        a[order] = self.cache[r[order]]
        return torch.from_numpy(a)

    def norm_stats(self):
        """Train-split statistics for in-model normalisation (ECG per lead; EHR per variable, observed only)."""
        tr = self.idx("train")
        if self.task == "ecg":
            s = self.x(tr[:5000])
            return s.mean((0, 2)).numpy(), s.std((0, 2)).numpy() + 1e-6
        if self.task == "ehr":
            v, m = self.vals[self.rows[tr]], self.mask[self.rows[tr]].astype(bool)
            mean = np.array([v[..., j][m[..., j]].mean() if m[..., j].any() else 0.0 for j in range(v.shape[-1])])
            std = np.array([v[..., j][m[..., j]].std() if m[..., j].any() else 1.0 for j in range(v.shape[-1])])
            return mean.astype(np.float32), (std + 1e-6).astype(np.float32)
        return None, None


def to_device_free(x):
    return x


def slice_x(x, i):
    """Index a model input (tensor or EHR tuple) by a tensor/array of positions."""
    if isinstance(x, tuple):
        return tuple(t[i] for t in x)
    return x[i]


def cat_x(parts):
    if isinstance(parts[0], tuple):
        return tuple(torch.cat([p[k] for p in parts]) for k in range(len(parts[0])))
    return torch.cat(parts)
