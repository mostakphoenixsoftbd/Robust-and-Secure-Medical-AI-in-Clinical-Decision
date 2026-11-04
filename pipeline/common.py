"""Shared helpers: DuckDB, leakage guards, splits, JSON output, threads.

The DuckDB and leakage helpers follow the sepsis project's db.py / leakage.py (same author group).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C


# --------------------------------------------------------------------------- runtime
def set_threads() -> None:
    import torch
    torch.set_num_threads(C.CPU_THREADS)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_jsonable), encoding="utf-8")
    os.replace(tmp, path)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def task_dir(task: str) -> Path:
    d = C.OUTPUT_DIR / task
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- duckdb
def connect():
    import duckdb
    C.DUCKDB_TEMP.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{C.DUCKDB_MEMORY}'")
    con.execute(f"SET threads={int(C.DUCKDB_THREADS)}")
    con.execute(f"SET temp_directory='{sql_path(C.DUCKDB_TEMP)}'")
    con.execute("SET preserve_insertion_order=false")
    return con


def sql_path(p) -> str:
    return str(p).replace("\\", "/").replace("'", "''")


def read_csv(path: Path) -> str:
    return (f"read_csv('{sql_path(path)}', header=true, all_varchar=true, delim=',', quote='\"', "
            f"escape='\"', max_line_size=10000000)")


def parquet(path: Path) -> str:
    return f"read_parquet('{sql_path(path)}')"


def in_list(values) -> str:
    return "(" + ", ".join(str(int(v)) for v in values) + ")"


# --------------------------------------------------------------------------- leakage
def fail(problems: list[str]) -> None:
    if problems:
        sys.exit("LEAKAGE CHECK FAILED: " + "; ".join(problems))


def disjoint_splits(subject, split) -> list[str]:
    subject, split = pd.Series(np.asarray(subject)), pd.Series(np.asarray(split))
    sets = {s: set(subject[split == s]) for s in pd.unique(split)}
    keys = sorted(sets)
    out = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if sets[a] & sets[b]:
                out.append(f"patients shared between {a} and {b}")
    return out


# --------------------------------------------------------------------------- splits / subsets
def patient_split(subjects: np.ndarray, labels: np.ndarray, seed: int = C.RANDOM_SEED) -> np.ndarray:
    """Patient-level split, stratified by whether the patient has any positive label."""
    df = pd.DataFrame({"s": subjects, "y": labels})
    pat = df.groupby("s")["y"].max().reset_index()
    rng = np.random.default_rng(seed)
    assign = {}
    for _, grp in pat.groupby("y"):
        ids = grp["s"].to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        n_tr = int(round(n * C.SPLIT_FRACTIONS["train"]))
        n_va = int(round(n * C.SPLIT_FRACTIONS["val"]))
        for i, s in enumerate(ids):
            assign[s] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    out = np.array([assign[s] for s in subjects])
    fail(disjoint_splits(subjects, out))
    return out


def stratified_subset(idx: np.ndarray, y: np.ndarray, subjects: np.ndarray, n: int, seed: int) -> np.ndarray:
    """One row per patient, label-stratified, at most n rows; returns positions into idx."""
    df = pd.DataFrame({"pos": np.arange(len(idx)), "y": y[idx], "s": subjects[idx]})
    rng = np.random.default_rng(seed)
    df = df.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).drop_duplicates("s")
    if len(df) <= n:
        return np.sort(df["pos"].to_numpy())
    frac = n / len(df)
    parts = [g.head(max(1, int(round(len(g) * frac)))) for _, g in df.groupby("y")]
    return np.sort(pd.concat(parts)["pos"].to_numpy()[:n])
