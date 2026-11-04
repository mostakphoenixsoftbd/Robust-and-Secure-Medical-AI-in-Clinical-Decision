"""ECG task (PROTOCOL 3.2): cohort, machine-statement labels, preprocessed waveforms.

    python build_ecg.py

Reads record_list.csv + machine_measurements.csv, keeps records whose waveform is on disk, takes the
first record per patient (independent samples), labels it from the machine statements
(C.ECG_LABEL_RULES), and writes
    ecg/cohort.parquet        subject_id, study_id, labels, split, row (index into the array)
    ecg/waveforms.npy         float16 (N, 12, 1000), mV, band-passed, 100 Hz
    ecg/cohort_summary.json   aggregate counts only
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy import signal

import common as U
import config as C


def label_table() -> pd.DataFrame:
    mm = pd.read_csv(C.ECG_ROOT / "machine_measurements.csv", low_memory=False,
                     usecols=lambda c: c in ("subject_id", "study_id") or c.startswith("report_"))
    rep = [c for c in mm.columns if c.startswith("report_")]
    text = mm[rep].fillna("").astype(str).agg(" | ".join, axis=1).str.lower()
    out = mm[["subject_id", "study_id"]].copy()
    for name, pats in C.ECG_LABEL_RULES.items():
        hit = np.zeros(len(text), bool)
        for p in pats:
            hit |= text.str.contains(p, regex=False).to_numpy()
        out[name] = hit.astype(np.int8)
    return out


_SOS = signal.butter(3, C.ECG_BANDPASS, btype="bandpass", fs=C.ECG_FS_IN, output="sos")


def load_wave(rel: str) -> np.ndarray | None:
    import wfdb
    try:
        rec = wfdb.rdrecord(str(C.ECG_ROOT / rel))
    except Exception:
        return None
    x = np.asarray(rec.p_signal, np.float64).T                 # (12, 5000) mV
    if x.shape != (12, C.ECG_FS_IN * 10) or rec.fs != C.ECG_FS_IN:
        return None
    if np.isnan(x).mean() > 0.1:
        return None
    x = np.nan_to_num(x)
    x = signal.sosfiltfilt(_SOS, x, axis=1)
    x = signal.resample_poly(x, 1, C.ECG_FS_IN // C.ECG_FS, axis=1)
    return np.clip(x, -C.ECG_CLIP_MV, C.ECG_CLIP_MV).astype(np.float16)


def main() -> int:
    out = U.task_dir("ecg")
    rl = pd.read_csv(C.ECG_ROOT / "record_list.csv", usecols=["subject_id", "study_id", "ecg_time", "path"])
    n_listed = len(rl)
    rl["on_disk"] = [(C.ECG_ROOT / (p + ".hea")).exists() for p in rl["path"]]
    rl = rl[rl["on_disk"]]
    n_disk = len(rl)
    lab = label_table()
    df = rl.merge(lab, on=["subject_id", "study_id"], how="inner")
    df["ecg_time"] = pd.to_datetime(df["ecg_time"])
    df = df.sort_values(["subject_id", "ecg_time"]).drop_duplicates("subject_id").reset_index(drop=True)
    U.log(f"listed {n_listed}, on disk {n_disk}, labelled first-per-patient {len(df)}")

    mm =np.lib.format.open_memmap(out / "waveforms.npy", mode="w+", dtype=np.float16,
                                   shape=(len(df), 12, C.ECG_SAMPLES))
    ok = np.zeros(len(df), bool)
    with ThreadPoolExecutor(6) as ex:
        for i, w in enumerate(ex.map(load_wave, df["path"], chunksize=64)):
            if w is not None:
                mm[i] = w
                ok[i] = True
            if i % 5000 == 0:
                U.log(f"  waveforms {i}/{len(df)}")
    mm.flush()
    del mm
    df["row"] = np.arange(len(df))
    df = df[ok].reset_index(drop=True)
    df["split"] = U.patient_split(df["subject_id"].to_numpy(), df[C.ECG_PRIMARY].to_numpy())
    df.drop(columns=["path", "on_disk"]).to_parquet(out / "cohort.parquet", index=False)

    summ = {"records_listed": int(n_listed), "records_on_disk": int(n_disk),
            "patients": int(len(df)), "unreadable_or_invalid": int((~ok).sum()),
            "splits": {s: {"n": int((df.split == s).sum()),
                           **{f"{k}_positives": int(df.loc[df.split == s, k].sum()) for k in C.ECG_LABEL_RULES}}
                       for s in ("train", "val", "test")},
            "prevalence": {k: round(float(df[k].mean()), 4) for k in C.ECG_LABEL_RULES}}
    U.save_json(summ, out / "cohort_summary.json")
    U.log(f"done: {summ['patients']} patients; AF prevalence {summ['prevalence']['af']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
