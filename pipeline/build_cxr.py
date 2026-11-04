"""CXR task (PROTOCOL 3.1): cohort, CheXpert labels, preprocessed 224 px images.

    python build_cxr.py

One frontal image per study (PA preferred over AP), only images whose JPG is on disk. Labels from
mimic-cxr-2.0.0-chexpert: 1 positive, 0/blank negative, -1 uncertain (stored as -1, masked = U-ignore).
Writes
    cxr/cohort.parquet        subject_id, study_id, dicom_id, labels, split, row
    cxr/images.npy            uint8 (N, 224, 224), centre square crop
    cxr/cohort_summary.json   aggregate counts only
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import common as U
import config as C


def jpg_path(r) -> str:
    s = str(int(r.subject_id))
    return str(C.CXR_JPG_ROOT / "files" / f"p{s[:2]}" / f"p{s}" / f"s{int(r.study_id)}" / f"{r.dicom_id}.jpg")


def load_img(path: str) -> np.ndarray | None:
    from PIL import Image
    try:
        with Image.open(path) as im:
            im = im.convert("L")
            side = min(im.size)
            left, top = (im.width - side) // 2, (im.height - side) // 2
            im = im.crop((left, top, left + side, top + side)).resize((C.CXR_SIZE, C.CXR_SIZE), Image.BILINEAR)
            return np.asarray(im, dtype=np.uint8)
    except Exception:
        return None


def main() -> int:
    out = U.task_dir("cxr")
    meta = pd.read_csv(C.CXR_JPG_ROOT / "mimic-cxr-2.0.0-metadata.csv.gz",
                       usecols=["dicom_id", "subject_id", "study_id", "ViewPosition"])
    meta = meta[meta["ViewPosition"].isin(C.CXR_VIEWS)].copy()
    n_frontal = len(meta)
    meta["path"] = [jpg_path(r) for r in meta.itertuples()]
    from pathlib import Path
    meta = meta[[Path(p).exists() for p in meta["path"]]]
    n_disk = len(meta)
    meta["pa"] = (meta["ViewPosition"] == "PA").astype(int)
    meta = meta.sort_values(["study_id", "pa", "dicom_id"], ascending=[True, False, True]).drop_duplicates("study_id")
    chx = pd.read_csv(C.CXR_JPG_ROOT / "mimic-cxr-2.0.0-chexpert.csv.gz")
    df = meta.merge(chx[["subject_id", "study_id", *C.CXR_LABELS]], on=["subject_id", "study_id"], how="inner")
    for k in C.CXR_LABELS:
        df[k] = df[k].fillna(0).astype(np.int8)
    df = df.reset_index(drop=True)
    U.log(f"frontal {n_frontal}, frontal on disk {n_disk}, studies with labels {len(df)}")

    mm = np.lib.format.open_memmap(out / "images.npy", mode="w+", dtype=np.uint8,
                                   shape=(len(df), C.CXR_SIZE, C.CXR_SIZE))
    ok = np.zeros(len(df), bool)
    with ThreadPoolExecutor(6) as ex:
        for i, im in enumerate(ex.map(load_img, df["path"], chunksize=32)):
            if im is not None:
                mm[i] = im
                ok[i] = True
            if i % 5000 == 0:
                U.log(f"  images {i}/{len(df)}")
    mm.flush()
    del mm
    df["row"] = np.arange(len(df))
    df = df[ok].reset_index(drop=True)
    prim = (df[C.CXR_PRIMARY] == 1).astype(int).to_numpy()
    df["split"] = U.patient_split(df["subject_id"].to_numpy(), prim)
    df.drop(columns=["path", "pa"]).to_parquet(out / "cohort.parquet", index=False)

    def counts(d):
        return {k.replace(" ", "_").lower(): {"positives": int((d[k] == 1).sum()), "uncertain_count": int((d[k] == -1).sum())}
                for k in C.CXR_LABELS}
    summ = {"frontal_images": int(n_frontal), "frontal_images_on_disk": int(n_disk),
            "studies": int(len(df)), "patients": int(df.subject_id.nunique()),
            "unreadable": int((~ok).sum()),
            "views": {v: int((df.ViewPosition == v).sum()) for v in C.CXR_VIEWS},
            "splits": {s: {"studies": int((df.split == s).sum()),
                           "patients": int(df.loc[df.split == s, "subject_id"].nunique()),
                           "labels": counts(df[df.split == s])} for s in ("train", "val", "test")}}
    U.save_json(summ, out / "cohort_summary.json")
    U.log(f"done: {summ['studies']} studies / {summ['patients']} patients")
    return 0


if __name__ == "__main__":
    sys.exit(main())
