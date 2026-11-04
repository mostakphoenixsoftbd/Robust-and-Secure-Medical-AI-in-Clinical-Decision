"""EHR task (PROTOCOL 3.3): ICU deterioration within 24 h after t0 = ICU intime + 24 h.

    python build_ehr.py

Population: adult (>= 18) first ICU stay per patient, still in the ICU at t0, alive at t0, no vasopressor
and no invasive ventilation started before t0. Outcome: death, vasopressor start or invasive ventilation
start in (t0, t0 + 24 h]. Inputs: vitals (chartevents) and labs (labevents) charted in [intime, t0) and
stored before t0 (storetime rule), last value per hourly bin.

Reads the filtered MIMIC-IV 3.1 tables extracted by the sepsis project (C.SEPSIS_CACHE, read-only).
Writes
    ehr/cohort.parquet        subject_id, stay_id, y (+ components), age, female, split, row
    ehr/arrays.npz            vals (N, 24, F) float32 raw units (0 where missing), mask (N, 24, F) uint8
    ehr/cohort_summary.json   aggregate counts only
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import common as U
import config as C


def cache(name: str) -> str:
    return U.parquet(C.SEPSIS_CACHE / f"{name}.parquet")


def cohort(con) -> tuple[pd.DataFrame, dict]:
    vaso, vent = U.in_list(C.VASO_ITEMIDS), int(C.INVASIVE_VENT_ITEMID)
    con.execute(f"""
    CREATE TEMP TABLE stays AS
    WITH first_stay AS (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY intime) AS rn FROM {cache('icustays')}
    )
    SELECT s.subject_id, s.hadm_id, s.stay_id, s.intime, s.outtime,
           s.intime + INTERVAL {C.OBS_HOURS} HOUR AS t0,
           p.anchor_age + (year(s.intime) - p.anchor_year) AS age,
           CASE WHEN p.gender = 'F' THEN 1 ELSE 0 END AS female,
           a.deathtime
    FROM first_stay s
    JOIN {cache('patients')} p USING (subject_id)
    JOIN {cache('admissions')} a ON a.hadm_id = s.hadm_id
    WHERE s.rn = 1
    """)
    con.execute(f"""
    CREATE TEMP TABLE events AS
    SELECT st.stay_id, v.vaso_start, p.vent_start
    FROM stays st
    LEFT JOIN (SELECT stay_id, MIN(starttime) AS vaso_start FROM {cache('inputevents')}
               WHERE itemid IN {vaso} GROUP BY stay_id) v USING (stay_id)
    LEFT JOIN (SELECT stay_id, MIN(starttime) AS vent_start FROM {cache('procedureevents')}
               WHERE itemid = {vent} GROUP BY stay_id) p USING (stay_id)
    """)
    df = con.execute("SELECT s.*, e.vaso_start, e.vent_start FROM stays s JOIN events e USING (stay_id)").df()
    flow = {"first_icu_stays": int(len(df))}
    df = df[df.age >= C.MIN_AGE]; flow["adults"] = int(len(df))
    df = df[df.outtime > df.t0]; flow["in_icu_at_t0"] = int(len(df))
    df = df[~(df.deathtime.notna() & (df.deathtime <= df.t0))]; flow["alive_at_t0"] = int(len(df))
    pre = (df.vaso_start.notna() & (df.vaso_start <= df.t0)) | (df.vent_start.notna() & (df.vent_start <= df.t0))
    flow["excluded_support_before_t0"] = int(pre.sum())
    df = df[~pre].copy(); flow["final_stays"] = int(len(df))
    end = df.t0 + pd.Timedelta(hours=C.HORIZON_HOURS)

    def inwin(t):
        return (t.notna() & (t > df.t0) & (t <= end)).astype(np.int8)
    df["y_death"], df["y_vaso"], df["y_vent"] = inwin(df.deathtime), inwin(df.vaso_start), inwin(df.vent_start)
    df["y"] = df[["y_death", "y_vaso", "y_vent"]].max(axis=1).astype(np.int8)
    return df.reset_index(drop=True), flow


def features(con, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    var_index = {v: i for i, v in enumerate(C.EHR_VARIABLES)}
    item_rows = [(it, var_index[v]) for v, (ids, _) in {**C.VITALS, **C.LABS}.items() for it in ids]
    con.register("itemmap_df", pd.DataFrame(item_rows, columns=["itemid", "var"]))
    con.register("coh_df", df[["subject_id", "stay_id", "intime", "t0"]].assign(row=np.arange(len(df))))
    rng = pd.DataFrame([(var_index[v], lo, hi) for v, (lo, hi) in C.EHR_RANGES.items()], columns=["var", "lo", "hi"])
    con.register("range_df", rng)
    chart = f"""
        SELECT c.row, m.var, e.charttime, e.itemid, e.valuenum AS v
        FROM {cache('chartevents')} e JOIN coh_df c ON e.stay_id = c.stay_id
        JOIN itemmap_df m ON m.itemid = e.itemid
        WHERE e.charttime >= c.intime AND e.charttime < c.t0 AND COALESCE(e.storetime, e.charttime) < c.t0
          AND e.valuenum IS NOT NULL"""
    labs = f"""
        SELECT c.row, m.var, e.charttime, e.itemid, e.valuenum AS v
        FROM {cache('labevents')} e JOIN coh_df c ON e.subject_id = c.subject_id
        JOIN itemmap_df m ON m.itemid = e.itemid
        WHERE e.charttime >= c.intime AND e.charttime < c.t0 AND COALESCE(e.storetime, e.charttime) < c.t0
          AND e.valuenum IS NOT NULL"""
    q = f"""
    WITH raw AS ({chart} UNION ALL {labs}),
    conv AS (
        SELECT r.row, r.var, r.charttime,
            CASE WHEN r.itemid = {C.TEMP_F_ITEMID} THEN (r.v - 32) * 5.0 / 9.0
                 WHEN r.itemid = {C.FIO2_ITEMID} AND r.v <= 1 THEN r.v * 100
                 ELSE r.v END AS v
        FROM raw r
    ),
    valid AS (SELECT c.* FROM conv c JOIN range_df g ON g.var = c.var WHERE c.v > g.lo AND c.v <= g.hi),
    binned AS (
        SELECT v.row, v.var,
            CAST(FLOOR(date_diff('second', k.intime, v.charttime) / 3600.0) AS INTEGER) AS bin,
            v.v, v.charttime
        FROM valid v JOIN coh_df k ON k.row = v.row
    )
    SELECT row, var, bin, arg_max(v, charttime) AS v FROM binned GROUP BY row, var, bin
    """
    tab = con.execute(q).df()
    # [intime, t0) with t0 = intime + 24 h maps to bins 0..23; anything else means the time filter broke
    U.fail([f"{int(((tab.bin < 0) | (tab.bin >= C.N_BINS)).sum())} values outside [intime, t0)"]
           if ((tab.bin < 0) | (tab.bin >= C.N_BINS)).any() else [])
    vals = np.zeros((len(df), C.N_BINS, len(C.EHR_VARIABLES)), np.float32)
    mask = np.zeros(vals.shape, np.uint8)
    r, b, v = tab.row.to_numpy(), tab.bin.to_numpy(), tab["var"].to_numpy()
    vals[r, b, v] = tab.v.to_numpy(np.float32)
    mask[r, b, v] = 1
    return vals, mask


def main() -> int:
    out = U.task_dir("ehr")
    con = U.connect()
    df, flow = cohort(con)
    U.log(f"cohort {flow}")
    vals, mask = features(con, df)
    df["row"] = np.arange(len(df))
    df["split"] = U.patient_split(df["subject_id"].to_numpy(), df["y"].to_numpy())
    np.savez(out / "arrays.npz", vals=vals, mask=mask)
    df[["subject_id", "stay_id", "y", "y_death", "y_vaso", "y_vent", "age", "female", "split", "row"]] \
        .to_parquet(out / "cohort.parquet", index=False)
    obs = mask.any(axis=1).mean(axis=0)
    summ = {"flow": flow,
            "splits": {s: {"n": int((df.split == s).sum()), "positives": int(df.loc[df.split == s, "y"].sum())}
                       for s in ("train", "val", "test")},
            "outcome_components": {"death_cases": int(df.y_death.sum()), "vaso_cases": int(df.y_vaso.sum()),
                                   "vent_cases": int(df.y_vent.sum()), "any_cases": int(df.y.sum())},
            "prevalence": round(float(df.y.mean()), 4),
            "variables": len(C.EHR_VARIABLES),
            "fraction_stays_with_variable": {v: round(float(obs[i]), 3) for i, v in enumerate(C.EHR_VARIABLES)},
            "median_age": float(np.median(df.age)), "female_fraction": round(float(df.female.mean()), 3)}
    U.save_json(summ, out / "cohort_summary.json")
    U.log(f"done: {flow['final_stays']} stays, prevalence {summ['prevalence']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
