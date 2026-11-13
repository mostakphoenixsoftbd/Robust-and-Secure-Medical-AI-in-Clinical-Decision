# Robust and Secure Medical AI

Code for the study *Robust and Secure Medical AI: Systematic Mitigation of Adversarial Attacks in Clinical
Decision Support Systems*.

One benchmark, three clinical decision support models on MIMIC data (pleural effusion on chest
radiographs, atrial fibrillation/flutter on 12-lead ECGs, ICU deterioration within 24 h from vitals and
laboratory values). Each model is attacked with the same family of evasion attacks (FGSM, PGD, APGD,
Square, transfer, plus adaptive attacks against each defence) under modality-specific, clinically
constrained budgets, and protected by the same defences (D0 undefended; D1 adversarial training;
D2 TRADES; D3 noise training with randomized smoothing; D4 feature squeezing; D5 feature-space
detection; D6 ensemble). Outcomes include AUROC under attack, robust accuracy, decisions flipped per
1,000 patients at a threshold fixed for 90% sensitivity, and certified accuracy.

## Data

No patient data, model weights, predictions or adversarial examples are included. The data come from
PhysioNet: MIMIC-IV v3.1 and MIMIC-CXR-JPG v2.1.0 (credentialed access, data use agreement) and
MIMIC-IV-ECG v1.0 (open access). Paths are set in `pipeline/config.py` (`PHYSIONET`). Row-level outputs
are written outside the repository and must not be shared.

## Requirements

Python 3.11, CPU only, packages in `requirements.txt`. 16 GB RAM; run one task at a time.

## Running (from `pipeline/`)

1. `python build_ehr.py`, `build_ecg.py`, `build_cxr.py` - cohorts, patient-level splits, features.
2. `bash run_retry.sh <ehr|ecg|cxr>` - train D0-D6, attack grid, certification (auto-restart; resumes
   from cached weights and attack results). Smoke test: `python run_task.py <task> --limit 50 --one-eps`.
   `PIPELINE_CXR_EPS=0.5,2,4` selects the CXR budgets (units of 1/255) used in the paper.
3. `python share_results.py`, `make_tables.py`, `make_figures.py` - aggregate results, tables, figures.
4. `python check_numbers.py ../manuscript/*.md` - checks every number in the manuscript against the
   aggregate result files.
5. `python -m pytest tests` - unit tests for the attacks.

Helpers used for the paper's CXR run: `bash finish_cxr.sh` waits for the CXR run to finish, then runs
steps 3-4 and the manuscript build (expects the `manuscript/` folder, which is not in this repository);
`python cxr_fill_values.py` prints each CXR result used in the manuscript with the key path it comes from.
Paths in `config.py`, `run_retry.sh` and `finish_cxr.sh` point at the authors' machine (`E:/physionet`);
change them (or set `ROBUST_OUTPUT_DIR`) before running.

## Changes

- 7 Nov 2026: `run_task.py` caches clean scores per model (local, resumes after a crash without
  rescoring) and runs CXR gradient attacks in batches of 8 images (`ATTACK_BS`; memory limit on a 16 GB
  CPU machine; per-image attacks, so only the seeded random start differs); `make_tables.py` rounds
  budget labels (0.5/255); added `finish_cxr.sh` and `cxr_fill_values.py`.
- 4 Nov 2026: first release.

The study protocol (budgets, defences, metrics, deviations) is in `PROTOCOL.md`.

## Licence

See `LICENSE`.
