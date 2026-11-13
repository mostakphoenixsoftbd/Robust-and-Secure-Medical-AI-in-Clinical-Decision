# Study protocol — DRAFT v0.1 (NOT YET APPROVED BY THE USER)

> Draft for discussion. Each `OPEN:` item has a **default** (Claude's call) and alternatives. Defaults
> apply unless the user changes them. **[blocking]** items must be confirmed before real-data runs;
> pipeline code and synthetic tests may be written against the defaults.

Title (locked): *Robust and Secure Medical AI: Systematic Mitigation of Adversarial Attacks in Clinical
Decision Support Systems.*
Target: niche Q2 journal (OPEN, see STATUS.md). Reporting: TRIPOD+AI items that apply (model development
and evaluation), plus a robustness-evaluation checklist (threat model, budgets, adaptive attacks;
section 9).

## 0. Research questions

1. **Vulnerability:** how far do clean-accurate CDSS models for chest X-ray, 12-lead ECG and ICU time
   series degrade under white-box, transfer (black-box) and query-based attacks at clinically
   plausible budgets?
2. **Clinical impact:** how many decisions flip at each model's clinical operating threshold, and in
   which direction (missed disease vs false alarm)?
3. **Mitigation:** which defences restore robustness, at what cost in clean accuracy, calibration and
   compute, and do they survive adaptive attacks?
4. **Consistency:** are defence rankings consistent across modalities (the "systematic" claim)?
5. (Secondary) **Training-time threats:** how vulnerable are the models to data poisoning / backdoors,
   and do standard detectors find them?

## 1. Data

| Source | Version | Used for | On disk (30 Oct) |
| --- | --- | --- | --- |
| MIMIC-CXR-JPG | 2.1.0 | frontal JPGs, CheXpert labels, metadata, official split | CSVs + ~26.5k JPGs |
| MIMIC-IV-ECG | 1.0 | 12-lead 10-s waveforms (500 Hz), record_list, machine_measurements (labels) | waveforms ~66k; machine_measurements downloading |
| MIMIC-IV | 3.1 | ICU stays, chartevents (vitals), labevents, inputevents (vasopressors), procedureevents (ventilation), patients/admissions (death) | complete |

Linkage across modalities by `subject_id` (needed only for the optional fusion analysis, section 3.4).

## 2. Common design rules

- Patient-level splits 70/10/20 (train/validation/test), stratified by label; no patient in two splits
  (scripts stop with `LEAKAGE CHECK FAILED`). Same seed list for all tasks (3 seeds).
- Everything tuned on validation (operating thresholds, attack step sizes, defence hyper-parameters);
  test used once per final configuration.
- Clean metrics on the full test set; attacks on a fixed, label-stratified **attack subset** of the
  test set (default n = 1,000 per task; CXR certification n = 200) because of CPU cost. Subset IDs
  are saved and reused for every attack/defence.
- Operating threshold per task: fixed on validation at sensitivity 0.90 (OPEN: or Youden J).

## 3. Tasks and base models

### 3.1 Chest X-ray (CXR) — multi-label findings
- Population: frontal (PA/AP) images with JPG on disk; one image per study.
- Labels: CheXpert five (atelectasis, cardiomegaly, consolidation, edema, pleural effusion) from
  `mimic-cxr-2.0.0-chexpert`; uncertain (-1) -> OPEN: **default U-ignore** (masked in loss and
  metrics), alternatives U-zeros/U-ones. Primary decision label: **pleural effusion** (OPEN) for the
  decision-flip analysis; all five for AUROC.
- Model: DenseNet-121 pretrained on CheXpert (torchxrayvision `densenet121-res224-chex`, the encoder
  already used locally by the sepsis project; it was **not** trained on MIMIC, so no test leakage),
  last dense block + head fine-tuned on CPU at 224 px. Attacks operate in pixel space through the
  whole network.
- Image count: ~26.5k on disk. OPEN: download more JPGs if the per-label positives in test are too few
  (decide from counts).

### 3.2 ECG — rhythm/abnormality classification
- Population: MIMIC-IV-ECG records on disk, one record per patient-visit (OPEN: all records,
  clustered bootstrap by patient).
- Labels from machine statements (`machine_measurements` report fields) by rule-based mapping fixed
  from the statement dictionary and counts only. **Default primary: atrial fibrillation/flutter vs
  not.** Secondary multi-label: sinus bradycardia, sinus tachycardia, LBBB, RBBB, ST/T abnormality
  (OPEN, depends on counts). Limitation: machine labels, not cardiologist labels;
  `waveform_note_links` used to report agreement on the linked subset if feasible.
- Input: 12 leads x 10 s resampled to 100 Hz (1,000 samples), band-pass filtered, per-lead z-score.
- Model: small 1-D ResNet trained from scratch on CPU.

### 3.3 ICU time series (EHR) — short-term deterioration
- Population: adult first ICU stays with length of stay >= 24 h (t0 = ICU intime + 24 h).
- Inputs: vitals (chartevents) and labs (labevents) in [intime, t0), hourly bins, last value + mask +
  time-since-last (GRU-D style); static age and sex. `storetime` availability rule (value used only if
  stored before t0).
- **OPEN [blocking]: outcome.** Default: **deterioration within 24 h after t0** = death, vasopressor
  start, or invasive mechanical ventilation start; patients with vasopressors/ventilation before t0
  excluded. Alternatives: in-ICU mortality (overlaps the sibling blockchain mortality paper), AKI.
- Models: GRU (attack target) and gradient-boosted trees (black-box target via transfer).

### 3.4 Fusion (secondary, OPEN: default include if time allows)
- EHR task patients with a frontal CXR in [t0 - 48 h, t0): late fusion of the CXR embedding and the GRU
  state. Question: does attacking **one** modality suffice to flip the fused decision?

## 4. Threat models

| Threat model | Attacker knowledge | Attacks |
| --- | --- | --- |
| White-box evasion | weights + gradients | FGSM, PGD-L-inf (multi-restart), PGD-L2, Carlini-Wagner L2, AutoAttack (APGD-CE, APGD-DLR/targeted, FAB, Square) |
| Black-box transfer | same data distribution, different seed/architecture surrogate | PGD/MI-FGSM crafted on the surrogate |
| Query-based black-box | scores only | Square attack (query budget 1,000/5,000) |
| (Secondary) training-time | can inject a fraction of training samples | label flipping; trigger backdoor (BadNets-style patch for CXR, low-amplitude waveform pattern for ECG, plausible value pattern for EHR) at 0.5/1/5 % poisoning |

**Perturbation budgets (clinically motivated; final values fixed on validation, OPEN):**
- CXR: L-inf epsilon in {0.5, 1, 2, 4, 8}/255 (intensity range 0-255).
- ECG: L-inf in microvolts {10, 25, 50, 100} µV (typical baseline noise tens of µV), plus a smoothed
  perturbation (Han et al. 2020) so attacks have no square-wave artefacts.
- EHR: only **mutable** measurements (labs and manually charted vitals; not age/sex/masks); each value
  stays within its physiological range and within k x its analytic coefficient of variation (k in {1, 2, 3});
  sparsity budget of at most m changed features (m in {1, 3, 5}); projected PGD onto this feasible
  set. Black-box against the GBM by transfer.

## 5. Defences (same set for every task where meaningful)

| Code | Defence | Notes |
| --- | --- | --- |
| D0 | none (standard training) | reference |
| D1 | PGD adversarial training (Madry) | CXR: last block + head only (CPU) |
| D2 | TRADES (beta tuned on validation) | |
| D3 | Gaussian noise training + randomized smoothing (Cohen) | certified L2 radius; N = 100 for prediction, N = 1,000 for certification on the certification subset |
| D4 | input purification: feature squeezing (CXR bit depth/median; ECG low-pass; EHR rounding to reporting precision) and a denoising autoencoder | |
| D5 | adversarial-input detection (Mahalanobis on penultimate features; LID) | reported as detection AUROC and robust accuracy after rejection |
| D6 | ensemble of D0 models with input-transformation diversity | |
| D7 (secondary) | poisoning/backdoor detection: spectral signatures, activation clustering | for section 4 training-time threats |

**Adaptive attacks are mandatory** for every defence: BPDA for non-differentiable purification, EOT for
randomized defences, attacks through the detector (joint loss). A defence counts as effective only
under its adaptive attack.

## 6. Outcomes and metrics

- Clean: AUROC, AUPRC, sensitivity/specificity at the operating threshold, Brier score, ECE.
- Robust: AUROC and accuracy under attack per epsilon (robustness curves); attack success rate;
  certified accuracy vs L2 radius (D3).
- **Clinical impact (primary decision metric):** decision-flip rate at the operating threshold per
  1,000 patients, split into induced false negatives (missed AF / effusion / deterioration) and induced
  false positives.
- Cost: clean-accuracy drop, calibration change, training and inference CPU time.
- Transferability matrix (surrogate -> target) per task.

## 7. Statistics

- 95 % CIs by patient-level bootstrap (1,000 resamples). Paired comparisons of defences vs D0 on the
  same attack subset (paired bootstrap of the difference); Holm correction across defences within a
  task. Mean +- SD over 3 seeds for training variability.
- Cross-modality consistency: Kendall's W of defence rankings across the three tasks (RQ4).

## 8. Compute plan (CPU only)

- Frozen/partly frozen CXR backbone; precomputed nothing for attacks (gradients must pass through the
  backbone), so CXR attacks run on the attack subset only, PGD-20 default, AutoAttack standard on
  n = 500 (OPEN).
- ECG and EHR models are small; full adversarial training feasible.
- Libraries (draft): PyTorch CPU, torchattacks or adversarial-robustness-toolbox, autoattack,
  torchxrayvision, DuckDB, scikit-learn/LightGBM. Pin versions in `requirements.txt`.

## 9. Reporting checklist (robustness)

Threat model stated; budgets justified clinically; attack hyper-parameters and restarts; adaptive
attacks per defence; gradient-masking sanity checks (unbounded attack -> 0 % accuracy; black-box not
stronger than white-box; more steps never weaker); clean vs robust trade-off; code release.

## 10. Ethics and data

PhysioNet credentialed access and DUA; secondary analysis of de-identified data (no IRB needed per
MIMIC terms; state this). No row-level data or images leave the local machine; figures show aggregate
results only (OPEN: whether example adversarial image panels are allowed under the DUA, check first).
Dual-use statement: attacks are evaluated to motivate defences; code releases defences and evaluation,
attack code uses public libraries.

## Change log

- 2026-10-30 v0.1 drafted by Claude from the user's title and "start" instruction; not approved.
- 2026-10-30 v0.2 (Claude, user asleep and asked for autonomous work; all defaults taken, user to review):
  - EHR outcome = default composite (death / vasopressor start / invasive ventilation start within 24 h
    after t0 = ICU intime + 24 h). Cohort 27,802 stays, prevalence 3.8 %. Blood-gas labs (pH, pO2,
    pCO2) left out of the inputs (arterial-specimen join); 24 variables (11 vitals, 13 labs).
  - EHR inputs read from the sepsis project's filtered MIMIC-IV 3.1 parquet cache (same release).
  - EHR attack set = whole test set (low prevalence; ~38 positives in a 1,000 subset would be too few);
    ECG attack subset 2,000; CXR attack subset 500, Square attack on 200 CXR images (CPU).
  - CXR: one frontal image per study (PA preferred) among the ~26.5k JPGs on disk; split by patient
    (own split, not the official one, which is defined for all 377k images).
  - CXR adversarial / noise fine-tuning: 1 epoch on a 6,000-image training subset with one-step
    random-start FGSM ("fast AT"); D0 trained on cached frozen activations (numerically identical).
  - D1/D2/D3 are fine-tuned from the seed-0 D0 weights (ECG 5 epochs, EHR 6 epochs, PGD-3 inner
    attack), model selection on validation AUROC under PGD-5 (robust overfitting).
  - D4 = feature squeezing only (CXR median 3x3 + 5-bit; ECG 50 ms moving average + 5 µV
    quantisation; EHR rounding to reporting precision); the denoising autoencoder was dropped (CPU).
  - "AutoAttack" replaced by its two components that apply to a single-logit binary decision:
    APGD-CE and Square (score-based); FAB/APGD-DLR need >= 3 classes. Stated as such in the paper.
  - Secondary analyses not run in this pass: poisoning/backdoor (RQ5) and the fusion model (3.4);
    both moved to "future work" unless time allows.
  - EHR immutable variables: GCS eye/verbal/motor (ordinal); all other observed values mutable.
- 2026-10-30 v0.3 (Claude, CPU budget; supersedes the v0.2 subset sizes above, user to review):
  - Attack sets (one record per patient, label-stratified): EHR 1,500, ECG 1,000, CXR 200.
    Square: EHR 300 / ECG 300 / CXR 100 inputs with 500 / 500 / 200 queries; D3 queries scored with
    8 noise draws. Certification sets: EHR 500, ECG 300, CXR 50 (n = 1000 / 1000 / 100 draws).
  - PGD: 1 restart; steps CXR 10, ECG 20, EHR 20. APGD steps 20 / 50 / 50. EOT 4 draws per step.
    Detector-aware attack: lambda in {0.1, 1.0}.
  - Training: ECG D0 6 epochs (patience 2); D1-D3 fine-tuning ECG 4 epochs on a 15,000-record training
    subset, EHR 6 epochs; inner attack PGD-3 (ECG, EHR), FGSM-RS (CXR).
  - ECG fine-tuning raised from 2 to 4 epochs (30 Oct 09:45) after the 2-epoch TRADES model had robust
    validation AUROC 0.17 (still rising; EHR TRADES also converged slowly). Decided on the validation
    trajectory only, before any test-set attack on ECG; D1 and D2 retrained from scratch.
- 2026-11-04 deviation (Claude, CPU time before the 5 Nov submission deadline; user to review):
  - CXR attack budget grid reduced from {0.5, 1, 2, 4, 8}/255 (Section 4) to {0.5, 2, 4}/255 via
    `PIPELINE_CXR_EPS` in `pipeline/config.py`; primary budget 2/255 unchanged. 1/255 and 8/255 can be
    added later from `cxr/adv_cache/`. Stated in Methods 2.5.
- 2026-11-06 deviation (Claude, memory on the shared 16 GB CPU machine; user to review):
  - CXR attack batch size reduced from 16 to 8 (`ATTACK_BS` in `pipeline/run_task.py`) after an out-of-memory
    crash on APGD for D3; applies to APGD D3-D6, Square, transfer and certification. Per-image gradients
    (models in eval mode), so only the partitioning of seeded random draws changes. Stated in Supplement S3.
