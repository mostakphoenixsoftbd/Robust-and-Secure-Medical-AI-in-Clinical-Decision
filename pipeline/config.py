"""Paths and study constants for the adversarial-robustness benchmark (PROTOCOL.md).

Edit only section 1 if the PhysioNet data move. Everything else is the pre-specified design; change it
only deliberately and record the change in PROTOCOL.md's change log.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Data locations
# ---------------------------------------------------------------------------
PHYSIONET = Path(r"E:/physionet")
MIMIC_IV_ROOT = PHYSIONET / "mimiciv/3.1"
CXR_JPG_ROOT = PHYSIONET / "mimic-cxr-jpg/2.1.0"
ECG_ROOT = PHYSIONET / "mimic-iv-ecg/1.0"
# Filtered MIMIC-IV tables already extracted by the sepsis project (read-only reuse; same source files,
# same MIMIC-IV 3.1 release). Columns: see build_ehr.py.
SEPSIS_CACHE = PHYSIONET / "sepsis_outputs/cache"

# Row-level outputs (IDs, arrays, predictions, weights): local only, outside the paper folder.
OUTPUT_DIR = Path(os.environ.get("ROBUST_OUTPUT_DIR", r"E:/physionet/robust_outputs"))
# Aggregate outputs only (copied by share_results.py). The only pipeline folder Claude reads.
SHARE_DIR = Path(__file__).resolve().parents[1] / "pipeline_share"
DUCKDB_TEMP = OUTPUT_DIR / "duckdb_tmp"
DUCKDB_MEMORY = os.environ.get("PIPELINE_DUCKDB_MEMORY", "4GB")
DUCKDB_THREADS = int(os.environ.get("PIPELINE_DUCKDB_THREADS", 4))
CPU_THREADS = int(os.environ.get("PIPELINE_CPU_THREADS", 6))

TASKS = ("ecg", "ehr", "cxr")

# ---------------------------------------------------------------------------
# 2. Common design (PROTOCOL section 2)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.10, "test": 0.20}
SEEDS = (0, 1, 2)                    # D0 trained with each; D6 = their ensemble; defences use seed 0
TARGET_SENSITIVITY = 0.90            # operating threshold fixed on validation
# CPU budget (shared machine): label-stratified, one record per patient.
ATTACK_SUBSET = {"ecg": 1000, "ehr": 1500, "cxr": 200}
CERT_SUBSET = {"ecg": 300, "ehr": 500, "cxr": 50}
BOOTSTRAP = 1000
SMALL_CELL = 10

# ---------------------------------------------------------------------------
# 3. Tasks
# ---------------------------------------------------------------------------
# CXR (PROTOCOL 3.1)
CXR_VIEWS = ("PA", "AP")
CXR_LABELS = ("Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion")
CXR_PRIMARY = "Pleural Effusion"
CXR_SIZE = 224
CXR_WEIGHTS = "densenet121-res224-chex"   # torchxrayvision, trained on CheXpert only (never saw MIMIC-CXR)

# ECG (PROTOCOL 3.2): labels from machine statements, lower-case substring rules
ECG_FS_IN, ECG_FS = 500, 100
ECG_SAMPLES = 1000
ECG_BANDPASS = (0.5, 40.0)
ECG_CLIP_MV = 5.0
ECG_LABEL_RULES = {
    "af": ["atrial fibrillation", "atrial flutter"],
    "sinus_brady": ["sinus bradycardia"],
    "sinus_tachy": ["sinus tachycardia"],
    "lbbb": ["left bundle branch block"],
    "rbbb": ["right bundle branch block"],
}
ECG_PRIMARY = "af"

# EHR (PROTOCOL 3.3): t0 = ICU intime + 24 h; outcome in (t0, t0 + 24 h]
MIN_AGE = 18
OBS_HOURS = 24
HORIZON_HOURS = 24
N_BINS = 24
# vitals: name -> (itemids, valid range (low, high]) -- from the sepsis project / mimic-code vitalsign.sql
VITALS = {
    "heart_rate":  ([220045], (0, 300)),
    "sbp":         ([220179, 220050, 225309], (0, 400)),
    "dbp":         ([220180, 220051, 225310], (0, 300)),
    "mbp":         ([220181, 220052, 225312], (0, 300)),
    "resp_rate":   ([220210, 224690], (0, 70)),
    "spo2":        ([220277], (0, 100.001)),
    "temperature": ([223762, 223761], (10, 50)),     # 223761 Fahrenheit, converted
    "gcs_eye":     ([220739], (0, 4.001)),
    "gcs_verbal":  ([223900], (0, 5.001)),
    "gcs_motor":   ([223901], (0, 6.001)),
    "fio2":        ([223835], (20, 100.001)),         # converted to %
}
TEMP_F_ITEMID = 223761
FIO2_ITEMID = 223835
# labs: name -> (itemids, valid range (low, high]); blood gases left out (arterial specimen join)
LABS = {
    "wbc":         ([51301, 51300], (0, 1000)),
    "lactate":     ([50813], (0, 50)),
    "creatinine":  ([50912], (0, 150)),
    "bun":         ([51006], (0, 300)),
    "bilirubin":   ([50885], (0, 150)),
    "platelets":   ([51265], (0, 10000)),
    "hemoglobin":  ([51222], (0, 30)),
    "sodium":      ([50983], (50, 200)),
    "potassium":   ([50971], (0, 30)),
    "chloride":    ([50902], (50, 200)),
    "bicarbonate": ([50882], (0, 100)),
    "glucose":     ([50931], (0, 3000)),
    "inr":         ([51237], (0, 30)),
}
EHR_VARIABLES = list(VITALS) + list(LABS)
EHR_RANGES = {k: v[1] for k, v in {**VITALS, **LABS}.items()}
# Mutable = continuous measurements an attacker could alter in the record. GCS items are ordinal
# scores (discrete) and are held fixed.
EHR_IMMUTABLE = ("gcs_eye", "gcs_verbal", "gcs_motor")
VASO_ITEMIDS = [221906, 221289, 221662, 221653, 221749, 222315]   # norepi, epi, dopa, dobu, phenyl, vasopressin
INVASIVE_VENT_ITEMID = 225792
DELTA_CAP_HOURS = 24

# ---------------------------------------------------------------------------
# 4. Threat models and budgets (PROTOCOL section 4)
# ---------------------------------------------------------------------------
# CXR pixels in [0, 1]; ECG in mV; EHR: relative change of each observed mutable value.
EPS = {
    "cxr": (0.5 / 255, 1 / 255, 2 / 255, 4 / 255, 8 / 255),
    "ecg": (0.010, 0.025, 0.050, 0.100),
    "ehr": (0.01, 0.02, 0.05, 0.10),
}
# Optional override of the CXR budget grid (units of 1/255, comma-separated, run in the given order), e.g.
# PIPELINE_CXR_EPS=0.5,2,4 for the deadline run of 4 Nov 2026; further budgets resume from adv_cache later.
if os.environ.get("PIPELINE_CXR_EPS"):
    EPS["cxr"] = tuple(float(v) / 255 for v in os.environ["PIPELINE_CXR_EPS"].split(","))
EPS_TRAIN = {"cxr": 2 / 255, "ecg": 0.050, "ehr": 0.05}       # adversarial training budget
EPS_PRIMARY = {"cxr": 2 / 255, "ecg": 0.050, "ehr": 0.05}     # used for single-budget experiments
EHR_SPARSE_FEATURES = 3                                        # sparse attack: at most 3 variables changed
PGD_STEPS = {"cxr": 10, "ecg": 20, "ehr": 20}
PGD_RESTARTS = {"cxr": 1, "ecg": 1, "ehr": 1}
APGD_STEPS = {"cxr": 20, "ecg": 50, "ehr": 50}
SQUARE_QUERIES = {"cxr": 200, "ecg": 500, "ehr": 500}
SQUARE_SUBSET = {"cxr": 100, "ecg": 300, "ehr": 300}   # score-based attack: first n of the attack set
ECG_SMOOTH_SIGMA = 3.0               # samples (30 ms at 100 Hz), Gaussian kernel for smooth PGD
EOT_SAMPLES = 4                      # expectation over transformation for randomized defences

# ---------------------------------------------------------------------------
# 5. Defences (PROTOCOL section 5)
# ---------------------------------------------------------------------------
DEFENCES = ("D0", "D1", "D2", "D3", "D4", "D5", "D6")
AT_STEPS = {"cxr": 1, "ecg": 3, "ehr": 3}       # CXR: FGSM with random start ("fast AT", CPU budget)
TRADES_BETA = 6.0
NOISE_SIGMA = {"cxr": 0.25, "ecg": 0.10, "ehr": 0.10}   # Gaussian augmentation / smoothing (EHR: relative)
SMOOTH_N_PREDICT = 32
SMOOTH_N_CERT = {"cxr": 100, "ecg": 1000, "ehr": 1000}
SMOOTH_ALPHA = 0.001
DETECT_TPR = 0.95                    # Mahalanobis detector threshold: 95 % of clean validation accepted
DETECT_ADAPTIVE_LAMBDA = (0.1, 1.0)
FT_TRAIN_SUBSET = {"cxr": 6000, "ecg": 15000}   # CPU budget: adversarial / noise fine-tuning on a training subset

# ---------------------------------------------------------------------------
# 6. Training
# ---------------------------------------------------------------------------
TRAIN = {
    "cxr": {"epochs": 6, "patience": 2, "batch": 32, "lr": 1e-4, "ft_epochs": 1, "ft_batch": 16},
    "ecg": {"epochs": 6, "patience": 2, "batch": 128, "lr": 1e-3, "ft_epochs": 4, "ft_batch": 128},
    "ehr": {"epochs": 25, "patience": 4, "batch": 256, "lr": 1e-3, "ft_epochs": 6, "ft_batch": 256},
}
WEIGHT_DECAY = 1e-4
GRU_HIDDEN = 64
GBM_PARAMS = {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.8,
              "subsample_freq": 1, "colsample_bytree": 0.8, "verbose": -1}
