"""Copy the aggregate outputs (and nothing else) from OUTPUT_DIR/<task>/ to SHARE_DIR as <task>_<name>.

    python share_results.py            # copy whatever aggregate files exist so far
    python share_results.py --list     # only show what would be copied

Adapted from the sepsis project's share_results.py. Each JSON is refused if it has identifier-like keys
or an 8+ digit integer (MIMIC IDs are 8 digits); count-like fields with values 1-9 become "<10".
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys

import config as C

SHAREABLE = ["cohort_summary.json", "results_*.json", "training_log.json", "*.png", "*.svg"]
ID_KEYS = re.compile(r"(^(subject|hadm|stay|study|dicom|patient|specimen)(_?ids?)?$|_ids?$|^ids?$|mrn|path)", re.I)
LONG_INT = re.compile(r"(?<![\d.])\d{8,}(?![\d.])")
COUNT_KEYS = re.compile(r"^(n|count|patients|stays|studies|records_.*|positives|.*_positives|.*_cases|.*_count|"
                        r"final_stays|first_icu_stays|adults|in_icu_at_t0|alive_at_t0|excluded_.*|unreadable.*)$", re.I)


def walk_keys(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k
            yield from walk_keys(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk_keys(v)


def suppress(o, key=None):
    if isinstance(o, dict):
        return {k: suppress(v, k) for k, v in o.items()}
    if isinstance(o, list):
        return [suppress(v, key) for v in o]
    if (key is not None and COUNT_KEYS.match(str(key)) and isinstance(o, int) and not isinstance(o, bool)
            and 0 < o < C.SMALL_CELL):
        return "<10"
    return o


def check_json(path):
    text = path.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return False, f"not valid JSON ({e})", None
    bad = sorted({k for k in walk_keys(obj) if ID_KEYS.search(str(k))})
    if bad:
        return False, f"identifier-like keys: {', '.join(bad[:5])}", None
    if LONG_INT.search(text):
        return False, "contains an 8+ digit integer (possible MIMIC ID)", None
    return True, "", suppress(obj)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    C.SHARE_DIR.mkdir(parents=True, exist_ok=True)
    n, refused = 0, 0
    for task in C.TASKS + ("figures",):
        d = C.OUTPUT_DIR / task
        if not d.exists():
            continue
        for pat in SHAREABLE:
            for f in sorted(p for p in d.glob(pat) if not p.stem.endswith("_smoke")):   # smoke runs are pipeline checks
                dest = C.SHARE_DIR / (f.name if task == "figures" else f"{task}_{f.name}")
                if f.suffix == ".json":
                    ok, why, clean = check_json(f)
                    if not ok:
                        print(f"  REFUSED {task}/{f.name}: {why}")
                        refused += 1
                        continue
                    if not args.list:
                        dest.write_text(json.dumps(clean, indent=2), encoding="utf-8")
                elif not args.list:
                    shutil.copy2(f, dest)
                print(f"  {'would copy' if args.list else 'copied'} {task}/{f.name} -> {dest.name}")
                n += 1
    print(f"{n} file(s) {'to share' if args.list else 'shared'}; {refused} refused")
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
