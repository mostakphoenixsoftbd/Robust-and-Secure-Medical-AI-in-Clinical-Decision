"""One-off: rebuild training histories in <task>/training_log.json from the run logs.

Needed once (30 Oct): earlier versions of run_task.py rewrote training_log.json on every run, so cached
models lost their histories. Epoch lines are aggregate only ("[task/mode] epoch k: loss L val AUROC (s)").
Models train in a fixed order (train_all), so the n-th epoch sequence in a log is the n-th model trained.

    python recover_training_log.py ehr ehr_smoke.log D0_seed0 D0_seed1 D0_seed2 surrogate D1_pgd_at D2_trades D3_noise
    python recover_training_log.py ecg ecg_train.log D0_seed0 D0_seed1
"""
from __future__ import annotations

import json
import re
import sys

import common as U
import config as C

EPOCH = re.compile(r"\[(\w+)/(\w+)\] epoch (\d+): loss ([\d.]+) val ([\d.]+) \((\d+)s\)")


def sequences(path):
    seqs = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = EPOCH.search(line)
        if not m:
            continue
        task, mode, ep, loss, val, sec = m.groups()
        if int(ep) == 1:
            seqs.append({"mode": mode, "history": [], "seconds": 0})
        seqs[-1]["history"].append({"epoch": int(ep), "train_loss": float(loss), "val_score": float(val)})
        seqs[-1]["seconds"] += int(sec)
    return seqs


def main(argv):
    task, logname, names = argv[0], argv[1], argv[2:]
    seqs = sequences(C.OUTPUT_DIR / "logs" / logname)
    if len(seqs) < len(names):
        U.fail(f"{logname}: {len(seqs)} epoch sequences, {len(names)} names given")
    path = U.task_dir(task) / "training_log.json"
    log = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    tr = log.setdefault("training", {})
    for name, s in zip(names, seqs):
        if name in tr and not tr[name].get("recovered_from_log"):
            continue                       # never overwrite a history written by run_task.py
        tr[name] = {"history": s["history"], "seconds": s["seconds"], "mode": s["mode"],
                    "recovered_from_log": logname}
        print(f"{task} {name}: {s['mode']}, {len(s['history'])} epochs")
    U.save_json(log, path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
