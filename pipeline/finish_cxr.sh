#!/usr/bin/env bash
# Step 6 after the CXR full run: share aggregates, tables, figures, number check, manuscript build.
#
#   bash pipeline/finish_cxr.sh            # waits for cxr_full.log "=== finished OK", then runs everything
#   bash pipeline/finish_cxr.sh --now      # skip the wait (only if the CXR run is known to be finished)
#
# Log: /e/physionet/robust_outputs/logs/finish_cxr.log (aggregate output only; no row-level data).
# Exits 2 if the CXR run gave up. check_numbers.py is expected to FAIL until the CXR [[R:]] placeholders are
# filled (it counts open placeholders); the build still runs so the draft can be inspected.
set -u
cd "$(dirname "$0")"                                   # pipeline/ (scripts import config from here)
PY=.venv/Scripts/python.exe
RUNLOG=/e/physionet/robust_outputs/logs/cxr_full.log
LOG=/e/physionet/robust_outputs/logs/finish_cxr.log
RES=/e/physionet/robust_outputs/cxr/results_cxr.json
SHARE=../pipeline_share
FIGS=../manuscript/figures

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# status of the current run_retry.sh invocation only: lines after the last "=== attempt 1 " marker
run_state() {
    local start
    start=$(grep -n '^=== attempt 1 ' "$RUNLOG" | tail -1 | cut -d: -f1)
    [[ -z "$start" ]] && start=1
    if tail -n +"$start" "$RUNLOG" | grep -q '^=== finished OK'; then echo ok
    elif tail -n +"$start" "$RUNLOG" | grep -q '^=== giving up'; then echo gaveup
    else echo running; fi
}

say "finish_cxr.sh started"
if [[ "${1:-}" != "--now" ]]; then
    while :; do
        s=$(run_state)
        [[ "$s" == ok ]] && break
        if [[ "$s" == gaveup ]]; then say "CXR run gave up; nothing done"; exit 2; fi
        sleep 60
    done
    say "CXR run finished OK"
fi

# the results file must be the completed one (run_task.py writes "finished" last)
if ! "$PY" -c "import json,sys; r=json.load(open(sys.argv[1],encoding='utf-8')); sys.exit(0 if 'finished' in r and 'certification' in r else 1)" "$RES"; then
    say "ERROR: $RES has no 'finished'/'certification' key (partial run?); stopping"; exit 3
fi

step() {   # step <name> <command...>; logs output, continues on failure
    local name=$1; shift
    say "--- $name"
    if "$@" >> "$LOG" 2>&1; then say "    $name OK"; else say "    $name exited $? (see log)"; return 1; fi
}

step share_results "$PY" -u share_results.py || { say "share_results failed (refused file?); stopping"; exit 4; }
step make_tables   "$PY" -u make_tables.py   || exit 5
step make_figures  "$PY" -u make_figures.py  || exit 6
step share_figures "$PY" -u share_results.py || exit 4     # copies robust_outputs/figures/*.png|svg to pipeline_share/

# figures: share_results puts them at pipeline_share/fig_<name>.*; build.py looks for manuscript/figures/figN_*.png
say "--- copy figures"
mkdir -p "$FIGS"
for pair in 2:robustness_curves 3:decision_flips 4:tradeoff 5:certified; do
    n=${pair%%:*}; name=${pair#*:}
    for ext in png svg; do
        if [[ -f "$SHARE/fig_${name}.${ext}" ]]; then
            cp -f "$SHARE/fig_${name}.${ext}" "$FIGS/fig${n}_${name}.${ext}"
            say "    fig_${name}.${ext} -> manuscript/figures/fig${n}_${name}.${ext}"
        else
            say "    MISSING $SHARE/fig_${name}.${ext}"
        fi
    done
done

step cxr_fill_values "$PY" -u cxr_fill_values.py "$SHARE/cxr_results_cxr.json"
# shellcheck disable=SC2086
step check_numbers "$PY" -u check_numbers.py ../manuscript/0[1-7]*.md || say "    (expected until CXR [[R:]] placeholders are filled)"
step build "$PY" -u ../manuscript/build/build.py || true
say "finish_cxr.sh done; next: fill CXR [[R:]] from the cxr_fill_values block above, rerun check_numbers and build.py"
