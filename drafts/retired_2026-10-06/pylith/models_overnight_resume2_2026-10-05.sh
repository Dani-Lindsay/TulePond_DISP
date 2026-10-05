#!/bin/bash
# Overnight MODEL rerun, 2026-10-05 (Dani): the model half of tonight's full update. Run it only AFTER notebooks
# 03-06 have been re-run on today's code (06 now corrects both tracks' real burst seams, so fig2_inputs.npz,
# fp_maps.npz and csi_mean_event_*.npz are new); the script refuses to start on an older 06 field. Then 09-12
# (12 collects Fig 2 from 07c) run after it.
#
# What changed in the code it runs (author-approved 2026-10-05):
#   - ONE surface % denominator everywhere: the median of the measured crossings in the SSE section
#     (scripts/pylith_helpers/surface_crossings.surface_pct_denominator_mm), used by 07a, 07b, 08a, 08b and the
#     driver; 08a/08b no longer re-express cached numbers (surface_pct_cached_scale is gone).
#   - every driver inversion (default, nuggetcd, e50/w50 pairs and the 16 sensitivity runs) is solved at 08a's
#     locked sigma (PYLITH_SIGMA=locked -> sigma_locked of scripts/08a_3d_model.py, 12); each L-curve is still
#     computed and saved, its own pick recorded as sigma_lcurve_pick. Each results table carries the fingerprint
#     of the 06 field it fits; 08a/08b/final_comparison.py refuse stale or mixed tables.
#   - the swath-7 and coherence-0.7 sample sets carry their data in their points npz: re-made here on the current
#     field (step 2), positions asserted unchanged (the cached PyLith Green's functions are computed at them).
#   - 07a no longer rescales by the orphan SFCVM profile csv (07a_hom_mw_sfcvm / _mu_effective_gpa retired);
#     Fig 2 K quotes the harmonic SFSVM Mw (08a [15c] row 'harmonic', registered 08a_mw_sfcvm_harmonic*).
#
#   bash drafts/pylith/models_overnight_2026-10-05.sh      (any directory; laptop plugged in; ~6 h, estimates below)
#   log: drafts/pylith/runs/models_overnight_2026-10-05.log (one line per step with its time; each step has its own log)
#
# Estimated runtimes (from the 2026-10-02/03 logs): 07a 37 min; point sets ~5 min; 22 inversions 95 min
# (~4.3 min each, shallow 12 min); PyLith tests 19 min; tables < 1 min; 08a 13 min (incl. the PyLith forward
# runs it re-makes); 08b 20 min; 07c 1 min; 07b 170 min. Total ~6 h.
#
# Failure behaviour: stops at the first error (set -e + ERR trap). A notebook that raises stops the run; the
# pytest anchor failures at the end of run_all.sh do NOT (the anchors are expected to move; listed in step 9).
# Backups (made once: a relaunch never overwrites them): drafts/pylith/gf/pre_2026-10-05/, modeling_SFCVM/gf/pre_2026-10-05/,
# drafts/pylith/points/pre_2026-10-05/, drafts/pylith/coh07/pre_2026-10-05/, outputs/registry_before_models_2026-10-05.csv;
# stale PyLith forward runs move to modeling_SFCVM/pylith_output/pre_2026-10-05/. Registry values change on
# purpose: TULEPOND_SUPERSEDE=1. Step 9 writes outputs/registry_diff_models_2026-10-05.txt.
set -Eeuo pipefail                       # -E: the ERR trap also fires inside the functions below
REPO=/Users/dlindsay/tools/internal/TulePond_DISP
D=$REPO/drafts/pylith
DATA=/Users/dlindsay/work/proj_TulePond_DISP
OUT=$DATA/outputs
M=$DATA/modeling_SFCVM
E=/Users/dlindsay/miniforge3/envs/opera_disp_feb13
TAG=2026-10-05
export PATH=$E/bin:$PATH PROJ_LIB=$E/share/proj PROJ_DATA=$E/share/proj MPLBACKEND=Agg PYTHONPATH=$REPO
export TULEPOND_SUPERSEDE=1
LOG=$D/runs/models_overnight_$TAG.log
T0=$(date +%s)
T_STEP=$T0
step() {   # one log line per step: time, minutes since the previous step and since the start
    local now; now=$(date +%s)
    echo "$(date '+%F %T') [+$(( (now - T_STEP) / 60 )) min, total $(( (now - T0) / 60 )) min] $*" | tee -a "$LOG"
    T_STEP=$now
}
trap 'step "FAILED (line $LINENO) -- see the step log named above"' ERR
caffeinate -imsw $$ &                    # keep the laptop awake for the whole script (each step also runs under caffeinate)

notebooks() {   # <log tag> <notebook prefixes...>: a notebook that raises stops the script; flagged anchors do not
    local tag=$1; shift
    step "notebooks $* (log: $D/runs/$tag.log)"
    if ! ( cd "$REPO" && caffeinate -ims ./run_all.sh "$@" ) > "$D/runs/$tag.log" 2>&1; then
        if grep -q "CellExecutionError\|Traceback" "$D/runs/$tag.log"; then
            step "FAILED: a notebook raised (log: $D/runs/$tag.log)"; exit 1
        fi
        step "  ran; pytest flagged anchors (expected: listed with the registry diff)"
    fi
}
run() {   # <log tag> <KEY=VALUE ...>: one driver inversion (cached Green's functions)
    local tag=$1; shift
    step "inversion $tag ($*)"
    bash "$D/run_notebook.sh" "$tag" "$@"
    tail -1 "$D/runs/notebook_$tag.log" | grep -q "^exit 0$" || { step "FAILED $tag (log: $D/runs/notebook_$tag.log)"; exit 1; }
}
py() {   # <log tag> <python args...>: a helper script, run from the repo root
    local tag=$1; shift
    step "python $* (log: $D/runs/$tag.log)"
    ( cd "$REPO" && caffeinate -ims python "$@" ) > "$D/runs/$tag.log" 2>&1 || { step "FAILED (log: $D/runs/$tag.log)"; exit 1; }
}

# 0. preconditions and backups
F06=$OUT/06_event_field_geometry
for f in "$F06/fp_maps.npz" "$F06/fig2_inputs.npz"; do
    for code in "$REPO"/scripts/0[3-6]_*.py "$REPO/insar_utils.py" "$REPO/TulePond_filepaths.py"; do
        [ "$f" -nt "$code" ] || { echo "$f is older than $code: run notebooks 03-06 first" | tee -a "$LOG"; exit 1; }
    done
done
step "0/9 06 field is newer than the 03-06 code; backups"
for g in "$D/gf" "$M/gf"; do
    if [ -e "$g/pre_$TAG" ]; then
        echo "  backup $g/pre_$TAG exists (relaunch): kept as it is" | tee -a "$LOG"
    else
        mkdir -p "$g/pre_$TAG"
        find "$g" -maxdepth 1 -type f ! -name 'G_*' -exec cp -p {} "$g/pre_$TAG/" \;
    fi
done
if [ ! -e "$D/points/pre_$TAG" ]; then
    mkdir -p "$D/points/pre_$TAG"
    cp -p "$D"/points/insar_points_swath7.* "$D"/points/insar_points_coh07.* "$D/points/pre_$TAG/"
fi
if [ ! -e "$D/coh07/pre_$TAG" ]; then
    mkdir -p "$D/coh07/pre_$TAG"
    cp -Rp "$D/coh07/coh07" "$D/coh07/coh09" "$D/coh07/pre_$TAG/"
fi
[ -e "$OUT/registry_before_models_$TAG.csv" ] || cp -p "$OUT/registry.csv" "$OUT/registry_before_models_$TAG.csv"

# 1. the uniform model (its cells [1]-[6] are also the data side of every PyLith inversion; writes the 07a
#    fig2_model_row.npz that step 2's swath-4 gate and the driver read)
step "7/9 08a + 08b + 07c"
export TULEPOND_RUN_PYLITH=1
notebooks r1005_08ab 08a 08b
notebooks r1005_07c 07c
unset TULEPOND_RUN_PYLITH

# 8. the long uniform-model tests, last
step "8/9 07b SKIPPED (Dani 2026-10-05 22:15: PyLith only; 07b out of the nightly chain)"
: # 07b retired from the nightly chain

# 9. what changed in the registry, and the anchors
step "9/9 registry diff -> $OUT/registry_diff_models_$TAG.txt"
python - <<EOF
import pandas as pd
D = "$OUT"
def latest(f):
    r = pd.read_csv(f)
    return r.groupby("key").tail(1).set_index("key")
old, new = latest(f"{D}/registry_before_models_$TAG.csv"), latest(f"{D}/registry.csv")
lines = ["Registry changes, model rerun $TAG (one surface % denominator; every driver run at the locked sigma on the "
         "current 06 field; harmonic SFSVM Mw; 07a SFCVM rescale retired)", ""]
for k in sorted(set(old.index) | set(new.index)):
    a = old["value"].get(k) if k in old.index else "(new)"
    b = new["value"].get(k) if k in new.index else "(gone)"
    if str(a) != str(b):
        lines.append(f"{k:45s} {str(a):>14s} -> {str(b):<14s}")
lines.append("NB: keys a notebook no longer registers (07a_hom_mw_sfcvm, 07a_hom_mu_effective_gpa) stay in the csv; "
             "registry.txt lists them as RETIRED.")
open(f"{D}/registry_diff_models_$TAG.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines[:80]))
EOF
( cd "$REPO" && python -m pytest -q tests/test_anchors.py 2>&1 | grep -E "AssertionError: |passed|failed" || true ) \
    | tee -a "$OUT/registry_diff_models_$TAG.txt"
step "DONE"
