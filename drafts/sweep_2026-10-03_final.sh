#!/usr/bin/env bash
# Clean sweep (Dani 2026-10-03): every notebook in order, from the prepared inputs to the paper figures,
# into Figures_2026-10-03_final (TulePond_filepaths fig_date). Nothing is downloaded; the PyLith Green's
# functions are reused and the six inversions the paper reads were re-made this morning
# (drafts/pylith/rerun_selected_2026-10-03.sh, stopped after its tests step: its copy step is done here).
# 07b (the 3 h uniform-model side checks) runs last so Figs 1-3 are not held up.
set -euo pipefail
REPO=/Users/dlindsay/tools/internal/TulePond_DISP
D=$REPO/drafts/pylith
DATA=/Users/dlindsay/work/proj_TulePond_DISP
M=$DATA/modeling_SFCVM
E=/Users/dlindsay/miniforge3/envs/opera_disp_feb13
export PATH=$E/bin:$PATH PROJ_LIB=$E/share/proj PROJ_DATA=$E/share/proj MPLBACKEND=Agg
export TULEPOND_SUPERSEDE=1 TULEPOND_RUN_PYLITH=1
LOG=$REPO/drafts/sweep_2026-10-03_final.log
step() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
trap 'step "FAILED (line $LINENO)"' ERR

# 1. the PyLith tests of this morning's rerun must have finished cleanly
step "waiting for the PyLith tests (drafts/pylith/runs/tests.log)"
while pgrep -f "pylith_sfcvm_tests.py" > /dev/null; do sleep 20; done
sleep 5                                   # run_tests.sh writes its exit line just after
tail -1 "$D/runs/tests.log" | grep -q "^exit 0$" || { step "FAILED: PyLith tests did not end with exit 0"; exit 1; }

# 2. the rerun's copy step: tables to the data side, stale forward runs out of the way (08a re-makes them)
step "copy draft tables to $M/gf; move fwd_fig2_/fwd_traction_ runs to pylith_output/pre_2026-10-03"
rsync -a --exclude 'G_*' --exclude 'pre_*' "$D/gf/" "$M/gf/"
mkdir -p "$M/pylith_output/pre_2026-10-03"
find "$M/pylith_output" -maxdepth 1 -type f \( -name 'fwd_fig2_*' -o -name 'fwd_traction_*' \) -exec mv {} "$M/pylith_output/pre_2026-10-03/" \;
cp -p "$DATA/outputs/registry.csv" "$DATA/outputs/registry_before_sweep_2026-10-03_final.csv"

# 3. every notebook, in order (a notebook that raises stops the sweep; flagged anchors do not)
step "notebooks (log: $REPO/drafts/sweep_2026-10-03_final_notebooks.log)"
if ! ( cd "$REPO" && caffeinate -ims ./run_all.sh 01_prep_inputs 01_prep_figures 02 03 04 05 06 07a 08a 08b 07c 09 10 11 12 07b ) \
        > "$REPO/drafts/sweep_2026-10-03_final_notebooks.log" 2>&1; then
    if grep -q "CellExecutionError\|Traceback" "$REPO/drafts/sweep_2026-10-03_final_notebooks.log"; then
        step "FAILED: a notebook raised (see the notebooks log)"; exit 1
    fi
    step "notebooks ran; pytest flagged anchors (expected: numbers moved today)"
fi
step "DONE"
