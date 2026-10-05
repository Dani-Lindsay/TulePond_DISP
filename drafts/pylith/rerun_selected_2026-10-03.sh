#!/usr/bin/env bash
# Minimum model rerun after the 04/06 onset-window fix (2026-10-03): the mean-event field that the models fit
# moved (~10 % in the FP step), so the models the paper quotes are re-inverted. The PyLith Green's functions do
# not depend on the data and are reused. NOT rerun (Dani 2026-10-03: not all the modelling): the 22-run
# sensitivity suite except the runs 08a/08b read for the paper (default + nuggetcd on both media, e50/w50
# alignment pairs); the other sensitivity rows in 08b's table therefore still fit the pre-fix field.
# Order: backups -> 07a -> 6 inversions -> PyLith tests -> copy tables -> 08a + 08b -> 07c -> 07b (3 h, last).
set -euo pipefail
REPO=/Users/dlindsay/tools/internal/TulePond_DISP
D=$REPO/drafts/pylith
DATA=/Users/dlindsay/work/proj_TulePond_DISP
M=$DATA/modeling_SFCVM
E=/Users/dlindsay/miniforge3/envs/opera_disp_feb13
export PATH=$E/bin:$PATH PROJ_LIB=$E/share/proj PROJ_DATA=$E/share/proj MPLBACKEND=Agg
export TULEPOND_SUPERSEDE=1
LOG=$D/runs/rerun_selected_2026-10-03.log
step() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
trap 'step "FAILED (line $LINENO) -- see the step log named above"' ERR

# 0. backups (the small tables only; the G matrices are not touched by this rerun)
for g in "$D/gf" "$M/gf"; do
    mkdir -p "$g/pre_2026-10-03"
    find "$g" -maxdepth 1 -type f ! -name 'G_*' -exec cp -p {} "$g/pre_2026-10-03/" \;
done
cp -p "$DATA/outputs/registry.csv" "$DATA/outputs/registry_before_model_rerun_2026-10-03.csv"
step "0 backups: gf/pre_2026-10-03 (draft + data), registry_before_model_rerun_2026-10-03.csv"

notebooks() {   # <log tag> <notebook numbers...>: a notebook that raises stops the script; flagged anchors do not
    local tag=$1; shift
    step "notebooks $* (log: $D/runs/$tag.log)"
    if ! ( cd "$REPO" && caffeinate -ims ./run_all.sh "$@" ) > "$D/runs/$tag.log" 2>&1; then
        if grep -q "CellExecutionError\|Traceback" "$D/runs/$tag.log"; then
            step "FAILED: a notebook raised (log: $D/runs/$tag.log)"; exit 1
        fi
        step "  ran; pytest flagged anchors (expected after the fix)"
    fi
}
run() {   # <log tag> <KEY=VALUE ...>
    local tag=$1; shift
    step "inversion $tag ($*)"
    bash "$D/run_notebook.sh" "$tag" "$@"
    tail -1 "$D/runs/notebook_$tag.log" | grep -q "^exit 0$" || { step "FAILED $tag (log: $D/runs/notebook_$tag.log)"; exit 1; }
}

# 1. the uniform model (its cells [1]-[6] are also the data side of every PyLith inversion)
notebooks r1003_07a 07a

# 2. the inversions the paper reads
BOTH=PYLITH_MEDIA=homog,sfcvm
run r1003_default          $BOTH
run r1003_nuggetcd         $BOTH PYLITH_CD=nugget
run r1003_align_e50_full   PYLITH_MEDIA=sfcvm_e50
run r1003_align_e50_nugget PYLITH_MEDIA=sfcvm_e50 PYLITH_CD=nugget
run r1003_align_w50_full   PYLITH_MEDIA=sfcvm_w50
run r1003_align_w50_nugget PYLITH_MEDIA=sfcvm_w50 PYLITH_CD=nugget

# 3. bootstrap / recovery / depth scan of the selected runs
step "tests: pylith_sfcvm_tests.py (log: $D/runs/tests.log)"
bash "$D/run_tests.sh"
tail -1 "$D/runs/tests.log" | grep -q "^exit 0$" || { step "FAILED tests (log: $D/runs/tests.log)"; exit 1; }

# 4. tables to the data side; stale forward runs out of the way so 08a re-makes them from the new slip
step "copy draft tables to $M/gf; move stale fwd_fig2_/fwd_traction_ runs to pylith_output/pre_2026-10-03"
rsync -a --exclude 'G_*' --exclude 'pre_*' "$D/gf/" "$M/gf/"
mkdir -p "$M/pylith_output/pre_2026-10-03"
find "$M/pylith_output" -maxdepth 1 -type f \( -name 'fwd_fig2_*' -o -name 'fwd_traction_*' \) -exec mv {} "$M/pylith_output/pre_2026-10-03/" \;

# 5. the paper's model notebooks and Fig 2, then the long uniform-model tests
export TULEPOND_RUN_PYLITH=1
notebooks r1003_08ab 08a 08b
notebooks r1003_07c 07c
notebooks r1003_07b 07b
step "DONE"
