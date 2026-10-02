#!/bin/bash
# Overnight rerun, 2026-10-02 (Dani): the 3D model's surface constraint is now the measured crossings, each tied
# to the slip at its own position (scripts/pylith_helpers/surface_crossings.py), and 07a's running-median window
# keeps its edge crossings (float roundoff had dropped half the windows). Everything downstream of the surface
# constraint is re-made from the same code, in order, and the run stops at the first failure.
#
#   bash drafts/pylith/overnight_2026-10-02.sh          (any directory; ~4-6 h; keeps the laptop awake)
#   log: drafts/pylith/runs/overnight_2026-10-02.log    (one line per step; each step has its own log too)
#
# Before this script ran, the old products were copied (nothing deleted): drafts/pylith/gf/pre_2026-10-02/,
# modeling_SFCVM/gf/pre_2026-10-02/, drafts/pylith/runs/pre_2026-10-02/, outputs/registry_pre_2026-10-02.{csv,txt}.
# The stale PyLith forward runs are moved (step 6) into modeling_SFCVM/pylith_output/pre_2026-10-02/.
# Registry values change on purpose: TULEPOND_SUPERSEDE=1. The last step writes outputs/registry_diff_2026-10-02.txt.
set -euo pipefail
REPO=/Users/dlindsay/tools/internal/TulePond_DISP
D=$REPO/drafts/pylith
DATA=/Users/dlindsay/work/proj_TulePond_DISP
M=$DATA/modeling_SFCVM
E=/Users/dlindsay/miniforge3/envs/opera_disp_feb13
export PATH=$E/bin:$PATH PROJ_LIB=$E/share/proj PROJ_DATA=$E/share/proj MPLBACKEND=Agg
export TULEPOND_SUPERSEDE=1
LOG=$D/runs/overnight_2026-10-02.log
step() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
trap 'step "FAILED (line $LINENO) -- see the step log named above"' ERR
for b in "$D/gf/pre_2026-10-02" "$M/gf/pre_2026-10-02" "$DATA/outputs/registry_pre_2026-10-02.csv"; do
    [ -e "$b" ] || { echo "missing backup $b: stop" | tee -a "$LOG"; exit 1; }
done

# 1. the CSI baseline (07a) and its tests (07b): 08a and the draft runs read 07b's fig2_model_row.npz
step "1/8 07a + 07b (log: $D/runs/r1002_07ab.log)"
# run_all.sh ends with pytest; a non-zero exit is a failure only if a notebook itself raised
if ! ( cd "$REPO" && caffeinate -ims ./run_all.sh 07a 07b ) > "$D/runs/r1002_07ab.log" 2>&1; then
    if grep -q "CellExecutionError\|Traceback" "$D/runs/r1002_07ab.log"; then
        step "FAILED: a notebook raised (log: $D/runs/r1002_07ab.log)"; exit 1
    fi
    step "07a/07b ran; pytest flagged something (see the log)"
fi

# 2. every PyLith inversion of the sensitivity table (cached Green's functions; ~5-10 min each)
run() {   # <log tag> <KEY=VALUE ...>
    local tag=$1; shift
    step "2/8 inversion $tag ($*)"
    bash "$D/run_notebook.sh" "$tag" "$@"
    tail -1 "$D/runs/notebook_$tag.log" | grep -q "^exit 0$" || { step "FAILED $tag (log: $D/runs/notebook_$tag.log)"; exit 1; }
}
BOTH=PYLITH_MEDIA=homog,sfcvm
run r1002_default          $BOTH
run r1002_nuggetcd         $BOTH PYLITH_CD=nugget
run r1002_diagcd           $BOTH PYLITH_CD=diag
run r1002_floor15          $BOTH PYLITH_CD=floor PYLITH_CD_FLOOR=15
run r1002_floor4           $BOTH PYLITH_CD=floor PYLITH_CD_FLOOR=4
run r1002_nosurf           $BOTH PYLITH_SURFACE=0
run r1002_nosurf_diagcd    $BOTH PYLITH_SURFACE=0 PYLITH_CD=diag
run r1002_shallow          $BOTH PYLITH_MESH_PRESET=shallow
run r1002_dip80e_full      $BOTH PYLITH_DIP=80e
run r1002_dip80e_nugget    $BOTH PYLITH_DIP=80e PYLITH_CD=nugget
run r1002_dip70e_full      $BOTH PYLITH_DIP=70e
run r1002_dip70e_nugget    $BOTH PYLITH_DIP=70e PYLITH_CD=nugget
run r1002_swath7_full      $BOTH PYLITH_POINTS=swath7
run r1002_swath7_nugget    $BOTH PYLITH_POINTS=swath7 PYLITH_CD=nugget
run r1002_order2_full      $BOTH PYLITH_ORDER=2
run r1002_order2_nugget    $BOTH PYLITH_ORDER=2 PYLITH_CD=nugget
run r1002_coh07_full       $BOTH PYLITH_POINTS=coh07
run r1002_coh07_nugget     $BOTH PYLITH_POINTS=coh07 PYLITH_CD=nugget
run r1002_align_e50_full   PYLITH_MEDIA=sfcvm_e50
run r1002_align_e50_nugget PYLITH_MEDIA=sfcvm_e50 PYLITH_CD=nugget
run r1002_align_w50_full   PYLITH_MEDIA=sfcvm_w50
run r1002_align_w50_nugget PYLITH_MEDIA=sfcvm_w50 PYLITH_CD=nugget

# 3. bootstrap, recovery and depth-scan tests of the four main models
step "3/8 pylith_sfcvm_tests.py (log: $D/runs/tests.log)"
bash "$D/run_tests.sh"
tail -1 "$D/runs/tests.log" | grep -q "^exit 0$" || { step "FAILED tests (log: $D/runs/tests.log)"; exit 1; }

# 4. the sensitivity table (every run in one csv)
step "4/8 final_comparison.py"
( cd "$D" && python final_comparison.py ) > "$D/runs/r1002_final_comparison.log" 2>&1

# 5. the draft products -> modeling_SFCVM/gf, where 08a/08b read them (G matrices unchanged, not copied)
step "5/8 copy draft tables to $M/gf"
rsync -a --exclude 'G_*' --exclude 'pre_*' "$D/gf/" "$M/gf/"

# 6. the old forward runs were made from the old slip: move them aside so 08a re-runs them (its slip check would fail)
step "6/8 move stale forward runs to $M/pylith_output/pre_2026-10-02"
mkdir -p "$M/pylith_output/pre_2026-10-02"
find "$M/pylith_output" -maxdepth 1 -type f \( -name 'fwd_fig2_*' -o -name 'fwd_traction_*' \) -exec mv {} "$M/pylith_output/pre_2026-10-02/" \;

# 7. the paper's model notebooks, Fig 2, with PyLith allowed to make the missing forward runs
step "7/8 08a + 08b + 07c (log: $D/runs/r1002_08ab_07c.log)"
# the anchor tests at the end of run_all.sh SHOULD flag the moved model numbers (the point of the rerun), so a
# non-zero exit is a failure only if a notebook itself raised; the flagged anchors are listed in step 8
if ! ( cd "$REPO" && TULEPOND_RUN_PYLITH=1 caffeinate -ims ./run_all.sh 08a 08b 07c ) > "$D/runs/r1002_08ab_07c.log" 2>&1; then
    if grep -q "CellExecutionError\|Traceback" "$D/runs/r1002_08ab_07c.log"; then
        step "FAILED: a notebook raised (log: $D/runs/r1002_08ab_07c.log)"; exit 1
    fi
    step "notebooks ran; pytest flagged anchors (expected: listed with the registry diff)"
fi

# 8. what changed in the registry
step "8/8 registry diff -> $DATA/outputs/registry_diff_2026-10-02.txt"
python - <<'EOF'
import pandas as pd
D = "/Users/dlindsay/work/proj_TulePond_DISP/outputs"
def latest(f):
    r = pd.read_csv(f)
    return r.groupby("key").tail(1).set_index("key")
old, new = latest(f"{D}/registry_pre_2026-10-02.csv"), latest(f"{D}/registry.csv")
lines = ["Registry changes, overnight rerun 2026-10-02 (surface constraint = measured crossings; 07a window fix)", ""]
for k in sorted(set(old.index) | set(new.index)):
    a = old["value"].get(k) if k in old.index else "(new)"
    b = new["value"].get(k) if k in new.index else "(gone)"
    if str(a) != str(b):
        lines.append(f"{k:45s} {str(a):>14s} -> {str(b):<14s}")
open(f"{D}/registry_diff_2026-10-02.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines[:80]))
EOF
( cd "$REPO" && python -m pytest -q tests/test_anchors.py 2>&1 | grep -E "AssertionError: |passed|failed" || true ) | tee -a "$DATA/outputs/registry_diff_2026-10-02.txt"
step "DONE"
