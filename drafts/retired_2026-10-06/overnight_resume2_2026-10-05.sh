#!/usr/bin/env bash
# Full update, night of 2026-10-05 (Dani): every notebook on today's fixed code, into a fresh
# Figures_<fig_date> folder (TulePond_filepaths fig_date). Order:
#   1. 01-06 and 09-11 (minutes; the forecast and Figs 1/3 are ready early)
#   2. the model chain, drafts/pylith/models_overnight_2026-10-05.sh (~6 h: 07a, point sets, 22 inversions at
#      sigma 12, PyLith tests, 08a, 08b, 07c, then 07b)
#   3. 12 (collects the paper figures, refuses stale ones)
#   4. a registry diff of the whole night: outputs/registry_diff_2026-10-05_full.txt
# A notebook that raises stops the night; pytest anchor failures at the end of run_all.sh do not (the anchors
# are expected to move: Dani updates the text from the diff, then the anchors).
#   caffeinate -ims bash drafts/overnight_2026-10-05.sh      (laptop plugged in, lid open)
set -Eeuo pipefail
REPO=/Users/dlindsay/tools/internal/TulePond_DISP
DATA=/Users/dlindsay/work/proj_TulePond_DISP
E=/Users/dlindsay/miniforge3/envs/opera_disp_feb13
export PATH=$E/bin:$PATH PROJ_LIB=$E/share/proj PROJ_DATA=$E/share/proj MPLBACKEND=Agg
export TULEPOND_SUPERSEDE=1
LOG=$REPO/drafts/overnight_2026-10-05.log
step() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
trap 'step "FAILED (line $LINENO)"' ERR

# run_all.sh exits non-zero both when a notebook raises and when pytest flags anchors: tell them apart
notebooks() {
    local nblog=$1; shift
    if ! ( cd "$REPO" && ./run_all.sh "$@" ) > "$nblog" 2>&1; then
        if grep -q "CellExecutionError\|Traceback" "$nblog"; then
            step "FAILED: a notebook raised (see $nblog)"; exit 1
        fi
        step "  ran; pytest flagged anchors (expected tonight)"
    fi
}

step "START full update; fig_date = $(cd "$REPO" && python -c 'import TulePond_filepaths as f; print(f.fig_date)')"
[ -e "$DATA/outputs/registry_before_full_2026-10-05.csv" ] || cp -p "$DATA/outputs/registry.csv" "$DATA/outputs/registry_before_full_2026-10-05.csv"

step "1. notebooks 01-06, 09-11"
step "  (part 1 done 20:43; resume 2 at 08a after the range-tolerance fix)"

step "2. model chain (log: drafts/pylith/runs/models_overnight_2026-10-05.log)"
bash "$REPO/drafts/pylith/models_overnight_resume2_2026-10-05.sh"

step "3. notebook 12"
notebooks "$REPO/drafts/overnight_2026-10-05_part3.log" 12

step "4. registry diff of the night"
python - <<'EOF'
import pandas as pd
out = "/Users/dlindsay/work/proj_TulePond_DISP/outputs/"
before = pd.read_csv(out + "registry_before_full_2026-10-05.csv")
after = pd.read_csv(out + "registry.csv")
key = before.columns[0]
val = "rendered"                                            # the number as the text would quote it
b = before.groupby(key).last()[val]
a = after.iloc[len(before):].groupby(key).last()[val]      # rows appended tonight only
lines = []
for k in sorted(set(a.index) | set(b.index)):
    if k in a.index and k in b.index:
        if str(a[k]) != str(b[k]):
            lines.append(f"CHANGED  {k}: {b[k]} -> {a[k]}")
    elif k in a.index:
        lines.append(f"NEW      {k}: {a[k]}")
    else:
        lines.append(f"NOT RE-REGISTERED  {k}: {b[k]}")
open(out + "registry_diff_2026-10-05_full.txt", "w").write("\n".join(lines) + "\n")
print(f"{sum(l.startswith('CHANGED') for l in lines)} changed, {sum(l.startswith('NEW') for l in lines)} new, "
      f"{sum(l.startswith('NOT') for l in lines)} not re-registered")
EOF
step "DONE"
