#!/usr/bin/env bash
# Run every notebook top to bottom, in order, from a fresh kernel, and stop at
# the first failure. This is the "restart and run all" that every number in
# the manuscript must come from. Pass notebook numbers to run a subset:
#     ./run_all.sh            # all
#     ./run_all.sh 03 05      # just those
#     ./run_all.sh 01_prep_inputs 07a 08a   # a full name or a prefix, run in the order given
#     ./run_all.sh 07a 08a 07c 08b 12      # the slip models and Fig 2, given their upstream products (04, 06)
#
# The sensitivity scripts (scripts/sensitivity/*.py, ~5 h, their csvs archived) are NOT run here: 08b_sensitivity draws
# their archived outputs. Run them by hand when a locked setting changes (how: scripts/sensitivity/common.py docstring).
#
# Uses the python next to this script's env (opera_disp_feb13); call it with
# the env's full path if `jupyter` is not on PATH:
#     PATH=/Users/dlindsay/miniforge3/envs/opera_disp_feb13/bin:$PATH ./run_all.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ $# -gt 0 ]; then
    nbs=""
    for n in "$@"; do nbs="$nbs $(ls ${n}*.ipynb)"; done
else
    # Name order, with two exceptions (the products each notebook reads must exist before it runs):
    #  - 07c_fig2 runs right after 08a_3d_model: it draws Fig 2 from 08a's selected model (2026-10-30 rewire; 07b, 08c and
    #    the old 08b are retired to drafts/retired_2026-10-28/ and no longer run);
    #  - 01_prep_figures runs right after 01_prep_inputs.
    nbs=""
    for nb in $(ls [0-9][0-9]*_*.ipynb); do
        case "$nb" in
            07c_*|01_prep_figures*) continue ;;
        esac
        nbs="$nbs $nb"
        case "$nb" in
            01_prep_inputs*) nbs="$nbs $(ls 01_prep_figures*.ipynb)" ;;   # inputs first, then their figures
            08a_*) nbs="$nbs $(ls 07c_*.ipynb)" ;;
        esac
    done
fi

# the HTML copies go to the results directory (TulePond_filepaths.executed_dir), not into the repo (2026-10-06)
EXECUTED=$(python -c "import TulePond_filepaths as f; print(f.executed_dir)")
mkdir -p "$EXECUTED"
for nb in $nbs; do
    echo ""
    echo "=============================== $nb  $(date '+%H:%M:%S')"
    start=$(date +%s)
    # sync the .py source into the .ipynb first, then execute in place
    jupytext --sync "$nb"
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 "$nb"
    jupyter nbconvert --to html "$nb" --output-dir "$EXECUTED" >/dev/null 2>&1
    echo "done in $(( $(date +%s) - start )) s"
done

echo ""
echo "=============================== pytest"
python -m pytest -q
