#!/usr/bin/env bash
# Run every notebook top to bottom, in order, from a fresh kernel, and stop at
# the first failure. This is the "restart and run all" that every number in
# the manuscript must come from. Pass notebook numbers to run a subset:
#     ./run_all.sh            # all
#     ./run_all.sh 03 05      # just those
#     ./run_all.sh 01_prep_inputs 07a 08a   # a full name or a prefix, run in the order given
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
    # Name order, except that 07c_fig2 runs right after 08b: 07c draws Fig 2 from 08a's 3D-model
    # products (and asserts they are on 07a's current samples), so in plain name order it would
    # run before 08a and fail, or draw the previous run's model. 08b re-uses 08a and does not
    # read 07c, so 07c goes after both. (The pattern takes a letter after the two digits, so
    # 07a-08b are included; the earlier [0-9][0-9]_* pattern skipped them, 2026-10-03.)
    nbs=""
    for nb in $(ls [0-9][0-9]*_*.ipynb); do
        case "$nb" in
            07c_*|01_prep_figures*) continue ;;
        esac
        nbs="$nbs $nb"
        case "$nb" in
            01_prep_inputs*) nbs="$nbs $(ls 01_prep_figures*.ipynb)" ;;   # inputs first, then their figures
            08b_*) nbs="$nbs $(ls 07c_*.ipynb)" ;;
        esac
    done
fi

mkdir -p executed
for nb in $nbs; do
    echo ""
    echo "=============================== $nb  $(date '+%H:%M:%S')"
    start=$(date +%s)
    # sync the .py source into the .ipynb first, then execute in place
    jupytext --sync "$nb"
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 "$nb"
    jupyter nbconvert --to html "$nb" --output-dir executed >/dev/null 2>&1
    echo "done in $(( $(date +%s) - start )) s"
done

echo ""
echo "=============================== pytest"
python -m pytest -q
