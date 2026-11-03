#!/usr/bin/env bash
# Run every notebook top to bottom, in order, from a fresh kernel, and stop at
# the first failure. This is the "restart and run all" that every number in
# the manuscript must come from. Pass notebook numbers to run a subset:
#     ./run_all.sh            # all
#     ./run_all.sh 03 05      # just those
#     ./run_all.sh 01_prep_inputs 07 08   # a full name or a prefix, run in the order given
#     ./run_all.sh 07 08 09 10         # the slip models and the model sensitivity (given 04 and 06)
#
# The sensitivity scripts (scripts/sensitivity/*.py, ~5 h) are not run here: 10_model_sensitivity draws their stored
# outputs. Run them by hand when a model setting changes (scripts/sensitivity/common.py).
#
# Uses the python next to this script's env (opera_disp_feb13); call it with
# the env's full path if `jupyter` is not on PATH:
#     PATH=/Users/dlindsay/miniforge3/envs/opera_disp_feb13/bin:$PATH ./run_all.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# the results directory must be named: TulePond_filepaths falls back to ~/work/proj_TulePond_DISP otherwise, which is
# not the release record
: "${TULEPOND_DATA_DIR:?set TULEPOND_DATA_DIR to the results directory, e.g. ~/work/proj_TulePond_DISP_release}"
export TULEPOND_DATA_DIR
# the commit and whether the tree was clean, recorded before any notebook runs: executing the notebooks in place makes
# the tree dirty, so insar_utils.git_hash reports this start state for the whole run
export TULEPOND_RUN_COMMIT=$(git rev-parse --short HEAD)
if [ -z "$(git status --porcelain)" ]; then export TULEPOND_RUN_CLEAN=1; else export TULEPOND_RUN_CLEAN=0; fi
echo "run_all: commit $TULEPOND_RUN_COMMIT, clean tree at start: $TULEPOND_RUN_CLEAN, results in $TULEPOND_DATA_DIR"

if [ $# -gt 0 ]; then
    nbs=""
    for n in "$@"; do nbs="$nbs $(ls ${n}*.ipynb)"; done
else
    # Name order, except that 01_prep_figures runs right after 01_prep_inputs (it draws the inputs' figures)
    nbs=""
    for nb in $(ls [0-9][0-9]*_*.ipynb); do
        case "$nb" in
            01_prep_figures*) continue ;;
        esac
        nbs="$nbs $nb"
        case "$nb" in
            01_prep_inputs*) nbs="$nbs $(ls 01_prep_figures*.ipynb)" ;;
        esac
    done
fi

# the HTML copies go to the results directory (TulePond_filepaths.executed_dir), not into the repo
EXECUTED=$(python -c "import TulePond_filepaths as f; print(f.executed_dir)")
mkdir -p "$EXECUTED"
for nb in $nbs; do
    echo ""
    echo "=============================== $nb  $(date '+%H:%M:%S')"
    start=$(date +%s)
    # execute in place (the notebooks are the source; no paired .py scripts since 2 Nov 2026)
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 "$nb"
    jupyter nbconvert --to html "$nb" --output-dir "$EXECUTED" >/dev/null 2>&1
    echo "done in $(( $(date +%s) - start )) s"
done

echo ""
echo "=============================== pytest"
python -m pytest -q
