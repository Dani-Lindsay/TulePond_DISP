#!/usr/bin/env bash
# Run every notebook top to bottom, in order, from a fresh kernel, and stop at
# the first failure. This is the "restart and run all" that every number in
# the manuscript must come from. Pass notebook numbers to run a subset:
#     ./run_all.sh            # all
#     ./run_all.sh 03 05      # just those
#
# Uses the python next to this script's env (opera_disp_feb13); call it with
# the env's full path if `jupyter` is not on PATH:
#     PATH=/Users/dlindsay/miniforge3/envs/opera_disp_feb13/bin:$PATH ./run_all.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ $# -gt 0 ]; then
    nbs=""
    for n in "$@"; do nbs="$nbs $(ls ${n}_*.ipynb)"; done
else
    nbs=$(ls [0-9][0-9]_*.ipynb)
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
