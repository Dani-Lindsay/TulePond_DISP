#!/bin/bash
# Sourced by run_gf.sh: where the runs are made and where their logs go.
#   HERE   this folder (pylith/), CFG its cfg/ folder (the parameter files the runs are made with)
#   WORK   the results directory data_dir (TulePond_filepaths.paths_pylith['workdir'], or $PYLITH_WORKDIR): every path in
#          the cfg files and on the command line is relative to it
#   log_dir <mat cfg>   the runs/ folder of the medium a material cfg writes to (its defaults.output_directory)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="$HERE/cfg"
WORK="${PYLITH_WORKDIR:-$(cd "$HERE/.." && python3 -c 'import TulePond_filepaths as f; print(f.paths_pylith["workdir"])')}"
log_dir() {
    local out
    out=$(grep -E "^defaults.output_directory" "$CFG/$1" | sed 's/.*= *//')
    [ -n "$out" ] || { echo "$1 sets no defaults.output_directory" >&2; exit 1; }
    echo "$WORK/$(dirname "$out")/runs"
}
