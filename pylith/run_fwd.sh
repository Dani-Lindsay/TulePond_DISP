#!/bin/bash
# One PyLith forward run (FaultCohesiveKin: a prescribed static slip, one time step), in the PyLith 5.0.2 bundle, made
# from the results directory: the validation runs (fwd_kin.cfg: 10 mm on km 57-61, top 1 km) and the forward runs of an
# inverted slip model (fwd_kin_simpledb.cfg with its slip database; notebook 08a [14], [15b]).
# Usage:
#   bash pylith/run_fwd.sh <name> <mat cfg> <mesh> <nprocs> <fwd cfg> [<extra cfg> ...] [--<pylith option>=<value> ...]
#     <mat cfg>, <fwd cfg>, <extra cfg>   files of pylith/cfg (e.g. order2_fwd.cfg after fwd_kin.cfg)
#     <mesh>        relative to the results directory (modeling_inputs/mesh/...)
#     --options     passed to pylith as they are (e.g. the points file, the slip database, the fault observer fields)
# e.g. bash pylith/run_fwd.sh fwd_default_homog mat_homog.cfg modeling_inputs/mesh/mesh_vertical_tet_default.msh 2 fwd_kin.cfg
# Writes <name>-points.h5, -fault.h5, ... into the medium's pylith_output/ (set by the material cfg); the log to
# <medium>/runs/<name>.log. About 2 min on 2 ranks (validation runs), 4 ranks for the Fig-2 forward runs.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
NAME=$1; MAT=$2; MESH=$3; NP=$4; shift 4
CFGS=""
OPTS=""
for a in "$@"; do
    case "$a" in
        --*) OPTS="$OPTS $(printf '%q' "$a")" ;;
        *.cfg) CFGS="$CFGS $CFG/$a" ;;
        *) echo "run_fwd.sh: $a is neither a cfg file nor a --option" >&2; exit 1 ;;
    esac
done
LOGS=$(log_dir "$MAT")
mkdir -p "$LOGS"
echo "$(date '+%Y-%m-%d %H:%M:%S') start $NAME ($MAT, $MESH, np $NP,$CFGS$OPTS)" >> "$LOGS/run_gf.log"
caffeinate -ims bash "$HERE/pylith_env.sh" "$WORK" pylith "$CFG/pylithapp.cfg" "$CFG/$MAT" $CFGS \
    --problem.defaults.name="$NAME" \
    --problem.mesh_initializer.phases.read_mesh.reader.filename="$MESH" \
    --nodes="$NP" $OPTS > "$LOGS/$NAME.log" 2>&1
echo "$(date '+%Y-%m-%d %H:%M:%S') done $NAME" >> "$LOGS/run_gf.log"
