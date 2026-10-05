#!/bin/bash
# One PyLith GreensFns run (FaultCohesiveImpulses: one unit left-lateral impulse per fault vertex of the impulse region,
# displacement written at the sample points), in the PyLith 5.0.2 bundle, made from the results directory.
# Usage:
#   bash pylith/run_gf.sh <name> <mat cfg> <mesh> <impulse db> <nprocs> <points> [<extra cfg> ...]
#     <mat cfg>     a file of pylith/cfg: mat_homog.cfg (outputs -> modeling_uniform/pylith_output) or mat_sfcvm*.cfg
#                   (-> modeling_SFCVM/pylith_output)
#     <mesh>, <impulse db>, <points>   relative to the results directory (modeling_inputs/...)
#     <extra cfg>   further files of pylith/cfg, e.g. order2.cfg (quadratic solution)
# e.g. bash pylith/run_gf.sh gf_default_homog mat_homog.cfg modeling_inputs/mesh/mesh_vertical_tet_default.msh \
#        modeling_inputs/spatialdb/impulses_km54-65_z3000.spatialdb 8 modeling_inputs/points/insar_points.txt
# Writes <name>-points.h5, -fault.h5, -elastic_info.h5, -parameters.json (+ the boundary outputs) into the medium's
# pylith_output/; the PyLith log to <medium>/runs/<name>.log and one start/done line to <medium>/runs/run_gf.log.
# About 33 min for the 2386 samples on 8 ranks (Apple M-series). See pylith/README.md for every run of the paper.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
NAME=$1; MAT=$2; MESH=$3; IMP=$4; NP=$5; PTS=$6; shift 6
EXTRA=""
for c in "$@"; do EXTRA="$EXTRA $CFG/$c"; done
LOGS=$(log_dir "$MAT")
mkdir -p "$LOGS"
echo "$(date '+%Y-%m-%d %H:%M:%S') start $NAME ($MAT, $MESH, $IMP, np $NP, points $PTS${EXTRA:+, extra$EXTRA})" >> "$LOGS/run_gf.log"
caffeinate -ims bash "$HERE/pylith_env.sh" "$WORK" pylith "$CFG/pylithapp.cfg" "$CFG/$MAT" "$CFG/gf.cfg" $EXTRA \
    --problem.defaults.name="$NAME" \
    --problem.mesh_initializer.phases.read_mesh.reader.filename="$MESH" \
    --problem.interfaces.fault.db_auxiliary_field.filename="$IMP" \
    --problem.solution_observers.points.reader.filename="$PTS" \
    --nodes="$NP" > "$LOGS/$NAME.log" 2>&1
echo "$(date '+%Y-%m-%d %H:%M:%S') done $NAME" >> "$LOGS/run_gf.log"
