#!/bin/bash
# As run_gf.sh, with cfg/order2.cfg appended (order-2 solution, materials order 1, slip impulses order 1).
# Usage: bash run_gf_o2.sh <name> <mat cfg> <mesh file> <impulse db> <nprocs> [<points file>]
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME=$1; MAT=$2; MESH=$3; IMP=$4; NP=${5:-1}; PTS=${6:-points/insar_points.txt}
mkdir -p "$HERE/runs"
echo "$(date '+%Y-%m-%d %H:%M:%S') start $NAME ($MAT, $MESH, $IMP, np $NP, points $PTS, order2)" >> "$HERE/runs/run_gf.log"
caffeinate -ims bash "$HERE/pylith_env.sh" "$HERE" pylith cfg/pylithapp.cfg "$MAT" cfg/gf.cfg cfg/order2.cfg \
    --problem.defaults.name="$NAME" \
    --problem.mesh_initializer.phases.read_mesh.reader.filename="$MESH" \
    --problem.interfaces.fault.db_auxiliary_field.filename="$IMP" \
    --problem.solution_observers.points.reader.filename="$PTS" \
    --nodes="$NP" > "$HERE/runs/$NAME.log" 2>&1
echo "$(date '+%Y-%m-%d %H:%M:%S') done $NAME" >> "$HERE/runs/run_gf.log"
