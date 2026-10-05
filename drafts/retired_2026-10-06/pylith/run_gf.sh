#!/bin/bash
# Run one PyLith GreensFns job in the PyLith env.
# Usage: bash run_gf.sh <name> <mat cfg> <mesh file> <impulse db> <nprocs> [<points file>]
# e.g. bash run_gf.sh gf_default_homog cfg/mat_homog.cfg mesh/mesh_vertical_tet_default.msh spatialdb/impulses_km54-65_z3000.spatialdb 8
# The optional 6th argument replaces the output points (default cfg/gf.cfg: points/insar_points.txt, the 2386 samples).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME=$1; MAT=$2; MESH=$3; IMP=$4; NP=${5:-1}; PTS=${6:-points/insar_points.txt}
mkdir -p "$HERE/runs"
echo "$(date '+%Y-%m-%d %H:%M:%S') start $NAME ($MAT, $MESH, $IMP, np $NP, points $PTS)" >> "$HERE/runs/run_gf.log"
caffeinate -ims bash "$HERE/pylith_env.sh" "$HERE" pylith cfg/pylithapp.cfg "$MAT" cfg/gf.cfg \
    --problem.defaults.name="$NAME" \
    --problem.mesh_initializer.phases.read_mesh.reader.filename="$MESH" \
    --problem.interfaces.fault.db_auxiliary_field.filename="$IMP" \
    --problem.solution_observers.points.reader.filename="$PTS" \
    --nodes="$NP" > "$HERE/runs/$NAME.log" 2>&1
echo "$(date '+%Y-%m-%d %H:%M:%S') done $NAME" >> "$HERE/runs/run_gf.log"
