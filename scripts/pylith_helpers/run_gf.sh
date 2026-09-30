#!/bin/bash
# Run one PyLith GreensFns job in the PyLith env.
# Usage: bash run_gf.sh <name> <mat cfg name in $PYLITH_CFG, e.g. mat_homog.cfg> <mesh file> <impulse db> <nprocs> [<points file>]   (mesh, db, points relative to $PYLITH_WORKDIR)
# e.g. bash run_gf.sh gf_default_homog mat_homog.cfg mesh/mesh_vertical_tet_default.msh spatialdb/impulses_km54-65_z3000.spatialdb 8
# The optional 6th argument replaces the output points (default cfg/gf.cfg: points/insar_points.txt, the 2386 samples).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
# Ported from drafts/pylith (2026-09-30): the run works in $PYLITH_WORKDIR (TulePond_filepaths.model_sfcvm_dir:
# mesh/, spatialdb/, points/ inputs, pylith_output/ results, runs/ logs) with the cfg files carried by the repo
# in scripts/pylith_cfg (or $PYLITH_CFG). Notebook 08a sets both when TULEPOND_RUN_PYLITH=1.
WORK="${PYLITH_WORKDIR:?set PYLITH_WORKDIR to TulePond_filepaths.model_sfcvm_dir}"
CFG="${PYLITH_CFG:-$HERE/../pylith_cfg}"
NAME=$1; MAT=$2; MESH=$3; IMP=$4; NP=${5:-1}; PTS=${6:-points/insar_points.txt}
mkdir -p "$WORK/runs"
echo "$(date '+%Y-%m-%d %H:%M:%S') start $NAME ($MAT, $MESH, $IMP, np $NP, points $PTS)" >> "$WORK/runs/run_gf.log"
caffeinate -ims bash "$HERE/pylith_env.sh" "$WORK" pylith "$CFG/pylithapp.cfg" "$CFG/$MAT" "$CFG/gf.cfg" \
    --problem.defaults.name="$NAME" \
    --problem.mesh_initializer.phases.read_mesh.reader.filename="$MESH" \
    --problem.interfaces.fault.db_auxiliary_field.filename="$IMP" \
    --problem.solution_observers.points.reader.filename="$PTS" \
    --nodes="$NP" > "$WORK/runs/$NAME.log" 2>&1
echo "$(date '+%Y-%m-%d %H:%M:%S') done $NAME" >> "$WORK/runs/run_gf.log"
