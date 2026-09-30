#!/bin/bash
# West-dip SENSITIVITY (geologically implausible): 80W homog + SFCVM, after the east-dip queue.
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
until grep -q "done gf_default_dip70e_sfcvm\|SKIPPED gf_default_dip70e_sfcvm" runs/run_gf.log; do sleep 60; done
bash run_gf.sh gf_default_dip80w_homog cfg/mat_homog.cfg mesh/mesh_dip80w_tet_default.msh spatialdb/impulses_km54-65_z3000.spatialdb 8
if [ -f spatialdb/mat_sfcvm_v26-0_aligned_dip80w.spatialdb ]; then
  bash run_gf.sh gf_default_dip80w_sfcvm cfg/mat_sfcvm_dip80w.cfg mesh/mesh_dip80w_tet_default.msh spatialdb/impulses_km54-65_z3000.spatialdb 8
fi
