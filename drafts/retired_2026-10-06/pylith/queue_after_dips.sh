#!/bin/bash
# After the east-dip runs: order-2 SFCVM (no nu cap) and order-2 homog on the vertical default mesh (all 2105 impulses,
# np 8), then the 80W sensitivity and the vertical per-depth-aligned control if the day allows.
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
until grep -q "done gf_default_dip70e_sfcvm\|SKIPPED gf_default_dip70e_sfcvm" runs/run_gf.log; do sleep 60; done
bash run_gf_o2.sh gf_default_sfcvm_order2 cfg/mat_sfcvm_nocap.cfg mesh/mesh_vertical_tet_default.msh spatialdb/impulses_km54-65_z3000.spatialdb 8
bash run_gf_o2.sh gf_default_homog_order2 cfg/mat_homog.cfg mesh/mesh_vertical_tet_default.msh spatialdb/impulses_km54-65_z3000.spatialdb 8
bash run_gf.sh gf_default_dip80w_homog cfg/mat_homog.cfg mesh/mesh_dip80w_tet_default.msh spatialdb/impulses_km54-65_z3000.spatialdb 8
bash run_gf.sh gf_default_dip80w_sfcvm cfg/mat_sfcvm_dip80w.cfg mesh/mesh_dip80w_tet_default.msh spatialdb/impulses_km54-65_z3000.spatialdb 8
bash run_gf.sh gf_default_sfcvmv cfg/mat_sfcvm_dip90v.cfg mesh/mesh_vertical_tet_default.msh spatialdb/impulses_km54-65_z3000.spatialdb 8
