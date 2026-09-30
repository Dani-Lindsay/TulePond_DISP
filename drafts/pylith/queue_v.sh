#!/bin/bash
# Control run: vertical default mesh, SFCVM contrast held on the fault at all depths (medium sfcvmv), after the 80W sensitivity.
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
until grep -q "done gf_default_dip80w_sfcvm" runs/run_gf.log; do sleep 60; done
[ -f spatialdb/mat_sfcvm_v26-0_aligned_dip90v.spatialdb ] && bash run_gf.sh gf_default_sfcvmv cfg/mat_sfcvm_dip90v.cfg mesh/mesh_vertical_tet_default.msh spatialdb/impulses_km54-65_z3000.spatialdb 8
