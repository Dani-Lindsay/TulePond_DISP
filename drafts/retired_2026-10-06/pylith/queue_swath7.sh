#!/bin/bash
# GF runs at the swath-7 sample set (vertical fault, default mesh), homog then SFCVM aligned, 8 ranks each.
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
bash run_gf.sh gf_default_homog_swath7 cfg/mat_homog.cfg mesh/mesh_vertical_tet_default.msh spatialdb/impulses_km54-65_z3000.spatialdb 8 points/insar_points_swath7.txt
bash run_gf.sh gf_default_sfcvm_swath7 cfg/mat_sfcvm.cfg mesh/mesh_vertical_tet_default.msh spatialdb/impulses_km54-65_z3000.spatialdb 8 points/insar_points_swath7.txt
