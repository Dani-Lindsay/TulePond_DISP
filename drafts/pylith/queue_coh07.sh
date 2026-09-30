#!/bin/bash
# Coherence-0.7 sensitivity: GFs at the coh07 quadtree samples (vertical fault, default mesh, order 1, np 8): SFCVM aligned, then homog.
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
bash run_gf.sh gf_default_sfcvm_coh07 cfg/mat_sfcvm.cfg mesh/mesh_vertical_tet_default.msh spatialdb/impulses_km54-65_z3000.spatialdb 8 points/insar_points_coh07.txt
bash run_gf.sh gf_default_homog_coh07 cfg/mat_homog.cfg mesh/mesh_vertical_tet_default.msh spatialdb/impulses_km54-65_z3000.spatialdb 8 points/insar_points_coh07.txt
echo "== coh07 GF queue done $(date '+%H:%M:%S')"
