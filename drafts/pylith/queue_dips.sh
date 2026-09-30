#!/bin/bash
# Dip GF runs (default mesh, 2386 default samples, np 8), started once the swath-7 queue is done: 80E and 70E, homog + SFCVM aligned to the dip.
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
until grep -q "done gf_default_sfcvm_swath7" runs/run_gf.log; do sleep 60; done
for d in 80e 70e; do
  bash run_gf.sh gf_default_dip${d}_homog cfg/mat_homog.cfg mesh/mesh_dip${d}_tet_default.msh spatialdb/impulses_km54-65_z3000.spatialdb 8
  if [ -f spatialdb/mat_sfcvm_v26-0_aligned_dip${d}.spatialdb ]; then
    bash run_gf.sh gf_default_dip${d}_sfcvm cfg/mat_sfcvm_dip${d}.cfg mesh/mesh_dip${d}_tet_default.msh spatialdb/impulses_km54-65_z3000.spatialdb 8
  else
    echo "$(date '+%Y-%m-%d %H:%M:%S') SKIPPED gf_default_dip${d}_sfcvm: no griddb" >> runs/run_gf.log
  fi
done
