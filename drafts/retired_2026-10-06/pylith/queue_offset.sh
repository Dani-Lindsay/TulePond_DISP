#!/bin/bash
# Alignment sensitivity: SFCVM boundary 50 m east / west of the trace. Grids (opera env), dbs (PyLith env), then two GF runs (np 8).
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
E=/Users/dlindsay/miniforge3/envs/opera_disp_feb13
export PATH=$E/bin:$PATH PROJ_LIB=$E/share/proj PROJ_DATA=$E/share/proj MPLBACKEND=Agg PYTHONPATH=/Users/dlindsay/tools/internal/TulePond_DISP
for o in 50 -50; do
  t=$([ $o = 50 ] && echo e50 || echo w50)
  (cd /Users/dlindsay/tools/internal/TulePond_DISP && python drafts/pylith/sfcvm_grid.py --offset $o) || { echo "FAILED grid $t"; continue; }
  bash pylith_env.sh "$HERE" python3 write_sfcvm_griddb.py spatialdb/sfcvm_v26-0_aligned_${t}_grid.npz spatialdb/mat_sfcvm_v26-0_aligned_${t}.spatialdb || { echo "FAILED db $t"; continue; }
done
for t in e50 w50; do
  [ -f spatialdb/mat_sfcvm_v26-0_aligned_${t}.spatialdb ] && bash run_gf.sh gf_default_sfcvm_${t} cfg/mat_sfcvm_${t}.cfg mesh/mesh_vertical_tet_default.msh spatialdb/impulses_km54-65_z3000.spatialdb 8
done
echo "== offset queue done $(date '+%H:%M:%S')"
