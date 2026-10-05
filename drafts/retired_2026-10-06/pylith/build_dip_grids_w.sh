#!/bin/bash
# SFCVM grids aligned to the dipping faults (opera env) and their SimpleGridDBs (PyLith env). Log: runs/dip_grids_w.log
HERE="$(cd "$(dirname "$0")" && pwd)"
E=/Users/dlindsay/miniforge3/envs/opera_disp_feb13
export PATH=$E/bin:$PATH PROJ_LIB=$E/share/proj PROJ_DATA=$E/share/proj MPLBACKEND=Agg PYTHONPATH=/Users/dlindsay/tools/internal/TulePond_DISP
cd /Users/dlindsay/tools/internal/TulePond_DISP
for d in 80w; do
  echo "== $(date '+%H:%M:%S') sfcvm_grid --dip $d"
  python drafts/pylith/sfcvm_grid.py --dip $d || { echo "FAILED sfcvm_grid --dip $d"; continue; }
  bash "$HERE/pylith_env.sh" "$HERE" python3 write_sfcvm_griddb.py spatialdb/sfcvm_v26-0_aligned_dip${d}_grid.npz spatialdb/mat_sfcvm_v26-0_aligned_dip${d}.spatialdb || echo "FAILED griddb $d"
done
echo "== $(date '+%H:%M:%S') done"
