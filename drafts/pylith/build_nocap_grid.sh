#!/bin/bash
HERE="$(cd "$(dirname "$0")" && pwd)"
E=/Users/dlindsay/miniforge3/envs/opera_disp_feb13
export PATH=$E/bin:$PATH PROJ_LIB=$E/share/proj PROJ_DATA=$E/share/proj MPLBACKEND=Agg PYTHONPATH=/Users/dlindsay/tools/internal/TulePond_DISP
cd /Users/dlindsay/tools/internal/TulePond_DISP
python drafts/pylith/sfcvm_grid.py --no-cap && bash "$HERE/pylith_env.sh" "$HERE" python3 write_sfcvm_griddb.py spatialdb/sfcvm_v26-0_aligned_nocap_grid.npz spatialdb/mat_sfcvm_v26-0_aligned_nocap.spatialdb
echo "== grid done $(date '+%H:%M:%S')"
cd "$HERE" && rm -f output/test_default_o2_nocap-* && bash run_gf_o2.sh test_default_o2_nocap cfg/mat_sfcvm_nocap.cfg mesh/mesh_vertical_tet_default.msh spatialdb/impulses_test20.spatialdb 2
echo "== test done $(date '+%H:%M:%S')"
