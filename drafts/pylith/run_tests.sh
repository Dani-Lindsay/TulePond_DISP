#!/bin/bash
HERE="$(cd "$(dirname "$0")" && pwd)"
E=/Users/dlindsay/miniforge3/envs/opera_disp_feb13
export PATH=$E/bin:$PATH PROJ_LIB=$E/share/proj PROJ_DATA=$E/share/proj MPLBACKEND=Agg PYTHONPATH=/Users/dlindsay/tools/internal/TulePond_DISP
cd "$HERE" && caffeinate -ims python pylith_sfcvm_tests.py > runs/tests.log 2>&1
echo "exit $?" >> runs/tests.log
