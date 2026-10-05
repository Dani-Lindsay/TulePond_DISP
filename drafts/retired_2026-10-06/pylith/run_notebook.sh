#!/bin/bash
# Run the notebook source in opera_disp_feb13 with the env switches given as KEY=VALUE arguments; log runs/notebook_<tag>.log.
# e.g. bash run_notebook.sh floor15 PYLITH_CD=floor PYLITH_CD_FLOOR=15 PYLITH_MEDIA=homog,sfcvm
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG=$1; shift
E=/Users/dlindsay/miniforge3/envs/opera_disp_feb13
export PATH=$E/bin:$PATH PROJ_LIB=$E/share/proj PROJ_DATA=$E/share/proj MPLBACKEND=Agg
export PYTHONPATH=/Users/dlindsay/tools/internal/TulePond_DISP
for kv in "$@"; do export "$kv"; done
cd "$HERE" && caffeinate -ims python pylith_sfcvm_gf.py > "runs/notebook_$LOG.log" 2>&1
echo "exit $?" >> "runs/notebook_$LOG.log"
