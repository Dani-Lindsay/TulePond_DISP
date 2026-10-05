#!/bin/bash
# Run the notebook source for the shallow-mesh sensitivity (homog + SFCVM aligned) in opera_disp_feb13.
# Products are suffixed _shallow; all PyLith outputs are re-used. Log: runs/notebook_shallow.log
HERE="$(cd "$(dirname "$0")" && pwd)"
E=/Users/dlindsay/miniforge3/envs/opera_disp_feb13
export PATH=$E/bin:$PATH PROJ_LIB=$E/share/proj PROJ_DATA=$E/share/proj MPLBACKEND=Agg
export PYTHONPATH=/Users/dlindsay/tools/internal/TulePond_DISP PYLITH_MESH_PRESET=shallow PYLITH_MEDIA=homog,sfcvm
cd "$HERE" && python pylith_sfcvm_gf.py > runs/notebook_shallow.log 2>&1
echo "exit $?" >> runs/notebook_shallow.log
