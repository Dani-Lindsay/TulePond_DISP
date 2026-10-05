#!/bin/bash
# Coherence-0.7 sensitivity inversions (PYLITH_POINTS=coh07), order 1, vertical, default mesh, surface ON.
#  1. as soon as gf_default_sfcvm_coh07 is done: SFCVM alone, nugget Cd (the selected setup) -> early numbers;
#  2. when gf_default_homog_coh07 is done: both media, full Cd, then both media, nugget Cd (overwrites 1 with a superset);
#  3. the Fig-2 product of the coh-0.7 nugget run (make_fig2_product.py _nuggetcd _coh07).
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
until grep -q "done gf_default_sfcvm_coh07" runs/run_gf.log; do sleep 30; done
echo "$(date '+%H:%M:%S') start coh07 sfcvm nugget (early)"; bash run_notebook.sh coh07_sfcvm_nugget PYLITH_POINTS=coh07 PYLITH_MEDIA=sfcvm PYLITH_CD=nugget
until grep -q "done gf_default_homog_coh07" runs/run_gf.log; do sleep 30; done
echo "$(date '+%H:%M:%S') start coh07 both full";   bash run_notebook.sh coh07_both_full   PYLITH_POINTS=coh07 PYLITH_MEDIA=homog,sfcvm
echo "$(date '+%H:%M:%S') start coh07 both nugget"; bash run_notebook.sh coh07_both_nugget PYLITH_POINTS=coh07 PYLITH_MEDIA=homog,sfcvm PYLITH_CD=nugget
E=/Users/dlindsay/miniforge3/envs/opera_disp_feb13
export PATH=$E/bin:$PATH PROJ_LIB=$E/share/proj PROJ_DATA=$E/share/proj MPLBACKEND=Agg PYTHONPATH=/Users/dlindsay/tools/internal/TulePond_DISP
echo "$(date '+%H:%M:%S') start fig2 product coh07 nugget"; python make_fig2_product.py _nuggetcd _coh07 > runs/fig2_coh07.log 2>&1; echo "fig2 exit $?"
echo "$(date '+%H:%M:%S') done"
