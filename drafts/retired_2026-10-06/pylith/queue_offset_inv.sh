#!/bin/bash
# Inversions of the alignment-sensitivity GFs as each finishes: medium sfcvm_<e50|w50> alone, full then nugget Cd.
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
for t in e50 w50; do
  until grep -q "done gf_default_sfcvm_$t" runs/run_gf.log; do sleep 30; done
  echo "$(date '+%H:%M:%S') start $t full";   bash run_notebook.sh align_${t}_full   PYLITH_MEDIA=sfcvm_$t
  echo "$(date '+%H:%M:%S') start $t nugget"; bash run_notebook.sh align_${t}_nugget PYLITH_MEDIA=sfcvm_$t PYLITH_CD=nugget
done
echo "$(date '+%H:%M:%S') done"
