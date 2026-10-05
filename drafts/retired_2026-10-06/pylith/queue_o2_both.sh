#!/bin/bash
# Order-2 inversions with both media: full Cd now; nugget Cd after the SFCVM-only nugget run has released its tag.
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
echo "$(date '+%H:%M:%S') start order2 both full"; bash run_notebook.sh order2_both_full PYLITH_ORDER=2 PYLITH_MEDIA=homog,sfcvm
until grep -q "^done" runs/queue_o2_sfcvm_inv.log 2>/dev/null; do sleep 15; done
echo "$(date '+%H:%M:%S') start order2 both nugget"; bash run_notebook.sh order2_both_nugget PYLITH_ORDER=2 PYLITH_MEDIA=homog,sfcvm PYLITH_CD=nugget
echo "$(date '+%H:%M:%S') done"
