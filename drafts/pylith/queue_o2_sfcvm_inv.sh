#!/bin/bash
# Order-2 SFCVM inversions (no order-2 homog yet): full Cd, then nugget Cd. Log: runs/queue_o2_sfcvm_inv.log
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
echo "$(date '+%H:%M:%S') start order2 sfcvm full"; bash run_notebook.sh order2_sfcvm_full PYLITH_ORDER=2 PYLITH_MEDIA=sfcvm
echo "$(date '+%H:%M:%S') start order2 sfcvm nugget"; bash run_notebook.sh order2_sfcvm_nugget PYLITH_ORDER=2 PYLITH_MEDIA=sfcvm PYLITH_CD=nugget
echo "$(date '+%H:%M:%S') done"
