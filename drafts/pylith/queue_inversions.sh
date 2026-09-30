#!/bin/bash
# Notebook inversions as the GF runs finish (each ~15 min, cached GFs): swath 7, dips 80E/70E, 80W sensitivity,
# each with 07's full Cd and with the nugget-only diagonal Cd. Log: runs/queue_inversions.log
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
wait_for() { until grep -q "done $1" runs/run_gf.log; do sleep 60; done; }
run_pair() {   # <label> <env...>
  L=$1; shift
  echo "$(date '+%H:%M:%S') start $L full"; bash run_notebook.sh ${L}_full "$@" PYLITH_MEDIA=homog,sfcvm
  echo "$(date '+%H:%M:%S') start $L nugget"; bash run_notebook.sh ${L}_nugget "$@" PYLITH_MEDIA=homog,sfcvm PYLITH_CD=nugget
  echo "$(date '+%H:%M:%S') done $L"
}
wait_for gf_default_sfcvm_swath7;  run_pair swath7 PYLITH_POINTS=swath7
wait_for gf_default_dip80e_sfcvm;  run_pair dip80e PYLITH_DIP=80e
wait_for gf_default_dip70e_sfcvm;  run_pair dip70e PYLITH_DIP=70e
wait_for gf_default_dip80w_sfcvm;  run_pair dip80w PYLITH_DIP=80w
