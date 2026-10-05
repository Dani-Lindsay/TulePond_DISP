#!/bin/bash
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
NAME=fwd_default_o2_homog
echo "$(date '+%Y-%m-%d %H:%M:%S') start $NAME (P2 forward validation)" >> runs/run_gf.log
caffeinate -ims bash pylith_env.sh "$HERE" pylith cfg/pylithapp.cfg cfg/mat_homog.cfg cfg/fwd_kin.cfg cfg/order2_fwd.cfg --problem.defaults.name=$NAME --nodes=2 > runs/$NAME.log 2>&1
echo "$(date '+%Y-%m-%d %H:%M:%S') done $NAME" >> runs/run_gf.log
