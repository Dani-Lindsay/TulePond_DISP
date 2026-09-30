#!/bin/bash
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
echo "$(date '+%H:%M:%S') start swath7 full"; bash run_notebook.sh swath7_full PYLITH_POINTS=swath7 PYLITH_MEDIA=homog,sfcvm
echo "$(date '+%H:%M:%S') start swath7 nugget"; bash run_notebook.sh swath7_nugget PYLITH_POINTS=swath7 PYLITH_MEDIA=homog,sfcvm PYLITH_CD=nugget
echo "$(date '+%H:%M:%S') done swath7"
