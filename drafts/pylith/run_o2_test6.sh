#!/bin/bash
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
until grep -q "done test_default_o2_homog" runs/run_gf.log; do sleep 20; done
MAT0="--problem.materials.elastic.auxiliary_subfields.density.basis_order=0 --problem.materials.elastic.bulk_rheology.auxiliary_subfields.bulk_modulus.basis_order=0 --problem.materials.elastic.bulk_rheology.auxiliary_subfields.shear_modulus.basis_order=0"
NAME=test_default_o2_mat0_cap
echo "$(date '+%Y-%m-%d %H:%M:%S') start $NAME (o2 tests)" >> runs/run_gf.log
caffeinate -ims bash pylith_env.sh "$HERE" pylith cfg/pylithapp.cfg cfg/mat_sfcvm.cfg cfg/gf.cfg cfg/order2.cfg $MAT0 --problem.defaults.name=$NAME --problem.interfaces.fault.db_auxiliary_field.filename=spatialdb/impulses_test20.spatialdb --nodes=2 > runs/$NAME.log 2>&1
echo "$(date '+%Y-%m-%d %H:%M:%S') done $NAME" >> runs/run_gf.log
