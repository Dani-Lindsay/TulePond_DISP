#!/bin/bash
# Separate the three order-2 changes on the 33 test impulses (np 2): P2 + materials order 0 + no cap; P1 + order 0 + no cap; P2 homog.
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
MAT0="--problem.materials.elastic.auxiliary_subfields.density.basis_order=0 --problem.materials.elastic.bulk_rheology.auxiliary_subfields.bulk_modulus.basis_order=0 --problem.materials.elastic.bulk_rheology.auxiliary_subfields.shear_modulus.basis_order=0"
run() { NAME=$1; shift; echo "$(date '+%Y-%m-%d %H:%M:%S') start $NAME (o2 tests)" >> runs/run_gf.log
  caffeinate -ims bash pylith_env.sh "$HERE" pylith "$@" --problem.defaults.name=$NAME --problem.interfaces.fault.db_auxiliary_field.filename=spatialdb/impulses_test20.spatialdb --nodes=2 > runs/$NAME.log 2>&1
  echo "$(date '+%Y-%m-%d %H:%M:%S') done $NAME" >> runs/run_gf.log; }
run test_default_o2_mat0_nocap cfg/pylithapp.cfg cfg/mat_sfcvm_nocap.cfg cfg/gf.cfg cfg/order2.cfg $MAT0
run test_default_o1_mat0_nocap cfg/pylithapp.cfg cfg/mat_sfcvm_nocap.cfg cfg/gf.cfg
rm -f output/test_default_o2-*; run test_default_o2_homog cfg/pylithapp.cfg cfg/mat_homog.cfg cfg/gf.cfg cfg/order2.cfg
echo "== o2 tests done $(date '+%H:%M:%S')"
