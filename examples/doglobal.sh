#!/bin/bash

# use 900s if solver failed
# reduce mintime to 0.1s, which fits better to local solvers
# disable examiner checks for optimality
python2.7 ../src/paver/paver.py \
  globallib/blitzen.trc \
  globallib/comet.trc \
  globallib/dasher.trc \
  globallib/prancer.trc \
  globallib/donner.trc \
  ../solu/globallib.solu \
  --failtime 900 \
  --mintime 0.1 \
  --ccopttol inf \
  --ccfeastol 1e-5 \
  --writehtml globallib.global
