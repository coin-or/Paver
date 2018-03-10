#!/bin/bash

set -e

# use 900s if solver failed
# reduce mintime to 0.1s, which fits better to local solvers
# disable examiner checks for feasibility and optimality
python2.7 ../src/paver/paver.py \
  globallib/dasher.trc \
  globallib/dancer.trc \
  globallib/comet.trc \
  globallib/cupid.trc \
  globallib/donner.trc \
  ../solu/globallib.solu \
  --failtime 900 \
  --mintime 0.1 \
  --ccopttol inf \
  --ccfeastol inf \
  --writehtml globallib.localExamNo

# as before, but enable examiner checks for feasibility/optimality
python2.7 ../src/paver/paver.py \
  globallib/dasher.trc \
  globallib/dancer.trc \
  globallib/comet.trc \
  globallib/cupid.trc \
  globallib/donner.trc \
  ../solu/globallib.solu \
  --failtime 900 \
  --mintime 0.1 \
  --ccopttol 1e-5 \
  --ccfeastol 1e-5 \
  --writehtml globallib.localExamYes
