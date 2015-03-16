#!/bin/bash

python2.7 ../src/paver/paver.py \
  linlib/A.trc \
  linlib/B.trc \
  linlib/C.trc \
  ../solu/linlib.solu \
  --writehtml linlib.all

python2.7 ../src/paver/paver.py \
  linlib/lp1gen.trc \
  linlib/lp2gen.trc \
  linlib/lp3gen.trc \
  ../solu/linlib.solu \
  --writehtml linlib.gen
