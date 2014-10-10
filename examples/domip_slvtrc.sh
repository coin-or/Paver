#!/bin/bash

python ../src/paver/paver.py \
  miplib2010_slvtrc/CBC.trc \
  miplib2010_slvtrc/CPLEX.trc \
  miplib2010_slvtrc/SCIPcpx.trc \
  miplib2010_slvtrc/SCIPspx.trc \
  miplib2010_slvtrc/*.solvetrace \
  ../solu/miplib2010.solu \
  --failtime 3600 \
  --writehtml miplib2010_slvtrc.all
