#!/bin/bash

python2.7 ../src/paver/paver.py \
  miplib2010_slvtrc/CBC.trc \
  miplib2010_slvtrc/CPLEX.trc \
  miplib2010_slvtrc/SCIPcpx.trc \
  miplib2010_slvtrc/SCIPspx.trc \
  miplib2010_slvtrc/*.solvetrace \
  ../solu/miplib2010.solu \
  --failtime 3600 \
  --writehtml miplib2010_slvtrc.all \
  --optfileisrunname

# we added --optfileisrunname to ignore the option file name
# this helps to match the solver ID from the solvetrace file with the solver names
