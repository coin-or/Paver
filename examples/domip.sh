#!/bin/bash

python ../src/paver/paver.py \
  miplib2010/gimli.trc \
  miplib2010/bombur.trc \
  miplib2010/thorin.trc \
  miplib2010/bifur.trc \
  miplib2010/balin.trc \
  ../solu/miplib2010.solu \
  --failtime 3600 \
  --refsolver Thorin \
  --writehtml miplib2010.all
