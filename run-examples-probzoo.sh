#!/bin/bash -e

export PYTHONPATH=$(pwd)

for f in $(ls examples/probzoo/*.py)
do
    echo "Running file $f ..."
    python $f
    echo ""
done