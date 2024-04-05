#!/bin/bash

set -ex

./check-warpii-env.sh

unamestr=$(uname)

if [[ "$unamestr" == 'Darwin' ]]; then
    brew install openmpi
else
    echo "We don't know how to install openmpi automatically on this system."
fi

# Check that cmake will be able to find, compile and link against MPI.
cmake --find-package -DNAME=MPI -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=COMPILE
cmake --find-package -DNAME=MPI -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=LINK
