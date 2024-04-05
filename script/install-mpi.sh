#!/bin/bash

set -ex

./check-warpii-env.sh

set +e
cmake --find-package -DNAME=MPI -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=EXIST
set -e
if [[ $? -eq 0 ]]; then
    echo "MPI exists, no need to install it."
else
    if [[ "$unamestr" == 'Darwin' ]]; then
        brew install --cc=clang openmpi
    else
        echo "We don't know how to install openmpi automatically on this system."
    fi
fi

# Check that cmake will be able to find, compile and link against MPI.
cmake --find-package -DNAME=MPI -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=COMPILE
cmake --find-package -DNAME=MPI -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=LINK

rm -rf CMakeFiles
