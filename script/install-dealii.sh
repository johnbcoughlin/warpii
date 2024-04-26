#!/bin/bash

set -ex

./check-warpii-env.sh

DEALII_BUILD_MODE=RelWithDebInfo

DEALII_SRCDIR=$WARPIISOFT/deps/dealii/src/dealii-${DEALII_VERSION}

pushd $DEALII_SRCDIR

mkdir -p build-${DEALII_BUILD_MODE}
pushd build-${DEALII_BUILD_MODE}

cmake -DCMAKE_INSTALL_PREFIX=$WARPIISOFT/deps/dealii/dealii-${DEALII_VERSION}-${DEALII_BUILD_MODE} \
    -DDEAL_II_WITH_MPI=ON \
    -DDEAL_II_WITH_MUPARSER=ON \
    -DDEAL_II_ALLOW_AUTODETECTION=OFF \ # https://www.dealii.org/current/users/cmake_dealii.html#configureautoconf \
    ..

make -j4
make install

popd # build-${DEALII_BUILD_MODE}
popd # $DEALII_SRCDIR
