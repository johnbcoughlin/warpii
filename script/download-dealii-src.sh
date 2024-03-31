#!/bin/bash

set -ex

source ../warpii.env

INSTALLDIR=$WARPIISOFT/deps/dealii

mkdir -p $INSTALLDIR
pushd $INSTALLDIR

mkdir -p dealii_src
pushd dealii_src

ZIPFILE="dealii-${DEALII_VERSION}.tar.gz"

curl -o dealii-${DEALII_VERSION}.tar.gz https://github.com/dealii/dealii/releases/download/v${DEALII_VERSION}/dealii-${DEALII_VERSION}.tar.gz

popd # dealii_src


popd # $INSTALLDIR
