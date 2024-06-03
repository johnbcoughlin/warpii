#!/bin/bash
#
# This test exercises the example Makefile that we generate as a starter
# for compiling Extensions. It should exercise both the include and linking
# flags in the Makefile, then test that the main function runs to completion.
#
# We don't care to test any actual functionality of Extensions in this test since
# that will be exercised by tests that compile under the gtest suite.
#

set -ex

# Set up a clean temporary directory to work in.
TMP=$(mktemp -d)

# We've been passed the Makefile in question by the ctest harness, grab its path here.
MAKEFILE=$1

pushd $TMP

# Create a minimal main.cc 
cat >main.cc <<EOF
#include <deal.II/base/mpi.h>
#include "src/five_moment/extension.h"

class TestExtension : warpii::five_moment::Extension<2> {};

int main(int argc, char** argv) {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    auto ext = std::make_shared<TestExtension>();
    std::cout << "Done with initialization" << std::endl;
}
EOF

cp $MAKEFILE Makefile
cat Makefile
make
./main | grep "Done with initialization"

popd
