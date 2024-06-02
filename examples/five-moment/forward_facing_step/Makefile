MPI_INCLUDE = /opt/homebrew/Cellar/open-mpi/5.0.2_1/include
MPI_LIB = /opt/homebrew/Cellar/open-mpi/5.0.2_1/lib
DEALII_INCLUDE = ${WARPIISOFT}/deps/dealii/dealii-9.5.1-RelWithDebInfo/include
DEALII_EXT_INCLUDE = ${WARPIISOFT}/deps/dealii/dealii-9.5.1-RelWithDebInfo/include/deal.II/bundled
DEALII_LIB = ${WARPIISOFT}/deps/dealii/dealii-9.5.1-RelWithDebInfo/lib

WARPII_INCLUDE = /Users/jack/src/warpii
WARPII_LIB = /Users/jack/src/warpii/builds/macos-debug/src

#CFLAGS = -Wall -Wextra -Wpedantic -Werror -g -std=gnu++17 -arch arm64 -Wno-deprecated-declarations
CFLAGS = -DDEBUG -isystem /Users/jack/warpiisoft/deps/dealii/dealii-9.5.1-RelWithDebInfo/include -isystem /Users/jack/warpiisoft/deps/dealii/dealii-9.5.1-RelWithDebInfo/include/deal.II/bundled -isystem /opt/homebrew/Cellar/open-mpi/5.0.2_1/include -Wall -Wextra -Wpedantic -Werror -g -std=gnu++17 -arch arm64 -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX13.3.sdk -pedantic -Wall -Wextra -Wmissing-braces -Woverloaded-virtual -Wpointer-arith -Wsign-compare -Wsuggest-override -Wswitch -Wsynth -Wwrite-strings -Wno-deprecated-declarations -Wno-psabi -Wfloat-conversion -Qunused-arguments -Wno-unsupported-friend -Wno-pass-failed -Wno-unused-local-typedefs -openmp-simd -ftemplate-depth=1024 -ffp-exception-behavior=strict -Og -ggdb -Wa,--compress-debug-sections


INCLUDES = -I${DEALII_INCLUDE} -I${DEALII_EXT_INCLUDE} -I${MPI_INCLUDE} -I${WARPII_INCLUDE}
LIB_DIRS = -L${DEALII_LIB} -L${MPI_LIB} -L${WARPII_LIB}

all: main

main: main.cc
	/usr/bin/clang++ ${CFLAGS} ${INCLUDES} main.cc -o main ${LIB_DIRS} -ldeal_II -llibwarpii -lmpi
