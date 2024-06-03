FILENAME = main

all: $(FILENAME)

$(FILENAME): $(FILENAME).o
	/usr/bin/clang++ $(FILENAME).o -o $(FILENAME) -L/Users/jack/warpiisoft/deps/dealii/dealii-9.5.1-RelWithDebInfo/lib -L/Users/jack/src/warpii/builds/macos-debug/src -ldeal_II -llibwarpii -L/opt/homebrew/Cellar/open-mpi/5.0.2_1/lib -lmpi

$(FILENAME).o: $(FILENAME).cc
	 /usr/bin/clang++ -DDEBUG -I/Users/jack/src/warpii -isystem /Users/jack/warpiisoft/deps/dealii/dealii-9.5.1-RelWithDebInfo/include -isystem /Users/jack/warpiisoft/deps/dealii/dealii-9.5.1-RelWithDebInfo/include/deal.II/bundled -isystem /opt/homebrew/Cellar/open-mpi/5.0.2_1/include -Wall -Wextra -Wpedantic -Werror -g -std=gnu++17 -arch arm64 -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX13.3.sdk -pedantic -Wall -Wextra -Wmissing-braces -Woverloaded-virtual -Wpointer-arith -Wsign-compare -Wsuggest-override -Wswitch -Wsynth -Wwrite-strings -Wno-deprecated-declarations -Wno-psabi -Wfloat-conversion -Qunused-arguments -Wno-unsupported-friend -Wno-pass-failed -Wno-unused-local-typedefs -openmp-simd -ftemplate-depth=1024 -ffp-exception-behavior=strict -Og -ggdb -Wa,--compress-debug-sections -o $(FILENAME).o -c $(FILENAME).cc

clean:
	rm $(FILENAME) $(FILENAME).o

.PHONY: all clean
