SHELL := /bin/bash

clean:
	rm -rf builds

builds/Debug: src codes
	cmake --fresh --preset clang-debug \
		&& cmake --build builds/Debug
	
builds/Release: src codes
	cmake --fresh --preset clang-release

