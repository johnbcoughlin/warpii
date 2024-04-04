SHELL := /bin/bash

clean:
	rm -rf builds

builds/Debug: src codes
	cmake --fresh --preset clang-debug \
		&& cmake --build builds/Debug --parallel
	
builds/Release: src codes
	cmake --fresh --preset clang-release \
		&& cmake --build builds/Release --parallel

test: builds/$(WARPII_BUILD_TYPE)
