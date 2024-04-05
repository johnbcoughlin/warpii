SHELL := /bin/bash

.PHONY: clean
clean:
	rm -rf builds

.PHONY: deps
deps:
	source warpii.env && cd script && $(MAKE) $(WARPIISOFT)/deps

build: builds/$(WARPII_BUILD_TYPE)

builds/Debug: deps src codes
	source warpii.env \
		&& cmake --fresh --preset clang-debug \
		&& cmake --build builds/Debug --parallel
	
builds/Release: deps src codes
	source warpii.env \
		&& cmake --fresh --preset clang-release \
		&& cmake --build builds/Release --parallel

test: build