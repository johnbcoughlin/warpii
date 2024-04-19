SHELL := /bin/bash

# Need to touch CMakeCache.txt in the case that nothing in it changed, to avoid rebuilding next time
builds/$(WARPII_CMAKE_PRESET)/configured: CMakePresets.json CMakeLists.txt
	source warpii.env \
		&& rm -rf builds/$(WARPII_CMAKE_PRESET)/configured \
		&& cmake --preset $(WARPII_CMAKE_PRESET) \
		&& touch builds/$(WARPII_CMAKE_PRESET)/configured

build: src codes builds/$(WARPII_CMAKE_PRESET)/configured
	source warpii.env && cmake --build --preset $(WARPII_CMAKE_PRESET) --parallel

test: build
	source warpii.env && cd builds/$(WARPII_CMAKE_PRESET)/test \
		&& ctest --output-on-failure

.PHONY: install-dealii
install-dealii:
	source warpii.env && cd script \
		&& $(MAKE) $(WARPIISOFT)/deps/dealii

.PHONY: clean
clean:
	rm -rf builds

