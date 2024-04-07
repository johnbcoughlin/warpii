SHELL := /bin/bash

$(WARPIISOFT)/deps: install-mpi $(WARPIISOFT)/deps/dealii

$(WARPIISOFT)/deps/dealii: install-mpi $(WARPIISOFT)/deps/dealii/dealii-$(DEALII_VERSION)-RelWithDebInfo

clean:
	rm -rf $(WARPIISOFT)/deps

### Targets for downloading, building and installing our main dependency, the deal.ii library

$(WARPIISOFT)/deps/dealii/src/dealii-$(DEALII_VERSION)/source: 
	mkdir -p $(WARPIISOFT)/deps/dealii/src \
		&& cd $(WARPIISOFT)/deps/dealii/src \
		&& curl -OL https://github.com/dealii/dealii/releases/download/v$(DEALII_VERSION)/dealii-$(DEALII_VERSION).tar.gz \
		&& gunzip dealii-$(DEALII_VERSION).tar.gz \
		&& tar -xf dealii-$(DEALII_VERSION).tar \
		&& rm dealii-$(DEALII_VERSION).tar

$(WARPIISOFT)/deps/dealii/dealii-$(DEALII_VERSION)-RelWithDebInfo: $(WARPIISOFT)/deps/dealii/src/dealii-$(DEALII_VERSION)/source install-dealii.sh
	./install-dealii.sh

### Targets for downloading, building and installing OpenMPI

.PHONY: install-mpi
install-mpi:
	./install-mpi.sh
