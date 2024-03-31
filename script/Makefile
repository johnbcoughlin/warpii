SHELL := /bin/bash

all: $(WARPIISOFT)/deps/dealii/dealii-Debug

clean:
	rm -rf $(WARPIISOFT)/deps

### Targets for downloading, building and installing our main dependency, the deal.ii library

$(WARPIISOFT)/deps/dealii/dealii-$(DEALII_VERSION): 
	mkdir -p $(WARPIISOFT)/deps/dealii \
		&& cd $(WARPIISOFT)/deps/dealii \
		&& curl -OL https://github.com/dealii/dealii/releases/download/v$(DEALII_VERSION)/dealii-$(DEALII_VERSION).tar.gz \
		&& gunzip dealii-$(DEALII_VERSION).tar.gz \
		&& tar -xf dealii-$(DEALII_VERSION).tar \
		&& rm dealii-$(DEALII_VERSION).tar

$(WARPIISOFT)/deps/dealii/dealii-Debug: $(WARPIISOFT)/deps/dealii/dealii-$(DEALII_VERSION)
	./install-dealii.sh --debug

$(WARPIISOFT)/deps/dealii/dealii-Release: warpiisoft-link
	./install-dealii.sh --release

