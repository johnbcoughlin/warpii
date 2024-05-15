FROM --platform=linux/amd64 ubuntu:22.04

# Install essential dependencies
RUN apt-get update \
    && apt-get install -y software-properties-common \
    && apt-get install -y build-essential cmake gdb git-core curl

# Install OpenMPI
RUN apt-get install -y openmpi-bin libopenmpi-dev

# Install deal.ii
RUN apt-add-repository ppa:ginggs/deal.ii-9.5.1-backports && apt-get install -y libdeal.ii-dev

# Install doxygen
RUN apt-get install -y doxygen

RUN cd $HOME \
    && git clone https://github.com/johnbcoughlin/warpii.git \
    && cd warpii \
    && git checkout d6c5e62ccdcdbd39156c2958df2d6a382229f9c9

ENV WARPIISOFT="/root/warpiisoft"

RUN cmake --version

ENTRYPOINT ["/bin/bash"]
