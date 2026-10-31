FROM nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04
LABEL org.opencontainers.image.source=https://github.com/accel-sim/accel-sim-framework
LABEL org.opencontainers.image.description="Accel-Sim container with Ubuntu 24.04 with CUDA 13.2.1 and CUDNN, for CI runs only"

SHELL ["/bin/bash", "-c"]

WORKDIR /accel-sim

ENV CUDA_INSTALL_PATH=/usr/local/cuda
ENV PTXAS_CUDA_INSTALL_PATH=/usr/local/cuda
ENV BOOST_ROOT=/usr/include/boost
ENV PATH=$CUDA_INSTALL_PATH/bin:$PATH

ENV GPUAPPS_ROOT=/accel-sim/gpu-app-collection

# Lets the image be validated against a feature branch before that branch
# lands on dev:  docker build --build-arg GPU_APP_COLLECTION_BRANCH=<branch> .
ARG GPU_APP_COLLECTION_BRANCH=dev

RUN apt-get update && apt-get install -y wget build-essential xutils-dev bison zlib1g-dev flex \
      libglu1-mesa-dev git g++ libssl-dev libxml2-dev libboost-all-dev git g++ \
      libxml2-dev vim python3-setuptools python3-pip python3-venv cmake \
      libfreeimage3 libfreeimage-dev freeglut3-dev pkg-config \
      python3-doc python3-tk binfmt-support psmisc apt-utils \
      libzstd-dev && apt-get clean


# Create and activate a virtual environment, venv is needed because of PEP 668
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN pip3 install --upgrade pip
RUN pip3 install pyyaml plotly psutil

# For CI, only build the CI apps and pull the regression data
# Clone the gpu-app-collection repository
RUN git clone --recurse-submodules --branch ${GPU_APP_COLLECTION_BRANCH} --single-branch \
      https://github.com/accel-sim/gpu-app-collection.git
# Build the CI apps and pull the regression data
# Invoke make directly instead of test-build.sh, which hardcodes a serial
# `make -C src/ ci`. src/setup_environment is still required: it exports
# BINDIR/BINSUBDIR/SETENV and the per-CUDA-version gencode flags the
# suite makefiles read. -j8 propagates to every recursive $(MAKE),
# including cutlass_mini's cmake-generated makefiles.
RUN cd gpu-app-collection && source ./src/setup_environment && make -j8 -C src/ ci && bash get_regression_data.sh
