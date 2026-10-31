FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04
LABEL org.opencontainers.image.source=https://github.com/accel-sim/accel-sim-framework
LABEL org.opencontainers.image.description="Accel-Sim container with Ubuntu 24.04 with CUDA 12.8.0 and CUDNN, for CI runs only"

SHELL ["/bin/bash", "-c"]

WORKDIR /accel-sim

ENV CUDA_INSTALL_PATH=/usr/local/cuda
ENV PTXAS_CUDA_INSTALL_PATH=/usr/local/cuda
ENV BOOST_ROOT=/usr/include/boost
ENV PATH=$CUDA_INSTALL_PATH/bin:$PATH

ENV GPUAPPS_ROOT=/accel-sim/gpu-app-collection

RUN apt-get update && apt-get install -y wget build-essential xutils-dev bison zlib1g-dev flex \
      libglu1-mesa-dev git g++ libssl-dev libxml2-dev libboost-all-dev git g++ \
      libxml2-dev vim python3-setuptools python3-pip python3-venv cmake \
      libfreeimage3 libfreeimage-dev freeglut3-dev pkg-config \
      python3-doc python3-tk binfmt-support psmisc apt-utils && apt-get clean 


# Create and activate a virtual environment, venv is needed because of PEP 668
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN pip3 install --upgrade pip
RUN pip3 install pyyaml plotly psutil

# For CI, only build the CI apps and pull the regression data
# Clone the gpu-app-collection repository
RUN git clone --recurse-submodules https://github.com/accel-sim/gpu-app-collection.git
# Build the CI apps and pull the regression data
RUN cd gpu-app-collection && bash test-build.sh ci && bash get_regression_data.sh