FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

SHELL ["/bin/bash", "-c"]

WORKDIR /accel-sim

ENV CUDA_INSTALL_PATH=/usr/local/cuda
ENV PTXAS_CUDA_INSTALL_PATH=/usr/local/cuda
ENV BOOST_ROOT=/usr/include/boost
ENV PATH=$CUDA_INSTALL_PATH/bin:$PATH

ENV GPUAPPS_ROOT=/accel-sim/gpu-app-collection

# NOTE: `bc` and `libzstd-dev` added to the upstream list -- both are load-bearing
# for the tracer build, which upstream does at runtime from the mounted source:
#   bc          -> tracer_tool/Makefile shells out to it for the nvcc>=11.7 ARCH check
#   libzstd-dev -> tracer_tool/traces-processing links -lzstd
RUN apt-get update && apt-get install -y \
    wget build-essential xutils-dev bison zlib1g-dev flex \
    libglu1-mesa-dev git g++ libssl-dev libxml2-dev libboost-all-dev \
    vim python3-setuptools python3-pip python3-venv cmake \
    libfreeimage3 libfreeimage-dev freeglut3-dev pkg-config \
    python3-doc python3-tk python3.12-venv python3.12-doc \
    binfmt-support psmisc apt-utils gdb curl bash-completion \
    bc libzstd-dev && \
    apt-get clean

# Create and activate a virtual environment, venv is needed because of PEP 668
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN pip3 install --upgrade pip
RUN pip3 install pyyaml plotly psutil

#get Nsys
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update --allow-insecure-repositories && apt update && apt install -y --no-install-recommends gnupg wget \
    && mkdir -p /etc/apt/keyrings
RUN wget -qO - https://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64/7fa2af80.pub | tee /etc/apt/keyrings/nvidia.asc
RUN echo "deb [signed-by=/etc/apt/keyrings/nvidia.asc] http://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64 /" | tee /etc/apt/sources.list.d/nvidia.list
RUN apt-get update --allow-insecure-repositories
RUN apt install  -y nsight-systems-cli --allow-unauthenticated

# ---------------------------------------------------------------------------
# vLLM + LLM tracing stack.
#
# Everything above is the upstream accel-sim Dockerfile unchanged (bar the two
# packages noted). Nothing below changes the upstream workflow: no source is
# baked in, WORKDIR stays /accel-sim, and there is no ENTRYPOINT override -- the
# nvidia/cuda base entrypoint already does `exec "$@"`, so you pass a command in
# and the mounted script builds the tracer, exactly as .github/scripts do.
#
# This is the single biggest layer in the image (~10 GB).
# ---------------------------------------------------------------------------
RUN pip3 install --no-cache-dir vllm transformers nvtx
RUN pip3 install --no-cache-dir monai diffusers accelerate