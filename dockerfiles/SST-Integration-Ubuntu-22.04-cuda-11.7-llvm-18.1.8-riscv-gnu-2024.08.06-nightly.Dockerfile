FROM ubuntu:22.04

SHELL ["/bin/bash", "-c"]

WORKDIR /accel-sim
ADD . /accel-sim

ENV CUDA_INSTALL_PATH /usr/local/cuda-11.7
ENV PTXAS_CUDA_INSTALL_PATH /usr/local/cuda-11.7
ENV GPUAPPS_ROOT /accel-sim/gpu-app-collection
ENV LLVM_INSTALL_PATH /accel-sim/llvm-install
ENV RISCV_TOOLCHAIN_INSTALL_PATH /accel-sim/riscv-gnu-install

# Install CUDA
RUN apt-get update \
&& apt-get install -y wget build-essential xutils-dev bison zlib1g-dev flex \
      libglu1-mesa-dev git g++ libssl-dev libxml2-dev libboost-all-dev git g++ \
      libxml2-dev vim python-setuptools python3-pip cmake \
&& apt-get clean \
&& pip3 install pyyaml plotly psutil \
&& wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run \
&& sh cuda_11.7.0_515.43.04_linux.run --silent --toolkit \
&& rm cuda_11.7.0_515.43.04_linux.run \
&& rm -rf /usr/local/cuda-11.7/nsight-compute-2022.2.0 \
&& rm -rf /usr/local/cuda-11.7/nsight-systems-2022.1.3

# Build LLVM 18.1.8
RUN git clone https://github.com/llvm/llvm-project.git \
&& mkdir llvm-install \
&& cd llvm-project \
&& git checkout llvmorg-18.1.8 \
&& mkdir build && cd build \
&& cmake -DLLVM_TARGETS_TO_BUILD="RISCV;X86;NVPTX" -DLLVM_DEFAULT_TARGET_TRIPLE=riscv64-unknown-linux-gnu \
         -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;lld" -DCMAKE_INSTALL_PREFIX=$LLVM_INSTALL_PATH ../llvm \
&& cmake --build . -j4 \
&& cmake --build . --target install \
&& cd .. && cd .. \
&& rm -rf llvm-project

# Build RISCV GNU Toolchain 2024.08.06.nightly
RUN apt-get update && apt-get -y install autoconf automake autotools-dev curl python3 python3-pip libmpc-dev libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo gperf libtool patchutils bc zlib1g-dev libexpat-dev ninja-build git cmake libglib2.0-dev libslirp-dev
RUN git clone https://github.com/riscv-collab/riscv-gnu-toolchain.git \
&& mkdir riscv-gnu-install \
&& cd riscv-gnu-toolchain \
&& git checkout 2024.08.06 \
&& ./configure --prefix=$RISCV_TOOLCHAIN_INSTALL_PATH \
&& make linux -j4 \
&& cd .. \
&& rm -rf riscv-gnu-toolchain

# Setup GPU app collection in SST mode
# Also remove prebuilt x86 binaries
# Also remove all data except for rodinia-2.0
RUN export PATH=$CUDA_INSTALL_PATH/bin:$PATH \
&& git clone https://github.com/accel-sim/gpu-app-collection \
&& cd gpu-app-collection \
&& git pull \
&& git checkout sst_support \
&& source ./src/setup_environment sst \
&& make -j -C ./src data \
&& mv ./data_dirs/cuda/rodinia/2.0-ft . \
&& rm -rf ./data_dirs \
&& mkdir -p ./data_dirs/cuda/rodinia/ \
&& mv ./2.0-ft ./data_dirs/cuda/rodinia/ \
&& ls ./data_dirs/cuda/rodinia/ \
&& rm gpucomputingsdk_4.2.9_linux.run \
&& rm -rf 4.2
