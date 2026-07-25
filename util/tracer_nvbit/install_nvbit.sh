#!/bin/bash
export BASH_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"

# NVBit 1.8 (CUDA 13.2 headers) is required for CUDA-13-era drivers (>=59x).
# NVBit 1.7.6 supports only driver <=575.xx and produces corrupt register
# operands on newer drivers (e.g. 596.xx), which crashes post-traces-processing.
NVBIT_VERSION="1.8"
rm -rf $BASH_ROOT/nvbit_release
mkdir -p "$BASH_ROOT/nvbit_release"
wget https://github.com/NVlabs/NVBit/releases/download/v${NVBIT_VERSION}/nvbit-Linux-x86_64-${NVBIT_VERSION}.tar.bz2
tar -xf nvbit-Linux-x86_64-${NVBIT_VERSION}.tar.bz2 -C "$BASH_ROOT/nvbit_release" --strip-components=1
rm nvbit-Linux-x86_64-${NVBIT_VERSION}.tar.bz2
