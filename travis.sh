#!/bin/bash

export PATH=$CUDA_INSTALL_PATH/bin:$PATH;
source ./gpu-simulator/setup_environment.sh
make -C ./gpu-simulator"
