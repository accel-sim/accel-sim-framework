# This bash script formats GPGPU-Sim using clang-format
THIS_DIR="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
clang-format -i ${THIS_DIR}/*.cu
clang-format -i ${THIS_DIR}/*.h
clang-format -i ${THIS_DIR}/traces-processing/*.cpp
