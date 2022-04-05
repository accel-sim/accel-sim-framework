#! /bin/sh

THIS_DIR="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
clang-format -i ${THIS_DIR}/ubench/*/*/*.cu
clang-format -i ${THIS_DIR}/ubench/*/*/*.h
clang-format -i ${THIS_DIR}/ubench/*/*/*.cpp
clang-format -i ${THIS_DIR}/hw_def/*/*.h
clang-format -i ${THIS_DIR}/hw_def/common/*.h
