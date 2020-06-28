# This bash script formats GPGPU-Sim using clang-format
THIS_DIR="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
clang-format -i ${THIS_DIR}/trace-driven/*.h
clang-format -i ${THIS_DIR}/trace-driven/*.cc
clang-format -i ${THIS_DIR}/trace-driven/ISA_Def/*.h
