# This bash script formats GPGPU-Sim using clang-format
THIS_DIR="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
clang-format -i ${THIS_DIR}/trace-driven/*.h --style=file:${THIS_DIR}/.clang-format
clang-format -i ${THIS_DIR}/trace-driven/*.cc --style=file:${THIS_DIR}/.clang-format
clang-format -i ${THIS_DIR}/ISA_Def/*.h --style=file:${THIS_DIR}/.clang-format
clang-format -i ${THIS_DIR}/trace-parser/*.h --style=file:${THIS_DIR}/.clang-format
clang-format -i ${THIS_DIR}/trace-parser/*.cc --style=file:${THIS_DIR}/.clang-format
