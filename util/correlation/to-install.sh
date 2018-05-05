pip install --target=./python-package/ pyyaml
export PYTHONPATH=./python-package/:$PYTHONPATH
export LD_LIBRARY_PATH=$CUDA_INSTALL_PATH/lib64:$LD_LIBRARY_PATH
