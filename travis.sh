#!/bin/bash

if [ ! -n "$CUDA_INSTALL_PATH" ]; then
	echo "ERROR ** Install CUDA Toolkit and set CUDA_INSTALL_PATH.";
	exit;
fi

#Make the simulator
export PATH=$CUDA_INSTALL_PATH/bin:$PATH;
source ./gpu-simulator/setup_environment.sh
make -C ./gpu-simulator

#Get the pre-run trace files
rm -rf ./rodinia_2.0-ft
wget https://engineering.purdue.edu/tgrogers/accel-sim/traces/tesla-v100/latest/rodinia_2.0-ft.tgz
tar -xzvf rodinia_2.0-ft.tgz
rm rodinia_2.0-ft.tgz

#Run the tests on the trace
pip install psutil
./util/job_launching/run_simulations.py -C QV100 -B rodinia_2.0-ft -T ./rodinia_2.0-ft -N myTest

# Wait for them to finish
./util/job_launching/monitor_func_test.py -N myTest
