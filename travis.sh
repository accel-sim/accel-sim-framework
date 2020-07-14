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
rm -rf ./hw_run/rodinia_2.0-ft
./get-accel-sim-traces.py -a tesla-v100/rodinia_2.0-ft
tar -xzvf ./hw_run/rodinia_2.0-ft.tgz -C ./hw_run
rm hw_run/rodinia_2.0-ft.tgz

#Run the tests on the trace
./util/job_launching/run_simulations.py -C QV100 -B rodinia_2.0-ft -T ./hw_run/rodinia_2.0-ft/9.1 -N myTest

# Wait for them to finish
./util/job_launching/monitor_func_test.py -v -N myTest
