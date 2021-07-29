#!/bin/bash
root_dir=`pwd`
if [ "$GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN" != "1" ]; then
    echo "ERROR - Please run setup_environment before running this script. For Example source $ACCELSIM_ROOT/gpu-simulator/setup_environment.sh"
    exit
fi

runs_dir="$ACCELSIM_ROOT/../hw_run/device-0/11.0"
if [ -d ${runs_dir} ] ; then
	rm -r ${runs_dir}
fi

$ACCELSIM_ROOT/../util/hw_stats/run_hw.py -B rodinia-3.1_validation_hw,parboil_validation,cuda_samples_11.0_validation,cutlass_5_trace_validation,cudaTensorCoreGemm_validation --collect other_stats --nsight_profiler --disable_nvprof

$ACCELSIM_ROOT/../accelwattch_hw_profiler/gen_hw_perf_csv.py -d $ACCELSIM_ROOT/../hw_run/device-0/11.0/

mv ${root_dir}/hw_perf.csv $ACCELSIM_ROOT/../accelwattch_hw_profiler/hw_perf.csv
echo "y" | cp $ACCELSIM_ROOT/../accelwattch_hw_profiler/hw_perf.csv $ACCELSIM_ROOT/gpgpu-sim/configs/tested-cfgs/SM7_QV100/hw_perf.csv
echo "Done"

