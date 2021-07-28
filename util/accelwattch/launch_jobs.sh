#!/bin/bash
root_dir=`pwd`
curr_dir=`pwd | awk -F'/' '{print $NF}'`
if [ ! "${curr_dir}" == "accel-sim-framework" ]; then
	echo "Please run this script from accel-sim-framework root directory."
	exit
fi

config=""
if [ "${1}" == "volta_sass_sim" ]; then
	config="QV100-Accelwattch_SASS_SIM"
elif [ "${1}" == "volta_sass_hybrid" ]; then
	config="QV100-Accelwattch_SASS_HYBRID"
elif [ "${1}" == "volta_sass_hw" ]; then
	config="QV100-Accelwattch_SASS_HW"
elif [ "${1}" == "volta_ptx_sim" ]; then
	config="QV100-Accelwattch_PTX_SIM"
else
	echo "Please provide accelwattch model; one of [volta_sass_sim, volta_sass_hybrid, volta_sass_hw, volta_ptx_sim]"
	echo "For example: ./util/accelwattch/launch_jobs.sh volta_sass_sim /nfs-scratch/vkandiah/accel-sim-framework/hw_run/power_validation"
	exit
fi

if [ "${2}" == "" ]; then
	if [ ! "${1}" == "volta_ptx_sim" ]; then
		echo "Please provide path to traces directory."
		echo "For example: ./util/accelwattch/launch_jobs.sh volta_sass_sim /nfs-scratch/vkandiah/accel-sim-framework/hw_run/power_validation"
		exit
	fi
fi

source ${root_dir}/gpu-simulator/setup_environment.sh
mkdir -p ${root_dir}/accelwattch_runs
runs_dir=${1}
if [ -d ${root_dir}/accelwattch_runs/${runs_dir} ] ; then
	rm -r ${root_dir}/accelwattch_runs/${runs_dir}
fi

if [ "${1}" == "volta_ptx_sim" ]; then
	${root_dir}/util/job_launching/run_simulations.py -B rodinia-3.1_validation,parboil_validation,cuda_samples_11.0_validation,cudaTensorCoreGemm_validation -C ${config} -N ${runs_dir} -r accelwattch_runs/${runs_dir} ${3}
elif [ "${1}" == "volta_sass_hybrid" ] || [ "${1}" == "volta_sass_hw" ]; then
	${root_dir}/util/job_launching/run_simulations.py -B rodinia-3.1_validation,parboil_validation,cuda_samples_11.0_validation,cutlass_5_trace_validation,cudaTensorCoreGemm_validation -a -C ${config} -T ${2} -N ${runs_dir} -r accelwattch_runs/${runs_dir} ${3}
else
	${root_dir}/util/job_launching/run_simulations.py -B rodinia-3.1_validation,parboil_validation,cuda_samples_11.0_validation,cutlass_5_trace_validation,cudaTensorCoreGemm_validation -C ${config} -T ${2} -N ${runs_dir} -r accelwattch_runs/${runs_dir} ${3}
fi
