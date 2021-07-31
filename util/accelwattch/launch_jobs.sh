#!/bin/bash

if [ "$GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN" != "1" ]; then
    echo "ERROR - Please run setup_environment before running this script"
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
elif [ "${1}" == "pascal_sass_sim" ]; then
	config="TITANX-Accelwattch_SASS_SIM"
elif [ "${1}" == "pascal_ptx_sim" ]; then
	config="TITANX-Accelwattch_PTX_SIM"
elif [ "${1}" == "turing_sass_sim" ]; then
	config="RTX2060_S-Accelwattch_SASS_SIM"
elif [ "${1}" == "turing_ptx_sim" ]; then
	config="RTX2060_S-Accelwattch_PTX_SIM"
else
	echo "Please provide accelwattch model; one of [volta_sass_sim, volta_sass_hybrid, volta_sass_hw, volta_ptx_sim, pascal_sass_sim, pascal_ptx_sim, turing_sass_sim, turing_ptx_sim ]"
	echo "For example: ./util/accelwattch/launch_jobs.sh volta_sass_sim /nfs-scratch/vkandiah/accelwattch_traces"
	exit
fi

if [ "${2}" == "" ]; then
	if [ ! "${1}" == "volta_ptx_sim" ] && [ ! "${1}" == "turing_ptx_sim" ] && [ ! "${1}" == "pascal_ptx_sim" ]; then
		echo "Please provide path to root accelwattch traces directory."
		echo "For example: ./util/accelwattch/launch_jobs.sh volta_sass_sim /nfs-scratch/vkandiah/accelwattch_traces"
		exit
	fi
fi


mkdir -p $ACCELSIM_ROOT/../accelwattch_runs
runs_dir=${1}
if [ -d $ACCELSIM_ROOT/../accelwattch_runs/${runs_dir} ] ; then
	rm -r $ACCELSIM_ROOT/../accelwattch_runs/${runs_dir}
fi

if [ "${1}" == "volta_ptx_sim" ] || [ "${1}" == "turing_ptx_sim" ]; then
	$ACCELSIM_ROOT/../util/job_launching/run_simulations.py -B rodinia-3.1_validation_ptx,parboil_validation,cuda_samples_11.0_validation,cudaTensorCoreGemm_validation -C ${config} -N ${runs_dir} -r $ACCELSIM_ROOT/../accelwattch_runs/${runs_dir} ${3}
elif [ "${1}" == "pascal_ptx_sim" ]; then
	$ACCELSIM_ROOT/../util/job_launching/run_simulations.py -B rodinia-3.1_validation_ptx,parboil_validation,cuda_samples_11.0_validation -C ${config} -N ${runs_dir} -r $ACCELSIM_ROOT/../accelwattch_runs/${runs_dir} ${3}
elif [ "${1}" == "volta_sass_hybrid" ] || [ "${1}" == "volta_sass_hw" ]; then
	$ACCELSIM_ROOT/../util/job_launching/run_simulations.py -B rodinia-3.1_validation_hw,parboil_validation,cuda_samples_11.0_validation,cutlass_5_trace_validation,cudaTensorCoreGemm_validation -a -C ${config} -T ${2}/accelwattch_volta_traces -N ${runs_dir} -r $ACCELSIM_ROOT/../accelwattch_runs/${runs_dir} ${3}
elif [ "${1}" == "volta_sass_sim" ]; then
	$ACCELSIM_ROOT/../util/job_launching/run_simulations.py -B rodinia-3.1_validation,parboil_validation,cuda_samples_11.0_validation,cutlass_5_trace_validation,cudaTensorCoreGemm_validation -C ${config} -T ${2}/accelwattch_volta_traces -N ${runs_dir} -r $ACCELSIM_ROOT/../accelwattch_runs/${runs_dir} ${3}
elif [ "${1}" == "turing_sass_sim" ]; then
	$ACCELSIM_ROOT/../util/job_launching/run_simulations.py -B rodinia-3.1_validation,parboil_validation,cuda_samples_11.0_validation,cutlass_5_trace_validation,cudaTensorCoreGemm_validation -C ${config} -T ${2}/accelwattch_turing_traces -N ${runs_dir} -r $ACCELSIM_ROOT/../accelwattch_runs/${runs_dir} ${3}
elif [ "${1}" == "pascal_sass_sim" ]; then
	$ACCELSIM_ROOT/../util/job_launching/run_simulations.py -B rodinia-3.1_validation,parboil_validation,cuda_samples_11.0_validation -C ${config} -T ${2}/accelwattch_pascal_traces -N ${runs_dir} -r $ACCELSIM_ROOT/../accelwattch_runs/${runs_dir} ${3}
else
	echo "Please provide accelwattch model; one of [volta_sass_sim, volta_sass_hybrid, volta_sass_hw, volta_ptx_sim, pascal_sass_sim, pascal_ptx_sim, turing_sass_sim, turing_ptx_sim ]"
	echo "For example: ./util/accelwattch/launch_jobs.sh volta_sass_sim /nfs-scratch/vkandiah/accelwattch_volta_traces"
	exit
fi
