#!/bin/bash

if [ "$GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN" != "1" ]; then
    echo "ERROR - Please run setup_environment before running this script"
    exit
fi


report_dir=$ACCELSIM_ROOT/../accelwattch_power_reports

if [ ! "${1}" == "volta_sass_sim" ] && [ ! "${1}" == "volta_sass_hybrid" ] && [ ! "${1}" == "volta_sass_hw" ] && [ ! "${1}" == "volta_ptx_sim" ]; then
	echo "Please provide accelwattch model; one of [volta_sass_sim, volta_sass_hybrid, volta_sass_hw, volta_ptx_sim]"
	echo "For example: ./util/accelwattch/collect_power_reports.sh volta_sass_sim"
	exit
fi
config=${1}
power_file=accelwattch_power_report.log
mkdir -p ${report_dir}
runs_dir=$ACCELSIM_ROOT/../accelwattch_runs/${config}
power_dir=${report_dir}/${config}
if [ -d ${power_dir} ] ; then
	rm -r ${power_dir}
fi
mkdir ${power_dir}
for bench in `ls ${runs_dir}`
do	
	if [ ! ${bench} == "gpgpu-sim-builds" ]; then
		for inp in `ls ${runs_dir}/${bench}/`
		do
			bench_dir=${runs_dir}/${bench}/${inp}/*
			if [ -f ${bench_dir}/${power_file} ] ; then
				#if [ "${bench}" == "cutlass_perf_test" ]; then	
					#cp ${bench_dir}/${power_file} ${power_dir}/${bench}_${inp}.log 
				#else
				cp ${bench_dir}/${power_file} ${power_dir}/${bench}.log 
				#fi
			else
				echo "Warning: No Accelwattch power report in ${bench_dir}."
			fi
		done
	fi
done
