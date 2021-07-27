#!/bin/bash
root_dir=`pwd`
curr_dir=`pwd | awk -F'/' '{print $NF}'`
if [ ! "${curr_dir}" == "accel-sim-framework" ]; then
	echo "Please run this script from accel-sim-framework root directory."
	exit
fi

if [ "${1}" == "" ]; then
	echo "Please provide path to traces directory."
	echo "For example: ./scripts_accelwattch/launch_jobs_all.sh /nfs-scratch/vkz4947/accel-sim-framework/hw_run/power_validation"
	exit
fi

if [ -d ${root_dir}/accelwattch_runs ] ; then
	rm -r ${root_dir}/accelwattch_runs
fi

${root_dir}/scripts_accelwattch/launch_jobs.sh volta_sass_sim ${1} ${2}
${root_dir}/scripts_accelwattch/launch_jobs.sh volta_sass_hw ${1} ${2}
${root_dir}/scripts_accelwattch/launch_jobs.sh volta_sass_hybrid ${1} ${2}
${root_dir}/scripts_accelwattch/launch_jobs.sh volta_ptx_sim ${1} ${2}
