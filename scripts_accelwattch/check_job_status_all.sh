#!/bin/bash
root_dir=`pwd`
curr_dir=`pwd | awk -F'/' '{print $NF}'`
if [ ! "${curr_dir}" == "accel-sim-framework" ]; then
	echo "Please run this script from accel-sim-framework root directory."
	exit
fi
echo "Job status for volta_sass_sim:"
${root_dir}/scripts_accelwattch/check_job_status.sh volta_sass_sim
echo "Job status for volta_sass_hw:"
${root_dir}/scripts_accelwattch/check_job_status.sh volta_sass_hw
echo "Job status for volta_sass_hybrid:"
${root_dir}/scripts_accelwattch/check_job_status.sh volta_sass_hybrid
echo "Job status for volta_ptx_sim:"
${root_dir}/scripts_accelwattch/check_job_status.sh volta_ptx_sim