#!/bin/bash
root_dir=`pwd`
curr_dir=`pwd | awk -F'/' '{print $NF}'`
if [ ! "${curr_dir}" == "accel-sim-framework" ]; then
	echo "Please run this script from accel-sim-framework root directory."
	exit
fi
${root_dir}/scripts_accelwattch/collect_power_reports.sh volta_sass_sim
${root_dir}/scripts_accelwattch/collect_power_reports.sh volta_sass_hw
${root_dir}/scripts_accelwattch/collect_power_reports.sh volta_sass_hybrid
${root_dir}/scripts_accelwattch/collect_power_reports.sh volta_ptx_sim
