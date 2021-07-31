#!/bin/bash

if [ "$GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN" != "1" ]; then
    echo "ERROR - Please run setup_environment before running this script"
    exit
fi

if [ "${1}" == "" ]; then
	echo "Please provide path to root accelwattch traces directory"
	echo "For example: ./util/accelwattch/launch_jobs_all.sh /nfs-scratch/vkandiah/accelwattch_traces"
	exit
fi

if [ -d $ACCELSIM_ROOT/../accelwattch_runs ] ; then
	rm -r $ACCELSIM_ROOT/../accelwattch_runs
fi

$ACCELSIM_ROOT/../util/accelwattch/launch_jobs.sh volta_sass_sim ${1} ${2}
$ACCELSIM_ROOT/../util/accelwattch/launch_jobs.sh volta_sass_hw ${1} ${2}
$ACCELSIM_ROOT/../util/accelwattch/launch_jobs.sh volta_sass_hybrid ${1} ${2}
$ACCELSIM_ROOT/../util/accelwattch/launch_jobs.sh volta_ptx_sim ${1} ${2}
$ACCELSIM_ROOT/../util/accelwattch/launch_jobs.sh pascal_sass_sim ${1} ${2}
$ACCELSIM_ROOT/../util/accelwattch/launch_jobs.sh pascal_ptx_sim ${1} ${2}
$ACCELSIM_ROOT/../util/accelwattch/launch_jobs.sh turing_sass_sim ${1} ${2}
$ACCELSIM_ROOT/../util/accelwattch/launch_jobs.sh turing_ptx_sim ${1} ${2}
