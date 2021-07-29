#!/bin/bash

if [ "$GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN" != "1" ]; then
    echo "ERROR - Please run setup_environment before running this script"
    exit
fi


if [ ! "${1}" == "volta_sass_sim" ] && [ ! "${1}" == "volta_sass_hybrid" ] && [ ! "${1}" == "volta_sass_hw" ] && [ ! "${1}" == "volta_ptx_sim" ]; then
	echo "Please provide accelwattch model; one of [volta_sass_sim, volta_sass_hybrid, volta_sass_hw, volta_ptx_sim]"
	echo "For example: ./util/accelwattch/check_job_status.sh volta_sass_sim"
	exit
fi

$ACCELSIM_ROOT/../util/job_launching/job_status.py -N ${1} -r accelwattch_runs/${1}
