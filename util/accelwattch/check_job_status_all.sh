#!/bin/bash

if [ "$GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN" != "1" ]; then
    echo "ERROR - Please run setup_environment before running this script"
    exit
fi


echo "Job status for volta_sass_sim:"
$ACCELSIM_ROOT/../util/accelwattch/check_job_status.sh volta_sass_sim
echo "Job status for volta_sass_hw:"
$ACCELSIM_ROOT/../util/accelwattch/check_job_status.sh volta_sass_hw
echo "Job status for volta_sass_hybrid:"
$ACCELSIM_ROOT/../util/accelwattch/check_job_status.sh volta_sass_hybrid
echo "Job status for volta_ptx_sim:"
$ACCELSIM_ROOT/../util/accelwattch/check_job_status.sh volta_ptx_sim
echo "Job status for turing_sass_sim:"
$ACCELSIM_ROOT/../util/accelwattch/check_job_status.sh turing_sass_sim
echo "Job status for turing_ptx_sim:"
$ACCELSIM_ROOT/../util/accelwattch/check_job_status.sh turing_ptx_sim
echo "Job status for pascal_sass_sim:"
$ACCELSIM_ROOT/../util/accelwattch/check_job_status.sh pascal_sass_sim
echo "Job status for pascal_ptx_sim:"
$ACCELSIM_ROOT/../util/accelwattch/check_job_status.sh pascal_ptx_sim