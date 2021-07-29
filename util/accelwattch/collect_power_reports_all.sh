#!/bin/bash

if [ "$GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN" != "1" ]; then
    echo "ERROR - Please run setup_environment before running this script"
    exit
fi

$ACCELSIM_ROOT/../util/accelwattch/collect_power_reports.sh volta_sass_sim
$ACCELSIM_ROOT/../util/accelwattch/collect_power_reports.sh volta_sass_hw
$ACCELSIM_ROOT/../util/accelwattch/collect_power_reports.sh volta_sass_hybrid
$ACCELSIM_ROOT/../util/accelwattch/collect_power_reports.sh volta_ptx_sim
