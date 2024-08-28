#!/bin/bash

# Copyright (c) 2018-2021, Vijay Kandiah, Junrui Pan, Mahmoud Khairy, Scott Peverelle, Timothy Rogers, Tor M. Aamodt, Nikos Hardavellas
# Northwestern University, Purdue University, The University of British Columbia
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer;
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution;
# 3. Neither the names of Northwestern University, Purdue University,
#    The University of British Columbia nor the names of their contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

if [ "$GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN" != "1" ]; then
    echo "ERROR - Please run setup_environment before running this script"
    exit
fi


report_dir=$ACCELSIM_ROOT/../accelwattch_power_reports

if [ ! "${1}" == "volta_sass_sim" ] && [ ! "${1}" == "volta_sass_hybrid" ] && [ ! "${1}" == "volta_sass_hw" ] && [ ! "${1}" == "volta_ptx_sim" ] && [ ! "${1}" == "pascal_sass_sim" ] && [ ! "${1}" == "pascal_ptx_sim" ] && [ ! "${1}" == "turing_sass_sim" ]&& [ ! "${1}" == "turing_ptx_sim" ]; then
	echo "Please provide accelwattch model; one of [volta_sass_sim, volta_sass_hybrid, volta_sass_hw, volta_ptx_sim, pascal_sass_sim, pascal_ptx_sim, turing_sass_sim, turing_ptx_sim ]"
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
				cp ${bench_dir}/${power_file} ${power_dir}/${bench}.log
			else
				echo "Warning: No Accelwattch power report in ${bench_dir}."
			fi
		done
	fi
done
