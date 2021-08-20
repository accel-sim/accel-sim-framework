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
	$ACCELSIM_ROOT/../util/job_launching/run_simulations.py -B rodinia-3.1_validation_ptx,parboil_validation,cuda_samples_11.0_validation,cudaTensorCoreGemm_validation -C ${config} -N ${runs_dir} -l local -r $ACCELSIM_ROOT/../accelwattch_runs/${runs_dir} ${3}
elif [ "${1}" == "pascal_ptx_sim" ]; then
	$ACCELSIM_ROOT/../util/job_launching/run_simulations.py -B rodinia-3.1_validation_ptx,parboil_validation,cuda_samples_11.0_validation -C ${config} -N ${runs_dir} -l local -r $ACCELSIM_ROOT/../accelwattch_runs/${runs_dir} ${3}
elif [ "${1}" == "volta_sass_hybrid" ] || [ "${1}" == "volta_sass_hw" ]; then
	$ACCELSIM_ROOT/../util/job_launching/run_simulations.py -B rodinia-3.1_validation_hw,parboil_validation,cuda_samples_11.0_validation,cutlass_5_trace_validation,cudaTensorCoreGemm_validation -a -C ${config} -T ${2}/accelwattch_volta_traces -N ${runs_dir} -l local -r $ACCELSIM_ROOT/../accelwattch_runs/${runs_dir} ${3}
elif [ "${1}" == "volta_sass_sim" ]; then
	$ACCELSIM_ROOT/../util/job_launching/run_simulations.py -B rodinia-3.1_validation,parboil_validation,cuda_samples_11.0_validation,cutlass_5_trace_validation,cudaTensorCoreGemm_validation -C ${config} -T ${2}/accelwattch_volta_traces -N ${runs_dir} -l local -r $ACCELSIM_ROOT/../accelwattch_runs/${runs_dir} ${3}
elif [ "${1}" == "turing_sass_sim" ]; then
	$ACCELSIM_ROOT/../util/job_launching/run_simulations.py -B rodinia-3.1_validation,parboil_validation,cuda_samples_11.0_validation,cutlass_5_trace_validation,cudaTensorCoreGemm_validation -C ${config} -T ${2}/accelwattch_turing_traces -N ${runs_dir} -l local -r $ACCELSIM_ROOT/../accelwattch_runs/${runs_dir} ${3}
elif [ "${1}" == "pascal_sass_sim" ]; then
	$ACCELSIM_ROOT/../util/job_launching/run_simulations.py -B rodinia-3.1_validation,parboil_validation,cuda_samples_11.0_validation -C ${config} -T ${2}/accelwattch_pascal_traces -N ${runs_dir} -l local -r $ACCELSIM_ROOT/../accelwattch_runs/${runs_dir} ${3}
else
	echo "Please provide accelwattch model; one of [volta_sass_sim, volta_sass_hybrid, volta_sass_hw, volta_ptx_sim, pascal_sass_sim, pascal_ptx_sim, turing_sass_sim, turing_ptx_sim ]"
	echo "For example: ./util/accelwattch/launch_jobs.sh volta_sass_sim /nfs-scratch/vkandiah/accelwattch_volta_traces"
	exit
fi
