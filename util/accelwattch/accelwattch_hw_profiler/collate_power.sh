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

SCRIPT_DIR=$ACCELSIM_ROOT/../util/accelwattch/accelwattch_hw_profiler

output_folder=$SCRIPT_DIR/collated_power
cd $SCRIPT_DIR
if [ ! "${1}" == "validation_power_reports" ] && [ ! "${1}" == "ubench_power_reports" ]; then
    echo "Please enter a correct power reports directory, one of: [ubench_power_reports,validation_power_reports]. For example: ./collate_power.sh validation_power_reports volta"
	exit 1
fi

if [ ! "${2}" == "volta" ] && [ ! "${2}" == "turing" ] && [ ! "${2}" == "pascal" ]; then
    echo "Please enter the GPU architecture; one of: [volta,turing,pascal]. For example: ./collate_power.sh validation_power_reports volta"
	exit 1
fi

if [ -e "$SCRIPT_DIR/${1}" ] && [ -d "$SCRIPT_DIR/${1}" ]; then
	if [ -d "$output_folder" ]; then
		rm -r "$output_folder"
	fi
	mkdir $output_folder
	for bm in `ls $SCRIPT_DIR/${1}`
	do
		for data in `ls $SCRIPT_DIR/${1}/$bm`
		do
			power=`cat $SCRIPT_DIR/${1}/$bm/$data | awk -F'Power draw = ' '{print $2}' | awk -F' W' '{print $1}'`
			echo $power >> $output_folder/$bm.rpt
		done
	done
	python gen_hw_power_csv.py $output_folder
    if [  "${1}" == "validation_power_reports" ]; then
        mv hw_power_results.csv hw_power_validation_${2}.csv
    else
        mv hw_power_results.csv hw_power_ubench_${2}.csv
    fi
    rm -r $output_folder
else
	echo "Please enter a correct power reports directory. Example: ./collate_power.sh validation_power_reports volta"
	exit 1
fi
