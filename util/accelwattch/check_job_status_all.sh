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
