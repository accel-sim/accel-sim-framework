#!/bin/bash
# Copyright (c) 2020 Timothy Rogers, Purdue University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
# Neither the name of The University of British Columbia nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

export ACCELSIM_SETUP_ENVIRONMENT_WAS_RUN=
export ACCELSIM_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"

if [ $# = '1' ] ;
then
    export ACCELSIM_CONFIG=$1
else
    export ACCELSIM_CONFIG=release
fi

if [ ! -d "$ACCELSIM_ROOT/gpgpu-sim" ] ; then
    git clone https://github.com/accel-sim/gpgpu-sim_distribution.git $ACCELSIM_ROOT/gpgpu-sim
    git -C $ACCELSIM_ROOT/gpgpu-sim/ checkout dev
fi

source $ACCELSIM_ROOT/gpgpu-sim/setup_environment $ACCELSIM_CONFIG

export ACCELSIM_SETUP_ENVIRONMENT_WAS_RUN=1
