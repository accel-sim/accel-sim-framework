#!/bin/bash
# Copyright (c) 2025 Timothy Rogers, Purdue University
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

#   Different branches of Accel-Sim should have different values here
#   For development, we use our internal repo and the dev branch
#       Ideally, when we release, it should be based off a GPGPU-Sim release.
export GPGPUSIM_REPO="${GPGPUSIM_REPO:=https://github.com/accel-sim/gpgpu-sim_distribution.git}"
export GPGPUSIM_BRANCH="${GPGPUSIM_BRANCH:=dev}"

# Help the user out by setting the default CUDA_INSTALL_PATH, if it is not already set
if [ -z "$CUDA_INSTALL_PATH" ]; then
    DEFAULT_CUDA_PATH="/usr/local/cuda"
    if [ ! -d "$DEFAULT_CUDA_PATH" ]; then
        echo "Error: Default CUDA installation directory ($DEFAULT_CUDA_PATH) does not exist."
        echo "Please set CUDA_INSTALL_PATH to your CUDA installation directory."
        return 1
    fi
    echo "CUDA_INSTALL_PATH is not set, setting it to $DEFAULT_CUDA_PATH"
    export CUDA_INSTALL_PATH=$DEFAULT_CUDA_PATH
fi

if [ $# = '1' ] ;
then
    export ACCELSIM_CONFIG=$1
else
    export ACCELSIM_CONFIG=release
fi

ln -sf $ACCELSIM_ROOT/build/$ACCELSIM_CONFIG/compile_commands.json $ACCELSIM_ROOT/../

# If we can't find an already set version of GPGPU-Sim, then pull one locally using the repos specificed above
if [ -z "$GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN" -o ! -d "$GPGPUSIM_ROOT" ]; then
    echo "No \$GPGPUSIM_ROOT, testing for local folder in: \"$ACCELSIM_ROOT/gpgpu-sim\""
    if [ ! -d "$ACCELSIM_ROOT/gpgpu-sim" ] ; then
        echo "No \$ACCELSIM_ROOT/gpgpu-sim."
        if [ ! -z "$PS1" ]; then
            read -e -p "Please specify the repo you want to sync for GPGPU-Sim (default is $GPGPUSIM_REPO):" user_repo
        fi
        if [ -z $user_repo ] ; then
            user_repo=$GPGPUSIM_REPO
        fi

        if [ ! -z "$PS1" ]; then
            read -e -p "Please specify the branch for GPGPU-Sim you would like to use (default is $GPGPUSIM_BRANCH):" user_branch
        fi
        if [ -z $user_branch ] ; then
            user_branch=$GPGPUSIM_BRANCH
        fi
        git clone $user_repo $ACCELSIM_ROOT/gpgpu-sim
        git -C $ACCELSIM_ROOT/gpgpu-sim/ checkout $user_branch
    else
        echo "Found $ACCELSIM_ROOT/gpgpu-sim, using existing local location. Not sycning anything."
    fi
    source $ACCELSIM_ROOT/gpgpu-sim/setup_environment $ACCELSIM_CONFIG || return 1
else
    source $GPGPUSIM_ROOT/setup_environment $ACCELSIM_CONFIG || return 1
fi

if [ ! -d "$ACCELSIM_ROOT/extern/pybind11" ] ; then
    git clone --depth 1 -b master https://github.com/pybind/pybind11.git $ACCELSIM_ROOT/extern/pybind11
fi

export PYTHONPATH=$ACCELSIM_ROOT/build/$ACCELSIM_CONFIG:$PYTHONPATH

echo "Using GPGPU-Sim in $GPGPUSIM_ROOT"
#echo "If that is not the intended behavior, then run: \"unset GPGPUSIM_ROOT; unset GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN\"."

echo "Accel-Sim setup succeeded."
export ACCELSIM_SETUP_ENVIRONMENT_WAS_RUN=1
