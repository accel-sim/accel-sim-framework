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

rate=100
temp=65
samples=600
sleep_time=30

SCRIPT_DIR=$ACCELSIM_ROOT/../util/accelwattch/accelwattch_hw_profiler
BINDIR="$ACCELSIM_ROOT/../util/accelwattch/accelwattch_benchmarks/validation"

if [ ! -d $ACCELSIM_ROOT/../util/accelwattch/accelwattch_benchmarks/data_dirs ]; then
	echo "Please create a data_dirs/ directory containing the datasets at $ACCELSIM_ROOT/../util/accelwattch/accelwattch_benchmarks/data_dirs"
    exit 1
fi

if [ ! -d $ACCELSIM_ROOT/../util/accelwattch/accelwattch_benchmarks/validation ]; then
	echo "Please create a validation/ directory with binaries at $ACCELSIM_ROOT/../util/accelwattch/accelwattch_benchmarks/validation"
    exit 1
fi

if [ ! "${1}" == "volta" ] && [ ! "${1}" == "turing" ] && [ ! "${1}" == "pascal" ]; then
    echo "Please enter the GPU architecture; one of: [volta,turing,pascal]. For example: ./profile_validation_power.sh volta 1"
    exit 1
fi

if [ "${2}" == "" ]; then
    echo "Please enter the GPU device ID;  For example: ./profile_validation_power.sh volta 1"
    exit 1
fi


cd $SCRIPT_DIR
RODINIA_DATADIR="$ACCELSIM_ROOT/../util/accelwattch/accelwattch_benchmarks/data_dirs/cuda/rodinia/3.1"
PARBOIL_DATADIR="$ACCELSIM_ROOT/../util/accelwattch/accelwattch_benchmarks/data_dirs/parboil"
PROFILER="$SCRIPT_DIR/measureGpuPower"
BENCH_FILE="$SCRIPT_DIR/validation_${1}.cfg"
DEVID=${2}
backprop_k1_r="$BINDIR/backprop_k1 65536"
backprop_k2_r="$BINDIR/backprop_k2 65536"
binomialOptions_k1_r="$BINDIR/binomialOptions_k1"
btree_k1_r="$BINDIR/btree_k1 file $RODINIA_DATADIR/b+tree-rodinia-3.1/data/mil.txt command $RODINIA_DATADIR/b+tree-rodinia-3.1/data/command.txt"
btree_k2_r="$BINDIR/btree_k2 file $RODINIA_DATADIR/b+tree-rodinia-3.1/data/mil.txt command $RODINIA_DATADIR/b+tree-rodinia-3.1/data/command.txt"
cudaTensorCoreGemm_k1_r="$BINDIR/cudaTensorCoreGemm_k1"
dct8x8_k1_r="$BINDIR/dct8x8_k1"
dct8x8_k2_r="$BINDIR/dct8x8_k2"
cutlass_k1_r="$BINDIR/cutlass_perf_test_k1 --seed=2020 --dist=0  --m=2560 --n=16 --k=2560 --kernels=wmma_gemm_nn  --iterations=500000000 --providers=cutlass"
cutlass_k2_r="$BINDIR/cutlass_perf_test_k2 --seed=2020 --dist=0  --m=4096 --n=128 --k=4096 --kernels=wmma_gemm_nn  --iterations=500000000 --providers=cutlass"
cutlass_k3_r="$BINDIR/cutlass_perf_test_k3 --seed=2020 --dist=0  --m=2560 --n=512 --k=2560 --kernels=wmma_gemm_nn  --iterations=500000000 --providers=cutlass"
fastWalshTransform_k1_r="$BINDIR/fastWalshTransform_k1"
fastWalshTransform_k2_r="$BINDIR/fastWalshTransform_k2"
histogram_k1_r="$BINDIR/histogram_k1"
hotspot_k1_r="$BINDIR/hotspot_k1 1024 2 2 $RODINIA_DATADIR/hotspot-rodinia-3.1/data/temp_1024 $RODINIA_DATADIR/hotspot-rodinia-3.1/data/power_1024 output.out"
kmeans_k1_r="$BINDIR/kmeans_k1 -o -i $RODINIA_DATADIR/kmeans-rodinia-3.1/data/819200.txt"
mergeSort_k1_r="$BINDIR/mergeSort_k1"
mergeSort_k2_r="$BINDIR/mergeSort_k2"
parboil_mriq_k1_r="$BINDIR/parboil_mriq_k1 -i $PARBOIL_DATADIR/parboil-mri-q/data/large/input/64_64_64_dataset.bin -o 64_64_64_dataset.out"
parboil_sad_k1_r="$BINDIR/parboil_sad_k1 -i $PARBOIL_DATADIR/parboil-sad/data/large/input/reference.bin,$PARBOIL_DATADIR/parboil-sad/data/large/input/frame.bin -o out.bin"
parboil_sgemm_k1_r="$BINDIR/parboil_sgemm_k1 -i $PARBOIL_DATADIR/parboil-sgemm/data/medium/input/matrix1.txt,$PARBOIL_DATADIR/parboil-sgemm/data/medium/input/matrix2t.txt,$PARBOIL_DATADIR/parboil-sgemm/data/medium/input/matrix2t.txt -o matrix3.txt"
pathfinder_k1_r="$BINDIR/pathfinder_k1 100000 100 20 "
quasirandomGenerator_k1_r="$BINDIR/quasirandomGenerator_k1"
quasirandomGenerator_k2_r="$BINDIR/quasirandomGenerator_k2"
sobolQRNG_k1_r="$BINDIR/sobolQRNG_k1"
srad_v1_k1_r="$BINDIR/srad_v1_k1 100 0.5 502 458"

if [ ! -f $PROFILER ]; then
	make -C $ACCELSIM_ROOT/../util/accelwattch/accelwattch_hw_profiler
fi

if [ -d $SCRIPT_DIR/validation_power_reports ]; then
	rm -r $SCRIPT_DIR/validation_power_reports
fi
mkdir $SCRIPT_DIR/validation_power_reports

if [ -d $SCRIPT_DIR/validation_profile_output ]; then
	rm -r $SCRIPT_DIR/validation_profile_output
fi
mkdir $SCRIPT_DIR/validation_profile_output

cd $SCRIPT_DIR/validation_profile_output
cp $ACCELSIM_ROOT/../util/accelwattch/accelwattch_benchmarks/data_dirs/dct8x8/data/* .

for run in {1..5}
do
    while IFS= read -r bm
    do  
        bm_name="${bm}_r"
        echo "Starting profiling of ${bm} "
        mkdir -p $SCRIPT_DIR/validation_power_reports/$bm
        ${!bm_name} >> $SCRIPT_DIR/validation_profile_output/$bm_output.txt &
        $PROFILER -t $temp -r $rate -n $samples -d $DEVID -o $SCRIPT_DIR/validation_power_reports/$bm/run_$run.rpt >> $SCRIPT_DIR/validation_profile_output/$bm.txt
        
        if [ $bm == "cutlass_k1" ] || [ $bm == "cutlass_k2" ] || [ $bm == "cutlass_k3" ]; then
            pid=`nvidia-smi | grep "cutlass_perf_test" | awk '{ print $5 }'`
        else
            pid=`nvidia-smi | grep $bm | awk '{ print $5 }'`
        fi
        echo "Profiling concluded. Killing $bm with pid: $pid"
        kill -9 $pid
        
        if cat $SCRIPT_DIR/validation_profile_output/$bm.txt | grep -q "WARNING: TEMPERATURE CUTTOFF NOT REACHED"; then
            echo "Heating up the GPU to >65C and rerunning kernel..."
            $BINDIR/backprop_k1 65536 &
            sleep 20
            pid=`nvidia-smi | grep backprop_k1 | awk '{ print $5 }'`
            kill -9 $pid
            ${!bm_name} >> $SCRIPT_DIR/validation_profile_output/$bm_output.txt &
            $PROFILER -t $temp -r $rate -n $samples -d $DEVID -o $SCRIPT_DIR/validation_power_reports/$bm/run_$run.rpt >> $SCRIPT_DIR/validation_profile_output/$bm.txt
            if [ $bm == "cutlass_k1" ] || [ $bm == "cutlass_k2" ] || [ $bm == "cutlass_k3" ]; then
                pid=`nvidia-smi | grep "cutlass_perf_test" | awk '{ print $5 }'`
            else
                pid=`nvidia-smi | grep $bm | awk '{ print $5 }'`
            fi
            echo "Profiling concluded. Killing $bm with pid: $pid"
            kill -9 $pid
        fi
        
        echo "Sleeping..."
        sleep $sleep_time
    done < $BENCH_FILE
done
