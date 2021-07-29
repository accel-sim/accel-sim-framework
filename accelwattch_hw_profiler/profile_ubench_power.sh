#!/bin/bash

if [ "$GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN" != "1" ]; then
    echo "ERROR - Please run setup_environment before running this script"
    exit
fi

rate=100
temp=65
samples=600
sleep_time=30

SCRIPT_DIR=`pwd`
BINDIR="$ACCELSIM_ROOT/../accelwattch_benchmarks/microbenchmarks"
PROFILER="$SCRIPT_DIR/measureGpuPower"
UBENCH_FILE="$SCRIPT_DIR/ubench.cfg"

if [ -d $ACCELSIM_ROOT/../accelwattch_benchmarks/microbenchmarks ]; then
	cd $ACCELSIM_ROOT/../accelwattch_benchmarks
    ./extract_binaries.sh
fi
cd $SCRIPT_DIR

if [ -f $SCRIPT_DIR/$PROFILER ]; then
	make
fi

if [ -d $SCRIPT_DIR/ubench_power_reports ]; then
	rm -r $SCRIPT_DIR/ubench_power_reports
fi
mkdir $SCRIPT_DIR/ubench_power_reports

if [ -d $SCRIPT_DIR/ubench_profile_output ]; then
	rm -r $SCRIPT_DIR/ubench_profile_output
fi
mkdir $SCRIPT_DIR/ubench_profile_output
mkdir -p $SCRIPT_DIR/ubench_profile_output

for run in {1..5}
do
    while IFS= read -r bm
    do  
        if [ `echo $bm | awk '{print NF}'` == "4" ]; then
            bm_name=`echo $bm | awk '{print $1"_"$2"_"$3"_"$4}'`
        else
            bm_name=`echo $bm | awk '{print $1"_"$2}'`
        fi
        echo "Starting profiling of $bm_name"
        mkdir -p $SCRIPT_DIR/ubench_power_reports/$bm_name				
        $BINDIR/$bm & 
        $PROFILER -t $temp -r $rate -n $samples -o $SCRIPT_DIR/ubench_power_reports/$bm_name/run_$run.rpt >> $SCRIPT_DIR/ubench_profile_output/$bm_name.txt
        kill_name=`echo $bm | awk '{print $1}'`
        pid=`nvidia-smi | grep $kill_name | awk '{ print $5 }'`
        echo "Profiling concluded. Killing $bm_name with pid: $pid"
        kill -9 $pid
        
        if [ 'grep "WARNING: TEMPERATURE CUTTOFF NOT REACHED" $SCRIPT_DIR/ubench_profile_output/$bm_name.txt' ]; then
            echo "Heating up the GPU to >65C and rerunning kernel..."
            $BINDIR/mix6 2000000000 &
            sleep 20
            pid=`nvidia-smi | grep mix6 | awk '{ print $5 }'`
            kill -9 $pid
            $BINDIR/$bm & 
            $PROFILER -t $temp -r $rate -n $samples -o $SCRIPT_DIR/ubench_power_reports/$bm_name/run_$run.rpt >> $SCRIPT_DIR/ubench_profile_output/$bm_name.txt
            pid=`nvidia-smi | grep $kill_name | awk '{ print $5 }'`
            echo "Profiling concluded. Killing $bm_name with pid: $pid"
            kill -9 $pid
        fi
        
        echo "Sleeping..."
        sleep $sleep_time
    done < $UBENCH_FILE
done
