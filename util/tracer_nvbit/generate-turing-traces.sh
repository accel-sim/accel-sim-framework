# This could be cleaner - but at least this is automated now
# Script for generating all the turing traces for a Purdue lab machine.
# since the disc is a little limited here - dump them out to our cluster as they finish

TURING_DEVICE=1

#echo "Starting uBench"
#time ./run_hw_trace.py -B GPU_Microbenchmark -D $TURING_DEVICE 2>&1 | tee ubench.out
#mv ../../hw_run/traces/device-$TURING_DEVICE/ ../../hw_run/traces/GPU_Microbenchmark
#ssh tgrogers-raid.ecn.purdue.edu mkdir -p /scratch/tgrogers-disk01/a/$USER/tmp-turing-traces
#echo "Copying uBench"
#rsync --progress -avhe ssh ../../hw_run/traces/GPU_Microbenchmark tgrogers-raid.ecn.purdue.edu:/scratch/tgrogers-disk01/a/$USER/tmp-turing-traces/GPU_Microbenchmark
#rm -rf ../../hw_run/traces/GPU_Microbenchmark

#echo "Starting Rodinia-3.1"
#time ./run_hw_trace.py -B rodinia-3.1 -D $TURING_DEVICE 2>&1 | tee rodinia.out
#mv ../../hw_run/traces/device-$TURING_DEVICE/ ../../hw_run/traces/rodinia-3.1
#ssh tgrogers-raid.ecn.purdue.edu mkdir -p /scratch/tgrogers-disk01/a/$USER/tmp-turing-traces
#echo "Copying rodinia-3.1"
#rsync --progress -avhe ssh ../../hw_run/traces/rodinia-3.1 tgrogers-raid.ecn.purdue.edu:/scratch/tgrogers-disk01/a/$USER/tmp-turing-traces
#rm -rf ../../hw_run/traces/rodinia-3.1

echo "Starting parboil"
time ./run_hw_trace.py -B parboil -D $TURING_DEVICE 2>&1 | tee parboil.out
mv ../../hw_run/traces/device-$TURING_DEVICE/ ../../hw_run/traces/parboil
ssh tgrogers-raid.ecn.purdue.edu mkdir -p /scratch/tgrogers-disk01/a/$USER/tmp-turing-traces
echo "Copying parboil"
rsync --progress -avhe ssh ../../hw_run/traces/parboil tgrogers-raid.ecn.purdue.edu:/scratch/tgrogers-disk01/a/$USER/tmp-turing-traces
rm -rf ../../hw_run/traces/parboil


echo "Starting polybench"
time ./run_hw_trace.py -B polybench -D $TURING_DEVICE 2>&1 | tee polybench.out
mv ../../hw_run/traces/device-$TURING_DEVICE/ ../../hw_run/traces/polybench
ssh tgrogers-raid.ecn.purdue.edu mkdir -p /scratch/tgrogers-disk01/a/$USER/tmp-turing-traces
echo "Copying polybench"
rsync --progress -avhe ssh ../../hw_run/traces/polybench tgrogers-raid.ecn.purdue.edu:/scratch/tgrogers-disk01/a/$USER/tmp-turing-traces
rm -rf ../../hw_run/traces/polybench

echo "Starting cutlass_5_trace"
time ./run_hw_trace.py -B cutlass_5_trace -D $TURING_DEVICE 2>&1 | tee cutlass.out
mv ../../hw_run/traces/device-$TURING_DEVICE/ ../../hw_run/traces/cutlass_5_trace
ssh tgrogers-raid.ecn.purdue.edu mkdir -p /scratch/tgrogers-disk01/a/$USER/tmp-turing-traces
echo "Copying cutlass_5_trace"
rsync --progress -avhe ssh ../../hw_run/traces/cutlass_5_trace tgrogers-raid.ecn.purdue.edu:/scratch/tgrogers-disk01/a/$USER/tmp-turing-traces
rm -rf ../../hw_run/traces/cutlass_5_trace

echo "Starting Deepbench_nvidia_tencore,Deepbench_nvidia_normal"
time ./run_hw_trace.py -B Deepbench_nvidia_tencore,Deepbench_nvidia_normal -D $TURING_DEVICE 2>&1 | tee deepbench.out
mv ../../hw_run/traces/device-$TURING_DEVICE/ ../../hw_run/traces/Deepbench_nvidia
ssh tgrogers-raid.ecn.purdue.edu mkdir -p /scratch/tgrogers-disk01/a/$USER/tmp-turing-traces
echo "Copying Deepbench_nvidia"
rsync --progress -avhe ssh ../../hw_run/traces/Deepbench_nvidia tgrogers-raid.ecn.purdue.edu:/scratch/tgrogers-disk01/a/$USER/tmp-turing-traces
rm -rf ../../hw_run/traces/Deepbench_nvidia
