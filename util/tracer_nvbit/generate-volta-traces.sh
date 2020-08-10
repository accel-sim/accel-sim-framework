# This could be cleaner - but at least this is automated now
# Script for generating all the volta traces in the Accel-Sim paper on a DGX-1

echo "Starting uBench"
time ./run_hw_trace.py -B GPU_Microbenchmark -D 0 2>&1 > ubench.out & 
ubenchPID=$!
echo "Starting Rodinia-3.1"
time ./run_hw_trace.py -B rodinia-3.1 -D 1 2>&1 > rodinia.out &
rodiniaPID=$!
echo "Starting parboil"
time ./run_hw_trace.py -B parboil -D 2 2>&1 > parboil.out &
parboilPID=$!
echo "Starting polybench"
time ./run_hw_trace.py -B polybench -D 3 2>&1 > polybench.out &
polybenchPID=$!
# 4/5 being used for MLPerf
echo "Starting cutlass_5_trace"
time ./run_hw_trace.py -B cutlass_5_trace -D 6 2>&1 > cutlass.out &
cutlassPID=$!
echo "Starting Deepbench_nvidia_tencore,Deepbench_nvidia_normal"
time ./run_hw_trace.py -B Deepbench_nvidia_tencore,Deepbench_nvidia_normal -D 7 2>&1 > deepbench.out & 
deepbenchPID=$!

echo "Wating for uBench"
wait $ubenchPID
print "uBench Done"
cat ubench.out
mv ../../hw_run/traces/device-0/ ../../hw_run/traces/GPU_Microbenchmark

echo "Wating for rodinia-3.1"
wait $rodiniaPID
print "rodinia Done"
cat rodinia.out
mv ../../hw_run/traces/device-1/ ../../hw_run/traces/rodinia-3.1

echo "Wating for parboil"
wait $parboilPID
print "parboil Done"
cat parboil.out
mv ../../hw_run/traces/device-2/ ../../hw_run/traces/parboil

echo "Wating for polybench"
wait $polybenchPID
print "poybench Done"
cat polybench.out
mv ../../hw_run/traces/device-3/ ../../hw_run/traces/polybench

echo "Wating for cutlass"
wait $cutlassPID
print "cutlass Done"
cat cutlass.out
mv ../../hw_run/traces/device-6/ ../../hw_run/traces/cutlass_5_trace

echo "Wating for deepbench"
wait $deepbenchPID
print "cutlass Done"
cat deepbench.out
mv ../../hw_run/traces/device-7/ ../../hw_run/traces/Deepbench_nvidia
