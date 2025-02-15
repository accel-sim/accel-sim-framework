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
echo "Starting rodinia_2.0-ft"
time ./run_hw_trace.py -B rodinia_2.0-ft -D 4 2>&1 > rodinia_2.0-ft.out &
rodinia20ftPID=$!
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
mv ../../hw_run/traces/device-0/ ../../hw_run/traces/ubench
cd ../../hw_run/traces/
tar  -I pigz -cf ubench.tgz ./ubench/
cd -

echo "Wating for rodinia_2.0-ft"
wait $rodinia20ftPID
print "rodinia_2.0-ft Done"
cat rodinia_2.0-ft.out
mv ../../hw_run/traces/device-4/ ../../hw_run/traces/rodinia_2.0-ft
cd ../../hw_run/traces/
tar -I pigz -cf rodinia_2.0-ft.tgz ./rodinia_2.0-ft/
cd -

echo "Wating for rodinia-3.1"
wait $rodiniaPID
print "rodinia Done"
cat rodinia.out
mv ../../hw_run/traces/device-1/ ../../hw_run/traces/rodinia-3.1
cd ../../hw_run/traces/
tar -I pigz -cf rodinia-3.1.tgz ./rodinia-3.1/
cd -

echo "Wating for parboil"
wait $parboilPID
print "parboil Done"
cat parboil.out
mv ../../hw_run/traces/device-2/ ../../hw_run/traces/parboil
cd ../../hw_run/traces/
tar -I pigz -cf parboil.tgz ./parboil/
cd -

echo "Wating for polybench"
wait $polybenchPID
print "poybench Done"
cat polybench.out
mv ../../hw_run/traces/device-3/ ../../hw_run/traces/polybench
cd ../../hw_run/traces/
tar -I pigz -cf polybench.tgz ./polybench/
cd -

echo "Wating for cutlass"
wait $cutlassPID
print "cutlass Done"
cat cutlass.out
mv ../../hw_run/traces/device-6/ ../../hw_run/traces/cutlass
cd ../../hw_run/traces/
tar -I pigz -cf cutlass.tgz ./cutlass/
cd -

echo "Wating for deepbench"
wait $deepbenchPID
print "cutlass Done"
cat deepbench.out
mv ../../hw_run/traces/device-7/ ../../hw_run/traces/deepbench
cd ../../hw_run/traces/
tar -I pigz -cf deepbench.tgz ./deepbench/
du -sh | tee trace.summary.txt
cd -
