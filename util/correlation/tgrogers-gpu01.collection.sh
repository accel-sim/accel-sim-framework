start=`date +%s`
DATE=`date '+%Y-%m-%d--%H.%M.%S'`
TARNAME=tesla.v100.tgz
#./run_hw.py -D 0 -B `cat ../job_launching/apps/all-apps.list` -R 10 -c
#./run_hw.py -D 0 -B GPU_Microbenchmark,Deepbench_nvidia,sdk-4.2-scaled,rodinia-3.1,parboil,polybench,cutlass_5_trace -R 10
./run_hw.py -D 7 -B cutlass_5_trace -R 10 -N
find ../../run_hw/ -name "*.out*" -exec rm {} \;
find ../../run_hw/ -type f -name "*output*" -exec rm {} \;
find ../../run_hw/ -type f -name "*reference.dat*" -exec rm {} \;
find ../../run_hw/ -type f -name "*result.txt*" -exec rm {} \;
cp -r ../../run_hw/device-7 ../../run_hw/TESLA-V100
cd ../../run_hw
tar zcvf $TARNAME ./TESLA-V100
ssh dynamo.ecn.purdue.edu mv /home/dynamo/a/tgrogers/website/gpgpu-sim/hw_data/$TARNAME /home/dynamo/a/tgrogers/website/gpgpu-sim/hw_data/bak.$DATE.$TARNAME
scp $TARNAME dynamo.ecn.purdue.edu:~/website/gpgpu-sim/hw_data/
end=`date +%s`
runtime=$((end-start))
echo "Runtime: $runtime"
