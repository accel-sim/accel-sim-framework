start=`date +%s`
DATE=`date '+%Y-%m-%d--%H.%M.%S'`
#./run_hw.py -D 0 -B GPU_Microbenchmark -R 10 -N
#./run_hw.py -D 0 -B `cat ../job_launching/apps/all-apps.list` -R 10 -c
#sudo nvidia-smi -i 2 -lgc 1417,1417

# QUADRO V100
sudo nvidia-smi -i 2 -lgc 1132,1132
./run_hw.py -D 0 -B Deepbench_nvidia -R 5 -N
sudo nvidia-smi -i 2 -rgc

# PASCAL TITANX
#sudo nvidia-smi -i 1 -lgc 1417,1417
#./run_hw.py -D 1 -B GPU_Microbenchmark,Deepbench_nvidia,sdk-4.2-scaled,rodinia-3.1,parboil,polybench,cutlass_5_trace -R 5
#./run_hw.py -D 1 -B parboil -R 5
#sudo nvidia-smi -i 1 -rgc

#./run_hw.py -D 0 -B GPU_Microbenchmark -R 10
#./run_hw.py -D 0 -B `cat ../job_launching/apps/correlation-apps.list` -R 10 -c
#./run_hw.py -D 0 -B polybench,shoc,deeplearning,microbench,cutlass,custom_apps -R 10

find ../../run_hw/ -name "*.out*" -exec rm {} \;
find ../../run_hw/ -type f -name "*output*" -exec rm {} \;
find ../../run_hw/ -type f -name "*reference.dat*" -exec rm {} \;
find ../../run_hw/ -type f -name "*result.txt*" -exec rm {} \;

# QUADRO V100
TARNAME=quadro.v100.cycle.tgz
cp -r ../../run_hw/device-0 ../../run_hw/QUADRO-V100
cd ../../run_hw
tar zcvf $TARNAME ./QUADRO-V100
ssh dynamo.ecn.purdue.edu mv /home/dynamo/a/tgrogers/website/gpgpu-sim/hw_data/$TARNAME /home/dynamo/a/tgrogers/website/gpgpu-sim/hw_data/bak.$DATE.$TARNAME
scp $TARNAME dynamo.ecn.purdue.edu:~/website/gpgpu-sim/hw_data/

# PASCAL TITAN X
#TARNAME=pascal.titanx.cycle.tgz
#cp -r ../../run_hw/device-1 ../../run_hw/TITAN-X-PASCAL
#cd ../../run_hw
#tar zcvf $TARNAME ./TITAN-X-PASCAL
#ssh dynamo.ecn.purdue.edu mv /home/dynamo/a/tgrogers/website/gpgpu-sim/hw_data/$TARNAME /home/dynamo/a/tgrogers/website/gpgpu-sim/hw_data/bak.$DATE.$TARNAME
#scp $TARNAME dynamo.ecn.purdue.edu:~/website/gpgpu-sim/hw_data/

end=`date +%s`
runtime=$((end-start))
echo "Runtime: $runtime"
