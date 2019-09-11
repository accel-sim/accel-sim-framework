start=`date +%s`
DATE=`date '+%Y-%m-%d--%H.%M.%S'`
TARNAME=volta.titanv.tgz
sudo nvidia-smi -i 2 -lgc 1417,1417
./run_hw.py -D 0 -B `cat ../job_launching/apps/all-apps.list` -R 10 -c
#./run_hw.py -D 0 -B polybench,shoc,deeplearning,microbench,cutlass,custom_apps -R 10
sudo nvidia-smi -i 2 -rgc
find ../../run_hw/ -name "*.out*" -exec rm {} \;
find ../../run_hw/ -type f -name "*output*" -exec rm {} \;
cp -r ../../run_hw/device-0 ../../run_hw/TITANV
cd ../../run_hw
tar zcvf $TARNAME ./TITANV
ssh dynamo.ecn.purdue.edu mv /home/dynamo/a/tgrogers/website/gpgpu-sim/hw_data/$TARNAME /home/dynamo/a/tgrogers/website/gpgpu-sim/hw_data/bak.$DATE.$TARNAME
scp $TARNAME dynamo.ecn.purdue.edu:~/website/gpgpu-sim/hw_data/
end=`date +%s`
runtime=$((end-start))
echo "Runtime: $runtime"
