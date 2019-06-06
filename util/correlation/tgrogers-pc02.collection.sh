start=`date +%s`
DATE=`date '+%Y-%m-%d--%H.%M.%S'`
TARNAME=quadro.v100.cycle.tgz
sudo nvidia-smi -i 1 -lgc 1132,1132
./run_hw.py -D 0 -B `cat ../job_launching/apps/all-apps.list` -R 10 -c
sudo nvidia-smi -i 1 -rgc
find ../../run_hw/ -name "*.out*" -exec rm {} \;
find ../../run_hw/ -type f -name "*output*" -exec rm {} \;
cp -r ../../run_hw/device-0 ../../run_hw/QUADRO-V100
cd ../../run_hw
tar zcvf $TARNAME ./QUADRO-V100
ssh dynamo.ecn.purdue.edu mv /home/dynamo/a/tgrogers/website/gpgpu-sim/hw_data/$TARNAME /home/dynamo/a/tgrogers/website/gpgpu-sim/hw_data/bak.$DATE.$TARNAME
scp $TARNAME dynamo.ecn.purdue.edu:~/website/gpgpu-sim/hw_data/
end=`date +%s`
runtime=$((end-start))
echo "Runtime: $runtime"
