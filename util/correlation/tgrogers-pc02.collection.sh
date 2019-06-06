start=`date +%s`
DATE=`date '+%Y-%m-%d--%H.%M.%S'`
sudo nvidia-smi -i 1 -lgc 1132,1132
./run_hw.py -D 0 -B `cat ../job_launching/apps/all-apps.list` -R 10 -c
sudo nvidia-smi -i 1 -rgc
cp -r ../../run_hw/device-0 ../../run_hw/QUADRO-V100
cd ../../run_hw
tar zcvf quadro.v100.cycle.tgz ./QUADRO-V100
#ssh dynamo.ecn.purdue.edu mv /home/dynamo/a/tgrogers/website/gpgpu-sim/hw_data/pascal.titanx.cycle.tgz /home/dynamo/a/tgrogers/website/gpgpu-sim/hw_data/bak.$DATE.pascal.titanx.cycle.tgz
scp quadro.v100.cycle.tgz dynamo.ecn.purdue.edu:~/website/gpgpu-sim/hw_data/
end=`date +%s`
runtime=$((end-start))
echo "Runtime: $runtime"
