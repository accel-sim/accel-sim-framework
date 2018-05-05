start=`date +%s`
DATE=`date '+%Y-%m-%d--%H.%M.%S'`
./run_hw.py -D 0 -B `cat ../job_launching/apps/all-apps.list`
cp -r ../../run_hw/device-0 ../../run_hw/TESLA-P100
cd ../../run_hw
tar zcvf pascal.tesla.p100.cycles.tgz ./TESLA-P100
ssh dynamo.ecn.purdue.edu mv /home/dynamo/a/tgrogers/website/gpgpu-sim/hw_data/pascal.tesla.p100.cycles.tgz /home/dynamo/a/tgrogers/website/gpgpu-sim/hw_data/bak.$DATE.pascal.tesla.p100.cycles.tgz
scp pascal.tesla.p100.cycles.tgz dynamo.ecn.purdue.edu:~/website/gpgpu-sim/hw_data/
end=`date +%s`
runtime=$((end-start))
echo "Runtime: $runtime"
