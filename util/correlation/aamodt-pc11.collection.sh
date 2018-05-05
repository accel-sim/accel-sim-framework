start=`date +%s`
DATE=`date '+%Y-%m-%d--%H.%M.%S'`
./run_hw.py -D 0 -B `cat ../job_launching/apps/all-apps.list` -R 5
cp -r ../../run_hw/device-0 ../../run_hw/1080TI_PASCAL
cd ../../run_hw
tar zcvf pascal.1080ti.cycle.tgz ./1080TI_PASCAL
ssh dynamo.ecn.purdue.edu mv /home/dynamo/a/tgrogers/website/gpgpu-sim/hw_data/pascal.1080ti.cycle.tgz /home/dynamo/a/tgrogers/website/gpgpu-sim/hw_data/bak.$DATE.pascal.1080ti.cycle.tgz
scp pascal.1080ti.cycle.tgz dynamo.ecn.purdue.edu:~/website/gpgpu-sim/hw_data/
end=`date +%s`
runtime=$((end-start))
echo "Runtime: $runtime"
