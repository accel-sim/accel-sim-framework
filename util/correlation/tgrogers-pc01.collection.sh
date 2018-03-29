DATE=`date '+%Y-%m-%d--%H.%M.%S'`
./run_hw.py -D 0 -c -B `cat ../job_launching/apps/all-apps.list`
cp -r ../../run_hw/device-0 ../../run_hw/TITAN-X-PASCAL
cd ../../run_hw
tar zcvf pascal.titanx.cycle.tgz ./TITAN-X-PASCAL
ssh dynamo.ecn.purdue.edu mv /home/dynamo/a/tgrogers/website/gpgpu-sim/hw_data/pascal.titanx.cycle.tgz /home/dynamo/a/tgrogers/website/gpgpu-sim/hw_data/bak.$DATE.pascal.titanx.cycle.tgz
scp pascal.titanx.cycle.tgz dynamo.ecn.purdue.edu:~/website/gpgpu-sim/hw_data/
