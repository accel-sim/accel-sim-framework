start=`date +%s`
DATE=`date '+%Y-%m-%d--%H.%M.%S'`
# Volta Titan
./run_hw.py -D 0 -B `cat ../job_launching/apps/all-apps.list` -R 10
cp -r ../../run_hw/device-0 ../../run_hw/TITANV
cd ../../run_hw
tar zcvf volta.titanv.tgz ./TITANV
ssh dynamo.ecn.purdue.edu mv /home/dynamo/a/tgrogers/website/gpgpu-sim/hw_data/volta.titanv.tgz /home/dynamo/a/tgrogers/website/gpgpu-sim/hw_data/bak.$DATE.volta.titanv.tgz
scp volta.titanv.tgz dynamo.ecn.purdue.edu:~/website/gpgpu-sim/hw_data/
# GTX 480
#./run_hw.py -D 1 -B `cat ../job_launching/apps/all-apps.list` -R 10
#cp -r ../../run_hw/device-1 ../../run_hw/GTX480
#cd ../../run_hw
#tar zcvf fermi.gtx480.cycle.tgz ./GTX480
#ssh dynamo.ecn.purdue.edu mv /home/dynamo/a/tgrogers/website/gpgpu-sim/hw_data/fermi.gtx480.cycle.tgz /home/dynamo/a/tgrogers/website/gpgpu-sim/hw_data/bak.$DATE.fermi.gtx480.cycle.tgz
#scp fermi.gtx480.cycle.tgz dynamo.ecn.purdue.edu:~/website/gpgpu-sim/hw_data/
end=`date +%s`
runtime=$((end-start))
echo "Runtime: $runtime"
