rm -rf gpgpu-sim_distribution
git clone https://github.com/gpgpu-sim/gpgpu-sim_distribution.git
cd gpgpu-sim_distribution
git checkout dev
GIT_COMMIT=`git rev-parse --short HEAD`
source setup_environment
make -j
cd ..

DATE=`date '+%b_%d_%Y_%H:%M.%S'`--$GIT_COMMIT
 ../job_launching/run_simulations.py -B rodinia_2.0-ft -C GTX480,GTX480-PTXPLUS -N regress-$$-1 && \
 PLOTDIR="jenkins/gpgpu-sim-public-dev/$DATE/4.2-rodinia" && ssh tgrogers@dynamo.ecn.purdue.edu mkdir -p /home/dynamo/a/tgrogers/website/gpgpu-sim-plots/$PLOTDIR && \
../job_launching/monitor_func_test.py -v -N regress-$$-1 -s stats-$$.csv && \
../plotting/plot-get-stats.py -c stats-$$.csv -p tgrogers@dynamo.ecn.purdue.edu:~/website/gpgpu-sim-plots/$PLOTDIR -w https://engineering.purdue.edu/tgrogers/gpgpu-sim-plots/$PLOTDIR -n $PLOTDIR
../job_launching/run_simulations.py -B sdk-4.2 -C GTX480 -N regress-$$-2 && \
PLOTDIR="jenkins/gpgpu-sim-public-dev/$DATE/4.2-sdk-4.2" && ssh tgrogers@dynamo.ecn.purdue.edu mkdir -p /home/dynamo/a/tgrogers/website/gpgpu-sim-plots/$PLOTDIR && \
../job_launching/monitor_func_test.py -v -N regress-$$-2 -s stats-$$.csv && \
../plotting/plot-get-stats.py -c stats-$$.csv -p tgrogers@dynamo.ecn.purdue.edu:~/website/gpgpu-sim-plots/$PLOTDIR -w https://engineering.purdue.edu/tgrogers/gpgpu-sim-plots/$PLOTDIR -n $PLOTDIR
