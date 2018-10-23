# The collection of sh calls that gather and publish correaltion data
# It is assumed that at least 1 argument is passed (configs list)
# If 3 argments are passed $1 -> PLOTDIR and $2 -> BUILDNUM then the results will be published
CORREL_THIS_DIR="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
EXTERNAL_PUBLISH_LOCATION="/home/dynamo/a/tgrogers/website/gpgpu-sim-plots/"
EXTERNAL_SERVER="tgrogers@dynamo.ecn.purdue.edu"

if [ $# -lt 1 ] ; then
    echo "Err - Pass 1 at least one arguemnt, which is the configurations list."
    exit 1
fi

rm -r $CORREL_THIS_DIR/sigmetrics-*
$CORREL_THIS_DIR/../../run_hw/get_hw_data.sh && rm -rf $CORREL_THIS_DIR/correl-html

echo "Plotting Balanced"
$CORREL_THIS_DIR/../job_launching/get_stats.py -R -K -k -B `cat $CORREL_THIS_DIR/../job_launching/apps/sigmetrics-2018-balanced.list` -C $1 > stats-$$.csv
$CORREL_THIS_DIR/plot-correlation.py -c stats-$$.csv -H $CORREL_THIS_DIR/../../run_hw/TITAN-X-PASCAL
mkdir $CORREL_THIS_DIR/sigmetrics-balanced/ &&  mv $CORREL_THIS_DIR/correl-html/*  $CORREL_THIS_DIR/sigmetrics-balanced/

echo "Plotting Cache-sens"
$CORREL_THIS_DIR/../job_launching/get_stats.py -R -K -k -B `cat $CORREL_THIS_DIR/../job_launching/apps/sigmetrics-2018-cache.list` -C $1 > stats-$$.csv
$CORREL_THIS_DIR/plot-correlation.py -c stats-$$.csv -H $CORREL_THIS_DIR/../../run_hw/TITAN-X-PASCAL
mkdir $CORREL_THIS_DIR/sigmetrics-cache-sens/ && mv $CORREL_THIS_DIR/correl-html/*  $CORREL_THIS_DIR/sigmetrics-cache-sens/
##
echo "Plotting Compute Intensive"
$CORREL_THIS_DIR/../job_launching/get_stats.py -I -R -K -k -B `cat $CORREL_THIS_DIR/../job_launching/apps/sigmetrics-2018-compute.list` -C $1 > stats-$$.csv
$CORREL_THIS_DIR/plot-correlation.py -c stats-$$.csv -H $CORREL_THIS_DIR/../../run_hw/TITAN-X-PASCAL
mkdir $CORREL_THIS_DIR/sigmetrics-compute-intense/ && mv $CORREL_THIS_DIR/correl-html/*  $CORREL_THIS_DIR/sigmetrics-compute-intense/
##
echo "Plotting Streaming"
$CORREL_THIS_DIR/../job_launching/get_stats.py -R -K -k -B `cat $CORREL_THIS_DIR/../job_launching/apps/sigmetrics-2018-streaming.list` -C $1 > stats-$$.csv
$CORREL_THIS_DIR/plot-correlation.py -c stats-$$.csv -H $CORREL_THIS_DIR/../../run_hw/TITAN-X-PASCAL
mkdir $CORREL_THIS_DIR/sigmetrics-streaming/ && mv $CORREL_THIS_DIR/correl-html/*  $CORREL_THIS_DIR/sigmetrics-streaming/
##
scp -r $CORREL_THIS_DIR/sigmetrics-* $EXTERNAL_SERVER:$EXTERNAL_PUBLISH_LOCATION/sigmetrics-apps/2018-paper-results/
