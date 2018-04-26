# The collection of sh calls that gather and publish correaltion data
# If an arguement is supplied - it is assumed to be the PLOTDIR on the external webserver
CORREL_THIS_DIR="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
EXTERNAL_PUBLISH_LOCATION="/home/dynamo/a/tgrogers/website/gpgpu-sim-plots/"
EXTERNAL_SERVER="tgrogers@dynamo.ecn.purdue.edu"

$CORREL_THIS_DIR/../job_launching/get_stats.py -R -K -k -B `cat $CORREL_THIS_DIR/../job_launching/apps/all-apps.list` -C TITANX_P102-2B,TITANX_P102-L1ON-2B,P100_HBM-2B > stats-$$.csv
$CORREL_THIS_DIR/../../run_hw/get_hw_data.sh && rm -rf $CORREL_THIS_DIR/correl-html
$CORREL_THIS_DIR/plot-correlation.py -c stats-$$.csv -H $CORREL_THIS_DIR/../../run_hw/

if [ $# -eq 2 ] ; then
    PLOTDIR=$1
    BUILDNUM=$2
    ssh $EXTERNAL_SERVER mkdir -p $EXTERNAL_PUBLISH_LOCATION/$PLOTDIR/$BUILDNUM
    scp  $CORREL_THIS_DIR/correl-html/* $EXTERNAL_SERVER:$EXTERNAL_PUBLISH_LOCATION/$PLOTDIR/$BUILDNUM
    ssh $EXTERNAL_SERVER "cd $EXTERNAL_PUBLISH_LOCATION/$PLOTDIR && ln -s -f $BUILDNUM latest"
fi
