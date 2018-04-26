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

$CORREL_THIS_DIR/../job_launching/get_stats.py -R -K -k -B `cat $CORREL_THIS_DIR/../job_launching/apps/all-apps.list` -C $1 > stats-$$.csv
$CORREL_THIS_DIR/../../run_hw/get_hw_data.sh && rm -rf $CORREL_THIS_DIR/correl-html
$CORREL_THIS_DIR/plot-correlation.py -c stats-$$.csv -H $CORREL_THIS_DIR/../../run_hw/

if [ $# -eq 3 ] ; then
    PLOTDIR=$2
    BUILDNUM=$3
    ssh $EXTERNAL_SERVER mkdir -p $EXTERNAL_PUBLISH_LOCATION/$PLOTDIR/$BUILDNUM
    scp  $CORREL_THIS_DIR/correl-html/* $EXTERNAL_SERVER:$EXTERNAL_PUBLISH_LOCATION/$PLOTDIR/$BUILDNUM
    ssh $EXTERNAL_SERVER "cd $EXTERNAL_PUBLISH_LOCATION/$PLOTDIR && ln -s -f $BUILDNUM latest"
fi
