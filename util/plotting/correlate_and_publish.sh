# The collection of sh calls that gather and publish correaltion data
# It is assumed that at least 1 argument is passed (configs list)
CORREL_THIS_DIR="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
EXTERNAL_PUBLISH_LOCATION="/home/dynamo/a/tgrogers/website/gpgpu-sim-plots/"
EXTERNAL_SERVER="tgrogers@dynamo.ecn.purdue.edu"


$CORREL_THIS_DIR/../../hw_run/get_hw_data.sh && rm -rf $CORREL_THIS_DIR/correl-html
$CORREL_THIS_DIR/plot-correlation.py -c $1 -H $CORREL_THIS_DIR/../../hw_run/ -b ./pact.blacklist

if [ $# -eq 3 ] ; then
    PLOTDIR=$2
    BUILDNUM=$3
    ssh $EXTERNAL_SERVER mkdir -p $EXTERNAL_PUBLISH_LOCATION/$PLOTDIR/$BUILDNUM
    scp  $CORREL_THIS_DIR/correl-html/* $EXTERNAL_SERVER:$EXTERNAL_PUBLISH_LOCATION/$PLOTDIR/$BUILDNUM
    ssh $EXTERNAL_SERVER "cd $EXTERNAL_PUBLISH_LOCATION/$PLOTDIR && rm -rf latest && cp -r $BUILDNUM latest"
fi
