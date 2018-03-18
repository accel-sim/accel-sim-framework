export DATA_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
TITANX_PASCAL_DATA=$DATA_ROOT/TITAN-X-PASCAL

if [ ! -d $TITANX_PASCAL_DATA ]; then
    echo "in"
    wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/hw_data/pascal.titanx.cycle.tgz .
    tar -xzvf pascal.titanx.cycle.tgz -C $DATA_ROOT
    rm pascal.titanx.cycle.tgz
fi



