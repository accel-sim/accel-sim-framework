export DATA_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
export DATA_ROOT=$DATA_ROOT/../../hw_run

mkdir -p $DATA_ROOT

TITANX_PASCAL_DATA=$DATA_ROOT/TITAN-X-PASCAL
P100_PASCAL_DATA=$DATA_ROOT/TESLA-P100
FERMI_GTX480_DATA=$DATA_ROOT/GTX480
TI1080_DATA=$DATA_ROOT/1080TI_PASCAL
TITANV_DATA=$DATA_ROOT/TITANV
QUADRO_V100_DATA=$DATA_ROOT/QUADRO-V100
VOLTA_TESLA_V100_DATA=$DATA_ROOT/volta-tesla-v100
TURING_RTX_DATA=$DATA_ROOT/TURING-RTX2060
KEPLER_DATA=$DATA_ROOT/KEPLER-TITAN
AMPERE_RTX_DATA=$DATA_ROOT/AMPERE-RTX3070

if [ ! -d $AMPERE_RTX_DATA ]; then
    TARFILE=ampere.rtx3070.tgz
    wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/hw_data/$TARFILE
    tar -xzvf $TARFILE -C $DATA_ROOT
    rm $TARFILE
fi

if [ ! -d $TURING_RTX_DATA ]; then
    TARFILE=turing.rtx2060.cycle.tgz
    wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/hw_data/$TARFILE
    tar -xzvf $TARFILE -C $DATA_ROOT
    rm $TARFILE
fi

if [ ! -d $TITANX_PASCAL_DATA ]; then
    wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/hw_data/pascal.titanx.cycle.tgz
    tar -xzvf pascal.titanx.cycle.tgz -C $DATA_ROOT
    rm pascal.titanx.cycle.tgz
fi

if [ ! -d $P100_PASCAL_DATA ]; then
    wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/hw_data/pascal.tesla.p100.cycles.tgz
    tar -xzvf pascal.tesla.p100.cycles.tgz -C $DATA_ROOT
    rm pascal.tesla.p100.cycles.tgz
fi

if [ ! -d $FERMI_GTX480_DATA ]; then
    wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/hw_data/fermi.gtx480.cycle.tgz
    tar -xzvf fermi.gtx480.cycle.tgz -C $DATA_ROOT
    rm fermi.gtx480.cycle.tgz
fi

if [ ! -d $TI1080_DATA ]; then
    wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/hw_data/pascal.1080ti.cycle.tgz
    tar -xzvf pascal.1080ti.cycle.tgz -C $DATA_ROOT
    rm pascal.1080ti.cycle.tgz
fi

if [ ! -d $TITANV_DATA ]; then
    wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/hw_data/volta.titanv.tgz
    tar -xzvf volta.titanv.tgz -C $DATA_ROOT
    rm volta.titanv.tgz
fi

if [ ! -d $QUADRO_V100_DATA ]; then
    TARFILE=quadro.v100.cycle.tgz
    wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/hw_data/$TARFILE
    tar -xzvf $TARFILE -C $DATA_ROOT
    rm $TARFILE
fi

if [ ! -d $VOLTA_TESLA_V100_DATA ]; then
    TARFILE=volta-tesla-v100.tgz
    wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/hw_data/$TARFILE
    tar -xzvf $TARFILE -C $DATA_ROOT
    rm $TARFILE
fi

if [ ! -d $KEPLER_DATA ]; then
    TARFILE=kepler.titan.cycle.tgz
    wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/hw_data/$TARFILE
    tar -xzvf $TARFILE -C $DATA_ROOT
    rm $TARFILE
fi
