export BASH_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
DATA_SUBDIR="/data_dirs/"
DATA_ROOT=$BASH_ROOT$DATA_SUBDIR

if [ ! -d $DATA_ROOT ]; then
    wget https://github.com/NVlabs/NVBit/releases/download/1.1/nvbit-Linux-x86_64-1.1.tar.bz2
    tar -xf nvbit-Linux-x86_64-1.1.tar.bz2 -C $BASH_ROOT
    rm nvbit-Linux-x86_64-1.1.tar.bz2
fi


