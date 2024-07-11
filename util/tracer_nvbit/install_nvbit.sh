export BASH_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"

rm -rf $BASH_ROOT/nvbit_release
wget https://github.com/NVlabs/NVBit/releases/download/1.7/nvbit-Linux-x86_64-1.7-1.tar.bz2
tar -xf nvbit-Linux-x86_64-1.7-1.tar.bz2 -C $BASH_ROOT
rm nvbit-Linux-x86_64-1.7-1.tar.bz2
