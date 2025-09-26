#!/bin/bash
export BASH_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"

rm -rf $BASH_ROOT/nvbit_release
mkdir -p "$BASH_ROOT/nvbit_release"
wget https://github.com/NVlabs/NVBit/releases/download/v1.7.6/nvbit-Linux-x86_64-1.7.6.tar.bz2
tar -xf nvbit-Linux-x86_64-1.7.6.tar.bz2 -C "$BASH_ROOT/nvbit_release" --strip-components=1
rm nvbit-Linux-x86_64-1.7.6.tar.bz2
