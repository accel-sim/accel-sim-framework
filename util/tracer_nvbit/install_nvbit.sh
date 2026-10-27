#!/bin/bash
export BASH_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"

# NVBit ships a separate tarball per host architecture; pick the one matching
# this machine rather than assuming x86_64, so the tracer builds on aarch64
# (Grace-Hopper, Grace-Blackwell) too.
NVBIT_VERSION=${NVBIT_VERSION:-1.8}
NVBIT_ARCH=$(uname -m)
case "$NVBIT_ARCH" in
    x86_64|aarch64) ;;
    *) echo "No NVBit release for $NVBIT_ARCH" >&2; exit 1 ;;
esac
NVBIT_TARBALL="nvbit-Linux-${NVBIT_ARCH}-${NVBIT_VERSION}.tar.bz2"

rm -rf $BASH_ROOT/nvbit_release
mkdir -p "$BASH_ROOT/nvbit_release"
wget "https://github.com/NVlabs/NVBit/releases/download/v${NVBIT_VERSION}/${NVBIT_TARBALL}"
tar -xf "$NVBIT_TARBALL" -C "$BASH_ROOT/nvbit_release" --strip-components=1
rm "$NVBIT_TARBALL"
