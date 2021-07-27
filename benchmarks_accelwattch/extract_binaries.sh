#!/bin/bash
export BASH_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
VALIDATION_SUBDIR="/validation/"
UBENCH_SUBDIR="/microbenchmarks/"
VALIDATION_ROOT=$BASH_ROOT$VALIDATION_SUBDIR
UBENCH_ROOT=$BASH_ROOT$UBENCH_SUBDIR

if [ ! -d $VALIDATION_ROOT ]; then
    tar xzvf validation.tgz -C $BASH_ROOT  
fi

if [ ! -d $UBENCH_ROOT ]; then
    tar xzvf microbenchmarks.tgz -C $BASH_ROOT  
fi
