#! /bin/sh

THIS_DIR="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

cd ${SCRIPT_DIR}/bin/
for f in ./*; do
    echo "running $f microbenchmark"
    $f
    echo "/////////////////////////////////"
done

