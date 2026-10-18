#!/bin/bash
# Run run_hw_trace.py on the GPU node, removing NVBit's temporary files.
#
#   .github/scripts/nvbit-trace.sh <run_hw_trace.py args...>
#
# NVBit puts /tmp/nvbit_tool_*, nvbit_code_* and nvbit_sass_* (hard-coded
# /tmp, TMPDIR is ignored) for every traced process and leaves them there;
# gpu01's /tmp is a 12 GB tmpfs, which they filled. main-tracer.sh runs this
# inside the gpu-lock singleton, so no other tracer of this user is running:
# remove this user's NVBit files before (what a killed run left) and after.
set -uo pipefail

nvbit_clean() {
    find /tmp -maxdepth 1 -user "$(id -un)" \
        \( -name 'nvbit_tool_*' -o -name 'nvbit_code_*' -o -name 'nvbit_sass_*' \) \
        -exec rm -rf -- {} +
}

nvbit_clean
rc=0
./util/tracer_nvbit/run_hw_trace.py "$@" || rc=$?
nvbit_clean
exit $rc
