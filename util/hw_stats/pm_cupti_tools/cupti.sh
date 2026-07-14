#!/bin/bash

print_usage() {
    echo "Usage: $0 [-m metric_group] <program> [args...]"
    echo ""
    echo "Metric groups (can specify multiple -m flags):"
    echo "  gpc     - LTS srcnode GPC"
    echo "  fbp     - LTS srcnode FBP"
    echo "  hub     - LTS srcnode HUB"
    echo "  dram    - DRAM sectors"
    echo "  nvlink  - NVLink bytes"
    echo ""
    echo "If no -m specified, runs with base metrics only (cycles, instructions, tensor)"
}

usage() {
    print_usage
    exit 1
}

SELECTED_GROUPS=()

while getopts "m:h" opt; do
    case $opt in
        m) SELECTED_GROUPS+=("$OPTARG") ;;
        h) usage ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

if [ $# -lt 1 ]; then
    usage
fi

# When no -m specified, show help and run with base metrics only
if [ ${#SELECTED_GROUPS[@]} -eq 0 ]; then
    print_usage
fi

METRICS="sm__cycles_elapsed.avg"
METRICS="$METRICS,sm__inst_executed.sum"
METRICS="$METRICS,sm__pipe_tensor_cycles_active_realtime.sum"

for group in "${SELECTED_GROUPS[@]}"; do
    case $group in
        gpc)
            # LTS srcnode GPC
            METRICS="$METRICS,lts__t_sectors.sum,lts__average_t_sector_srcnode_gpc_aperture_device_op_read.pct,lts__average_t_sector_srcnode_gpc_aperture_device_op_write.pct"
            ;;
        fbp)
            # LTS srcnode FBP
            METRICS="$METRICS,lts__t_sectors.sum,lts__average_t_sector_srcnode_fbp_aperture_device_op_read.pct,lts__average_t_sector_srcnode_fbp_aperture_device_op_write.pct"
            ;;
        hub)
            # LTS srcnode HUB
            METRICS="$METRICS,lts__t_sectors.sum,lts__average_t_sector_srcnode_hub_aperture_device_op_read.pct,lts__average_t_sector_srcnode_hub_aperture_device_op_write.pct"
            ;;
        dram)
            # DRAM
            METRICS="$METRICS,dram__sectors_read.sum,dram__sectors_write.sum,dram__sectors.sum"
            ;;
        nvlink)
            # NVLink
            METRICS="$METRICS,nvlrx__bytes.sum,nvltx__bytes.sum,dram__sectors_read.sum,dram__sectors_write.sum,dram__sectors.sum"
            ;;
        *)
            echo "Unknown metric group: $group"
            usage
            ;;
    esac
done

# lts__t_sectors.sum, 
# lts__t_sectors_srcunit_tex.sum, 
# lts__t_sectors_srcnode_gpc.sum, 
# lts__average_t_sector_srcnode_fbp.pct
# lts__average_t_sector_srcnode_fbp_aperture_device.pct
# lts__average_t_sector_srcnode_gpc_aperture_device.pct
# lts__average_t_sector_srcnode_gpc_aperture_peer.pct
# lts__average_t_sector_srcunit_tex_aperture_device.pct
# lts__average_t_sector_srcunit_tex_aperture_device_lookup_hit.pct

export PM_SAMPLING_MAX_SAMPLES=160000
export INJECTION_METRICS=$METRICS
export INJECTION_KERNEL_COUNT=20
export PM_SAMPLING_HW_BUFFER_BYTES=9388608000
export PM_SAMPLING_INTERVAL_SYSCLK=3000
# export PM_SAMPLING_CSV_PATH=$PWD/pm_samples.csv
# export LD_LIBRARY_PATH=/usr/local/cuda-12.8/extras/CUPTI/lib64:$LD_LIBRARY_PATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_INJECTION64_PATH="$SCRIPT_DIR/build/libpmsampling_injection.so"

exec "$@"

