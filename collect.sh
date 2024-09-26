#! /bin/bash

# configs="-C ORIN-SASS,ORIN-SASS-concurrent,ORIN-SASS-concurrent-7GR,ORIN-SASS-concurrent-7GR-MIG,ORIN-SASS-concurrent-7GR-finegrain,ORIN-SASS-concurrent-8GR,ORIN-SASS-concurrent-8GR-MIG,ORIN-SASS-concurrent-8GR-finegrain"
name="-N run-20230403-1728"

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

./util/job_launching/get_stats.py -k -R -B vulkan:pbrtexture_2k $configs $name-pbrtexture_2k > pbrtexture_2k.csv &
./util/job_launching/get_stats.py -k -R -B vulkan:pbrtexture_4k $configs $name-pbrtexture_4k > pbrtexture_4k.csv &
./util/job_launching/get_stats.py -k -R -B vulkan:instancing_2k $configs $name-instancing_2k > instancing_2k.csv &
./util/job_launching/get_stats.py -k -R -B vulkan:instancing_4k $configs $name-instancing_4k > instancing_4k.csv &
./util/job_launching/get_stats.py -k -R -B vulkan:render_passes_2k $configs $name-render_passes_2k > render_passes_2k.csv &
./util/job_launching/get_stats.py -k -R -B vulkan:render_passes_4k $configs $name-render_passes_4k > render_passes_4k.csv &

wait < <(jobs -p)
