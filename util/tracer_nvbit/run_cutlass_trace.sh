LD_PRELOAD=./tracer_tool/tracer_tool.so ~/accel-sim-updated/accel-sim-framework/gpu-app-collection/bin/11.0/release/cutlass_profiler

export CUDA_VERSION="11.0"; 

export CUDA_VISIBLE_DEVICES="7" 

#LD_PRELOAD=/scratch/tgrogers-disk01/a/tgrogers/accel-sim-framework/util/tracer_nvbit/tracer_tool/tracer_tool.so /home/tgrogers-raid/a/tgrogers/github/accel-sim/accel-sim-framework/gpu-app-collection/src/..//bin/11.0/release/cutlass_perf_test --seed=2020 --dist=0  --m=2560 --n=1024 --k=2560 --kernels=sgemm_nn  --iterations=5 --providers=cutlass



/scratch/tgrogers-disk01/a/tgrogers/accel-sim-framework/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing /scratch/tgrogers-disk01/a/tgrogers/accel-sim-framework/hw_run/traces/device-7/11.0/cutlass_perf_test/__seed_2020___dist_0____m_2560___n_1024___k_2560___kernels_sgemm_nn____iterations_5___providers_cutlass/traces/kernelslist

rm -f /scratch/tgrogers-disk01/a/tgrogers/accel-sim-framework/hw_run/traces/device-7/11.0/cutlass_perf_test/__seed_2020___dist_0____m_2560___n_1024___k_2560___kernels_sgemm_nn____iterations_5___providers_cutlass/traces/*.trace

rm -f /scratch/tgrogers-disk01/a/tgrogers/accel-sim-framework/hw_run/traces/device-7/11.0/cutlass_perf_test/__seed_2020___dist_0____m_2560___n_1024___k_2560___kernels_sgemm_nn____iterations_5___providers_cutlass/traces/kernelslist

