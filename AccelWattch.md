# Welcome to the AccelWattch MICRO'21 Artifact Appendix Manual

![AccelWattch Overview](https://github.com/VijayKandiah/accel-sim.github.io/blob/master/assets/img/accelwattch-flowchart.svg)

AccelWattch is a cycle-level constant, static and dynamic power modeling framework tuned for the NVIDIA Volta GV100 GPU. If you use any component of AccelWattch, please cite:

```
Vijay Kandiah, Scott Peverelle, Mahmoud Khairy, Amogh Manjunath, Junrui Pan, Timothy G. Rogers, Tor Aamodt, Nikos Hardavellas,
AccelWattch: A Power Modeling Framework for Modern GPUs,
in 2021 IEEE/ACM International Symposium on Microarchitecture (MICRO)
```
This Repository serves as an Artifact for the paper above and includes scripts to reproduce the figures 7 to 12 presented in our MICRO'21 paper. Please find Accel-Sim traces and benchmark datasets used for AccelWattch at the archived repository pointed to by the Artifact Appendix in our MICRO'21 paper: https://doi.org/10.5281/zenodo.5398781

Please see [AccelWattch Overview](https://github.com/accel-sim/accel-sim-framework/blob/release-accelwattch/README.md#accelwattch-overview) entry in the main read-me page if you are only looking for information on running AccelWattch power estimations for your applications.


## Dependencies

This package is meant to be run on a modern linux distro.
There is nothing special here in terms of dependencies that isn't already required by Accel-Sim which can be resolved with the following commands: 
```bash
sudo apt-get install  -y wget build-essential xutils-dev bison zlib1g-dev flex \
      libglu1-mesa-dev git g++ libssl-dev libxml2-dev libboost-all-dev git g++ \
      libxml2-dev vim python-setuptools python-dev build-essential python-pip makedepend
pip install pyyaml==5.1 plotly psutil pandas
wget http://developer.download.nvidia.com/compute/cuda/11.0.1/local_installers/cuda_11.0.1_450.36.06_linux.run
sh cuda_11.0.1_450.36.06_linux.run --silent --toolkit
rm cuda_11.0.1_450.36.06_linux.run
```
Note that we use python 2.7.18 by default for our python scripts unless explicitly specified with `#!/usr/bin/python3` at the top of a python script.


## Setting up validation benchmarks for AccelWattch
### Compiling benchmarks from gpu-app-collection repository

There is an additional repo where we have collected a set of common GPU applications and a common infrastructure for building
them with different versions of CUDA. The `release-accelwattch` branch of this repository also contains all the microbenchmarks used to build AccelWattch and all the benchmakrs used to validate AccelWattch. 

[GPU App Collection](https://github.com/accel-sim/gpu-app-collection/tree/release-accelwattch)

First run setup_environment from the root directory of gpu-app-collection repository. 
```
# Make sure CUDA_INSTALL_PATH is set.
cd gpu-app-collection
source src/setup_environment
```
This will set environment variable $GPUAPPS_ROOT to be the path to gpu-app-collection root directory. We refer to the root directory of gpu-app-collection with this environment variable in the steps below.

The source code for AccelWattch Microbenchmarks are located at: 
```
$GPUAPPS_ROOT/src/cuda/accelwattch-ubench
```
The source code for AccelWattch Microbenchmarks are located at: 
```
$GPUAPPS_ROOT/src/cuda/accelwattch-ubench
```

To compile AccelWattch Microbenchmarks: 
```
make accelwattch_ubench -C $GPUAPPS_ROOT/src
```
To compile AccelWattch validation set benchmarks for simulator runs:
```
make accelwattch_validation -C $GPUAPPS_ROOT/src
```
To compile AccelWattch validation set benchmarks for power profiling individual-kernels:
```
make accelwattch_hw_power -C $GPUAPPS_ROOT/src
```
To compile everything above for AccelWattch:
```
make accelwattch -C $GPUAPPS_ROOT/src
```

Please run setup_environment from the root directory of accel-sim-framework before doing anything below:
```
cd accel-sim-framework
source gpu-simulator/setup_environment.sh
```
This will set environment variable $ACCELSIM_ROOT to be the path to **gpu-simulator/** directory inside accel-sim-framework repository. We refer to the root directory of accel-sim-framework using this environment variable in the steps below. This will also fetch the release version sources of GPGPU-Sim with AccelWattch into **$ACCELSIM_ROOT/gpgpu-sim**.

**NOTE:** If you are compiling the binaries for AccelWattch validation as above, please paste the binaries found at the [GPU App Collection](https://github.com/accel-sim/gpu-app-collection/tree/release-accelwattch) repository at **$GPUAPPS_ROOT/bin/11.0/release** into the accel-sim-framework repository at **$ACCELSIM_ROOT/../util/accelwattch/accelwattch_benchmarks/validation** with 
```
mkdir -p $ACCELSIM_ROOT/../util/accelwattch/accelwattch_benchmarks/validation
cp $GPUAPPS_ROOT/bin/11.0/release/* $ACCELSIM_ROOT/../util/accelwattch/accelwattch_benchmarks/validation
```

### Using pre-compiled binaries for AccelWattch

Alternatively, you can use pre-compiled binaries located at **$ACCELSIM_ROOT/../util/accelwattch/accelwattch_benchmarks/**
To extract pre-compiled binaries for AccelWattch:
```
cd $ACCELSIM_ROOT/../util/accelwattch/accelwattch_benchmarks
./extract_binaries.sh
```
This will create a folder **$ACCELSIM_ROOT/../util/accelwattch/accelwattch_benchmarks/validation** with the binaries for validation workloads and a folder **$ACCELSIM_ROOT/../util/accelwattch/accelwattch_benchmarks/microbenchmarks** with the binaries for AccelWattch microbenchmarks.

**NOTE:** These binaries include dynamically-linked libraries and may not run on your system. Hence, we **STRONGLY** recommend compiling the binaries yourself following the previous step above.


### Setting up datasets for AccelWattch Validation Benchmarks


Extract the datasets required by AccelWattch Validation benchmarks into the accelwattch_benchmarks directory with:
```
$ACCELSIM_ROOT/../util/accelwattch/accelwattch_benchmarks/get_data.sh <path to accelwattch_traces.tgz file>
```
This will create a data_dirs/ folder in **$ACCELSIM_ROOT/../util/accelwattch/accelwattch_benchmarks** containing all the input datasets required by the validation benchmarks used for AccelWattch.

## Hardware profiling for AccelWattch validation
Please do a `export CUDA_VISIBLE_DEVICES=<GPU_DEVID>` before proceeding below. You can find out <GPU_devid> of the target GPU by just doing a `nvidia-smi` while a GPU workload is running in the background.
### Measuring power for validation kernels
Once all the validation suite binaries are located at $ACCELSIM_ROOT/../util/accelwattch/accelwattch_benchmarks/validation, run this per GPU Arch among [volta, turing, pascal]:
```
make -C $ACCELSIM_ROOT/../util/accelwattch/accelwattch_hw_profiler
$ACCELSIM_ROOT/../util/accelwattch/accelwattch_hw_profiler/profile_validation_power.sh <GPU_Arch> <GPU_devid>
```
to measure power five times for each validation set kernel. Note that you need to have a GPU card in your system. Please also specify <GPU_devid> for the target GPU device if it's not 0. You can find out <GPU_devid> by just doing a `nvidia-smi`.

To collect the power reports generated above and create **hw_power_validaton_<GPU_Arch>.csv** containing the mean of the five power measurements recorded per validation kernel, run: 
```
$ACCELSIM_ROOT/../util/accelwattch/accelwattch_hw_profiler/collate_power.sh validation_power_reports <GPU_Arch> <GPU_devid>
```
This should replace the pre-existing hw_power_validaton_<GPU_Arch>.csv with new results. 

**NOTE:** We provide hw_power_validaton_volta.csv with pre-filled hardware power measurements. We also provide pre-filled pascal and turing power measurements directly in the excel sheet **$ACCELSIM_ROOT/../util/accelwattch/AccelWattch_graphs.xlsx**

### Collecting hardware performance counter information for validation kernels
These are required for AccelWattch HW and AccelWattch HYBRID configurations of AccelWattch.
Once all the validation suite binaries are located at **$ACCELSIM_ROOT/../util/accelwattch/accelwattch_benchmarks/validation**, run:
```
$ACCELSIM_ROOT/../util/accelwattch/accelwattch_hw_profiler/profile_validation_perf.sh
```
This will replace the pre-existing hw_perf.csv with new results. The hw_perf.csv is also copied over to **$ACCELSIM_ROOT/../gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100/** for use in subsequent AccelWattch HW and HYBRID runs.

## Building AccelWattch
To build AccelWattch and Accel-Sim using a single CPU core, please run:
```
make -C $ACCELSIM_ROOT/
```
To build AccelWattch and Accel-Sim using multiple CPU cores, please run:
```
make -j -C $ACCELSIM_ROOT/
```
This will produce an executable `$ACCELSIM_ROOT/bin/release/accel-sim.out`.

## Running AccelWattch and collecting power model results

### Recollecting traces
If you are recollecting Accel-Sim traces:
```
# Make sure CUDA_INSTALL_PATH is set, and PATH includes nvcc  

# Run the applications with the tracer (remember you need a real Volta or Turing GPU for this):  
$ACCELSIM_ROOT/../util/tracer_nvbit/run_hw_trace.py -B rodinia-3.1_validation,parboil_validation,cuda_samples_11.0_validation,cutlass_5_trace_validation,cudaTensorCoreGemm_validation -D <gpu-device-num-to-run-on>  
# Since Pascal does not have tensor cores, use this command instead if you're collecting Pascal traces:
# Run the applications with the tracer (remember you need a real Pascal GPU for this):  
$ACCELSIM_ROOT/../util/tracer_nvbit/run_hw_trace.py -B rodinia-3.1_validation,parboil_validation,cuda_samples_11.0_validation -D <gpu-device-num-to-run-on>  
```
Make sure the collected traces are placed in these respective folders per GPU arch: accelwattch_volta_traces/, accelwattch_pascal_traces/, accelwattch_turing_traces/. Thesse diectories should all be present inside the same directory such as 'accelwattch_traces/' before proceeding below. These traces are required for the SASS mode of Accel-Sim simulations.

### Using the provided traces
Alternatively, you can use the traces provided at the DOI repository at accelwattch_traces/. You would have to extract them with:
```
tar -xf accelwattch_traces/accelwattch_volta_traces.tgz -C accelwattch_traces/
tar -xf accelwattch_traces/accelwattch_pascal_traces.tgz -C accelwattch_traces/
tar -xf accelwattch_traces/accelwattch_turing_traces.tgz -C accelwattch_traces/
```
This will create three folders inside accelwattch_traces/ namely accelwattch_volta_traces/, accelwattch_pascal_traces/, accelwattch_turing_traces/.

### Launching AccelWattch jobs
To launch jobs for a specific AccelWattch configuration among [volta_sass_sim, volta_sass_hybrid, volta_sass_hw, volta_ptx_sim, pascal_sass_sim, pascal_ptx_sim, turing_sass_sim, turing_ptx_sim], run:
```
$ACCELSIM_ROOT/../util/accelwattch/launch_jobs.sh <accelwattch_configuration> <path to root accelwattch traces directory>  
```
To launch jobs for all AccelWattch configurations needed to reproduce Figures 7 to 12 in our MICRO'21 paper, run:
```
$ACCELSIM_ROOT/../util/accelwattch/launch_jobs_all.sh <path to root accelwattch traces directory>  
```
The above will create **$ACCELSIM_ROOT/../accelwattch_runs/** directory which contain all job runs, AccelWattch configuration files, Accel-Sim configuration files, and simulator output.

**NOTE:** Some Accel-Sim jobs like `cudaTensorCoreGemm` require ~20G memory to run and might get killed when out-of-memory. To relaunch individual jobs, go to the run directory and run the launch command as shown in this example:
```
cd $ACCELSIM_ROOT/../accelwattch_runs/volta_sass_sim/cudaTensorCoreGemm/NO_ARGS/QV100-Accelwattch_SASS_SIM
$ACCELSIM_ROOT/bin/release/accel-sim.out  -config ./gpgpusim.config -trace ./traces/kernelslist.g
```


### Monitoring AccelWattch jobs
You can monitor the job status for a specific AccelWattch configuration among [volta_sass_sim, volta_sass_hybrid, volta_sass_hw, volta_ptx_sim, pascal_sass_sim, pascal_ptx_sim, turing_sass_sim, turing_ptx_sim] using:
```
$ACCELSIM_ROOT/../util/accelwattch/check_job_status.sh <accelwattch_configuration>
```
You can monitor the job status for all AccelWattch configurations:
```
$ACCELSIM_ROOT/../util/accelwattch/check_job_status_all.sh
```
You can use the same command in any run directory to rerun the respective job. The AccelWattch power output for all runs will be stored at a new file `accelwattch_power_report.log` in the run directory. This is all we need to validate Accelwattch. To save Accel-Sim performance output, simply append a `>>accelsim_out.txt` or something similar to the run command.

### Collecting AccelWattch results
You can collect all available power reports for a specific AccelWattch configuration among [volta_sass_sim, volta_sass_hybrid, volta_sass_hw, volta_ptx_sim, pascal_sass_sim, pascal_ptx_sim, turing_sass_sim, turing_ptx_sim] using:
```
$ACCELSIM_ROOT/../util/accelwattch/collect_power_reports.sh <accelwattch_configuration>
```
You can collect all available power reports for all AccelWattch configurations:
```
$ACCELSIM_ROOT/../util/accelwattch/collect_power_reports_all.sh
```
The above will create **$ACCELSIM_ROOT/../accelwattch_power_reports/** directory containing AccelWattch output for each validation benchmark run. Note that this script copies over AccelWattch output power reports found in $ACCELSIM_ROOT/../accelwattch_runs/ into a simpler directory structure at $ACCELSIM_ROOT/../accelwattch_power_reports/

### Generating per-component power breakdown CSVs
To generate CSV files with per-component power breakdowns for each validation kernel for a specific AccelWattch configuration among [volta_sass_sim, volta_sass_hybrid, volta_sass_hw, volta_ptx_sim, pascal_sass_sim, pascal_ptx_sim, turing_sass_sim, turing_ptx_sim], run:
```
$ACCELSIM_ROOT/../util/accelwattch/gen_sim_power_csv.py <accelwattch_configuration>
```
To generate CSV files with per-component power breakdowns for each validation kernel for all AccelWattch configurations, run:
```
$ACCELSIM_ROOT/../util/accelwattch/gen_sim_power_csv.py all
```
The above will create **$ACCELSIM_ROOT/../accelwattch_results/** directory with a CSV per AccelWattch configuration containing per-component power breakdowns for each validation kernel.


## Generating validation figures presented in our MICRO'21 paper
At this point you should have a CSV file (like accelwattch_volta_sass_sim.csv) containing per-component power breakdowns from AccelWattch runs for each validation kernel and a CSV file (like hw_power_validation_volta.csv) containing hardware power measurements per validation kernel recorded on a real GPU card.

The provided excel file *AccelWattch_graphs.xlsx* contains all the graphs with the raw data pre-filled. 

For volta_sass_sim AccelWattch configuration: Open the provided excel file **$ACCELSIM_ROOT/../util/accelwattch/AccelWattch_graphs.xlsx** and copy paste the numbers from $ACCELSIM_ROOT/../accelwattch_results/accelwattch_volta_sass_sim.csv into the correct columns in sheet *Volta_SASS_SIM*. 
Make sure to check if the hw_power_validation_volta.csv file contains the benchmarks in the same order as the excel sheet because different AccelWattch configurations have a different set of validation benchmarks. Paste the mean hardware power measurements from $ACCELSIM_ROOT/../util/accelwattch/accelwattch_hw_profiler/hw_power_validation_volta.csv into the correct rows in column AF on the same sheet 'Volta_SASS_SIM'. This should update all the graphs shown in the excel file that are generated from the Power per Component table in sheet 'Volta_SASS_SIM'.

The same process as above can be repeated for each AccelWattch configuration. 

Note that we apply technology node scaling for Pascal configurations. Hence, paste the data from your CSV files into the table marked as *BEFORE SCALING* in those respective sheets in the excel file. The table above that will be updated automatically with the technology-scaled power readings. This is what we present in our MICRO'21 paper. 
Also, note that we apply a fudge factor for constant power component in all turing configurations. Hence, make sure to not overwrite the *Sim_Total (in Watts)* column in the respective sheets for turing configurations.

The sheet *Correlation plots* contains Fig 7 and Fig 10 in our MICRO'21 paper. Similarly, the sheet *Power Breakdowns* contains Fig 8,9,11, and the sheet *Relative Accuracy* contains Fig 12.
