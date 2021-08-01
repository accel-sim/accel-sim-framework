# Welcome to the AccelWattch MICRO'21 Artifact Appendix Manual

AccelWattch is a cycle-level constant, static and dynamic power model for the NVIDIA Volta GV100 GPU architecture.If you use any component of AccelWattch, please cite:

```
Vijay Kandiah, Scott Peverelle, Mahmoud Khairy, Amogh Manjunath, Junrui Pan, Timothy G. Rogers, Tor Aamodt, Nikos Hardavellas,
AccelWattch: A Power Modeling Framework for Modern GPUs,
in 2021 IEEE/ACM International Symposium on Microarchitecture (MICRO)
```
This Repository serves as an Artifact for the paper above and includes scripts to reproduce the figures 7 to 12 presented in our MICRO'21 paper. Please find Accel-Sim traces and benchmark datasets used for AccelWattch at the archived repository pointed to by the Artifact Appendix in our MICRO'21 paper.


## Dependencies

This package is meant to be run on a modern linux distro.
There is nothing special here that isn't already required by Accel-Sim.


## Setting up validation benchmarks for AccelWattch
### Compiling benchmarks from gpu-app-collection repository

There is an additional repo where we have collected a set of common GPU applications and a common infrastructure for building
them with different versions of CUDA. This repository also contains all the microbenchmarks used to build AccelWattch and all the benchmakrs used to validate AccelWattch.

[GPU App Collection](https://github.com/VijayKandiah/gpu-app-collection)
First run setup_environment from the gpu-app-collection repository.
```
# Make sure CUDA_INSTALL_PATH is set.
source ./src/setup_environment
```

The source code for AccelWattch Microbenchmarks are located at: 
```
src/cuda/accelwattch-ubench
```
The source code for AccelWattch Microbenchmarks are located at: 
```
src/cuda/accelwattch-ubench
```

To compile AccelWattch Microbenchmarks: 
```
make accelwattch_ubench -C ./src
```
To compile AccelWattch validation set benchmarks for simulator runs:
```
make accelwattch_validation -C ./src
```
To compile AccelWattch validation set benchmarks for power profiling individual-kernels:
```
make accelwattch_hw_power -C ./src
```
To compile everything above for AccelWattch:
```
make accelwattch -C ./src
```
If you are compiling the binaries for AccelWattch validation as shown above, please paste the binaries found at the [GPU App Collection](https://github.com/VijayKandiah/gpu-app-collection) repository at gpu-app-collection/bin/11.0/release into the accel-sim-framework repository: accel-sim-framework/accelwattch_benchmarks/validation/

### Using pre-compiled binaries for AccelWattch

Pre-compiled binaries obtained are located at accelwattch_benchmarks/
To extract pre-compiled binaries for AccelWattch:
```
cd accelwattch_benchmarks
./extract_binaries.sh
```

### Setting up datasets for AccelWattch Validation Benchmarks
Please run setup_environment this before doing anything below:
```
source gpu-simulator/setup_environment.sh
```

Extract the datasets required by AccelWattch Validation benchmarks into the accelwattch_benchmarks directory with:
```
./accelwattch_benchmarks/get_data.sh <path to accelwattch_traces.tgz file>
```


## Hardware profiling for AccelWattch validation
### Measuring power for validation kernels
Once all the validation suite binaries are located at accelwattch_benchmarks/validation, run:
```
./accelwattch_hw_profiler/profile_validation_power.sh validation_power_reports volta
```
to measure power five times for each validation set kernel. Note that you need to have a GPU card in your system.

To collect the power reports generated above and create hw_power_validaton_volta.csv containing the mean of the five power measurements recorded per validation kernel, run: 
```
./accelwattch_hw_profiler/collate_power.sh validation_power_reports volta
```
This should replace the pre-existing hw_power_validaton_volta.csv with new results.

### Collecting hardware performance counter information for validation kernels
These are required for AccelWattch HW and AccelWattch HYBRID configurations of AccelWattch.
Once all the validation suite binaries are located at accelwattch_benchmarks/validation, run:
```
./accelwattch_hw_profiler/profile_validation_perf.sh
```
This will replace the pre-existing hw_perf.csv with new results. The hw_perf.csv is also copied over to gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100/ for use in subsequent AccelWattch HW and HYBRID runs.


## Running AccelWattch and collecting power model results

### Recollecting traces
If you are recollecting Accel-Sim traces:
```
# Make sure CUDA_INSTALL_PATH is set, and PATH includes nvcc  
  
# Get the applications, their data files and build them:  
git clone https://github.com/accel-sim/gpu-app-collection  
source ./gpu-app-collection/src/setup_environment  
make -j -C ./gpu-app-collection/src rodinia_2.0-ft  
make -C ./gpu-app-collection/src data  
  
# Run the applications with the tracer (remember you need a real GPU for this):  
./util/tracer_nvbit/run_hw_trace.py -B rodinia-3.1_validation,parboil_validation,cuda_samples_11.0_validation,cutlass_5_trace_validation,cudaTensorCoreGemm_validation -D <gpu-device-num-to-run-on>  
# Since Pascal does not have tensor cores, use this command instead if you're collecting Pascal traces:
# Run the applications with the tracer (remember you need a real Volta GPU for this):  
./util/tracer_nvbit/run_hw_trace.py -B rodinia-3.1_validation,parboil_validation,cuda_samples_11.0_validation -D <gpu-device-num-to-run-on>  
```
Make sure the collected traces or the traces provided at the DOI repository inside accelwattch_traces/ are extracted and folders accelwattch_volta_traces/, accelwattch_pascal_traces/, accelwattch_turing_traces/ are present inside the same directory like accelwattch_traces/ before proceeding below. These traces are required for SASS mode of Accel-Sim simulations.

### Launching AccelWattch jobs
To launch jobs for a specific AccelWattch configuration among [volta_sass_sim, volta_sass_hybrid, volta_sass_hw, volta_ptx_sim, pascal_sass_sim, pascal_ptx_sim, turing_sass_sim, turing_ptx_sim], run:
```
./util/accelwattch/launch_jobs.sh <accelwattch_configuration> <path to root accelwattch traces directory>  
```
To launch jobs for all AccelWattch configurations needed to reproduce Figures 7 to 12 in our MICRO'21 paper, run:
```
./util/accelwattch/launch_jobs_all.sh <path to root accelwattch traces directory>  
```
The above will create accelwattch_runs/ directory which contain all the job runs, AccelWattch configuration files, Accel-Sim configuration files, and simulator output.

### Monitoring AccelWattch jobs
You can monitor the job status for a specific AccelWattch configuration among [volta_sass_sim, volta_sass_hybrid, volta_sass_hw, volta_ptx_sim, pascal_sass_sim, pascal_ptx_sim, turing_sass_sim, turing_ptx_sim] using:
```
./util/accelwattch/check_job_status.sh <accelwattch_configuration>
```
You can monitor the job status for all AccelWattch configurations:
```
./util/accelwattch/check_job_status_all.sh
```

### Collecting AccelWattch results
You can collect all available power reports for a specific AccelWattch configuration among [volta_sass_sim, volta_sass_hybrid, volta_sass_hw, volta_ptx_sim, pascal_sass_sim, pascal_ptx_sim, turing_sass_sim, turing_ptx_sim] using:
```
./util/accelwattch/collect_power_reports.sh <accelwattch_configuration>
```
You can collect all available power reports for all AccelWattch configurations:
```
./util/accelwattch/check_job_status_all.sh
```
The above will create accelwattch_power_reports/ directory containing AccelWattch output for each validation benchmark run.

### Generating per-component power breakdown CSVs
To generate CSV files with per-component power breakdowns for each validation kernel for a specific AccelWattch configuration among [volta_sass_sim, volta_sass_hybrid, volta_sass_hw, volta_ptx_sim, pascal_sass_sim, pascal_ptx_sim, turing_sass_sim, turing_ptx_sim], run:
```
./util/accelwattch/gen_sim_power_csv.py <accelwattch_configuration>
```
To generate CSV files with per-component power breakdowns for each validation kernel for all AccelWattch configurations, run:
```
./util/accelwattch/gen_sim_power_csv.py all
```
The above will create accelwattch_results/ directory with a CSV per AccelWattch configuration containing per-component power breakdowns for each validation kernel.


## Generating validation figures presented in our MICRO'21 paper
At this point you should have a CSV file (like accelwattch_volta_sass_sim.csv) containing per-component power breakdowns from AccelWattch runs for each validation kernel and a CSV file (like hw_power_validation_volta.csv) containing hardware power measurements per validation kernel recorded on a real GPU card.

The provided excel file AccelWattch_graphs.xlsx contains all the graphs with the raw data pre-filled. 

For volta_sass_sim AccelWattch configuration: Open the provided excel file AccelWattch_graphs.xlsx and paste the numbers from accelwattch_volta_sass_sim.csv into the correct columns in sheet 'Volta_SASS_SIM'. 
Make sure to check if the hw_power_validation_volta.csv file contains the benchmarks in the same order as the excel sheet because different configurations have a different set of validation benchmarks. Paste the mean hardware power measurements from hw_power_validation_volta.csv into the correct rows in column AF on the same sheet 'Volta_SASS_SIM'. This should update all the graphs shown in the excel file that are generated from the Power per Component table in sheet 'Volta_SASS_SIM'.

You can repeat the process above for each AccelWattch configuration. 
Note that we apply technology node scaling for Pascal configurations. Hence, paste the data from your CSV files into the table marked at 'BEFORE SCALING' in those respective sheets in the excel file. The table above that will be updated automatically with the technology-scaled power readings which is what we present in our MICRO'21 paper. 
Note that we apply a fudge factor for constant power component in all turing configurations. Hence, make sure to not overwrite the 'Sim_Total (in Watts)' column in the respective sheets for turing configurations.

The sheet 'Correlation plots' contains Fig 7 and Fig 10 in our MICRO'21 paper. Similarly, the sheet 'Power Breakdowns' contains Fig 8,9,11, and the sheet 'Relative Accuracy' contains Fig 12.
