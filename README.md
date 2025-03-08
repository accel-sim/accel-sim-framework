# Welcome to the top-level repo of Accel-Sim and AccelWattch
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/accel-sim/accel-sim-framework)  
[![Long Tests](https://github.com/accel-sim/accel-sim-framework/actions/workflows/long-tests.yml/badge.svg)](https://github.com/accel-sim/accel-sim-framework/actions/workflows/long-tests.yml)
[![Short Tests](https://github.com/accel-sim/accel-sim-framework/actions/workflows/short-tests.yml/badge.svg)](https://github.com/accel-sim/accel-sim-framework/actions/workflows/short-tests.yml)
[![Tracer Tool](https://github.com/accel-sim/accel-sim-framework/actions/workflows/tracer-tool.yml/badge.svg)](https://github.com/accel-sim/accel-sim-framework/actions/workflows/tracer-tool.yml)
[![Weekly Tests](https://github.com/accel-sim/accel-sim-framework/actions/workflows/weekly.yml/badge.svg)](https://github.com/accel-sim/accel-sim-framework/actions/workflows/weekly.yml)
- [Welcome to the top-level repo of Accel-Sim and AccelWattch](#welcome-to-the-top-level-repo-of-accel-sim-and-accelwattch)
  - [Dependencies](#dependencies)
  - [Overview](#overview)
  - [Accel-Sim Components](#accel-sim-components)
    - [Accel-Sim Tracer](#accel-sim-tracer)
      - [A simple example](#a-simple-example)
      - [Pre-traced applications](#pre-traced-applications)
    - [Accel-Sim SASS Frontend and Simulation Engine](#accel-sim-sass-frontend-and-simulation-engine)
    - [Accel-Sim Correlator](#accel-sim-correlator)
    - [Accel-Sim Tuner](#accel-sim-tuner)
    - [How do I quickly just run what GitHub action runs?](#how-do-i-quickly-just-run-what-github-action-runs)
  - [AccelWattch Overview](#accelwattch-overview)


The [ISCA 2020 paper](https://conferences.computer.org/isca/pdfs/ISCA2020-4QlDegUf3fKiwUXfV0KdCm/466100a473/466100a473.pdf)
describes the goals of Accel-Sim and introduces the tool. This readme is meant to provide tutorial-like details on how to use the Accel-Sim
framework. If you use any component of Accel-Sim, please cite:

```
Mahmoud Khairy, Zhensheng Shen, Tor M. Aamodt, Timothy G. Rogers,
Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling,
in 2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA)
```

This repository also includes AccelWattch: A Power Modeling Framework for Modern GPUs. The [MICRO 2021 paper](http://paragon.cs.northwestern.edu/papers/2021-MICRO-AccelWattch-Kandiah.pdf) introduces AccelWattch. Please look at our [AccelWattch MICRO'21 Artifact Manual](https://github.com/accel-sim/accel-sim-framework/blob/release/AccelWattch.md) for detailed information on various AccelWattch components. For information on just running AccelWattch, please look at the [AccelWattch Overview](https://github.com/accel-sim/accel-sim-framework/blob/release/README.md#accelwattch-overview) section in this read-me.
If you use any component of AccelWattch, please cite:

```
Vijay Kandiah, Scott Peverelle, Mahmoud Khairy, Amogh Manjunath, Junrui Pan, Timothy G. Rogers, Tor Aamodt, Nikos Hardavellas,
AccelWattch: A Power Modeling Framework for Modern GPUs,
in 2021 IEEE/ACM International Symposium on Microarchitecture (MICRO)
```


## Dependencies

This package is meant to be run on a modern linux distro.
A docker image that works with this repo can be found [here](https://github.com/accel-sim/Dockerfile/pkgs/container/accel-sim-framework).
The dockerfile used to build this image can be found [here](https://github.com/accel-sim/Dockerfile), which built on top of `nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04`.

To build on local machine, install the following packages with CUDA toolkit:
```bash
# Assuming running on Ubuntu 24.04 and installing CUDA 12.8
sudo apt-get install  -y wget build-essential xutils-dev bison zlib1g-dev flex \
      libglu1-mesa-dev git g++ libssl-dev libxml2-dev libboost-all-dev git g++ \
      libxml2-dev vim python-setuptools build-essential python3-pip

pip3 install pyyaml plotly psutil
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
sh cuda_12.8.1_570.124.06_linux.run --silent --toolkit
rm cuda_12.8.1_570.124.06_linux.run
```

## Overview

The code for the Accel-Sim and AccelWattch frameworks are in this repo. Accel-Sim 1.0 uses the
[GPGPU-Sim 4.0](https://github.com/accel-sim/accel-sim-framework/blob/dev/gpu-simulator/gpgpu-sim4.md) performance model, which was released as part of the original
Accel-Sim paper. Building the trace-based Accel-Sim will pull the right version of
GPGPU-Sim 4.0 and the AccelWattch power model to use in Accel-Sim. AccelWattch replaces the GPUWattch power model in GPGPU-Sim 4.0.

There is an additional repo where we have collected a set of common GPU applications and a common infrastructure for building
them with different versions of CUDA. If you use/extend this app framework, it makes Accel-Sim easily usable
with a few simple command lines. The instructions in this README will take you through how to use Accel-Sim with
the apps in from this collection as well as just on your own, with your own apps.

[GPU App Collection](https://github.com/accel-sim/gpu-app-collection)

AccelWattch microbenchmarks and AccelWattch validation set benchmarks are also included. For more information on these benchmarks, please look at our [MICRO 2021 paper](http://paragon.cs.northwestern.edu/papers/2021-MICRO-AccelWattch-Kandiah.pdf) and [AccelWattch MICRO'21 Artifact Manual](https://github.com/accel-sim/accel-sim-framework/blob/release/AccelWattch.md).

## Accel-Sim Components

![Accel-Sim Overview](https://accel-sim.github.io/assets/img/accel-sim-crop.svg)

> Note, that all the python scripts in the following sections have more detailed options explanations when run with `--help`

### Accel-Sim Tracer

An NVBit tool for generating SASS traces from CUDA applications. Code for the tool lives in `./util/tracer_nvbit/`. To make the tool:

```bash
export CUDA_INSTALL_PATH=<your_cuda>
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
./util/tracer_nvbit/install_nvbit.sh
make -C ./util/tracer_nvbit/
```
#### A simple example

The following example demonstrates how to trace the simple rodinia functional tests
that get run in our travis regressions:

```bash
# Make sure CUDA_INSTALL_PATH is set, and PATH includes nvcc
# Get the applications, their data files and build them:
git clone https://github.com/accel-sim/gpu-app-collection
source ./gpu-app-collection/src/setup_environment
make -j -C ./gpu-app-collection/src rodinia_2.0-ft
make -C ./gpu-app-collection/src data

# Run the applications with the tracer (remember you need a real GPU for this):
./util/tracer_nvbit/run_hw_trace.py -B rodinia_2.0-ft -D <gpu-device-num-to-run-on>
```

That's it. The traces for the short-running rodinia tests will be generated in:
```bash
./hw_run/traces/
```

To extend the tracer, use other apps and understand what, exactly is going on, read [this](https://github.com/accel-sim/accel-sim-framework/blob/dev/util/tracer_nvbit/README.md).

#### Pre-traced applications
For convience, we have included a repository of pre-traced applications - to get all those traces, simply run:
```bash
./get-accel-sim-traces.py
```
and follow the instructions.

### Accel-Sim SASS Frontend and Simulation Engine

A simulator frontend that consumes SASS traces and feeds them into a performance model. The intial release of Accel-Sim coincides with the release of [GPGPU-Sim 4.0](https://github.com/accel-sim/accel-sim-framework/blob/dev/gpu-simulator/gpgpu-sim4.md), which acts as the detailed performance model. To build the Accel-Sim simulator that uses the traces, do the following:

```bash
pip3 install -r requirements.txt
source ./gpu-simulator/setup_environment.sh

# Build with make
make -j -C ./gpu-simulator/

# Build with CMake
cmake -S ./gpu-simulator/ -B ./gpu-simulator/build
cmake --build ./gpu-simulator/build -j8
cmake --install ./gpu-simulator/build
```

This will produce an executable in:
```bash
./gpu-simulator/bin/release/accel-sim.out
```

Running the [simple example](#a-simple-example) in the [tracer section](#accel-sim-tracer):

```bash
./util/job_launching/run_simulations.py -B rodinia_2.0-ft -C QV100-SASS -T ./hw_run/traces/device-<device-num>/<cuda-version>/ -N myTest
```

The above command will run the workloads in Accel-Sim's SASS traces-driven mode. You can also run the workloads in PTX mode using:

```txt
PTX mode usage: ./util/job_launching/run_simulations.py -B <benchmark> -C <gpu_config> -N <run_identifier>
Optional:
[-B benchmark]              (From the gpu-app-collection compiled in Step 1)
[-C gpu_config]             (List of supported configs: accel-sim-framework/util/job_launching/configs/define-standard-cfgs.yml)
```
Eg:
```bash
./util/job_launching/run_simulations.py -B rodinia_2.0-ft -C QV100-PTX -N myTest-PTX
```


You can monitor the tests using:
```bash
./util/job_launching/monitor_func_test.py -v -N myTest
```
After the jobs finish - you can collect all the stats using:
```bash
./util/job_launching/get_stats.py -N myTest | tee stats.csv
```

If you want to run the accel-sim.out executable command itself for specific workload, you can use:
```bash
/gpu-simulator/bin/release/accel-sim.out -trace ./hw_run/rodinia_2.0-ft/9.1/backprop-rodinia-2.0-ft/4096___data_result_4096_txt/traces/kernelslist.g -config ./gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100/gpgpusim.config -config ./gpu-simulator/configs/tested-cfgs/SM7_QV100/trace.config
```
However, we encourage you to use our workload launch manager 'run_simulations' script as shown above, which will greatly simplify the simulation process and increase productivity.

To understand what is going on and how to just run the simulator in isolation without the framework, read [this](https://github.com/accel-sim/accel-sim-framework/tree/dev/util/job_launching/README.md).

To better undersatnd the Accel-Sim front-end and the interface with GPGPU-Sim, read [this](https://github.com/accel-sim/accel-sim-framework/blob/dev/gpu-simulator/README.md).

### Accel-Sim Correlator
A tool that matches, plots and correlates statistics from the performance model with real hardware statistics generated by profiling tools. To use the correlator, you must first generate hardware output and simulation statistics. To generate output from the GPU, use the scripts in [./util/hw_stats](./util/hw_stats).
For example, to generate the profiler numbers for the short-running apps in our running example, do the following:

> Note: this step assumes you have already built the apps using the instructions from [simple example](#a-simple-example) in the [tracer section](#accel-sim-tracer).
```bash
./util/hw_stats/run_hw.py -B rodinia_2.0-ft
```

> Note: Different cards support different profilers. By default - this script will use nvprof. However, you can use nsight-cli instead using:
```bash
./util/hw_stats/run_hw.py -B rodinia_2.0-ft --nsight_profiler --disable_nvprof
```

All the stats will be output in:
```bash
./hw_run/...
```

> Note: that in order to correlate our running example with your local machine - you need to have a QV100 card.

However - we also provide a comprehensive suite of hardware profiling results, which can be obtained by running:
```bash
./util/hw_stats/get_hw_data.sh
```

Now you can use the statistics from the simulation run you did in (2) to correlate with these results.
To generate stats that can be correlated - do the following:
```bash
./util/job_launching/get_stats.py -R -k -K -B rodinia_2.0-ft -C QV100-SASS | tee per.kernel.stats.csv
```

To run the correlator - do the following:
```
./util/plotting/plot-correlation.py -c per.kernel.stats.csv -H ./hw_run/QUADRO-V100/device-0/9.1/
```

The script may take a few minutes to run (primarily because it is parsing a large amount of hardware data for >150 apps).
Stdout will print the summary of counters error, correlation, etc. and a set of correlation plots will be generated
in:
```
./util/plotting/correl-html/
```

Here you will find interactive HTML plots, csvs and textual summaries of how well the simulator correlated against hardware on both a per-kernel and per-app basis.
Note that the simple tests we ran in this tutorial are short running and not generally representative of scaled GPU apps and are just meant to quickly validate you can get Accel-Sim working.
For a true validation, you should attempt correlating the fully-scaled set of apps used in the paper.
**These will take hours to run (even on a cluster), and some consume significant memory**, but can be run using:

```bash
./util/job_launching/run_simulations.py -B rodinia-3.1,GPU_Microbenchmark,sdk-4.2-scaled,parboil,polybench,cutlass_5_trace,Deepbench_nvidia -C QV100-SASS -T ~/../common/accel-sim/traces/tesla-v100/latest/ -N all-apps -M 70G

# Once complete, collect the stats and plot
./util/job_launching/get_stats.py -k -K -R -N all-apps | tee all-apps.csv
./util/plotting/plot-correlation.py -c all-apps.csv -H ./hw_run/QUADRO-V100/device-0/9.1/
```

### Accel-Sim Tuner

An automated tuner that automates configuration file generation from a detailed microbenchmark suite. You need to provide a C header file `hw_def` that contains minimal information about the hardware model. This file is used to configure and tune the microbenchmarks for the unduerline hardware. See an example of Ampere RTX 3060 card [here](https://github.com/accel-sim/accel-sim-framework/blob/dev/util/tuner/GPU_Microbenchmark/hw_def/ampere_RTX3070_hw_def.h). Then, compile and run the microbenchmarks and the tuner:

```bash
# Make sure PATH includes nvcc
# If your hardware has new compute capability, ensure to add it in the /GPU_Microbenchmark/common/common.mk
# Compile microbenchmarks
make -C ./util/tuner/GPU_Microbenchmark/

# Set the device id that you want to tune to
# If you do not know the device id, run ./tuner/GPU_Microbenchmark/bin/list_devices
export CUDA_VISIBLE_DEVICES=0

# Run the ubench and save output in stats.txt
./util/tuner/GPU_Microbenchmark/run_all.sh | tee stats.txt
# Run the tuner with the stats.txt from the previous step
./util/tuner/tuner.py -s stats.txt
```

The tuner.py script will parse the microbenchmarks output and generate a folder with the same device name (e.g. "RTX_3060"). The folder will contain the config files for GPGPU-Sim performance model and Accel-Sim trace-driven front-end that matche and model the underline hardware as much as possible. For more detilas about the Accel-Sim tuner and the microbemcakring suite, read [this](https://github.com/accel-sim/accel-sim-framework/tree/dev/util/tuner#readme).


### How do I quickly just run what GitHub action runs?

Install docker, then simply run:

```bash
docker run -v `pwd`:/accel-sim:rw ghcr.io/accel-sim/accel-sim-framework:ubuntu-24.04-cuda-12.8 /bin/bash short-tests.sh
```

If something is dying and you want to debug it - you can always run it in interactive mode:

```bash
docker run -it -v `pwd`:/accel-sim:rw ghcr.io/accel-sim/accel-sim-framework:ubuntu-24.04-cuda-12.8 /bin/bash
```

Then from within the docker run:
```bash
./short-tests.sh
```

You can also play around and do stuff inside the image (even debug the
simulator) - if you want to do this, installing gdb will help:
```bash
apt-get install gdb
```

Don't want to install docker?
Just use a linux distro with the packages detailed in dependencies, set
`CUDA_INSTALL_PATH`./short-tests.sh, the run `./short-tests.sh`.


## AccelWattch Overview

![AccelWattch Overview](https://github.com/VijayKandiah/accel-sim.github.io/blob/master/assets/img/accelwattch-flowchart.svg)

1. **Running AccelWattch SASS SIM**: To run *the simple example from bullet 1* with AccelWattch power estimations enabled using the *AccelWattch SASS SIM* model,
```bash
./util/job_launching/run_simulations.py -B rodinia_2.0-ft -C GV100-Accelwattch_SASS_SIM -T ./hw_run/traces/device-<device-num>/<cuda-version>/ -N myTest
```
This will use the *AccelWattch SASS SIM* xml configuration file for the power model. The configuration files for the AccelWattch power model presented in our [MICRO 2021 paper](http://paragon.cs.northwestern.edu/papers/2021-MICRO-AccelWattch-Kandiah.pdf) can be found [here](https://github.com/accel-sim/gpgpu-sim_distribution/tree/release-accelwattch/configs/tested-cfgs/SM7_QV100). Please look at `./util/job_launching/configs/define-standard-cfgs.yml` for a list of provided AccelWattch configurations. The *AccelWattch HYBRID* configuration provided there uses activity factors for L2 and NOC from Accel-Sim and the rest from hardware performance counters. You can create your own *AccelWattch HYBRID* configuration in this file with a different mix of AccelWattch activity factors from Accel-Sim and hardware execution.
Upon completion of simulations, AccelWattch power estimations are stored in a *accelwattch_power_report.log* in a per-kernel format in the run directory.

2. **Running AccelWattch HW or AccelWattch HYBRID:** To run *the simple example from bullet 1* with *AccelWattch HW* or *AccelWattch HYBRID* configurations,
```bash
./util/job_launching/run_simulations.py -B rodinia_2.0-ft -a -C <GV100-Accelwattch_SASS_HW or GV100-Accelwattch_SASS_HYBRID> -T ./hw_run/traces/device-<device-num>/<cuda-version>/ -N myTest
```
Note that *AccelWattch HW* and *AccelWattch HYBRID* configurations require hardware performance counter information for the target application stored in a *hw_perf.csv* file in the run directory. A sample *hw_perf.csv* file with performance counter information collected from a GV100 card for validation suite benchmarks used in our [MICRO 2021 paper](http://paragon.cs.northwestern.edu/papers/2021-MICRO-AccelWattch-Kandiah.pdf) is copied over to the run directory by default with the above *run_simulations.py* command. The *-a* argument for *run_simulations.py* is used to feed the application name to AccelWattch. Please make sure that there is a hardware performance counter information entry with the same application name in *hw_perf.csv* for AccelWattch to obtain activity factors from. Please look at example entries in the provided `./util/accelwattch/accelwattch_hw_profiler/hw_perf.csv`.

3. **Running AccelWattch PTX SIM**: To run *the simple example from bullet 1* with AccelWattch power estimations enabled using the *AccelWattch PTX SIM* model,
```bash
./util/job_launching/run_simulations.py -B rodinia_2.0-ft -C GV100-Accelwattch_PTX_SIM -N myTest
```

4. **Hardware Power and Performance Profiler**: The AccelWattch hardware profiler scripts are located at `./util/accelwattch/accelwattch_hw_profiler/` in this repository. For more information on how to use them, please look at [this](https://github.com/accel-sim/accel-sim-framework/blob/release/AccelWattch.md#hardware-profiling-for-accelwattch-validation) section in our MICRO'21 Artifact Manual.

5. **Microbenchmarks and Quadratic Optimization Solver**: The source code for the microbenchmarks used for AccelWattch dynamic power modeling are located [here](https://github.com/accel-sim/gpu-app-collection/tree/release-accelwattch/src/cuda/accelwattch-ubench) and can be compiled by following the README [here](https://github.com/accel-sim/gpu-app-collection/tree/release-accelwattch). The Quadratic Optimization Solver MATLAB script is located at `./util/accelwattch/quadprog_solver.m`.

6. **SASS to Power Component Mapping**: The header file `gpu-simulator/ISA_Def/accelwattch_component_mapping.h` contains the Accel-Sim instruction opcode to AccelWattch power component mapping and can be extended to support new SASS instructions for future architectures. Please look at the *opcode.h* files for respective GPU Architectures in the same directory `gpu-simulator/ISA_Def/` for SASS instruction to Accel-Sim opcode mapping.
