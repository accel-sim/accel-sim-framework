# Welcome to the top-level repo of Accel-Sim

The [ISCA 2020 paper](https://conferences.computer.org/isca/pdfs/ISCA2020-4QlDegUf3fKiwUXfV0KdCm/466100a473/466100a473.pdf)
describes the goals of Accel-Sim and introduces the tool. This readme is meant to provide tutorial-like details on how to use the Accel-Sim
framework. If you use any component of Accel-Sim, please cite:

```
Mahmoud Khairy, Zhensheng Shen, Tor M. Aamodt, Timothy G. Rogers,
Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling,
in 2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA)
```

## Dependencies

This package is meant to be run on a modern linux distro.
A docker image that works with this repo can be found [here](https://hub.docker.com/repository/docker/accelsim/ubuntu-18.04_cuda-11).
There is nothing special here, just Ubuntu 18.04 with the following commands
run:

```bash
sudo apt-get install  -y wget build-essential xutils-dev bison zlib1g-dev flex \
      libglu1-mesa-dev git g++ libssl-dev libxml2-dev libboost-all-dev git g++ \
      libxml2-dev vim python-setuptools python-dev build-essential python-pip
pip3 install pyyaml plotly psutil
wget http://developer.download.nvidia.com/compute/cuda/11.0.1/local_installers/cuda_11.0.1_450.36.06_linux.run
sh cuda_11.0.1_450.36.06_linux.run --silent --toolkit
rm cuda_11.0.1_450.36.06_linux.run
```

Note, that all the python scripts have more detailed options explanations when run with "--help"


## Accel-Sim Repo Overview

The code for the Accel-Sim framework is in this repo. Accel-Sim 1.0 uses the
GPGPU-Sim 4.0 performance model, which was released as part of the original
Accel-Sim paper. Building the trace-based Accel-Sim will pull the right version of
GPGPU-Sim 4.0 to use in Accel-Sim.

There is an additional repo where we have collected a set of common GPU applications and a common infrastructure for building
them with different versions of CUDA. If you use/extend this app framework, it makes Accel-Sim easily usable
with a few simple command lines. The instructions in this README will take you through how to use Accel-Sim with
the apps in from this collection as well as just on your own, with your own apps.

[GPU App Collection](https://github.com/accel-sim/gpu-app-collection)

## Accel-Sim Components

![Accel-Sim Overview](https://accel-sim.github.io/assets/img/accel-sim-crop.svg)

1. **Accel-Sim Tracer**: An NVBit tool for generating SASS traces from CUDA applications. Code for the tool lives in ./util/tracer\_nvbit/. To make the tool:  
  
    ```bash  
    export CUDA_INSTALL_PATH=<your_cuda>  
    export PATH=$CUDA_INSTALL_PATH/bin:$PATH  
    ./util/tracer_nvbit/install_nvbit.sh  
    make -C ./util/tracer_nvbit/  
    ```  
    ---
    *A simple example*  
      
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
      
    ---
    For convience, we have included a repository of pre-traced applications - to get all those traces, simply run:  
    ```bash  
    ./get-accel-sim-traces.py  
    ```  
    and follow the instructions.  

2. **Accel-Sim SASS Frontend**: A simulator frontend that consumes SASS traces and feeds them into a performance model. The intial release of Accel-Sim coincides with the release of GPGPU-Sim 4.0, which acts as the detailed performance model. To build the Accel-Sim simulator that uses the traces, do the following:
    ```bash
    source ./gpu-simulator/setup_environment.sh
    make -j -C ./gpu-simulator/
    ```
    This will produce an executable in:
    ```bash
    ./gpu-simulator/bin/release/accel-sim.out
    ```

    *Running the simple example from bullet 1*
    ```bash
    ./util/job_launching/run_simulations.py -B rodinia_2.0-ft -C QV100-SASS -T ./hw_run/traces/device-<device-num>/<cuda-version>/ -N myTest
    ```
    The above command will run the worklaods in accel-sim's SASS traces-driven mode. You can also run the workloads in PTX mode using:
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

3. **Accel-Sim Correlator**: A tool that matches, plots and correlates statistics from the performance model with real hardware statistics generated by profiling tools. To use the correlator, you must first generate hardware output and simulation statistics. To generate output from the GPU, use the scripts in [./util/hw_stats](./util/hw_stats).
For example, to generate the profiler numbers for the short-running apps in our running example, do the following:
Note that this step assumes you have already built the apps using the instructions from (1).
```bash
./util/hw_stats/run_hw.py -B rodinia_2.0-ft
```

Note: Different cards support different profilers. By default - this script will use nvprof. However, you can use nsight-cli instead using:
```bash
./util/hw_stats/run_hw.py -B rodinia_2.0-ft --nsight_profiler --disable_nvprof
```

All the stats will be output in:
```bash
./hw_run/...
```

Note - that in order to correlate our running example with your local machine - you need to have a QV100 card.
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


4. **Accel-Sim Tuner**: An automated tuner that automates configuration file generation from a detailed microbenchmark suite. To be released soon!



### How do I quickly just run what Travis runs?

Install docker, then simply run:

```
docker run --env CUDA_INSTALL_PATH=/usr/local/cuda-11.0 -v `pwd`:/accel-sim:rw accelsim/ubuntu-18.04_cuda-11:latest /bin/bash travis.sh
```

If something is dying and you want to debug it - you can always run it in interactive mode:

```
docker run -it --env CUDA_INSTALL_PATH=/usr/local/cuda-11.0 -v `pwd`:/accel-sim:rw accelsim/ubuntu-18.04_cuda-11:latest /bin/bash
```

Then from within the docker run:
```
./travis.sh
```

You can also play around and do stuff inside the image (even debug the
simulator) - if you want to do this, installing gdb will help:
```
apt-get install gdb
```

Don't want to install docker?
Just use a linux ditro with the packages detailed in dependencies, set
CUDA\_INSTALL\_PATH, the run ./travis.sh.


