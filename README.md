# A simulation repository that compliments gpgpu-sim.

## Dependencies

This package is meant to be run on a modern linux distro and requires python and torque to be installed.
Note, that all the python scripts have more detailed options explanations when run with "--help"

## Purpose

The repository provides 3 things:

1. A centralized place for GPGPU benchmark code + data + arguments that run in a reasonable amount of time on GPGPU-Sim (in a few minutes for basic tests and within a few days for long-running tests). 
2. A collection of scripts the easily facilitate the launching, error checking and statistics collection of cluster-based systems running GPGPU-Sim.
3. The regression recipes run by Travis when regressing pull requests to the GPGPU-Sim dev branch. If you want to know if you pass Travis, then run this first.


### How do I quickly just run what Travis runs?

Travis uses a docker image that has a copy of this repository on it.
Each time travis runs will pulls the lastest version of this repo (this is defined in the .travis.yml file for gpgpusim\_distribution.
You can quickly just run what travis runs by copying the command line from the last line
of the file: https://github.com/gpgpu-sim/gpgpu-sim\_distribution/blob/dev/.travis.yml

### Running test generally

The scripts here provide a set of defaults that make it relatively painless to run the default regression test.
Remember, the following things should be setup:

1. CUDA toolkit installed
2. nvcc is in the path
3. Torque is installed (even if you are not on a cluster, installing this locally is required).
4. You have sourced the `setup_environment` file inside the GPGPU-Sim dev branch (and built the simulator).

If all this is true, then running the following commands will verify that the configs that ship in GPGPU-Sim are finishing and functionally correct:

```
git checkout travis-regress
cd ./benchmarks/src
make all -j
# All the apps will be output to: ./benchmarks/bin/4.2/release
# Now that everything is built, lets run the tests:
cd ../../util/job_launching

# This actually runs the tests
./run_simulations.py -N travis-test
# This will probe all the tests you just ran and will inform you once they pass or fail
./monitor_func_test.py -N travis-test
```

If you see `Congratulations! All tests pass!` then everything is good to go.
If you see anything else, look at the error log indicated and debug what is wrong. The scripts will print out the error logs for every run that failed as well at the last 10 lines of the stdout.
If you need more than that, you can do to the benchmark's directory and run gdb manually.

### The Benchmarks

The initial iteration of this repository contains a set of functional tests based on Rodinia 2.0 created by Andrew Boktor from UBC.
Over time, we will add workloads to this tree for other updated benchmarks suites.
It is also a long-term goal that papers published using GPGPU-Sim can submit their workloads here as a centralized place where people looking to reproduce work can gather them.
The SDK is conspicuously absent due to the lengthy license agreement associated with the SDK code. It might be alright to post it here, but we don't have time to make sure.

#### ./benchmarks/data_dirs

The repo itself has no data (since git is bad with big files). git-lfs is one option that could be explored, but since the public, free version of github limits the
capacity and bandwidth using git-lfs, the data is simply retrieved via wget form a tarball hosted on Purdue's servers.

#### ./benchmarks/src/

This is where all the code for the apps go.
A top-level makefile  is here for allowing all of these to be built easily. In the initial commit, only rodinia-2.0-ft apps are included, but more will be added later.
It should also be noted that the common.mk file from the CUDA 4.2 SDK is included here, since the rodinia benchmarks rely on it.

### The simulation scripts

Everybody doing any serious research with GPGPU-Sim wants to do a few things:

1. Run a bunch of benchmarks on a bunch of different configs
2. Make sure that all their test actually finished correctly and if they didn't, debug them quickly.
3. Once you are sure everything worked, you want to collect a grid of statistics for plotting in papers, which typically look like:
```
IPC:
    , Config1, Config2, ...
App1,     X  ,   Y    , ...
App2,     A  ,   B    , ...
...
```

This repo provides a standardized, flexible way of doing all this very quickly.

#### ./util/job_launching

There are 3 core scripts in here:

1. `run_simulations.py # Launches jobs`
2. `job_status.py # Checks if jobs pass and get errors when they don't`
3. `get_stat.py # Collects the data you need to publish :)`

`run_simulations.py`:

This file handles everything that needs to be done to launch jobs.
It is configured by two yaml scripts, one for benchmarks and one for configs.
The default files here are in `./util/job_launching/regression_recipies/rodinia_2.0-ft/`.
The comments in these files should be self-explanatory. The overall intent here is that you point run_simulations.py at different benchmark.yml and configs.yml files for whatever you want to run.
run_simualtions.py does all the following:

1. Copies the .so file already compiled in gpgpu-sim root into the "running directory" (explained below) inside `<running_directory>/gpgpu-sim-builds/<gpgpu-sim-git-commit-#>`. This is a nice feature because if you rebuilt the simulator while tests are running, you will not be changing the .so that the tests are actually using (when torque runs, it adds this new directory to the LD_LIBRARY_PATH).
2. It creates the right directory structure for running multiple benchmarks, with multiple arguments over multiple configs.
3. It copies all the files GPGPU-Sim needs to run into the newly created directories (interconnect files, gpuwattch files, any pre-generated PTX files, etc...)
4. It copies and (symbolically linking all the data) that each benchmark needs into the directories the apps are running from.
5. It applies any benchmark-specific options to the config files (you might want to do this if some apps require old sm versions for example).
6. After everything is setup it launches the jobs via torque. The core command here is `qsub` - if you need to modify these scripts for use with another job manager, look for `qsub` in this file.
7. It creates a log of all the jobs you launch this time so you can collect stats and status for just these jobs if you so choose.

`job_status.py`


`get_stats.py`


### The Running directory:

These scripts create a directory structure for all the jobs that they launch.

The format is:
./running_directory/app_name/dirsafe_app_args/config/
Inside each of these directories is where GPGPU-Sim is run by the launched torque jobs.
The default running directory is `sim_run_<toolkit_version>` for example `sim_run_4.2`.
If you ever need to debug specific runs interactively, you can `cd` to the right directory then run:

`gdb --args `tail -1 torque.sim``

This works because the last line of the torque.sim file contains the commandline for running the program with these arguments.
When jobs complete in torque, they output a .o<jobId> and .e<jobId> file. This is what all the stat collection scripts parse when collecting statistics.
It is intended that there will be multiple such output files in each of these directories, as you change GPGPU-Sim to fix bugs and model additional features.
Everything in the running directory is recreated when you launch new jobs, so as long as you are fine loosing all your job outputs, this directory is safe to delete.
