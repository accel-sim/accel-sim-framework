### Running test generally

The scripts here provide a set of defaults that make it relatively painless to run the default regression test.
Remember, the following things should be setup:

1. CUDA toolkit installed
2. nvcc is in the path
3. You have sourced the `setup_environment` file inside the accel-sim-framework/gpu-simulator branch (and built the simulator).
4. If you are running SASS-mode, that you have either generated or downloaded the traces (to download them use accel-sim-framework/get-accel-sim-traces.py)
5. If you are running PTX-mode, that you have downloaded the apps with all their data and built them (see https://github.com/accel-sim/gpu-app-collection.git).

### The simulation scripts

Everybody doing any serious research with a simulator wants to do a few things:

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

Accel-Sim provides a standardized, flexible way of doing all this very quickly.
One big advantage of using this system, instead of rolling your own, is that simulations, hardware statistics, trace generation and statistic collection all use the same directory structure and basic interface for collecting results.
It makes it easy to make sure you are running the same app, with the same arguments in all situations and perform true apples-to-apple comparisons on
results across different simulator configurations and hardware. Of course you are free to simply ignore all this and essentially reimplement everything
here if you choose, but if you spend an hour learning how this works it can save you countless hours marshalling properly marshalling results.

#### ./util/job\_launching

There are 3 core scripts in here:

1. `run_simulations.py # Launches jobs. Current supports torque (qsub), slurm (sbatch), and a custom local job manager with no dependencies (procman.py)`
2. `monitor_func_test.py # Checks if jobs pass and aggregates the errors when they don't`
3. `job_status.py # Prints an easy-to-read summary of what state all jobs are in`
4. `get_stat.py # Collects the data you need to publish :)`

**run\_simulations.py **:

This file handles everything that needs to be done to launch jobs.
It is configured by two yaml scripts, one for benchmarks and one for configs.
The file for defining all the apps is here: `./util/job_launching/apps/define-all-apps.yml`,
and all the default configs is here: `./util/job_launching/configs.define-standard-cfgs.yml`
run\_simulatons.py will parse any define-\*.yml file in both directories when it is launched to determine what valid benchmarks and configs are.
You can to add your own configs or apps, just follow the same format as the other apps/configs defined in the yaml file and any app in any
location can be run using this collection of scripts. If you want to avoid interfering with the "default" files, just make your own `./util/job_launching/apps/define-myapps.yml` that contains only your applications. Same for configs.

To summarize what run\_simulations does:

1. Copies the accel-sim executable (or GPGPU-Sim .so file) already compiled in into the "running directory" (explained below) inside `<running_directory>/gpgpu-sim-builds/<gpgpu-sim-git-commit-#>`. This is a nice feature because if you rebuilt the simulator while tests are running, you will not be changing the .so that the tests are actually using.
2. It creates the right directory structure for running multiple benchmarks, with multiple arguments over multiple configs.
3. It copies all the files simulation needs to run into the newly created directories (interconnect files, gpuwattch files, any pre-generated PTX files, etc...)
4. It copies and symbolically links all the data and trace directories that each benchmark needs into the directories the apps are running from.
5. It applies any benchmark-specific options to the config files (you might want to do this if some apps require old sm versions for example).
6. After everything is setup it launches the jobs via a job manager.
7. It creates a log of all the jobs you launch this time so you can collect stats and status for just these jobs if you so choose.

**job\_status.py**:


**get\_stats.py**:


### The Running directory:

These scripts create a directory structure for all the jobs that they launch.

The format is:
./<running_directory>/app_name/dirsafe_app_args/config/
Inside each of these directories is where the simulaton is run by the launched jobs.
The default running directory is `sim_run_<toolkit_version>` for example `sim_run_11.0`.
If you ever need to debug specific runs interactively, you can `cd` to the right directory then run:

`gdb --args `./justrun.sh``

When jobs complete, they output a .o<jobId> and .e<jobId> file. This is what all the stat collection scripts parse when collecting statistics.
It is intended that there will be multiple such output files in each of these directories.
Everything in the running directory is recreated when you launch new jobs, so as long as you are fine loosing all your job outputs, this directory is safe to delete.
