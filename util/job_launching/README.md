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

**run\_simulations.py**:

This file handles everything that needs to be done to launch jobs.
It is configured by two yaml scripts, one for benchmarks and one for configs.
The file for defining all the apps is here: `./util/job_launching/apps/define-all-apps.yml`,
and all the default configs is here: `./util/job_launching/configs/define-standard-cfgs.yml`
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

**job\_status.py**: This script will check all the jobs you ran and print details on their state (i.e. running, waiting or done). For each done job, some basic stats are also collected and printed. This is meant to be called with a -N parameter that indicates which launch of `run_simulations.py` you want the status for example:

```bash
# Run simulations using SASS (assumes you have built accel-sim and pulled the trace files)
./run_simulations.py -B rodinia_2.0-ft -C QV100-SASS -T ../../hw_run/rodinia_2.0-ft/9.1/ -N rodinia-sass-test
./job_status.py -N rodinia-sass-test
```

```bash
# Output:
squeue.id       Node                            App                     AppArgs                 Version                 Config          RunningTime     Mem         JobStatus                       Basic GPGPU-Sim Stats
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
3207            tgrogers-littleram-03           backprop-rodinia-2.0    4096___data_result_4    backprop-rodinia-2.0    QV100-SASS      0:31            702 M       RUNNING
3208            tgrogers-littleram-03           bfs-rodinia-2.0-ft      __data_graph4096_txt    bfs-rodinia-2.0-ft.a    QV100-SASS      0:31            391 M       RUNNING
3209            tgrogers-littleram-04           hotspot-rodinia-2.0-    30_6_40___data_resul    hotspot-rodinia-2.0-    QV100-SASS      0:31            581 M       RUNNING
3210            tgrogers-littleram-04           heartwall-rodinia-2.    __data_test_avi_1___    heartwall-rodinia-2.    QV100-SASS      00:00:22        578 M       COMPLETE_NO_OTHER_INFO          SIMRATE_IPS=349 K       SIM_TIME=21 sec (21 sec)        TOT_IPC=883     TOT_INSN=7 M    TOT_CYCLE=8 K
3211            tgrogers-littleram-04           kmeans-rodinia-2.0-f    _i_data_400_txt__g_d    kmeans-rodinia-2.0-f    QV100-SASS      0:28            332 M       RUNNING
3212            tgrogers-littleram-04           lud-rodinia-2.0-ft      _v__b__i___data_64_d    lud-rodinia-2.0-ft.a    QV100-SASS      0:28            326 M       RUNNING
3213            tgrogers-littleram-04           nw-rodinia-2.0-ft       128_10___data_result    nw-rodinia-2.0-ft.ac    QV100-SASS      0:28            324 M       RUNNING
3214            tgrogers-littleram-04           nn-rodinia-2.0-ft       __data_filelist_4_3_    nn-rodinia-2.0-ft.ac    QV100-SASS      0:28            517 M       RUNNING
3215            tgrogers-littleram-05           pathfinder-rodinia-2    1000_20_5___data_res    pathfinder-rodinia-2    QV100-SASS      0:28            358 M       RUNNING
3216            tgrogers-littleram-05           srad_v2-rodinia-2.0-    __data_matrix128x128    srad_v2-rodinia-2.0-    QV100-SASS      0:28            504 M       RUNNING
3217            tgrogers-littleram-05           streamcluster-rodini    3_6_16_1024_1024_100    streamcluster-rodini    QV100-SASS      0:28            358 M       RUNNING
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```


**get\_stats.py**: When the tests are all done and you want to aggregate results, this script does through all the oupt and aggregates useful statistics, outputting the result to stdout in csv format. There are many options on how to organize the output CSV data - which you can then plot however you like. You can define the stats you want to collect in a yaml file, by default, the [./stats/example_stats.yml](./stats/example_stats.yml) is used. Each line in the file corresponds to a regrex to list that stat from the Accel-Sim/GPGPU-Sim output.
Some useful ways to use `get_stats.py` are as follows:

```bash
# Just get me all the per-app stats for the set of
./get_stats.py -N rodinia-sass-test | tee per-app-fromlaunch.csv

# Ignoring how the tests were run, get me the latest results for a particular set of configs/apps
# output will be a convenient output format that can be just copy/pasted into excel, etc... to plot
# grouped bar chats. Note that you should either '-N' or '-B <apps> -C <configs>', not both.
./get_stats.py -B rodinia_2.0-ft -C QV100-SASS,QV100-PTX | tee per-app-app-cfgs.csv

# Same as above, but for each kernel launched, not for each end-result for the app:
./get_stats.py -K -k -B rodinia_2.0-ft -C QV100-SASS,QV100-PTX | tee per-kernel-instance.csv

# Get output you can feed to the auto-plotting ../plotting/plot-get-stats.py:
./get_stats.py -R -B rodinia_2.0-ft -C QV100-SASS,QV100-PTX | tee per-app-for-autoplot.csv

# Get output you can feed to the correlation plotter ../plotting/plot-correlation.py:
./get_stats.py -K -k -R -B rodinia_2.0-ft -C QV100-SASS,QV100-PTX | tee per-app-for-correlation.csv
```


### The Running directory:

These scripts create a directory structure for all the jobs that they launch.

The format is:
./<running_directory>/app_name/dirsafe_app_args/config/
Inside each of these directories is where the simulaton is run by the launched jobs.
The default running directory is `sim_run_<toolkit_version>` for example `sim_run_11.0`.
If you ever need to debug specific runs interactively, you can `cd` to the right directory then run:

```bash
gdb --args `./justrun.sh`
```

When jobs complete, they output a .o<jobId> and .e<jobId> file. This is what all the stat collection scripts parse when collecting statistics.
It is intended that there will be multiple such output files in each of these directories.
Everything in the running directory is recreated when you launch new jobs, so as long as you are fine loosing all your job outputs, this directory is safe to delete.
