# Quick bar-charts

* Make sure CUDA\_INSTALL\_PATH is set and bin/lib directories are in PATH and LD\_LIBRARY\_PATH
* You can use `./util/job_launching/get_stats.py` to generate csvs that can be plotted in any way you like.
    In this folder `./plot-get-stats.py` provides a quick way to generate bar charts from the output of `./get-stats.py`
    For example:
```bash
# Run simulations using SASS (assumes you have built accel-sim and pulled the trace files)
../job_launching/run_simulations.py -B rodinia_2.0-ft -C QV100-SASS -T ../../hw_run/rodinia_2.0-ft/9.1/ -N rodinia-sass

# Run simulations using PTX (assumes you have built accel-sim, the apps and pulled their data)
../job_launching/run_simulations.py -B rodinia_2.0-ft -C QV100-PTX  -N rodinia-ptx

# Wait for jobs to finish
../job_launching/monitor_func_test.py -N rodinia-sass
../job_launching/monitor_func_test.py -N rodinia-ptx

# Collect the stats from both jobs:
../job_launching/get_stats.py -R -C QV100-SASS,QV100-PTX -B rodinia_2.0-ft | tee per-app-stats.csv
./plot-get-stats.py -c per-app-stats.csv
```

All stats collected by the `get_stats.py` file will be plotted and placed in ./htmls/.
[An example for the IPC is here](https://engineering.purdue.edu/tgrogers/accel-sim/example-plots/example.plot.rodinia_2.0-ft.html).

# Single-page dashboard (all stats in one HTML)

`./plot-get-stats.py` writes one HTML file per stat (~25 files). If you'd rather
review everything in one place, use `./plot-get-stats-dashboard.py`, which reads
the exact same CSV and produces a single, self-contained `dashboard.html` with a
searchable sidebar. Only one stat's chart is shown at a time; click a stat in the
sidebar to switch. Stat names are cleaned up from the collection regexes (e.g.
`gpgpu_simulation_time\s*=.*\(([0-9]+) sec\).*` becomes `gpgpu_simulation_time (sec)`),
and Plotly is embedded so the file opens in any browser with no internet.

```bash
# Collect the stats (same as above):
../job_launching/get_stats.py -R -C QV100-SASS,QV100-PTX -B rodinia_2.0-ft | tee per-app-stats.csv

# Build the dashboard:
./plot-get-stats-dashboard.py -c per-app-stats.csv
# open ./htmls/dashboard.html
```

Options:
* `-c/--csv_file`  the get_stats.py CSV to plot (required)
* `-o/--output`    output HTML path (default `./htmls/dashboard.html`)
* `-n/--basename`  dashboard title
* `-s/--stats_yml` stats yml used to group the sidebar into categories (defaults to `../job_launching/stats/example_stats.yml`)

# Instructions on plotting correlation graphs

* Make sure CUDA\_INSTALL\_PATH is set and bin/lib directories are in PATH and LD\_LIBRARY\_PATH
* This is reliant on:
    * The benchmarks have been run using the `run_simulations.py` script
    * The hardware apps have been run using the run\_hw.py script (or downloaded our posted statistics using `./util/get_hw_data.sh`)
    * Plottly is installed
        ```bash
        pip install --target=~/python-package plotly
        export PYTHONPATH=~/python-package:$PYTHONPATH
        echo "export PYTHONPATH=~/python-package:$PYTHONPATH" >> ~/.bashrc
        ```
    * Generate the statistics file for the simulator using:
    ```bash
    ../job_launching/get_stats.py -R -K -k -C <Your config name> -B <simulator apps> > correl.stats.csv
    # An example: ../job_launching/get_stats.py -R -K -k -C QV100-SASS,QV100-PTX -B rodinia_2.0-ft > correl.stats.csv
    ./plot-correlation.py -c correl.stats.csv
    # Default: one offline dashboard at ./correl-html/dashboard.html
    #   (sidebar + Per-app/Per-kernel toggle; no per-stat HTML spam)
    # Per-stat HTML files only (old behavior, no dashboard):
    #   ./plot-correlation.py -c correl.stats.csv -H ... --individual
    # Both dashboard and per-stat HTML:
    #   ./plot-correlation.py -c correl.stats.csv -H ... --all-html
    # You can generate pdf files instead using
    ./plot-correlation.py -c correl.stats.csv -H ../../hw_run/QUADRO-V100/9.1/
    # You can also generate pdf files for the correaltions using "-i pdf"
    ```
For day-to-day review, prefer the default `./correl-html/dashboard.html`. Use `--individual`
only when you need the separate per-app / per-kernel HTML files for archival or sharing.
[Here is an example correlation plot for the simple rodinia tests aggregated per-app](https://engineering.purdue.edu/tgrogers/accel-sim/example-plots/gv100-cycles.QV100-PTX.QV100-SASS.per-app.html).
[And per-kernel](https://engineering.purdue.edu/tgrogers/accel-sim/example-plots/gv100-cycles.QV100-PTX.QV100-SASS.per-kernel.html).
Note again - that these short-running tests are not representative of longer running GPU apps and the correlation on these applications should
be taken into context.

We map the hardware Nvprof and Nvsight statistics with the Accel-Sim reported statistics. To see the 1:1 mapping, read [this speadsheet](https://docs.google.com/spreadsheets/d/1oLbNX-5qTnF9x4v-GCUKuA5zvjx72yD0D2rJIllBHds/edit#gid=0). Also, see the [./correl_mappings.py](https://github.com/accel-sim/accel-sim-framework/blob/dev/util/plotting/correl_mappings.py) file for the exact mapping in python script.
