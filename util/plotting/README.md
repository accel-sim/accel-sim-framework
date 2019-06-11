# Instructions on plotting correlation graphs

* Make sure CUDA\_INSTALL\_PATH is set and bin/lib directories are in PATH and LD\_LIBRARY\_PATH
* This is reliant on:
    * The benchmarks have been run using the job\_launching.py script
    * The hardware apps have been run using the run\_hw.py script (or you are validating against apps/machines that are already on posted on https://engineering.purdue.edu/tgrogers/gpgpu-sim/hw\_data/)
    * Plottly is installed
        ```bash
        pip install --target=~/python-package plotly
        export PYTHONPATH=~/python-package:$PYTHONPATH
        echo "export PYTHONPATH=~/python-package:$PYTHONPATH" >> ~/.bashrc
        ```
    * Generate the statistics file for the simulator using:
    ```bash
    ../job_launching/get_stats.py -R -K -k -C <Your config name> -B <simulator apps> > correl.stats.csv
    # An example: ../job_launching/get_stats.py -R -K -k -C TITANV -B rodinia_2.0-ft > correl.stats.csv
    ./plot-correlation.py -c correl.stats.csv
    # stdout will print summary statistics and html files will be generated in ./correl-html/
    # You can generate png files instead using 
    ./plot-correlation.py -c correl.stats.csv -i png
    # You also generate pdfs for submission quality images - but for that you must pay plotly :)
    ```
