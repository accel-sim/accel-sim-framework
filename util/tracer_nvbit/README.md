# Instructions on using tracer tool for Accel-sim 

* First insall the nvbit tool:
    ```bash
    # Insall the nvbit
    ./install_nvbit.sh
    # compile the tools
    ./make
    ```
* To generate traces for benchamrk suite, run the following command:
    ```bash
    # example: to run the tracer on hardware device 0 for rodinia workloads
    ./run_hw_trace.py -B rodinia-3.1 -D 0
    ```
    the traces will be found in ../../run_hw/traces/device-0/

* To generate traces for specific individual application:
    ```bash
    # example: to run the tracer on hardware device 0 for vecadd app
    export CUDA_VISIBLE_DEVICES=0
    LD_PRELOAD=./tracer_tool/tracer_tool.so ./nvbit_release/test-apps/vectoradd/vectoradd
    ```
    the traces will be found in "traces" folder, This folder will contain:
    * 1- kernel traces files with .trace (one file per kernel), e.g. kernel-1.trace, kernel-2.trace, etc.
    * 2- kernelslist (one file), this contains the kernel files list that have been traced along with the cuda memcpy commands
    * 3- stats.csv (one file), this contains the statistcs of the kernels, e.g. how many kernels traced, traced intructions, etc.

    Next, you will have to do post processing for the traces. The generated traces above are not structured, we have to group them by thread block Id. To do this, run the following:

    ```bash
    ./tracer_tool/traces-processing/post-traces-processing ./traces/kernelslist
    ```

    the post-traces-processing will go through all the kernels, one by one, and generate new file ".traceg", and it will also generate the kernelslist.g file. These are the final files that shold be given to accel-sim simulatror. Example:

    ```bash
    ./accelsim.out -trace ../../../../../../Nvbits/nvbit_1.1/nvbit_release/traces/kernelslist.g 
    ```

    .trace files are not required anymore. These are intermediate files and you can delete them to save disk space. 
    Note that the above run_hw_trace.py script do all the steps automatically for you.

* Tracing Specific kernels (kernel-based checkpointing):

    Set enviroment variables as below wil only report kernels 3,4,5.
    ```bash
    export DYNAMIC_KERNEL_LIMIT_START=3
    export DYNAMIC_KERNEL_LIMIT_END=5
    ```

    Set enviroment variables as below wil only report kernel 3 only
    ```bash
    export DYNAMIC_KERNEL_LIMIT_START=3
    export DYNAMIC_KERNEL_LIMIT_END=3
    ```

    If you do not really know the kernel id that you are intersted in, You can set kernel start with a big number like 1000 and the tracer will trace nothing. But it will still list kernels name and id in stats.csv. So, see the stats.csv file and you can check the file and see the exact kernel Id you want to trace. This feature is very important if your application generates large traces, and you want to skip some kernels and trace specific important kernels.

