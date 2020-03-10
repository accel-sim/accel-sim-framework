# Instructions on using tracer tools for GPGPU-sim 

* First insall the nvbit tool:
    ```bash
    # Insall the nvbit
    ./install_nvbit.sh
    # compile the tools
    ./make
    ```
* To generate traces for GPGPU-sim trace-driven mode, run the following command:
    ```bash
    # example: to run the tracer on hardware device 0 for rodinia workloads
    ./run_hw_trace.py -B rodinia-3.1 -D 0
    ```
    the traces will be found in ../../run_hw/traces/device-0/

* To generate silicon checkpoints for execution-driven:
    ```bash
    # TO DO
    ```

