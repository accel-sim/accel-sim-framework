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
    The traces will be found in `../../hw_run/traces/device-0/`

* To generate traces for specific individual application:
    ```bash
    # example: to run the tracer on hardware device 0 for vecadd app
    export CUDA_VISIBLE_DEVICES=0
    LD_PRELOAD=./tracer_tool/tracer_tool.so ./nvbit_release/test-apps/vectoradd/vectoradd
    ```
    the traces will be found in `traces` folder, This folder will contain:
    * 1- kernel traces files with `.trace` extension (one file per kernel), e.g. `kernel-1.trace`, `kernel-2.trace`, etc.
    * 2- `kernelslist` (one file), this contains the kernel files list that have been traced along with the CUDA memcpy commands
    * 3- `stats.csv` (one file), this contains the statistcs of the kernels, e.g. how many kernels traced, traced intructions, etc.

    Next, you will have to do post processing for the traces. The generated traces above are not structured, we have to group them by thread block Id. To do this, run the following:

    ```bash
    ./tracer_tool/traces-processing/post-traces-processing ./traces/kernelslist
    ```

    The post-traces-processing program will go through all the kernels, one by one, and generate new file ".traceg", and it will also generate the "kernelslist.g" file. These are the final files that should be given to Accel-Sim simulator. Example:

    ```bash
    ./gpu-simulator/bin/release/accel-sim.out -trace ./hw_run/rodinia_2.0-ft/9.1/backprop-rodinia-2.0-ft/4096___data_result_4096_txt/traces/kernelslist.g -config ./gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100/gpgpusim.config -config ./gpu-simulator/configs/tested-cfgs/SM7_QV100/trace.config
    ```

    .trace files are not required anymore. These are intermediate files and you can delete them to save disk space. 
    Note that the above run_hw_trace.py script do all the steps automatically for you.

* Tracing Specific kernels (kernel-based checkpointing):

    Set environment variables as below will only report kernels 3,4,5.
    ```bash
    export DYNAMIC_KERNEL_LIMIT_START=3
    export DYNAMIC_KERNEL_LIMIT_END=5
    ```

    Set environment variables as below will only report kernel 3.
    ```bash
    export DYNAMIC_KERNEL_LIMIT_START=3
    export DYNAMIC_KERNEL_LIMIT_END=3
    ```

    If you do not really know the kernel id that you are interested in, you can set kernel start with a big number like 1000000 
    ```bash
    export DYNAMIC_KERNEL_LIMIT_START=1000000
    ```
    In this case, the tracer will trace nothing. However, it will still list kernels name and ids in stats.csv file. So, check the stats.csv file and see the exact kernel Ids you want to trace. This feature is very important if your application generates large traces, and you want to skip some kernels and trace specific important kernels. 

    As an alternative to the method described above, you can wrap the region you want to trace with `cudaProfileStart()` and `cudaProfilerStop()` calls then set the following environment variable to trace only within that region. Note: setting `ACTIVE_FROM_START` to zero disables the effects of the `DYNAMIC_KERNEL_LIMIT_START/STOP` variables.
    ```bash
    export ACTIVE_FROM_START=0
    ```

* Traces format:

    The instruction format contains the following columns. Any column that is NOT contained in brackets [] must exist in any instruction format, so any instruction should have at least 10 column entries as reported below. 
    
    ```bash
    #traces format = threadblock_x threadblock_y threadblock_z warpid_tb PC mask dest_num [reg_dests] opcode src_num [reg_srcs] mem_width [adrrescompress?] [mem_addresses]
    ```
    
    The other columns that are in brackets [] may or may not exist based on the instruction characteristics, for example:
    "dest_num" tells us the number of destination registers.
    If dest_num=0, then "reg_dests" will be empty and not exist in the trace.
    If dest_num>0, this means that this instruction has dest_num destination registers, the [reg_dests] will list these registers values. 
	Similarly, the "src_num" and "reg_srcs".

    Finally, the mem_width rule is as following:
    If mem_width=0, this implies that it is not a memory instruction and [adrrescompress?] [mem_addresses] will be empty.
    If mem_width>0, this implies that this is a memory instruction with mem_width as the memory width of the data to be loaded per thread, and [adrrescompress?] [mem_addresses] will list the memory addresses in a compressed format.

    Example: 


		31 0 0 3 0000 ffffffff 1 R1 IMAD.MOV.U32 2 R255 R255 0


    This is interpreted as following: <br />
    threadblock_x threadblock_y threadblock_z=31 0 0 <br />
    warpid_tb=3 <br />
    PC =0000 (hexa) <br />
    mask=ffffffff (hexa) <br />
    dest_num=1 (how many destination registers) <br />
    reg_dests=R1 (if dest_num=0, then this would be empty) <br />
    opcode=IMAD.MOV.U32  <br />
    src_num=2  <br />
    reg_srcs=R255 R255  <br />
    mem_width = 0 (if mem_width>0, then there will be some addresses listed afterwards) <br />


