
  # Accel-Sim Tuner
  This is a tool that tunes your Accel-Sim performance model to the underline hardware. In this process, we try to generate GPGPU-Sim and Accel-Sim configuration files that match and model the hardware to the best effort. 
  
  ![Accel-Sim Class Overview](https://accel-sim.github.io/assets/img/tuner.png)

  Our tuner collects and demystifies the hardware parameters from four different resources:
1. User input HW_def header file. This file is provided by the user and contains some public information about the hardware card that is hard to be determined from micro-benchmarking (e.g. memory model, core model, etc). See an example [here](https://github.com/accel-sim/accel-sim-framework/tree/dev/util/tuner/GPU_Microbenchmark/hw_def)
  
2. Accel-Sim Microbenchmarks suite. The microbenchmarks suite demystify cache, memory, and execution unit configuration. See [./GPU_Microbenchmark/ubench](https://github.com/accel-sim/accel-sim-framework/tree/dev/util/tuner/GPU_Microbenchmark/ubench) for further details. 

3. CUDA device query. The CUDA runtime provides us with some hardware configurations, including SM number, memory width, etc. See [Device Query uebnch](https://github.com/accel-sim/accel-sim-framework/blob/dev/util/tuner/GPU_Microbenchmark/ubench/system/deviceQuery/deviceQuery.cpp) for a complete list of these reported parameters.

4. For other parameters that cannot be directly determined by our microbenchmarks (such as warp scheduling, memory scheduling, the L2 cache interleaving granularity, and the L2
cache hashing function), we do an extensive searching by simulating each possible combination of these four parameters on a set of memory bandwidth microbenchmarks.

# Tuning Steps:
The following steps demonstrate how to tune the Accel-Sim config files to a specific GPU hardware. We assume that you already have the GPU hardware in question.

1. **Provide HW def file and run microbenchmarks**:
You need to provide a C header file `hw_def` that contains minimal information about the hardware model. This file is used to configure and tune the microbenchmarks for the unduerline hardware. See an example of Ampere RTX 3060 card [here](https://github.com/accel-sim/accel-sim-framework/blob/dev/util/tuner/GPU_Microbenchmark/hw_def/ampere_RTX3070_hw_def.h). These information can be gathered from Nvidia whitepaper and public website. 
After you write the HW file for the underline card, ensure to add it in [/GPU_Microbenchmark/hw_def/hw_def.h](https://github.com/accel-sim/accel-sim-framework/blob/dev/util/tuner/GPU_Microbenchmark/hw_def/hw_def.h).
Then, compile microbenchmarks and run:

  ```bash
  # Make sure PATH includes nvcc  
  # If your hardware has new compute capability, ensure to add it in the /GPU_Microbenchmark/common/common.mk
  # compile microbenchmarks
  make -C ./GPU_Microbenchmark/
  # set the device id that you want to tune to 
  # if you do not know the device id, run ./GPU_Microbenchmark/bin/list_devices
  export CUDA_VISIBLE_DEVICES=0  
  #run the ubench and save output in stats.txt
  ./GPU_Microbenchmark/run_all.sh | tee stats.txt
  ```  
2. **Run the tuner**:
The tuner.py script will parse the microbenchmarks output and generate a folder of the HW device name (e.g. "TITAN_V"). The folder will contain the config files for GPGPU-Sim performance model and Accel-Sim trace-driven front-end (gpgpusim.config and trace.config files)

  ```bash
  # run the tuner with the stats.txt from the previous step
 ./tuner.py -s stats.txt
 
 #ensure to copy the generated folder to the gpgpu-sim and accel-sim directories
 cp -r TITAN_V ../../gpu-simulator/gpgpu-sim/configs/tested-cfgs/
 cp -r TITAN_V ../../gpu-simulator/configs/tested-cfgs/
  ``` 
   Ensure to add an entry in the [config.yml](https://github.com/accel-sim/accel-sim-framework/blob/dev/util/job_launching/configs/define-standard-cfgs.yml) with the new generated config file. For exmaple:
 ```bash
# Volta
TITANV:
    base_file: "$GPGPUSIM_ROOT/configs/tested-cfgs/TITAN_V/gpgpusim.config"
 ``` 
 
3. **Tuner searching**: Some parameters are hard to determine from microbenchmarking. Accel-Sim simulates each possible combination of these four parameters on a set of memory
bandwidth microbenchmarks (l1-bw, l2-bw, shd-bw, mem-bw, and maxflops). In the table below, we list the four undemystified parameters in question, each has two possible combinations, with a total of 16 possible cases.

| HW Parameter | Possible Options | GPGPU-Sim Options
| ------------- | ------------- | ------------- |
| Warp Scheduling  | Loose Round-Robin vs Greedy | lrr, gto |
| L2 cache interleaving granularity  | Fine vs Corase grain  | 32B, 268B  |
| L2 cache hashing function | Linear vs IPOLY  hashing | 'L', 'P'  |
| Memory Scheduling  | FCFS vs First Row-ready | FCFS, FR-FCFS  |

First, We use our run_simulations.py script to launch all 16 possible combinations as listed below. Note that, we assume you already generated the traces for the microbenchmark suite. Also, ensure to replace the "TITANV" name in the command below with the new config name entry that you have added in yml file from the previous step.

  ```bash
../job_launching/run_simulations.py \
 -T /scratch/tgrogers-disk01/a/common/accel-sim/traces/volta-tesla-v100/latest/ \
 -C TITANV-SASS,\
TITANV-SASS-LINEAR-RR-32B-FRFCFS,\
TITANV-SASS-LINEAR-RR-32B-FCFS,\
TITANV-SASS-LINEAR-RR-256B-FRFCFS,\
TITANV-SASS-LINEAR-RR-256B-FCFS,\
TITANV-SASS-LINEAR-GTO-32B-FRFCFS,\
TITANV-SASS-LINEAR-GTO-32B-FCFS,\
TITANV-SASS-LINEAR-GTO-256B-FRFCFS,\
TITANV-SASS-LINEAR-GTO-256B-FCFS,\
TITANV-SASS-IPOLY-RR-32B-FRFCFS,\
TITANV-SASS-IPOLY-RR-32B-FCFS,\
TITANV-SASS-IPOLY-RR-256B-FRFCFS,\
TITANV-SASS-IPOLY-RR-256B-FCFS,\
TITANV-SASS-IPOLY-GTO-32B-FRFCFS,\
TITANV-SASS-IPOLY-GTO-32B-FCFS,\
TITANV-SASS-IPOLY-GTO-256B-FRFCFS,\
TITANV-SASS-IPOLY-GTO-256B-FCFS \
-N tuning -B GPU_Microbenchmark

  ``` 
  
The above command typically takes few hours to finish simulation on a single machine. Once they are finished, you should collect hardware statistics and do a correlation with hardware (see Accel-Sim Correlator paragraph in the [main page readme](https://github.com/accel-sim/accel-sim-framework/blob/release/README.md) for further details).  
In the generated correlation folder, see the cycle correlation results file "*-cycles.TITANV-SASS.apps.txt". The combination with the highest average hardware correlation and the lowest error should be chosen and the config file has to be updated accordingly.

Using the correlation graphs generated by the Accel-Sim correlator, the user can pinpoint the microbenchmarks that are not well correlated to the hardware, and thus specify which simulator component needs to be further tuned and fixed. This process can be repeated over various workloads. We highly recommend researchers to correlate their benchmark suite before doing their research to ensure that the simulator gives you sane results that match the hardware. 

We have run our microbenchmark suite, the tuner, and perform tune searching as described above over four different cards, Kepler, Pascal, Volta, and Turing.  The results of these experiments can be found in [this excel sheet](https://docs.google.com/spreadsheets/d/1FlSb6XusECmG-WNSghbGcZuhFFLr4L7d5f1wsOaMFDA/edit?usp=sharing). 
  
  
