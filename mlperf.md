
# 🔧 Environment Initialization Instructions

### **Step 1: Pull the Docker Image**

Download the official Accel-Sim Framework Docker image:

```bash
docker pull ghcr.io/accel-sim/accel-sim-framework:ubuntu-24.04-cuda-12.8
```

---

### **Step 2: Start the Docker Container**

Run the container with GPU support:

```bash
docker run --name <container_name> --gpus all -it ghcr.io/accel-sim/accel-sim-framework:ubuntu-24.04-cuda-12.8 /bin/bash
```

> 🔁 Replace `<container_name>` with a name of your choice.

---

### **Step 3: Run MLPerf BERT Test**

Execute the pre-installed MLPerf BERT inference script:

```bash
/accel-sim/gpu-app-collection/bin/12.8/release/mlperf_inference/inference_mlperf_bert_test.sh
```

To change the number of inference queries, edit the script and modify the `--test_query_count` parameter:

```bash
--test_query_count=500
```

---

# 📍 Tracing Instructions with Accel-Sim

### **Step 1: Clone the Accel-Sim Repository**

```bash
git clone https://github.com/accel-sim/accel-sim-framework.git
```

---

### **Step 2: Install and Build the Tracer**

Navigate to the tracer directory and install NVBit:

```bash
cd /accel-sim/accel-sim-framework/util/tracer_nvbit/
./install_nvbit.sh
```

Compile the tracing tool:

```bash
make -j
```

---

### **Step 3: Generate Tracing Scripts**

Use the helper script to generate a tracing script for your application (e.g., MLPerf):

```bash
./run_hw_trace.py -B mlperf_inference -n
```

This creates a `hw_run` directory. Navigate to:

```bash
cd /accel-sim/accel-sim-framework/hw_run/traces/device-0/12.8/inference_mlperf_bert_test.sh/NO_ARGS/
```

---

### **Step 4: Configure Kernel Range to Trace**

Open the generated `run.sh` and locate the following environment variables:

```bash
export DYNAMIC_KERNEL_LIMIT_START=0
export DYNAMIC_KERNEL_LIMIT_END=0
```

These control which kernel IDs are traced.

To perform a **dry run** and list all launched kernels without tracing, set the start to a very high number:

```bash
export DYNAMIC_KERNEL_LIMIT_START=1000000
```

Then run the script:

```bash
./run.sh
```

> 📄 Kernel names and IDs will be logged in the stats files inside the traces directory.  
> Each layer has **12 kernels**, typically starting from kernel ID 13 to 24.

---

### **Step 5: Trace Selected Kernels**

Update `run.sh` with the kernel range you want to trace. For example, to trace the first 24 kernels:

```bash
export DYNAMIC_KERNEL_LIMIT_START=0
export DYNAMIC_KERNEL_LIMIT_END=24
```

Then rerun:

```bash
./run.sh
```

---

### **Step 6: Post-Processing**

After tracing, post-processing will run automatically to generate simulation-ready trace files.  
These processed traces will be stored in the corresponding directory under `traces`.  
`KernelsList.g` is the configuration file used by Accel-Sim to specify which GPU kernels should be simulated.

---

# 🚀 Simulating with Accel-Sim

### **Step 1: Compile the Simulator**

Navigate to the GPU simulator directory:

```bash
cd /accel-sim/accel-sim-framework/gpu-simulator/
```

Set up the environment:

```bash
source setup_environment.sh
```

Then compile the simulator:

```bash
make -j
```

---

### **Step 2: Run Simulation Using the Job Launching Script**

Navigate to the job launching utility:

```bash
cd /accel-sim/accel-sim-framework/util/job_launching/
```

Run the simulation with the following command:

```bash
./run_simulations.py -B <benchmark name> -C <GPU config name> -T <trace dir> -N <job name>
```

> 📌 **Example for BERT using NVIDIA RTX 3070 configuration**:

```bash
./run_simulations.py -B mlperf_inference -C RTX3070 -T /accel-sim/accel-sim-framework/hw_run/traces/device-0/12.8/ -N bert_simulation
```

---

---

# 📊 Profiling MLPerf BERT (NVIDIA Nsight Compute)

> ⚠️ **Note**: `ncu` has a bug and may get stuck. Use this alternative manual approach.

### **Step 1: Generate the Profiling Script**

Navigate to the hardware stats utility:

```bash
cd /accel-sim/accel-sim-framework/util/hw_stats/
./run_hw.py -B mlperf_inference -n
```

This creates a directory under:

```bash
cd accel-sim-framework/hw_run/device-0/12.8/inference_mlperf_bert_test.sh/NO_ARGS/
```

There you'll find a `run.sh` script with the computation command.

---

### **Step 2: Profile with Nsight Compute**

Copy the profiling command and prepend it to the execution line inside:

```bash
/root/MLC/repos/mlcommons@mlperf-automations/script/app-mlperf-inference-mlcommons-python/customize.py
```

Go to line 314 and prepend the following to the `cmd =` line:

```bash
ncu --metrics [LONG_METRICS_LIST] --csv --page raw --target-processes all -o /accel-sim/.../ncu_stat
```

Example format:

```python
cmd = "ncu --metrics ... --csv ... -o output_path " + env['MLC_PYTHON_BIN_WITH_PATH'] + " run.py --backend=..."
```

Replace `[LONG_METRICS_LIST]` with the actual metrics list.

---

### **Step 3: Profile for Cycles Only**

To collect only cycle data:

```bash
ncu --target-processes all --metrics gpc__cycles_elapsed.avg --csv -o /accel-sim/.../ncu_cycles.0
```

Follow the same approach: prepend this command to the `cmd` in `customize.py`.


