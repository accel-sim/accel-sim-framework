# Accel-Sim Trace-Driven Front-end

![Accel-Sim Class Overview](https://accel-sim.github.io/assets/img/accel-sim-class.png)

The Accel-Sim's performance model relies on GPGPU-Sim 4.0 performance model. We created new classes with "exec_" and "trace_" prefix that are inherited from GPGPU-Sim performance model classes, then we moved some function implementations from the performance model to the new inherited classes using virtual functions. These functions are the ones that differ from exec-driven mode versus trace-driven mode. For example, when GPGPU-Sim calls the function "get_next_inst()", the exec_shader_core_ctx implementation will get the next instruction from the functional model, whereas the trace_shader_core_ctx will get the next inst from the traces.
The blue blocks in the image are maintained in GPGPU-Sim 4.0 repo [here](https://github.com/gpgpu-sim/gpgpu-sim_distribution), whereas the green blocks are maintained by Accel-Sim in this repo.

Our new frontend supports both vISA (PTX) execution-driven and mISA (SASS) trace-driven simulation. In traced-riven mode, mISA traces are converted into an ISA-independent intermediate representation, that has a 1:1 correspondence to the original SASS instructions. We generate the traces from NVIDIA GPUs using Accel-Sim’s tracer tool that is built on top of Nvbit. For further details about the tracer, see [this](https://github.com/accel-sim/accel-sim-framework/blob/dev/util/tracer_nvbit/README.md). These compatible traces are parsed by our [trace-parser](https://github.com/accel-sim/accel-sim-framework/tree/dev/gpu-simulator/trace-parser) component and feed up the performance model with these traces. The trace parser is a standalone component and can be utilized in other simulation engines for different use cases.

For each new GPU generation, we have to crease ISA_def file that specifies the SASS instruction types and where each instruction should be executed. For now, we have created the ISA_def files for NVIDIA's Kepler, Pascal, Turing and Volta generations. Please see the directory [./ISA_Def](./ISA_Def).
We were able to generate these files using the NVIDIA's CUDA Binary Utilities documentation from [here](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-ref).

# GPGPU-SIM 4.x

You do not need to clone the GPGPU-Sim 4.x performance model by yourself. The [./setup_environment.sh](./setup_environment.sh) will clone the recent GPGPU-Sim model and integrate it with Accel-Sim. For more info on the Accel-Sim front-end and how to compile, please see "Accel-Sim SASS Frontend" entry in the main read-me page [here](https://github.com/accel-sim/accel-sim-framework/blob/dev/README.md).

For more information about the major changes and new features in GPGPU-Sim 4.x, read [this](https://github.com/accel-sim/accel-sim-framework/blob/dev/gpu-simulator/gpgpu-sim4.md).

For more information about GPGPU-Sim, see the [original GPGPU-Sim manual](http://gpgpu-sim.org/manual/index.php/Main_Page).

The GPGPU-SIM 4.x integrated with Accel-Sim includes AccelWattch. For more information on AccelWattch, please see [AccelWattch Overview](https://github.com/VijayKandiah/accel-sim-framework#accelwattch-overview) entry in the main read-me page and the [AccelWattch MICRO'21 Artifact Manual](https://github.com/VijayKandiah/accel-sim-framework/blob/release/AccelWattch.md).
