# Accel-Sim Trace-Driven Front-end

![Accel-Sim Class Overview](https://accel-sim.github.io/assets/img/accel-sim-class.png)

The Accel-Sim's performance model relies on GPGPU-Sim 4.0 performance model. We created new classes with "exec_" and "trace_" prefix that are inherited from gpgpu-sim performance model classes, then we moved some function implementations from the performance model to the new inherited classes using virtual functions. These functions are the ones that differ from exec-driven mode versus trace-driven mode. For example, when gpgpu-sim call the function "get_next_inst()", the exec_shader_core_ctx implementation will get the next instruction from the functional model, whereas the trace_shader_core_ctx will get the next inst from the traces.
The blue blocks in the image are maintained in GPGPU-Sim 4.0 repo [here](https://github.com/gpgpu-sim/gpgpu-sim_distribution), whereas the green blocks are maintained by accel-sim in this repo.

For each new GPU generation, we have to crease ISA_def file that specifies the SASS instructions. For now, we have created the ISA_def files for NVIDIA's Kepler, Pascal, Turing and Volta generations. Please see the directory [./ISA_Def](./ISA_Def).
We were able to generate these files using the NVIDIA's CUDA Binary Utilities documentation from [here](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-ref).

You do not need to clone the GPGPU-Sim 4.0 by your self. The [./setup_environment.sh](./setup_environment.sh) will clone the recent GPGPU-Sim model and integrate it with Accel-Sim. For more info on the Accel-Sim front-end and how to compile, please see "Accel-Sim SASS Frontend" entry in the main read-me page [here](https://github.com/accel-sim/accel-sim-framework/blob/dev/README.md).
