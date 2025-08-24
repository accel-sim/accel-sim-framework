#include "common.h"
#include <bitset>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <trace_file>\n", argv[0]);
    return 1;
  }

  std::string filepath = argv[1];
  FILE *file;
  FILE *output_file;

  try {
    // Use utility function to open input file
    file = openFileForReading(filepath);

    // Open output file
    std::string output_filepath = filepath + ".txt";
    output_file = openFileForWriting(output_filepath, false);
  } catch (const std::runtime_error &e) {
    fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }

  // Read the kernel header
  std::string kernel_name;
  uint64_t name_size;
  fread(&name_size, sizeof(uint64_t), 1, file);

  kernel_name.resize(name_size);
  fread(kernel_name.data(), name_size, 1, file);

  // Read the kernel header
  kernel_header header;
  fread(&header, sizeof(kernel_header), 1, file);

  fprintf(output_file, "-kernel name = %s\n", kernel_name.c_str());
  fprintf(output_file, "-kernel id = %d\n", header.kernel_id);
  fprintf(output_file, "-grid dim = (%d,%d,%d)\n", header.grid_dim_x,
          header.grid_dim_y, header.grid_dim_z);
  fprintf(output_file, "-block dim = (%d,%d,%d)\n", header.block_dim_x,
          header.block_dim_y, header.block_dim_z);
  fprintf(output_file, "-shmem = %d\n", header.shared_mem_bytes);
  fprintf(output_file, "-nregs = %d\n", header.nregs);
  fprintf(output_file, "-binary version = %d\n", header.binary_version);
  fprintf(output_file, "-cuda stream id = %lu\n", header.cuda_stream_id);
  fprintf(output_file, "-shmem base_addr = 0x%016lx\n", header.shmem_base_addr);
  fprintf(output_file, "-local mem base_addr = 0x%016lx\n",
          header.local_mem_base_addr);
  fprintf(output_file, "-nvbit version = %s\n", header.nvbit_version);
  fprintf(output_file, "-accelsim tracer version = %s\n",
          header.accelsim_tracer_version);
  fprintf(output_file, "-enable lineinfo = %d\n", header.enable_lineinfo);

  // Read the traces
  unsigned tot_warp_in_id;
  unsigned tb_count = 0;
  while (fread(&tot_warp_in_id, sizeof(unsigned), 1, file) == 1) {
    // TB
    fprintf(output_file, "#BEGIN_TB\n");
    fprintf(output_file, "thread block = %d,%d,%d\n", tb_count, 0, 0);

    unsigned read_tb = 0;
    while (read_tb < tot_warp_in_id) {
      // WARP
      unsigned num_insts;
      fread(&num_insts, sizeof(unsigned), 1, file);
      fprintf(output_file, "warp = %d\n", read_tb);
      fprintf(output_file, "insts = %d\n", num_insts);

      unsigned read_inst = 0;
      while (read_inst < num_insts) {
        // INST
        sim_inst_u full_inst;
        inst_type_t inst_type;
        fread(&inst_type, sizeof(inst_type), 1, file);
        unsigned size;
        switch (inst_type) {
        case INST_BASE:
          size = sizeof(sim_inst_trace_t);
          fread(&full_inst.sim_inst_base, size, 1, file);
          break;
        case INST_FLAT:
          size = sizeof(sim_inst_trace_flat_t);
          fread(&full_inst.sim_inst_flat, size, 1, file);
          break;
        case INST_DELTA:
          size = sizeof(sim_inst_trace_delta_t);
          fread(&full_inst.sim_inst_delta, size, 1, file);
          break;
        case INST_STRIDE:
          size = sizeof(sim_inst_trace_stride_t);
          fread(&full_inst.sim_inst_stride, size, 1, file);
          break;
        default:
          assert(0);
          exit(1);
        }

        sim_inst_trace_t inst = full_inst.sim_inst_base;
        // print VPC
        fprintf(output_file, "%04x ", inst.vpc);

        // print active mask
        fprintf(output_file, "%08x ", inst.active_mask & inst.predicate_mask);

        // print GPRDst
        if (inst.GPRDst >= 0) {
          fprintf(output_file, "1 R%d ", inst.GPRDst);
        } else {
          fprintf(output_file, "0 ");
        }

        // print opcode
        fprintf(output_file, "%s ", inst.opcode);

        // print src count
        fprintf(output_file, "%d ", inst.numSrcs);

        // print GPRSrcs
        for (int i = 0; i < MAX_SRC; i++) {
          if (inst.GPRSrcs[i] >= 0) {
            fprintf(output_file, "R%d ", inst.GPRSrcs[i]);
          }
        }

        // print width
        fprintf(output_file, "%d ", inst.width);

        // print is_mem
        if (inst.is_mem) {
          std::bitset<32> mask(inst.active_mask & inst.predicate_mask);
          if (inst_type == INST_FLAT) {
            // default no compression for now
            fprintf(output_file, "0 ");

            for (int i = 0; i < 32; i++) {
              if (mask[i]) {
                fprintf(output_file, "0x%llx ",
                        (unsigned long long)full_inst.sim_inst_flat.addrs[i]);
              }
            }
          } else if (inst_type == INST_DELTA) {
            // enabled compression
            fprintf(output_file, "2 ");
            fprintf(output_file, "0x%llx ",
                    (unsigned long long)full_inst.sim_inst_delta.base_addr);

            for (int i = 0; i < 32; i++) {
              if (mask[i]) {
                fprintf(output_file, "%llx ",
                        (unsigned long long)full_inst.sim_inst_delta.base_addr +
                            full_inst.sim_inst_delta.delta[i]);
              }
            }
          } else if (inst_type == INST_STRIDE) {
            // enabled compression
            fprintf(output_file, "1 ");
            fprintf(output_file, "0x%llx ",
                    (unsigned long long)full_inst.sim_inst_stride.base_addr);

            for (int i = 0; i < 32; i++) {
              if (mask[i]) {
                fprintf(
                    output_file, "%llx ",
                    (unsigned long long)full_inst.sim_inst_stride.base_addr +
                        full_inst.sim_inst_stride.stride * i);
              }
            }
          }
        }

        // print imm
        fprintf(output_file, "%ld ", inst.imm);

        // newline
        fprintf(output_file, "\n");

        read_inst++;
      }
      read_tb++;
    }
    fprintf(output_file, "#END_TB\n");
    tb_count++;
  }

  fclose(file);
  fclose(output_file);

  return 0;
}