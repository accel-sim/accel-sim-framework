#include "common.h"
#include <fstream>
#include <iostream>
#include <string>

#define DEBUG 1

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
    output_file = fopen(output_filepath.c_str(), "w");
    if (!output_file) {
      throw std::runtime_error("Failed to open output file: " +
                               output_filepath);
    }
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

  if (DEBUG) {
    printf("-kernel id = %d\n", header.kernel_id);
    printf("-grid dim = (%d,%d,%d)\n", header.grid_dim_x, header.grid_dim_y,
           header.grid_dim_z);
    printf("-block dim = (%d,%d,%d)\n", header.block_dim_x, header.block_dim_y,
           header.block_dim_z);
    printf("-shmem = %d\n", header.shared_mem_bytes);
    printf("-nregs = %d\n", header.nregs);
    printf("-binary version = %d\n", header.binary_version);
    printf("-cuda stream id = %lu\n", header.cuda_stream_id);
    printf("-shmem base_addr = 0x%016lx\n", header.shmem_base_addr);
    printf("-local mem base_addr = 0x%016lx\n", header.local_mem_base_addr);
    printf("-nvbit version = %s\n", header.nvbit_version);
    printf("-accelsim tracer version = %s\n", header.accelsim_tracer_version);
    printf("-enable lineinfo = %d\n", header.enable_lineinfo);
  }

  // Read the traces
  inst_trace_t trace;
  unsigned tot_warp_in_id;
  unsigned read_tb = 0;
  while (fread(&tot_warp_in_id, sizeof(unsigned), 1, file) == 1) {
    // TB
    printf("Total warp in TB: %d\n", tot_warp_in_id);
    while (read_tb < tot_warp_in_id) {
      // WARP
      unsigned num_insts;
      fread(&num_insts, sizeof(unsigned), 1, file);
      printf("insts in warp: %d\n", num_insts);

      unsigned read_inst = 0;
      while (read_inst < num_insts) {
        // INST
        unsigned inst_size;
        fread(&inst_size, sizeof(unsigned), 1, file);
        fread(&trace, inst_size, 1, file);

        printf("Trace: %s\n", trace.opcode);

        // addr
        read_inst++;
      }
      read_tb++;
    }
  }

  return 0;
}