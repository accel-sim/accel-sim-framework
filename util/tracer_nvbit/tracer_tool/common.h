/* Author1: Mahmoud Khairy, abdallm@purdue.com - 2019 */
/* Author2: Jason Shen, shen203@purdue.edu - 2019 */

#ifndef COMMON_H
#define COMMON_H

#include <cstdint>
#include <stdint.h>
#include <string>

/* information collected in the instrumentation function and passed
 * on the channel from the GPU to the CPU */
#define MAX_SRC 5
#define MAX_OPCODE_LENGTH 32

typedef struct {
  unsigned kernel_id;
  unsigned grid_dim_x;
  unsigned grid_dim_y;
  unsigned grid_dim_z;
  unsigned block_dim_x;
  unsigned block_dim_y;
  unsigned block_dim_z;
  unsigned shared_mem_bytes;
  unsigned shmem;
  unsigned nregs;
  unsigned binary_version;
  uint64_t cuda_stream_id;
  uint64_t shmem_base_addr;
  uint64_t local_mem_base_addr;
  char nvbit_version[10];
  char accelsim_tracer_version[10];
  bool enable_lineinfo;
} kernel_header;

typedef struct {
  int cta_id_x;
  int cta_id_y;
  int cta_id_z;
  int warpid_tb;
  int warpid_sm;
  int sm_id;
  int opcode_id;
  char opcode[MAX_OPCODE_LENGTH];
  uint32_t line_num;
  uint32_t vpc;
  bool is_mem;
  int32_t GPRDst;
  int32_t GPRSrcs[MAX_SRC];
  int32_t numSrcs;
  int32_t width;
  uint32_t active_mask;
  uint32_t predicate_mask;
  uint64_t imm;

  // variable size
  uint64_t addrs[32];
} inst_trace_t;

inline bool hasEnding(const std::string &fullString,
                      const std::string &ending) {
  if (fullString.length() >= ending.length()) {
    return (0 == fullString.compare(fullString.length() - ending.length(),
                                    ending.length(), ending));
  }
  return false;
}

#endif