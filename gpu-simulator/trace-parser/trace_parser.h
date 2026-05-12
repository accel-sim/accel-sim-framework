// developed by Mahmoud Khairy, Purdue Univ

#ifndef TRACE_PARSER_H
#define TRACE_PARSER_H
#include <assert.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <array>
#include <bitset>
#include <memory>
#include <string>
#include <vector>

#define WARP_SIZE 32
#define MAX_DST 1
#define MAX_SRC 4

enum command_type {
  kernel_launch = 1,
  cpu_gpu_mem_copy,
  gpu_cpu_mem_copy,
};

enum address_space { GLOBAL_MEM = 1, SHARED_MEM, LOCAL_MEM, TEX_MEM };

enum address_scope {
  L1_CACHE = 1,
  L2_CACHE,
  SYS_MEM,
};

// TODO Fix this shared enum with tracer tool
enum address_format {
  list_all = 0,
  base_stride = 1,
  base_delta = 2,
  tma_list_all = 3,
  tma_base_delta = 4
};

struct trace_command {
  std::string command_string;
  command_type m_type;
};

struct inst_memadd_info_t {
  uint64_t addrs[WARP_SIZE];
  int32_t width;

  void base_stride_decompress(unsigned long long base_address, int stride,
                              const std::bitset<WARP_SIZE> &mask);
  void base_delta_decompress(unsigned long long base_address,
                             const std::vector<long long> &deltas,
                             const std::bitset<WARP_SIZE> &mask);
};

struct tma_inst_memaddr_info_t {
  std::vector<uint64_t> addrs;
  int32_t width;
  void base_delta_decompress(unsigned long long base_address,
                             const std::vector<long long> &deltas,
                             const std::bitset<WARP_SIZE> &mask);
};

typedef std::array<uint64_t, WARP_SIZE> reg_val_t;
typedef enum {
  REG = 0,
  UREG,
  PRED,
  UPRED,
} trace_reg_type_t;

typedef struct {
  uint32_t num;
  trace_reg_type_t type;

} trace_reg_t;

struct inst_trace_t {
  inst_trace_t();
  inst_trace_t(const inst_trace_t &) = delete;
  inst_trace_t &operator=(const inst_trace_t &) = delete;
  inst_trace_t(inst_trace_t &&) = default;
  inst_trace_t &operator=(inst_trace_t &&) = default;

  dim3 cta_id;
  dim3 cluster_cta_id;
  dim3 cluster_id;
  unsigned cluster_rank;
  unsigned line_num;
  unsigned m_pc;
  unsigned mask;
  unsigned reg_dsts_num;
  trace_reg_t reg_dest[MAX_DST];
  std::vector<reg_val_t> reg_dest_vals;
  std::string opcode;
  unsigned reg_srcs_num;
  trace_reg_t reg_src[MAX_SRC];
  std::vector<reg_val_t> reg_src_vals;
  unsigned is_gmma_commit_group;
  uint64_t imm;
  // Optional second immediate value (used e.g. by Hopper BAR instructions)
  uint64_t imm2;
  std::unique_ptr<inst_memadd_info_t> memadd_info;
  std::unique_ptr<tma_inst_memaddr_info_t> tma_memadd_info;
  bool tma_is_multicast = false;
  uint16_t tma_multicast_cta_mask = 0;
  uint32_t tma_mbar_addr = 0;
  size_t tma_byte_count = 0;
  size_t tma_oob_byte_count = 0;

  bool parse_from_string(std::string trace, unsigned tracer_version,
                         unsigned enable_lineinfo,
                         dim3 header_cta_id = dim3(-1, -1, -1),
                         dim3 header_cluster_cta_id = dim3(-1, -1, -1),
                         dim3 header_cluster_id = dim3(-1, -1, -1),
                         unsigned header_cluster_rank = 0);

  bool check_opcode_contain(const std::vector<std::string> &opcode,
                            std::string param) const;

  unsigned get_datawidth_from_opcode(
      const std::vector<std::string> &opcode) const;

  std::vector<std::string> get_opcode_tokens() const;

  ~inst_trace_t();
};

class PipeReader {
 public:
  PipeReader() {}  // default: no file opened
  PipeReader(const std::string &filePath);

  // Destructor to close the pipe
  ~PipeReader() {
    if (pipe) {
      pclose(pipe);  // Close the pipe when done
    }
  }

  // It does not make sense to implement copy semantics for PipeReader,
  // because each instance should hold a unique Linux pipe handle
  PipeReader(const PipeReader &) = delete;
  PipeReader &operator=(const PipeReader &) = delete;

  // Move semantics can be supported
  PipeReader(PipeReader &&) noexcept;
  PipeReader &operator=(PipeReader &&) noexcept;

  // Read one line
  bool readLine(std::string &line);

 private:
  FILE *pipe = NULL;    // Store the pipe
  std::string command;  // Store the shell command to be executed

  // Helper function to check if a string ends with a specific suffix (file
  // extension)
  bool hasEnding(const std::string &fullString, const std::string &ending);

  void OpenFile(const std::string &filePath);
};

class TracezReader;  // forward declaration

struct kernel_trace_t {
  kernel_trace_t(const std::string &filePath);
  ~kernel_trace_t();

  std::string kernel_name;
  unsigned kernel_id;
  unsigned grid_dim_x;
  unsigned grid_dim_y;
  unsigned grid_dim_z;
  unsigned tb_dim_x;
  unsigned tb_dim_y;
  unsigned tb_dim_z;
  unsigned shmem;
  unsigned nregs;
  unsigned long long cuda_stream_id;
  unsigned binary_verion;
  unsigned enable_lineinfo;
  unsigned trace_verion;
  std::string nvbit_verion;
  unsigned long long shmem_base_addr;
  unsigned long long local_base_addr;

  // Text/pipe path (used for .traceg / .traceg.xz)
  PipeReader pipeReader;

  // .tracez path
  bool is_tracez = false;
  TracezReader *tracez_reader = nullptr;  // owned, non-null when is_tracez
};

class trace_parser {
 public:
  trace_parser() {}
  trace_parser(const char *kernellist_filepath);

  std::vector<trace_command> parse_commandlist_file();

  kernel_trace_t *parse_kernel_info(const std::string &kerneltraces_filepath);

  void parse_memcpy_info(const std::string &memcpy_command, size_t &add,
                         size_t &count);

  void get_next_threadblock_traces(
      std::vector<std::vector<inst_trace_t> *> threadblock_traces,
      unsigned trace_version, unsigned enable_lineinfo,
      class PipeReader &pipeReader);

  void kernel_finalizer(kernel_trace_t *trace_info);

 private:
  std::string kernellist_filename;
};

#endif
