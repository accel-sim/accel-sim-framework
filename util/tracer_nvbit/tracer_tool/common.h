/* Author1: Mahmoud Khairy, abdallm@purdue.com - 2019 */
/* Author2: Jason Shen, shen203@purdue.edu - 2019 */

#ifndef COMMON_H
#define COMMON_H

#include <cstdint>
#include <cstdio>
#include <stdexcept>
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

/**
 * Opens a file for reading, automatically handling .xz decompression
 * @param filepath Path to the file to open
 * @return FILE pointer to the opened file/pipe, or nullptr on failure
 * @throws std::runtime_error if file type is unsupported or opening fails
 */
inline FILE *openFileForReading(const std::string &filepath) {
  FILE *file = nullptr;

  if (hasEnding(filepath, ".xz")) {
    // Use xz command to decompress .xz files
    std::string command = "xz -dc " + filepath;
    file = popen(command.c_str(), "r");
  } else if (hasEnding(filepath, ".trace") || hasEnding(filepath, ".traceg")) {
    // Use fopen for regular trace files
    file = fopen(filepath.c_str(), "rb");
  } else {
    throw std::runtime_error("Unsupported file type: " + filepath);
  }

  if (!file) {
    throw std::runtime_error("Failed to open file for reading: " + filepath);
  }

  return file;
}

/**
 * Opens a file for writing, automatically handling .xz compression
 * @param filepath Base filepath (without extension)
 * @param use_xz_compression Whether to compress with xz
 * @return FILE pointer to the opened file/pipe, or nullptr on failure
 * @throws std::runtime_error if opening fails
 */
inline FILE *openFileForWriting(const std::string &filepath,
                                bool use_xz_compression = false) {
  FILE *file = nullptr;

  if (use_xz_compression) {
    std::string command = "xz -1 -T0 > " + filepath + ".xz";
    file = popen(command.c_str(), "w");
    if (!file) {
      throw std::runtime_error(
          "Failed to open xz compression pipe for: " + filepath + ".xz");
    }
  } else {
    file = fopen(filepath.c_str(), "wb");
    if (!file) {
      throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
  }

  return file;
}

/**
 * Opens a file for writing with xz compression using a command buffer
 * @param base_filepath Base filepath (without extension)
 * @param cmd_buffer Buffer to store the xz command
 * @param buffer_size Size of the command buffer
 * @return FILE pointer to the opened pipe, or nullptr on failure
 * @throws std::runtime_error if buffer is too small or opening fails
 */
inline FILE *openFileForWritingXz(const std::string &base_filepath,
                                  char *cmd_buffer, size_t buffer_size) {
  if (snprintf(cmd_buffer, buffer_size, "xz -1 -T0 > %s.xz",
               base_filepath.c_str()) >= (int)buffer_size) {
    throw std::runtime_error("Command buffer too small for xz compression");
  }

  FILE *file = popen(cmd_buffer, "w");
  if (!file) {
    throw std::runtime_error(
        "Failed to open xz compression pipe for: " + base_filepath + ".xz");
  }

  return file;
}

/**
 * Generates output filepath based on input filepath and desired extension
 * @param input_filepath Input file path
 * @param new_extension New extension to append (without dot)
 * @return Output filepath with new extension
 */
inline std::string generateOutputFilepath(const std::string &input_filepath,
                                          const std::string &new_extension) {
  if (hasEnding(input_filepath, ".xz")) {
    // Remove .xz and add new extension
    return input_filepath.substr(0, input_filepath.find_last_of(".")) +
           new_extension;
  } else {
    // Add new extension to existing path
    return input_filepath + "." + new_extension;
  }
}

#endif