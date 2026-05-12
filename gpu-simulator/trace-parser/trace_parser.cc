// developed by Mahmoud Khairy, Purdue Univ
// abdallm@purdue.edu

#include <bits/stdc++.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <errno.h>
#include <signal.h>
#include <unistd.h>

#include "trace_parser.h"
#include "warp_trace_stream.h"

bool is_number(const std::string &s) {
  std::string::const_iterator it = s.begin();
  while (it != s.end() && std::isdigit(*it)) ++it;
  return !s.empty() && it == s.end();
}

void split(const std::string &str, std::vector<std::string> &cont,
           char delimi = ' ') {
  std::stringstream ss(str);
  std::string token;
  while (std::getline(ss, token, delimi)) {
    cont.push_back(token);
  }
}

inst_trace_t::inst_trace_t() {
  imm = 0;
  imm2 = 0;
}

inst_trace_t::~inst_trace_t() {}

bool inst_trace_t::check_opcode_contain(const std::vector<std::string> &opcode,
                                        std::string param) const {
  for (unsigned i = 0; i < opcode.size(); ++i)
    if (opcode[i] == param) return true;

  return false;
}

std::vector<std::string> inst_trace_t::get_opcode_tokens() const {
  std::istringstream iss(opcode);
  std::vector<std::string> opcode_tokens;
  std::string token;
  while (std::getline(iss, token, '.')) {
    if (!token.empty()) opcode_tokens.push_back(token);
  }
  return opcode_tokens;
}

unsigned inst_trace_t::get_datawidth_from_opcode(
    const std::vector<std::string> &opcode) const {
  for (unsigned i = 0; i < opcode.size(); ++i) {
    if (is_number(opcode[i])) {
      return (std::stoi(opcode[i], NULL) / 8);
    } else if (opcode[i][0] == 'U' && is_number(opcode[i].substr(1))) {
      // handle the U* case
      unsigned bits;
      sscanf(opcode[i].c_str(), "U%u", &bits);
      return bits / 8;
    }
  }

  return 4;  // default is 4 bytes
}

static bool hasEnding(const std::string &s, const std::string &suffix) {
  if (s.length() < suffix.length()) return false;
  return s.compare(s.length() - suffix.length(), suffix.length(), suffix) == 0;
}

kernel_trace_t::kernel_trace_t(const std::string &filePath) {
  kernel_name = filePath;
  shmem_base_addr = 0;
  local_base_addr = 0;
  binary_verion = 0;
  trace_verion = 0;

  if (hasEnding(filePath, ".tracez")) {
    is_tracez = true;
    // PipeReader left default-constructed (unused for .tracez)
  } else {
    is_tracez = false;
    pipeReader = PipeReader(filePath);
  }
}

kernel_trace_t::~kernel_trace_t() { delete tracez_reader; }

void inst_memadd_info_t::base_stride_decompress(
    unsigned long long base_address, int stride,
    const std::bitset<WARP_SIZE> &mask) {
  bool first_bit1_found = false;
  bool last_bit1_found = false;
  unsigned long long addra = base_address;
  for (int s = 0; s < WARP_SIZE; s++) {
    if (mask.test(s) && !first_bit1_found) {
      first_bit1_found = true;
      addrs[s] = base_address;
    } else if (first_bit1_found && !last_bit1_found) {
      if (mask.test(s)) {
        addra += stride;
        addrs[s] = addra;
      } else
        last_bit1_found = true;
    } else
      addrs[s] = 0;
  }
}

void inst_memadd_info_t::base_delta_decompress(
    unsigned long long base_address, const std::vector<long long> &deltas,
    const std::bitset<WARP_SIZE> &mask) {
  bool first_bit1_found = false;
  long long last_address = 0;
  unsigned delta_index = 0;
  for (int s = 0; s < 32; s++) {
    if (mask.test(s) && !first_bit1_found) {
      addrs[s] = base_address;
      first_bit1_found = true;
      last_address = base_address;
    } else if (mask.test(s) && first_bit1_found) {
      assert(delta_index < deltas.size());
      addrs[s] = last_address + deltas[delta_index++];
      last_address = addrs[s];
    } else
      addrs[s] = 0;
  }
}

void tma_inst_memaddr_info_t::base_delta_decompress(
    unsigned long long base_address, const std::vector<long long> &deltas,
    const std::bitset<WARP_SIZE> &mask) {
  if (mask.any()) {
    addrs.push_back(base_address);
    for (auto delta : deltas) {
      addrs.push_back(addrs.back() + delta);
    }
  }
}

bool inst_trace_t::parse_from_string(std::string trace, unsigned trace_version,
                                     unsigned enable_lineinfo,
                                     dim3 header_cta_id,
                                     dim3 header_cluster_cta_id,
                                     dim3 header_cluster_id,
                                     unsigned header_cluster_rank) {
  std::stringstream ss;
  ss.str(trace);

  std::string temp;

  // Start Parsing

  if (trace_version < 3) {
    // for older trace version, read the tb ids and ignore
    unsigned threadblock_x = 0, threadblock_y = 0, threadblock_z = 0,
             warpid_tb = 0;

    ss >> std::dec >> threadblock_x >> threadblock_y >> threadblock_z >>
        warpid_tb;
  } else {
    // Set the cta ids
    cta_id = header_cta_id;
    cluster_cta_id = header_cluster_cta_id;
    cluster_id = header_cluster_id;
    cluster_rank = header_cluster_rank;
  }
  if (enable_lineinfo) {
    ss >> std::dec >> line_num;
  }

  ss >> std::hex >> m_pc;
  ss >> std::hex >> mask;

  std::bitset<WARP_SIZE> mask_bits(mask);

  ss >> std::dec >> reg_dsts_num;
  assert(reg_dsts_num <= MAX_DST);
  auto parse_trace_reg_num = [&](trace_reg_t &reg) {
    std::string reg_str;
    ss >> reg_str;
    // Parse the register type and number
    if (reg_str.find("UR") != std::string::npos)
      reg.type = UREG;
    else if (reg_str.find("R") != std::string::npos)
      reg.type = REG;
    else if (reg_str.find("UP") != std::string::npos)
      reg.type = UPRED;
    else if (reg_str.find("P") != std::string::npos)
      reg.type = PRED;
    reg.num = std::stoi(reg_str.substr(reg_str.find_first_not_of("RURPUP")));
  };
  for (unsigned i = 0; i < reg_dsts_num; ++i) {
    parse_trace_reg_num(reg_dest[i]);
  }

  ss >> opcode;

  ss >> reg_srcs_num;
  assert(reg_srcs_num <= MAX_SRC);
  for (unsigned i = 0; i < reg_srcs_num; ++i) {
    parse_trace_reg_num(reg_src[i]);
  }

  // Parse is_gmma_commit_group flag if the instruction is a GMMA instruction
  if (strstr(opcode.c_str(), "GMMA") != NULL) {
    ss >> std::dec >> is_gmma_commit_group;
  }

  // parse mem info
  unsigned address_mode = 0;
  unsigned mem_width = 0;

  ss >> mem_width;
  // Check if the list of string is included in "opcode"
  std::vector<std::string> tma_opcodes = {"UTMALDG",  "UTMASTG", "UTMAPF",
                                          "UTMAREDG", "UBLKCP",  "UBLKPF",
                                          "UBLKRED"};
  bool is_tma = false;
  for (auto op : tma_opcodes) {
    if (opcode.find(op) != std::string::npos) {
      is_tma = true;
      break;
    }
  }

  if (is_tma) {
    tma_memadd_info = std::make_unique<tma_inst_memaddr_info_t>();
    tma_memadd_info->width = mem_width;
    ss >> std::hex >> tma_mbar_addr;
    ss >> std::dec >> tma_is_multicast;
    if (tma_is_multicast) {
      ss >> std::hex >> tma_multicast_cta_mask;
    }
    ss >> std::dec >> tma_byte_count;
    ss >> std::dec >> tma_oob_byte_count;
    ss >> std::dec >> address_mode;
    int32_t transfer_count = 0;
    if (address_mode == address_format::tma_list_all) {
      ss >> std::dec >> transfer_count;
      for (int s = 0; s < transfer_count; s++) {
        uint64_t addr = 0;
        ss >> std::hex >> addr;
        tma_memadd_info->addrs.push_back(addr);
      }
    } else if (address_mode == address_format::tma_base_delta) {
      uint64_t base_addr = 0;
      std::vector<long long> deltas;
      ss >> std::hex >> base_addr;
      ss >> std::dec >> transfer_count;
      // If there are actual accesses, need to parse the deltas
      if (transfer_count > 0) {
        for (int s = 1; s < transfer_count; s++) {
          long long delta = 0;
          ss >> std::dec >> delta;
          deltas.push_back(delta);
        }
        tma_memadd_info->base_delta_decompress(base_addr, deltas, mask_bits);
      } else {
        tma_memadd_info->addrs.clear();
      }
    } else {
      assert(0 && "Unsupported address mode");
    }
  }

  if (!is_tma && mem_width > 0) {
    // then it is a memory inst
    memadd_info = std::make_unique<inst_memadd_info_t>();

    // read the memory width from the opcode, as nvbit can report it incorrectly
    std::vector<std::string> opcode_tokens = get_opcode_tokens();
    memadd_info->width = get_datawidth_from_opcode(opcode_tokens);

    ss >> std::dec >> address_mode;
    if (address_mode == address_format::list_all) {
      // read addresses one by one from the file
      for (int s = 0; s < WARP_SIZE; s++) {
        if (mask_bits.test(s))
          ss >> std::hex >> memadd_info->addrs[s];
        else
          memadd_info->addrs[s] = 0;
      }
    } else if (address_mode == address_format::base_stride) {
      // read addresses as base address and stride
      unsigned long long base_address = 0;
      int stride = 0;
      ss >> std::hex >> base_address;
      ss >> std::dec >> stride;
      memadd_info->base_stride_decompress(base_address, stride, mask_bits);
    } else if (address_mode == address_format::base_delta) {
      unsigned long long base_address = 0;
      std::vector<long long> deltas;
      // read addresses as base address and deltas
      ss >> std::hex >> base_address;
      // No delta will be pushed if there is only one bit set in the mask
      // Find the first bit set
      int first_bit_set = -1;
      for (int s = 0; s < WARP_SIZE; s++) {
        if (mask_bits.test(s)) {
          first_bit_set = s;
          break;
        }
      }
      for (int s = first_bit_set + 1; s < WARP_SIZE; s++) {
        if (mask_bits.test(s)) {
          long long delta = 0;
          ss >> std::dec >> delta;
          deltas.push_back(delta);
        }
      }
      memadd_info->base_delta_decompress(base_address, deltas, mask_bits);
    } else {
      assert(0 && "Unsupported address mode");
    }
  }

  ss >> imm;
  imm2 = 0;
  if (trace_version == 6) {
    // Try to read a second immediate; if it fails (old traces), leave imm2=0.
    uint64_t tmp_imm2 = 0;
    if (ss >> tmp_imm2) {
      imm2 = tmp_imm2;
    } else {
      ss.clear();
    }
  }

  if (trace_version == 6) {
    // check Val or NoVal
    std::string val_or_no_val;
    ss >> val_or_no_val;
    if (val_or_no_val == "Val") {
      // Parse the register values
      auto parse_trace_reg_vals = [&](reg_val_t &reg_val) {
        uint32_t distinct_values;
        ss >> std::dec >> distinct_values;
        if (distinct_values == 1) {
          // All the values are the same
          uint64_t value;
          ss >> std::hex >> value;
          for (int j = 0; j < WARP_SIZE; j++) {
            reg_val[j] = value;
          }
        } else if (distinct_values == WARP_SIZE) {
          // Different values
          for (int j = 0; j < WARP_SIZE; j++) {
            ss >> std::hex >> reg_val[j];
          }
        } else {
          // Invalid number of distinct values
          assert(0 && "Invalid number of distinct values");
        }
      };
      // Resize the register values
      reg_dest_vals.resize(reg_dsts_num);
      reg_src_vals.resize(reg_srcs_num);
      // Parse the destination register values
      for (unsigned i = 0; i < reg_dsts_num; i++) {
        parse_trace_reg_vals(reg_dest_vals[i]);
      }
      // Parse the source register values
      for (unsigned i = 0; i < reg_srcs_num; i++) {
        parse_trace_reg_vals(reg_src_vals[i]);
      }
    }
  }

  // Finish Parsing

  return true;
}

trace_parser::trace_parser(const char *kernellist_filepath) {
  kernellist_filename = kernellist_filepath;
}

std::vector<trace_command> trace_parser::parse_commandlist_file() {
  std::ifstream fs;
  fs.open(kernellist_filename);

  if (!fs.is_open()) {
    std::cerr << "Unable to open file: " << kernellist_filename << std::endl;
    exit(1);
  }

  std::string directory(kernellist_filename);
  const size_t last_slash_idx = directory.rfind('/');
  if (std::string::npos != last_slash_idx) {
    directory = directory.substr(0, last_slash_idx);
  }

  std::string line, filepath;
  std::vector<trace_command> commandlist;
  while (!fs.eof()) {
    getline(fs, line);
    if (line.empty())
      continue;
    else if (line.substr(0, 10) == "MemcpyHtoD") {
      trace_command command;
      command.command_string = line;
      command.m_type = command_type::cpu_gpu_mem_copy;
      commandlist.push_back(command);
    } else if (line.substr(0, 6) == "kernel") {
      trace_command command;
      command.m_type = command_type::kernel_launch;
      filepath = directory + "/" + line;
      command.command_string = filepath;
      commandlist.push_back(command);
    }
    // ignore gpu_to_cpu_memory_cpy
  }

  fs.close();
  return commandlist;
}

void trace_parser::parse_memcpy_info(const std::string &memcpy_command,
                                     size_t &address, size_t &count) {
  std::vector<std::string> params;
  split(memcpy_command, params, ',');
  assert(params.size() == 3);
  std::stringstream ss;
  ss.str(params[1]);
  ss >> std::hex >> address;
  ss.clear();
  ss.str(params[2]);
  ss >> std::dec >> count;
}

// Parse a single kernel header line (shared between text and tracez paths)
static void parse_kernel_header_line(const std::string &line,
                                     kernel_trace_t *kernel_info) {
  if (line.empty() || line[0] != '-') return;

  std::stringstream ss;
  std::string string1, string2;
  ss.str(line);
  ss.ignore();
  ss >> string1 >> string2;

  if (string1 == "kernel" && string2 == "name") {
    const size_t equal_idx = line.find('=');
    kernel_info->kernel_name = line.substr(equal_idx + 2);
  } else if (string1 == "kernel" && string2 == "id") {
    sscanf(line.c_str(), "-kernel id = %d", &kernel_info->kernel_id);
  } else if (string1 == "grid" && string2 == "dim") {
    sscanf(line.c_str(), "-grid dim = (%d,%d,%d)", &kernel_info->grid_dim_x,
           &kernel_info->grid_dim_y, &kernel_info->grid_dim_z);
  } else if (string1 == "block" && string2 == "dim") {
    sscanf(line.c_str(), "-block dim = (%d,%d,%d)", &kernel_info->tb_dim_x,
           &kernel_info->tb_dim_y, &kernel_info->tb_dim_z);
  } else if (string1 == "shmem" && string2 == "=") {
    sscanf(line.c_str(), "-shmem = %d", &kernel_info->shmem);
  } else if (string1 == "nregs") {
    sscanf(line.c_str(), "-nregs = %d", &kernel_info->nregs);
  } else if (string1 == "cuda" && string2 == "stream") {
    sscanf(line.c_str(), "-cuda stream id = %llu",
           &kernel_info->cuda_stream_id);
  } else if (string1 == "binary" && string2 == "version") {
    sscanf(line.c_str(), "-binary version = %d", &kernel_info->binary_verion);
  } else if (string1 == "enable" && string2 == "lineinfo") {
    sscanf(line.c_str(), "-enable lineinfo = %d",
           &kernel_info->enable_lineinfo);
  } else if (string1 == "nvbit" && string2 == "version") {
    const size_t equal_idx = line.find('=');
    kernel_info->nvbit_verion = line.substr(equal_idx + 1);
  } else if (string1 == "accelsim" && string2 == "tracer") {
    sscanf(line.c_str(), "-accelsim tracer version = %d",
           &kernel_info->trace_verion);
  } else if (string1 == "shmem" && string2 == "base_addr") {
    const size_t equal_idx = line.find('=');
    ss.str(line.substr(equal_idx + 1));
    ss >> std::hex >> kernel_info->shmem_base_addr;
  } else if (string1 == "local" && string2 == "mem") {
    const size_t equal_idx = line.find('=');
    ss.str(line.substr(equal_idx + 1));
    ss >> std::hex >> kernel_info->local_base_addr;
  }
  std::cout << line << std::endl;
}

kernel_trace_t *trace_parser::parse_kernel_info(
    const std::string &kerneltraces_filepath) {
  std::cout << "Processing kernel " << kerneltraces_filepath << std::endl;
  kernel_trace_t *kernel_info = new kernel_trace_t(kerneltraces_filepath);
  kernel_info->enable_lineinfo = 0;  // default disabled

  if (kernel_info->is_tracez) {
    // .tracez path: use TracezReader to parse header and index
    kernel_info->tracez_reader = new TracezReader();
    std::string header_text =
        kernel_info->tracez_reader->open(kerneltraces_filepath);

    // Parse header lines from the header text
    std::istringstream iss(header_text);
    std::string line;
    while (std::getline(iss, line)) {
      if (line.empty()) continue;
      if (line[0] == '#') break;
      parse_kernel_header_line(line, kernel_info);
    }
  } else {
    // Text path: use PipeReader
    std::string line;
    while (kernel_info->pipeReader.readLine(line)) {
      if (line.length() == 0) {
        continue;
      } else if (line[0] == '#') {
        break;
      } else if (line[0] == '-') {
        parse_kernel_header_line(line, kernel_info);
        continue;
      }
    }
  }

  // do not close the file ifs, the kernel_finalizer will close it
  return kernel_info;
}

void trace_parser::kernel_finalizer(kernel_trace_t *trace_info) {
  assert(trace_info);

  // The pipe read/write end file descriptors held by the child process would
  // have been automatically closed when it terminated. But the parent
  // process may read an arbitrary amount of trace files, so it has to close
  // all file descriptors.
  delete trace_info;
}

void trace_parser::get_next_threadblock_traces(
    std::vector<std::vector<inst_trace_t> *> threadblock_traces,
    unsigned trace_version, unsigned enable_lineinfo,
    class PipeReader &pipeReader) {
  for (unsigned i = 0; i < threadblock_traces.size(); ++i) {
    threadblock_traces[i]->clear();
  }

  dim3 block_id;
  dim3 cluster_cta_id;
  dim3 cluster_id;
  unsigned cluster_rank = 0;
  bool start_of_tb_stream_found = false;

  unsigned warp_id = 0;
  unsigned insts_num = 0;
  unsigned inst_count = 0;
  std::string line;
  while (pipeReader.readLine(line)) {
    std::stringstream ss;
    std::string string1, string2;

    if (line.length() == 0) {
      continue;
    } else {
      ss.str(line);
      ss >> string1 >> string2;
      if (string1 == "#BEGIN_TB") {
        if (!start_of_tb_stream_found) {
          start_of_tb_stream_found = true;
        } else
          assert(0 &&
                 "Parsing error: thread block start before the previous one "
                 "finishes");
      } else if (string1 == "#END_TB") {
        assert(start_of_tb_stream_found);
        break;  // end of TB stream
      } else if (string1 == "thread" && string2 == "block") {
        assert(start_of_tb_stream_found);
        sscanf(line.c_str(), "thread block = %d,%d,%d", &block_id.x,
               &block_id.y, &block_id.z);
        std::cout << line << std::endl;
      } else if (string1 == "cluster" && string2 == "id") {
        assert(start_of_tb_stream_found);
        sscanf(line.c_str(), "cluster id = %d,%d,%d", &cluster_id.x,
               &cluster_id.y, &cluster_id.z);
        std::cout << line << std::endl;
      } else if (string1 == "cluster" && string2 == "cta") {
        assert(start_of_tb_stream_found);
        sscanf(line.c_str(), "cluster cta = %d,%d,%d", &cluster_cta_id.x,
               &cluster_cta_id.y, &cluster_cta_id.z);
        std::cout << line << std::endl;
      } else if (string1 == "cluster" && string2 == "rank") {
        assert(start_of_tb_stream_found);
        sscanf(line.c_str(), "cluster rank = %d", &cluster_rank);
        std::cout << line << std::endl;
      } else if (string1 == "warp") {
        // the start of new warp stream
        assert(start_of_tb_stream_found);
        sscanf(line.c_str(), "warp = %d", &warp_id);
      } else if (string1 == "insts") {
        assert(start_of_tb_stream_found);
        sscanf(line.c_str(), "insts = %d", &insts_num);
        threadblock_traces[warp_id]->resize(
            insts_num);  // allocate all the space at once
        inst_count = 0;
      } else {
        assert(start_of_tb_stream_found);
        threadblock_traces[warp_id]
            ->at(inst_count)
            .parse_from_string(line, trace_version, enable_lineinfo, block_id,
                               cluster_cta_id, cluster_id, cluster_rank);
        inst_count++;
      }
    }
  }
}

PipeReader::PipeReader(const std::string &filePath) { OpenFile(filePath); }

void PipeReader::OpenFile(const std::string &filePath) {
  if (hasEnding(filePath, ".xz")) {
    // Use xz command to decompress .xz files
    command = "xz -dc " + filePath;
  } else if (hasEnding(filePath, ".traceg")) {
    // Use cat command for regular trace files
    command = "cat " + filePath;
  } else {
    throw std::runtime_error("Unsupported file type!");
  }

  // Open the pipe
  pipe = popen(command.c_str(), "r");
  if (!pipe) {
    throw std::runtime_error("Failed to open pipe!");
  }
}

bool PipeReader::readLine(std::string &line) {
  char *buffer = nullptr;
  size_t len = 0;
  ssize_t nread;

  // Use getline() to read from the pipe
  if ((nread = getline(&buffer, &len, pipe)) != -1) {
    line.assign(buffer, nread);  // Assign the read line to the std::string
    assert(line.back() == '\n');
    line.pop_back();  // Remove the newline character
    free(buffer);     // Free the buffer allocated by getline
    return true;
  }

  free(buffer);  // Free the buffer if getline failed or reached EOF
  return false;  // End of pipe or error
}

// Helper function to check if a string ends with a specific suffix (file
// extension)
bool PipeReader::hasEnding(const std::string &fullString,
                           const std::string &ending) {
  if (fullString.length() >= ending.length()) {
    return (0 == fullString.compare(fullString.length() - ending.length(),
                                    ending.length(), ending));
  }
  return false;
}

PipeReader::PipeReader(PipeReader &&other) noexcept
    : pipe(other.pipe), command(other.command) {
  other.pipe = NULL;
  other.command = {};
}

PipeReader &PipeReader::operator=(PipeReader &&other) noexcept {
  if (this != &other) {
    pipe = other.pipe;
    command = other.command;

    other.pipe = NULL;
    other.command = {};
  }
  return *this;
}