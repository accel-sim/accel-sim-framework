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

#include "trace_parser.h"

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

inst_trace_t::inst_trace_t() { memadd_info = NULL; }

inst_trace_t::~inst_trace_t() {
  if (memadd_info != NULL) delete memadd_info;
}

inst_trace_t::inst_trace_t(const inst_trace_t &b) {
  if (memadd_info != NULL) {
    memadd_info = new inst_memadd_info_t();
    memadd_info = b.memadd_info;
  }
}

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

kernel_trace_t::kernel_trace_t() {
  kernel_name = "Empty";
  shmem_base_addr = 0;
  local_base_addr = 0;
  binary_verion = 0;
  trace_verion = 0;
}

void inst_memadd_info_t::base_stride_decompress(
    unsigned long long base_address, int stride,
    unsigned mask) {
  bool first_bit1_found = false;
  bool last_bit1_found = false;
  unsigned long long addra = base_address;
  for (int s = 0; s < warp_size; s++) {
    if (((mask >> s) & 0x1) && !first_bit1_found) {
      first_bit1_found = true;
      addrs[s] = base_address;
    } else if (first_bit1_found && !last_bit1_found) {
      if ((mask >> s) & 0x1) {
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
    unsigned mask) {
  bool first_bit1_found = false;
  long long last_address = 0;
  unsigned delta_index = 0;
  for (int s = 0; s < warp_size; s++) {
    if (((mask >> s) & 0x1) && !first_bit1_found) {
      addrs[s] = base_address;
      first_bit1_found = true;
      last_address = base_address;
    } else if (((mask >> s) & 0x1) && first_bit1_found) {
      assert(delta_index < deltas.size());
      addrs[s] = last_address + deltas[delta_index++];
      last_address = addrs[s];
    } else
      addrs[s] = 0;
  }
}

void inst_trace_t::set_warp_size(unsigned given_warp_size) {
  this->warp_size = given_warp_size;
}

// Parse every trace in string to instruction type?
bool inst_trace_t::parse_from_string(std::string trace,
                                     unsigned trace_version) {
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
  }

  ss >> std::hex >> m_pc;
  ss >> std::hex >> mask;

  ss >> std::dec >> reg_dsts_num;
  assert(reg_dsts_num <= MAX_DST);
  for (unsigned i = 0; i < reg_dsts_num; ++i) {
    ss >> temp;
    // Handle different reg types
    const char * regName = temp.c_str();
    if (regName[0] == 'R' && isdigit(regName[1])) {  // Vector register
      sscanf(temp.c_str(), "R%d", &reg_dest[i]);
    } else if (regName[0] == 'S' && isdigit(regName[1])) { // Scalar register, treated as vector reg for now
      // TODO Might cause conflict with the vector regs?
      sscanf(temp.c_str(), "S%d", &reg_dest[i]);
    } else { // Special registers, ignored
      // do nothing
    }
  }

  ss >> opcode;

  ss >> reg_srcs_num;
  assert(reg_srcs_num <= MAX_SRC);
  for (unsigned i = 0; i < reg_srcs_num; ++i) {
    ss >> temp;
    // Handle different reg types
    const char * regName = temp.c_str();
    if (regName[0] == 'R' && isdigit(regName[1])) {  // Vector register
      sscanf(temp.c_str(), "R%d", &reg_dest[i]);
    } else if (regName[0] == 'S' && isdigit(regName[1])) { // Scalar register, treated as vector reg for now
      // TODO Might cause conflict with the vector regs?
      sscanf(temp.c_str(), "S%d", &reg_dest[i]);
    } else { // Special registers, ignored
      // do nothing
    }
  }

  // parse mem info
  unsigned address_mode = 0;
  unsigned mem_width = 0;

  ss >> mem_width;

  if (mem_width > 0)  // then it is a memory inst
  {
    memadd_info = new inst_memadd_info_t();
    memadd_info->warp_size = warp_size;
    memadd_info->addrs.resize(warp_size);

    // read the memory width from the opcode, as nvbit can report it incorrectly
    // TODO Ignore this for AMD GPUs
    if (warp_size == DEFAULT_WARP_SIZE) {
      std::vector<std::string> opcode_tokens = get_opcode_tokens();
      memadd_info->width = get_datawidth_from_opcode(opcode_tokens);
    } else {  // MGPUSim trace has accurate width info
      memadd_info->width = mem_width;
    }


    ss >> std::dec >> address_mode;
    if (address_mode == address_format::list_all) {
      // read addresses one by one from the file
      for (int s = 0; s < warp_size; s++) {
        if ((mask >> s) & 0x1)
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
      memadd_info->base_stride_decompress(base_address, stride, mask);
    } else if (address_mode == address_format::base_delta) {
      unsigned long long base_address = 0;
      std::vector<long long> deltas;
      // read addresses as base address and deltas
      ss >> std::hex >> base_address;
      for (int s = 0; s < warp_size; s++) {
        if ((mask >> s) & 0x1) {
          long long delta = 0;
          ss >> std::dec >> delta;
          deltas.push_back(delta);
        }
      }
      memadd_info->base_delta_decompress(base_address, deltas, mask);
    }
  }
  // Finish Parsing

  return true;
}

trace_parser::trace_parser(const char *kernellist_filepath) {
  kernellist_filename = kernellist_filepath;
  warp_size = DEFAULT_WARP_SIZE;
}

// Read the kernel commandlist file: kernelslist.g
std::vector<trace_command> trace_parser::parse_commandlist_file() {
  ifs.open(kernellist_filename);

  if (!ifs.is_open()) {
    std::cout << "Unable to open file: " << kernellist_filename << std::endl;
    exit(1);
  }

  std::string directory(kernellist_filename);
  const size_t last_slash_idx = directory.rfind('/');
  if (std::string::npos != last_slash_idx) {
    directory = directory.substr(0, last_slash_idx);
  }

  std::string line, filepath;
  std::vector<trace_command> commandlist;
  while (!ifs.eof()) {
    getline(ifs, line);
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

  ifs.close();
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

// Read the meta info of .traceg file
kernel_trace_t trace_parser::parse_kernel_info(
    const std::string &kerneltraces_filepath) {
  ifs.open(kerneltraces_filepath.c_str());

  if (!ifs.is_open()) {
    std::cout << "Unable to open file: " << kerneltraces_filepath << std::endl;
    exit(1);
  }

  std::cout << "Processing kernel " << kerneltraces_filepath << std::endl;

  kernel_trace_t kernel_info;

  // Default warp size
  kernel_info.warp_size = DEFAULT_WARP_SIZE;

  std::string line;

  while (!ifs.eof()) {
    getline(ifs, line);

    if (line.length() == 0) {
      continue;
    } else if (line[0] == '#') {
      // the trace format, ignore this and assume fixed format for now
      break;  // the begin of the instruction stream
    } else if (line[0] == '-') {
      // Possess header configs
      std::stringstream ss;
      std::string string1, string2;

      ss.str(line);
      ss.ignore();
      ss >> string1 >> string2;

      if (string1 == "kernel" && string2 == "name") {
        const size_t equal_idx = line.find('=');
        kernel_info.kernel_name = line.substr(equal_idx + 2);
      } else if (string1 == "kernel" && string2 == "id") {
        sscanf(line.c_str(), "-kernel id = %d", &kernel_info.kernel_id);
      } else if (string1 == "grid" && string2 == "dim") {
        sscanf(line.c_str(), "-grid dim = (%d,%d,%d)", &kernel_info.grid_dim_x,
               &kernel_info.grid_dim_y, &kernel_info.grid_dim_z);
      } else if (string1 == "block" && string2 == "dim") {
        sscanf(line.c_str(), "-block dim = (%d,%d,%d)", &kernel_info.tb_dim_x,
               &kernel_info.tb_dim_y, &kernel_info.tb_dim_z);
      } else if (string1 == "shmem" && string2 == "=") {
        /**
         * shmem = Static shmem bytes + p->sharedMemBytes
         * 
         * Static part, from https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html
         * The size in bytes of statically-allocated shared memory per block required 
         * by this function. This does not include dynamically-allocated shared memory 
         * requested by the user at runtime.
         * 
         * p->sharedMemBytes
         */
        sscanf(line.c_str(), "-shmem = %d", &kernel_info.shmem);
      } else if (string1 == "nregs") {  
        // The number of registers used by each thread of this function.
        sscanf(line.c_str(), "-nregs = %d", &kernel_info.nregs);
      } else if (string1 == "cuda" && string2 == "stream") {
        sscanf(line.c_str(), "-cuda stream id = %d",
               &kernel_info.cuda_stream_id);
      } else if (string1 == "binary" && string2 == "version") { 
        // SASS version? Used to differentiate different NVIDIA arch
        /**
         * From https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html
         * The binary architecture version for which the function was compiled. 
         * This value is the major binary version * 10 + the minor binary version, 
         * so a binary version 1.3 function would return the value 13. Note that this 
         * will return a value of 10 for legacy cubins that do not have a 
         * properly-encoded binary architecture version.
         */
        sscanf(line.c_str(), "-binary version = %d",
               &kernel_info.binary_verion);
      } else if (string1 == "nvbit" && string2 == "version") {
        const size_t equal_idx = line.find('=');
        kernel_info.nvbit_verion = line.substr(equal_idx + 1);

      } else if (string1 == "accelsim" && string2 == "tracer") {
        sscanf(line.c_str(), "-accelsim tracer version = %d",
               &kernel_info.trace_verion);

      } else if (string1 == "shmem" && string2 == "base_addr") {
        /** 
         * From uint64_t nvbit_get_shmem_base_addr(CUcontext cuctx);
         * 
         * Allows to get shmem base address from CUcontext
         * shmem range is [shmem_base_addr, shmem_base_addr+16MB) and
         * the base address is 16MB aligned.  
         */
        const size_t equal_idx = line.find('=');
        ss.str(line.substr(equal_idx + 1));
        ss >> std::hex >> kernel_info.shmem_base_addr;

      } else if (string1 == "local" && string2 == "mem") {
        /**
         * From uint64_t nvbit_get_local_mem_base_addr(CUcontext cuctx);
         * 
         * Allows to get local memory base address from CUcontext
         * local mem range is [shmem_base_addr, shmem_base_addr+16MB) and
         * the base address is 16MB aligned.  
         */
        const size_t equal_idx = line.find('=');
        ss.str(line.substr(equal_idx + 1));
        ss >> std::hex >> kernel_info.local_base_addr;
      } else if (string1 == "warp" && string2 == "size") {
        // Get the warp size info, default is 32
        int tmp;
        sscanf(line.c_str(), "-warp size = %d",
               &tmp);
        kernel_info.warp_size = tmp;
        this->warp_size = tmp;
      }
      std::cout << line << std::endl;
      continue;
    }
  }

  // do not close the file ifs, the kernel_finalizer will close it
  return kernel_info;
}

void trace_parser::kernel_finalizer() {
  if (ifs.is_open()) ifs.close();
}

// Parse the trace of each threadblock that gets executed in the kernel 
bool trace_parser::get_next_threadblock_traces(
    std::vector<std::vector<inst_trace_t> *> threadblock_traces,
    unsigned trace_version) {
  for (unsigned i = 0; i < threadblock_traces.size(); ++i) {
    threadblock_traces[i]->clear();
  }

  unsigned block_id_x = 0, block_id_y = 0, block_id_z = 0;
  bool start_of_tb_stream_found = false;

  unsigned warp_id = 0;
  unsigned insts_num = 0;
  unsigned inst_count = 0;

  while (!ifs.eof()) {
    std::string line;
    std::stringstream ss;
    std::string string1, string2;

    getline(ifs, line);

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
        sscanf(line.c_str(), "thread block = %d,%d,%d", &block_id_x,
               &block_id_y, &block_id_z);
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
        // Set the warp size for the trace
        threadblock_traces[warp_id]
            ->at(inst_count)
            .set_warp_size(warp_size);
        threadblock_traces[warp_id]
            ->at(inst_count)
            .parse_from_string(line, trace_version);
        inst_count++;
      }
    }
  }

  return true;
}
