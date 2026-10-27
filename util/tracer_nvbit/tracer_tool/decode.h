#pragma once

#include <bitset>
#include <cassert>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

constexpr int WARP_SIZE = 32;

enum address_format {
  list_all = 0,
  base_stride = 1,
  base_delta = 2,
  tma_list_all = 3,
  tma_base_delta = 4
};

inline std::vector<uint64_t> decode_list_all(std::stringstream &ss,
                                             std::bitset<WARP_SIZE> mask_bits) {
  std::vector<uint64_t> addrs;
  uint64_t addr = 0;

  // read addresses one by one from the file
  for (int s = 0; s < WARP_SIZE; s++) {
    if (mask_bits.test(s)) {
      ss >> std::hex >> addr;
      if (ss.fail())
        throw std::runtime_error("String stream input failed prematurely");
      addrs.push_back(addr);
    } else
      addrs.push_back(0);
  }
  assert(addrs.size() == WARP_SIZE);
  return addrs;
}

inline std::vector<uint64_t>
decode_base_stride(std::stringstream &ss, std::bitset<WARP_SIZE> mask_bits) {
  std::vector<uint64_t> addrs;
  uint64_t addr = 0;

  // read addresses as base address and stride
  int stride = 0;
  ss >> std::hex >> addr;
  ss >> std::dec >> stride;
  if (ss.fail())
    throw std::runtime_error("String stream input failed prematurely");
  bool first_bit1_found = false;
  bool last_bit1_found = false;

  for (int s = 0; s < WARP_SIZE; s++) {
    if (!first_bit1_found && mask_bits.test(s)) {
      first_bit1_found = true;
      addrs.push_back(addr);
    } else if (first_bit1_found && !last_bit1_found) {
      if (mask_bits.test(s)) {
        addr += stride;
        addrs.push_back(addr);
      } else
        last_bit1_found = true;
      addrs.push_back(0);
    } else
      addrs.push_back(0);
  }
  assert(addrs.size() == WARP_SIZE);
  return addrs;
}

inline std::vector<uint64_t>
decode_base_delta(std::stringstream &ss, std::bitset<WARP_SIZE> mask_bits) {
  std::vector<uint64_t> addrs;
  uint64_t addr = 0;
  int64_t delta = 0;

  // read addresses as base address and deltas
  ss >> std::hex >> addr;
  if (ss.fail())
    throw std::runtime_error("String stream input failed prematurely");
  bool first_bit1_found = false;
  for (int s = 0; s < WARP_SIZE; s++) {
    if (!first_bit1_found && mask_bits.test(s)) {
      first_bit1_found = true;
      addrs.push_back(addr);
    } else if (first_bit1_found && mask_bits.test(s)) {
      ss >> std::dec >> delta;
      if (ss.fail())
        throw std::runtime_error("String stream input failed prematurely");
      addr += delta;
      addrs.push_back(addr);
    } else
      addrs.push_back(0);
  }
  assert(addrs.size() == WARP_SIZE);
  return addrs;
}

inline std::vector<uint64_t> decode(unsigned address_mode,
                                    std::stringstream &ss,
                                    std::bitset<WARP_SIZE> mask_bits) {

  if (address_mode == 0)
    return decode_list_all(ss, mask_bits);
  if (address_mode == 1)
    return decode_base_stride(ss, mask_bits);
  if (address_mode == 2)
    return decode_base_delta(ss, mask_bits);
  if (address_mode == 3 || address_mode == 4)
    throw std::runtime_error(
        "decode() does not support TMA addressing modes as they are written "
        "usage-specific; reference usage-specific code at other callsites (for "
        "instance, trace_parser.cc or post-traces-processing.cpp) and do not "
        "invoke decode() here");
  throw std::runtime_error("Unsupported address mode");
}