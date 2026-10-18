#pragma once

#include <vector>
#include <bitset>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <cassert>

constexpr int WARP_SIZE = 32;

inline std::vector<uint64_t> decode_list_all(
  std::stringstream &ss,
  std::bitset<WARP_SIZE> mask_bits
) {
  std::vector<uint64_t> addrs;
  uint64_t addr = 0;

  // read addresses one by one from the file
  for (int s = 0; s < WARP_SIZE; s++) {
    if (mask_bits.test(s)) {
      ss >> std::hex >> addr;
      assert(!ss.fail());
      addrs.push_back(addr);
    } else
      addrs.push_back(0);
  }

  return addrs;
}

inline std::vector<uint64_t> decode_base_stride(
  std::stringstream &ss,
  std::bitset<WARP_SIZE> mask_bits
) {
  std::vector<uint64_t> addrs;
  uint64_t addr = 0;

  // read addresses as base address and stride
  int stride = 0;
  ss >> std::hex >> addr;
  ss >> std::dec >> stride;
  assert(!ss.fail());
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
    } else
      addrs.push_back(0);
  }

  return addrs;
}

inline std::vector<uint64_t> decode_base_delta(
  std::stringstream &ss,
  std::bitset<WARP_SIZE> mask_bits
) {
  std::vector<uint64_t> addrs;
  uint64_t addr = 0;
  int64_t delta = 0;

  // read addresses as base address and deltas
  ss >> std::hex >> addr;
  assert(!ss.fail());
  bool first_bit1_found = false;
  for (int s = 0; s < WARP_SIZE; s++) {
    if (!first_bit1_found && mask_bits.test(s)) {
      first_bit1_found = true;
      addrs.push_back(addr);
    } else if (first_bit1_found && mask_bits.test(s)) {
      ss >> std::dec >> delta;
      assert(!ss.fail());
      addr += delta;
      addrs.push_back(addr);
    } else
      addrs.push_back(0);
  }

  return addrs;
}

inline std::vector<uint64_t> decode(
    unsigned address_mode,
    std::stringstream &ss,
    std::bitset<WARP_SIZE> mask_bits
) {

    if(address_mode == 0) return decode_list_all(ss, mask_bits);
    if(address_mode == 1) return decode_base_stride(ss, mask_bits);
    if(address_mode == 2) return decode_base_delta(ss, mask_bits);
    throw std::runtime_error("Unsupported address mode");
  }