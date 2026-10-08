#include <vector>
#include <bitset>

#define WARP_SIZE 32


enum address_format {
  list_all = 0,
  base_stride = 1,
  base_delta = 2,
  tma_list_all = 3,
  tma_base_delta = 4
};

inline vector<uint64_t> decode(
    address_format address_mode,
    std::stringstream &ss,
    std::bitset<WARP_SIZE> mask_bits
) {
  switch(address_mode) {
    case address_format::list_all: return decode_list_all(ss, mask_bits);
    case address_format::base_stride: return decode_base_stride(ss, mask_bits);
    case address_format::base_delta: return decode_base_delta(ss, mask_bits);
    default: throw std::runtime_error("Unsupported address mode");
  }
}

inline vector<uint64_t> decode_list_all(
  std::stringstream &ss,
  std::bitset<WARP_SIZE> mask_bits,
) {
  vector<uint64_t> addrs;
  uint64_t addr = 0;

  // read addresses one by one from the file
  for (int s = 0; s < WARP_SIZE; s++) {
    if (mask_bits.test(s)) {
      ss >> std::hex >> addr;
      addrs.push_back(addr);
    } else addrs.push_back(0);
  }

  return addrs;
}

inline vector<uint64_t> decode_base_stride(
  std::stringstream &ss,
  std::bitset<WARP_SIZE> mask_bits
) {
  vector<uint64_t> addrs;
  uint64_t addr = 0;

  // read addresses as base address and stride
  int stride = 0;
  ss >> std::hex >> addr;
  ss >> std::dec >> stride;
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
    } else addrs.push_back(0);
  }

  return addrs;
}

inline vector<uint64_t> decode_base_delta(
  std::stringstream &ss,
  std::bitset<WARP_SIZE> mask_bits
) {
  vector<uint64_t> addrs;
  uint64_t addr = 0;
  int64_t delta;

  // read addresses as base address and deltas
  ss >> std::hex >> addr;
  bool first_bit1_found = false;
  for(int s = 0; s < WARP_SIZE; s++) {
    if(!first_bit1_found && mask_bits.test(s)) {
      first_bit1_found = true;
      addrs.push_back(addr);
    } else if(first_bit1_found && mask_bits.test(s)) {
      ss >> std::dec >> delta;
      addr += delta;
      addrs.push_back(addr);
    } else addrs.push_back(0);
  }

  return addrs;
}