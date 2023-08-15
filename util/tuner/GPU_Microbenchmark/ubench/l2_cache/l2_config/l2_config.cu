#include <iostream>
using namespace std;

#include "../../../hw_def/hw_def.h"

// We know the below information from running our ubench, we copy and paste the
// ubench results below manullay
// TODO: we will automate this process

// we know sector size from l2_access_grain ubench
#define L2_CACHE_LINE_SIZE 128
#define L2_SECTOR_SIZE 32
#define IS_SECTOR 1

// It is hard to know the exact l2 assoc from ubenhmarking
// Thus, based on previous work, we assume assoc is constant and = 16
// similar to AMD GPU:
// https://www.techpowerup.com/gpu-specs/docs/amd-gcn1-architecture.pdf
#define L2_CACHE_ASSOC 16

// L2 cache cache since kepler and above is write-allocate, subsector-write,
// write-back. We know that from l2_write_policy ubench and has been consistent
// since kepler. Change it accordingly if it changes in new generations
static const char *L2_Cache_Write_Policy = ",L:B:m:L:";

// For now, accel-sim only supoprts ipoly for 64 and less
#define ACCELSIM_IPOLY_HASH_SUPPORT 64
// 8 byte for icnt control
#define ACCELSIM_ICNT_CONTROL 8

int main() {
  intilizeDeviceProp(0);

  if (deviceProp.l2CacheSize) {
    printf("L2 Cache Size = %.0f MB\n",
           static_cast<float>(deviceProp.l2CacheSize / 1048576.0f));
  }

  unsigned mem_channel = get_num_channels(MEM_BITWIDTH, DRAM_MODEL);
  unsigned l2_banks_num = mem_channel * L2_BANKS_PER_MEM_CHANNEL;

  std::cout << "L2 Banks number = " << l2_banks_num << std::endl;

  if (ACCEL_SIM_MODE) {

    std::cout << "\n//Accel_Sim config: \n";

    unsigned l2_size_per_bank = L2_SIZE / l2_banks_num;
    unsigned assoc, sets_num;
    char set_indexing = 'L'; // by default assume linear indexing
    char is_sector = IS_SECTOR ? 'S' : 'N';
    if (isPowerOfTwo(l2_size_per_bank)) {
      assoc = L2_CACHE_ASSOC;
      sets_num = l2_size_per_bank / L2_CACHE_LINE_SIZE / assoc;
      if (sets_num <= ACCELSIM_IPOLY_HASH_SUPPORT)
        set_indexing = 'P';
      else
        set_indexing = 'X'; // bitwise xoring
    } else {
      // if not power of two, assume it is 24, as most NVidia GPU L2 cache size
      // that is not power of two, is actually divisble by 24
      assoc = 24;
      // ensure that our assumption is true
      assert((l2_size_per_bank / L2_CACHE_LINE_SIZE) % assoc == 0);
      sets_num = l2_size_per_bank / L2_CACHE_LINE_SIZE / assoc;
      if (isPowerOfTwo(sets_num) && l2_banks_num <= ACCELSIM_IPOLY_HASH_SUPPORT)
        set_indexing = 'P';
      else if (isPowerOfTwo(sets_num))
        set_indexing = 'X'; // bitwise xoring
    }

    std::cout << "-gpgpu_n_sub_partition_per_mchannel "
              << L2_BANKS_PER_MEM_CHANNEL << std::endl;
    std::cout << "-icnt_flit_size "
              << L2_BANK_WIDTH_in_BYTE + ACCELSIM_ICNT_CONTROL
              << std::endl; // 8bytes for control
    if (isPowerOfTwo(l2_banks_num) &&
        l2_banks_num <= ACCELSIM_IPOLY_HASH_SUPPORT)
      std::cout << "-gpgpu_memory_partition_indexing 2" << std::endl;
    else
      std::cout << "-gpgpu_memory_partition_indexing 0" << std::endl;
    std::cout << "-gpgpu_cache:dl2 " << is_sector << ":" << sets_num << ":"
              << L2_CACHE_LINE_SIZE << ":" << assoc << L2_Cache_Write_Policy
              << set_indexing << ","
              << "A:192:4,32:0,32" << std::endl;
  }

  return 1;
}
