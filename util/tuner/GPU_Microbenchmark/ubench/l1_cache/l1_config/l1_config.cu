#include <iostream>
#include <sstream>
using namespace std;

#include "../../../hw_def/hw_def.h"

/*
We know the below information from running our ubench, we copy and paste the
ubench results below manullay
TODO: we will automate this process
*/

// We know cache line size from l1_assoc ubench
#define L1_CACHE_LINE_SIZE 128

// We know #sets from l1_assoc ubench (the l1 cache has 4 sets, since kepler
// and in volta and turing)
#define L1_CACHE_SETS 4

// we know sector size from l1_assoc and l1_acces_grain ubenches and has
// been consistent over generations, change it accordingly
#define L1_SECTOR_SIZE 32

// we know the mshr throughput from l1_mshr ubench
// we find that each warp can issue up to two pending cache lines (8 sector
// reqs)
#define L1_ACCESS_FACTOR L1_CACHE_LINE_SIZE / L1_SECTOR_SIZE
#define L1_MSHR_ENTRIES_PER_WARP L1_ACCESS_FACTOR * 2

// L1 cache cache in Volta and above is write allocate, subsector write, write-
// through we know that from l1_write_policy ubench and has been consistent
// after Volta. Change it accordingly if it changes in new generations
static const char *After_Volta_L1_Cache_Write_Policy = ",L:T:m:L:L,";

// L1 cache bfore Volta was write-no-allocate, write-evict with only local
// accsses to be write-back
static const char *Before_Volta_L1_Cache_Write_Policy = ",L:L:m:N:L,";

// Adaptive cache config option
static const char *SHMEM_ADAPTIVE_OPTION = "0,8,16,32,64";

int main() {
  intilizeDeviceProp(0);

  if (ACCEL_SIM_MODE) {

    std::cout << "\n//Accel_Sim config: \n";

    bool adaptive_cache;
    string cache_write_string;
    string adaptive_shmem_option_string;
    unsigned write_cache_ratio;
    unsigned unified_l1d_size_inKB;
    unsigned config_l1_size;
    // l1 cache is sector since pascal
    char is_sector = (deviceProp.major >= 6) ? 'S' : 'N';
    // for volta and above, l1 is write allocate and adative
    if (deviceProp.major >= 7) {
      // configure based on min l1 cache
      // l1 cache is adpative
      adaptive_cache = true;
      adaptive_shmem_option_string = SHMEM_ADAPTIVE_OPTION;
      std::stringstream large_shmem_size;
      unsigned shd_mem_inKB = deviceProp.sharedMemPerMultiprocessor / 1024;
      large_shmem_size << "," << shd_mem_inKB;
      adaptive_shmem_option_string += large_shmem_size.str();
      unified_l1d_size_inKB = L1_SIZE / 1024;
      //increase unified cache by 32KB in case the shd is larger
      //this case happens in Turing, we need to write ubench to get the exact size
      if(unified_l1d_size_inKB <= shd_mem_inKB)
        unified_l1d_size_inKB = unified_l1d_size_inKB + 32;
      // set l1 write allocation policy (write allocate, write through)
      cache_write_string = After_Volta_L1_Cache_Write_Policy;
      // L1 write-to-read ratio (25%) based on rodinia kmeans workload
      // benchmarking
      write_cache_ratio = 25;
      //always configure l1 as 32KB in adaptive cache
      //accel-sim will adjust the assoc adpatively during run-time
      config_l1_size = 32*1024;
      //ensure unified cache is multiple of l1 cache size
      assert((unified_l1d_size_inKB*1024) % config_l1_size == 0);
    } else {
      adaptive_cache = false;
      cache_write_string = Before_Volta_L1_Cache_Write_Policy;
      write_cache_ratio = 0;
      unified_l1d_size_inKB = L1_SIZE / 1024;
      config_l1_size = L1_SIZE;
    }

    // lines per set
    unsigned assoc = config_l1_size / L1_CACHE_LINE_SIZE / L1_CACHE_SETS;

    unsigned warps_num_per_sm = MAX_THREADS_PER_SM / WARP_SIZE;
    // each warp can issue up to two pending cache lines (this is based on our
    // l1_mshr ubench)
    unsigned mshr = warps_num_per_sm * L1_MSHR_ENTRIES_PER_WARP;

    std::cout << "-gpgpu_adaptive_cache_config " << adaptive_cache << std::endl;
    std::cout << "-gpgpu_shmem_option " << adaptive_shmem_option_string
              << std::endl;
    std::cout << "-gpgpu_unified_l1d_size " << unified_l1d_size_inKB << std::endl;
    std::cout << "-gpgpu_l1_banks " << WARP_SCHEDS_PER_SM << std::endl;
    std::cout << "-gpgpu_cache:dl1 " << is_sector << ":" << L1_CACHE_SETS << ":"
              << L1_CACHE_LINE_SIZE << ":" << assoc << cache_write_string
              << "A:" << mshr << ":" << warps_num_per_sm << ",16:0,32"
              << std::endl;
    std::cout << "-gpgpu_gmem_skip_L1D " << !deviceProp.globalL1CacheSupported
              << std::endl;
    std::cout << "-gpgpu_l1_cache_write_ratio " << write_cache_ratio
              << std::endl;
  }

  return 1;
}
