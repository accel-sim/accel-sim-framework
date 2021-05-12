#include <iostream>
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
#define L1_MSHR_ENTRIES_PER_WARP 8

// L1 cache cache in Volta and above is write allocate, subsector write, write-
// through we know that from l1_write_policy ubench and has been consistent
// after Volta. Change it accordingly if it changes in new generations
static const char *After_Volta_L1_Cache_Write_Policy = ",L:T:m:L:L,";

// L1 cache bfore Volta was write-no-allocate, write-evict with only local
// accsses to be write-back
static const char *Before_Volta_L1_Cache_Write_Policy = ",L:L:m:N:L,";

int main() {
  intilizeDeviceProp(0);

  if (ACCEL_SIM_MODE) {

    std::cout << "\n//Accel_Sim config: \n";

    unsigned MIN_L1_SIZE;
    bool adaptive_cache;
    string cache_write_string;
    // l1 cache is sector since pascal
    char is_sector = (deviceProp.major >= 6) ? 'S' : 'N';
    // for volta and above, l1 cache is adpative
    if (deviceProp.major >= 7) {
      // configure based on min l1 cache
      MIN_L1_SIZE = L1_SIZE - deviceProp.sharedMemPerMultiprocessor;
      adaptive_cache = true;
      cache_write_string = After_Volta_L1_Cache_Write_Policy;
    } else {
      MIN_L1_SIZE = L1_SIZE;
      adaptive_cache = false;
      cache_write_string = Before_Volta_L1_Cache_Write_Policy;
    }

    // lines per set
    unsigned assoc = MIN_L1_SIZE / L1_CACHE_LINE_SIZE / L1_CACHE_SETS;

    unsigned warps_num_per_sm = MAX_THREADS_PER_SM / WARP_SIZE;
    unsigned access_factor = L1_CACHE_LINE_SIZE / L1_SECTOR_SIZE;
    // each warp can issue up to two pending cache lines (this is based on our
    // l1_mshr ubench)
    unsigned mshr = warps_num_per_sm * L1_MSHR_ENTRIES_PER_WARP;

    std::cout << "-gpgpu_adaptive_cache_config " << adaptive_cache << std::endl;
    std::cout << "-gpgpu_l1_banks " << WARP_SCHEDS_PER_SM << std::endl;
    // set old write settings for now as accel-sim does not support l1 write
    // allocate yet
    std::cout << "-gpgpu_cache:dl1 " << is_sector << ":" << L1_CACHE_SETS << ":"
              << L1_CACHE_LINE_SIZE << ":" << assoc
              << Before_Volta_L1_Cache_Write_Policy << "A:" << mshr << ":"
              << warps_num_per_sm << ",16:0,32" << std::endl;
    std::cout << "-gpgpu_gmem_skip_L1D " << !deviceProp.globalL1CacheSupported
              << std::endl;
  }

  return 1;
}