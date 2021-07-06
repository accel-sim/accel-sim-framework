/*
This microbenchmark detects L2 write policy
check the nvprof or nvsight for received l2 reads and writes to detect the
policy check the comments below for further details and also see our arvix
paper: https://arxiv.org/pdf/1810.07269.pdf

Compile this file using the following command to disable L1 cache:
    nvcc -Xptxas -dlcm=cg l2_write_policy.cu

to run the program with nvsight
 make nvsight ./l2_write_policy
*/

#include <assert.h>
#include <cstdio>
#include <iostream>
using namespace std;

#include "../../../hw_def/hw_def.h"

#define THREADS_NUM 1   // Launch only one thread
#define ARRAY_SIZE 1024 // size of the array

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Device code

/*
check the nvprof or nvsight to see the L2/DRAM reads and write hits
in the below mb, we have 6 reads and 4 writes

1. Check the write allocation policy
 we have three policies: write no-allocate vs write-allocate fetch-on-write vs
vs write-allocate sub-sector write?? if only two write hits (C[i] and A[i] at
lines 3&4) ==> then write no-allocate, else if three write hits (C[i+1], C[i]
and A[i] at lines 2&3&4) then it is write-allocate. if one read miss (A[i] at
line1) and 5 reads hits ==> then fetch-on-write, as the miss at line1 will fetch
the whole sector, and C[i] and line 3 is hit else if two read misses (A[i] at
lines 1 and C[i] at line 3) ==> then sub-sector write with write bit-mask, the
sector will be fetched on read miss, not write miss

2. check if write-back or write-through
check the DRAM writes, if four writes are received ==> then write-through
if less than four writes ==> then write-back

to run the program with nvsight:
 make nvsight ./l2_write_policy

 stats to look at:
l2 reads: lts__t_sectors_srcunit_tex_op_read.sum
l2 writes: lts__t_sectors_srcunit_tex_op_write.sum
l2 read hits: lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum
l2 write hits: lts__t_sectors_srcunit_tex_op_write_lookup_hit.

The comments below shows a case of write-allocate with sub-sector mask as in
Pascal, Volta, Turing and Ampere HW Results found: Pascal, Volta, Turing and
Ampere: write allocate &  sub-sector write  + write-back
*/

__global__ void write_policy_mb(float *A, float *C) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i == 0) {
    C[i] = A[i];        // write to C[i] is a miss (cache line is missing)
    C[i + 1] = A[i];    // write to C[i+1] is a hit (cache line is found)
    C[i] = C[i] + A[i]; // read of C[i] is a miss (entire sector is missing,
                        // fetch it from memory)
    A[i] =
        C[i] + C[i + 1]; // read C[i] and C[i+1] are hits (entire sector exists)
  }
}

//////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
  intilizeDeviceProp(0);

  BLOCKS_NUM = 1;
  TOTAL_THREADS = THREADS_NUM * BLOCKS_NUM;
  THREADS_PER_SM = THREADS_NUM * BLOCKS_NUM;

  // create 4KB buffers of A&C
  assert(ARRAY_SIZE * sizeof(float) < L2_SIZE);

  float *A = (float *)malloc(ARRAY_SIZE * sizeof(float));
  float *C = (float *)malloc(ARRAY_SIZE * sizeof(float));

  float *A_g;
  float *C_g;

  gpuErrchk(cudaMalloc(&A_g, ARRAY_SIZE * sizeof(float)));
  gpuErrchk(cudaMalloc(&C_g, ARRAY_SIZE * sizeof(float)));

  for (uint32_t i = 0; i < ARRAY_SIZE; i++)
    A[i] = (float)i;

  gpuErrchk(
      cudaMemcpy(A_g, A, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));

  write_policy_mb<<<1, THREADS_NUM>>>(A_g, C_g);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(C, C_g, ARRAY_SIZE * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));

  std::cout << "\nThis microbenchmark detects L2 write policy.\n";
  std::cout << "check the nvprof or nvsight for received L2 reads and writes "
               "to detect the policy.\n";
  std::cout << "see the code comments for further details\n";
  std::cout << "to run the program with nvsight: make nvsight ./2\n";
  std::cout << "stats to look at: llts__t_sectors_srcunit_tex_op_read.sum & "
               "lts__t_sectors_srcunit_tex_op_write.sum & "
               "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum & "
               "lts__t_sectors_srcunit_tex_op_write_lookup_hit.sum \n\n";

  return 1;
}
