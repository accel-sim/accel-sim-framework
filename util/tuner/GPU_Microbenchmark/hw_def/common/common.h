#ifndef COMMON_H
#define COMMON_H

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define ACCEL_SIM_MODE 1

enum issue_model { single = 1, dual = 2 };

static const char *issue_model_str[] = {"none", "single", "dual"};

enum core_model { shared = 0, subcore = 1 };

static const char *core_model_str[] = {"none", "shared", "subcore"};

enum dram_model { GDDR5 = 1, GDDR5X = 2, GDDR6 = 3, HBM = 4 };

// GPU error check
#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

// source:
// https://stackoverflow.com/questions/466204/rounding-up-to-next-power-of-2
unsigned round_up_2n(unsigned v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;

  return v;
}

unsigned round_up_2n(float n) { return round_up_2n((unsigned)ceil(n)); }

bool isPowerOfTwo(int n) {
  if (n == 0)
    return false;

  return (ceil(log2(n)) == floor(log2(n)));
}

static const char *dram_model_str[] = {"none", "GDDR5", "GDDR5X", "GDDR6",
                                       "HBM"};
static const unsigned dram_model_bus_width[] = {0, 32, 32, 16, 128}; // in bits
static const unsigned dram_model_mem_per_ctrlr[] = {0, 1, 1, 1, 1};
static const unsigned dram_model_burst_length[] = {0, 8, 8, 16, 2};
static const unsigned dram_model_freq_ratio[] = {0, 4, 4, 4, 2};
// atom size =
// dram_model_channel_width*dram_model_mem_per_ctrlr*dram_model_burst_length
unsigned get_atom_size_inByte(enum dram_model model) {
  return (dram_model_bus_width[model] / 8) * dram_model_mem_per_ctrlr[model] *
         dram_model_burst_length[model];
}
// CCD = dram_model_burst_length/dram_model_freq_ratio
unsigned get_adjusted_CCD(enum dram_model model) {
  assert(dram_model_burst_length[model] % dram_model_freq_ratio[model] == 0);
  return dram_model_burst_length[model] / dram_model_freq_ratio[model];
}

unsigned get_num_channels(unsigned total_memory_width, enum dram_model model) {
  unsigned channel_width =
      dram_model_bus_width[model] * dram_model_mem_per_ctrlr[model];
  assert(total_memory_width % channel_width == 0);
  return total_memory_width / channel_width;
}

// DDR timing struct
struct DDR_Timing {
  unsigned freq;
  unsigned nbk;
  unsigned CCD;
  unsigned RRD;
  unsigned RCD;
  unsigned RAS;
  unsigned RP;
  unsigned RC;
  unsigned CL;
  unsigned WL;
  unsigned CDLR;
  unsigned WR;
  unsigned nbkgrp;
  unsigned CCDL;
  unsigned RTPL;

  DDR_Timing(unsigned mfreq, unsigned n_bk, unsigned tCCD, unsigned tRRD,
             unsigned tRCD, unsigned tRAS, unsigned tRP, unsigned tRC,
             unsigned tCL, unsigned tWL, unsigned tCDLR, unsigned tWR,
             unsigned n_bkgrp, unsigned tCCDL, unsigned tRTPL) {
    freq = mfreq;
    nbk = n_bk;
    CCD = tCCD;
    RRD = tRRD;
    RCD = tRCD;
    RAS = tRAS;
    RP = tRP;
    RC = tRC;
    CL = tCL;
    WL = tWL;
    CDLR = tCDLR;
    WR = tWR;
    nbkgrp = n_bkgrp;
    CCDL = tCCDL;
    RTPL = tRTPL;
  }

  void scale_timing_for_new_freq(float newfreq) {
    float freq_scale = freq / newfreq;
    RRD = ceil(RRD / freq_scale);
    RCD = ceil(RCD / freq_scale);
    RAS = ceil(RAS / freq_scale);
    RP = ceil(RP / freq_scale);
    RC = ceil(RC / freq_scale);
    CL = ceil(CL / freq_scale);
    WL = ceil(WL / freq_scale);
    CDLR = ceil(CDLR / freq_scale);
    WR = ceil(WR / freq_scale);
    CCDL = ceil(CCDL / freq_scale);
    RTPL = ceil(RTPL / freq_scale);
  }
};

// GDDR5 timing from hynix H5GQ1H24AFR
//-gpgpu_dram_timing_opt "nbk=16:CCD=2:RRD=6:RCD=12:RAS=28:RP=12:RC=40:
//                        CL=12:WL=4:CDLR=5:WR=12:nbkgrp=4:CCDL=3:RTPL=2"

static const DDR_Timing GDDR5_Timing_1800MHZ(1800, 16, 2, 6, 12, 28, 12, 40, 12,
                                             4, 5, 12, 4, 3, 2);

// HBM timing are adopted from hynix JESD235 standered and nVidia HPCA 2017
// paper (http://www.cs.utah.edu/~nil/pubs/hpca17.pdf)
// Timing for 1 GHZ:
//-gpgpu_dram_timing_opt "nbk=16:CCD=1:RRD=4:RCD=14:RAS=33:RP=14:RC=47:
//                        CL=14:WL=2:CDLR=3:WR=12:nbkgrp=4:CCDL=2:RTPL=4"

static const DDR_Timing HBM_Timing_1000MHZ(1000, 16, 1, 4, 14, 33, 14, 47, 14,
                                           2, 3, 12, 4, 2, 4);

#endif