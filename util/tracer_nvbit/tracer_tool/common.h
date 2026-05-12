/* Author1: Mahmoud Khairy, abdallm@purdue.com - 2019 */
/* Author2: Jason Shen, shen203@purdue.edu - 2019 */

#include <stdint.h>

static __managed__ uint64_t total_dynamic_instr_counter = 0;
static __managed__ uint64_t reported_dynamic_instr_counter = 0;
static __managed__ bool stop_report = false;

/* information collected in the instrumentation function and passed
 * on the channel from the GPU to the CPU */
#define MAX_SRC 5
// Unified registers are offset by 256 from the regular registers
#define UREG_OFFSET 256

typedef enum MyEnum {
  INST_REGULAR,
  INST_TMA,
} TracerInstrType;

typedef struct {
  uint32_t active_mask;
  uint32_t predicate_mask;
  int cta_id_x;
  int cta_id_y;
  int cta_id_z;
  // CTA id within the cluster
  int cluster_cta_id_x;
  int cluster_cta_id_y;
  int cluster_cta_id_z;
  // CTA rank within the cluster
  int cluster_rank;
  // Cluster id within the grid
  int cluster_id_x;
  int cluster_id_y;
  int cluster_id_z;
  int warpid_tb;
  int warpid_sm;
  int sm_id;
  int opcode_id;
  // Whether this is a GMMA commit group instruction
  int is_gmma_commit_group;
  uint32_t vpc;
  uint32_t line_num;
  uint32_t instr_idx;
  TracerInstrType inst_type;
  union {
    // For regular instructions
    struct {
      uint64_t addrs[32];
      bool is_mem;
      int32_t GPRDst;
      int32_t GPRSrcs[MAX_SRC];
      int32_t numSrcs;
      int32_t width;
      uint64_t imm;
      uint64_t imm2;
      uint32_t desRegVal[32];
      uint32_t srcRegVals[MAX_SRC][32];
      // Opcode-defined extra per-lane value. Currently used by SYNCS.EXCH.64
      // to carry the hi half (Rn+1) of its 64-bit src2 pair.
      uint32_t auxRegVals[32];
    } regular;
    // For TMA instructions
    struct {
      uint64_t tma_param_handle_ptr;
      uint32_t tma_param_handle_size;
      uint8_t tma_param_handle[128];
    } tma;
  } inst;
} inst_trace_t;
