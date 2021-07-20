// developed by Weili An, Purdue Univ
// an107@purdue.edu

#ifndef GCN3_OPCODE_H
#define GCN3_OPCODE_H

#include <string>
#include <unordered_map>
#include "abstract_hardware_model.h"
#include "trace_opcode.h"

#define GCN3_BINART_VERSION 100

// TO DO: moving this to a yml or def files

/// GCN3 ISA
// Refered the Ampere_opcode for opcode matching
// TODO How to format this?
// TODO How these are invoked?
static const std::unordered_map<std::string, OpcodeChar> GCN3_OpcodeMap = {
    // Floating Point 32 Instructions
    {"FADD", OpcodeChar(OP_FADD, SP_OP)},
    {"FADD32I", OpcodeChar(OP_FADD32I, SP_OP)},
    {"FCHK", OpcodeChar(OP_FCHK, SP_OP)},
    {"FFMA32I", OpcodeChar(OP_FFMA32I, SP_OP)},
    {"FFMA", OpcodeChar(OP_FFMA, SP_OP)},
    {"FMNMX", OpcodeChar(OP_FMNMX, SP_OP)},
    {"FMUL", OpcodeChar(OP_FMUL, SP_OP)},
    {"FMUL32I", OpcodeChar(OP_FMUL32I, SP_OP)},
    {"FSEL", OpcodeChar(OP_FSEL, SP_OP)},
    {"FSET", OpcodeChar(OP_FSET, SP_OP)},
    {"FSETP", OpcodeChar(OP_FSETP, SP_OP)},
    {"FSWZADD", OpcodeChar(OP_FSWZADD, SP_OP)},
    // SFU
    {"MUFU", OpcodeChar(OP_MUFU, SFU_OP)},

    // Floating Point 16 Instructions
    {"HADD2", OpcodeChar(OP_HADD2, SP_OP)},
    {"HADD2_32I", OpcodeChar(OP_HADD2_32I, SP_OP)},
    {"HFMA2", OpcodeChar(OP_HFMA2, SP_OP)},
    {"HFMA2_32I", OpcodeChar(OP_HFMA2_32I, SP_OP)},
    {"HMUL2", OpcodeChar(OP_HMUL2, SP_OP)},
    {"HMUL2_32I", OpcodeChar(OP_HMUL2_32I, SP_OP)},
    {"HSET2", OpcodeChar(OP_HSET2, SP_OP)},
    {"HSETP2", OpcodeChar(OP_HSETP2, SP_OP)},
    {"HMNMX2", OpcodeChar(OP_HMNMX2, SP_OP)},

    // Tensor Core Instructions
    // Execute Tensor Core Instructions on SPECIALIZED_UNIT_3
    {"HMMA", OpcodeChar(OP_HMMA, SPECIALIZED_UNIT_3_OP)},
    {"DMMA", OpcodeChar(OP_DMMA, SPECIALIZED_UNIT_3_OP)},
    {"BMMA", OpcodeChar(OP_BMMA, SPECIALIZED_UNIT_3_OP)},
    {"IMMA", OpcodeChar(OP_IMMA, SPECIALIZED_UNIT_3_OP)},

    // Double Point Instructions
    {"DADD", OpcodeChar(OP_DADD, DP_OP)},
    {"DFMA", OpcodeChar(OP_DFMA, DP_OP)},
    {"DMUL", OpcodeChar(OP_DMUL, DP_OP)},
    {"DSETP", OpcodeChar(OP_DSETP, DP_OP)},

    // Integer Instructions
    {"BMSK", OpcodeChar(OP_BMSK, INTP_OP)},
    {"BREV", OpcodeChar(OP_BREV, INTP_OP)},
    {"FLO", OpcodeChar(OP_FLO, INTP_OP)},
    {"IABS", OpcodeChar(OP_IABS, INTP_OP)},
    {"IADD", OpcodeChar(OP_IADD, INTP_OP)},
    {"IADD3", OpcodeChar(OP_IADD3, INTP_OP)},
    {"IADD32I", OpcodeChar(OP_IADD32I, INTP_OP)},
    {"IDP", OpcodeChar(OP_IDP, INTP_OP)},
    {"IDP4A", OpcodeChar(OP_IDP4A, INTP_OP)},
    {"IMAD", OpcodeChar(OP_IMAD, INTP_OP)},
    {"IMNMX", OpcodeChar(OP_IMNMX, INTP_OP)},
    {"IMUL", OpcodeChar(OP_IMUL, INTP_OP)},
    {"IMUL32I", OpcodeChar(OP_IMUL32I, INTP_OP)},
    {"ISCADD", OpcodeChar(OP_ISCADD, INTP_OP)},
    {"ISCADD32I", OpcodeChar(OP_ISCADD32I, INTP_OP)},
    {"ISETP", OpcodeChar(OP_ISETP, INTP_OP)},
    {"LEA", OpcodeChar(OP_LEA, INTP_OP)},
    {"LOP", OpcodeChar(OP_LOP, INTP_OP)},
    {"LOP3", OpcodeChar(OP_LOP3, INTP_OP)},
    {"LOP32I", OpcodeChar(OP_LOP32I, INTP_OP)},
    {"POPC", OpcodeChar(OP_POPC, INTP_OP)},
    {"SHF", OpcodeChar(OP_SHF, INTP_OP)},
    {"SHL", OpcodeChar(OP_SHL, INTP_OP)},  //////////
    {"SHR", OpcodeChar(OP_SHR, INTP_OP)},
    {"VABSDIFF", OpcodeChar(OP_VABSDIFF, INTP_OP)},
    {"VABSDIFF4", OpcodeChar(OP_VABSDIFF4, INTP_OP)},

    // Conversion Instructions
    {"F2F", OpcodeChar(OP_F2F, ALU_OP)},
    {"F2I", OpcodeChar(OP_F2I, ALU_OP)},
    {"I2F", OpcodeChar(OP_I2F, ALU_OP)},
    {"I2I", OpcodeChar(OP_I2I, ALU_OP)},
    {"I2IP", OpcodeChar(OP_I2IP, ALU_OP)},
    {"I2FP", OpcodeChar(OP_I2FP, ALU_OP)},
    {"F2IP", OpcodeChar(OP_F2IP, ALU_OP)},
    {"FRND", OpcodeChar(OP_FRND, ALU_OP)},

    // Movement Instructions
    {"MOV", OpcodeChar(OP_MOV, ALU_OP)},
    {"MOV32I", OpcodeChar(OP_MOV32I, ALU_OP)},
    {"MOVM", OpcodeChar(OP_MOVM, ALU_OP)},  // move matrix
    {"PRMT", OpcodeChar(OP_PRMT, ALU_OP)},
    {"SEL", OpcodeChar(OP_SEL, ALU_OP)},
    {"SGXT", OpcodeChar(OP_SGXT, ALU_OP)},
    {"SHFL", OpcodeChar(OP_SHFL, ALU_OP)},

    // Predicate Instructions
    {"PLOP3", OpcodeChar(OP_PLOP3, ALU_OP)},
    {"PSETP", OpcodeChar(OP_PSETP, ALU_OP)},
    {"P2R", OpcodeChar(OP_P2R, ALU_OP)},
    {"R2P", OpcodeChar(OP_R2P, ALU_OP)},

    // Load/Store Instructions
    {"LD", OpcodeChar(OP_LD, LOAD_OP)},
    // For now, we ignore constant loads, consider it as ALU_OP, TO DO
    {"LDC", OpcodeChar(OP_LDC, ALU_OP)},
    {"LDG", OpcodeChar(OP_LDG, LOAD_OP)},
    {"LDL", OpcodeChar(OP_LDL, LOAD_OP)},
    {"LDS", OpcodeChar(OP_LDS, LOAD_OP)},
    {"LDSM", OpcodeChar(OP_LDSM, LOAD_OP)},  //
    {"ST", OpcodeChar(OP_ST, STORE_OP)},
    {"STG", OpcodeChar(OP_STG, STORE_OP)},
    {"STL", OpcodeChar(OP_STL, STORE_OP)},
    {"STS", OpcodeChar(OP_STS, STORE_OP)},
    {"MATCH", OpcodeChar(OP_MATCH, ALU_OP)},
    {"QSPC", OpcodeChar(OP_QSPC, ALU_OP)},
    {"ATOM", OpcodeChar(OP_ATOM, STORE_OP)},
    {"ATOMS", OpcodeChar(OP_ATOMS, STORE_OP)},
    {"ATOMG", OpcodeChar(OP_ATOMG, STORE_OP)},
    {"RED", OpcodeChar(OP_RED, STORE_OP)},
    {"CCTL", OpcodeChar(OP_CCTL, ALU_OP)},
    {"CCTLL", OpcodeChar(OP_CCTLL, ALU_OP)},
    {"ERRBAR", OpcodeChar(OP_ERRBAR, ALU_OP)},
    {"MEMBAR", OpcodeChar(OP_MEMBAR, MEMORY_BARRIER_OP)},
    {"CCTLT", OpcodeChar(OP_CCTLT, ALU_OP)},

    {"LDGDEPBAR", OpcodeChar(OP_LDGDEPBAR, ALU_OP)},
    {"LDGSTS", OpcodeChar(OP_LDGSTS, LOAD_OP)},

    // Uniform Datapath Instruction
    // UDP unit
    // for more info about UDP, see
    // https://www.hotchips.org/hc31/HC31_2.12_NVIDIA_final.pdf
    {"R2UR", OpcodeChar(OP_R2UR, SPECIALIZED_UNIT_4_OP)},
    {"REDUX", OpcodeChar(OP_REDUX, SPECIALIZED_UNIT_4_OP)},
    {"S2UR", OpcodeChar(OP_S2UR, SPECIALIZED_UNIT_4_OP)},
    {"UBMSK", OpcodeChar(OP_UBMSK, SPECIALIZED_UNIT_4_OP)},
    {"UBREV", OpcodeChar(OP_UBREV, SPECIALIZED_UNIT_4_OP)},
    {"UCLEA", OpcodeChar(OP_UCLEA, SPECIALIZED_UNIT_4_OP)},
    {"UF2FP", OpcodeChar(OP_UF2FP, SPECIALIZED_UNIT_4_OP)},
    {"UFLO", OpcodeChar(OP_UFLO, SPECIALIZED_UNIT_4_OP)},
    {"UIADD3", OpcodeChar(OP_UIADD3, SPECIALIZED_UNIT_4_OP)},
    {"UIMAD", OpcodeChar(OP_UIMAD, SPECIALIZED_UNIT_4_OP)},
    {"UISETP", OpcodeChar(OP_UISETP, SPECIALIZED_UNIT_4_OP)},
    {"ULDC", OpcodeChar(OP_ULDC, SPECIALIZED_UNIT_4_OP)},
    {"ULEA", OpcodeChar(OP_ULEA, SPECIALIZED_UNIT_4_OP)},
    {"ULOP", OpcodeChar(OP_ULOP, SPECIALIZED_UNIT_4_OP)},
    {"ULOP3", OpcodeChar(OP_ULOP3, SPECIALIZED_UNIT_4_OP)},
    {"ULOP32I", OpcodeChar(OP_ULOP32I, SPECIALIZED_UNIT_4_OP)},
    {"UMOV", OpcodeChar(OP_UMOV, SPECIALIZED_UNIT_4_OP)},
    {"UP2UR", OpcodeChar(OP_UP2UR, SPECIALIZED_UNIT_4_OP)},
    {"UPLOP3", OpcodeChar(OP_UPLOP3, SPECIALIZED_UNIT_4_OP)},
    {"UPOPC", OpcodeChar(OP_UPOPC, SPECIALIZED_UNIT_4_OP)},
    {"UPRMT", OpcodeChar(OP_UPRMT, SPECIALIZED_UNIT_4_OP)},
    {"UPSETP", OpcodeChar(OP_UPSETP, SPECIALIZED_UNIT_4_OP)},
    {"UR2UP", OpcodeChar(OP_UR2UP, SPECIALIZED_UNIT_4_OP)},
    {"USEL", OpcodeChar(OP_USEL, SPECIALIZED_UNIT_4_OP)},
    {"USGXT", OpcodeChar(OP_USGXT, SPECIALIZED_UNIT_4_OP)},
    {"USHF", OpcodeChar(OP_USHF, SPECIALIZED_UNIT_4_OP)},
    {"USHL", OpcodeChar(OP_USHL, SPECIALIZED_UNIT_4_OP)},
    {"USHR", OpcodeChar(OP_USHR, SPECIALIZED_UNIT_4_OP)},
    {"VOTEU", OpcodeChar(OP_VOTEU, SPECIALIZED_UNIT_4_OP)},

    // Texture Instructions
    // For now, we ignore texture loads, consider it as ALU_OP
    {"TEX", OpcodeChar(OP_TEX, SPECIALIZED_UNIT_2_OP)},
    {"TLD", OpcodeChar(OP_TLD, SPECIALIZED_UNIT_2_OP)},
    {"TLD4", OpcodeChar(OP_TLD4, SPECIALIZED_UNIT_2_OP)},
    {"TMML", OpcodeChar(OP_TMML, SPECIALIZED_UNIT_2_OP)},
    {"TXD", OpcodeChar(OP_TXD, SPECIALIZED_UNIT_2_OP)},
    {"TXQ", OpcodeChar(OP_TXQ, SPECIALIZED_UNIT_2_OP)},

    // Surface Instructions //
    {"SUATOM", OpcodeChar(OP_SUATOM, ALU_OP)},
    {"SULD", OpcodeChar(OP_SULD, ALU_OP)},
    {"SUQUERY", OpcodeChar(OP_SUQUERY, ALU_OP)},
    {"SURED", OpcodeChar(OP_SURED, ALU_OP)},
    {"SUST", OpcodeChar(OP_SUST, ALU_OP)},

    // Control Instructions
    // execute branch insts on a dedicated branch unit (SPECIALIZED_UNIT_1)
    {"BMOV", OpcodeChar(OP_BMOV, SPECIALIZED_UNIT_1_OP)},
    {"BPT", OpcodeChar(OP_BPT, SPECIALIZED_UNIT_1_OP)},
    {"BRA", OpcodeChar(OP_BRA, SPECIALIZED_UNIT_1_OP)},
    {"BREAK", OpcodeChar(OP_BREAK, SPECIALIZED_UNIT_1_OP)},
    {"BRX", OpcodeChar(OP_BRX, SPECIALIZED_UNIT_1_OP)},
    {"BRXU", OpcodeChar(OP_BRXU, SPECIALIZED_UNIT_1_OP)},  //
    {"BSSY", OpcodeChar(OP_BSSY, SPECIALIZED_UNIT_1_OP)},
    {"BSYNC", OpcodeChar(OP_BSYNC, SPECIALIZED_UNIT_1_OP)},
    {"CALL", OpcodeChar(OP_CALL, SPECIALIZED_UNIT_1_OP)},
    {"EXIT", OpcodeChar(OP_EXIT, EXIT_OPS)},
    {"JMP", OpcodeChar(OP_JMP, SPECIALIZED_UNIT_1_OP)},
    {"JMX", OpcodeChar(OP_JMX, SPECIALIZED_UNIT_1_OP)},
    {"JMXU", OpcodeChar(OP_JMXU, SPECIALIZED_UNIT_1_OP)},  ///
    {"KILL", OpcodeChar(OP_KILL, SPECIALIZED_UNIT_3_OP)},
    {"NANOSLEEP", OpcodeChar(OP_NANOSLEEP, SPECIALIZED_UNIT_1_OP)},
    {"RET", OpcodeChar(OP_RET, SPECIALIZED_UNIT_1_OP)},
    {"RPCMOV", OpcodeChar(OP_RPCMOV, SPECIALIZED_UNIT_1_OP)},
    {"RTT", OpcodeChar(OP_RTT, SPECIALIZED_UNIT_1_OP)},
    {"WARPSYNC", OpcodeChar(OP_WARPSYNC, SPECIALIZED_UNIT_1_OP)},
    {"YIELD", OpcodeChar(OP_YIELD, SPECIALIZED_UNIT_1_OP)},

    // Miscellaneous Instructions
    {"B2R", OpcodeChar(OP_B2R, ALU_OP)},
    {"BAR", OpcodeChar(OP_BAR, BARRIER_OP)},
    {"CS2R", OpcodeChar(OP_CS2R, ALU_OP)},
    {"CSMTEST", OpcodeChar(OP_CSMTEST, ALU_OP)},
    {"DEPBAR", OpcodeChar(OP_DEPBAR, ALU_OP)},
    {"GETLMEMBASE", OpcodeChar(OP_GETLMEMBASE, ALU_OP)},
    {"LEPC", OpcodeChar(OP_LEPC, ALU_OP)},
    {"NOP", OpcodeChar(OP_NOP, ALU_OP)},
    {"PMTRIG", OpcodeChar(OP_PMTRIG, ALU_OP)},
    {"R2B", OpcodeChar(OP_R2B, ALU_OP)},
    {"S2R", OpcodeChar(OP_S2R, ALU_OP)},
    {"SETCTAID", OpcodeChar(OP_SETCTAID, ALU_OP)},
    {"SETLMEMBASE", OpcodeChar(OP_SETLMEMBASE, ALU_OP)},
    {"VOTE", OpcodeChar(OP_VOTE, ALU_OP)},
    {"VOTE_VTG", OpcodeChar(OP_VOTE_VTG, ALU_OP)},


    // Tentative GCN3 opcode mapping
    // TODO Token get ride of the last width part (I and U and B only)
    // TODO also drop the DWORD part and UBYTE0
    // TODO For V_CMP, return just two first words
    // AMD SOP2 instructions
    {"S_ABSDIFF_I32", OpcodeChar(OP_VABSDIFF, INTP_OP)},
    {"S_ADD", OpcodeChar(OP_IADD, INTP_OP)},
    {"S_ADDC_U32", OpcodeChar(OP_IADD, INTP_OP)},
    {"S_AND", OpcodeChar(OP_IADD, INTP_OP)},    // TODO need to find equivalent for this
    {"S_ANDN2", OpcodeChar(OP_IADD, INTP_OP)},
    {"S_ASHR", OpcodeChar(OP_SHR, INTP_OP)},
    {"S_BFE", OpcodeChar(OP_IADD, INTP_OP)},    // TODO
    {"S_BFM", OpcodeChar(OP_BMSK, INTP_OP)},    // Bitfield mask
    // Conditional branch using branch stack. Arg0 = compare mask (VCC or any SGPR), Arg1 =
    // 64-bit byte address of target instruction.
    {"S_CBRANCH_G_FORK", OpcodeChar(OP_BRA, SPECIALIZED_UNIT_1_OP)}, // TODO
    {"S_CSELECT", OpcodeChar(OP_SEL, ALU_OP)},    // TODO
    {"S_LSHL", OpcodeChar(OP_SHL, INTP_OP)},
    {"S_LSHR", OpcodeChar(OP_SHR, INTP_OP)},
    {"S_MAX", OpcodeChar(OP_SEL, INTP_OP)}, // TODO
    {"S_MIN", OpcodeChar(OP_SHR, INTP_OP)},
    {"S_MUL", OpcodeChar(OP_IMUL, INTP_OP)},
    {"S_NAND", OpcodeChar(OP_IADD, INTP_OP)},   // TODO
    {"S_NOR", OpcodeChar(OP_IADD, INTP_OP)},   // TODO
    {"S_OR", OpcodeChar(OP_IADD, INTP_OP)}, // TODO
    {"S_ORN2", OpcodeChar(OP_IADD, INTP_OP)},
    {"S_SUB", OpcodeChar(OP_ISUB, INTP_OP)},
    {"S_SUBB", OpcodeChar(OP_ISUB, INTP_OP)},
    {"S_XNOR", OpcodeChar(OP_IADD, INTP_OP)},
    {"S_XOR", OpcodeChar(OP_IADD, INTP_OP)},

    // AMD GCN3 SOPK instructions
    {"S_ADDK", OpcodeChar(OP_IADD, INTP_OP)},
    {"S_CBRANCH_I_FORK", OpcodeChar(OP_BRA, SPECIALIZED_UNIT_1_OP)},    // TODO
    {"S_CMOVK", OpcodeChar(OP_BMOV, SPECIALIZED_UNIT_1_OP)},  // TODO
    {"S_CMPK_EQ", OpcodeChar(OP_ICMP, INTP_OP)},  // TODO
    {"S_CMPK_GE", OpcodeChar(OP_ICMP, INTP_OP)},  // TODO
    {"S_CMPK_GT", OpcodeChar(OP_ICMP, INTP_OP)},  // TODO
    {"S_CMPK_LE", OpcodeChar(OP_ICMP, INTP_OP)},  // TODO
    {"S_CMPK_LG", OpcodeChar(OP_ICMP, INTP_OP)},  // TODO
    {"S_CMPK_LT", OpcodeChar(OP_ICMP, INTP_OP)},  // TODO
    {"S_GETREG", OpcodeChar(OP_MOV, ALU_OP)},  // TODO
    {"S_MOVK", OpcodeChar(OP_MOV, ALU_OP)},
    {"S_MULK", OpcodeChar(OP_IMUL, INTP_OP)},
    {"S_SETREG", OpcodeChar(OP_MOV, ALU_OP)},
    {"S_SETREG_IMM32", OpcodeChar(OP_MOV, ALU_OP)},

    // AMD GCN3 SOP1 instructions
    {"S_ABS", OpcodeChar(OP_IABS, INTP_OP)},
    {"S_AND_SAVEEXEC", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    {"S_ANDN2_SAVEEXEC", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    {"S_BCNT0_I32", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    {"S_BCNT1_I32", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    {"S_BITSET0", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    {"S_BITSET1", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    {"S_BREV", OpcodeChar(OP_BREV, INTP_OP)},
    {"S_CBRANCH_JOIN", OpcodeChar(OP_BSYNC, SPECIALIZED_UNIT_1_OP)},  // TODO
    {"S_CMOV", OpcodeChar(OP_BMOV, SPECIALIZED_UNIT_1_OP)},
    {"S_FF0_I32", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    {"S_FF1_I32", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    {"S_FLBIT", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    {"S_FLBIT_I32", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    {"S_GETPC", OpcodeChar(OP_PCNT, RET_OPS)},  // TODO
    {"S_MOV", OpcodeChar(OP_MOV, ALU_OP)},
    {"S_MOVRELD", OpcodeChar(OP_MOV, ALU_OP)},
    {"S_MOVRELS", OpcodeChar(OP_MOV, ALU_OP)},
    {"S_NAND_SAVEEXEC", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    {"S_NOR_SAVEEXEC", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    {"S_OR_SAVEEXEC", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    {"S_ORN2_SAVEEXEC", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    {"S_XNOR_SAVEEXEC", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    {"S_XOR_SAVEEXEC", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    {"S_NOT", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    {"S_QUADMASK", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    {"S_RFE", OpcodeChar(OP_RET, SPECIALIZED_UNIT_1_OP)},  // TODO Return from exception, set PC
    {"S_SET_GPR_IDX_IDX", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    {"S_SETPC", OpcodeChar(OP_JMP, SPECIALIZED_UNIT_1_OP)},
    {"S_SEXT_I32", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    {"S_SWAPPC", OpcodeChar(OP_JMP, SPECIALIZED_UNIT_1_OP)},
    {"S_WQM", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    
    // AMD GCN3 SOPC instructions
    {"S_BITCMP0", OpcodeChar(OP_ICMP, INTP_OP)},
    {"S_BITCMP1", OpcodeChar(OP_ICMP, INTP_OP)},
    {"S_CMP_EQ", OpcodeChar(OP_ICMP, INTP_OP)},
    {"S_CMP_GE", OpcodeChar(OP_ICMP, INTP_OP)},
    {"S_CMP_GT", OpcodeChar(OP_ICMP, INTP_OP)},
    {"S_CMP_LE", OpcodeChar(OP_ICMP, INTP_OP)},
    {"S_CMP_LG", OpcodeChar(OP_ICMP, INTP_OP)},
    {"S_CMP_LT", OpcodeChar(OP_ICMP, INTP_OP)},
    {"S_CMP_NE", OpcodeChar(OP_ICMP, INTP_OP)},
    {"S_SET_GPR_IDX_ON", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    {"S_SETVSKIP", OpcodeChar(OP_IADD, INTP_OP)},  // TODO

    // AMD GCN3 SOPP instructions
    {"S_BARRIER", OpcodeChar(OP_BAR, BARRIER_OP)},
    {"S_BRANCH", OpcodeChar(OP_BRA, SPECIALIZED_UNIT_1_OP)},
    {"S_CBRANCH_CDBGSYS", OpcodeChar(OP_BRA, SPECIALIZED_UNIT_1_OP)},
    {"S_CBRANCH_CDBGSYS_AND_USER", OpcodeChar(OP_BRA, SPECIALIZED_UNIT_1_OP)},
    {"S_CBRANCH_CDBGSYS_OR_USER", OpcodeChar(OP_BRA, SPECIALIZED_UNIT_1_OP)},
    {"S_CBRANCH_EXECNZ", OpcodeChar(OP_BRA, SPECIALIZED_UNIT_1_OP)},
    {"S_CBRANCH_EXECZ", OpcodeChar(OP_BRA, SPECIALIZED_UNIT_1_OP)},
    {"S_CBRANCH_SCC0", OpcodeChar(OP_BRA, SPECIALIZED_UNIT_1_OP)},
    {"S_CBRANCH_SCC1", OpcodeChar(OP_BRA, SPECIALIZED_UNIT_1_OP)},
    {"S_CBRANCH_VCCNZ", OpcodeChar(OP_BRA, SPECIALIZED_UNIT_1_OP)},
    {"S_CBRANCH_VCCZ", OpcodeChar(OP_BRA, SPECIALIZED_UNIT_1_OP)},
    {"S_DECPERFLEVEL", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    {"S_ENDPGM", OpcodeChar(OP_EXIT, EXIT_OPS)},
    {"S_ENDPGM_SAVED", OpcodeChar(OP_EXIT, EXIT_OPS)},
    {"S_ICACHE_INV", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    {"S_INCPERFLEVEL", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    {"S_SENDMSG", OpcodeChar(OP_NOP, ALU_OP)},  // TODO
    {"S_SENDMSGHALT", OpcodeChar(OP_NOP, ALU_OP)},  // TODO
    {"S_SET_GPR_IDX_MODE", OpcodeChar(OP_NOP, ALU_OP)},  // TODO
    {"S_SET_GPR_IDX_OFF", OpcodeChar(OP_NOP, ALU_OP)},  // TODO
    {"S_SETHALT", OpcodeChar(OP_NOP, ALU_OP)},  // TODO
    {"S_SETKILL", OpcodeChar(OP_NOP, ALU_OP)},  // TODO
    {"S_SETPRIO", OpcodeChar(OP_NOP, ALU_OP)},  // TODO
    {"S_SLEEP", OpcodeChar(OP_NANOSLEEP, SPECIALIZED_UNIT_1_OP)},
    {"S_TRAP", OpcodeChar(OP_CALL, SPECIALIZED_UNIT_1_OP)},
    {"S_TTRACEDATA", OpcodeChar(OP_NOP, ALU_OP)},  // TODO
    {"S_WAITCNT", OpcodeChar(OP_NOP, ALU_OP)},  // TODO

    // AMD GCN3 SMEM instructions
    {"S_ATC_PROBE", OpcodeChar(OP_LDS, LOAD_OP)},  // TODO
    {"S_ATC_PROBE_BUFFER", OpcodeChar(OP_LDS, LOAD_OP)},  // TODO
    {"S_BUFFER_LOAD", OpcodeChar(OP_LDC, ALU_OP)},
    {"S_BUFFER_STORE", OpcodeChar(OP_STL, STORE_OP)},  // TODO
    // Invalidate DCACHE insts, TODO
    {"S_DCACHE_INV", OpcodeChar(OP_NOP, ALU_OP)},  // TODO
    {"S_DCACHE_INV_VOL", OpcodeChar(OP_NOP, ALU_OP)},  // TODO
    {"S_DCACHE_WB", OpcodeChar(OP_NOP, ALU_OP)},  // TODO
    {"S_DCACHE_WB_VOL", OpcodeChar(OP_NOP, ALU_OP)},  // TODO
    {"S_LOAD", OpcodeChar(OP_LDL, LOAD_OP)},
    // TODO Return current 64-bit RTC.
    {"S_MEMREALTIME", OpcodeChar(OP_NOP, ALU_OP)},
    {"S_MEMTIME", OpcodeChar(OP_NOP, ALU_OP)},
    {"S_STORE", OpcodeChar(OP_STL, STORE_OP)},

    // AMD GCN3 VOP2 Insts
    {"V_ADD_F16", OpcodeChar(OP_HADD2, SP_OP)},
    {"V_ADD_F32", OpcodeChar(OP_FADD, SP_OP)},
    {"V_ADD", OpcodeChar(OP_IADD, INTP_OP)},
    {"V_ADDC", OpcodeChar(OP_IADD, INTP_OP)},
    {"V_AND", OpcodeChar(OP_IADD, INTP_OP)},  // TODO need to find equivalent for this
    {"V_ASHRREV", OpcodeChar(OP_SHR, INTP_OP)},
    {"V_CNDMASK", OpcodeChar(OP_SEL, ALU_OP)},  // TODO
    {"V_LDEXP_F16", OpcodeChar(OP_IMUL, INTP_OP)},  // TODO
    {"V_LSHLREV", OpcodeChar(OP_SHL, INTP_OP)},
    // TODO Rest are multiply-add ops, which used MUL to simulate for now
    {"V_MAC_F16", OpcodeChar(HMUL2, SP_OP)},
    {"V_MAC_F32", OpcodeChar(OP_FMUL, SP_OP)},
    {"V_MADAK_F16", OpcodeChar(HMUL2, SP_OP)},
    {"V_MADAK_F32", OpcodeChar(OP_FMUL, SP_OP)},
    {"V_MADMK_F16", OpcodeChar(HMUL2, SP_OP)},
    {"V_MADMK_F32", OpcodeChar(OP_FMUL, SP_OP)},
    // end of multiple-add ops
    {"V_MAX_F16", OpcodeChar(OP_SEL, SP_OP)}, // TODO
    {"V_MAX_F32", OpcodeChar(OP_SEL, SP_OP)}, // TODO
    {"V_MAX", OpcodeChar(OP_SEL, INTP_OP)},
    {"V_MBCNT_HI_U32", OpcodeChar(OP_IADD, INTP_OP)}, // TODO
    {"V_MBCNT_LO_U32", OpcodeChar(OP_IADD, INTP_OP)}, // TODO
    {"V_MIN_F16", OpcodeChar(OP_SEL, SP_OP)}, // TODO
    {"V_MIN_F32", OpcodeChar(OP_SEL, SP_OP)}, // TODO
    {"V_MAX", OpcodeChar(OP_SEL, INTP_OP)},
    {"V_MUL_F16", OpcodeChar(OP_HMUL2, SP_OP)},
    {"V_MUL_F32", OpcodeChar(OP_FMUL, SP_OP)},
    {"V_MUL_HI_I32", OpcodeChar(OP_IMUL, INTP_OP)},
    {"V_MUL_HI_U32", OpcodeChar(OP_IMUL, INTP_OP)},
    {"V_MUL_I32", OpcodeChar(OP_IMUL, INTP_OP)},
    {"V_MUL_LEGACY_F32", OpcodeChar(OP_FMUL, SP_OP)},
    {"V_MUL_LO", OpcodeChar(OP_IMUL, INTP_OP)},
    {"V_MUL_U32", OpcodeChar(OP_IMUL, INTP_OP)},
    {"V_OR", OpcodeChar(OP_IADD, INTP_OP)},
    {"V_SUB_F16", OpcodeChar(OP_HADD2, SP_OP)},
    {"V_SUB_F32", OpcodeChar(OP_FADD, SP_OP)},
    {"V_SUB", OpcodeChar(OP_ISUB, INTP_OP)},
    {"V_SUBB", OpcodeChar(OP_ISUB, INTP_OP)},
    {"V_SUBBREV", OpcodeChar(OP_ISUB, INTP_OP)},
    {"V_SUBREV_F16", OpcodeChar(OP_HADD2, SP_OP)},
    {"V_SUBREV_F32", OpcodeChar(OP_FADD, SP_OP)},
    {"V_SUBREV", OpcodeChar(OP_ISUB, INTP_OP)},
    {"V_WRITELANE", OpcodeChar(OP_IADD, INTP_OP)},  // TODO 
    {"V_XOR", OpcodeChar(OP_IADD, INTP_OP)},

    // AMD GCN3 VOP1 Insts
    {"V_BFREV", OpcodeChar(OP_BREV, INTP_OP)},
    // TODO Ceil op treat as add
    {"V_CEIL_F16", OpcodeChar(OP_HADD2, SP_OP)},
    {"V_CEIL_F32", OpcodeChar(OP_FADD, SP_OP)},
    {"V_CEIL_F64", OpcodeChar(OP_DADD, SP_OP)},
    {"V_CLREXCP", OpcodeChar(OP_NOP, ALU_OP)},  // TODO
    // TODO Cos treat as mul
    {"V_COS_F16", OpcodeChar(OP_HMUL2, SP_OP)},
    {"V_COS_F32", OpcodeChar(OP_FMUL, SP_OP)},
    // TODO Convert treated as add
    {"V_CVT_F16_F32", OpcodeChar(OP_FADD, SP_OP)},
    {"V_CVT_F16", OpcodeChar(OP_FADD, SP_OP)},
    {"V_CVT_F32_F16", OpcodeChar(OP_FADD, SP_OP)},
    {"V_CVT_F32_F64", OpcodeChar(OP_FADD, SP_OP)},
    {"V_CVT_F32", OpcodeChar(OP_FADD, SP_OP)},
    // TODO Ceil treated as add
    {"V_CEIL_F32", OpcodeChar(OP_HADD2, SP_OP)},
    {"V_CEIL_F32", OpcodeChar(OP_FADD, SP_OP)},
    {"V_CVT_F64_F32", OpcodeChar(OP_FADD, SP_OP)},
    {"V_CVT_F64", OpcodeChar(OP_FADD, SP_OP)},
    // TODO Convert ops as int ops
    {"V_CVT_FLR_I32_F32", OpcodeChar(OP_IADD, INTP_OP)},
    {"V_CVT_I16_F16", OpcodeChar(OP_IADD, INTP_OP)},
    {"V_CVT_I32_F32", OpcodeChar(OP_IADD, INTP_OP)},
    {"V_CVT_I32_F64", OpcodeChar(OP_IADD, INTP_OP)},
    {"V_CVT_OFF_F32", OpcodeChar(OP_FADD, SP_OP)},
    {"V_CVT_RPI_I32_F32", OpcodeChar(OP_IADD, INTP_OP)},
    {"V_CVT_U16_F16", OpcodeChar(OP_IADD, INTP_OP)},
    {"V_CVT_U32_F32", OpcodeChar(OP_IADD, INTP_OP)},
    {"V_CVT_U32_F64", OpcodeChar(OP_IADD, INTP_OP)},
    // TODO Exp as MUL
    {"V_EXP_F32", OpcodeChar(OP_FMUL, SP_OP)},
    {"V_EXP_F16", OpcodeChar(OP_HMUL2, SP_OP)},
    {"V_EXP_LEGACY_F32", OpcodeChar(OP_FMUL, SP_OP)},
    // TODO Find bit treated as add
    {"V_FFBH", OpcodeChar(OP_IADD, INTP_OP)},
    {"V_FFBL", OpcodeChar(OP_IADD, INTP_OP)},
    // TODO Float floor treated as add
    {"V_FLOOR_F16", OpcodeChar(OP_HADD2, SP_OP)},
    {"V_FLOOR_F32", OpcodeChar(OP_FADD, SP_OP)},
    {"V_FLOOR_F64", OpcodeChar(OP_DADD, SP_OP)},
    {"V_FRACT_F16", OpcodeChar(OP_HADD2, SP_OP)},
    {"V_FRACT_F32", OpcodeChar(OP_FADD, SP_OP)},
    {"V_FRACT_F64", OpcodeChar(OP_DADD, SP_OP)},
    // TODO Exp as MUL
    {"V_FREXP_EXP_I16_F16", OpcodeChar(OP_HMUL2, SP_OP)},
    {"V_FREXP_EXP_I32_F32", OpcodeChar(OP_FMUL, SP_OP)},
    {"V_FREXP_EXP_I32_F64", OpcodeChar(OP_DMUL, DP_OP)},
    {"V_FREXP_MANT_F16", OpcodeChar(OP_HMUL2, SP_OP)},
    {"V_FREXP_MANT_F32", OpcodeChar(OP_FMUL, SP_OP)},
    {"V_FREXP_MANT_F64", OpcodeChar(OP_DMUL, DP_OP)},
    // TODO LOG as MUL
    {"V_LOG_F16", OpcodeChar(OP_HMUL2, SP_OP)},
    {"V_LOG_F32", OpcodeChar(OP_FMUL, SP_OP)},
    {"V_LOG_LEGACY_F32", OpcodeChar(OP_FMUL, SP_OP)},
    {"V_MOV", OpcodeChar(OP_MOV, ALU_OP)},
    {"V_MOVRELD", OpcodeChar(OP_MOV, ALU_OP)},
    {"V_MOVRELS", OpcodeChar(OP_MOV, ALU_OP)},
    {"V_MOVRELSD", OpcodeChar(OP_MOV, ALU_OP)},
    {"V_NOP", OpcodeChar(OP_NOP, ALU_OP)},
    {"V_NOT", OpcodeChar(OP_IADD, INTP_OP)},  // todo
    // TODO Reciprocal
    {"V_RCP_F16", OpcodeChar(OP_HMUL2, SP_OP)},
    {"V_RCP_F32", OpcodeChar(OP_FMUL, SP_OP)},
    {"V_RCP_F64", OpcodeChar(OP_DMUL, DP_OP)},
    {"V_RCP_IFLAG_F32", OpcodeChar(OP_FMUL, SP_OP)},
    {"V_READFIRSTLANE", OpcodeChar(OP_MOV, ALU_OP)},
    // TODO Rounding
    {"V_RNDNE_F16", OpcodeChar(OP_HADD2, SP_OP)},
    {"V_RNDNE_F32", OpcodeChar(OP_FADD, SP_OP)},
    {"V_RNDNE_F64", OpcodeChar(OP_DADD, SP_OP)},
    // TODO Reciprocal sqrt
    {"V_RSQ_F16", OpcodeChar(OP_HMUL2, SP_OP)},
    {"V_RSQ_F32", OpcodeChar(OP_FMUL, SP_OP)},
    {"V_RSQ_F64", OpcodeChar(OP_DMUL, DP_OP)},
    // TODO Sin
    {"V_SIN_F16", OpcodeChar(OP_HMUL2, SP_OP)},
    {"V_SIN_F32", OpcodeChar(OP_FMUL, SP_OP)},
    // TODO SQRT
    {"V_SQRT_F16", OpcodeChar(OP_HMUL2, SP_OP)},
    {"V_SQRT_F32", OpcodeChar(OP_FMUL, SP_OP)},
    {"V_SQRT_F64", OpcodeChar(OP_DMUL, DP_OP)},
    // TODO TRUNC
    {"V_TRUNC_F16", OpcodeChar(OP_HADD2, SP_OP)},
    {"V_TRUNC_F32", OpcodeChar(OP_FADD, SP_OP)},
    {"V_TRUNC_F64", OpcodeChar(OP_DADD, SP_OP)},

    // AMD GCN3 VOPC Insts
    // Just use first two words in this type of insts for look up
    {"V_CMP", OpcodeChar(OP_ICMP, INTP_OP)},  // TODO
    {"V_CMPX", OpcodeChar(OP_ICMP, INTP_OP)},  // TODO
    {"V_CMPS", OpcodeChar(OP_ICMP, INTP_OP)},  // TODO
    {"V_CMPSX", OpcodeChar(OP_ICMP, INTP_OP)},  // TODO

    // AMD GCN3 VOP3a Insts
    {"V_ADD_F64", OpcodeChar(OP_DADD, SP_OP)},
    {"V_ALIGNBIT", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    {"V_ALIGNBYTE", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    {"V_BCNT_U32", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    {"V_BFE", OpcodeChar(OP_IADD, INTP_OP)},    // TODO
    {"V_BFI", OpcodeChar(OP_IADD, INTP_OP)},    // TODO
    {"V_BFM", OpcodeChar(OP_BMSK, INTP_OP)},    // Bitfield mask
    // TODO CUBE map insts
    {"V_CUBEID_F32", OpcodeChar(OP_FADD, SP_OP)},
    {"V_CUBEMA_F32", OpcodeChar(OP_FADD, SP_OP)},
    {"V_CUBESC_F32", OpcodeChar(OP_FADD, SP_OP)},
    {"V_CUBETC_F32", OpcodeChar(OP_FADD, SP_OP)},
    // TODO Convert ops as int ops
    {"V_CVT_PK_I16", OpcodeChar(OP_IADD, INTP_OP)},
    {"V_CVT_PK_U8_F32", OpcodeChar(OP_IADD, INTP_OP)},
    {"V_CVT_PK_U16", OpcodeChar(OP_IADD, INTP_OP)},
    {"V_CVT_PKACCUM_U8_F32", OpcodeChar(OP_IADD, INTP_OP)},
    {"V_CVT_PKNORM_I16_F32", OpcodeChar(OP_FADD, SP_OP)},
    {"V_CVT_PKNORM_U16_F32", OpcodeChar(OP_IADD, INTP_OP)},
    {"V_CVT_PKRTZ_F16_F32", OpcodeChar(OP_IADD, INTP_OP)},
    // TODO DIV as MUL
    {"V_DIV_FIXUP_F16", OpcodeChar(OP_HMUL2, SP_OP)},
    {"V_DIV_FIXUP_F32", OpcodeChar(OP_FMUL, SP_OP)},
    {"V_DIV_FIXUP_F64", OpcodeChar(OP_DMUL, DP_OP)},
    {"V_DIV_FMAS_F32", OpcodeChar(OP_FMUL, SP_OP)},
    {"V_DIV_FMAS_F64", OpcodeChar(OP_DMUL, DP_OP)},
    // TODO Fused multiply add as MUL
    {"V_FMA_F16", OpcodeChar(OP_HMUL2, SP_OP)},
    {"V_FMA_F32", OpcodeChar(OP_FMUL, SP_OP)},
    {"V_FMA_F64", OpcodeChar(OP_DMUL, DP_OP)},
    // TODO ldexp as MUL
    {"V_LDEXP_F32", OpcodeChar(OP_FMUL, SP_OP)},
    {"V_LDEXP_F64", OpcodeChar(OP_DMUL, DP_OP)},
    // TODO Unsigned eight-bit pixel average on packed unsigned bytes
    {"V_LERP", OpcodeChar(OP_IADD, INTP_OP)},
    {"V_LSHLREV", OpcodeChar(OP_SHL, INTP_OP)},
    {"V_LSHLRREV", OpcodeChar(OP_SHR, INTP_OP)},
    // TODO Multiply-add as MUL
    {"V_MAD_F16", OpcodeChar(OP_HMUL2, SP_OP)},
    {"V_MAD_F32", OpcodeChar(OP_FMUL, SP_OP)},
    {"V_MAD", OpcodeChar(OP_IMUL, INTP_OP)},
    {"V_MAD_I32", OpcodeChar(OP_IMUL, INTP_OP)},
    {"V_MAD_I64", OpcodeChar(OP_IMUL, INTP_OP)},
    {"V_MAD_LEGACY_F32", OpcodeChar(OP_FMUL, SP_OP)},
    {"V_MAD_U32", OpcodeChar(OP_IMUL, INTP_OP)},
    {"V_MAD_U64", OpcodeChar(OP_IMUL, INTP_OP)},
    {"V_MAX_F64", OpcodeChar(OP_SEL, DP_OP)}, // TODO
    {"V_MAX3_F32", OpcodeChar(OP_SEL, SP_OP)}, // TODO
    {"V_MAX3", OpcodeChar(OP_SEL, INTP_OP)}, // TODO
    {"V_MBCNT_LO_U32", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    // TODO Median as SEL
    {"V_MED3_F32", OpcodeChar(OP_SEL, SP_OP)},
    {"V_MED3", OpcodeChar(OP_SEL, INTP_OP)},
    {"V_MIN_F64", OpcodeChar(OP_SEL, DP_OP)}, // TODO
    {"V_MIN3_F32", OpcodeChar(OP_SEL, SP_OP)}, // TODO
    {"V_MIN3", OpcodeChar(OP_SEL, INTP_OP)}, // TODO
    // TODO Masked Quad-Byte SAD with accum_lo/hi
    {"V_MQSAD_PK_U16", OpcodeChar(OP_IADD, INTP_OP)},
    {"V_MQSAD_U32", OpcodeChar(OP_IADD, INTP_OP)},
    {"V_MSAD", OpcodeChar(OP_IADD, INTP_OP)},
    {"V_MUL_F64", OpcodeChar(OP_DMUL, DP_OP)},
    {"V_MUL_HI", OpcodeChar(OP_IMUL, INTP_OP)},
    {"V_MUL_LO", OpcodeChar(OP_IMUL, INTP_OP)},
    // TODO Bit permute
    {"V_PERM", OpcodeChar(OP_IADD, INTP_OP)},
    {"V_QSAD_PK_U16", OpcodeChar(OP_IADD, INTP_OP)},  // TODO
    {"V_READLANE", OpcodeChar(OP_MOV, ALU_OP)},
    // TODO Sum of absolute differences with accumulation.
    {"V_SAD_HI", OpcodeChar(OP_IADD, INTP_OP)},
    {"V_SAD", OpcodeChar(OP_IADD, INTP_OP)},
    {"V_TRIG_PREOP_F64", OpcodeChar(OP_DMUL, DP_OP)},  // todo
    


};

#endif
