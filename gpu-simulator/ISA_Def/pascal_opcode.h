// developed by Mahmoud Khairy, Purdue Univ
// abdallm@purdue.edu

#ifndef PASCAL_OPCODE_H
#define PASCAL_OPCODE_H

#include "abstract_hardware_model.h"
#include "trace_opcode.h"
#include <string>
#include <unordered_map>

#define PASCAL_TITANX_BINART_VERSION 61
#define PASCAL_P100_BINART_VERSION 60

// TO DO: moving this to a yml or def files

/// Pascal SM_61 ISA
// see: https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
static const std::unordered_map<std::string, OpcodeChar> Pascal_OpcodeMap = {
    // Floating Point 32 Instructions
    {"FADD", OpcodeChar(OP_FADD, SP_OP, FP__OP, FP_OP)},
    {"FADD32I", OpcodeChar(OP_FADD32I, SP_OP, FP__OP, FP_OP)},
    {"FCHK", OpcodeChar(OP_FCHK, SP_OP, FP__OP, FP_OP)},
    {"FCMP", OpcodeChar(OP_FCMP, SP_OP, FP__OP, FP_OP)},
    {"FFMA32I", OpcodeChar(OP_FFMA32I, SP_OP, FP_MUL_OP, FP_OP)},
    {"FFMA", OpcodeChar(OP_FFMA, SP_OP, FP_MUL_OP, FP_OP)},
    {"FMNMX", OpcodeChar(OP_FMNMX, SP_OP, FP__OP, FP_OP)},
    {"FMUL", OpcodeChar(OP_FMUL, SP_OP, FP_MUL_OP, FP_OP)},
    {"FMUL32I", OpcodeChar(OP_FMUL32I, SP_OP, FP_MUL_OP, FP_OP)},
    {"FSEL", OpcodeChar(OP_FSEL, SP_OP, FP__OP, FP_OP)},
    {"FSET", OpcodeChar(OP_FSET, SP_OP, FP__OP, FP_OP)},
    {"FSETP", OpcodeChar(OP_FSETP, SP_OP, FP__OP, FP_OP)},
    {"FSWZADD", OpcodeChar(OP_FSWZADD, SP_OP, FP__OP, FP_OP)},
    {"RRO", OpcodeChar(OP_RRO, SP_OP, FP__OP, FP_OP)},

    // SFU
    {"MUFU", OpcodeChar(OP_MUFU, SFU_OP, FP_SIN_OP, FP_OP)},

    // Floating Point 16 Instructions
    {"HADD2", OpcodeChar(OP_HADD2, SP_OP, FP__OP, FP_OP)},
    {"HFMA2", OpcodeChar(OP_HFMA2, SP_OP, FP_MUL_OP, FP_OP)},
    {"HMUL2", OpcodeChar(OP_HMUL2, SP_OP, FP_MUL_OP, FP_OP)},
    {"HSET2", OpcodeChar(OP_HSET2, SP_OP, FP__OP, FP_OP)},
    {"HSETP2", OpcodeChar(OP_HSETP2, SP_OP, FP__OP, FP_OP)},

    // Double Point Instructions
    {"DADD", OpcodeChar(OP_DADD, DP_OP, DP___OP, FP_OP)},
    {"DFMA", OpcodeChar(OP_DFMA, DP_OP, DP_MUL_OP, FP_OP)},
    {"DMUL", OpcodeChar(OP_DMUL, DP_OP, DP_MUL_OP, FP_OP)},
    {"DSETP", OpcodeChar(OP_DSETP, DP_OP, DP___OP, FP_OP)},
    {"DMNMX", OpcodeChar(OP_DMNMX, DP_OP, DP___OP, FP_OP)},
    {"DSET", OpcodeChar(OP_DSET, DP_OP, DP___OP, FP_OP)},

    // Integer Instructions
    {"BMSK", OpcodeChar(OP_BMSK, INTP_OP, INT__OP, INT_OP)},
    {"BREV", OpcodeChar(OP_BREV, INTP_OP, INT__OP, INT_OP)},
    {"FLO", OpcodeChar(OP_FLO, INTP_OP, INT__OP, INT_OP)},
    {"IABS", OpcodeChar(OP_IABS, INTP_OP, INT__OP, INT_OP)},
    {"IADD", OpcodeChar(OP_IADD, INTP_OP, INT__OP, INT_OP)},
    {"IADD3", OpcodeChar(OP_IADD3, INTP_OP, INT__OP, INT_OP)},
    {"IADD32I", OpcodeChar(OP_IADD32I, INTP_OP, INT__OP, INT_OP)},
    {"IDP", OpcodeChar(OP_IDP, INTP_OP, INT_MUL_OP, INT_OP)},
    {"IDP4A", OpcodeChar(OP_IDP4A, INTP_OP, INT_MUL_OP, INT_OP)},
    {"IMAD", OpcodeChar(OP_IMAD, INTP_OP, INT_MUL_OP, INT_OP)},
    {"IMMA", OpcodeChar(OP_IMMA, INTP_OP, TENSOR__OP, INT_OP)},
    {"IMNMX", OpcodeChar(OP_IMNMX, INTP_OP, INT__OP, INT_OP)},
    {"IMUL", OpcodeChar(OP_IMUL, INTP_OP, INT_MUL_OP, INT_OP)},
    {"IMUL32I", OpcodeChar(OP_IMUL32I, INTP_OP, INT_MUL_OP, INT_OP)},
    {"ISCADD", OpcodeChar(OP_ISCADD, INTP_OP, INT_MUL_OP, INT_OP)},
    {"ISCADD32I", OpcodeChar(OP_ISCADD32I, INTP_OP, INT_MUL_OP, INT_OP)},
    {"ISETP", OpcodeChar(OP_ISETP, INTP_OP, INT__OP, INT_OP)},
    {"ISET", OpcodeChar(OP_ISET, INTP_OP, INT__OP, INT_OP)},
    {"LEA", OpcodeChar(OP_LEA, INTP_OP, INT_MUL_OP, INT_OP)},
    {"LOP", OpcodeChar(OP_LOP, INTP_OP, INT__OP, INT_OP)},
    {"LOP3", OpcodeChar(OP_LOP3, INTP_OP, INT__OP, INT_OP)},
    {"LOP32I", OpcodeChar(OP_LOP32I, INTP_OP, INT__OP, INT_OP)},
    {"POPC", OpcodeChar(OP_POPC, INTP_OP, INT__OP, INT_OP)},
    {"SHF", OpcodeChar(OP_SHF, INTP_OP, INT__OP, INT_OP)},
    {"SHR", OpcodeChar(OP_SHR, INTP_OP, INT__OP, INT_OP)},
    {"VABSDIFF", OpcodeChar(OP_VABSDIFF, INTP_OP, INT__OP, INT_OP)},
    {"VABSDIFF4", OpcodeChar(OP_VABSDIFF4, INTP_OP, INT__OP, INT_OP)},
    {"BFE", OpcodeChar(OP_BFE, INTP_OP, INT__OP, INT_OP)},
    {"BFI", OpcodeChar(OP_BFI, INTP_OP, INT__OP, INT_OP)},
    {"ICMP", OpcodeChar(OP_ICMP, INTP_OP, INT__OP, INT_OP)},
    {"IMADSP", OpcodeChar(OP_IMADSP, INTP_OP, INT_MUL_OP, INT_OP)},
    {"SHL", OpcodeChar(OP_SHL, INTP_OP, INT__OP, INT_OP)},
    {"XMAD", OpcodeChar(OP_XMAD, INTP_OP, INT_MUL_OP, INT_OP)},
    {"VMNMX", OpcodeChar(OP_VMNMX, INTP_OP, INT__OP, INT_OP)},

    // Conversion Instructions
    {"F2F", OpcodeChar(OP_F2F, ALU_OP, FP__OP, FP_OP)},
    {"F2I", OpcodeChar(OP_F2I, ALU_OP, FP__OP, FP_OP)},
    {"I2F", OpcodeChar(OP_I2F, ALU_OP, FP__OP, FP_OP)},
    {"I2I", OpcodeChar(OP_I2I, ALU_OP, INT__OP, INT_OP)},
    {"I2IP", OpcodeChar(OP_I2IP, ALU_OP, INT__OP, INT_OP)},
    {"FRND", OpcodeChar(OP_FRND, ALU_OP, INT__OP, FP_OP)},


    // Movement Instructions
    {"MOV", OpcodeChar(OP_MOV, ALU_OP, INT__OP, FP_OP)},
    {"MOV32I", OpcodeChar(OP_MOV32I, ALU_OP, INT__OP, FP_OP)},
    {"PRMT", OpcodeChar(OP_PRMT, ALU_OP, INT__OP, FP_OP)},
    {"SEL", OpcodeChar(OP_SEL, ALU_OP, INT__OP, INT_OP)},
    {"SGXT", OpcodeChar(OP_SGXT, ALU_OP, INT__OP, INT_OP)},
    {"SHFL", OpcodeChar(OP_SHFL, ALU_OP, INT__OP, INT_OP)},

    // Predicate Instructions
    {"PLOP3", OpcodeChar(OP_PLOP3, ALU_OP, INT__OP, INT_OP)},
    {"PSETP", OpcodeChar(OP_PSETP, ALU_OP, INT__OP, INT_OP)},
    {"P2R", OpcodeChar(OP_P2R, ALU_OP, INT__OP, INT_OP)},
    {"R2P", OpcodeChar(OP_R2P, ALU_OP, INT__OP, INT_OP)},
    {"CSET", OpcodeChar(OP_CSET, ALU_OP, INT__OP, INT_OP)},
    {"CSETP", OpcodeChar(OP_CSETP, ALU_OP, INT__OP, INT_OP)},
    {"PSET", OpcodeChar(OP_PSET, ALU_OP, INT__OP, INT_OP)},

    // Load/Store Instructions
    {"LD", OpcodeChar(OP_LD, LOAD_OP, OTHER_OP, FP_OP)},
    {"LDC", OpcodeChar(OP_LDC, LOAD_OP, OTHER_OP, FP_OP)},
    {"LDG", OpcodeChar(OP_LDG, LOAD_OP, OTHER_OP, FP_OP)},
    {"LDL", OpcodeChar(OP_LDL, LOAD_OP, OTHER_OP, FP_OP)},
    {"LDS", OpcodeChar(OP_LDS, LOAD_OP, OTHER_OP, FP_OP)},
    {"ST", OpcodeChar(OP_ST, STORE_OP, OTHER_OP, FP_OP)},
    {"STG", OpcodeChar(OP_STG, STORE_OP, OTHER_OP, FP_OP)},
    {"STL", OpcodeChar(OP_STL, STORE_OP, OTHER_OP, FP_OP)},
    {"STS", OpcodeChar(OP_STS, STORE_OP, OTHER_OP, FP_OP)},
    {"MATCH", OpcodeChar(OP_MATCH, ALU_OP, OTHER_OP, FP_OP)},
    {"QSPC", OpcodeChar(OP_QSPC, ALU_OP, OTHER_OP, FP_OP)},
    {"ATOM", OpcodeChar(OP_ATOM, STORE_OP, OTHER_OP, FP_OP)},
    {"ATOMS", OpcodeChar(OP_ATOMS, STORE_OP, OTHER_OP, FP_OP)},
    {"ATOMG", OpcodeChar(OP_ATOMG, STORE_OP, OTHER_OP, FP_OP)},
    {"RED", OpcodeChar(OP_RED, STORE_OP, OTHER_OP, FP_OP)},
    {"CCTL", OpcodeChar(OP_CCTL, ALU_OP, OTHER_OP, FP_OP)},
    {"CCTLL", OpcodeChar(OP_CCTLL, ALU_OP, OTHER_OP, FP_OP)},
    {"ERRBAR", OpcodeChar(OP_ERRBAR, ALU_OP, OTHER_OP, UN_OP)},
    {"MEMBAR", OpcodeChar(OP_MEMBAR, MEMORY_BARRIER_OP, OTHER_OP, UN_OP)},
    {"CCTLT", OpcodeChar(OP_CCTLT, ALU_OP, OTHER_OP, FP_OP)},

    // Texture Instructions
    // For now, we ignore texture loads, consider it as ALU_OP
    {"TEX", OpcodeChar(OP_TEX, ALU_OP, TEX__OP, FP_OP)},
    {"TLD", OpcodeChar(OP_TLD, ALU_OP, TEX__OP, FP_OP)},
    {"TLD4", OpcodeChar(OP_TLD4, ALU_OP, TEX__OP, FP_OP)},
    {"TMML", OpcodeChar(OP_TMML, ALU_OP, TEX__OP, FP_OP)},
    {"TXD", OpcodeChar(OP_TXD, ALU_OP, TEX__OP, FP_OP)},
    {"TXQ", OpcodeChar(OP_TXQ, ALU_OP, TEX__OP, FP_OP)},
    {"TEXS", OpcodeChar(OP_TEXS, ALU_OP, TEX__OP, FP_OP)},
    {"TLD4S", OpcodeChar(OP_TLD4S, ALU_OP, TEX__OP, FP_OP)},
    {"TLDS", OpcodeChar(OP_TLDS, ALU_OP, TEX__OP, FP_OP)},

    // Control Instructions
    {"BMOV", OpcodeChar(OP_BMOV, BRANCH_OP, OTHER_OP, UN_OP)},
    {"BPT", OpcodeChar(OP_BPT, BRANCH_OP, OTHER_OP, UN_OP)},
    {"BRA", OpcodeChar(OP_BRA, BRANCH_OP, OTHER_OP, UN_OP)},
    {"BREAK", OpcodeChar(OP_BREAK, BRANCH_OP, OTHER_OP, UN_OP)},
    {"BRX", OpcodeChar(OP_BRX, BRANCH_OP, OTHER_OP, UN_OP)},
    {"BSSY", OpcodeChar(OP_BSSY, BRANCH_OP, OTHER_OP, UN_OP)},
    {"BSYNC", OpcodeChar(OP_BSYNC, BRANCH_OP, OTHER_OP, UN_OP)},
    {"CALL", OpcodeChar(OP_CALL, CALL_OPS, OTHER_OP, UN_OP)},
    {"EXIT", OpcodeChar(OP_EXIT, EXIT_OPS, OTHER_OP, UN_OP)},
    {"JMP", OpcodeChar(OP_JMP, BRANCH_OP, OTHER_OP, UN_OP)},
    {"SSY", OpcodeChar(OP_SSY, BRANCH_OP, OTHER_OP, UN_OP)},
    {"SYNC", OpcodeChar(OP_SYNC, BRANCH_OP, OTHER_OP, UN_OP)},
    {"JMX", OpcodeChar(OP_JMX, BRANCH_OP, OTHER_OP, UN_OP)},
    {"KILL", OpcodeChar(OP_KILL, BRANCH_OP, OTHER_OP, UN_OP)},
    {"NANOSLEEP", OpcodeChar(OP_NANOSLEEP, BRANCH_OP, OTHER_OP, UN_OP)},
    {"RET", OpcodeChar(OP_RET, RET_OPS, OTHER_OP, UN_OP)},
    {"RPCMOV", OpcodeChar(OP_RPCMOV, BRANCH_OP, OTHER_OP, UN_OP)},
    {"RTT", OpcodeChar(OP_RTT, RET_OPS, OTHER_OP, UN_OP)},
    {"WARPSYNC", OpcodeChar(OP_WARPSYNC, BRANCH_OP, OTHER_OP, UN_OP)},
    {"YIELD", OpcodeChar(OP_YIELD, BRANCH_OP, OTHER_OP, UN_OP)},
    {"CAL", OpcodeChar(OP_CAL, CALL_OPS, OTHER_OP, UN_OP)},
    {"JCAL", OpcodeChar(OP_JCAL, CALL_OPS, OTHER_OP, UN_OP)},
    {"PRET", OpcodeChar(OP_PRET, CALL_OPS, OTHER_OP, UN_OP)},
    {"BRK", OpcodeChar(OP_BRK, CALL_OPS, OTHER_OP, UN_OP)},
    {"PBK", OpcodeChar(OP_PBK, CALL_OPS, OTHER_OP, UN_OP)},
    {"CONT", OpcodeChar(OP_CONT, CALL_OPS, OTHER_OP, UN_OP)},
    {"PCNT", OpcodeChar(OP_PCNT, CALL_OPS, OTHER_OP, UN_OP)},
    {"PEXIT", OpcodeChar(OP_PEXIT, CALL_OPS, OTHER_OP, UN_OP)},

    // Miscellaneous Instructions
    {"B2R", OpcodeChar(OP_B2R, ALU_OP, OTHER_OP, FP_OP)},
    {"BAR", OpcodeChar(OP_BAR, BARRIER_OP, OTHER_OP, UN_OP)},
    {"CS2R", OpcodeChar(OP_CS2R, ALU_OP, INT__OP, FP_OP)},
    {"CSMTEST", OpcodeChar(OP_CSMTEST, ALU_OP, OTHER_OP, FP_OP)},
    {"DEPBAR", OpcodeChar(OP_DEPBAR, ALU_OP, OTHER_OP, UN_OP)},
    {"GETLMEMBASE", OpcodeChar(OP_GETLMEMBASE, ALU_OP, OTHER_OP, FP_OP)},
    {"LEPC", OpcodeChar(OP_LEPC, ALU_OP, OTHER_OP, FP_OP)},
    {"NOP", OpcodeChar(OP_NOP, ALU_OP, OTHER_OP, UN_OP)},
    {"PMTRIG", OpcodeChar(OP_PMTRIG, ALU_OP, OTHER_OP, FP_OP)},
    {"R2B", OpcodeChar(OP_R2B, ALU_OP, OTHER_OP, FP_OP)},
    {"S2R", OpcodeChar(OP_S2R, ALU_OP, OTHER_OP, FP_OP)},
    {"SETCTAID", OpcodeChar(OP_SETCTAID, ALU_OP, OTHER_OP, FP_OP)},
    {"SETLMEMBASE", OpcodeChar(OP_SETLMEMBASE, ALU_OP, OTHER_OP, FP_OP)},
    {"VOTE", OpcodeChar(OP_VOTE, ALU_OP, OTHER_OP, FP_OP)},
    {"VOTE_VTG", OpcodeChar(OP_VOTE_VTG, ALU_OP, OTHER_OP, FP_OP)},
    
};

#endif
