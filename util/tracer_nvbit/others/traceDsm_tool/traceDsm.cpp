// traceDsm: Disassembler for .tracez files
// Outputs either simulator-compatible text (.traceg format) or annotated
// human-readable format.
//
// Usage:
//   traceDsm <file.tracez>              # writes <file>.traceg next to input
//   traceDsm <file.tracez> --annotate   # human-readable format on stdout

#include <zstd.h>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "tracez_format.h"

using tracez::tb_index_t;
using tracez::warp_index_t;

static void fread_exact(void *buf, size_t size, size_t nmemb, FILE *fp,
                        const char *what) {
  if (fread(buf, size, nmemb, fp) != nmemb) {
    fprintf(stderr, "FATAL: short read on %s (corrupted .tracez?)\n", what);
    exit(1);
  }
}

enum OutputMode { MODE_SIM, MODE_ANNOTATE };

// Print instruction in annotated single-line format with labeled fields.
// Same structure as sim format but with field names for readability.
static void write_annotated(FILE *out, const std::string &line) {
  std::istringstream ss(line);
  std::string token;

  unsigned pc, mask;
  ss >> std::hex >> pc >> mask;
  std::bitset<32> active(mask);

  fprintf(out, "pc=0x%04x mask=0x%08x(%u) ", pc, mask,
          (unsigned)active.count());

  // Dest regs
  unsigned dest_count;
  ss >> std::dec >> dest_count;
  fprintf(out, "dst=");
  if (dest_count == 0) {
    fprintf(out, "- ");
  } else {
    for (unsigned i = 0; i < dest_count; i++) {
      ss >> token;
      fprintf(out, "%s", token.c_str());
      if (i < dest_count - 1) fprintf(out, ",");
    }
    fprintf(out, " ");
  }

  // Opcode
  ss >> token;
  fprintf(out, "op=%s ", token.c_str());
  std::string opcode = token;

  // Src regs
  unsigned src_count;
  ss >> std::dec >> src_count;
  fprintf(out, "src=");
  if (src_count == 0) {
    fprintf(out, "- ");
  } else {
    for (unsigned i = 0; i < src_count; i++) {
      ss >> token;
      fprintf(out, "%s", token.c_str());
      if (i < src_count - 1) fprintf(out, ",");
    }
    fprintf(out, " ");
  }

  // GMMA commit group
  if (opcode.find("GMMA") != std::string::npos) {
    unsigned commit;
    ss >> commit;
    fprintf(out, "gmma_commit=%u ", commit);
  }

  // Memory
  unsigned mem_width;
  ss >> std::dec >> mem_width;
  if (mem_width > 0) {
    unsigned addr_mode;
    ss >> addr_mode;

    fprintf(out, "width=%u ", mem_width);

    if (addr_mode == 1) {  // base_stride
      uint64_t base;
      int stride;
      ss >> std::hex >> base >> std::dec >> stride;
      fprintf(out, "[stride] base=0x%lx stride=%d ", base, stride);
    } else if (addr_mode == 2) {  // base_delta
      uint64_t base;
      ss >> std::hex >> base;
      fprintf(out, "[delta] base=0x%lx deltas=", base);
      // Deltas are interleaved with remaining fields; print rest of line
      std::string rest;
      std::getline(ss, rest);
      fprintf(out, "%s", rest.c_str());
      fprintf(out, "\n");
      return;
    } else if (addr_mode == 0) {  // list_all
      fprintf(out, "[flat] addrs=");
      for (unsigned i = 0; i < 32; i++) {
        if (active.test(i)) {
          uint64_t addr;
          ss >> std::hex >> addr;
          fprintf(out, "0x%lx,", addr);
        }
      }
      fprintf(out, " ");
    } else if (addr_mode == 3 || addr_mode == 4) {  // TMA
      const char *tma_mode = (addr_mode == 3) ? "tma_flat" : "tma_delta";
      fprintf(out, "[%s] ", tma_mode);
      // Print remaining TMA fields as-is
      std::string rest;
      std::getline(ss, rest);
      fprintf(out, "%s", rest.c_str());
      fprintf(out, "\n");
      return;
    }
  }

  // Remaining: imm, imm2, Val/NoVal, reg values
  std::string rest;
  std::getline(ss, rest);
  if (!rest.empty()) {
    size_t start = rest.find_first_not_of(" \t");
    if (start != std::string::npos) {
      fprintf(out, "imm=%s", rest.substr(start).c_str());
    }
  }
  fprintf(out, "\n");
}

int main(int argc, char *argv[]) {
  OutputMode mode = MODE_SIM;

  if (argc < 2 || argc > 3) {
    fprintf(stderr,
            "Usage: %s <file.tracez> [--annotate]\n"
            "  Default: write <file>.traceg next to the input.\n"
            "  --annotate: human-readable format with labeled fields (stdout).\n",
            argv[0]);
    return 1;
  }

  std::string filepath = argv[1];
  if (argc == 3) {
    if (std::string(argv[2]) == "--annotate")
      mode = MODE_ANNOTATE;
    else {
      fprintf(stderr, "Unknown option: %s\n", argv[2]);
      return 1;
    }
  }

  FILE *fp = fopen(filepath.c_str(), "rb");
  if (!fp) {
    fprintf(stderr, "Error: cannot open %s\n", filepath.c_str());
    return 1;
  }

  // MODE_SIM writes the .traceg next to the input (input must end in .tracez).
  // MODE_ANNOTATE stays on stdout.
  FILE *out = stdout;
  if (mode == MODE_SIM) {
    const std::string ext = ".tracez";
    if (filepath.size() <= ext.size() ||
        filepath.compare(filepath.size() - ext.size(), ext.size(), ext) != 0) {
      fprintf(stderr, "Error: input must have .tracez suffix: %s\n",
              filepath.c_str());
      fclose(fp);
      return 1;
    }
    std::string out_path =
        filepath.substr(0, filepath.size() - ext.size()) + ".traceg";
    out = fopen(out_path.c_str(), "w");
    if (!out) {
      fprintf(stderr, "Error: cannot open %s for writing\n", out_path.c_str());
      fclose(fp);
      return 1;
    }
    fprintf(stderr, "Writing to %s\n", out_path.c_str());
  }

  std::string header = tracez::read_kernel_header(fp);

  uint32_t dict_size, subchunk_insts;
  tracez::read_compression_header(fp, dict_size, subchunk_insts);

  // Read dictionary
  std::vector<char> dict_data(dict_size);
  if (dict_size > 0) {
    if (fread(dict_data.data(), 1, dict_size, fp) != dict_size) {
      fprintf(stderr, "FATAL: short read on dictionary (corrupted .tracez?)\n");
      return 1;
    }
  }
  ZSTD_DDict *ddict = nullptr;
  if (dict_size > 0) {
    ddict = ZSTD_createDDict(dict_data.data(), dict_size);
  }
  ZSTD_DCtx *dctx = ZSTD_createDCtx();

  std::vector<tb_index_t> tb_index;
  tracez::read_index(fp, tb_index);

  // Print file info
  if (mode == MODE_ANNOTATE) {
    fprintf(stdout, "=== TRACEZ File Info ===\n");
    fprintf(stdout, "Dict size: %u bytes\n", dict_size);
    fprintf(stdout, "Subchunk insts: %u\n", subchunk_insts);
    fprintf(stdout, "Thread blocks: %zu\n", tb_index.size());
    uint64_t total_insts = 0;
    for (auto &tb : tb_index)
      for (auto &w : tb.warps) total_insts += w.inst_count;
    fprintf(stdout, "Total instructions: %lu\n\n", total_insts);
  }

  // Print kernel header
  if (mode == MODE_SIM) {
    // Output header lines as-is
    std::istringstream hss(header);
    std::string line;
    while (std::getline(hss, line)) {
      fprintf(out, "%s\n", line.c_str());
    }
  } else {
    fprintf(stdout, "=== Kernel Header ===\n%s\n\n", header.c_str());
  }

  // Process each TB
  for (uint32_t t = 0; t < tb_index.size(); t++) {
    const auto &tb = tb_index[t];

    if (mode == MODE_SIM) {
      fprintf(out, "\n#BEGIN_TB\n");
      fprintf(out, "\nthread block = %u,%u,%u\n", tb.tb_id_x, tb.tb_id_y,
              tb.tb_id_z);
      fprintf(out, "cluster id = %u,%u,%u\n", tb.cluster_id_x,
              tb.cluster_id_y, tb.cluster_id_z);
      fprintf(out, "cluster cta = %u,%u,%u\n", tb.cluster_cta_id_x,
              tb.cluster_cta_id_y, tb.cluster_cta_id_z);
      fprintf(out, "cluster rank = %u\n", tb.cluster_rank);
    } else {
      fprintf(stdout, "=== TB %u,%u,%u (cluster=%u,%u,%u rank=%u) ===\n",
              tb.tb_id_x, tb.tb_id_y, tb.tb_id_z, tb.cluster_id_x,
              tb.cluster_id_y, tb.cluster_id_z, tb.cluster_rank);
    }

    for (uint32_t w = 0; w < tb.warp_count; w++) {
      const auto &warp = tb.warps[w];

      // Seek to warp data and skip warp header (inst_count + num_subchunks)
      fseeko(fp, warp.file_offset + 8, SEEK_SET);

      // Decompress all sub-chunks
      std::vector<std::string> lines;
      for (uint32_t sc = 0; sc < warp.num_subchunks; sc++) {
        uint32_t comp_size;
        fread_exact(&comp_size, sizeof(comp_size), 1, fp, "subchunk comp_size");
        std::vector<char> comp_buf(comp_size);
        fread_exact(comp_buf.data(), 1, comp_size, fp, "subchunk data");

        unsigned long long decomp_size =
            ZSTD_getFrameContentSize(comp_buf.data(), comp_size);
        if (decomp_size == ZSTD_CONTENTSIZE_ERROR ||
            decomp_size == ZSTD_CONTENTSIZE_UNKNOWN) {
          fprintf(stderr,
                  "FATAL: zstd frame missing content size (corrupted .tracez?)\n");
          return 1;
        }

        std::vector<char> decomp_buf(decomp_size + 1);
        size_t actual;
        if (ddict)
          actual = ZSTD_decompress_usingDDict(dctx, decomp_buf.data(),
                                              decomp_buf.size(),
                                              comp_buf.data(), comp_size,
                                              ddict);
        else
          actual = ZSTD_decompressDCtx(dctx, decomp_buf.data(),
                                       decomp_buf.size(), comp_buf.data(),
                                       comp_size);
        if (ZSTD_isError(actual)) {
          fprintf(stderr, "zstd error: %s\n", ZSTD_getErrorName(actual));
          return 1;
        }

        decomp_buf[actual] = '\0';
        char *p = decomp_buf.data();
        char *end = p + actual;
        while (p < end) {
          char *nl = (char *)memchr(p, '\n', end - p);
          if (nl) {
            lines.emplace_back(p, nl - p);
            p = nl + 1;
          } else {
            if (p < end) lines.emplace_back(p, end - p);
            break;
          }
        }
      }

      if (mode == MODE_SIM) {
        fprintf(out, "\nwarp = %u\n", w);
        fprintf(out, "insts = %u\n", warp.inst_count);
        for (const auto &line : lines) {
          fprintf(out, "%s\n", line.c_str());
        }
      } else {
        fprintf(stdout, "  Warp %u: %u insts, %u subchunks\n", w,
                warp.inst_count, warp.num_subchunks);
        for (unsigned i = 0; i < lines.size(); i++) {
          write_annotated(stdout, lines[i]);
        }
      }
    }

    if (mode == MODE_SIM) {
      fprintf(out, "\n#END_TB\n");
    }
  }

  ZSTD_freeDCtx(dctx);
  if (ddict) ZSTD_freeDDict(ddict);
  fclose(fp);
  if (out != stdout) fclose(out);
  return 0;
}
