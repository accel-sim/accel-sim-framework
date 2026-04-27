// On-disk format definitions for the .tracez compressed trace format.
// Shared by the writer (post-traces-processing), the simulator reader
// (warp_trace_stream), and the traceDsm disassembler. Header-only so it
// can be included from both the tracer Makefile build and the simulator
// CMake build without a link dependency.
//
// File layout:
//   [null-terminated kernel header text]
//   [u32 dict_size][u32 subchunk_insts]
//   [dict_size bytes of zstd dictionary]
//   [per-warp zstd sub-chunk payloads ...]
//   [index footer: num_tbs + per-TB records]
//   [u64 index_offset]  (last 8 bytes)

#ifndef TRACEZ_FORMAT_H
#define TRACEZ_FORMAT_H

#include <stdint.h>
#include <stdio.h>
#include <string>
#include <vector>

namespace tracez {

struct warp_index_t {
  uint32_t inst_count;
  uint32_t num_subchunks;
  uint64_t file_offset; // offset to warp header in file
  std::vector<uint32_t> subchunk_inst_counts;
};

struct tb_index_t {
  uint32_t tb_id_x, tb_id_y, tb_id_z;
  uint32_t cluster_id_x, cluster_id_y, cluster_id_z;
  uint32_t cluster_cta_id_x, cluster_cta_id_y, cluster_cta_id_z;
  uint32_t cluster_rank;
  uint32_t warp_count;
  std::vector<warp_index_t> warps;
};

// Read the null-terminated kernel header text at the start of the file.
inline std::string read_kernel_header(FILE *fp) {
  std::string header;
  int ch;
  while ((ch = fgetc(fp)) != EOF && ch != '\0')
    header += (char)ch;
  return header;
}

// Read the two u32s following the kernel header.
inline void read_compression_header(FILE *fp, uint32_t &dict_size,
                                    uint32_t &subchunk_insts) {
  fread(&dict_size, sizeof(dict_size), 1, fp);
  fread(&subchunk_insts, sizeof(subchunk_insts), 1, fp);
}

// Seek to the footer, read the index, leave fp positioned after the index.
inline void read_index(FILE *fp, std::vector<tb_index_t> &out) {
  fseeko(fp, -8, SEEK_END);
  uint64_t index_offset;
  fread(&index_offset, sizeof(index_offset), 1, fp);
  fseeko(fp, (off_t)index_offset, SEEK_SET);
  uint32_t num_tbs;
  fread(&num_tbs, sizeof(num_tbs), 1, fp);
  out.resize(num_tbs);
  for (uint32_t i = 0; i < num_tbs; ++i) {
    auto &tb = out[i];
    fread(&tb.tb_id_x, sizeof(uint32_t), 1, fp);
    fread(&tb.tb_id_y, sizeof(uint32_t), 1, fp);
    fread(&tb.tb_id_z, sizeof(uint32_t), 1, fp);
    fread(&tb.cluster_id_x, sizeof(uint32_t), 1, fp);
    fread(&tb.cluster_id_y, sizeof(uint32_t), 1, fp);
    fread(&tb.cluster_id_z, sizeof(uint32_t), 1, fp);
    fread(&tb.cluster_cta_id_x, sizeof(uint32_t), 1, fp);
    fread(&tb.cluster_cta_id_y, sizeof(uint32_t), 1, fp);
    fread(&tb.cluster_cta_id_z, sizeof(uint32_t), 1, fp);
    fread(&tb.cluster_rank, sizeof(uint32_t), 1, fp);
    fread(&tb.warp_count, sizeof(uint32_t), 1, fp);
    tb.warps.resize(tb.warp_count);
    for (uint32_t j = 0; j < tb.warp_count; ++j) {
      auto &w = tb.warps[j];
      fread(&w.inst_count, sizeof(uint32_t), 1, fp);
      fread(&w.num_subchunks, sizeof(uint32_t), 1, fp);
      fread(&w.file_offset, sizeof(uint64_t), 1, fp);
      w.subchunk_inst_counts.resize(w.num_subchunks);
      for (uint32_t k = 0; k < w.num_subchunks; ++k) {
        fread(&w.subchunk_inst_counts[k], sizeof(uint32_t), 1, fp);
      }
    }
  }
}

// Write the index footer and the trailing u64 index_offset.
inline void write_index(FILE *fp, const std::vector<tb_index_t> &index,
                        uint64_t index_offset) {
  uint32_t n = (uint32_t)index.size();
  fwrite(&n, sizeof(n), 1, fp);
  for (const auto &tb : index) {
    fwrite(&tb.tb_id_x, sizeof(uint32_t), 1, fp);
    fwrite(&tb.tb_id_y, sizeof(uint32_t), 1, fp);
    fwrite(&tb.tb_id_z, sizeof(uint32_t), 1, fp);
    fwrite(&tb.cluster_id_x, sizeof(uint32_t), 1, fp);
    fwrite(&tb.cluster_id_y, sizeof(uint32_t), 1, fp);
    fwrite(&tb.cluster_id_z, sizeof(uint32_t), 1, fp);
    fwrite(&tb.cluster_cta_id_x, sizeof(uint32_t), 1, fp);
    fwrite(&tb.cluster_cta_id_y, sizeof(uint32_t), 1, fp);
    fwrite(&tb.cluster_cta_id_z, sizeof(uint32_t), 1, fp);
    fwrite(&tb.cluster_rank, sizeof(uint32_t), 1, fp);
    fwrite(&tb.warp_count, sizeof(uint32_t), 1, fp);
    for (const auto &w : tb.warps) {
      fwrite(&w.inst_count, sizeof(uint32_t), 1, fp);
      fwrite(&w.num_subchunks, sizeof(uint32_t), 1, fp);
      fwrite(&w.file_offset, sizeof(uint64_t), 1, fp);
      for (uint32_t sc = 0; sc < w.num_subchunks; ++sc) {
        fwrite(&w.subchunk_inst_counts[sc], sizeof(uint32_t), 1, fp);
      }
    }
  }
  fwrite(&index_offset, sizeof(index_offset), 1, fp);
}

} // namespace tracez

#endif // TRACEZ_FORMAT_H
