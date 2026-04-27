// Per-warp streaming reader for .tracez (zstd-compressed text trace) files.
// Supports on-demand sub-chunk decompression with a sliding window.
// Uses pread() on a shared fd for lock-free parallel warp reads.

#ifndef WARP_TRACE_STREAM_H
#define WARP_TRACE_STREAM_H

#include <fcntl.h>
#include <unistd.h>
#include <zstd.h>
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "trace_parser.h"
#include "tracez_format.h"

using tracez::tb_index_t;
using tracez::warp_index_t;

// Helper: read exactly 'count' bytes from fd at offset using pread.
static inline void pread_exact(int fd, void *buf, size_t count, off_t offset) {
  size_t total = 0;
  while (total < count) {
    ssize_t n = pread(fd, (char *)buf + total, count - total, offset + total);
    if (n <= 0) {
      fprintf(stderr,
              "FATAL: pread failed at offset %ld, requested %zu bytes: %s\n",
              (long)(offset + total), count - total,
              n == 0 ? "unexpected EOF" : strerror(errno));
      abort();
    }
    total += n;
  }
}

// Reads .tracez file: header, dictionary, index.
// Owns the single fd shared by all WarpTraceStreams for this kernel.
class TracezReader {
 public:
  TracezReader() = default;
  ~TracezReader() {
    if (fd_ >= 0) close(fd_);
    if (ddict_) ZSTD_freeDDict(ddict_);
  }

  TracezReader(const TracezReader &) = delete;
  TracezReader &operator=(const TracezReader &) = delete;

  // Open a .tracez file: read header, dict, and index from footer.
  // Returns the kernel header text (lines before the null terminator).
  std::string open(const std::string &filepath) {
    // Use FILE* for initial sequential parsing, then switch to fd for pread
    FILE *fp = fopen(filepath.c_str(), "rb");
    if (!fp) throw std::runtime_error("TracezReader: cannot open " + filepath);

    // Read kernel header (null-terminated text)
    std::string header;
    int ch;
    while ((ch = fgetc(fp)) != EOF && ch != '\0') {
      header += (char)ch;
    }

    // Read compression header
    uint32_t subchunk_insts_unused;
    fread(&dict_size_, sizeof(dict_size_), 1, fp);
    fread(&subchunk_insts_unused, sizeof(subchunk_insts_unused), 1, fp);

    // Read dictionary
    if (dict_size_ > 0) {
      dict_data_.resize(dict_size_);
      fread(dict_data_.data(), 1, dict_size_, fp);
      ddict_ = ZSTD_createDDict(dict_data_.data(), dict_size_);
    }

    // Read index from footer: seek to end - 8 (u64 index_offset)
    fseeko(fp, -8, SEEK_END);
    uint64_t index_offset;
    fread(&index_offset, sizeof(index_offset), 1, fp);

    // Read index
    fseeko(fp, index_offset, SEEK_SET);
    uint32_t num_tbs;
    fread(&num_tbs, sizeof(num_tbs), 1, fp);
    tb_index_.resize(num_tbs);
    for (uint32_t i = 0; i < num_tbs; ++i) {
      auto &tb = tb_index_[i];
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
        fread(&tb.warps[j].inst_count, sizeof(uint32_t), 1, fp);
        fread(&tb.warps[j].num_subchunks, sizeof(uint32_t), 1, fp);
        fread(&tb.warps[j].file_offset, sizeof(uint64_t), 1, fp);
        tb.warps[j].subchunk_inst_counts.resize(tb.warps[j].num_subchunks);
        for (uint32_t k = 0; k < tb.warps[j].num_subchunks; ++k) {
          fread(&tb.warps[j].subchunk_inst_counts[k], sizeof(uint32_t), 1, fp);
        }
      }
    }

    fclose(fp);

    // Open fd for pread (shared by all WarpTraceStreams)
    fd_ = ::open(filepath.c_str(), O_RDONLY);
    assert(fd_ >= 0);

    tb_counter_ = 0;
    return header;
  }

  int fd() const { return fd_; }
  ZSTD_DDict *ddict() const { return ddict_; }

  const tb_index_t *get_next_tb_index() {
    if (tb_counter_ >= tb_index_.size()) return nullptr;
    return &tb_index_[tb_counter_++];
  }

 private:
  int fd_ = -1;
  uint32_t dict_size_ = 0;
  std::vector<char> dict_data_;
  ZSTD_DDict *ddict_ = nullptr;
  std::vector<tb_index_t> tb_index_;
  size_t tb_counter_ = 0;
};

// Per-warp double-buffered streaming over zstd-compressed instruction
// sub-chunks. One buffer serves the simulator while a background thread
// prefetches the next sub-chunk into the other buffer. Sub-chunks may have
// variable sizes (replay regions are never split across boundaries).
class WarpTraceStream {
 public:
  WarpTraceStream() = default;
  ~WarpTraceStream() {
    join_prefetch();
    for (int i = 0; i < 2; i++) {
      if (buf_[i].dctx) ZSTD_freeDCtx(buf_[i].dctx);
    }
    // fd_ is NOT owned — TracezReader owns it
  }

  WarpTraceStream(const WarpTraceStream &) = delete;
  WarpTraceStream &operator=(const WarpTraceStream &) = delete;

  // Initialize: record warp metadata, scan sub-chunk offsets via pread.
  void init(int fd, const warp_index_t &w_idx, ZSTD_DDict *ddict,
            unsigned trace_version, unsigned enable_lineinfo, dim3 block_id,
            dim3 cluster_cta_id, dim3 cluster_id, unsigned cluster_rank) {
    fd_ = fd;
    ddict_ = ddict;
    w_idx_ = w_idx;
    trace_version_ = trace_version;
    enable_lineinfo_ = enable_lineinfo;
    block_id_ = block_id;
    cluster_cta_id_ = cluster_cta_id;
    cluster_id_ = cluster_id;
    cluster_rank_ = cluster_rank;

    // Create a ZSTD_DCtx per buffer for thread safety
    for (int i = 0; i < 2; i++) {
      buf_[i].dctx = ZSTD_createDCtx();
    }

    // Build cumulative instruction count array from per-subchunk counts
    uint32_t num_sc = w_idx_.num_subchunks;
    sc_cumulative_.resize(num_sc + 1);
    sc_cumulative_[0] = 0;
    for (uint32_t i = 0; i < num_sc; i++) {
      sc_cumulative_[i + 1] =
          sc_cumulative_[i] + w_idx_.subchunk_inst_counts[i];
    }

    // Scan sub-chunk byte offsets from the warp's data region.
    off_t pos = w_idx_.file_offset;
    uint32_t inst_count_file, num_subchunks_file;
    pread_exact(fd_, &inst_count_file, sizeof(inst_count_file), pos);
    pos += sizeof(inst_count_file);
    pread_exact(fd_, &num_subchunks_file, sizeof(num_subchunks_file), pos);
    pos += sizeof(num_subchunks_file);

    subchunk_offsets_.resize(num_subchunks_file);
    for (uint32_t sc = 0; sc < num_subchunks_file; ++sc) {
      subchunk_offsets_[sc] = pos;
      uint32_t comp_size;
      pread_exact(fd_, &comp_size, sizeof(comp_size), pos);
      pos += sizeof(comp_size) + comp_size;
    }
  }

  uint32_t total_inst_count() const { return w_idx_.inst_count; }

  // Get instruction at global index. Uses double-buffered prefetch.
  inst_trace_t &get(unsigned index) {
    assert(index < w_idx_.inst_count);
    unsigned sc = which_subchunk(index);

    if (sc == buf_[active_].sc_id) {
      // Hit: in active buffer
    } else if (sc == buf_[1 - active_].sc_id) {
      // In back buffer (prefetched) — swap and prefetch next
      join_prefetch();
      active_ = 1 - active_;
      start_prefetch(sc + 1);
    } else {
      // Miss: not in either buffer — load synchronously
      join_prefetch();
      load_into_buffer(active_, sc);
      start_prefetch(sc + 1);
    }

    unsigned local = index - sc_cumulative_[sc];
    assert(local < buf_[active_].insts.size());
    return buf_[active_].insts[local];
  }

  // Get the first instruction's PC.
  unsigned first_inst_pc() {
    if (w_idx_.inst_count == 0) return 0;
    return get(0).m_pc;
  }

 private:
  struct SubchunkBuffer {
    std::vector<inst_trace_t> insts;
    unsigned sc_id = UINT_MAX;
    ZSTD_DCtx *dctx = nullptr;
  };

  // Binary search: find which sub-chunk contains inst_index
  unsigned which_subchunk(unsigned inst_index) const {
    // sc_cumulative_[i] = first inst index of sub-chunk i
    // Find largest i such that sc_cumulative_[i] <= inst_index
    unsigned lo = 0, hi = (unsigned)subchunk_offsets_.size();
    while (lo + 1 < hi) {
      unsigned mid = (lo + hi) / 2;
      if (sc_cumulative_[mid] <= inst_index)
        lo = mid;
      else
        hi = mid;
    }
    return lo;
  }

  void load_into_buffer(unsigned buf_idx, unsigned sc) {
    assert(sc < subchunk_offsets_.size());
    SubchunkBuffer &buf = buf_[buf_idx];

    off_t pos = subchunk_offsets_[sc];

    // Read compressed size + data
    uint32_t comp_size;
    pread_exact(fd_, &comp_size, sizeof(comp_size), pos);
    pos += sizeof(comp_size);

    std::vector<char> comp_buf(comp_size);
    pread_exact(fd_, comp_buf.data(), comp_size, pos);

    // Decompress
    unsigned long long decomp_size =
        ZSTD_getFrameContentSize(comp_buf.data(), comp_size);
    if (decomp_size == ZSTD_CONTENTSIZE_ERROR ||
        decomp_size == ZSTD_CONTENTSIZE_UNKNOWN) {
      fprintf(stderr,
              "FATAL: zstd frame missing content size (corrupted .tracez?)\n");
      abort();
    }

    std::vector<char> decomp_buf(decomp_size + 1);
    size_t actual_size;
    if (ddict_) {
      actual_size = ZSTD_decompress_usingDDict(
          buf.dctx, decomp_buf.data(), decomp_buf.size(), comp_buf.data(),
          comp_size, ddict_);
    } else {
      actual_size =
          ZSTD_decompressDCtx(buf.dctx, decomp_buf.data(), decomp_buf.size(),
                              comp_buf.data(), comp_size);
    }
    if (ZSTD_isError(actual_size)) {
      fprintf(stderr, "FATAL: zstd decompression error: %s\n",
              ZSTD_getErrorName(actual_size));
      abort();
    }

    // Split into lines
    decomp_buf[actual_size] = '\0';
    std::vector<std::string> lines;
    char *p = decomp_buf.data();
    char *end = p + actual_size;
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

    // Parse each line into inst_trace_t
    buf.insts.clear();
    buf.insts.resize(lines.size());
    for (unsigned i = 0; i < lines.size(); ++i) {
      buf.insts[i].parse_from_string(lines[i], trace_version_, enable_lineinfo_,
                                     block_id_, cluster_cta_id_, cluster_id_,
                                     cluster_rank_);
    }
    buf.sc_id = sc;
  }

  void start_prefetch(unsigned sc) {
    if (sc >= subchunk_offsets_.size()) return;
    join_prefetch();
    unsigned target = 1 - active_;
    if (buf_[target].sc_id == sc) return;  // already loaded
    prefetch_pending_ = true;
    prefetch_thread_ = std::thread([this, target, sc] {
      try {
        load_into_buffer(target, sc);
      } catch (const std::exception &e) {
        fprintf(stderr, "FATAL: prefetch thread exception: %s\n", e.what());
        abort();
      }
    });
  }

  void join_prefetch() {
    if (prefetch_pending_ && prefetch_thread_.joinable()) {
      prefetch_thread_.join();
      prefetch_pending_ = false;
    }
  }

  int fd_ = -1;  // NOT owned — shared from TracezReader
  ZSTD_DDict *ddict_ = nullptr;
  warp_index_t w_idx_{};
  unsigned trace_version_ = 0;
  unsigned enable_lineinfo_ = 0;
  dim3 block_id_;
  dim3 cluster_cta_id_;
  dim3 cluster_id_;
  unsigned cluster_rank_ = 0;

  // Sub-chunk layout
  std::vector<uint64_t> subchunk_offsets_;  // byte offset per sub-chunk
  std::vector<uint32_t> sc_cumulative_;     // prefix-sum of inst counts

  // Double buffer
  SubchunkBuffer buf_[2];
  unsigned active_ = 0;
  std::thread prefetch_thread_;
  bool prefetch_pending_ = false;
};

#endif  // WARP_TRACE_STREAM_H
