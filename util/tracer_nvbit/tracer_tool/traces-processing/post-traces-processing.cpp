#include <condition_variable>
#include <deque>
#include <fstream>
#include <iostream>
#include <math.h>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdio.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <errno.h>
#include <signal.h>
#include <unistd.h>

#include <filesystem>

#include <zdict.h>
#include <zstd.h>

#include "../tracez_format.h"

using namespace std;

struct threadblock_info {
  bool initialized;
  // CTA id within the grid
  unsigned tb_id_x, tb_id_y, tb_id_z;
  // Cluster information
  unsigned cluster_id_x, cluster_id_y, cluster_id_z;
  // CTA id within the cluster
  unsigned cluster_cta_id_x, cluster_cta_id_y, cluster_cta_id_z;
  // CTA rank within the cluster
  unsigned cluster_rank;
  vector<deque<const string *>> warp_insts_array;
  threadblock_info() {
    initialized = false;
    tb_id_x = tb_id_y = tb_id_z = 0;
    cluster_id_x = cluster_id_y = cluster_id_z = 0;
    cluster_cta_id_x = cluster_cta_id_y = cluster_cta_id_z = 0;
    cluster_rank = 0;
  }
};

/// @brief There exist significant repetition in the trace. The WarpInstLUT
/// registers recurrent trace fragments in a hash map. Strings (trace fragments)
/// are mapped to a pointer to a unique copy of that string, which is guaranteed
/// to live throughout the scope of the lifetime of this WarpInstLUT.
struct WarpInstLUT {
  // A mapping from "raw instruction string" to "a pointer to a global copy of
  // that string". For any element (x->y) of this map, *y==x holds.
  unordered_map<string, unique_ptr<string>> registration_table;

  /// @brief Is a string already registered?
  /// @param s The probing string.
  /// @return nullptr if the probing string does not exist in the look up table.
  /// Otherwise, a const pointer to a unique copy of that string.
  const string *lookup_entry(const string s) const {
    const auto it = registration_table.find(s);

    // not registered
    if (it == registration_table.end()) {
      return nullptr;
    } else {
      return it->second.get();
    }
  }

  /// @brief Add a string to the look up table.
  /// @param s The string to be added.
  /// @return A const pointer to the unique copy of the string.
  const string *register_new_entry(const string s) {
    // Check if the string is already in the LUT.
    const string *entry_ptr = lookup_entry(s);
    if (entry_ptr) {
      // just in case a rare hash collision happens, we panic
      if (s != *entry_ptr) {
        cerr << "FATAL: new string insertion " << s
             << "collides with the hash of a different string in the "
                "registration table "
             << *entry_ptr << "\n";
        abort();
      }
      return entry_ptr;
    }

    // Create a new string
    auto new_string_ptr = std::make_unique<string>(s);
    entry_ptr = new_string_ptr.get();
    registration_table.insert({s, std::move(new_string_ptr)});

    return entry_ptr;
  }
};

void group_per_block(const string &filepath, bool use_tracez);
void group_per_core(const char *filepath);

// Mutex for thread-safe stderr output
std::mutex stderr_mutex;

// Simple thread pool for processing kernel files
class ThreadPool {
  std::vector<std::thread> workers;
  std::deque<std::string> tasks;
  std::mutex queue_mutex;
  std::condition_variable cv;
  std::condition_variable done_cv;
  bool stop = false;
  size_t in_flight = 0; // tasks being processed + tasks in queue
  bool m_use_tracez = false;

public:
  explicit ThreadPool(size_t num_threads, bool use_tracez = false)
      : m_use_tracez(use_tracez) {
    for (size_t i = 0; i < num_threads; ++i) {
      workers.emplace_back([this] {
        while (true) {
          std::string filepath;
          {
            std::unique_lock<std::mutex> lock(queue_mutex);
            cv.wait(lock, [this] { return stop || !tasks.empty(); });
            if (stop && tasks.empty())
              return;
            filepath = std::move(tasks.front());
            tasks.pop_front();
          }
          group_per_block(filepath, m_use_tracez);
          {
            std::lock_guard<std::mutex> lock(queue_mutex);
            --in_flight;
          }
          done_cv.notify_all();
        }
      });
    }
  }

  void enqueue(const std::string &filepath) {
    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      tasks.push_back(filepath);
      ++in_flight;
    }
    cv.notify_one();
  }

  // Wait for all currently enqueued tasks to complete
  void wait_for_tasks() {
    std::unique_lock<std::mutex> lock(queue_mutex);
    done_cv.wait(lock, [this] { return in_flight == 0; });
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      stop = true;
    }
    cv.notify_all();
    for (auto &worker : workers) {
      worker.join();
    }
  }
};

std::vector<std::string> kernelslist_list;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  string kernellist_filepath;
  string filepath;
  int max_threads = 8; // default thread limit

  bool use_tracez = true;

  // Parse arguments
  if (argc == 1) {
    cerr << "Usage: " << argv[0] << " <path> [-j N] [--text]\n";
    cerr << "  path: kernelslist file or directory containing kernelslist "
            "files\n";
    cerr << "  -j N: limit to N parallel threads (default: 8)\n";
    cerr << "  --text: output .traceg (plain text) instead of .tracez\n";
    return 1;
  }

  filepath = argv[1];
  for (int i = 2; i < argc; i++) {
    if (string(argv[i]) == "-j" && i + 1 < argc) {
      max_threads = atoi(argv[++i]);
      if (max_threads < 1)
        max_threads = 1;
    } else if (string(argv[i]) == "--text") {
      use_tracez = false;
    }
  }

  // Initialize thread pool
  ThreadPool pool(max_threads, use_tracez);
  cerr << "Using " << max_threads << " parallel threads\n";
  if (use_tracez)
    cerr << "Output format: .tracez (zstd compressed)\n";
  else
    cerr << "Output format: .traceg (plain text)\n";

  ifstream ifs;
  ofstream ofs;

  // We can now pass a directory or a file as the input argument
  std::filesystem::path p(filepath);
  if (std::filesystem::is_directory(p)) {
    for (const auto &entry : std::filesystem::directory_iterator(p)) {
      std::string filename = entry.path().filename();
      // Skip output files (kernelslist.g) - only process input kernelslist
      // files
      if (filename.find("kernelslist") != std::string::npos &&
          filename.find(".g") == std::string::npos) {
        kernelslist_list.push_back(entry.path().string());
      }
    }
  } else if (std::filesystem::is_regular_file(p)) {
    kernelslist_list.push_back(p);
  } else {
    cerr << "Invalid file path\n";
    return 1;
  }

  for (auto kernellist_filepath : kernelslist_list) {
    string directory(kernellist_filepath);
    const size_t last_slash_idx = directory.rfind('/');
    if (std::string::npos != last_slash_idx) {
      directory = directory.substr(0, last_slash_idx);
    }

    ifs.open(kernellist_filepath.c_str());
    // If we have only one context, name it kernelslist.g by default
    if (kernelslist_list.size() == 1 ||
        kernelslist_list[0] == kernellist_filepath)
      ofs.open((string(directory) + "/kernelslist.g").c_str());
    else
      ofs.open((string(kernellist_filepath) + ".g").c_str());

    if (!ifs.is_open()) {
      cerr << "Unable to open file: " << kernellist_filepath << endl;
      return 1;
    }

    // First pass: collect all kernel files and output lines
    vector<string> kernel_filepaths;
    vector<string> output_lines;

    string line;
    string filepath;
    while (!ifs.eof()) {
      getline(ifs, line);
      if (line.empty())
        continue;
      else if (line.substr(0, 6) == "Memcpy") {
        output_lines.push_back(line);
      } else if (line.substr(0, 6) == "kernel") {
        filepath = directory + "/" + line;
        kernel_filepaths.push_back(filepath);

        int _l = line.length();
        if (use_tracez) {
          // .trace.xz or .trace -> .tracez
          if (_l > 3 && line.substr(_l - 3, 3) == ".xz") {
            output_lines.push_back(line.substr(0, _l - 3) + "z");
          } else {
            output_lines.push_back(line + "z");
          }
        } else {
          if (_l > 3 && line.substr(_l - 3, 3) == ".xz") {
            output_lines.push_back(line.substr(0, _l - 3) + "g.xz");
          } else {
            output_lines.push_back(line + "g");
          }
        }
      } else {
        cerr << "Undefined command: " << line << endl;
        continue;
      }
    }

    // Enqueue kernel files to thread pool
    for (const auto &kf : kernel_filepaths) {
      pool.enqueue(kf);
    }

    // Wait for all tasks to complete before writing output
    pool.wait_for_tasks();

    // Write output file after all processing is done
    for (const auto &out_line : output_lines) {
      ofs << out_line << endl;
    }

    ifs.close();
    ofs.close();
  }
  return 0;
}

// Helper to read a line from FILE* into a string
static bool fgets_string(FILE *fp, string &out) {
  out.clear();
  char buf[4096];
  while (fgets(buf, sizeof(buf), fp)) {
    out += buf;
    if (!out.empty() && out.back() == '\n') {
      out.pop_back(); // remove trailing newline
      return true;
    }
  }
  return !out.empty(); // return true if we got partial line at EOF
}

// Write helper for binary data
static void fwrite_u32(FILE *fp, uint32_t v) { fwrite(&v, sizeof(v), 1, fp); }
static void fwrite_u64(FILE *fp, uint64_t v) { fwrite(&v, sizeof(v), 1, fp); }

// Extract opcode from an instruction text line.
// Format: pc mask dest_count [dest_regs...] opcode ...
static string extract_opcode(const string &line) {
  stringstream ss(line);
  string token;
  // skip pc, mask
  ss >> token >> token;
  // read dest_count
  unsigned dest_count;
  ss >> dest_count;
  // skip dest regs
  for (unsigned i = 0; i < dest_count; i++)
    ss >> token;
  // opcode
  ss >> token;
  return token;
}

// Build replay-safe sub-chunk boundaries for a warp.
// Returns a vector of (start, count) pairs. Replay regions are never split.
static vector<pair<size_t, size_t>>
build_subchunk_ranges(const deque<const string *> &warp, uint32_t sc_target) {
  vector<pair<size_t, size_t>> ranges;
  size_t total = warp.size();
  if (total == 0)
    return ranges;
  if (sc_target == 0)
    sc_target = (uint32_t)total; // whole warp

  size_t pos = 0;
  while (pos < total) {
    size_t end = min(pos + sc_target, total);
    // Check if we're splitting inside a replay region.
    // Scan from pos to end for REPLAY_START without matching REPLAY_END.
    int replay_depth = 0;
    for (size_t k = pos; k < end; k++) {
      string op = extract_opcode(*warp[k]);
      if (op == "REPLAY_START")
        replay_depth++;
      else if (op == "REPLAY_END")
        replay_depth--;
    }
    // If replay_depth > 0, we have unclosed REPLAY_START — extend until closed
    while (replay_depth > 0 && end < total) {
      string op = extract_opcode(*warp[end]);
      if (op == "REPLAY_START")
        replay_depth++;
      else if (op == "REPLAY_END")
        replay_depth--;
      end++;
    }
    ranges.push_back({pos, end - pos});
    pos = end;
  }
  return ranges;
}

// Collect instruction lines for a warp, joining with newlines
static string join_warp_lines(const deque<const string *> &lines, size_t start,
                              size_t count) {
  string result;
  for (size_t i = start; i < start + count && i < lines.size(); ++i) {
    if (i > start)
      result += '\n';
    result += *lines[i];
  }
  return result;
}

// Write .tracez format output
static void write_tracez(FILE *out_fp, const vector<string> &header_lines,
                         const vector<threadblock_info> &insts) {
  // Get subchunk size from environment (default: 2048)
  uint32_t subchunk_insts = 2048;
  const char *env = getenv("TRACEZ_SUBCHUNK_INSTS");
  if (env)
    subchunk_insts = (uint32_t)atoi(env);

  // 1. Write kernel header as null-terminated text
  for (const auto &hline : header_lines) {
    fprintf(out_fp, "%s\n", hline.c_str());
  }
  fputc('\0', out_fp);

  // 2. Collect samples for dictionary training
  vector<string> samples;
  size_t total_sample_bytes = 0;
  const size_t MAX_SAMPLE_BYTES = 1024 * 1024; // 1MB of samples
  for (unsigned i = 0;
       i < insts.size() && total_sample_bytes < MAX_SAMPLE_BYTES; ++i) {
    if (!insts[i].initialized)
      continue;
    for (unsigned j = 0; j < insts[i].warp_insts_array.size(); ++j) {
      const auto &warp = insts[i].warp_insts_array[j];
      // Sample every 8th instruction
      for (size_t k = 0;
           k < warp.size() && total_sample_bytes < MAX_SAMPLE_BYTES; k += 8) {
        samples.push_back(*warp[k]);
        total_sample_bytes += warp[k]->size();
      }
    }
  }

  // Train dictionary
  vector<char> dict_buf(64 * 1024); // 64KB max dictionary
  vector<size_t> sample_sizes;
  string all_samples;
  for (const auto &s : samples) {
    sample_sizes.push_back(s.size());
    all_samples += s;
  }

  size_t dict_size = 0;
  if (!samples.empty()) {
    dict_size = ZDICT_trainFromBuffer(dict_buf.data(), dict_buf.size(),
                                      all_samples.data(), sample_sizes.data(),
                                      (unsigned)sample_sizes.size());
    if (ZDICT_isError(dict_size)) {
      {
        lock_guard<mutex> lock(stderr_mutex);
        cerr << "Warning: Dictionary training failed: "
             << ZDICT_getErrorName(dict_size) << ", proceeding without dict\n";
      }
      dict_size = 0;
    }
  }

  // 3. Write compression header
  fwrite_u32(out_fp, (uint32_t)dict_size);
  fwrite_u32(out_fp, subchunk_insts);

  // 4. Write dictionary
  if (dict_size > 0) {
    fwrite(dict_buf.data(), 1, dict_size, out_fp);
  }

  // Create compression dict and context
  ZSTD_CDict *cdict = nullptr;
  if (dict_size > 0) {
    cdict = ZSTD_createCDict(dict_buf.data(), dict_size, 3);
  }
  ZSTD_CCtx *cctx = ZSTD_createCCtx();

  // 5. Write per-TB per-warp compressed data, building index
  vector<tracez::tb_index_t> index;

  for (unsigned i = 0; i < insts.size(); ++i) {
    if (!insts[i].initialized || insts[i].warp_insts_array.empty())
      continue;

    tracez::tb_index_t tb_idx;
    tb_idx.tb_id_x = insts[i].tb_id_x;
    tb_idx.tb_id_y = insts[i].tb_id_y;
    tb_idx.tb_id_z = insts[i].tb_id_z;
    tb_idx.cluster_id_x = insts[i].cluster_id_x;
    tb_idx.cluster_id_y = insts[i].cluster_id_y;
    tb_idx.cluster_id_z = insts[i].cluster_id_z;
    tb_idx.cluster_cta_id_x = insts[i].cluster_cta_id_x;
    tb_idx.cluster_cta_id_y = insts[i].cluster_cta_id_y;
    tb_idx.cluster_cta_id_z = insts[i].cluster_cta_id_z;
    tb_idx.cluster_rank = insts[i].cluster_rank;
    tb_idx.warp_count = (uint32_t)insts[i].warp_insts_array.size();

    for (unsigned j = 0; j < insts[i].warp_insts_array.size(); ++j) {
      const auto &warp = insts[i].warp_insts_array[j];
      uint32_t inst_count = (uint32_t)warp.size();

      // Build replay-safe sub-chunk ranges
      auto ranges = build_subchunk_ranges(warp, subchunk_insts);
      uint32_t num_subchunks = (uint32_t)ranges.size();

      tracez::warp_index_t w_idx;
      w_idx.inst_count = inst_count;
      w_idx.num_subchunks = num_subchunks;
      w_idx.file_offset = (uint64_t)ftello(out_fp);

      // Write warp header
      fwrite_u32(out_fp, inst_count);
      fwrite_u32(out_fp, num_subchunks);

      // Write sub-chunks
      for (uint32_t sc = 0; sc < num_subchunks; ++sc) {
        size_t start = ranges[sc].first;
        size_t count = ranges[sc].second;
        w_idx.subchunk_inst_counts.push_back((uint32_t)count);

        string text = join_warp_lines(warp, start, count);

        size_t bound = ZSTD_compressBound(text.size());
        vector<char> comp_buf(bound);
        size_t comp_size;
        if (cdict) {
          comp_size = ZSTD_compress_usingCDict(cctx, comp_buf.data(), bound,
                                               text.data(), text.size(), cdict);
        } else {
          comp_size = ZSTD_compressCCtx(cctx, comp_buf.data(), bound,
                                        text.data(), text.size(), 3);
        }
        if (ZSTD_isError(comp_size)) {
          lock_guard<mutex> lock(stderr_mutex);
          cerr << "FATAL: zstd compression error: "
               << ZSTD_getErrorName(comp_size) << "\n";
          abort();
        }

        fwrite_u32(out_fp, (uint32_t)comp_size);
        fwrite(comp_buf.data(), 1, comp_size, out_fp);
      }

      tb_idx.warps.push_back(w_idx);
    }
    index.push_back(tb_idx);
  }

  // 6. Write index
  uint64_t index_offset = (uint64_t)ftello(out_fp);
  tracez::write_index(out_fp, index, index_offset);

  ZSTD_freeCCtx(cctx);
  if (cdict)
    ZSTD_freeCDict(cdict);
}

// Classify an instruction line for the unused-mbarrier scan.
// Input is the post-cluster portion of a trace line: [linenum] pc mask <dst...>
//   <opcode> <src...> <mem_width> [<addr_data>] ...
// Returns the mbarrier address (lower 32 bits) for ARRIVE-on / TRYWAIT
// instructions, or 0 if the inst is unrelated or cannot be parsed.
struct mbar_inst_info {
  enum kind_t { OTHER, ARRIVE, TRYWAIT };
  kind_t kind;
  uint32_t mbar_addr;
};
static mbar_inst_info classify_mbar_inst(const string &line,
                                         unsigned lineinfo) {
  mbar_inst_info r{mbar_inst_info::OTHER, 0};
  stringstream ss(line);
  string tok;
  if (lineinfo) {
    unsigned linenum;
    ss >> dec >> linenum;
  }
  unsigned pc, mask, n_dst;
  ss >> hex >> pc >> mask;
  ss >> dec >> n_dst;
  for (unsigned i = 0; i < n_dst; i++)
    ss >> tok;
  string opcode;
  ss >> opcode;

  bool is_arrive_syncs = (opcode.rfind("SYNCS.ARRIVE", 0) == 0);
  bool is_arrive_ldgsts = (opcode.rfind("ARRIVES.LDGSTSBAR", 0) == 0);
  bool is_trywait = (opcode.find("PHASECHK") != string::npos &&
                     opcode.find("TRYWAIT") != string::npos);
  bool is_tma_load = (opcode.rfind("UTMALDG", 0) == 0);
  if (!(is_arrive_syncs || is_arrive_ldgsts || is_trywait || is_tma_load))
    return r;

  unsigned n_src;
  ss >> dec >> n_src;
  for (unsigned i = 0; i < n_src; i++)
    ss >> tok;
  unsigned mem_width;
  ss >> dec >> mem_width;
  if (mem_width == 0)
    return r;

  uint64_t addr = 0;
  if (is_tma_load) {
    // TMA format: tma_mbar_addr is the next field after mem_width.
    ss >> hex >> addr;
  } else {
    unsigned addr_mode;
    ss >> dec >> addr_mode;
    // address_format::list_all=0 -> first listed addr is for the first
    // active lane. base_stride=1 / base_delta=2 -> base_address is first.
    if (addr_mode == 0 || addr_mode == 1 || addr_mode == 2) {
      ss >> hex >> addr;
    } else {
      return r;
    }
  }
  if (!ss)
    return r;

  r.mbar_addr = (uint32_t)addr;
  r.kind = is_trywait ? mbar_inst_info::TRYWAIT : mbar_inst_info::ARRIVE;
  return r;
}

// Drop SYNCS.PHASECHK.*.TRYWAIT instructions targeting mbarriers that no
// instruction in this kernel ever ARRIVE-s on (SYNCS.ARRIVE*,
// ARRIVES.LDGSTSBAR, or TMA load via tma_mbar_addr). Such trywaits would
// deadlock the simulator since the mbarrier phase never advances.
static void drop_unused_trywaits(const string &filepath_str,
                                 vector<threadblock_info> &insts,
                                 unsigned lineinfo) {
  unordered_set<uint32_t> arrived;
  for (const auto &tb : insts) {
    for (const auto &warp : tb.warp_insts_array) {
      for (const string *inst_ptr : warp) {
        auto c = classify_mbar_inst(*inst_ptr, lineinfo);
        if (c.kind == mbar_inst_info::ARRIVE && c.mbar_addr != 0)
          arrived.insert(c.mbar_addr);
      }
    }
  }

  unordered_map<uint32_t, size_t> dropped;
  for (auto &tb : insts) {
    for (auto &warp : tb.warp_insts_array) {
      deque<const string *> filtered;
      for (const string *inst_ptr : warp) {
        auto c = classify_mbar_inst(*inst_ptr, lineinfo);
        if (c.kind == mbar_inst_info::TRYWAIT && c.mbar_addr != 0 &&
            arrived.count(c.mbar_addr) == 0) {
          dropped[c.mbar_addr]++;
          continue;
        }
        filtered.push_back(inst_ptr);
      }
      warp = std::move(filtered);
    }
  }

  if (!dropped.empty()) {
    lock_guard<mutex> lock(stderr_mutex);
    for (auto &kv : dropped) {
      cout << "[" << filepath_str << "] Dropped " << kv.second
           << " TRYWAIT(s) on mbarrier 0x" << hex << kv.first << dec
           << " (no ARRIVE in this kernel)\n";
    }
  }
}

// Thread-safe function using popen/pclose for I/O instead of stdin/stdout
// redirection. Each thread processes one kernel trace file independently.
void group_per_block(const string &filepath, bool use_tracez) {
  const string &filepath_str = filepath;
  WarpInstLUT warp_inst_lut;

  string trace_source_cmd;
  string output_filepath;

  int _l = filepath_str.length();
  if (_l > 3 && filepath_str.substr(_l - 3, 3) == ".xz") {
    trace_source_cmd = "xz -dc " + filepath_str;
    if (use_tracez) {
      output_filepath = filepath_str.substr(0, _l - 3) + "z";
    } else {
      output_filepath = filepath_str.substr(0, _l - 3) + "g.xz";
    }
  } else if (_l > 6 && filepath_str.substr(_l - 6, 6) == ".trace") {
    trace_source_cmd = "cat " + filepath_str;
    if (use_tracez) {
      output_filepath = filepath_str + "z";
    } else {
      output_filepath = filepath_str + "g";
    }
  } else {
    lock_guard<mutex> lock(stderr_mutex);
    cerr << "Only support xz or raw text format. Unable to process - and "
            "skipping - trace file "
         << filepath_str << endl;
    return;
  }

  // Open pipe for reading input
  FILE *source_fp = popen(trace_source_cmd.c_str(), "r");
  if (!source_fp) {
    lock_guard<mutex> lock(stderr_mutex);
    cerr << "Failed to open source pipe for " << filepath_str << endl;
    return;
  }

  {
    lock_guard<mutex> lock(stderr_mutex);
    cerr << "Processing file " << filepath_str << endl;
  }

  vector<threadblock_info> insts;
  vector<string> header_lines; // collect header lines for tracez
  unsigned grid_dim_x, grid_dim_y, grid_dim_z, tb_dim_x, tb_dim_y, tb_dim_z;
  unsigned tb_id_x, tb_id_y, tb_id_z, tb_id, warpid_tb;
  // Cluster information
  unsigned cluster_id_x, cluster_id_y, cluster_id_z;
  // CTA id within the cluster
  unsigned cluster_cta_id_x, cluster_cta_id_y, cluster_cta_id_z;
  // CTA rank within the cluster
  unsigned cluster_rank;
  unsigned lineinfo = 0, linenum;
  string line;
  stringstream ss;
  string string1, string2;
  bool found_grid_dim = false, found_block_dim = false;

  // Add a flag for LDGSTS instruction to indicate which one to remove
  vector<vector<bool>> ldgsts_flags; // true to remove, false to not

  while (fgets_string(source_fp, line)) {
    if (line.length() == 0 || line[0] == '#') {
      header_lines.push_back(line);
      continue;
    }

    else if (line[0] == '-') {
      ss.clear();
      ss.str(line);
      ss.ignore();
      ss >> string1 >> string2;
      if (string1 == "grid" && string2 == "dim") {
        sscanf(line.c_str(), "-grid dim = (%d,%d,%d)", &grid_dim_x, &grid_dim_y,
               &grid_dim_z);
        found_grid_dim = true;
      } else if (string1 == "block" && string2 == "dim") {
        sscanf(line.c_str(), "-block dim = (%d,%d,%d)", &tb_dim_x, &tb_dim_y,
               &tb_dim_z);
        found_block_dim = true;
      } else if (string1 == "enable" && string2 == "lineinfo") {
        sscanf(line.c_str(), "-enable lineinfo = %d", &lineinfo);
      }

      if (found_grid_dim && found_block_dim) {
        insts.resize(grid_dim_x * grid_dim_y * grid_dim_z);

        // Size the ldgsts_flags vector
        ldgsts_flags.resize(grid_dim_x * grid_dim_y * grid_dim_z);

        for (unsigned i = 0; i < insts.size(); ++i) {
          insts[i].warp_insts_array.resize(
              ceil(float(tb_dim_x * tb_dim_y * tb_dim_z) / 32));

          // Size the ldgsts_flags vector
          ldgsts_flags[i].resize(
              ceil(float(tb_dim_x * tb_dim_y * tb_dim_z) / 32));
          for (unsigned j = 0; j < ldgsts_flags[i].size(); j++) {
            ldgsts_flags[i][j] = true;
          }
        }
      }
      header_lines.push_back(line);
      continue;
    } else {
      ss.clear();
      ss.str(line);
      ss >> tb_id_x >> tb_id_y >> tb_id_z >> warpid_tb >> cluster_id_x >>
          cluster_id_y >> cluster_id_z >> cluster_cta_id_x >>
          cluster_cta_id_y >> cluster_cta_id_z >> cluster_rank;
      tb_id =
          tb_id_z * grid_dim_y * grid_dim_x + tb_id_y * grid_dim_x + tb_id_x;
      if (!insts[tb_id].initialized) {
        insts[tb_id].tb_id_x = tb_id_x;
        insts[tb_id].tb_id_y = tb_id_y;
        insts[tb_id].tb_id_z = tb_id_z;
        insts[tb_id].cluster_id_x = cluster_id_x;
        insts[tb_id].cluster_id_y = cluster_id_y;
        insts[tb_id].cluster_id_z = cluster_id_z;
        insts[tb_id].cluster_cta_id_x = cluster_cta_id_x;
        insts[tb_id].cluster_cta_id_y = cluster_cta_id_y;
        insts[tb_id].cluster_cta_id_z = cluster_cta_id_z;
        insts[tb_id].cluster_rank = cluster_rank;
        insts[tb_id].initialized = true;
      }
      string rest_of_line(ss.str().substr(static_cast<size_t>(ss.tellg()) + 1));

      // Ni: ignore the shmem LDGSTS instruction
      stringstream opcode_ss;
      string opcode, temp;
      unsigned dest_num;
      opcode_ss << rest_of_line;
      for (int i = 0; i < 2; i++) {
        opcode_ss >> temp;
      }
      opcode_ss >> dest_num;
      for (unsigned i = 0; i < dest_num; i++) {
        opcode_ss >> temp;
      }
      opcode_ss >> opcode;

      // Look up the warp inst table to see if this instruction has been
      // registered. If yes, we just copy the pointer to that string.
      const string *inst_ptr = warp_inst_lut.lookup_entry(rest_of_line);
      if (!inst_ptr)
        inst_ptr = warp_inst_lut.register_new_entry(rest_of_line);

      // One actual LDGSTS instruction includes 2 LDGSTS instructions in the
      // trace, because it has two memory references. This is trying to remove
      // the one with the shared memory address.

      // Check if opcode starts with "LDGSTS" (not just contains it, to avoid
      // matching ARRIVES.LDGSTSBAR)
      if (opcode.rfind("LDGSTS", 0) == 0) {
        if (!ldgsts_flags[tb_id][warpid_tb]) {
          insts[tb_id].warp_insts_array[warpid_tb].push_back(inst_ptr);
        }
        ldgsts_flags[tb_id][warpid_tb] = !ldgsts_flags[tb_id][warpid_tb];
      } else {
        insts[tb_id].warp_insts_array[warpid_tb].push_back(inst_ptr);
      }
    }
  }

  pclose(source_fp);

  drop_unused_trywaits(filepath_str, insts, lineinfo);

  if (use_tracez) {
    // Write .tracez format
    FILE *out_fp = fopen(output_filepath.c_str(), "wb");
    if (!out_fp) {
      lock_guard<mutex> lock(stderr_mutex);
      cerr << "Failed to open output file " << output_filepath << endl;
      return;
    }
    write_tracez(out_fp, header_lines, insts);
    fclose(out_fp);
  } else {
    // Write .traceg format (original text path)
    string trace_sink_cmd;
    if (output_filepath.size() > 3 &&
        output_filepath.substr(output_filepath.size() - 3) == ".xz") {
      trace_sink_cmd = "xz -1 -T0 > " + output_filepath;
    } else {
      trace_sink_cmd = "cat > " + output_filepath;
    }
    FILE *sink_fp = popen(trace_sink_cmd.c_str(), "w");
    if (!sink_fp) {
      lock_guard<mutex> lock(stderr_mutex);
      cerr << "Failed to open sink pipe for " << output_filepath << endl;
      return;
    }

    // Write header
    for (const auto &hline : header_lines) {
      fprintf(sink_fp, "%s\n", hline.c_str());
    }

    for (unsigned i = 0; i < insts.size(); ++i) {
      if (insts[i].initialized && insts[i].warp_insts_array.size() > 0) {
        fprintf(sink_fp, "\n#BEGIN_TB\n");
        fprintf(sink_fp, "\nthread block = %u,%u,%u\n", insts[i].tb_id_x,
                insts[i].tb_id_y, insts[i].tb_id_z);
        fprintf(sink_fp, "cluster id = %u,%u,%u\n", insts[i].cluster_id_x,
                insts[i].cluster_id_y, insts[i].cluster_id_z);
        fprintf(sink_fp, "cluster cta = %u,%u,%u\n", insts[i].cluster_cta_id_x,
                insts[i].cluster_cta_id_y, insts[i].cluster_cta_id_z);
        fprintf(sink_fp, "cluster rank = %u\n", insts[i].cluster_rank);
      } else {
        lock_guard<mutex> lock(stderr_mutex);
        cerr << "Warning: Thread block " << insts[i].tb_id_x << ","
             << insts[i].tb_id_y << "," << insts[i].tb_id_z << " is empty"
             << "\n";
        continue;
      }
      for (unsigned j = 0; j < insts[i].warp_insts_array.size(); ++j) {
        fprintf(sink_fp, "\nwarp = %u\n", j);
        fprintf(sink_fp, "insts = %zu\n", insts[i].warp_insts_array[j].size());
        if (insts[i].warp_insts_array[j].size() == 0) {
          lock_guard<mutex> lock(stderr_mutex);
          cerr << "Warning: Warp " << j << " in thread block"
               << insts[i].tb_id_x << "," << insts[i].tb_id_y << ","
               << insts[i].tb_id_z << " is empty"
               << "\n";
        }
        for (auto it = insts[i].warp_insts_array[j].cbegin();
             it != insts[i].warp_insts_array[j].cend(); ++it) {
          fprintf(sink_fp, "%s\n", (*it)->c_str());
        }
      }
      fprintf(sink_fp, "\n#END_TB\n");
    }
    pclose(sink_fp);
  }
}

void group_per_core(const char *filepath) {

  // TO DO
}
