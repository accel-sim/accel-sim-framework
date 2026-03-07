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
#include <vector>

#include <errno.h>
#include <signal.h>
#include <unistd.h>

#include <filesystem>

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

void group_per_block(const string &filepath);
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

public:
  explicit ThreadPool(size_t num_threads) {
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
          group_per_block(filepath);
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

  // Parse arguments
  if (argc == 1) {
    cerr << "Usage: " << argv[0] << " <path> [-j N]\n";
    cerr << "  path: kernelslist file or directory containing kernelslist "
            "files\n";
    cerr << "  -j N: limit to N parallel threads (default: 8)\n";
    return 1;
  }

  filepath = argv[1];
  for (int i = 2; i < argc; i++) {
    if (string(argv[i]) == "-j" && i + 1 < argc) {
      max_threads = atoi(argv[++i]);
      if (max_threads < 1)
        max_threads = 1;
    }
  }

  // Initialize thread pool
  ThreadPool pool(max_threads);
  cerr << "Using " << max_threads << " parallel threads\n";

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
        if (_l > 3 && line.substr(_l - 3, 3) == ".xz") {
          output_lines.push_back(line.substr(0, _l - 3) + "g.xz");
        } else {
          output_lines.push_back(line + "g");
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

// Thread-safe function using popen/pclose for I/O instead of stdin/stdout
// redirection. Each thread processes one kernel trace file independently.
void group_per_block(const string &filepath) {
  const string &filepath_str = filepath;
  WarpInstLUT warp_inst_lut;

  string trace_source_cmd;
  string trace_sink_cmd;
  string output_filepath;

  int _l = filepath_str.length();
  if (_l > 3 && filepath_str.substr(_l - 3, 3) == ".xz") {
    // kernel-1.trace.xz --(xz -dc)--> f --(xz -1 -T0)--> kernel-1.traceg.xz
    output_filepath = filepath_str.substr(0, _l - 3) + "g.xz";
    trace_source_cmd = "xz -dc " + filepath_str;
    trace_sink_cmd = "xz -1 -T0 > " + output_filepath;
  } else if (_l > 6 && filepath_str.substr(_l - 6, 6) == ".trace") {
    // kernel-2.trace --(cat)--> f --(cat)--> kernel-2.traceg
    output_filepath = filepath_str + "g";
    trace_source_cmd = "cat " + filepath_str;
    trace_sink_cmd = "cat > " + output_filepath;
  } else {
    lock_guard<mutex> lock(stderr_mutex);
    cerr << "Only support xz or raw text format. Unable to process - and "
            "skipping - trace file "
         << filepath_str << endl;
    return;
  }

  // Open pipes for reading input and writing output
  FILE *source_fp = popen(trace_source_cmd.c_str(), "r");
  if (!source_fp) {
    lock_guard<mutex> lock(stderr_mutex);
    cerr << "Failed to open source pipe for " << filepath_str << endl;
    return;
  }

  FILE *sink_fp = popen(trace_sink_cmd.c_str(), "w");
  if (!sink_fp) {
    lock_guard<mutex> lock(stderr_mutex);
    cerr << "Failed to open sink pipe for " << filepath_str << endl;
    pclose(source_fp);
    return;
  }

  {
    lock_guard<mutex> lock(stderr_mutex);
    cerr << "Processing file " << filepath_str << endl;
  }

  vector<threadblock_info> insts;
  unsigned grid_dim_x, grid_dim_y, grid_dim_z, tb_dim_x, tb_dim_y, tb_dim_z;
  unsigned tb_id_x, tb_id_y, tb_id_z, tb_id, warpid_tb;
  // Cluster information
  unsigned cluster_id_x, cluster_id_y, cluster_id_z;
  // CTA id within the cluster
  unsigned cluster_cta_id_x, cluster_cta_id_y, cluster_cta_id_z;
  // CTA rank within the cluster
  unsigned cluster_rank;
  unsigned lineinfo, linenum;
  string line;
  stringstream ss;
  string string1, string2;
  bool found_grid_dim = false, found_block_dim = false;

  // Add a flag for LDGSTS instruction to indicate which one to remove
  vector<vector<bool>> ldgsts_flags; // true to remove, false to not

  while (fgets_string(source_fp, line)) {
    if (line.length() == 0 || line[0] == '#') {
      fprintf(sink_fp, "%s\n", line.c_str());
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
      fprintf(sink_fp, "%s\n", line.c_str());
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
        cerr << "Warning: Warp " << j << " in thread block" << insts[i].tb_id_x
             << "," << insts[i].tb_id_y << "," << insts[i].tb_id_z
             << " is empty"
             << "\n";
      }
      for (auto it = insts[i].warp_insts_array[j].cbegin();
           it != insts[i].warp_insts_array[j].cend(); ++it) {
        // dereference once: const string*
        // dereference twice: const string
        fprintf(sink_fp, "%s\n", (*it)->c_str());
      }
    }
    fprintf(sink_fp, "\n#END_TB\n");
  }

  pclose(source_fp);
  pclose(sink_fp);
}

void group_per_core(const char *filepath) {

  // TO DO
}
