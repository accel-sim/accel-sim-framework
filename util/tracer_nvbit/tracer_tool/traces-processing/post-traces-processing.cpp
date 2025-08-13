#include <deque>
#include <fstream>
#include <iostream>
#include <math.h>
#include <memory>
#include <sstream>
#include <stdio.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "../common.h"
#include <errno.h>
#include <signal.h>
#include <unistd.h>

#include <filesystem>

using namespace std;

struct threadblock_info {
  vector<deque<inst_trace_t>> warp_insts_array;
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

void group_per_block(const char *filepath);
void group_per_core(const char *filepath);

// This program works by redirecting the stdin/stdout to child processes. The
// stdin is piped to a process that reads from disk the input trace file. The
// stdout is piped to a process that writes to disk the post-process trace. We
// should preserve the original file descriptors for stdin/stdout before doing
// redirections.
int preserved_stdin_fileno;
int preserved_stdout_fileno;

std::vector<std::string> kernelslist_list;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  string kernellist_filepath;
  string filepath;
  bool is_per_core;
  if (argc == 1) {
    cerr << "File path is missing\n";
    return 1;
  } else if (argc == 2) {
    filepath = argv[1];
    is_per_core = true;

  } else if (argc == 3) {
    filepath = argv[1];
    is_per_core = bool(argv[2]);
  } else {
    cerr << "Too Many Arguemnts!\n";
    return 1;
  }

  ifstream ifs;
  ofstream ofs;

  // We can now pass a directory or a file as the input argument
  std::filesystem::path p(filepath);
  if (std::filesystem::is_directory(p)) {
    for (const auto &entry : std::filesystem::directory_iterator(p)) {
      std::string filename = entry.path().filename();
      if (filename.find("kernelslist") != std::string::npos) {
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

    string line;
    string filepath;
    while (!ifs.eof()) {
      getline(ifs, line);
      if (line.empty())
        continue;
      else if (line.substr(0, 6) == "Memcpy") {
        ofs << line << endl;
      } else if (line.substr(0, 6) == "kernel") {
        filepath = directory + "/" + line;
        group_per_block(filepath.c_str());

        int _l = line.length();
        if (_l > 3 && line.substr(_l - 3, 3) == ".xz") {
          ofs << line.substr(0, _l - 3) << "g.xz" << endl;
        } else {
          ofs << line + "g" << endl;
        }
      } else {
        cerr << "Undefined command: " << line << endl;
        return 1;
      }
    }

    ifs.close();
    ofs.close();
  }
  return 0;
}

// This function redirects stdin and stdout for trace processing.
// For error/warning/info message to print to the terminal, always use the
// stderr stream. The io redirection will be restored by the time the function
// returns.
void group_per_block(const char *filepath) {
  string output_filepath;
  // Open the pipe
  FILE *pipe;
  FILE *kernel_out;

  try {
    // Use utility function to open input file
    pipe = openFileForReading(filepath);

    // Generate output filepath and open output file
    if (hasEnding(filepath, ".xz")) {
      output_filepath = generateOutputFilepath(filepath, "g");
      kernel_out =
          openFileForWriting(output_filepath, true); // Use xz compression
    } else if (hasEnding(filepath, ".trace")) {
      output_filepath = string(filepath) + "g";
      kernel_out = openFileForWriting(output_filepath, false); // No compression
    } else {
      throw std::runtime_error("Unsupported file type!");
    }
  } catch (const std::runtime_error &e) {
    throw std::runtime_error("Failed to open files: " + string(e.what()));
  }

  cerr << "Processing file " << filepath << endl;

  vector<threadblock_info> insts;

  // Read the kernel header
  std::string kernel_name;
  uint64_t name_size;
  fread(&name_size, sizeof(uint64_t), 1, pipe);
  kernel_name.resize(name_size);
  fread(kernel_name.data(), name_size, 1, pipe);

  // Read the kernel header
  kernel_header header;
  fread(&header, sizeof(kernel_header), 1, pipe);

  insts.resize(header.grid_dim_x * header.grid_dim_y * header.grid_dim_z);
  vector<vector<bool>> ldgsts_flags;

  for (unsigned tb = 0; tb < insts.size(); ++tb) {
    insts[tb].warp_insts_array.resize(ceil(
        float(header.block_dim_x * header.block_dim_y * header.block_dim_z) /
        32));

    ldgsts_flags.resize(insts[tb].warp_insts_array.size());
    for (unsigned j = 0; j < ldgsts_flags[tb].size(); j++) {
      ldgsts_flags[tb][j] = true;
    }
  }

  unsigned size;
  while (fread(&size, sizeof(unsigned), 1, pipe)) {
    inst_trace_t inst = {0};
    fread(&inst, size, 1, pipe);

    unsigned tb_id_x = inst.cta_id_x;
    unsigned tb_id_y = inst.cta_id_y;
    unsigned tb_id_z = inst.cta_id_z;
    unsigned tb_id = tb_id_z * header.grid_dim_y * header.grid_dim_x +
                     tb_id_y * header.grid_dim_x + tb_id_x;
    unsigned warp_id = inst.warpid_tb;

    std::string opcode = inst.opcode;
    if (opcode.find("LDGSTS") != string::npos) {
      if (!ldgsts_flags[tb_id][warp_id]) {
        insts[tb_id].warp_insts_array[warp_id].push_back(inst);
      }
      ldgsts_flags[tb_id][warp_id] = !ldgsts_flags[tb_id][warp_id];
    } else {
      insts[tb_id].warp_insts_array[warp_id].push_back(inst);
    }
  }

  fwrite(&name_size, sizeof(uint64_t), 1, kernel_out);
  fwrite(kernel_name.c_str(), kernel_name.size(), 1, kernel_out);

  fwrite(&header, sizeof(kernel_header), 1, kernel_out);

  for (unsigned tb_id = 0; tb_id < insts.size(); ++tb_id) {
    if (insts[tb_id].warp_insts_array.size() > 0) {
      // print total warp count in this thread block
      unsigned total_warp_count = insts[tb_id].warp_insts_array.size();
      fwrite(&total_warp_count, sizeof(unsigned), 1, kernel_out);

      for (unsigned warp_id = 0; warp_id < insts[tb_id].warp_insts_array.size();
           ++warp_id) {
        // print total inst count in this warp
        unsigned total_inst_count =
            insts[tb_id].warp_insts_array[warp_id].size();
        fwrite(&total_inst_count, sizeof(unsigned), 1, kernel_out);

        for (unsigned inst_id = 0;
             inst_id < insts[tb_id].warp_insts_array[warp_id].size();
             ++inst_id) {
          inst_trace_t &inst = insts[tb_id].warp_insts_array[warp_id][inst_id];

          // Write the inst_trace_t structure as binary data to the file
          unsigned size = sizeof(inst_trace_t);
          if (!inst.is_mem) {
            // write only the part without addrs
            size = offsetof(inst_trace_t, addrs);
          }
          fwrite(&size, sizeof(unsigned), 1, kernel_out);
          fwrite(&inst, size, 1, kernel_out);
        }
      }
    }
  }
  fclose(kernel_out);

  /*
  // legacy code starts here. Pending to be removed.

  // Important... without clear(), cin.eof() may evaluate to true on the
  // second kernel
  cin.clear();
  clearerr(stdin);
  while (!cin.eof()) {
    getline(cin, line);

    if (line.length() == 0 || line[0] == '#') {
      cout << line << endl;
      continue;
    }

    else if (line[0] == '-') {
      ss.str(line);
      ss.ignore();
      ss >> string1 >> string2;
      if (string1 == "grid" && string2 == "dim") {
        sscanf(line.c_str(), "-grid dim = (%d,%d,%d)", &grid_dim_x,
  &grid_dim_y, &grid_dim_z); found_grid_dim = true; } else if (string1 ==
  "block" && string2 == "dim") { sscanf(line.c_str(), "-block dim =
  (%d,%d,%d)", &tb_dim_x, &tb_dim_y, &tb_dim_z); found_block_dim = true; }
  else if (string1 == "enable" && string2 == "lineinfo") {
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
      cout << line << endl;
      continue;
    } else {

      ss.str(line);
      ss >> tb_id_x >> tb_id_y >> tb_id_z >> warpid_tb;
      tb_id =
          tb_id_z * grid_dim_y * grid_dim_x + tb_id_y * grid_dim_x +
  tb_id_x; if (!insts[tb_id].initialized) { insts[tb_id].tb_id_x = tb_id_x;
        insts[tb_id].tb_id_y = tb_id_y;
        insts[tb_id].tb_id_z = tb_id_z;
        insts[tb_id].initialized = true;
      }
      // ss.ignore(); //remove the space
      // rest_of_line.clear();
      // getline(ss, rest_of_line); //get rest of the string!
      string rest_of_line(ss.str().substr(ss.tellg() + 1));

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
      // trace, because it has two memory references. This is trying to
  remove
      // the one with the shared memory address.

      if (opcode.find("LDGSTS") != string::npos) {
        if (!ldgsts_flags[tb_id][warpid_tb]) {
          insts[tb_id].warp_insts_array[warpid_tb].push_back(inst_ptr);
        }
        ldgsts_flags[tb_id][warpid_tb] = !ldgsts_flags[tb_id][warpid_tb];
      } else {
        insts[tb_id].warp_insts_array[warpid_tb].push_back(inst_ptr);
      }
    }
  }

  for (unsigned tb_id = 0; tb_id < insts.size(); ++tb_id) {
    // ofs<<string<<"\n";
    if (insts[tb_id].initialized && insts[tb_id].warp_insts_array.size() >
  0) { cout << "\n"
           << "#BEGIN_TB"
           << "\n";
      cout << "\n"
           << "thread block = " << insts[tb_id].tb_id_x << ","
           << insts[tb_id].tb_id_y << "," << insts[tb_id].tb_id_z << "\n";
    } else {
      cerr << "Warning: Thread block " << insts[tb_id].tb_id_x << ","
           << insts[tb_id].tb_id_y << "," << insts[tb_id].tb_id_z << " is
  empty"
           << "\n";
      continue;
    }
    for (unsigned warp_id = 0; warp_id <
  insts[tb_id].warp_insts_array.size();
         ++warp_id) {
      cout << "\n"
           << "warp = " << warp_id << "\n";
      cout << "insts = " << insts[tb_id].warp_insts_array[warp_id].size()
           << "\n";
      if (insts[tb_id].warp_insts_array[warp_id].size() == 0) {
        cerr << "Warning: Warp " << warp_id << " in thread block"
             << insts[tb_id].tb_id_x << "," << insts[tb_id].tb_id_y << ","
             << insts[tb_id].tb_id_z << " is empty"
             << "\n";
      }
      for (auto it = insts[tb_id].warp_insts_array[warp_id].cbegin();
           it != insts[tb_id].warp_insts_array[warp_id].cend(); ++it) {
        // dereference once: const string*
        // dereference twice: const string
        cout << **it << "\n";
      }
    }
    cout << endl << "#END_TB" << endl;
  }
  */
}

void group_per_core(const char *filepath) {

  // TO DO
}
