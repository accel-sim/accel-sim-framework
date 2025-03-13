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

#include <errno.h>
#include <signal.h>
#include <unistd.h>

#include <filesystem>

using namespace std;

struct threadblock_info {
  bool initialized;
  unsigned tb_id_x, tb_id_y, tb_id_z;
  vector<deque<const string *>> warp_insts_array;
  threadblock_info() {
    initialized = false;
    tb_id_x = tb_id_y = tb_id_z = 0;
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
  if (std::filesystem::is_directory(p))
  {
    for (const auto & entry : std::filesystem::directory_iterator(p))
    {
      std::string filename = entry.path().filename();
      if (filename.find("kernelslist") != std::string::npos)
      {
        kernelslist_list.push_back(entry.path().string());
      }
    }
  }
  else if (std::filesystem::is_regular_file(p))
  {
    kernelslist_list.push_back(kernellist_filepath);
  }
  else
  {
    cerr << "Invalid file path\n";
    return 1;
  }

  for (auto kernellist_filepath : kernelslist_list){
    string directory(kernellist_filepath);
    const size_t last_slash_idx = directory.rfind('/');
    if (std::string::npos != last_slash_idx) {
      directory = directory.substr(0, last_slash_idx);
    }

    ifs.open(kernellist_filepath.c_str());
    // If we have only one context, name it kernelslist.g by default
    if(kernelslist_list.size() == 1 || kernelslist_list[0] == kernellist_filepath)
      ofs.open((string(directory)+"/kernelslist.g").c_str());
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
  preserved_stdin_fileno = dup(STDIN_FILENO);
  preserved_stdout_fileno = dup(STDOUT_FILENO);

  string filepath_str{filepath};
  WarpInstLUT warp_inst_lut;

  pid_t sink_process_pid = 0;
  string trace_sink_cmd;
  int sink_pipe_fd[2];

  pid_t source_process_pid = 0;
  string trace_source_cmd;
  int source_pipe_fd[2];
  string output_filepath;

  bool input_file_is_xz = false;
  int _l = filepath_str.length();
  if (_l > 3 && filepath_str.substr(_l - 3, 3) == ".xz") {
    // kernel-1.trace.xz --(xz -dc)--> f --(xz -1 -T0)--> kernel-1.traceg.xz
    input_file_is_xz = true;
    output_filepath = filepath_str.substr(0, _l - 3) + "g.xz";
    trace_source_cmd = "xz -dc " + filepath_str;
    trace_sink_cmd = "xz -1 -T0 > " + output_filepath;
  } else if (_l > 6 && filepath_str.substr(_l - 6, 6) == ".trace") {
    // kernel-2.trace --(cat)--> f --(cat)--> kernel-2.traceg
    input_file_is_xz = false;
    output_filepath = filepath_str + "g";
    trace_source_cmd = "cat " + filepath_str;
    trace_sink_cmd = "cat > " + output_filepath;
  } else {
    cerr << "Only support xz or raw text format. Unable to process - and "
            "skipping - trace file "
         << filepath_str << endl;
    close(preserved_stdin_fileno);
    close(preserved_stdout_fileno);
    return;
  }

  // cerr << "source cmd is "<<trace_source_cmd<<"\n";
  // cerr << "sink cmd is "<<trace_sink_cmd<<"\n";

  // fork a child process as the trace source
  if (pipe(source_pipe_fd) != 0) {
    cerr << "Failed to create pipe\n";
    perror("pipe");
    exit(1);
  }
  source_process_pid = fork();
  if (source_process_pid == 0) {
    //  child process
    close(source_pipe_fd[0]);
    dup2(source_pipe_fd[1], STDOUT_FILENO);

    // When using GDB, sending Ctrl+C to the program will send a SIGINT signal
    // to the child process as well, subsequently causing it to terminate. To
    // avoid this, we let the child process ignore (SIG_IGN) the SIGINT signal.
    // Reference:
    // https://stackoverflow.com/questions/38404925/gdb-interrupt-running-process-without-killing-child-processes
    signal(SIGINT, SIG_IGN);

    execle("/bin/sh", "sh", "-c", trace_source_cmd.c_str(), NULL, environ);
    perror("execle"); // child shouldn't reach here if all is well.
    exit(1);
  } else if (source_process_pid > 0) {
    // parent process - the trace post processor
    // stdin is now redirected to the read end of the source_pipe
    close(source_pipe_fd[1]);
    int r = dup2(source_pipe_fd[0], STDIN_FILENO);
  } else {
    cerr << "Failed to fork data source process\n";
    perror("fork");
    exit(1);
  }

  // fork a child process as the trace sink
  if (pipe(sink_pipe_fd) != 0) {
    cerr << "Failed to create pipe\n";
    perror("pipe");
    exit(1);
  }
  sink_process_pid = fork();
  if (sink_process_pid == 0) {
    // child process
    close(sink_pipe_fd[1]);
    dup2(sink_pipe_fd[0], STDIN_FILENO);
    signal(SIGINT, SIG_IGN); // ignore SIGINT
    execle("/bin/sh", "sh", "-c", trace_sink_cmd.c_str(), NULL, environ);
    perror("execle"); // child shouldn't reach here if all is well.
    exit(1);
  } else if (sink_process_pid > 0) {
    // parent process - the trace post processor
    // stdout is now redirected to the write end of the sink_pipe
    close(sink_pipe_fd[0]);
    int r = dup2(sink_pipe_fd[1], STDOUT_FILENO);
  } else {
    cerr << "Failed to fork data sink process\n";
    perror("fork");
    exit(1);
  }

  cerr << "Processing file " << filepath << endl;

  vector<threadblock_info> insts;
  unsigned grid_dim_x, grid_dim_y, grid_dim_z, tb_dim_x, tb_dim_y, tb_dim_z;
  unsigned tb_id_x, tb_id_y, tb_id_z, tb_id, warpid_tb;
  unsigned lineinfo, linenum;
  string line;
  stringstream ss;
  string string1, string2;
  bool found_grid_dim = false, found_block_dim = false;

  // Add a flag for LDGSTS instruction to indicate which one to remove
  vector<vector<bool>> ldgsts_flags; // true to remove, false to not

  // Important... without clear(), cin.eof() may evaluate to true on the second
  // kernel
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
      cout << line << endl;
      continue;
    } else {

      ss.str(line);
      ss >> tb_id_x >> tb_id_y >> tb_id_z >> warpid_tb;
      tb_id =
          tb_id_z * grid_dim_y * grid_dim_x + tb_id_y * grid_dim_x + tb_id_x;
      if (!insts[tb_id].initialized) {
        insts[tb_id].tb_id_x = tb_id_x;
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
      // trace, because it has two memory references. This is trying to remove
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

  for (unsigned i = 0; i < insts.size(); ++i) {
    // ofs<<string<<"\n";
    if (insts[i].initialized && insts[i].warp_insts_array.size() > 0) {
      cout << "\n"
           << "#BEGIN_TB"
           << "\n";
      cout << "\n"
           << "thread block = " << insts[i].tb_id_x << "," << insts[i].tb_id_y
           << "," << insts[i].tb_id_z << "\n";
    } else {
      cerr << "Warning: Thread block " << insts[i].tb_id_x << ","
           << insts[i].tb_id_y << "," << insts[i].tb_id_z << " is empty"
           << "\n";
      continue;
    }
    for (unsigned j = 0; j < insts[i].warp_insts_array.size(); ++j) {
      cout << "\n"
           << "warp = " << j << "\n";
      cout << "insts = " << insts[i].warp_insts_array[j].size() << "\n";
      if (insts[i].warp_insts_array[j].size() == 0) {
        cerr << "Warning: Warp " << j << " in thread block" << insts[i].tb_id_x
             << "," << insts[i].tb_id_y << "," << insts[i].tb_id_z
             << " is empty"
             << "\n";
      }
      for (auto it = insts[i].warp_insts_array[j].cbegin();
           it != insts[i].warp_insts_array[j].cend(); ++it) {
        // dereference once: const string*
        // dereference twice: const string
        cout << **it << "\n";
      }
    }
    cout << endl << "#END_TB" << endl;
  }

  close(source_pipe_fd[0]);
  close(source_pipe_fd[1]);
  close(sink_pipe_fd[0]);
  close(sink_pipe_fd[1]);

  // restore stdin/stdout file descriptor
  dup2(preserved_stdin_fileno, STDIN_FILENO);
  dup2(preserved_stdout_fileno, STDOUT_FILENO);
  close(preserved_stdin_fileno);
  close(preserved_stdout_fileno);
}

void group_per_core(const char *filepath) {

  // TO DO
}
