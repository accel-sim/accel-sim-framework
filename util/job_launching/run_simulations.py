#!/usr/bin/env python

from optparse import OptionParser
import os
import subprocess
import sys
import re
import shutil
import glob
import datetime
import yaml
import common

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
# This function will pull the SO name out of the shared object,
# which will have current GIT commit number attatched.
def extract_so_name( so_path ):
    objdump_out_filename = this_directory + "so_objdump_out.{0}.txt".format(os.getpid())
    objdump_out_file = open(objdump_out_filename, 'w+')
    subprocess.call(["objdump", "-p", so_path], stdout=objdump_out_file)
    objdump_out_file.seek(0)
    returnStr = re.sub( r".*SONAME\s+([^\s]+).*", r"\1", objdump_out_file.read().strip().replace("\n", " ") )
    objdump_out_file.close()
    os.remove(objdump_out_filename)
    return returnStr

#######################################################################################
# Class the represents each configuration you are going to run
# For example, if your sweep file has 2 entries 32k-L1 and 64k-L1 there will be 2
# ConfigurationSpec classes and the run_subdir name for each will be 32k-L1 and 64k-L1
# respectively
class ConfigurationSpec:
    #########################################################################################
    # Public Interface methods
    #########################################################################################
    # Class is constructed with a single line of text from the sweep_param file
    def __init__(self, ( name, params, config_file ) ):
        self.run_subdir = name
        self.params = params
        self.config_file = config_file

    def my_print(self):
        print "Run Subdir = " + self.run_subdir
        print "Parameters = " + self.params
        print "Base config file = " + self.config_file

    def run(self, build_handle, benchmarks, run_directory, cuda_version, libdir):
        for dir_bench in benchmarks:
            exec_dir, run_dir, benchmark, self.command_line_args_list = dir_bench
            full_exec_dir = os.path.join( this_directory, exec_dir )
            full_run_dir = os.path.join( this_directory, run_dir, benchmark.replace('/','_') )

            self.benchmark_args_subdirs = {}
            for args in self.command_line_args_list:
                if args == "" or args == None:
                    self.benchmark_args_subdirs[args] = "NO_ARGS"
                else:
                    self.benchmark_args_subdirs[args] = re.sub(r"[^a-z^A-Z^0-9]", "_", str(args).strip())


            for args in self.command_line_args_list:
                this_run_dir = run_directory +\
                            "/" + benchmark.replace('/','_') + "/" + self.benchmark_args_subdirs[args] +\
                            "/" + self.run_subdir + "/"
                self.setup_run_directory(full_run_dir, this_run_dir)

                self.text_replace_torque_sim(full_run_dir,this_run_dir,benchmark,cuda_version, args, libdir, full_exec_dir,build_handle)
                self.append_gpgpusim_config(benchmark, this_run_dir, self.config_file)
                
                # Submit the job to torque and dump the output to a file
                if not options.no_launch:
                    torque_out_filename = this_directory + "torque_out.{0}.txt".format(os.getpid())
                    torque_out_file = open(torque_out_filename, 'w+')
                    saved_dir = os.getcwd()
                    os.chdir(this_run_dir)
                    if subprocess.call(["qsub",\
                                        "-W", "umask=022",\
                                       this_run_dir + "torque.sim"],\
                                       stdout=torque_out_file) < 0:
                        exit("Error Launching Torque Job")
                    else:
                        # Parse the torque output for just the numeric ID
                        torque_out_file.seek(0)
                        torque_out = re.sub(r"(^\d+).*", r"\1",
                            torque_out_file.read().strip())
                        print("Job " + torque_out + " queued (" +\
                            benchmark + "-" + self.benchmark_args_subdirs[args] +\
                            " " + self.run_subdir + ")")
                    torque_out_file.close()
                    os.remove(torque_out_filename)
                    os.chdir(saved_dir)

                    if len(torque_out) > 0:
                        # Dump the benchmark description to the logfile
                        if not os.path.exists(this_directory + "logfiles/"):
                            # In the very rare case that concurrent builds try to make the directory at the same time
                            # (after the test to os.path.exists -- this has actually happened...)
                            try:
                                os.makedirs(this_directory + "logfiles/")
                            except:
                                pass
                        now_time = datetime.datetime.now()
                        day_string = now_time.strftime("%y.%m.%d-%A")
                        time_string = now_time.strftime("%H:%M:%S")
                        log_name = "sim_log.{0}".format(options.launch_name)
                        logfile = open(this_directory +\
                                       "logfiles/"+ log_name + "." +\
                                       day_string + ".txt",'a')
                        print >> logfile, "%s %6s %-22s %-100s %-25s %s.%s" %\
                               ( time_string ,\
                               torque_out ,\
                               benchmark ,\
                               self.benchmark_args_subdirs[args] ,\
                               self.run_subdir,\
                               benchmark,\
                               build_handle )
                        logfile.close()
            self.benchmark_args_subdirs.clear()

    #########################################################################################
    # Internal utility methods
    #########################################################################################
    # copies and links the necessary files to the run directory
    def setup_run_directory(self, full_bin_dir, this_run_dir):
        if not os.path.isdir(this_run_dir):
            os.makedirs(this_run_dir)

        files_to_copy_to_run_dir = glob.glob(os.path.join(full_bin_dir, "*.ptx")) +\
                                   glob.glob(os.path.join(full_bin_dir, "*.cl")) +\
                                   glob.glob(os.path.join(full_bin_dir, "*.h")) +\
                                   glob.glob(os.path.dirname(self.config_file) + "/*.icnt") +\
                                   glob.glob(os.path.dirname(self.config_file) + "/*.xml")

        for file_to_cp in files_to_copy_to_run_dir:
            new_file = this_run_dir +\
                       os.path.basename(this_directory + file_to_cp)
            if os.path.isfile(new_file):
                os.remove(new_file)
            shutil.copyfile(file_to_cp,new_file)
        
        # link the data directory
        if os.path.isdir(os.path.join(full_bin_dir, "data")):
            if os.path.lexists(os.path.join(this_run_dir, "data")):
                os.remove(os.path.join(this_run_dir, "data"))
            os.symlink(os.path.join(full_bin_dir, "data"), os.path.join(this_run_dir,"data"))

    # replaces all the "REAPLCE_*" strings in the torque.sim file
    def text_replace_torque_sim( self,full_run_dir,this_run_dir,benchmark, cuda_version, command_line_args,
                                 libpath, exec_dir, gpgpusim_build_handle ):
        # get the pre-launch sh commands
        prelaunch_filename =  full_run_dir +\
                             "benchmark_pre_launch_command_line.txt"
        benchmark_command_line = ""
        if(os.path.isfile(prelaunch_filename)):
            f = open(prelaunch_filename)
            benchmark_command_line = f.read().strip()
            f.close()
        
        exec_name = options.benchmark_exec_prefix + " " +\
                    os.path.join(this_directory, exec_dir, benchmark)

        # Test the existance of required env variables
        if str(os.getenv("GPGPUSIM_ROOT")) == "None":
            exit("\nERROR - Specify GPGPUSIM_ROOT prior to running this script")
        if str(os.getenv("OPENCL_REMOTE_GPU_HOST")) == "None":
            os.environ["OPENCL_REMOTE_GPU_HOST"] = ""
        if str(os.getenv("GPGPUSIM_CONFIG")) == "None":
            exit("\nERROR - Specify GPGPUSIM_CONFIG prior to running this script")

        # do the text replacement for the torque.sim file
        if command_line_args == None:
            txt_args = ""
        else:
            txt_args = command_line_args

        replacement_dict = {"NAME":benchmark + "-" + self.benchmark_args_subdirs[command_line_args] + "." +\
                                gpgpusim_build_handle,
                            "NODES":"1", 
                            "GPGPUSIM_ROOT":os.getenv("GPGPUSIM_ROOT"),
                            "LIBPATH": libpath,
                            "SUBDIR":this_run_dir,
                            "OPENCL_REMOTE_GPU_HOST":os.getenv("OPENCL_REMOTE_GPU_HOST"),
                            "BENCHMARK_SPECIFIC_COMMAND":benchmark_command_line,
                            "PATH":os.getenv("PATH"),
                            "EXEC_NAME":exec_name,
                            "COMMAND_LINE":txt_args}
        torque_text = open(this_directory + "torque.sim").read().strip()
        for entry in replacement_dict:
            torque_text = re.sub("REPLACE_" + entry,
                                 str(replacement_dict[entry]),
                                 torque_text)
        open(this_run_dir + "torque.sim", 'w').write(torque_text)

    # replaces all the "REPLACE_*" strings in the gpgpusim.config file
    def append_gpgpusim_config(self, bench_name, this_run_dir, config_text_file):
        benchmark_spec_opts_file = os.path.join( this_directory, "..", "..", "benchmarks",
            "app-specific-gpgpu-sim-options", bench_name, "benchmark_options.txt" )
        benchmark_spec_opts = ""
        if(os.path.isfile(benchmark_spec_opts_file)):
            f = open(benchmark_spec_opts_file)
            benchmark_spec_opts = f.read().strip()
            f.close()

        config_text = open(config_text_file).read()
        config_text += "\n" + benchmark_spec_opts + "\n" + self.params

        open(this_run_dir + "gpgpusim.config", 'w').write(config_text)

#-----------------------------------------------------------
# main script start
#-----------------------------------------------------------
(options, args) = common.parse_run_simulations_options()

if str(os.getenv("GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN")) != "1":
    sys.exit("ERROR - Please run setup_environment before running this script")


cuda_version = common.get_cuda_version( this_directory )

if options.run_directory == "":
    options.run_directory = os.path.join(this_directory, "../../sim_run_%s"%cuda_version)
else:
    options.run_directory = os.path.join(os.getcwd(), options.run_directory)

# Let's copy out the .so file so that builds don't interfere with running tests
# If the user does not specify a so file, then use the one in the git repo and copy it out.
options.so_dir = common.dir_option_test(
    options.so_dir, os.path.join( os.getenv("GPGPUSIM_ROOT"), "lib", os.getenv("GPGPUSIM_CONFIG") ),
    this_directory )
so_path = os.path.join( options.so_dir, "libcudart.so" )
version_string = extract_so_name( so_path )
running_so_dir = os.path.join( options.run_directory, "gpgpu-sim-builds", version_string )
if not os.path.exists( running_so_dir ):
    # In the very rare case that concurrent builds try to make the directory at the same time
    # (after the test to os.path.exists -- this has actually happened...)
    try:
        os.makedirs( running_so_dir )
    except:
        pass
    shutil.copy( so_path, running_so_dir )
options.so_dir = running_so_dir

common.load_defined_yamls()

# Test for the existance of torque on the system
if not any([os.path.isfile(os.path.join(p, "qsub")) for p in os.getenv("PATH").split(os.pathsep)]):
    exit("ERROR - Cannot find qsub in PATH... Is torque installed on this machine?")

if not any([os.path.isfile(os.path.join(p, "nvcc")) for p in os.getenv("PATH").split(os.pathsep)]):
    exit("ERROR - Cannot find nvcc PATH... Is CUDA_INSTALL_PATH/bin in the system PATH?")

benchmarks = []
benchmarks = common.gen_apps_from_suite_list(options.benchmark_list.split(","))

cfgs = common.gen_configs_from_list( options.configs_list.split(",") )
configurations = []
for config in cfgs:
    configurations.append( ConfigurationSpec( config ) )

print("Running Simulations with GPGPU-Sim built from \n{0}\n ".format(version_string) +
      "\nUsing configs: " + options.configs_list +
      "\nBenchmark: " + options.benchmark_list)

for config in configurations:
    config.my_print()
    config.run(version_string, benchmarks, options.run_directory, cuda_version, options.so_dir)
