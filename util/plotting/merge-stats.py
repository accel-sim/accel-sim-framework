#!/usr/bin/env python3

from __future__ import print_function
from optparse import OptionParser
import os
import subprocess
import shutil
import glob

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

import sys
sys.path.insert(0,os.path.join(this_directory,"..","job_launching"))
import common

import numpy as np  # (*) numpy for math functions and arrays
import csv

def get_csv_data_for_merge(filepath):
    all_named_kernels = {}
    stat_map = {}
    apps = []
    cached_apps = []
    configs = []
    cached_configs = []
    stats = []
    gpgpu_build_num = None
    gpgpu_build_nums = set()
    accel_build_num = None
    accel_build_nums = set()
    data = {}
    with open(filepath, 'r') as data_file:
        reader = csv.reader(data_file)        # define reader object
        state = "start"
        for row in reader:                    # loop through rows in csv file
            if len(row) != 0 and row[0].startswith("----"):
                state = "find-stat"
                continue
            if state == "find-stat":
                current_stat = row[0]
                stats.append(current_stat)
                state = "find-apps"
                continue
            if state == "find-apps":
                apps = row[1:]
                state = "process-cfgs"
                continue
            if state == "process-cfgs":
                if len(row) == 0:
                    if any_data:
                        cached_apps = apps
                        cached_configs = configs
                        for config in configs:
                            count = 0
                            for appargs_kname in apps:
                                first_delimiter = appargs_kname.find("--")
                                appargs = appargs_kname[ :first_delimiter ]
                                kname = appargs_kname[ first_delimiter + 2: ]
                                if current_stat == "GPGPU-Sim-build" and data[config][count] != "NA":
                                    gpgpu_build_num = data[config][count][21:28]
                                    gpgpu_build_nums.add(gpgpu_build_num)
                                if current_stat == "Accel-Sim-build" and data[config][count] != "NA":
                                    accel_build_num = data[config][count][16:23]
                                    accel_build_nums.add(accel_build_num)
                                stat_map[kname + appargs + config + current_stat ] = data[config][count]
                                count += 1
                    apps = []
                    configs = []
                    data = {}
                    state = "start"
                    any_data = False
                    continue
                else:
                    any_data = True
                    if accel_build_num != None and gpgpu_build_num != None:
                        full_config = row[0] + "-accel-" + str(accel_build_num) + "-gpgpu-" + str(gpgpu_build_num)
                    else:
                        full_config = row[0]
                    configs.append(full_config)
                    data[full_config] = row[1:]

    app_and_args = []
    for appargs_kname in cached_apps:
        first_delimiter = appargs_kname.find("--")
        appargs = appargs_kname[ :first_delimiter ]
        kname = appargs_kname[ first_delimiter + 2: ]
        if appargs not in all_named_kernels:
            all_named_kernels[appargs] = []
            app_and_args.append(appargs)
        all_named_kernels[appargs].append( kname )

    # The assumption here is that every entry in each stats file is run with the same
    # git hash number, if not we are just going to warn and fail.
    if len(gpgpu_build_nums) > 1 or len(accel_build_nums) > 1:
        exit("File {0} contains more than one gpgpu_build_num or accelsim_build_num - this assumes one stats file has one build num: gpgpu: {1}" +\
            " accel: {2}"\
            .format(filepath, gpgpu_build_nums, accel_build_nums))
    return all_named_kernels, stat_map, app_and_args, cached_configs, stats, gpgpu_build_nums

parser = OptionParser()
parser.add_option("-c", "--csv_files", dest="csv_files",
                  help="Files to merge",
                  default="")
parser.add_option("-R", "--configs_as_rows", dest="configs_as_rows",
                  help="When printing merged files, are configs as rows?",
                  action="store_true")
(options, args) = parser.parse_args()

csv_files = []
for csvf in options.csv_files.split(","):
    try:
        csv_files.append( common.file_option_test( csvf, "", this_directory ) )
    except common.PathMissing as e:
        print("Warning path {0}. Continuing".format(e), file=sys.stderr)

stats_per_file = {}
for csvf in csv_files:
    stats_per_file[csvf] = get_csv_data_for_merge(csvf)

new_stats = {}
new_configs = []
union_apps_args = set()
union_stats = set()
union_configs = set()
for csvf in csv_files:
    ( all_named_kernels, stat_map, apps_and_args, configs, stats, gpgpu_build_nums ) = stats_per_file[csvf]
    print("Processing {0}".format(csvf), file=sys.stderr)
    new_stats = dict(new_stats, **stat_map)
    for config in configs:
        if config not in union_configs:
            union_configs.add(config)
            new_configs.append(config)
        else:
            print("Found redundant config: {0} in csvf: \"{1}\" - filtering it out.".format(config, csvf), file=sys.stderr)

    if len(union_apps_args) == 0:
        union_apps_args = set(apps_and_args)
    else:
        union_apps_args &= set(apps_and_args)

    if len(union_stats) == 0:
        union_stats = set(stats)
    else:
        union_stats &= set(stats)

for stat in stats:
    common.print_stat( stat, all_named_kernels, apps_and_args, new_configs, new_stats, options.configs_as_rows, False )
