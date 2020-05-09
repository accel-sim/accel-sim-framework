#!/usr/bin/env python

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
    build_num = None
    build_nums = set()
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
                                    build_num = data[config][count][21:28]
                                    build_nums.add(build_num)
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
                    full_config = row[0] + "-" + str(build_num)
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
    if len(build_nums) > 1:
        exit("File {0} contains more than one build_num - this assumes one stats file has one build num: {1}"
            .format(filepath, build_nums))
    return all_named_kernels, stat_map, app_and_args, cached_configs, stats, build_nums


# After collection, spew out the tables
def print_stat(stat_name, all_named_kernels, apps_and_args, configs, stat_map, cfg_as_rows):
    csv_str = ""
    DIVISION = "-" * 100
    csv_str += DIVISION + "\n"
    csv_str += stat_name + "\n,"
    if cfg_as_rows:
        for appargs in apps_and_args:
            knames = all_named_kernels[appargs]
            for kname in knames:
                if kname == "":
                    continue
                csv_str += appargs + "--" + kname + ","

        csv_str = csv_str[:-1]
        csv_str += "\n"
        for config in configs:
            csv_str += config + ","
            for appargs in apps_and_args:
                knames = all_named_kernels[appargs]
                for kname in knames:
                    if kname == "":
                        continue
                    if kname + appargs + config + stat_name in stat_map:
                        csv_str += str(stat_map[kname + appargs + config + stat_name]) + ","
                    else:
                        csv_str += "NA,"
            csv_str = csv_str[:-1]
            csv_str += "\n"
    else:
        for config in configs:
            csv_str += config + ","
        csv_str = csv_str[:-1]
        csv_str += "\n"
        for appargs in apps_and_args:
            knames = all_named_kernels[appargs]
            for kname in knames:
                if kname == "":
                    continue
                csv_str += appargs + "--" + kname + ","
                for config in configs:
                    if kname + appargs + config + stat_name in stat_map:
                        csv_str += str(stat_map[kname + appargs + config + stat_name]) + ","
                    else:
                        csv_str += "NA,"
                csv_str = csv_str[:-1]
                csv_str += "\n"

    csv_str = csv_str[:-1]
    csv_str += "\n"
    print csv_str

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
    csv_files.append( common.file_option_test( csvf, "", this_directory ) )

stats_per_file = {}
for csvf in csv_files:
    stats_per_file[csvf] = get_csv_data_for_merge(csvf)

new_stats = {}
new_configs = []
union_apps_args = set()
union_stats = set()
for fname, ( all_named_kernels, stat_map, apps_and_args, configs, stats, build_nums ) in stats_per_file.iteritems():
    print "Processing {0}".format(fname)
    new_stats = dict(new_stats, **stat_map)
    new_configs += configs

    if len(union_apps_args) == 0:
        union_apps_args = set(apps_and_args)
    else:
        union_apps_args &= set(apps_and_args)

    if len(union_stats) == 0:
        union_stats = set(stats)
    else:
        union_stats &= set(stats)

print union_stats
print union_apps_args
print new_configs

for stat in stats:
    print_stat( stat, all_named_kernels, apps_and_args, new_configs, new_stats, options.configs_as_rows )
