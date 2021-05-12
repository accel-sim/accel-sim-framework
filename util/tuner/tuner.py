#!/usr/bin/env python3

from optparse import OptionParser
import os
import subprocess
import os
this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
import sys
import re
import shutil
import glob
import datetime
import string

parser = OptionParser()
parser.add_option("-s", "--stats_output", dest="stats_output",
                 help="the path to the stats output generated from the GPU_Microbenchmark suite",
                 default="./output.txt")

(options, args) = parser.parse_args()

# parse stats output
stats = {}
device_name_key = "Device Name"
device_name = "undefined"
stats_file = open(options.stats_output, 'r')
lines = stats_file.readlines()
for line in lines:
    if line[0] == '-':
        cols = line.split()
        stats[cols[0]] = line
    elif device_name_key in line:
        cols = line.split(' = ')
        device_name = cols[1].replace(" ","_").strip('\n')
print('parsing', device_name, 'stats is done')

#create a config folder for the device name
config = []
new_config_dir = os.path.join(this_directory, device_name)
print('creating', device_name, 'folder')
if not os.path.exists(new_config_dir):
    os.makedirs(new_config_dir)

files = ['gpgpusim.config', 'trace.config']
for config_file in files:
    config_temp = os.path.join(this_directory, "config_template")
    gpgpusim_file = os.path.join(config_temp, config_file)

    #copy config tempalte to the new device folder
    shutil.copy(gpgpusim_file, new_config_dir)
    with open(gpgpusim_file) as my_file:
        config = my_file.readlines()

    #for each config param, replace it with the one from the output ubench stats
    for idx, item in enumerate(config):
        if item[0] == '-':
            cols = item.split()
            if cols[0] in stats.keys():
                config[idx] = stats[cols[0]]

    #write the new config param in the config files
    new_gpgpusim_file = os.path.join(new_config_dir, config_file)
    print('writing new stats file to', config_file)
    with open(new_gpgpusim_file, 'w') as f:
        for item in config:
            f.write("%s" % item)

print('Done!')
