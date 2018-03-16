#!/usr/bin/env python

from optparse import OptionParser
import plotly
import plotly.plotly as py
import plotly.tools as tls
from plotly.graph_objs import *
import os
import plotly.graph_objs as go

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

import sys
sys.path.insert(0,os.path.join(this_directory,"..","job_launching"))
import common

import numpy as np  # (*) numpy for math functions and arrays
import csv
import re
import glob


def get_sim_csv_data(filepath):
    all_kerns = {}
    klist = []
    with open(filepath, 'r') as data_file:
        reader = csv.reader(data_file)        # define reader object
        state = "start"
        first_stat = True
        for row in reader:                    # loop through rows in csv file
            if len(row) != 0 and row[0].startswith("----"):
                state = "find-stat"
                continue
            if state == "find-stat":
                current_stat = row[0]
                state = "find-apps"
                continue
            if state == "find-apps":
                if first_stat:
                    last_appargs = ""
                    for item in row[1:]:
                        split = item.split("--")
                        if len(split) > 1:
                            appargs = split[0]
                            kname = split[1]
                            if kname == "all_kernels":
                                continue
                            if appargs == last_appargs:
                                num += 1
                            else:
                                last_appargs = appargs
                                num = 0
                            klist.append((appargs, kname, num))
                            if appargs not in all_kerns:
                                all_kerns[appargs] = []
                            all_kerns[appargs].append({})
                            all_kerns[appargs][-1]["Kernel"] = kname
                if not kname == "all_kernels":
                    state = "process-cfgs"
                continue
            if state == "process-cfgs":
                if len(row) == 0:
                    first_stat = False
                    state = "start"
                    continue
                count = 0
                for x in row[1:]:
                    try:
                        appargs,kname,num = klist[count]
                        all_kerns[appargs][num][current_stat] = float(x)
                    except ValueError:
                        all_kerns[appargs][num][current_stat] = float(0)
                    count += 1
    return all_kerns

def parse_hw_csv(csv_file):
    kdata = []
    processFiles = True
    processedCycle = False
    while processFiles:
        with open(csv_file, 'r') as data_file:
            reader = csv.reader(data_file)        # define reader object
            state = "start"
            header = []
            kcount = 0
            for row in reader:                    # loop through rows in csv file
                if state == "start":
                    if "Profiling result" in row[0]:
                        state = "header_proc"
                    continue
                if state == "header_proc":
                    header = row
                    state = "blanc_proc"
                    continue
                if state == "blanc_proc":
                    state = "kernel_proc"
                    continue
                if state == "kernel_proc":
                    if processedCycle:
                        if "[CUDA " in "".join(row):
                            continue
                        count = 0
                        for elem in row:
                            kdata[kcount][header[count]] = elem
                            count += 1
                        kcount += 1
                    else:
                        kstat = {}
                        count = 0
                        for elem in row:
                            kstat[header[count]] = elem
                            count += 1
                        kdata.append(kstat)
                    continue
        if os.path.exists(csv_file + ".cycle") and not processedCycle and len(kdata) > 0:
            processedCycle = True
            csv_file += ".cycle"
        else:
            processFiles = False

    return kdata


parser = OptionParser()
parser = OptionParser()
parser.add_option("-B", "--benchmark_list", dest="benchmark_list",
                  help="a comma seperated list of benchmark suites to run. See apps/define-*.yml for " +\
                        "the benchmark suite names.",
                  default="rodinia_2.0-ft")
parser.add_option("-C", "--configs_list", dest="configs_list",
                  help="a comma seperated list of configs to run. See configs/define-*.yml for " +\
                        "the config names.",
                  default="TITANX-P102")
parser.add_option("-H", "--hardware_dir", dest="hardware_dir",
                  help="The hardware stats directories",
                  default="")
parser.add_option("-c", "--csv_file", dest="csv_file",
                  help="File to parse",
                  default="")
(options, args) = parser.parse_args()
common.load_defined_yamls()

benchmarks = []
benchmarks = common.gen_apps_from_suite_list(options.benchmark_list.split(","))
options.hardware_dir = common.dir_option_test( options.hardware_dir, "../../run_hw/", this_directory )

# Get the hardware Data
hw_data = {}
for root, dirs, files in os.walk(options.hardware_dir):
    for d in dirs:
        csvs = glob.glob(os.path.join(root, d,"*.csv"))
        if len(csvs) > 0:
            hw_data[os.path.join(os.path.basename(root),d)] = parse_hw_csv(max(csvs, key=os.path.getctime))


#Get the simulator data
sim_data = get_sim_csv_data(options.csv_file)

hw_array = []
sim_array = []
label_array = []
for appargs,sim_klist in sim_data.iteritems():
#    print appargs
    if appargs in hw_data:
        hw_klist = hw_data[appargs]
        if len(hw_klist) == len(sim_klist):
            count = 0
            for k in sim_klist:
                hw_num = hw_klist[count]["Duration"]
                sw_num = k["gpu_tot_sim_cycle\s*=\s*(.*)"]
                # HW in us - clock is 1417 MHz
                # HW_CYCLES=hw_us/10^6
                hw_array.append(float(hw_num)*1417)
                sim_array.append(float(sw_num))
                label_array.append(appargs + "--" + hw_klist[count]["Name"])
                count += 1
#            print "Sane"
#        else:
#            print "HW List"
#            for k in hw_klist:
#                print k["Kernel"]
#            print "SIM List"
#            for k in sim_klist:
#                print k["Kernel"]
#            print "INSANE hw={0},sim={1}".format(len(hw_klist),len(sim_klist))



# Create a trace
trace = go.Scatter(
    x = hw_array,
    y = sim_array,
    mode = 'markers',
    text=label_array,
)
layout = Layout(
    title="TITANX-P102 Correlation on rodinia_2.0-ft",
     xaxis=dict(
        title='HW Cycles (us x 1417)',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='GPGPU-Sim Cycles',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

data = [trace]

# Plot and embed in ipython notebook!
plotly.offline.plot(Figure(data=data,layout=layout), filename=os.path.join("first-correl.html"),auto_open=False)
