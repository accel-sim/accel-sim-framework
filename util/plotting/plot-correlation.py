#!/usr/bin/env python

from optparse import OptionParser
import plotly
import plotly.plotly as py
import plotly.tools as tls
from plotly.graph_objs import *
import os

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
                        print split
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
                        print str(count) + "--" + str(len(klist))
                        print x
                        appargs,kname,num = klist[count]
                        all_kerns[appargs][num][current_stat] = float(x)
                    except ValueError:
                        all_kerns[appargs][num][current_stat] = float(0)
                    count += 1
    return all_kerns

def parse_hw_csv(csv_file):
    kdata = []
    with open(csv_file, 'r') as data_file:
        reader = csv.reader(data_file)        # define reader object
        state = "start"
        header = []
        for row in reader:                    # loop through rows in csv file
            if state == "start":
                if "Profiling result" in row[0]:
                    state = "header_proc"
                continue
            if state == "header_proc":
                header = row
                state = "kernel_proc"
                continue
            if state == "kernel_proc":
                count = 0
                kstat = {}
                for elem in row:
                    kstat[header[count]] = elem
                    count += 1
                kdata.append(kstat)
                continue
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

#print hw_data
#exit()

#Get the simulator data
sim_data = get_sim_csv_data(options.csv_file)

print sim_data

exit()


for bench in benchmarks:
    edir, ddir, exe, args = bench
    ddir = os.path.join(this_directory,ddir,exe)
    if args[0] == "" or args[0] == None:
        run_name= os.path.join(exe,"NO_ARGS")
    else:
        run_name = os.path.join(exe, re.sub(r"[^a-z^A-Z^0-9]", "_", str(args).strip()))

    this_run_dir = os.path.join(this_directory, "..", "..", "run_hw", "device-" + options.device_num, cuda_version, run_name)
    print this_run_dir

exit()

all_stats = get_csv_data(options.csv_file)

colors= ['#0F8C79','#BD2D28','#E3BA22']
stat_count = 0
for stat,value in all_stats.iteritems():
    traces = []
    cfg_count = 0
    apps, data = value
    for k,v in data.iteritems():
        traces.append(Bar(
            x= apps,
            y= v,
            name=k,
            marker=Marker(color=colors[cfg_count]),
            xaxis='x1',
            yaxis='y{}'.format(stat_count+1)
            )
        )
        cfg_count += 1

    data = Data(traces)
    layout = Layout(
        title=stat,
        barmode='group',
        bargroupgap=0,
        bargap=0.25,
        showlegend=True,
        yaxis=YAxis(
            title="test",
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    )
    fig = Figure(data=data, layout=layout)
    figure_name = options.basename+"--"+stat
    print "plotting: " + figure_name
    outdir = (os.path.join(this_directory,"htmls"))
    if not os.path.exists( outdir ):
        os.makedirs(outdir)
    plotly.offline.plot(fig, filename=os.path.join(outdir,figure_name + ".html"),auto_open=False)
    stat_count += 1
