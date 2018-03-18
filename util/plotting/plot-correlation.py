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
import copy
import ast
import numpy
import datetime

correl_log = ""

def get_sim_csv_data(filepath, correl_log):
    all_kerns = {}
    all_kern_cfg = {}
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
                correl_log += "Processing Stat {0}\n".format(current_stat)
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
                            correl_log += "Found appargs {0}. Kernel {1}\n".format(appargs,kname)
                            klist.append((appargs, kname, num))
                            if appargs not in all_kern_cfg:
                                all_kern_cfg[appargs] = []
                            all_kern_cfg[appargs].append({})
                            all_kern_cfg[appargs][-1]["Kernel"] = kname
                if not kname == "all_kernels":
                    state = "process-cfgs"
                continue
            if state == "process-cfgs":
                if len(row) == 0:
                    first_stat = False
                    state = "start"
                    continue
                count = 0
                cfg = row[0]
                correl_log += "Processing config: {0}\n".format(cfg)
                if cfg not in all_kerns:
                    all_kerns[cfg] = copy.deepcopy(all_kern_cfg)
                for x in row[1:]:
                    try:
                        appargs,kname,num = klist[count]
                        all_kerns[cfg][appargs][num][current_stat] = float(x)
                    except ValueError:
                        all_kerns[cfg][appargs][num][current_stat] = float(0)
                    count += 1
    return all_kerns

def parse_hw_csv(csv_file, correl_log):
    kdata = []
    processFiles = True
    processedCycle = False
    cfg = ""
    cfg_col = None
    while processFiles:
        with open(csv_file, 'r') as data_file:
            correl_log +="Parsing HW csv file {0}\n".format(csv_file)
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
                    count = 0

                    # get the device column
                    for elem in row:
                        if elem == "Device":
                            cfg_col = count
                        count += 1

                    state = "blanc_proc"
                    continue
                if state == "blanc_proc":
                    state = "kernel_proc"
                    continue
                if state == "kernel_proc":
                    if "[CUDA " in "".join(row):
                        continue
                    if processedCycle:
                        count = 0
                        for elem in row:
                            kdata[kcount][header[count]] = elem
                            count += 1
                        kname = kdata[kcount]["Kernel"]
                        correl_log += "Kernel Launch {0}: HW Kernel {1} found\n".format(kcount,kname)
                        kcount += 1
                    else:
                        # Set the Device
                        if cfg != "" and cfg != row[cfg_col]:
                            print "data for more than one device in {0}..{1}:{2}".format(csv_file,cfg,elem)
                            exit()
                        cfg = row[cfg_col]

                        kstat = {}
                        count = 0
                        for elem in row:
                            kstat[header[count]] = elem
                            count += 1
                        kname = kstat["Name"]
                        correl_log += "Kernel Launch {0}: HW Kernel {1} found\n".format(kcount,kname)
                        kdata.append(kstat)
                    continue
        # Drop the .cycle off the name
        if os.path.exists(csv_file[:-6]) and not processedCycle and len(kdata) > 0:
            processedCycle = True
            csv_file = csv_file[:-6]
        else:
            processFiles = False

    return kdata, cfg


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
parser.add_option("-d", "--data_mappings", dest="data_mappings",
                  help="python file that descibes your desired data mappings",
                  default="")
(options, args) = parser.parse_args()
common.load_defined_yamls()

benchmarks = []
benchmarks = common.gen_apps_from_suite_list(options.benchmark_list.split(","))
options.hardware_dir = common.dir_option_test( options.hardware_dir, "../../run_hw/", this_directory )
options.data_mappings = common.file_option_test( options.data_mappings, "correl_mappings.py", this_directory )

# Get the hardware Data
correl_log += "Getting HW data\n"
hw_data = {}
tmp = {}
for root, dirs, files in os.walk(options.hardware_dir):
    for d in dirs:
        csv_dir = os.path.join(root, d)
        csvs = glob.glob(os.path.join(csv_dir,"*.cycle"))
        correl_log += "Found HW {0} csvs in {1}\n".format(len(csvs),csv_dir)
        if len(csvs) > 0:
            tmp[os.path.join(os.path.basename(root),d)],cfgname\
                = parse_hw_csv(max(csvs, key=os.path.getctime),correl_log)
            hw_data[cfgname] = tmp


#Get the simulator data
correl_log += "Processing simulator data\n"
sim_data = get_sim_csv_data(options.csv_file, correl_log)

exec(open(options.data_mappings,'r').read())

hw_array = []
sim_array = []
label_array = []
color_array = []

for cfg,sim_for_cfg in sim_data.iteritems():
    if cfg not in config_maps:
        continue

    hw_cfg = None
    for device in hw_data.iterkeys():
        if config_maps[cfg] in device:
            hw_cfg = device
            continue

    if hw_cfg == None:
        msg =  "Cannot find HW data for {0} skipping plots.\n".format(hw_cfg)
        print msg
        correl_log += msg
        continue

    for correl in correl_list:
        if correl.config != "all" and cfg != correl.config:
            print "for cfg:{0} - Skipping plot:\n{1}\n".format(cfg, correl)
            continue

        appcount = 0
        kernelcount = 0
        num_under = 0
        num_over = 0
        errs = []
        sim_appargs_leftover = set(copy.deepcopy(sim_for_cfg.keys()))
        hw_appargs_leftover = set(copy.deepcopy(hw_data[hw_cfg].keys()))
        for appargs,sim_klist in sim_for_cfg.iteritems():
            if appargs in hw_data[hw_cfg]:
                hw_klist = hw_data[hw_cfg][appargs]
                if len(hw_klist) == len(sim_klist):
                    correl_log += "Found hw/sim match for {0}\n".format(appargs)
                    sim_appargs_leftover.remove(appargs)
                    hw_appargs_leftover.remove(appargs)
                    appcount += 1
                    count = 0
                    for sim in sim_klist:
                        kernelcount += 1
                        hw = hw_klist[count]
                        try:
                            hw_pass = True
                            hw_array.append(eval(correl.hw_eval))
                        except KeyError:
                            hw_pass = False
                            correl_log += "Potentially uncollected stat in {0}]\n".format(correl.hw_eval)
                        try:
                            if hw_pass:
                                sim_array.append(eval(correl.sim_eval))
                        except KeyError:
                            correl_log += "Potentially uncollected stat in {0}]\n".format(correl.sim_eval)

                        err = None
                        if hw_array[-1] > 0:
                            err = sim_array[-1] - hw_array[-1]
                            err = (err / hw_array[-1]) * 100
                            if err > 0:
                                num_over += 1
                            else:
                                num_under += 1
                            errs.append(abs(err))
                        label_array.append(appargs + "--" + hw_klist[count]["Name"] + " (Err={0:.2f}%)".format(err))
                        count += 1
                else:
                    msg = "For appargs={0}, HW/SW kernels do not match HW={1}, SIM={2}\n"\
                        .format(appargs, len(hw_klist), len(sim_klist))
                    print msg
                    correl_log += msg
    
        correl_log += "Sim apps no HW:\n{0}\nHW apps no sim data:\n{1}\n"\
            .format(sim_appargs_leftover, hw_appargs_leftover)

        now_time = datetime.datetime.now()
        day_string = now_time.strftime("%y.%m.%d-%A")
        time_string = now_time.strftime("%H:%M:%S")
        logfile = "correl_log--" + day_string + "--" + time_string + ".log"
        open(os.path.join(this_directory,logfile),"w").write(correl_log)

        correl_co = numpy.corrcoef(hw_array, sim_array)[0][1]
        avg_err = 0
        for err in errs:
            avg_err += err
        avg_err = avg_err / len(errs)

        trace = go.Scatter(
            x = hw_array,
            y = sim_array,
            mode = 'markers',
            text=label_array,
        )
        layout = Layout(
            title=correl.chart_name + " for " + cfg + " ({0} apps, {1} kernels({4} under, {5} over)) [Correl={2:.2} Err={3:.2f}%]".format(appcount, kernelcount,correl_co, avg_err,num_under,num_over),
             xaxis=dict(
                title='Hardware',
            ),
            yaxis=dict(
                title='GPGPU-Sim',
            )
        )
        
        data = [trace]

        plotname = filename=os.path.join("correl-html", cfg + "." + correl.plotfile)
        print "Plotting {0}".format(plotname)
        if not os.path.isdir("correl-html"):
            os.makedirs("correl-html")
        plotly.offline.plot(Figure(data=data,layout=layout), filename=plotname, auto_open=False)
