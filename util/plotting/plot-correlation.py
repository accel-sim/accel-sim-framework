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


class Logger:
    def __init__(self, verbose):
        self.correl_log = ""
        self.verbose = verbose

    def log(self, msg):
        if self.verbose:
            print msg
        self.correl_log += msg + "\n"

    def write_log(self):
        now_time = datetime.datetime.now()
        day_string = now_time.strftime("%y.%m.%d-%A")
        time_string = now_time.strftime("%H:%M:%S")
        logfile = "correl_log--" + day_string + "--" + time_string + ".log"
        log_dir = os.path.join(this_directory, "correl_logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        open(os.path.join(log_dir,logfile),"w").write(self.correl_log)

def get_sim_csv_data(filepath, logger):
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
                logger.log("Processing Stat {0}".format(current_stat))
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
                            logger.log("Found appargs {0}. Kernel {1}".format(appargs,kname))
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
                logger.log("Processing config: {0}".format(cfg))
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

def parse_hw_csv(csv_file, logger):
    kdata = []
    processFiles = True
    processedCycle = False
    cfg = ""
    cfg_col = None
    while processFiles:
        with open(csv_file, 'r') as data_file:
            logger.log("Parsing HW csv file {0}".format(csv_file))
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
                    if len(row) == 1:
                        logger.log("Bad line - possibly the app failed -- {0}".format(row))
                        continue
                    if processedCycle:
                        count = 0
                        for elem in row:
                            kdata[kcount][header[count]] = elem
                            count += 1
                        kname = kdata[kcount]["Kernel"]
                        logger.log("Kernel Launch {0}: HW Kernel {1} found".format(kcount,kname))
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
                        logger.log("Kernel Launch {0}: HW Kernel {1} found".format(kcount,kname))
                        kdata.append(kstat)
                        kcount += 1
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
parser.add_option("-v", "--verbose", dest="verbose",
                  help="spew the log to stdout",
                  action="store_true")
(options, args) = parser.parse_args()
common.load_defined_yamls()

benchmarks = []
benchmarks = common.gen_apps_from_suite_list(options.benchmark_list.split(","))
options.hardware_dir = common.dir_option_test( options.hardware_dir, "../../run_hw/", this_directory )
options.data_mappings = common.file_option_test( options.data_mappings, "correl_mappings.py", this_directory )

logger = Logger(options.verbose)

# Get the hardware Data
logger.log("Getting HW data\n")
hw_data = {}
tmp = {}
for root, dirs, files in os.walk(options.hardware_dir):
    for d in dirs:
        csv_dir = os.path.join(root, d)
        csvs = glob.glob(os.path.join(csv_dir,"*.cycle"))
        logger.log("Found HW {0} csvs in {1}\n".format(len(csvs),csv_dir))
        if len(csvs) > 0:
            tmp[os.path.join(os.path.basename(root),d)],cfgname\
                = parse_hw_csv(max(csvs, key=os.path.getctime),logger)
            hw_data[cfgname] = tmp


#Get the simulator data
logger.log("Processing simulator data\n")
sim_data = get_sim_csv_data(options.csv_file, logger)

exec(open(options.data_mappings,'r').read())

fig_data = {} # map of HW config to a list of scatters
for cfg,sim_for_cfg in sim_data.iteritems():
    if cfg not in config_maps:
        continue

    hw_cfg = None
    for device in hw_data.iterkeys():
        if config_maps[cfg] in device:
            hw_cfg = device
            continue

    if hw_cfg == None:
        logger.log("Cannot find HW data for {0} skipping plots.".format(hw_cfg))
        continue

    for correl in correl_list:
        if correl.config != "all" and cfg != correl.config:
            logger.log("for cfg:{0} - Skipping plot:\n{1}".format(cfg, correl))
            continue

        hw_array = []
        sim_array = []
        label_array = []
        color_array = []

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
                processAnyKernels = False
                if len(hw_klist) == len(sim_klist):
                    logger.log("Found hw/sim match for {0}".format(appargs))
                    sim_appargs_leftover.remove(appargs)
                    hw_appargs_leftover.remove(appargs)
                    count = 0
                    for sim in sim_klist:
                        hw = hw_klist[count]
                        try:
                            hw_array.append(eval(correl.hw_eval))
                        except KeyError as e:
                            logger.log("Potentially uncollected stat in {0}.Error: {1}".format(correl.hw_eval, e))
                            continue
                        try:
                            sim_array.append(eval(correl.sim_eval))
                        except KeyError as e:
                            print e
                            logger.log("Potentially uncollected stat in {0}.Error: {1}".format(correl.sim_eval, e))
                            hw_array = hw_array[:-1]
                            continue
                        kernelcount += 1
                        processAnyKernels = True
                        err = 99999
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
                    logger.log("For appargs={0}, HW/SW kernels do not match HW={1}, SIM={2}\n"\
                        .format(appargs, len(hw_klist), len(sim_klist)))
                if processAnyKernels:
                    appcount += 1
        logger.log("Sim apps no HW:\n{0}\nHW apps no sim data:\n{1}"\
            .format(sim_appargs_leftover, hw_appargs_leftover))

        logger.write_log()

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
            name=cfg,
        )
        chart_info = correl.chart_name + " for " + cfg + "\n({0} apps, {1} kernels({4} under, {5} over))\n[Correl={2:.2} Err={3:.2f}%]".format(appcount, kernelcount,correl_co, avg_err,num_under,num_over)
        layout = Layout(
            title=chart_info,
             xaxis=dict(
                title='Hardware',
            ),
            yaxis=dict(
                title='GPGPU-Sim',
            )
        )
        data = [trace]
        correl_outdir = os.path.join(this_directory, "correl-html")
        plotname = filename=os.path.join(correl_outdir, cfg + "." + correl.plotfile)
        if not os.path.isdir(correl_outdir):
            os.makedirs(correl_outdir)
        if hw_cfg not in fig_data:
            fig_data[hw_cfg] = []
        fig_data[hw_cfg].append((trace, layout, plotname, chart_info))


for hw_cfg, traces in fig_data.iteritems():
    print "Plotting HW cfg:{0}".format(hw_cfg)
    data = []
    markers =[dict(size = 10, color = 'rgba(152, 0, 0, .8)', line = dict(width = 2,color = 'rgb(0, 0, 0)')),
              dict(size = 10,color = 'rgba(255, 182, 193, .9)',line = dict(width = 2,))]
    count = 0
    agg_info = ""
    SEP = "  :  "
    agg_plotname = ""
    for trace, layout, plotname, chart_info in traces:
        trace.marker = markers[count %len(markers)]
        trace.mode = "markers"
        data.append(trace)
        agg_info +=  chart_info + SEP
        if agg_plotname == "":
            agg_plotname = plotname
        agg_plotname += os.path.basename(plotname)
        count += 1

    agg_info = agg_info[:-len(SEP)]
    layout.title = agg_info
    print "Plotting {0}: {1}".format(agg_plotname, agg_info)
    plotly.offline.plot(Figure(data=data,layout=layout), filename=agg_plotname, auto_open=False)
