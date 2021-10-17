#!/usr/bin/env python3

from optparse import OptionParser
import plotly
import plotly.tools as tls
from plotly.graph_objs import *
import os
import subprocess
import shutil
import glob
import re

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

import sys
sys.path.insert(0,os.path.join(this_directory,"..","job_launching"))
import common

import numpy as np  # (*) numpy for math functions and arrays
import csv


def get_csv_data(filepath):
    all_stats = {}
    apps = []
    data = {}
    any_data = False
    with open(filepath, 'r') as data_file:
        reader = csv.reader(data_file)        # define reader object
        state = "start"
        for row in reader:                    # loop through rows in csv file
            if len(row) != 0 and row[0].startswith("----"):
                state = "find-stat"
                continue
            if state == "find-stat":
                current_stat = row[0]
                state = "find-apps"
                continue
            if state == "find-apps":
                apps = [item.upper() for item in row[1:]]
                state = "process-cfgs"
                continue
            if state == "process-cfgs":
                if len(row) == 0:
                    if any_data:
                        all_stats[current_stat] = apps,data
                    apps = []
                    data = {}
                    state = "start"
                    any_data = False
                    continue
                temp = []
                for x in row[1:]:
                    try:
                        temp.append(float(x))
                        any_data = True
                    except ValueError:
                        temp.append(0)
                data[row[0]] = np.array(temp)

    return all_stats

parser = OptionParser()
parser = OptionParser()
parser.add_option("-n", "--basename", dest="basename",
                  help="Basename for plot generation",
                  default="gpgpu-sim")
parser.add_option("-c", "--csv_file", dest="csv_file",
                  help="File to parse",
                  default="")
parser.add_option("-p", "--publish_path", dest="publish_path",
                  help="After the htmls are generated - they will get published here."+\
                  " Assumes you can scp to this directory.",
                  default="")
parser.add_option("-w", "--publish_web", dest="publish_web",
                  help="After the htmls are generated - they will get published here."+\
                  " Assumes you can scp to this directory.",
                  default="")
parser.add_option("-P", "--plotname", dest="plotname",
                  help="String appended to the filenames",
                  default="")
(options, args) = parser.parse_args()
options.csv_file = common.file_option_test( options.csv_file, "", this_directory )

all_stats = get_csv_data(options.csv_file)

colors= ['#0F8C79','#BD2D28','#E3BA22','#E6842A','#137B80','#8E6C8A','#9A3E25', '#E6842A']
stat_count = 0
for stat,value in all_stats.items():
    traces = []
    cfg_count = 0
    apps, data = value
    for k,v in data.items():
        traces.append(Bar(
            x= apps,
            y= v,
            name=k,
            marker=Marker(color=colors[cfg_count % len(colors)]),
            xaxis='x1',
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
            title=stat,
        )
    )
    fig = Figure(data=data, layout=layout)
    figure_name = re.sub('[^0-9a-zA-Z]+','_',stat) + "_" + options.plotname
    print("plotting: " + figure_name)
    outdir = (os.path.join(this_directory,"htmls"))
    if not os.path.exists( outdir ):
        os.makedirs(outdir)
    plotly.offline.plot(fig, filename=os.path.join(outdir,figure_name + ".html"),auto_open=False)
    stat_count += 1


shutil.copy2(options.csv_file, outdir)
if options.publish_path != None and options.publish_path != "":
    files = glob.glob(os.path.join(outdir, "*"))
    if subprocess.call(["scp"] + files + [options.publish_path]) != 0:
        print("Error Publishing via scp")
    else:
        print("Successfully pushed results to: {0}".format(options.publish_web))
