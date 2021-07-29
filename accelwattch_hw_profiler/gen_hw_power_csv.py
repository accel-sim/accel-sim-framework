import csv
import numpy
import os 
import collections 
import sys
import pandas as pd
from os.path import dirname, basename, isfile, join

ubenchmarks = []
basedir = os.getcwd();
reportspath = basedir + '/collated_power'

for reportfile in os.listdir(reportspath):
	ubenchmarks.append(reportfile[:-4])

power_dict = collections.OrderedDict()
for benchmark in ubenchmarks :
	f = open((reportspath + '/' + benchmark + '.rpt'), 'r')
	x = f.readlines()
	samples = []
	mean = 0
	for each in x:
		if ((float(each)) == 0): #ignore erronous datapoint- 0Watts
			continue
	 	samples.append(float(each))
	datapoints = numpy.array(samples)
	power_dict[benchmark] = collections.OrderedDict()
	power_dict[benchmark]["#samples"] = datapoints.size
	if datapoints.size == 0:
		print 'No Data for :' + benchmark 
	else:
		power_dict[benchmark]["mean HW_power"] = datapoints.mean()
		power_dict[benchmark]["st_dev"] = datapoints.std()
		power_dict[benchmark]["var"] = datapoints.var()

df = pd.DataFrame.from_dict(power_dict, orient='index')
cwd = 'hw_power_results.csv'
df.to_csv(cwd)
