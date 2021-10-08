import csv
import numpy
import os 
import collections 
import sys
import pandas as pd
from os.path import dirname, basename, isfile, join

# Copyright (c) 2018-2021, Vijay Kandiah, Junrui Pan, Mahmoud Khairy, Scott Peverelle, Timothy Rogers, Tor M. Aamodt, Nikos Hardavellas
# Northwestern University, Purdue University, The University of British Columbia
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer;
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution;
# 3. Neither the names of Northwestern University, Purdue University,
#    The University of British Columbia nor the names of their contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

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

	if datapoints.size == 0:
		print 'No Data for :' + benchmark 
	else:
		power_dict[benchmark]["mean HW_power"] = datapoints.mean()
		power_dict[benchmark]["st_dev"] = datapoints.std()
		power_dict[benchmark]["var"] = datapoints.var()
		power_dict[benchmark]["#samples"] = datapoints.size
df = pd.DataFrame.from_dict(power_dict, orient='index')
cwd = 'hw_power_results.csv'
df.to_csv(cwd)