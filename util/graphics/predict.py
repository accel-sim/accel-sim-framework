#! /usr/bin/env python3

import pandas as pd
import numpy as np
import os



### Loop the data lines
with open("render_passes_series.csv", 'r') as temp_f:
    # get No of columns in each line
    col_count = [ len(l.split(",")) for l in temp_f.readlines() ]

temp_f.close()
### Generate column names  (names will be 0, 1, 2, ..., maximum columns - 1)
column_names = [i for i in range(0, max(col_count))]
    
file = pd.read_csv("render_passes_series.csv", header=None, delimiter=",", names=column_names)
file = file.fillna(0)
file.columns = file.iloc[0]
file.set_index('count', inplace=True)
file = file.drop(file.index[0]).astype(int)
# print(file)

# average of last 3 frames
print("Average of last 3 frames")
avg = []
linear_reg = []
linear_reg_5 = []
for frame in range(3,file.shape[0]):
    frame_cycle = 0
    real_cycle = 0
    for draw in range(0,24):
        predicted_cycle = (file.iloc[frame-1,draw] + file.iloc[frame-2,draw] + file.iloc[frame-3,draw])/3
        frame_cycle = frame_cycle + predicted_cycle
        real_cycle = real_cycle + file.iloc[frame,draw]
    print("Frame: ", frame+1, " Cycle: ", frame_cycle, " Real: ", real_cycle, " percent error: ", "%.2f" % ((frame_cycle-real_cycle)/real_cycle*100), "%")
    avg.append((frame_cycle-real_cycle)/real_cycle*100)

# linear regression
print("Linear Regression")
for frame in range(3,file.shape[0]):
    frame_cycle = 0
    real_cycle = 0
    for draw in range(0,24):
        y = np.array([file.iloc[frame-3,draw], file.iloc[frame-2,draw], file.iloc[frame-1,draw]])
        x = np.array([1, 2, 3])
        m,b = np.polyfit(x, y, 1)
        predicted_cycle = m*4 +b
        frame_cycle = frame_cycle + predicted_cycle
        real_cycle = real_cycle + file.iloc[frame,draw]
        # print("predicted: ", predicted_cycle)
        # print(str((predicted_cycle-file.iloc[frame,draw])/file.iloc[frame,draw]) + ", ", end='')
    # print("")

    print("Frame: ", frame+1, " Cycle: ", frame_cycle, " Real: ", real_cycle, " percent error: ", "%.2f" % ((frame_cycle-real_cycle)/real_cycle*100), "%")
    linear_reg.append((frame_cycle-real_cycle)/real_cycle*100)