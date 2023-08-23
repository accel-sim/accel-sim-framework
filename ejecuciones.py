# programa que ejecuta varios programas

import os

directorios = "backprop-rodinia-2.0-ft/4096___data_result_4096_txt bfs-rodinia-2.0-ft/__data_graph4096_txt___data_graph4096_result_txt heartwall-rodinia-2.0-ft/__data_test_avi_1___data_result_1_txthotspot-rodinia-2.0-ft/30_6_40___data_result_30_6_40_txt kmeans-rodinia-2.0-ft/_i_data_400_txt__g_data_400_result_txt__o lud-rodinia-2.0-ft/_v__b__i___data_64_dat nn-rodinia-2.0-ft/__data_filelist_4_3_30_90___data_filelist_4_3_30_90_result_txt nw-rodinia-2.0-ft/128_10___data_result_128_10_txt pathfinder-rodinia-2.0-ft/1000_20_5___data_result_1000_20_5_txt srad_v2-rodinia-2.0-ft/__data_matrix128x128_txt_0_127_0_127__5_2___data_result_matrix128x128_1_150_1_100__5_2_txt streamcluster-rodinia-2.0-ft/3_6_16_1024_1024_100_none_output_txt_1___data_result_3_6_16_1024_1024_100_none_1_txt" 
directorios = directorios.split(" ")
for i in directorios:
    os.system("./gpu-simulator/bin/release/accel-sim.out -trace ./hw_run/rodinia_2.0-ft/9.1/" + i + "/traces/kernelslist.g -config ./gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100/gpgpusim.config -config ./gpu-simulator/configs/tested-cfgs/SM7_QV100/trace.config > executions_stadistics/rodinia/" + i.split("/")[0] + ".txt")
print("Rodinia ejecutado")