#! /usr/bin/python3

import os

cwd = os.path.dirname(os.path.realpath(__file__)) + "/"
trace_dir = cwd + "../../hw_run/traces/vulkan/"

gs = ["pbrtexture_2k","pbrtexture_4k", "render_passes_2k", "render_passes_4k", "instancing_2k","instancing_4k","render_passes_dev"]
cs = ["vpi_sample_03_harris_corners", "klt_tracker", "vpi_sample_11_fisheye", "vpi_sample_12_optflow_lk_refined"]

# to clean up
for g in gs:
    for c in cs:
        if os.path.exists(trace_dir + g + "/" + c):
            os.system("rm -rf " + trace_dir + g + "/" + c)
    # rm all
    if os.path.exists(trace_dir + g + "/" + "all"):
        os.system("rm -rf " + trace_dir + g + "/" + "all")

# exit()

for g in gs:
    if os.path.exists(trace_dir + g + "/" + "all"):
        print("skipping " + g + "/" + "all")
    else:
        os.makedirs(trace_dir + g + "/" + "all/traces", exist_ok=True)
        # copy over traces
        os.system("ln -s " + trace_dir + g + "/NO_ARGS/traces/*.traceg " + trace_dir + g + "/" + "all" + "/traces/")
        # copy over kernelslist.g
        os.system("cp " + trace_dir + g + "/NO_ARGS/traces/kernelslist.g " + trace_dir + g + "/" + "all" + "/traces/")
    for c in cs:
        # check if dir exits
        if os.path.exists(trace_dir + g + "/" + c):
            print("skipping " + g + "/" + c)
        else:
            print("creating " + g + "/" + c)
            os.makedirs(trace_dir + g + "/" + c + "/traces", exist_ok=True)
            # copy over traces
            os.system("ln -s " + trace_dir + g + "/NO_ARGS/traces/*.traceg " + trace_dir + g + "/" + c + "/traces/")
            # copy over kernelslist.g
            os.system("cp " + trace_dir + g + "/NO_ARGS/traces/kernelslist.g " + trace_dir + g + "/" + c + "/traces/")

        # get sub dir of c
        sub_dir = trace_dir + c + "/" + os.listdir(trace_dir + c + "/")[0]
        # read in file traceg
        
        kernelslist = open(trace_dir + g + "/" + c + "/traces/kernelslist.g", "a+")
        kernelslist_all = open(trace_dir + g + "/" + "all" + "/traces/kernelslist.g", "a+")
        # add lines to kernelslist.g
        for i in range(0, 1):
            kernelslist.write("\n")
            kernelslist_all.write("\n")
            kernelslist_c = open(sub_dir + "/traces/kernelslist.g", "r")
            for line in kernelslist_c:
                kernelslist.write(c + "-" + line)
                kernelslist_all.write(c + "-" + line)
            kernelslist_c.close()
        kernelslist.close()
        kernelslist_all.close()
        # copy over files in sub_dir and rename
        for file in os.listdir(sub_dir + "/traces"):
            if file == "kernelslist.g" or file == "stats.csv":
                continue
            os.system("ln -s " + sub_dir + "/traces/" + file + " " + trace_dir + g + "/" + c + "/traces/" + c + "-" + file)
            os.system("ln -s " + sub_dir + "/traces/" + file + " " + trace_dir + g + "/" + "all/" + "/traces/" + c + "-" + file)


