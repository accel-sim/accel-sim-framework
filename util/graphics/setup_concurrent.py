#! /usr/bin/python3

import os

cwd = os.path.dirname(os.path.realpath(__file__)) + "/"
trace_dir = cwd + "../../hw_run/traces/vulkan/"

gs = ["pbrtexture_2k","pbrtexture_4k", "render_passes_2k", "render_passes_4k", "instancing_2k","instancing_4k", "sponza_2k", "sponza_4k", "materials_2k", "materials_4k", "platformer_2k", "platformer_4k", "demo_2k", "demo_4k"]
cs = ["vpi_sample_03_harris_corners", "klt_tracker", "vpi_sample_11_fisheye", "vpi_sample_12_optflow_lk_refined"]
css = ["ritnet", "hotlab", 
    # "slam_lidar_steady", "slam_rgbd_steady"
]
vio = 1
# if vio != 1:
all_name = "all" + str(vio)
# else:
    # all_name = "all"

# to clean up
for g in gs:
    for c in cs + css:
        if os.path.exists(trace_dir + g + "/" + c):
            os.system("rm -rf " + trace_dir + g + "/" + c)
    # rm all
    if os.path.exists(trace_dir + g + "/" + all_name):
        os.system("rm -rf " + trace_dir + g + "/" + all_name)

# exit()

for g in gs:
    if os.path.exists(trace_dir + g + "/" + all_name):
        print("skipping " + g + "/" + all_name)
    else:
        os.makedirs(trace_dir + g + "/" + all_name + "/traces", exist_ok=True)
        # copy over traces
        os.system("ln -s " + trace_dir + g + "/NO_ARGS/traces/*.traceg " + trace_dir + g + "/" + all_name + "/traces/")
        # copy over kernelslist.g
        os.system("cp " + trace_dir + g + "/NO_ARGS/traces/kernelslist.g " + trace_dir + g + "/" + all_name + "/traces/")
    for c in cs + css:
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

# write compute kernels to each cs kernelslist.g
    for c in cs:
        # get sub dir of c
        sub_dir = trace_dir + c + "/" + os.listdir(trace_dir + c + "/")[0]
        # read in file traceg
        
        kernelslist = open(trace_dir + g + "/" + c + "/traces/kernelslist.g", "a+")
        
        # add lines to kernelslist.g
        kernelslist.write("\n")
        kernelslist_c = open(sub_dir + "/traces/kernelslist.g", "r")
        for line in kernelslist_c:
            kernelslist.write(c + "-" + line)
        kernelslist_c.close()
        kernelslist.close()

# write compute kernels (up to n times) to all kernelslist.g
    for i in range(0,vio):
        for c in cs:
            sub_dir = trace_dir + c + "/" + os.listdir(trace_dir + c + "/")[0]
            kernelslist_all = open(trace_dir + g + "/" + all_name + "/traces/kernelslist.g", "a+")
            kernelslist_all.write("\n")
            kernelslist_c = open(sub_dir + "/traces/kernelslist.g", "r")
            for line in kernelslist_c:
                kernelslist_all.write(c + "-" + line)
            kernelslist_all.close()

# link the compute trace files
    for c in cs:
        sub_dir = trace_dir + c + "/" + os.listdir(trace_dir + c + "/")[0]
        # copy over files in sub_dir and rename
        for file in os.listdir(sub_dir + "/traces"):
            if file == "kernelslist.g" or file == "stats.csv":
                continue
            os.system("ln -s " + sub_dir + "/traces/" + file + " " + trace_dir + g + "/" + c + "/traces/" + c + "-" + file)
            os.system("ln -s " + sub_dir + "/traces/" + file + " " + trace_dir + g + "/" + all_name + "/" + "/traces/" + c + "-" + file)

# write compute kernels to each css kernelslist.g (not part of all)
    for c in css:
        # get sub dir of c
        sub_dir = trace_dir + c + "/" + os.listdir(trace_dir + c + "/")[0]
        # read in file traceg
        
        kernelslist = open(trace_dir + g + "/" + c + "/traces/kernelslist.g", "a+")
        
        # add lines to kernelslist.g
        kernelslist.write("\n")
        kernelslist_c = open(sub_dir + "/traces/kernelslist.g", "r")
        for line in kernelslist_c:
            kernelslist.write(c + "-" + line)
        kernelslist_c.close()
        kernelslist.close()
        for file in os.listdir(sub_dir + "/traces"):
            if file == "kernelslist.g" or file == "stats.csv":
                continue
            os.system("ln -s " + sub_dir + "/traces/" + file + " " + trace_dir + g + "/" + c + "/traces/" + c + "-" + file)
            # os.system("ln -s " + sub_dir + "/traces/" + file + " " + trace_dir + g + "/" + all_name + "/" + "/traces/" + c + "-" + file)