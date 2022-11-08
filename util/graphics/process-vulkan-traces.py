#! /usr/bin/python3
import os

file = "/home/pan251/Vulkan-Samples/traces.traceg"
folder = "../../hw_run/traces/vulkan/RENAME_ME/NO_ARGS/traces/"

trace = open(file, 'r')
lines = trace.readlines()
warp_range = []
max_warp = 0
total = []
kernel_name = []
block_dim = 2
big_str=[]
counter = 0

file = folder + "kernelslist.g"
os.system("mkdir -p " + folder)
infof = open(file, "w")
warp_range.append(0)
for line in lines:
    if 'graphics kernel end:' in line:
        warp_range.append(max_warp+1)
        kernel_name.append(line.split(': ')[1].replace("\n", ""))
        continue
    if 'MemcpyHtoD' in line:
        continue
    substr = line.split(', ')
    assert(len(substr) == 2)
    warp_id = int(substr[0])
    inst = substr[1]
    if warp_id > max_warp:
        max_warp = warp_id

block_id = 0
for dumb in range(0,max_warp+1):
    # num_warps_kernel[n] is number of warps
    # append big_str for each warp
    big_str.append([])

# start parsing!
index_k = 0
for line in lines:
    if 'MemcpyHtoD' in line:
        infof.write(line)
        continue
    if 'graphics kernel end:' in line:
        file = folder + "kernel-" + kernel_name[index_k] + "_" + str(counter) + ".traceg"
        infof.write("kernel-" + kernel_name[index_k] + "_" + str(counter) + ".traceg\n")
        counter = counter + 1
        print(file)
        f = open(file, "w")

        # write kernel info
        f.write("-kernel name = " + kernel_name[index_k] + "\n")
        f.write("-kernel id = " + str(index_k) + "\n")
        f.write("-grid dim = (" + str(int((warp_range[index_k+1]-warp_range[index_k])/2)) + ",1,1)\n")
        f.write("-block dim = (64,1,1)\n")
        f.write("-shmem = 0\n")
        f.write("-nregs = /*UPDATE_ME*/\n")
        f.write("-binary version = 80\n")
        f.write("-cuda stream id = " + str(index_k) + "\n")
        f.write("-shmem base_addr = 0xffffffff\n")
        f.write("-local mem base_addr = 0xc0000000\n")
        f.write("-nvbit version = 1.5.3\n")
        f.write("-accelsim tracer version = 3\n")
        f.write("#traces format = threadblock_x threadblock_y threadblock_z warpid_tb PC mask dest_num [reg_dests] opcode src_num [reg_srcs] mem_width [adrrescompress?] [mem_addresses]\n")

        warp_count = 0
        # 17325
        for warp_id in range(warp_range[index_k],warp_range[index_k+1]):
            if warp_count % block_dim == 0 or warp_count == 0:
                f.write("\n#BEGIN_TB\n\n")
                f.write("thread block = " + str(int((warp_id-warp_range[index_k])/block_dim)) + ",0,0\n")
            f.write("\nwarp = " + str(warp_count) + "\n")
            f.write("insts = " + str(len(big_str[warp_id])) + "\n")
            for inst in big_str[warp_id]:
                f.write(inst)
            if warp_count % block_dim == 1:
                f.write("\n#END_TB\n\n")
                warp_count = 0
            else:
                warp_count = warp_count + 1
        f.close()
        index_k = index_k + 1
        continue
    substr = line.split(', ')
    assert(len(substr) == 2)
    warp_id = int(substr[0])
    inst = substr[1]
    big_str[warp_id].append(inst)

# make sure we printed all kernel
assert(index_k == len(kernel_name))