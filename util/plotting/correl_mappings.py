# This file is eval'd inside the plot-correlation.py file

# This maps the named GPGPU-Sim config to the card name reported in the nvprof file.
#   Every time you want to correlate a new configuration, you need to map it here.
config_maps = \
{
    "TITANX": set("TITAN X (Pascal)"),
    "P100_HBM" : set("Tesla P100"),
    "GTX480" : set("GeForce GTX 480"),
    "GTX1080Ti" : set("GeForce GTX 1080 Ti"),
    "TITANK" : set("GeForce GTX TITAN"),
    "QV100" : set(["TITAN V", "Quadro GV100","Tesla V100-SXM2-32GB"]),
    "RTX2060" : set("GeForce RTX 2060"),
    "RTX3070" : set("GeForce RTX 3070"),
}


# Every stat you want to correlate gets an entry here.
#   For cycles, the math is different for every card so we have differnt stats baed on the hardware.
import collections
CorrelStat = collections.namedtuple('CorrelStat', 'chart_name hw_eval hw_error sim_eval hw_name plotfile drophwnumbelow plottype stattype')
correl_list = \
[
    # 1200 MHz
    CorrelStat(chart_name="Cycles",
        plotfile="titanv-cycles",
        hw_eval="np.average(hw[\"Duration\"])*1200",
        hw_error="np.max(hw[\"Duration\"])*1200 - np.average(hw[\"Duration\"])*1200,"+\
                 "np.average(hw[\"Duration\"])*1200 - np.min(hw[\"Duration\"])*1200",
        sim_eval="float(sim[\"gpu_tot_sim_cycle\s*=\s*(.*)\"])",
        hw_name="TITAN V",
        drophwnumbelow=0,
		plottype="log",
        stattype="counter"
    ),
    # 1417 MHz
    CorrelStat(chart_name="Cycles",
        plotfile="titanx-p102-cycles",
        hw_eval="np.average(hw[\"Duration\"])*1417",
        hw_error="np.max(hw[\"Duration\"])*1417 - np.average(hw[\"Duration\"])*1417,"+\
                 "np.average(hw[\"Duration\"])*1417 - np.min(hw[\"Duration\"])*1417",
        sim_eval="float(sim[\"gpu_tot_sim_cycle\s*=\s*(.*)\"])",
        hw_name="TITAN X (Pascal)",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
    # (1400 MHz - 16-wide SIMD)
    CorrelStat(chart_name="Cycles",
        plotfile="gtx480-cycles",
        hw_eval="np.average(hw[\"Duration\"])*1400",
        hw_error="np.max(hw[\"Duration\"])*1400 - np.average(hw[\"Duration\"])*1400,"+\
                 "np.average(hw[\"Duration\"])*1400 - np.min(hw[\"Duration\"])*1400",
        sim_eval="float(sim[\"gpu_tot_sim_cycle\s*=\s*(.*)\"])*2",
        hw_name="GeForce GTX 480",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
    # 1480 MHz
    CorrelStat(chart_name="Cycles",
        plotfile="p100-cycles",
        hw_eval="np.average(hw[\"Duration\"])*1480",
        hw_error="np.max(hw[\"Duration\"])*1480 - np.average(hw[\"Duration\"])*1480,"+\
                 "np.average(hw[\"Duration\"])*1480 - np.min(hw[\"Duration\"])*1480",
        sim_eval="float(sim[\"gpu_tot_sim_cycle\s*=\s*(.*)\"])",
        hw_name="Tesla P100",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
    # 1480 MHz
    CorrelStat(chart_name="Cycles",
        plotfile="1080ti-cycles",
        hw_eval="np.average(hw[\"Duration\"])*1480",
        hw_error="np.max(hw[\"Duration\"])*1480 - np.average(hw[\"Duration\"])*1480,"+\
                 "np.average(hw[\"Duration\"])*1480 - np.min(hw[\"Duration\"])*1480",
        sim_eval="float(sim[\"gpu_tot_sim_cycle\s*=\s*(.*)\"])",
        hw_name="GeForce GTX 1080 Ti",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
    # 1132 MHz
    CorrelStat(chart_name="Cycles",
        plotfile="gv100-cycles",
        hw_eval="np.average(hw[\"Duration\"])*1132",
        hw_error="np.max(hw[\"Duration\"])*1132 - np.average(hw[\"Duration\"])*1132,"+\
                 "np.average(hw[\"Duration\"])*1132 - np.min(hw[\"Duration\"])*1132",
        sim_eval="float(sim[\"gpu_tot_sim_cycle\s*=\s*(.*)\"])",
        hw_name="Quadro GV100",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
    CorrelStat(chart_name="Cycles",
        plotfile="qv100_sm_cycles",
        hw_eval="np.average(hw[\"elapsed_cycles_sm\"])/80",
        hw_error="np.max(hw[\"elapsed_cycles_sm\"])/80 - np.average(hw[\"elapsed_cycles_sm\"])/80,"+\
                 "np.average(hw[\"elapsed_cycles_sm\"])/80 - np.min(hw[\"elapsed_cycles_sm\"])/80",
        sim_eval="float(sim[\"gpu_tot_sim_cycle\s*=\s*(.*)\"])",
        hw_name="Quadro GV100",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
    CorrelStat(chart_name="GPC Cycles",
        plotfile="gpc_cycles",
        hw_eval="np.average(hw[\"gpc__cycles_elapsed.avg\"])",
        hw_error="np.max(hw[\"gpc__cycles_elapsed.avg\"]) - np.average(hw[\"gpc__cycles_elapsed.avg\"]),"+\
                 "np.average(hw[\"gpc__cycles_elapsed.avg\"]) - np.min(hw[\"gpc__cycles_elapsed.avg\"])",
        sim_eval="float(sim[\"gpu_tot_sim_cycle\s*=\s*(.*)\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
	# 1455 MHz
    CorrelStat(chart_name="Cycles",
        plotfile="tv100-cycles",
        hw_eval="np.average(hw[\"Duration\"])*1455",
        hw_error="np.max(hw[\"Duration\"])*1455 - np.average(hw[\"Duration\"])*1455,"+\
                 "np.average(hw[\"Duration\"])*1455 - np.min(hw[\"Duration\"])*1455",
        sim_eval="float(sim[\"gpu_tot_sim_cycle\s*=\s*(.*)\"])",
        hw_name="Tesla V100-SXM2-32GB",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
    CorrelStat(chart_name="TESLA V100 SM Cycles",
        plotfile="tv100_sm_cycles",
        hw_eval="np.average(hw[\"elapsed_cycles_sm\"])/80",
        hw_error="np.max(hw[\"elapsed_cycles_sm\"])/80 - np.average(hw[\"elapsed_cycles_sm\"])/80,"+\
                 "np.average(hw[\"elapsed_cycles_sm\"])/80 - np.min(hw[\"elapsed_cycles_sm\"])/80",
        sim_eval="float(sim[\"gpu_tot_sim_cycle\s*=\s*(.*)\"])",
        hw_name="Tesla V100-SXM2-32GB",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
	#837 MHZ
    CorrelStat(chart_name="Cycles",
        plotfile="kepler-cycles",
        hw_eval="np.average(hw[\"Duration\"])*837",
        hw_error="np.max(hw[\"Duration\"])*837 - np.average(hw[\"Duration\"])*837,"+\
                 "np.average(hw[\"Duration\"])*837 - np.min(hw[\"Duration\"])*837",
        sim_eval="float(sim[\"gpu_tot_sim_cycle\s*=\s*(.*)\"])",
        hw_name="GeForce GTX TITAN",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
    CorrelStat(chart_name="TITAN KEPLER Cycles",
        plotfile="kepler_sm_cycles",
        hw_eval="np.average(hw[\"elapsed_cycles_sm\"])/14",
        hw_error="np.max(hw[\"elapsed_cycles_sm\"])/14 - np.average(hw[\"elapsed_cycles_sm\"])/14,"+\
                 "np.average(hw[\"elapsed_cycles_sm\"])/14 - np.min(hw[\"elapsed_cycles_sm\"])/14",
        sim_eval="float(sim[\"gpu_tot_sim_cycle\s*=\s*(.*)\"])",
        hw_name="GeForce GTX TITAN",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
	#Turing
    CorrelStat(chart_name="TITAN TURING Cycles",
        plotfile="turing_sm_cycles",
        hw_eval="np.average(hw[\"gpc__cycles_elapsed.avg\"])",
        hw_error="np.max(hw[\"gpc__cycles_elapsed.avg\"]) - np.average(hw[\"gpc__cycles_elapsed.avg\"]),"+\
                 "np.average(hw[\"gpc__cycles_elapsed.avg\"]) - np.min(hw[\"gpc__cycles_elapsed.avg\"])",
        sim_eval="float(sim[\"gpu_tot_sim_cycle\s*=\s*(.*)\"])",
        hw_name="GeForce RTX 2060",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),

    # Common, non-cycle stats for nvprof
    CorrelStat(chart_name="Warp Instructions",
        plotfile="warp-inst",
        hw_eval="np.average(hw[\"inst_issued\"])",
        hw_error=None,
        sim_eval="float(sim[\"gpgpu_n_tot_w_icount\s*=\s*(.*)\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
    CorrelStat(chart_name="L2 Read Hits",
        plotfile="l2-read-hits",
        hw_eval="np.average(hw[\"l2_tex_read_transactions\"])*np.average(hw[\"l2_tex_read_hit_rate\"])/100",
        hw_error=None,
        sim_eval="float(sim[\"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_R\]\[HIT\]\s*=\s*(.*)\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
    CorrelStat(chart_name="L2 Reads",
        plotfile="l2-read-transactions",
        hw_eval="np.average(hw[\"l2_tex_read_transactions\"])",
        hw_error=None,
        sim_eval="float(sim[\"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_R\]\[TOTAL_ACCESS\]\s*=\s*(.*)\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
    CorrelStat(chart_name="L2 Writes",
        plotfile="l2-write-transactions",
        hw_eval="np.average(hw[\"l2_tex_write_transactions\"])",
        hw_error=None,
        sim_eval="float(sim[\"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_W\]\[TOTAL_ACCESS\]\s*=\s*(.*)\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
    CorrelStat(chart_name="L2 Write Hits",
        plotfile="l2-write-hits",
        hw_eval="np.average(hw[\"l2_tex_write_transactions\"]) * np.average(hw[\"l2_tex_write_hit_rate\"]) / 100.0",
        hw_error=None,
        sim_eval="float(sim[\"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_W\]\[HIT\]\s*=\s*(.*)\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
    CorrelStat(chart_name="L2 BW",
        plotfile="l2_bw",
        hw_eval="np.average(hw[\"l2_tex_read_throughput\"])",
        hw_error=None,
        sim_eval="float(sim[\"L2_BW\s*=\s*(.*)GB\/Sec\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="linear",
        stattype="rate"
    ),
    CorrelStat(chart_name="L2 Read Hit Rate",
        plotfile="l2-read-hitrate",
        hw_eval="np.average(hw[\"l2_tex_read_hit_rate\"])",
        hw_error=None,
        sim_eval=
            "100*float(sim[\"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_R\]\[HIT\]\s*=\s*(.*)\"])/"+\
            "float(sim[\"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_R\]\[TOTAL_ACCESS\]\s*=\s*(.*)\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="linear",
        stattype="rate"
    ),
    CorrelStat(chart_name="L2 Write Hit Rate",
        plotfile="l2-write-hitrate",
        hw_eval="np.average(hw[\"l2_tex_write_hit_rate\"])",
        hw_error=None,
        sim_eval=
            "100*float(sim[\"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_W\]\[HIT\]\s*=\s*(.*)\"])/"+\
            "float(sim[\"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_W\]\[TOTAL_ACCESS\]\s*=\s*(.*)\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="linear",
        stattype="rate"
    ),
    CorrelStat(chart_name="Occupancy",
        plotfile="occupancy",
        hw_eval="np.average(hw[\"achieved_occupancy\"])*100",
        hw_error=None,
        sim_eval="float(sim[\"gpu_occupancy\s*=\s*(.*)%\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="linear",
        stattype="rate"
    ),
    CorrelStat(chart_name="L1D Hit Rate",
        plotfile="l1hitrate",
        hw_eval="np.average(hw[\"tex_cache_hit_rate\"])",
        hw_error=None,
        sim_eval="float(sim[\"\s+Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[HIT\]\s*=\s*(.*)\"])" +\
                 "/(float(sim[\"\s+Total_core_cache_stats_breakdown\[GLOBAL_ACC_W\]\[TOTAL_ACCESS\]\s*=\s*(.*)\"])" +\
                 "+float(sim[\"\s+Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[TOTAL_ACCESS\]\s*=\s*(.*)\"]) + 1) * 100",
        hw_name="all",
        drophwnumbelow=0,
        plottype="linear",
        stattype="rate"
    ),
    CorrelStat(chart_name="L1D Hit Rate (global_hit_rate match)",
        plotfile="l1hitrate.global",
        hw_eval="np.average(hw[\"global_hit_rate\"])",
        hw_error=None,
        sim_eval="(float(sim[\"\s+Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[HIT\]\s*=\s*(.*)\"])" +\
                " + float(sim[\"\s+Total_core_cache_stats_breakdown\[GLOBAL_ACC_W\]\[HIT\]\s*=\s*(.*)\"]))" +\
                 "/(float(sim[\"\s+Total_core_cache_stats_breakdown\[GLOBAL_ACC_W\]\[TOTAL_ACCESS\]\s*=\s*(.*)\"])" +\
                 "+float(sim[\"\s+Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[TOTAL_ACCESS\]\s*=\s*(.*)\"]) + 1) * 100",
        hw_name="all",
        drophwnumbelow=0,
        plottype="linear",
        stattype="rate"
    ),
    CorrelStat(chart_name="L1D Reads",
        plotfile="l1readaccess",
        hw_eval="np.average(hw[\"gld_transactions\"])",
        hw_error=None,
        sim_eval="float(sim[\"\s+Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[TOTAL_ACCESS\]\s*=\s*(.*)\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
	CorrelStat(chart_name="L1 BW",
        plotfile="l1_bw",
        hw_eval="np.average(hw[\"tex_cache_throughput\"])",
        hw_error=None,
        sim_eval="((float(sim[\"\s+Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[TOTAL_ACCESS\]\s*=\s*(.*)\"])" +\
                " + float(sim[\"\s+Total_core_cache_stats_breakdown\[GLOBAL_ACC_W\]\[TOTAL_ACCESS\]\s*=\s*(.*)\"])) * 32 * 1.132)/" +\
				"float(sim[\"gpu_tot_sim_cycle\s*=\s*(.*)\"])",
        hw_name="Quadro GV100",
        drophwnumbelow=0,
        plottype="linear",
        stattype="rate"
    ),
	CorrelStat(chart_name="DRAM Reads",
        plotfile="dram-read-transactions",
        hw_eval="np.average(hw[\"dram_read_transactions\"])",
        hw_error=None,
#        sim_eval="float(sim[\"Read\s*=\s*(.*)\"])+float(sim[\"L2_Alloc\s*=\s*(.*)\"])*24",
        sim_eval="float(sim[\"total dram reads\s*=\s*(.*)\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
	CorrelStat(chart_name="DRAM Writes",
        plotfile="dram-write-transactions",
        hw_eval="np.average(hw[\"dram_write_transactions\"])",
        hw_error=None,
        sim_eval="float(sim[\"total dram writes\s*=\s*(.*)\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
	
	
	 ### Common, non-cycle stats for nvsight
    CorrelStat(chart_name="Warp Instructions",
        plotfile="warp-inst",
        hw_eval="np.average(hw[\"smsp__inst_executed.sum\"])",
        hw_error=None,
        sim_eval="float(sim[\"gpgpu_n_tot_w_icount\s*=\s*(.*)\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
    CorrelStat(chart_name="L2 Read Hits",
        plotfile="l2-read-hits",
        hw_eval="np.average(hw[\"lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum\"])",
        hw_error=None,
        sim_eval="float(sim[\"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_R\]\[HIT\]\s*=\s*(.*)\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
    CorrelStat(chart_name="L2 Reads",
        plotfile="l2-read-transactions",
        hw_eval="np.average(hw[\"lts__t_sectors_srcunit_tex_op_read.sum\"])",
        hw_error=None,
        sim_eval="float(sim[\"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_R\]\[TOTAL_ACCESS\]\s*=\s*(.*)\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
    CorrelStat(chart_name="L2 Writes",
        plotfile="l2-write-transactions",
        hw_eval="np.average(hw[\"lts__t_sectors_srcunit_tex_op_write.sum\"])",
        hw_error=None,
        sim_eval="float(sim[\"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_W\]\[TOTAL_ACCESS\]\s*=\s*(.*)\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
    CorrelStat(chart_name="L2 Write Hits",
        plotfile="l2-write-hits",
        hw_eval="np.average(hw[\"lts__t_sectors_srcunit_tex_op_write_lookup_hit.sum\"])",
        hw_error=None,
        sim_eval="float(sim[\"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_W\]\[HIT\]\s*=\s*(.*)\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
    CorrelStat(chart_name="L2 BW",
        plotfile="l2_bw",
        hw_eval="np.average(hw[\"lts__t_sectors_srcunit_tex_op_read.sum.per_second\"] * 32)",
        hw_error=None,
        sim_eval="float(sim[\"L2_BW\s*=\s*(.*)GB\/Sec\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="linear",
        stattype="rate"
    ),
    CorrelStat(chart_name="L2 Read Hit Rate",
        plotfile="l2-read-hitrate",
        hw_eval="np.average(hw[\"lts__t_sector_op_read_hit_rate.pct\"])",
        hw_error=None,
        sim_eval=
            "100*float(sim[\"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_R\]\[HIT\]\s*=\s*(.*)\"])/"+\
            "float(sim[\"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_R\]\[TOTAL_ACCESS\]\s*=\s*(.*)\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="linear",
        stattype="rate"
    ),
    CorrelStat(chart_name="L2 Write Hit Rate",
        plotfile="l2-write-hitrate",
        hw_eval="np.average(hw[\"lts__t_sector_op_write_hit_rate.pct\"])",
        hw_error=None,
        sim_eval=
            "100*float(sim[\"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_W\]\[HIT\]\s*=\s*(.*)\"])/"+\
            "float(sim[\"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_W\]\[TOTAL_ACCESS\]\s*=\s*(.*)\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="linear",
        stattype="rate"
    ),
    CorrelStat(chart_name="Occupancy",
        plotfile="occupancy",
        hw_eval="np.average(hw[\"sm__warps_active.avg.pct_of_peak_sustained_active\"])",
        hw_error=None,
        sim_eval="float(sim[\"gpu_occupancy\s*=\s*(.*)%\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="linear",
        stattype="rate"
    ),
    CorrelStat(chart_name="L1D Read Hits",
        plotfile="l1hitreads",
        hw_eval="np.average(hw[\"l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum\"])",
        hw_error=None,
        sim_eval="float(sim[\"\s+Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[HIT\]\s*=\s*(.*)\"])" +\
                "+float(sim[\"\s+Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[MSHR_HIT\]\s*=\s*(.*)\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
    CorrelStat(chart_name="L1D Write Hits",
        plotfile="l1hitwrites",
        hw_eval="np.average(hw[\"l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum\"])",
        hw_error=None,
        sim_eval="float(sim[\"\s+Total_core_cache_stats_breakdown\[GLOBAL_ACC_W\]\[HIT\]\s*=\s*(.*)\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
    CorrelStat(chart_name="L1D Read Access",
        plotfile="l1readaccess",
        hw_eval="np.average(hw[\"l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum\"])",
        hw_error=None,
        sim_eval="float(sim[\"\s+Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[TOTAL_ACCESS\]\s*=\s*(.*)\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
	CorrelStat(chart_name="L1D Write Access",
        plotfile="l1writeaccess",
        hw_eval="np.average(hw[\"l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum\"])",
        hw_error=None,
        sim_eval="float(sim[\"\s+Total_core_cache_stats_breakdown\[GLOBAL_ACC_W\]\[TOTAL_ACCESS\]\s*=\s*(.*)\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
	CorrelStat(chart_name="DRAM Reads",
        plotfile="dram-read-transactions",
        hw_eval="np.average(hw[\"dram__sectors_read.sum\"])",
        hw_error=None,
#        sim_eval="float(sim[\"Read\s*=\s*(.*)\"])+float(sim[\"L2_Alloc\s*=\s*(.*)\"])*24",
        sim_eval="float(sim[\"total dram reads\s*=\s*(.*)\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),
	CorrelStat(chart_name="DRAM Writes",
        plotfile="dram-write-transactions",
        hw_eval="np.average(hw[\"dram__sectors_write.sum\"])",
        hw_error=None,
        sim_eval="float(sim[\"total dram writes\s*=\s*(.*)\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="log",
        stattype="counter"
    ),

    CorrelStat(chart_name="IPC",
        plotfile="ipc",
        hw_eval="np.average(hw[\"inst_issued\"])/(np.average(hw[\"elapsed_cycles_sm\"])/80)",
        hw_error="np.average(hw[\"inst_issued\"])/(np.max(hw[\"elapsed_cycles_sm\"])/80) - np.average(hw[\"inst_issued\"])/(np.average(hw[\"elapsed_cycles_sm\"])/80),"+\
                 "np.average(hw[\"inst_issued\"])/(np.average(hw[\"elapsed_cycles_sm\"])/80) - np.average(hw[\"inst_issued\"])/(np.min(hw[\"elapsed_cycles_sm\"])/80)",
        sim_eval="np.average(hw[\"inst_issued\"])/float(sim[\"gpu_tot_sim_cycle\s*=\s*(.*)\"])",
        hw_name="all",
        drophwnumbelow=0,
        plottype="linear",
        stattype="rate"
    ),

]
