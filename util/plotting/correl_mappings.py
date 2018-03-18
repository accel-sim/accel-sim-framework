config_maps = \
{
    "TITANX-P102": "TITAN X (Pascal)",
    "P100-HBM" : "Tesla P100",
}


import collections
CorrelStat = collections.namedtuple('CorrelStat', 'chart_name hw_eval sim_eval config plotfile')
correl_list = \
[
    CorrelStat(chart_name="Execution Cycles (1417 MHz)",
        plotfile="titanx-p102-cycles.html",
        hw_eval="float(hw[\"Duration\"])*1417",
        sim_eval="float(sim[\"gpu_tot_sim_cycle\s*=\s*(.*)\"])",
        config="TITANX-P102"
    ),
    CorrelStat(chart_name="Execution Cycles (1480 MHz)",
        plotfile="p100-cycles.html",
        hw_eval="float(hw[\"Duration\"])*1480",
        sim_eval="float(sim[\"gpu_tot_sim_cycle\s*=\s*(.*)\"])",
        config="P100-HBM"
    ),
#    CorrelStat(chart_name="Global Load Transactions",
#        plotfile="global-load.html",
#        hw_eval="float(hw[\"gld_transactions\"])",
#        sim_eval="float(sim[\"L1_GLOBAL_R\s*=\s*(.*)\"])",
#        config="all"
#    ),
#    CorrelStat(chart_name="Warp Instructions",
#        plotfile="warp-inst.html",
#        hw_eval="float(hw[\"inst_executed\"])",
#        sim_eval="float(sim[\"gpgpu_n_tot_w_icount\s*=\s*(.*)\"])",
#        config="all"
#    ),
#    CorrelStat(chart_name="L2 read transactions",
#        plotfile="l2-read-transactions.html",
#        hw_eval="float(hw[\"l2_read_transactions\"])",
#        sim_eval="float(sim[\"GLOBAL_ACC_R\s*=\s*(.*)\"]) + "\
#            "float(sim[\"LOCAL_ACC_R\s*=\s*(.*)\"]) + "\
#            "float(sim[\"CONST_ACC_R\s*=\s*(.*)\"]) + "\
#            "float(sim[\"TEXTURE_ACC_R\s*=\s*(.*)\"]) + "\
#            "float(sim[\"INST_ACC_R\s*=\s*(.*)\"]) + "\
#            "float(sim[\"L1_WR_ALLOC_R\s*=\s*(.*)\"])",
#        config="all"
#    ),
]
