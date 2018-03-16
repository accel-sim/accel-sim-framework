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
    CorrelStat(chart_name="Instructions Issued",
        plotfile="instrs.html",
        hw_eval="float(hw[\"inst_issued\"])",
        sim_eval="float(sim[\"gpu_tot_sim_insn\s*=\s*(.*)\"])",
        config="all"
    ),
]
