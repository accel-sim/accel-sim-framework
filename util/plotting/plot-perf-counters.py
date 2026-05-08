import argparse
import re
import plotly.graph_objects as go
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot data from a CSV file")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file to load")
    parser.add_argument("--normalize", action="store_true", help="Normalize metrics per cycle")
    return parser.parse_args()


def get_to_plot(df: pd.DataFrame, normalize: bool = False) -> tuple[pd.Series, pd.DataFrame]:
    """
    Get the global cycles and the to plot dataframe.
    """
    global_cycles = df.filter(regex=r'gpu_sim_cycle$').sum(axis=1)
    cycle_diff = global_cycles.diff()
    to_plot = pd.DataFrame()

    # ---------------------------
    norm_factor = cycle_diff if normalize else 1

    to_plot["L2 Read Misses"] = df.filter(regex=r'L2.*GLOBAL_ACC_R_MISS$').sum(axis=1).diff() / norm_factor
    to_plot["Accumulated L2 Read Misses"] = df.filter(regex=r'L2.*GLOBAL_ACC_R_MISS$').sum(axis=1)
    to_plot["Normalized IPC"] = df.filter(regex=r'sim_insn$').sum(axis=1).diff() / cycle_diff
    to_plot["L2 Bandwidth - Replies in parallel"] = df.filter(regex=r'^partiton_replys_in_parallel$').sum(axis=1).diff() / norm_factor
    to_plot["L2 Bandwidth - Reqs in parallel"] = df.filter(regex=r'^partiton_reqs_in_parallel$').sum(axis=1).diff() / norm_factor
    to_plot["SIMT to Mem"] = df.filter(regex=r'n_simt_to_mem_').sum(axis=1).diff() / norm_factor
    to_plot["Mem to SIMT"] = df.filter(regex=r'n_mem_to_simt_').sum(axis=1).diff() / norm_factor
    to_plot["L2 Writes"] = df.filter(regex=r'L2_bank_.*_GLOBAL_ACC_W_(HIT$|MISS|HIT_RESERVED|SECTOR_MISS)').sum(axis=1).diff() / norm_factor
    to_plot["L2 Reads"] = df.filter(regex=r'L2_bank_.*_GLOBAL_ACC_R_(HIT$|MISS|HIT_RESERVED|SECTOR_MISS)').sum(axis=1).diff() / norm_factor
    to_plot["L1 Write RESERVATION_FAIL"] = df.filter(regex=r'L1D_.*_GLOBAL_ACC_W_RESERVATION_FAIL').sum(axis=1).diff() / norm_factor
    to_plot["LRC ICNT to LRC sectors"] = df.filter(regex=r'LRC_subpartition_num_icnt_to_lrc_sectors_.*').sum(axis=1).diff() / norm_factor
    to_plot["LRC LRC to L2 sectors"] = df.filter(regex=r'LRC_subpartition_num_lrc_to_l2_sectors.*').sum(axis=1).diff() / norm_factor
    to_plot["LRC L2 stall due to LRC queue full"] = df.filter(regex=r'LRC_subpartition_l2_stall_due_to_lrc_full.*').sum(axis=1).diff() / norm_factor
    to_plot["LRC average queue size"] = df.filter(regex=r'LRC_subpartition_lrc_queue_size.*').mean(axis=1)
    to_plot["LRC max coalesced count"] = df.filter(regex=r'LRC_subpartition_current_max_coalesced_count.*').max(axis=1)
    to_plot["LRC average max coalesced count"] = df.filter(regex=r'LRC_subpartition_current_max_coalesced_count.*').mean(axis=1)
    to_plot["LRC average coalesced count"] = df.filter(regex=r'LRC_subpartition_current_avg_coalesced_count.*').mean(axis=1)
    # to_plot["Shader Idle"] = df.filter(regex=r'shader_cycle_distro_0_').diff()
    # to_plot["Shader Waiting for RAW"] = df.filter(regex=r'shader_cycle_distro_1_').diff()
    # to_plot["Shader Stalled"] = df.filter(regex=r'shader_cycle_distro_2_').diff()

    # Simulation rate over time (requires wall_clock_ms column)
    if 'wall_clock_ms' in df.columns:
        wall_diff_sec = df['wall_clock_ms'].diff() / 1000.0
        to_plot["Simulation Rate (inst/sec)"] = df.filter(regex=r'sim_insn$').sum(axis=1).diff() / wall_diff_sec
        to_plot["Simulation Rate (cycle/sec)"] = df.filter(regex=r'sim_cycle$').sum(axis=1).diff() / wall_diff_sec
    # ---------------------------

    return global_cycles, to_plot


def plot_series(x_series, df, config_name, existing_figs=None):
    figs = {}
    for col in df.columns:
        if existing_figs is None or col not in existing_figs:
            fig = go.Figure(
                data=[go.Scatter(x=x_series, y=df[col], name=config_name)],
                layout=go.Layout(title=col, xaxis=dict(title="Cycle"))
            )
            fig.update_layout(
                title=dict(x=0.5, y=0.95, xanchor='center', yanchor='top'),
                margin=dict(l=30, r=30, t=50, b=30),
                height=400
            )
        else:
            fig = existing_figs[col]
            fig.add_trace(go.Scatter(x=x_series, y=df[col], name=config_name))
        figs[col] = fig

    return figs


def plot_series_single(x_series, df, title, yaxis_title="", existing_figs=None):
    fig = go.Figure(
        layout=go.Layout(title=title, xaxis=dict(title="Cycle"), yaxis=dict(title=yaxis_title))
    )
    for col in df.columns:
        number = re.search(r'_(\d+)', col)
        if number is not None:
            name = int(number.group(1))
        else:
            name = col
        fig.add_trace(go.Scatter(x=x_series, y=df[col], name=name))

    if existing_figs is not None:
        existing_figs[title] = fig
        return existing_figs
    else:
        figs = {}
        figs[title] = fig
        return figs


def save_figs(figs, out_path="PerfCounters.html", combine: bool = True):
    with open(out_path, 'w') as f:
        f.write('')
    if combine:
        with open(out_path, 'a') as f:
            for fig in figs.values():
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    else:
        for fig in figs:
            fig.write_html(f"{fig.layout.title.text}.html")


def show_figs(figs):
    for fig in figs.values():
        fig.show()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv_file)

    global_cycles, to_plot = get_to_plot(df, normalize=args.normalize)
    figs = plot_series(global_cycles, to_plot, config_name=args.csv_file)
    save_figs(figs)
    # show_figs(figs)


if __name__ == "__main__":
    main()
