#!/usr/bin/env python3
import argparse
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Plot PM sampling CSV data")
    parser.add_argument(
        "csv", nargs="?", default="output_0.csv", help="CSV file to plot"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize counters by sm__cycles_elapsed.avg",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    # drop first
    df = df.iloc[1:]

    # drop na
    df = df.dropna()

    # Auto-compute: for .pct columns, find matching .sum columns and create
    # derived columns = pct * sum / 100.
    # E.g. lts__average_t_sector_srcnode_gpc_aperture_device_op_read.pct
    #    * lts__t_sectors.sum / 100
    #    -> lts__t_sectors_srcnode_gpc_aperture_device_op_read
    pct_cols = [c for c in df.columns if c.endswith(".pct")]
    sum_cols = [c for c in df.columns if c.endswith(".sum")]
    for pct_col in pct_cols:
        pct_base = pct_col.rsplit(".", 1)[0]  # e.g. lts__average_t_sector_srcnode_gpc...
        pct_prefix = pct_base.split("__", 1)[0]  # e.g. lts
        for sum_col in sum_cols:
            sum_base = sum_col.rsplit(".", 1)[0]  # e.g. lts__t_sectors
            sum_prefix = sum_base.split("__", 1)[0]  # e.g. lts
            if pct_prefix != sum_prefix:
                continue
            # Check measurement root match: "average_t_sector" <-> "t_sectors"
            pct_measurement = pct_base.split("__", 1)[1]  # average_t_sector_srcnode_gpc...
            sum_measurement = sum_base.split("__", 1)[1]  # t_sectors
            # The .pct measurement should start with "average_" + singular of sum measurement
            # e.g. pct has "average_t_sector_*", sum has "t_sectors"
            if sum_measurement.endswith("s"):
                singular = sum_measurement[:-1]  # t_sector
            else:
                singular = sum_measurement
            if not pct_measurement.startswith(f"average_{singular}"):
                continue
            # Extract the suffix after the measurement root
            suffix = pct_measurement[len(f"average_{singular}"):]  # e.g. _srcnode_gpc...
            new_name = f"{pct_prefix}__{sum_measurement}{suffix}"
            df[new_name] = df[pct_col] * df[sum_col] / 100
            print(f"Derived: {new_name} = {pct_col} * {sum_col} / 100")

    # Ignore timestamp columns
    ignore_cols = {"StartTimestamp", "EndTimestamp"}
    cols = [c for c in df.columns if c not in ignore_cols]

    # Check for sm__cycles_elapsed.avg for x-axis
    cycles_col = "sm__cycles_elapsed.avg"
    if cycles_col not in cols:
        raise ValueError(f"{cycles_col} not found in CSV columns: {cols}")

    # Accumulated cycles from start
    df["accum_cycles"] = df[cycles_col].cumsum()

    # Find first non-zero sm__inst_executed.sum index
    first_non_zero_inst = df["sm__inst_executed.sum"].ne(0).idxmax()
    df = df.iloc[first_non_zero_inst - 5 :].reset_index(drop=True)

    # Find last non-zero sm__inst_executed.sum index
    last_non_zero_inst = df["sm__inst_executed.sum"].ne(0)[::-1].idxmax()
    df = df.iloc[: last_non_zero_inst + 100].reset_index(drop=True)

    # reset x to start from 0
    df["accum_cycles"] = df["accum_cycles"] - df["accum_cycles"].iloc[0]

    # Other columns as y
    y_cols = [c for c in cols if c != cycles_col]

    # Normalize counters by sm__cycles_elapsed.avg if requested
    if args.normalize:
        for col in y_cols:
            df[col] = df[col] / df[cycles_col]

    fig = go.Figure()
    for col in y_cols:
        fig.add_trace(
            go.Scatter(x=df["accum_cycles"], y=df[col], mode="lines", name=col)
        )

    y_axis_label = "Value / cycle" if args.normalize else "Value"
    fig.update_layout(
        xaxis_title="sm__cycles_elapsed",
        yaxis_title=y_axis_label,
        title="PM Sampling Metrics",
        showlegend=True,
        legend=dict(title="Metrics", bordercolor="black", orientation="h"),
        height=600,
    )
    fig.write_html("pm_sampling_plot.html", include_plotlyjs="cdn")


if __name__ == "__main__":
    main()
