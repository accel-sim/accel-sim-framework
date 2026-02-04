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
