#!/usr/bin/env python3
import argparse
import pandas as pd
import plotly.graph_objects as go


def main():
    parser = argparse.ArgumentParser(description="Plot PM sampling CSV data")
    parser.add_argument("csv", nargs="?", default="output_0.csv", help="CSV file to plot")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # Ignore timestamp columns
    ignore_cols = {"StartTimestamp", "EndTimestamp"}
    cols = [c for c in df.columns if c not in ignore_cols]

    # Check for sm__cycles_elapsed.avg for x-axis
    cycles_col = "sm__cycles_elapsed.avg"
    if cycles_col not in cols:
        raise ValueError(f"{cycles_col} not found in CSV columns: {cols}")

    # Accumulated cycles from start
    x = df[cycles_col].cumsum()

    # Other columns as y
    y_cols = [c for c in cols if c != cycles_col]

    fig = go.Figure()
    for col in y_cols:
        fig.add_trace(go.Scatter(x=x, y=df[col], mode="lines", name=col))

    fig.update_layout(
        xaxis_title="sm__cycles_elapsed",
        yaxis_title="Value",
        title="PM Sampling Metrics",
        showlegend=True,
        legend=dict(title="Metrics", bordercolor='black', orientation='h'),
    )
    fig.write_html("pm_sampling_plot.html", include_plotlyjs='cdn')


if __name__ == "__main__":
    main()
