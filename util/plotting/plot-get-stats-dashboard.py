#!/usr/bin/env python3

"""Build a single, self-contained HTML dashboard from a get_stats.py CSV.

Unlike plot-get-stats.py (which writes one HTML file per stat), this script
emits a single dashboard.html with a searchable sidebar. Only one stat's chart
is shown at a time; clicking a different stat in the sidebar switches the view.
The Plotly library is embedded so the resulting file works fully offline.
"""

from optparse import OptionParser
import plotly
import os
import re
import json
import html

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

import sys

sys.path.insert(0, os.path.join(this_directory, "..", "job_launching"))
import common

import numpy as np
import csv


def get_csv_data(filepath):
    all_stats = {}
    apps = []
    data = {}
    any_data = False
    with open(filepath, "r") as data_file:
        reader = csv.reader(data_file)  # define reader object
        state = "start"
        for row in reader:  # loop through rows in csv file
            if len(row) != 0 and row[0].startswith("----"):
                state = "find-stat"
                continue
            if state == "find-stat":
                current_stat = row[0]
                state = "find-apps"
                continue
            if state == "find-apps":
                apps = [item.upper() for item in row[1:]]
                state = "process-cfgs"
                continue
            if state == "process-cfgs":
                if len(row) == 0:
                    if any_data:
                        all_stats[current_stat] = apps, data
                    apps = []
                    data = {}
                    state = "start"
                    any_data = False
                    continue
                temp = []
                for x in row[1:]:
                    try:
                        temp.append(float(x))
                        any_data = True
                    except ValueError:
                        temp.append(0)
                data[row[0]] = np.array(temp)

    return all_stats


def _detect_unit(s):
    """Infer a human-readable unit from an (unescaped) collection regex."""
    if re.search(r"inst/sec", s):
        return "inst/sec"
    if re.search(r"cycle/sec", s):
        return "cycle/sec"
    if re.search(r"GB/Sec", s, re.IGNORECASE):
        return "GB/Sec"
    if re.search(r"\bsec\b", s):
        return "sec"
    if "%" in s:
        return "%"
    if re.search(r"\(\.\*\)x\b", s):
        return "x"
    return ""


def pretty_label(raw):
    """Turn a collection regex into a readable stat name.

    e.g. 'gpgpu_simulation_time\\s*=.*\\(([0-9]+) sec\\).*' -> 'gpgpu_simulation_time (sec)'
         '\\s+L2_cache_stats_breakdown\\[GLOBAL_ACC_R\\]\\[HIT\\]\\s*=\\s*(.*)'
             -> 'L2_cache_stats_breakdown[GLOBAL_ACC_R][HIT]'
         'gpgpu_simulation_rate\\s+=\\s+(.*)\\s+\\(inst\\/sec\\)'
             -> 'gpgpu_simulation_rate (inst/sec)'
    """
    # Unescape common regex escapes so unit detection and names read cleanly.
    s = raw
    for esc, plain in (
        ("\\/", "/"),
        ("\\[", "["),
        ("\\]", "]"),
        ("\\(", "("),
        ("\\)", ")"),
    ):
        s = s.replace(esc, plain)

    unit = _detect_unit(s)

    # Keep only the part before the '=' assignment in the regex.
    name = re.split(r"=", s, maxsplit=1)[0]

    # Strip regex whitespace tokens and any leftover escapes/quantifiers.
    name = name.replace("\\s+", "").replace("\\s*", "").replace("\\s", "")
    name = name.replace("\\", "")
    name = name.strip(" +*")

    if unit and unit not in name:
        name = "{0} ({1})".format(name, unit)
    return name


def load_categories(stats_yml):
    """Map each raw stat regex to a sidebar category using the stats yml.

    Returns a dict {raw_stat: category_label}. Best-effort: if the yml is
    missing or unreadable, an empty dict is returned and everything falls
    back to the 'Other' group.
    """
    labels = {
        "collect_aggregate": "Aggregate",
        "collect_abs": "Per-kernel (absolute)",
        "collect_rates": "Rates",
    }
    mapping = {}
    if not stats_yml or not os.path.exists(stats_yml):
        return mapping
    try:
        import yaml

        parsed = yaml.load(open(stats_yml), Loader=yaml.FullLoader)
    except Exception:
        return mapping
    for key, label in labels.items():
        for raw in parsed.get(key, []) or []:
            mapping[raw] = label
    return mapping


colors = [
    "#0F8C79",
    "#BD2D28",
    "#E3BA22",
    "#E6842A",
    "#137B80",
    "#8E6C8A",
    "#9A3E25",
    "#3B7DD8",
]


def build_figure(stat, apps, data):
    """Build a Plotly grouped-bar figure dict for one stat."""
    label = pretty_label(stat)
    traces = []
    cfg_count = 0
    for cfg, values in data.items():
        traces.append(
            {
                "type": "bar",
                "x": apps,
                "y": [float(v) for v in values],
                "name": cfg,
                "marker": {"color": colors[cfg_count % len(colors)]},
            }
        )
        cfg_count += 1

    layout = {
        "barmode": "group",
        "bargap": 0.25,
        "bargroupgap": 0.05,
        "showlegend": True,
        "legend": {"orientation": "h", "y": -0.35, "x": 0},
        "margin": {"l": 80, "r": 40, "t": 20, "b": 160},
        "yaxis": {
            "title": {"text": label},
            "gridcolor": "#e9edf2",
            "zerolinecolor": "#d0d7de",
        },
        "xaxis": {
            "automargin": True,
            "tickangle": -35,
        },
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {
            "family": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
            "color": "#24292f",
        },
    }
    return {"data": traces, "layout": layout, "label": label}


def render_html(basename, figures, order, categories):
    """Assemble the self-contained dashboard HTML string.

    figures: {stat_id: {data, layout, label}}
    order:   list of stat_ids in display order
    categories: {stat_id: category_label}
    """
    plotly_js = plotly.offline.get_plotlyjs()

    # Group stat ids by category, preserving encounter order within each group.
    grouped = {}
    for sid in order:
        cat = categories.get(sid, "Other")
        grouped.setdefault(cat, []).append(sid)

    category_order = ["Aggregate", "Per-kernel (absolute)", "Rates", "Other"]
    ordered_cats = [c for c in category_order if c in grouped]
    ordered_cats += [c for c in grouped if c not in ordered_cats]

    sidebar_items = []
    for cat in ordered_cats:
        sidebar_items.append(
            '<div class="group-label">{0}</div>'.format(html.escape(cat))
        )
        for sid in grouped[cat]:
            label = figures[sid]["label"]
            sidebar_items.append(
                '<button class="stat-item" data-id="{0}" '
                'data-search="{1}">{2}</button>'.format(
                    html.escape(sid, quote=True),
                    html.escape(label.lower(), quote=True),
                    html.escape(label),
                )
            )
    sidebar_html = "\n".join(sidebar_items)

    figures_payload = {
        sid: {"data": figures[sid]["data"], "layout": figures[sid]["layout"], "label": figures[sid]["label"]}
        for sid in order
    }
    figures_json = json.dumps(figures_payload)
    order_json = json.dumps(order)
    title = html.escape(basename)

    return TEMPLATE.format(
        title=title,
        plotly_js=plotly_js,
        sidebar_html=sidebar_html,
        figures_json=figures_json,
        order_json=order_json,
    )


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title} - Stats Dashboard</title>
<script>{plotly_js}</script>
<style>
  :root {{
    --bg: #f6f8fa;
    --panel: #ffffff;
    --border: #d0d7de;
    --text: #24292f;
    --muted: #57606a;
    --accent: #0969da;
    --accent-soft: #ddf4ff;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ height: 100%; margin: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    display: flex;
    height: 100vh;
    overflow: hidden;
  }}
  #sidebar {{
    width: 340px;
    min-width: 260px;
    background: var(--panel);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    height: 100%;
  }}
  #sidebar-header {{
    padding: 18px 18px 12px 18px;
    border-bottom: 1px solid var(--border);
  }}
  #sidebar-header h1 {{
    font-size: 16px;
    margin: 0 0 4px 0;
  }}
  #sidebar-header p {{
    font-size: 12px;
    color: var(--muted);
    margin: 0 0 12px 0;
  }}
  #search {{
    width: 100%;
    padding: 8px 10px;
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 13px;
    outline: none;
  }}
  #search:focus {{ border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }}
  #stat-list {{ overflow-y: auto; padding: 8px; flex: 1; }}
  .group-label {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
    padding: 12px 10px 4px 10px;
    font-weight: 600;
  }}
  .stat-item {{
    display: block;
    width: 100%;
    text-align: left;
    background: transparent;
    border: none;
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 13px;
    color: var(--text);
    cursor: pointer;
    word-break: break-word;
  }}
  .stat-item:hover {{ background: var(--bg); }}
  .stat-item.active {{ background: var(--accent-soft); color: var(--accent); font-weight: 600; }}
  #main {{ flex: 1; display: flex; flex-direction: column; height: 100%; padding: 24px; overflow: hidden; }}
  #chart-title {{ font-size: 20px; font-weight: 600; margin: 0 0 16px 4px; }}
  #chart-card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    box-shadow: 0 1px 3px rgba(27,31,36,0.08);
    flex: 1;
    padding: 16px;
    min-height: 0;
  }}
  #chart {{ width: 100%; height: 100%; }}
  #empty {{ color: var(--muted); font-size: 14px; padding: 40px; }}
</style>
</head>
<body>
  <div id="sidebar">
    <div id="sidebar-header">
      <h1>{title}</h1>
      <p>Select a stat to view. One chart at a time.</p>
      <input id="search" type="text" placeholder="Filter stats..." autocomplete="off"/>
    </div>
    <div id="stat-list">
      {sidebar_html}
    </div>
  </div>
  <div id="main">
    <div id="chart-title"></div>
    <div id="chart-card"><div id="chart"></div></div>
  </div>

<script>
  var FIGURES = {figures_json};
  var ORDER = {order_json};
  var chartDiv = document.getElementById('chart');
  var titleDiv = document.getElementById('chart-title');
  var config = {{ responsive: true, displaylogo: false, toImageButtonOptions: {{ format: 'png', scale: 2 }} }};

  function show(id) {{
    var fig = FIGURES[id];
    if (!fig) return;
    titleDiv.textContent = fig.label;
    Plotly.react(chartDiv, fig.data, fig.layout, config);
    document.querySelectorAll('.stat-item').forEach(function(el) {{
      el.classList.toggle('active', el.getAttribute('data-id') === id);
    }});
    var active = document.querySelector('.stat-item[data-id="' + (window.CSS && CSS.escape ? CSS.escape(id) : id) + '"]');
    if (active) active.scrollIntoView({{ block: 'nearest' }});
    if (location.hash.slice(1) !== id) {{
      history.replaceState(null, '', '#' + encodeURIComponent(id));
    }}
  }}

  document.querySelectorAll('.stat-item').forEach(function(el) {{
    el.addEventListener('click', function() {{ show(el.getAttribute('data-id')); }});
  }});

  document.getElementById('search').addEventListener('input', function(e) {{
    var q = e.target.value.toLowerCase();
    document.querySelectorAll('.stat-item').forEach(function(el) {{
      var match = el.getAttribute('data-search').indexOf(q) !== -1;
      el.style.display = match ? 'block' : 'none';
    }});
    document.querySelectorAll('.group-label').forEach(function(g) {{
      var n = g.nextElementSibling, visible = false;
      while (n && n.classList.contains('stat-item')) {{
        if (n.style.display !== 'none') visible = true;
        n = n.nextElementSibling;
      }}
      g.style.display = visible ? 'block' : 'none';
    }});
  }});

  window.addEventListener('resize', function() {{ Plotly.Plots.resize(chartDiv); }});

  var initial = decodeURIComponent(location.hash.slice(1));
  if (!FIGURES[initial]) initial = ORDER[0];
  if (initial) show(initial);
</script>
</body>
</html>
"""


def main():
    parser = OptionParser()
    parser.add_option(
        "-c", "--csv_file", dest="csv_file", help="File to parse", default=""
    )
    parser.add_option(
        "-o",
        "--output",
        dest="output",
        help="Output HTML file path.",
        default=os.path.join(this_directory, "htmls", "dashboard.html"),
    )
    parser.add_option(
        "-n",
        "--basename",
        dest="basename",
        help="Dashboard title.",
        default="gpgpu-sim",
    )
    parser.add_option(
        "-s",
        "--stats_yml",
        dest="stats_yml",
        help="Stats yml used to group stats into sidebar categories.",
        default=os.path.join(
            this_directory, "..", "job_launching", "stats", "example_stats.yml"
        ),
    )
    (options, args) = parser.parse_args()
    options.csv_file = common.file_option_test(options.csv_file, "", this_directory)
    if options.csv_file == "":
        parser.error("Please supply a csv file with -c/--csv_file")

    all_stats = get_csv_data(options.csv_file)
    if not all_stats:
        print("No stats found in {0}".format(options.csv_file))
        return

    categories_raw = load_categories(options.stats_yml)

    figures = {}
    order = []
    categories = {}
    seen = set()
    for stat, (apps, data) in all_stats.items():
        sid = re.sub("[^0-9a-zA-Z]+", "_", stat).strip("_")
        base_sid = sid or "stat"
        n = 1
        while sid in seen:
            n += 1
            sid = "{0}_{1}".format(base_sid, n)
        seen.add(sid)

        figures[sid] = build_figure(stat, apps, data)
        order.append(sid)
        categories[sid] = categories_raw.get(stat, "Other")
        print("added: " + figures[sid]["label"])

    out_html = render_html(options.basename, figures, order, categories)

    outdir = os.path.dirname(os.path.abspath(options.output))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(options.output, "w") as f:
        f.write(out_html)

    print("\nDashboard written to: {0}".format(os.path.abspath(options.output)))
    print("Open it in any browser (no internet required).")


if __name__ == "__main__":
    main()
