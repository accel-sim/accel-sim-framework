#!/usr/bin/env python3
"""Self-contained correlation dashboard (sidebar + per-app/per-kernel toggle).

Consumed by plot-correlation.py after it builds the figs = {kernel, app} dicts.
"""

from __future__ import print_function

import html
import json
import os
import re

import plotly


def figure_to_payload(fig):
    """Convert a Plotly Figure to a JSON-serializable payload for the dashboard.

    Pulls the long gray summary annotations out of the Plotly layout into a
    plain `summary` string so the dashboard can wrap them in HTML (Plotly
    annotations clip at the plot edge and truncate).
    """
    raw = fig.to_plotly_json()
    layout = raw.get("layout", {}) or {}
    summaries = []
    for ann in layout.get("annotations") or []:
        text = ann.get("text") if isinstance(ann, dict) else None
        if text:
            summaries.append(re.sub(r"<[^>]+>", "", str(text)).strip())
    # Drop in-plot annotations; shown in the HTML banner instead.
    layout["annotations"] = []
    # Less top margin needed without the overlay banner.
    margin = layout.get("margin")
    if not isinstance(margin, dict):
        margin = {}
    else:
        margin = dict(margin)
    margin["t"] = max(int(margin.get("t") or 0), 40)
    layout["margin"] = margin
    return {
        "data": raw.get("data", []),
        "layout": layout,
        "summary": "\n".join(summaries),
    }


def _basename_key(fig_key):
    """Strip path and _app/_kernel suffix from a figs dict key.

    e.g. '.../correl-html/gpc_cycles.RTX3070-SASS_app' -> 'gpc_cycles.RTX3070-SASS'
    """
    base = os.path.basename(fig_key)
    if base.endswith("_app"):
        return base[: -len("_app")]
    if base.endswith("_kernel"):
        return base[: -len("_kernel")]
    return base


def _pretty_label(stat_key, fig):
    """Human label: prefer axis/layout titles, else clean up plotfile.CFG."""

    def _title_text(obj):
        if obj is None:
            return None
        if isinstance(obj, str):
            return obj
        return getattr(obj, "text", None)

    layout = getattr(fig, "layout", None)
    candidates = []
    if layout is not None:
        candidates.append(_title_text(getattr(layout, "title", None)))
        for axis in ("xaxis", "yaxis"):
            ax = getattr(layout, axis, None)
            if ax is not None:
                candidates.append(_title_text(getattr(ax, "title", None)))

    for raw in candidates:
        if not raw:
            continue
        text = re.sub(r"</?b>", "", str(raw)).strip()
        text = re.sub(r"\s*\[Correl=.*$", "", text).strip()
        # "Hardware GPC Cycles" / "Simulation GPC Cycles" / "Per App GPC Cycles"
        text = re.sub(
            r"^(Hardware|Simulation|Per App|Per-App|Per Kernel|Per-Kernel)\s+",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        if text:
            return text

    # Fallback: gpc_cycles.RTX3070-SASS -> Gpc Cycles (RTX3070-SASS)
    parts = stat_key.rsplit(".", 1)
    stem = parts[0].replace("-", " ").replace("_", " ")
    stem = " ".join(w.capitalize() for w in stem.split())
    if len(parts) == 2:
        return "{0} ({1})".format(stem, parts[1])
    return stem


def _stable_id(stat_key):
    sid = re.sub(r"[^0-9a-zA-Z]+", "_", stat_key).strip("_")
    return sid or "stat"


def write_correl_dashboard(outdir, figs, title="Correlation"):
    """Write correl-html/dashboard.html from figs={'kernel':..., 'app':...}.

    Returns the absolute path of the written file.
    """
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # Pair app/kernel figures by shared plotname stem.
    app_figs = figs.get("app", {}) or {}
    kernel_figs = figs.get("kernel", {}) or {}

    by_stat = {}  # stat_key -> {app, kernel, label}
    for key, fig in app_figs.items():
        sk = _basename_key(key)
        by_stat.setdefault(sk, {})["app"] = fig
    for key, fig in kernel_figs.items():
        sk = _basename_key(key)
        by_stat.setdefault(sk, {})["kernel"] = fig

    order = []
    labels = {}
    payload_app = {}
    payload_kernel = {}

    for sk in sorted(by_stat.keys()):
        entry = by_stat[sk]
        sid = _stable_id(sk)
        # Prefer app fig for the label (same chart_name).
        label_fig = entry.get("app") or entry.get("kernel")
        label = _pretty_label(sk, label_fig)
        order.append(sid)
        labels[sid] = label
        if "app" in entry:
            p = figure_to_payload(entry["app"])
            p["label"] = label
            payload_app[sid] = p
        if "kernel" in entry:
            p = figure_to_payload(entry["kernel"])
            p["label"] = label
            payload_kernel[sid] = p

    if not order:
        print("No correlation figures to put in the dashboard.")
        return None

    out_path = os.path.join(outdir, "dashboard.html")
    html_str = _render_html(
        title=title,
        order=order,
        labels=labels,
        figures_app=payload_app,
        figures_kernel=payload_kernel,
    )
    with open(out_path, "w") as f:
        f.write(html_str)
    return os.path.abspath(out_path)


def _render_html(title, order, labels, figures_app, figures_kernel):
    plotly_js = plotly.offline.get_plotlyjs()

    sidebar_items = []
    for sid in order:
        label = labels[sid]
        sidebar_items.append(
            '<button class="stat-item" data-id="{0}" '
            'data-search="{1}">{2}</button>'.format(
                html.escape(sid, quote=True),
                html.escape(label.lower(), quote=True),
                html.escape(label),
            )
        )
    sidebar_html = "\n".join(sidebar_items)

    return TEMPLATE.format(
        title=html.escape(title),
        plotly_js=plotly_js,
        sidebar_html=sidebar_html,
        order_json=json.dumps(order),
        labels_json=json.dumps(labels),
        figures_app_json=json.dumps(figures_app),
        figures_kernel_json=json.dumps(figures_kernel),
    )


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title} - Correlation Dashboard</title>
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
  .mode-toggle {{
    display: flex;
    gap: 6px;
    margin-bottom: 12px;
  }}
  .mode-btn {{
    flex: 1;
    padding: 7px 8px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--bg);
    font-size: 12px;
    font-weight: 600;
    color: var(--muted);
    cursor: pointer;
  }}
  .mode-btn.active {{
    background: var(--accent-soft);
    border-color: var(--accent);
    color: var(--accent);
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
  #chart-title {{ font-size: 20px; font-weight: 600; margin: 0 0 4px 4px; }}
  #chart-subtitle {{ font-size: 13px; color: var(--muted); margin: 0 0 8px 4px; }}
  #chart-summary {{
    display: none;
    background: #F5F3F2;
    border: 1px solid #FFFFFF;
    border-radius: 6px;
    padding: 10px 12px;
    margin: 0 0 12px 0;
    font-size: 12px;
    line-height: 1.45;
    color: var(--text);
    white-space: pre-wrap;
    word-break: break-word;
    overflow-wrap: anywhere;
  }}
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
</style>
</head>
<body>
  <div id="sidebar">
    <div id="sidebar-header">
      <h1>{title}</h1>
      <p>HW vs sim correlation. One chart at a time.</p>
      <div class="mode-toggle">
        <button class="mode-btn active" data-mode="app" id="btn-app">Per-app</button>
        <button class="mode-btn" data-mode="kernel" id="btn-kernel">Per-kernel</button>
      </div>
      <input id="search" type="text" placeholder="Filter stats..." autocomplete="off"/>
    </div>
    <div id="stat-list">
      {sidebar_html}
    </div>
  </div>
  <div id="main">
    <div id="chart-title"></div>
    <div id="chart-subtitle"></div>
    <div id="chart-summary"></div>
    <div id="chart-card"><div id="chart"></div></div>
  </div>

<script>
  var FIGURES_APP = {figures_app_json};
  var FIGURES_KERNEL = {figures_kernel_json};
  var ORDER = {order_json};
  var LABELS = {labels_json};
  var mode = 'app';
  var currentId = null;
  var chartDiv = document.getElementById('chart');
  var titleDiv = document.getElementById('chart-title');
  var subtitleDiv = document.getElementById('chart-subtitle');
  var summaryDiv = document.getElementById('chart-summary');
  var config = {{ responsive: true, displaylogo: false, toImageButtonOptions: {{ format: 'png', scale: 2 }} }};

  function figuresForMode() {{
    return mode === 'kernel' ? FIGURES_KERNEL : FIGURES_APP;
  }}

  function setHash(id, m) {{
    var h = encodeURIComponent(id) + '&mode=' + m;
    if (location.hash.slice(1) !== h) {{
      history.replaceState(null, '', '#' + h);
    }}
  }}

  function parseHash() {{
    var raw = location.hash.slice(1);
    if (!raw) return {{ id: null, mode: null }};
    var parts = raw.split('&');
    var id = decodeURIComponent(parts[0]);
    var m = null;
    for (var i = 1; i < parts.length; i++) {{
      if (parts[i].indexOf('mode=') === 0) m = parts[i].slice(5);
    }}
    return {{ id: id, mode: m }};
  }}

  function show(id) {{
    var figs = figuresForMode();
    var fig = figs[id];
    if (!fig) {{
      // Fall back to the other mode if this stat only exists there.
      var other = mode === 'app' ? FIGURES_KERNEL : FIGURES_APP;
      if (other[id]) {{
        mode = mode === 'app' ? 'kernel' : 'app';
        updateModeButtons();
        fig = other[id];
      }} else {{
        return;
      }}
    }}
    currentId = id;
    titleDiv.textContent = LABELS[id] || fig.label || id;
    subtitleDiv.textContent = (mode === 'kernel' ? 'Per-kernel' : 'Per-app') + ' aggregation';
    if (fig.summary) {{
      summaryDiv.textContent = fig.summary;
      summaryDiv.style.display = 'block';
    }} else {{
      summaryDiv.textContent = '';
      summaryDiv.style.display = 'none';
    }}
    Plotly.react(chartDiv, fig.data, fig.layout, config);
    document.querySelectorAll('.stat-item').forEach(function(el) {{
      el.classList.toggle('active', el.getAttribute('data-id') === id);
    }});
    var esc = (window.CSS && CSS.escape) ? CSS.escape(id) : id;
    var active = document.querySelector('.stat-item[data-id="' + esc + '"]');
    if (active) active.scrollIntoView({{ block: 'nearest' }});
    setHash(id, mode);
  }}

  function updateModeButtons() {{
    document.querySelectorAll('.mode-btn').forEach(function(el) {{
      el.classList.toggle('active', el.getAttribute('data-mode') === mode);
    }});
  }}

  function setMode(m) {{
    if (m !== 'app' && m !== 'kernel') return;
    mode = m;
    updateModeButtons();
    if (currentId) show(currentId);
  }}

  document.querySelectorAll('.mode-btn').forEach(function(el) {{
    el.addEventListener('click', function() {{ setMode(el.getAttribute('data-mode')); }});
  }});

  document.querySelectorAll('.stat-item').forEach(function(el) {{
    el.addEventListener('click', function() {{ show(el.getAttribute('data-id')); }});
  }});

  document.getElementById('search').addEventListener('input', function(e) {{
    var q = e.target.value.toLowerCase();
    document.querySelectorAll('.stat-item').forEach(function(el) {{
      var match = el.getAttribute('data-search').indexOf(q) !== -1;
      el.style.display = match ? 'block' : 'none';
    }});
  }});

  window.addEventListener('resize', function() {{ Plotly.Plots.resize(chartDiv); }});

  var parsed = parseHash();
  if (parsed.mode === 'app' || parsed.mode === 'kernel') {{
    mode = parsed.mode;
    updateModeButtons();
  }}
  var initial = parsed.id;
  if (!initial || !(FIGURES_APP[initial] || FIGURES_KERNEL[initial])) {{
    initial = ORDER[0];
  }}
  if (initial) show(initial);
</script>
</body>
</html>
"""
