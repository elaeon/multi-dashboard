---
name: dashboard-guide
description: Best practices for building Polars + Plotly + Dash dashboards with optional static HTML export. Use at the start of any new dashboard task.
---

# Dashboard Building Guide

**Stack:** Python · Polars · Plotly (px + go) · Dash · dash-bootstrap-components · uv

---

## Phase 1 — Explore before coding

Run these before writing a single chart. Wrong assumptions here cause rewrites.

```python
# Column names and sample rows
import polars as pl
df = pl.read_csv("data/file.csv")
print(df.schema)
print(df.head(5))

# Unique values of every categorical column
for col in df.columns:
    if df[col].dtype == pl.String:
        print(col, df[col].n_unique(), df[col].unique().to_list()[:10])

# Numeric ranges — catch mixed-unit columns early
for col in df.columns:
    if df[col].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
        print(col, df[col].min(), df[col].max(), df[col].mean())
```

**What to flag:**
- Categorical columns with fewer values than expected (e.g., TIPO_MERCADO = 2, not 4)
- Numeric columns with wildly different scales per category → likely mixed units (filter before averaging)
- Nulls in columns you plan to aggregate
- Date/year columns that need casting (`pl.col("AÑO").cast(pl.Int32)`)

---

## Large datasets (≥ 100k rows) — ask first, then optimize

If the dataset has **100k+ rows**, stop before writing any chart and confirm the optimization approach with the user. Show this checklist and agree on which steps apply:

1. **`scan_parquet` / `scan_csv`** — use lazy evaluation; never `read_parquet` at module level.
2. **Replace `unpivot` with `sum_horizontal`** — `unpivot` on wide data multiplies rows (e.g. ×12 for monthly columns); `sum_horizontal` stays at the original count.
3. **Pre-aggregate at startup, not in callbacks** — `group_by().agg()` once at startup into small summary DataFrames; callbacks filter those, never the raw data.
4. **`pl.collect_all([q1, q2, q3])`** — collect multiple lazy queries in a single file scan instead of N separate passes.
5. **Never call `.to_pandas()`** — Polars DataFrames pass directly to Plotly Express; converting adds memory and requires pandas as a dependency.
6. **Only `.collect()` on aggregated frames** — the raw scan should never be collected; only the already-grouped result frames get `.collect()`.

```python
# Pattern: scan → horizontal sum → collect_all
lf = (
    pl.scan_parquet("data/file.parquet")
    .with_columns(pl.sum_horizontal(*[pl.col(m).fill_null(0) for m in MONTH_COLS]).alias("total"))
    .select(["KEY_COL1", "KEY_COL2", "total"])
)
q1 = lf.group_by(["KEY_COL1"]).agg(pl.col("total").sum())
q2 = lf.group_by(["KEY_COL1", "KEY_COL2"]).agg(pl.col("total").sum())
agg1, agg2 = pl.collect_all([q1, q2])   # single scan, two results
```

---

## Phase 2 — Architecture decisions

### Figure factories
One function per chart. Always accepts a filtered `pl.DataFrame`, returns `go.Figure`.

```python
def fig_something(d: pl.DataFrame) -> go.Figure:
    agg = d.group_by("COL").agg(pl.col("VAL").sum())
    fig = px.bar(agg, ...)
    fig.update_layout(height=420, **CHART_LAYOUT)
    return fig
```

Never hardcode data inside figure functions. Never call global `df` inside them.

### Single callback
Filter once, pass the filtered DataFrame to all figure factories.

```python
@app.callback(Output(...), ..., Input("year-range", "value"), Input("dropdown", "value"))
def update_all(year_range, filter_val):
    d = df.filter(pl.col("AÑO").is_between(*year_range))
    if filter_val:
        d = d.filter(pl.col("COL") == filter_val)
    return fig_a(d), fig_b(d), fig_c(d), ...
```

### New feature → new callback
Never expand an existing callback. Add a second callback with its own inputs/outputs.

```python
@app.callback(Output("new-graph", "figure"), Input("new-dropdown", "value"), ...)
def update_new_feature(...):
    ...
```

---

## Phase 3 — Dark theme (reuse verbatim)

```python
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#CBD5E1",
    xaxis=dict(gridcolor="#334155"),
    yaxis=dict(gridcolor="#334155"),
)
CARD_STYLE = {
    "background": "#1E293B", "border": "1px solid #334155",
    "borderRadius": "8px", "padding": "16px", "textAlign": "center",
}
TAB_STYLE = {"backgroundColor": "#0F172A", "color": "#94A3B8", "borderTop": "none"}
TAB_SEL   = {"backgroundColor": "#1E293B", "color": "#F8FAFC",
             "borderTop": "2px solid #2E86AB", "fontWeight": "600"}
```

Body background: `#0F172A`. Cards: `#1E293B`. Text: `#CBD5E1`. Muted: `#94A3B8 / #64748B`.
Accent blue: `#2E86AB`. Green: `#3BB273`. Orange: `#F4A261`. Red: `#E84855`.

---

## Phase 4 — Chart selection

### Decision table

| Data relationship | First choice | Notes |
|---|---|---|
| Change over time, 1–3 series | Line / area | — |
| Change over time, many overlapping series | Small multiples | Never >5 lines on one chart |
| Change over time, few discrete points | Column bar | — |
| Before → after, 2 time points, absolute scale | **Dumbbell plot** | Shows starting value + magnitude of change |
| Before → after, many entities, direction story | **Slope chart** | Red = got worse, green = got better |
| Rankings + variability across a period | **Dot-and-range** | Mean dot + min/max tick marks |
| Rankings, single value | Horizontal bar (sorted) | — |
| Part-of-whole, any number of categories | **Stacked 100% horizontal bar** | Default choice; beats donut in almost every case |
| Hierarchical part-of-whole | Treemap | Only when hierarchy matters |
| Geographic, admin regions | Choropleth (`px.choropleth_map`) | `map_style="carto-darkmatter"` |
| Geographic, point locations with a category | **Symbol map** (`go.Scattermap`) | Color = category, filter to valid bounds first |
| Correlation / 3 variables | Scatter → Bubble | — |
| Distribution, one variable | Histogram | — |
| Seasonality (year × month) | Heatmap (`go.Heatmap`) | Best pattern for cyclical data |

**Skip always:** 3D charts, animated charts (break static export), tables (use sorted bars instead).

---

### Donut / pie — narrow use case

**Default: use a stacked 100% horizontal bar instead.** It reads more precisely, scales to any number of categories, and is consistent with the rest of the dashboard.

**Use a donut only when:** ≤4 categories AND the reader only needs a rough gestalt ("roughly half vs half") AND exact differences don't matter.

**Always replace a donut with a stacked bar when:**
- Any two categories are within ~15% of each other
- More than 4 categories
- The chart sits in a tab or row with other bar charts (consistency)
- The absolute counts also matter (bar can show both)

Stacked 100% bar pattern (works for 2–N categories):
```python
for cat in categories:          # iterate in display order
    pct = count_map[cat] / total * 100
    fig.add_trace(go.Bar(
        x=[pct], y=["Label"], orientation="h", name=cat,
        marker_color=COLORS[cat],
        text=f"{pct:.1f}%", textposition="inside", insidetextanchor="middle",
        customdata=[count_map[cat]],
        hovertemplate=f"<b>{cat}</b>: %{{x:.1f}}%  (n=%{{customdata[0]:,}})<extra></extra>",
    ))
fig.update_layout(barmode="stack", xaxis=dict(range=[0, 100], visible=False))
```

---

### Dumbbell / arrow plot (before → after, absolute scale)

Use when the story needs BOTH the starting value AND the change for many entities.
Classic cases: temperature baseline vs current, pay in two periods, scores before/after intervention.

```python
# Connector lines using None-separator trick
x_lines, y_lines = [], []
for start, end, label in zip(starts, ends, labels):
    x_lines += [start, end, None]
    y_lines += [label, label, None]

fig.add_trace(go.Scatter(x=x_lines, y=y_lines, mode="lines",
                          line=dict(color="#475569", width=1.5),
                          showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=starts, y=labels, mode="markers", name="Periodo A",
                          marker=dict(color="#64748B", size=10, symbol="circle-open",
                                      line=dict(color="#64748B", width=2))))
fig.add_trace(go.Scatter(x=ends, y=labels, mode="markers", name="Periodo B",
                          marker=dict(color=colors, size=10),
                          customdata=deltas,
                          hovertemplate="<b>%{y}</b><br>%{x:.2f}  Δ: %{customdata:.2f}<extra></extra>"))
```

---

### Slope chart (first year → last year, direction story)

Use when the story is **which direction did each entity move** across two time points.
Sort by end value. Color red = worsened, green = improved.

```python
for row in result.iter_rows(named=True):
    color = "#E84855" if row["delta"] > 0 else "#3BB273"
    fig.add_trace(go.Scatter(
        x=[str(y0), str(y1)], y=[row["r0"], row["r1"]],
        mode="lines+markers",
        line=dict(color=color, width=1.5), marker=dict(color=color, size=7),
        showlegend=False,
        hovertemplate=f"<b>{row['ENTITY']}</b><br>%{{x}}: %{{y:.1f}}<extra></extra>",
    ))
fig.update_xaxes(type="category", gridcolor="rgba(0,0,0,0)")

# Legend proxy traces to summarize count
fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines+markers",
                          line=dict(color="#E84855"), marker=dict(color="#E84855"),
                          name=f"▲ Aumentó ({n_up})"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines+markers",
                          line=dict(color="#3BB273"), marker=dict(color="#3BB273"),
                          name=f"▼ Disminuyó ({n_dn})"))
```

---

### Dot-and-range (mean + min/max across a period)

Use when you need to show both **typical level** and **variability**. Pair with a slope chart: slope = direction, dot-and-range = stability.

```python
x_lines, y_lines = [], []
for mn, mx, s in zip(mins, maxs, states):
    x_lines += [mn, mx, None]
    y_lines += [s, s, None]

fig.add_trace(go.Scatter(x=x_lines, y=y_lines, mode="lines",
                          line=dict(color="#334155", width=2),
                          showlegend=False, hoverinfo="skip"))
for x_vals, name in [(mins, "Mínimo"), (maxs, "Máximo")]:
    fig.add_trace(go.Scatter(x=x_vals, y=states, mode="markers", name=name,
                              marker=dict(symbol="line-ew", size=8, color="#475569",
                                          line=dict(width=2, color="#475569"))))
fig.add_trace(go.Scatter(x=means, y=states, mode="markers", name="Promedio",
                          marker=dict(color=colors, size=10)))
fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
```

---

### Symbol map (point locations with a categorical variable)

Use when data has lat/lon coordinates and you want to show spatial clustering. Filter to valid coordinate bounds before plotting.

```python
# Filter to valid bounds first (example: Mexico)
geo = d.with_columns(
    pl.col("LAT_COL").cast(pl.Float64, strict=False).alias("lat"),
    pl.col("LON_COL").cast(pl.Float64, strict=False).alias("lon"),
).filter(pl.col("lat").is_between(14, 33) & pl.col("lon").is_between(-118, -86))

center_lat = float(geo["lat"].mean())
center_lon = float(geo["lon"].mean())

for cat, color in COLORS.items():
    subset = geo.filter(pl.col("CATEGORY") == cat)
    fig.add_trace(go.Scattermap(
        lat=subset["lat"].to_list(), lon=subset["lon"].to_list(),
        mode="markers", marker=dict(size=7, color=color, opacity=0.7),
        name=cat,
        customdata=list(zip(subset["LABEL1"].fill_null("—").to_list(),
                            subset["LABEL2"].fill_null("—").to_list())),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
    ))
fig.update_layout(map=dict(style="carto-darkmatter",
                            center=dict(lat=center_lat, lon=center_lon), zoom=6),
                  paper_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1")
```

---

### Height rules
- Fixed charts: 360–420 px
- Horizontal bars with variable rows: `max(300, n_rows * 28 + 80)`
- Dot-and-range / slope charts: `max(300, n_entities * 22 + 80)`
- Maps (choropleth): 580–620 px
- Maps (symbol): 480–520 px

### Stacked bar legend placement
Legend at `y=1.1` overlaps the title or tallest bars. Move it below:
```python
legend=dict(orientation="h", y=-0.18, x=0),
margin=dict(t=40, b=70, l=10, r=10),
```

---

## Phase 5 — Static HTML export

Pattern: pre-compute aggregates per filter value → embed as JSON → `Plotly.newPlot()` on selection.

```python
# Python side: batch aggregation (never filter per-item in a loop)
result = {}
grouped = df.group_by("FILTER_COL", "AÑO").agg(pl.col("VAL").sum())
for row in grouped.group_by("FILTER_COL").agg(...).iter_rows(named=True):
    result[row["FILTER_COL"]] = {"y": row["years"], "v": row["vals"]}
json.dumps(result, separators=(",", ":"))  # compact
```

```js
// JS side
function updateCharts(filterVal) {
    const D = DATA[filterVal];
    const layout = Object.assign({}, document.getElementById("chart-id").layout,
                                 {title: {text: "Title for " + filterVal}});
    Plotly.newPlot("chart-id", [{type:"scatter", x:D.y, y:D.v, ...}], layout, CFG);
}
document.getElementById("select").addEventListener("change", function() {
    const url = new URL(window.location.href);
    url.searchParams.set("filter", this.value);
    window.location.href = url.toString();  // URL = shareable state
});
window.addEventListener("load", function() {
    const val = new URLSearchParams(window.location.search).get("filter");
    if (val && DATA[val]) { document.getElementById("select").value = val; updateCharts(val); }
});
```

**Decide static vs dynamic before writing the exporter.** This decision shapes the whole architecture — changing it later requires restructuring the JS.

| Chart type | Rule |
|---|---|
| Trend, monthly, age, KPIs | Dynamic — update on every filter change |
| Choropleth map | Static — always full dataset |
| State/category ranking bar | Static — always full dataset |
| Scatter, technology comparison | Static — always full dataset |

**Two filters → compound key.** Pre-compute all combinations and index by `"A|B"`. With Polars, 130+ combos typically runs in under 2 seconds — the naive loop is fine.

```python
for filter_a in ALL_A:
    for filter_b in ALL_B:
        d = df.filter(...)
        result[f"{filter_a}|{filter_b}"] = build_payload(d)
```

```js
const key = selectA.value + "|" + selectB.value;
updateCharts(ALL_DATA[key]);
```

**`include_plotlyjs="cdn"` only on the first pre-rendered chart.** All subsequent `pio.to_html()` calls pass `False`. The order in the HTML file determines which chart loads the library — make it a regular bar or line chart, not a map.

```python
chart_map    = chart_div(fig_map(df),    "chart-map",    first=True)   # loads Plotly CDN
chart_states = chart_div(fig_states(df), "chart-states", first=False)  # reuses it
```

**`applyWhenReady` must anchor to a regular chart, not a map.** Choropleth maps use WebGL/maplibre and never produce `svg.main-svg`. Wait on a bar or line chart that renders as SVG instead.

```js
function applyWhenReady(key, attempt) {
    attempt = attempt || 0;
    if (attempt > 120) { updateCharts(key); return; }
    // ✓ bar/line chart — renders as SVG
    const el = document.getElementById("chart-bar");
    // ✗ don't use "chart-map" — choropleth uses WebGL, no svg.main-svg
    if (el && el.querySelector("svg.main-svg")) {
        updateCharts(key);
    } else {
        setTimeout(() => applyWhenReady(key, attempt + 1), 80);
    }
}
```

---

## What NOT to do

| Don't | Do instead |
|---|---|
| Use a donut with 5+ categories | Horizontal bar sorted by value |
| Use a donut when small differences matter | Stacked 100% horizontal bar — 3% is invisible in a circle, readable in a bar |
| Put 6+ overlapping lines on one chart | Small multiples — one panel per series |
| Use a ranked bar when change over time matters | Slope chart (direction) or dot-and-range (stability) |
| Plot lat/lon data only on a choropleth | Add a symbol map — spatial clustering is invisible in state-level aggregates |
| Filter `df` in a loop for 300+ items | Batch `group_by` → build index dict |
| Average a numeric column across mixed units | Check `UNIDAD_MEDIDA` first, filter to dominant unit |
| Add features "just in case" | Build exactly what was asked |
| Expand an existing callback | Add a second callback |
| Use `px.choropleth_mapbox` | Use `px.choropleth_map` (Plotly ≥ 5.12) |
| Use `go.Scattermapbox` | Use `go.Scattermap` with `layout.map` (Plotly ≥ 5.12) |
| Render server-side for each static filter | Pre-compute JSON, swap client-side |
| Guess categorical values | Check `.n_unique()` and `.unique()` first |
| Commit without testing figures | Run all figure factories on full df + each filter value |

---

## Phase 6 — Insight analysis (always run after the dashboard is built)

Once the dashboard is working, run a structured analysis on the dataframe to surface findings the user should know about. Don't wait to be asked.

### What to compute

```python
# 1. Baseline: % or mean for every key column
for col in KEY_COLS:
    valid = df[col].drop_nulls()
    print(col, f"{float(valid.mean()*100):.1f}%", f"n={len(valid):,}")

# 2. Trend over time: does anything get better or worse?
agg = df.group_by("YEAR_COL").agg([pl.col(c).mean() for c in KEY_COLS]).sort("YEAR_COL")
print(agg)

# 3. Cross-tabulations: do bad things cluster together?
# e.g. records where condition_A AND condition_B AND condition_C
bad = df.filter((pl.col("A") == BAD) & (pl.col("B") == BAD) & (pl.col("C") == BAD))
base = df.filter(pl.col("A").is_not_null() & pl.col("B").is_not_null() & pl.col("C").is_not_null())
print(f"Triple problem: {len(bad)/len(base)*100:.1f}%")

# 4. Geographic/categorical breakdown: who has it worst?
agg_geo = (
    df.filter(pl.col("KEY").is_not_null() & pl.col("GEO").str.len_chars().gt(3))
    .group_by("GEO")
    .agg(pl.col("KEY").mean().alias("pct"), pl.col("KEY").count().alias("n"))
    .filter(pl.col("n") >= 50)
    .sort("pct", descending=True)
)
print(agg_geo.head(10))

# 5. Contradictions: records where two logically opposed things are both true
contra = df.filter((pl.col("BAD_THING") == 1) & (pl.col("CLAIMS_GOOD") == 1))
print(f"Contradiction: {len(contra)/len(base)*100:.1f}%")
```

### How to present findings

Order by urgency, not by column order. Lead with numbers. Use this structure:

1. **The most alarming single number** — one sentence, one stat.
2. **A trend that got worse over time** — name the before/after values and the break point.
3. **A clustering effect** — two or three bad conditions that co-occur more than expected.
4. **Geographic or categorical extremes** — who has it worst (name the specific entity).
5. **A contradiction** — something that shouldn't be true simultaneously but is.
6. **The most actionable finding** — what could actually be fixed, and why it's tractable.

Present your findings to the user to choice which are worth to add in the dashboard

### What makes a finding "shocking"

- A rate above 90% or below 10% for something that should be the opposite
- A metric that got significantly worse across time periods (not noise — a sustained drop)
- A co-occurrence rate that is much higher than the individual rates would predict
- A geographic unit that is an extreme outlier with a large enough sample to trust
- A regulation or policy that exists on paper but the data shows near-zero compliance

### What to skip

- Findings that are obvious from the domain (e.g. "more sales in December")
- Differences smaller than 5 percentage points between groups
- Patterns with n < 50 — too small to be reliable
- Restatements of what the charts already show without adding interpretation

---

## Verification checklist

```python
# Before running the server
import your_dashboard as m
import polars as pl

d = m.df
all_fns = [m.fig_a, m.fig_b, ...]  # list every figure factory
for fn in all_fns:
    fn(d)                                    # full dataset
    fn(d.filter(pl.col("STATE") == "X"))     # filtered

# KPIs
print(m.compute_kpis(d))   # check values make sense (units, scale)
```

```bash
# Smoke-test the server
timeout 8 uv run python your_dashboard.py 2>/dev/null
# exit 124 = timeout (running fine), exit 1 = error
```

---

## File naming convention (this project)

| File | Purpose |
|---|---|
| `dashboard/<dataset>.py` | Interactive Dash app |
| `dashboard/__init__.py` | Package init (required for imports) |
| `dashboard/assets/` | CSS served by Dash |
| `static_dashboard/export_<dataset>_html.py` | Static HTML exporter → `site/<dataset>.html` |
| `scripts/` | One-off data preparation scripts (e.g. merge xlsx → parquet) |
| `data/` | Source CSVs, xlsx, GeoJSON, and parquet files |
| `site/` | Static output for GitHub Pages |

**Run commands (always from project root):**
```bash
uv run python dashboard/<dataset>.py              # interactive Dash app
uv run python static_dashboard/export_<dataset>_html.py  # generate static HTML
uv run python scripts/merge_incidencia.py <input_dir> <output.parquet>
```

**Import pattern in export scripts:**
```python
from dashboard.<dataset> import df, fig_a, fig_b, ...
```
