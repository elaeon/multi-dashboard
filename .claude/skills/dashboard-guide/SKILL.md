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

**Always useful:**
- Line/area trend over time (the "so what" chart — shows growth, drops, recovery)
- Horizontal bar top-N (sortable, easy to read at any length)
- Choropleth map with GeoJSON (`px.choropleth_map`, `map_style="carto-darkmatter"`)
- Treemap for part-of-whole with many categories
- Donut for 2–4 categories only
- Bubble scatter for 3-variable relationships (x, y, size)

**Skip unless clearly requested:**
- Pie charts with > 4 slices
- 3D charts
- Animated charts (they break static export)
- Tables (use bar charts instead)

**Height rules:**
- Fixed charts: 360–420 px
- Horizontal bars with variable rows: `max(300, n_rows * 28 + 80)`
- Maps: 580–620 px

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

Charts that should update on filter: trend, top-N bar, KPIs.  
Charts that stay static (full-dataset context): maps, technology comparisons, scatter.

---

## What NOT to do

| Don't | Do instead |
|---|---|
| Filter `df` in a loop for 300+ items | Batch `group_by` → build index dict |
| Average a numeric column across mixed units | Check `UNIDAD_MEDIDA` first, filter to dominant unit |
| Add features "just in case" | Build exactly what was asked |
| Expand an existing callback | Add a second callback |
| Use `px.choropleth_mapbox` | Use `px.choropleth_map` (Plotly ≥ 5.12) |
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
| `<dataset>.py` | Interactive Dash app |
| `export_<dataset>_html.py` | Static HTML exporter → `site/<dataset>.html` |
| `data/` | Source CSVs and GeoJSON |
| `site/` | Static output for GitHub Pages |
