---
name: dashboard-guide
description: Best practices for building Polars + Plotly + Dash dashboards — insight-first pipeline, chart selection, annotation, narrative composition, optional static HTML export. Use at the start of any new dashboard task.
---

# Dashboard Building Guide

**Stack:** Python · Polars · Plotly (px + go) · Dash · dash-bootstrap-components · uv

---

## Pipeline — insights first, charts second

The dashboard is an argument with exhibits, not a data inventory. Build it in this order:

1. **Explore** (Phase 1) — schema, categories, ranges, artifacts.
2. **Mine insights** (Phase 2) — run the recipes; pass findings through the artifact gate.
3. **Write claims** (Phase 3) — one sentence per finding; present them to the user; claims pick the charts.
4. **Build** (Phases 4–6) — architecture, theme, charts, annotations.
5. **Verify** — every claim must still be true on the final, filtered data.

Never write a chart before you can state its claim. A hypothesis before exploring is fine; the *committed* claim comes after mining. If a chart has no claim, cut it.

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

### Large datasets (≥ 100k rows) — ask first, then optimize

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

## Phase 2 — Insight mining (before any chart exists)

Run a structured analysis on the dataframe to surface what the dashboard should say. The findings decide the layout — don't build first and analyze later.

### Level 1 — baselines and extremes

```python
# 1. Baseline: % or mean for every key column
for col in KEY_COLS:
    valid = df[col].drop_nulls()
    print(col, f"{float(valid.mean()*100):.1f}%", f"n={len(valid):,}")

# 2. Trend over time: does anything get better or worse?
agg = df.group_by("YEAR_COL").agg([pl.col(c).mean() for c in KEY_COLS]).sort("YEAR_COL")
print(agg)

# 3. Cross-tabulations: do bad things cluster together?
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

### Level 2 — the hidden findings

These are what separate a data inventory from a story. Run every one that applies.

```python
# 6. Mix-shift decomposition (Simpson's paradox check)
# Is the aggregate trend real behavior, or just composition changing?
overall = df.group_by("AÑO").agg(pl.col("VAL").mean()).sort("AÑO")
by_group = df.group_by("GRUPO", "AÑO").agg(pl.col("VAL").mean()).sort("GRUPO", "AÑO")
# If most groups move opposite to the aggregate, the composition shift IS the headline.

# 7. Deviation from expected (baseline projection)
# Fit a pre-period trend, project it, chart observed − expected.
import numpy as np
trend = df.group_by("AÑO").agg(pl.col("VAL").sum()).sort("AÑO")
base = trend.filter(pl.col("AÑO") <= BASELINE_END)          # e.g. 2019
coef = np.polyfit(base["AÑO"].to_list(), base["VAL"].to_list(), 1)
expected = np.polyval(coef, trend["AÑO"].to_list())
excess = trend["VAL"].to_numpy() - expected                  # the story is the gap

# 8. Concentration: who accounts for most of it?
ranked = df.group_by("ENTIDAD").agg(pl.col("VAL").sum()).sort("VAL", descending=True)
cum = ranked.with_columns((pl.col("VAL").cum_sum() / pl.col("VAL").sum() * 100).alias("cum_pct"))
print(cum.head(5))   # "top 3 states = 58%" beats a 32-bar ranking

# 9. Change-point: WHEN did the trend break?
vals, years = trend["VAL"].to_list(), trend["AÑO"].to_list()
deltas = sorted(((years[i], vals[i] - vals[i-1]) for i in range(1, len(vals))),
                key=lambda t: abs(t[1]), reverse=True)
print(deltas[:3])    # candidate break years → annotate the winner on the chart

# 10. Per-capita normalization check
# Rank by absolute AND by rate. If the top-5 differ, show the rate and say so.
# A raw-count choropleth is just a population map.

# 11. Common-trend residuals: who deviates from the shared pattern?
nat = df.group_by("AÑO").agg(pl.col("VAL").mean().alias("nat"))
resid = (df.group_by("ENTIDAD", "AÑO").agg(pl.col("VAL").mean())
           .join(nat, on="AÑO")
           .with_columns((pl.col("VAL") - pl.col("nat")).alias("resid")))
# Small multiples of `resid` surface the outlier entities; 32 parallel lines never do.
```

### Artifact gate — run before promoting any finding

Public datasets break more often than reality does. For each candidate finding, check:

```python
# Partial final year → fake "drop". Compare record counts per year.
print(df.group_by("AÑO").len().sort("AÑO"))

# Category renamed/merged mid-series → fake trend break.
print(df.group_by("AÑO").agg(pl.col("CAT").n_unique()).sort("AÑO"))

# Mixed units within a column → fake averages. Check UNIDAD_MEDIDA or per-category scales.
```

Also ask: did the collection methodology or registry change in the break year? If you can't rule out an artifact, present the finding with that caveat — don't silently promote it.

### What makes a finding "shocking"

- A rate above 90% or below 10% for something that should be the opposite
- A metric that got significantly worse across time periods (not noise — a sustained drop)
- A co-occurrence rate much higher than the individual rates would predict
- A geographic unit that is an extreme outlier with a large enough sample to trust
- An aggregate trend that reverses inside every subgroup (mix-shift)
- A policy that exists on paper but the data shows near-zero compliance

### What to skip

- Findings obvious from the domain (e.g. "more sales in December")
- Differences smaller than 5 percentage points between groups
- Patterns with n < 50 — too small to be reliable
- Anything that didn't survive the artifact gate

---

## Phase 3 — Claims and narrative composition

### One claim per finding

Order findings by urgency and present them to the user to choose which go on the dashboard:

1. **The most alarming single number** — one sentence, one stat.
2. **A trend that got worse over time** — name the before/after values and the break point.
3. **A clustering effect** — bad conditions that co-occur more than expected.
4. **Geographic or categorical extremes** — name the specific entity.
5. **A contradiction** — something that shouldn't be true simultaneously but is.
6. **The most actionable finding** — what could be fixed, and why it's tractable.

### The claim is the chart title

Topic labels waste the strongest line on the chart. The title states the finding; a muted subtitle carries the metric and units.

| ✗ Topic label | ✓ Claim |
|---|---|
| "Distribución por sexo" | "3 de cada 4 desaparecidos adultos son hombres" |
| "Defunciones por año" | "Las defunciones superaron la tendencia 2010–2019 por 600 mil" |
| "Tasa por entidad" | "Chihuahua duplica la tasa nacional de suicidio" |

```python
fig.update_layout(title=dict(
    text="<b>Chihuahua duplica la tasa nacional</b>"
         "<br><sup style='color:#94A3B8'>Tasa de suicidio por 100k hab., 2010–2024</sup>",
))
```

**Rule:** if the claim names a year, entity, or threshold, that thing must be marked on the chart (see annotation toolkit, Phase 6).

### Page composition — inverted pyramid

- **KPI row = the headline.** Every KPI card carries a comparison — delta vs. prior period or vs. benchmark. A lone number is not an insight.

```python
def kpi_card(label, value, prev, harm_when_up=True):
    delta = (value - prev) / prev * 100
    worsened = (delta > 0) == harm_when_up
    color = "#E84855" if worsened else "#3BB273"
    arrow = "▲" if delta > 0 else "▼"
    # render value large, then html.Span(f"{arrow} {abs(delta):.1f}% vs periodo anterior",
    #                                    style={"color": color, "fontSize": "0.85rem"})
```

- **Sections answer questions in order:** what happened → since when → where / who → why or what's associated. Each tab answers the question the previous one raises.
- **≤ 2 charts per claim.** If two charts say the same thing, keep the stronger one.

---

## Phase 4 — Architecture decisions

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

### Cross-filtering — let the reader drill down

A dashboard the reader can interrogate beats one they can only look at. One pattern: click an entity on the map (or a bar in the ranking) → detail charts re-render for that entity.

```python
@app.callback(
    Output("detail-graph", "figure"),
    Output("drill-label", "children"),
    Input("map-graph", "clickData"),
    Input("clear-drill", "n_clicks"),
)
def drill_down(click, _clear):
    from dash import ctx
    if ctx.triggered_id == "clear-drill" or not click:
        return fig_detail(df), "Nacional"
    entity = click["points"][0]["location"]   # or ["customdata"][0]
    return fig_detail(df.filter(pl.col("ENTIDAD") == entity)), entity
```

Layout needs an `html.Button("✕ Ver nacional", id="clear-drill")` next to the drill label so the reader can always get back. Follows the new-feature → new-callback rule: drill-down is its own callback, never merged into `update_all`.

---

## Phase 5 — Dark theme and color semantics

### Theme (reuse verbatim)

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

### Color is emphasis, not decoration (focus + context)

In any multi-entity chart, the entity the claim is about gets the accent color; everything else gets context gray. Rainbow categorical palettes say "these are all equally important" — which is never the claim.

```python
FOCUS, CONTEXT = "#2E86AB", "#475569"
colors = [FOCUS if e == focus_entity else CONTEXT for e in entities]
# Works for ranking bars, slope charts, line charts, scatter points.
```

Reserve red `#E84855` / green `#3BB273` for direction-of-harm semantics (slope charts, deltas) — don't spend them on neutral categories.

---

## Phase 6 — Chart selection and annotation

### Decision table

| Data relationship | First choice | Notes |
|---|---|---|
| Change over time, 1–3 series | Line | — |
| Change over time, many overlapping series | Small multiples | Never >5 lines on one chart |
| Change over time, few discrete points | Column bar | — |
| Observed vs expected over time | Line + baseline projection | Annotate the gap (recipe 7) |
| Before → after, 2 time points, absolute scale | **Dumbbell plot** | Shows starting value + magnitude of change |
| Before → after, many entities, direction story | **Slope chart** | Red = direction of harm, green = improvement (sign depends on metric) |
| Rankings with mean + min/max across a period | **Dot-and-range** | Mean dot + min/max ticks; use when rank order matters |
| Rankings, single value | Horizontal bar (sorted) | Focus color on the claim's entity |
| Part-of-whole, any number of categories | **Stacked 100% horizontal bar** | Donut only if ≤4 categories AND only a rough gestalt matters |
| Hierarchical part-of-whole | Treemap | Only when hierarchy matters |
| Geographic, admin regions | Choropleth (`px.choropleth_map`) | `map_style="carto-darkmatter"`; rates, not raw counts |
| Geographic, point locations with a category | **Symbol map** (`go.Scattermap`) | Color = category, filter to valid bounds first |
| Correlation / 3 variables | Scatter → Bubble | — |
| Distribution, one variable | Histogram | — |
| Full distribution shape across categories | Box plot (or violin) | Median, IQR, outliers — use when distribution shape matters |
| Seasonality (year × month) | Heatmap (`go.Heatmap`) | Best pattern for cyclical data |

**Skip always:** 3D charts.
**Skip in static export:** animated charts (break export); fine in interactive Dash if used sparingly.
**Prefer sorted bars over tables for rankings.** Use tables only when readers need to look up exact values.

### Annotation toolkit — mark the claim on the chart

Every specific year, entity, or threshold named in the chart's claim gets marked. Unannotated charts make the reader re-derive the finding.

```python
# Break year (from change-point recipe 9)
fig.add_vline(x=2018, line_dash="dot", line_color="#94A3B8",
              annotation_text="2018: cambio de tendencia",
              annotation_font_color="#94A3B8", annotation_position="top left")

# Event window
fig.add_vrect(x0=2020, x1=2021, fillcolor="rgba(244,162,97,0.12)", line_width=0,
              annotation_text="pandemia", annotation_font_color="#94A3B8")

# Reference level (national mean, target, threshold)
fig.add_hline(y=nat_mean, line_dash="dash", line_color="#64748B",
              annotation_text="media nacional", annotation_font_color="#94A3B8")

# Callout on the outlier the claim names
fig.add_annotation(x=x_out, y=y_out, text="<b>Guanajuato</b>: 3× la media",
                   font=dict(color="#F4A261", size=12),
                   arrowcolor="#94A3B8", ax=40, ay=-30)
```

### Stacked 100% bar (part-of-whole default)

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

### Slope chart (first year → last year, direction story)

Use when the story is **which direction did each entity move** across two time points.
Sort by end value. Color the *direction of harm* in red and *improvement* in green — which sign that is depends on the metric (rising poverty = harm; rising graduation = improvement).

```python
WORSE_WHEN_UP = True   # set False for benefit metrics (graduation, income, etc.)
for row in result.iter_rows(named=True):
    worsened = (row["delta"] > 0) == WORSE_WHEN_UP
    color = "#E84855" if worsened else "#3BB273"
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

## Phase 7 — Static HTML export

**Decide static vs dynamic before writing the exporter.** This decision shapes the whole architecture — changing it later requires restructuring the JS.

| Chart type | Rule |
|---|---|
| Trend, monthly, age, KPIs | Dynamic — update on every filter change |
| Choropleth map | Static — always full dataset |
| State/category ranking bar | Static — always full dataset |
| Scatter, technology comparison | Static — always full dataset |

Implementation plumbing (JSON embedding, compound keys, Plotly CDN loading order, `applyWhenReady`): read [references/static-export.md](references/static-export.md) when you start writing the exporter.

---

## What NOT to do

| Don't | Do instead |
|---|---|
| Use a topic label as a chart title ("Casos por año") | Title states the claim; muted subtitle carries metric/units |
| Name a year/entity/threshold in a claim without marking it | Annotate it on the chart (vline, hline, callout) |
| Ship a KPI card with a lone number | Add delta vs prior period or benchmark |
| Give every entity its own color | Focus color on the claim's entity, context gray for the rest |
| Sort categorical bars alphabetically | Sort by value (descending for rankings) |
| Truncate the y-axis on a bar chart | Bars start at 0; lines/scatter use a data-appropriate range |
| Dual y-axis on one chart | Two stacked panels or small multiples |
| Use a donut with 5+ categories or when small differences matter | Stacked 100% horizontal bar — 3% is invisible in a circle, readable in a bar |
| Put 6+ overlapping lines on one chart | Small multiples — one panel per series |
| Use a ranked bar when change over time matters | Slope chart (direction) or dot-and-range (stability) |
| Map raw counts on a choropleth | Per-capita rate — raw counts are a population map |
| Plot lat/lon data only on a choropleth | Add a symbol map — spatial clustering is invisible in state-level aggregates |
| Report a trend ending in a suspicious final-year drop | Check record counts per year — partial years fake declines |
| Trust an aggregate trend without a subgroup check | Run the mix-shift decomposition (recipe 6) |
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

**Claim audit:** re-compute every chart title's number/claim on the final data (after all filters and cleaning applied at startup). A title that was true during exploration can become false after filtering — stale claims are worse than topic labels.

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
