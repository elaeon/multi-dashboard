# Static HTML export — plumbing reference

Implementation details for `static_dashboard/export_<dataset>_html.py`. The architectural
decision (static vs dynamic per chart) lives in the main SKILL.md — make it before opening
this file.

## Core pattern

Pre-compute aggregates per filter value → embed as JSON → `Plotly.newPlot()` on selection.

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

## Two filters → compound key

Pre-compute all combinations and index by `"A|B"`. With Polars, 130+ combos typically runs
in under 2 seconds — the naive loop is fine.

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

## Plotly CDN loading order

`include_plotlyjs="cdn"` only on the first pre-rendered chart. All subsequent
`pio.to_html()` calls pass `False`. The order in the HTML file determines which chart loads
the library — make it a regular bar or line chart, not a map.

```python
chart_map    = chart_div(fig_map(df),    "chart-map",    first=True)   # loads Plotly CDN
chart_states = chart_div(fig_states(df), "chart-states", first=False)  # reuses it
```

## applyWhenReady must anchor to a regular chart, not a map

Choropleth maps use WebGL/maplibre and never produce `svg.main-svg`. Wait on a bar or line
chart that renders as SVG instead.

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
