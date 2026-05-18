"""
Exports a fully self-contained index.html for GitHub Pages.

The "centro de trabajo" dropdown works entirely client-side:
  - Per-centro aggregates are pre-computed in Python and embedded as JSON
  - Plotly.restyle() swaps chart data on selection — no server needed

Charts that respond to the dropdown:
  annual trend, YoY change, visitor-type donut, monthly pattern, foreign-% trend

Charts that stay as full-dataset context (not meaningful per-site):
  Mexico map, top states, foreign % by state, top sites, sites-by-state, etc.

Run: uv run python export_html.py
"""

import json
import polars as pl
import plotly.io as pio

from dashboard import (
    df, compute_kpis, TOTAL_SITES, YEAR_MIN, YEAR_MAX, VISITOR_SHORT,
    fig_annual_trend, fig_yoy_change, fig_site_donut, fig_visitor_type_donut,
    fig_top_states, fig_foreign_ratio,
    fig_top_sites, fig_foreign_by_site_type, fig_sites_by_state,
    fig_visitor_trend, fig_paid_free_ratio, fig_foreign_trend,
    fig_seasonality_heatmap, fig_monthly_pattern, fig_states_map,
)

d = df  # full dataset (already excludes partial 2026)
total, pct_foreign, peak_yr = compute_kpis(d)

CHART_CFG = {"responsive": True, "displayModeBar": False}


# ── HTML helpers ──────────────────────────────────────────────────────────────

def div(fig, div_id: str, first=False) -> str:
    return pio.to_html(
        fig, full_html=False,
        include_plotlyjs="cdn" if first else False,
        config=CHART_CFG,
        div_id=div_id,
    )


def row(*cols) -> str:
    """cols: (html_string, bootstrap_width)"""
    inner = "".join(f'<div class="col-{w} mb-3">{h}</div>' for h, w in cols)
    return f'<div class="row g-3">{inner}</div>'


def kpi(title, value, sub, kpi_id=None):
    val_id  = f' id="{kpi_id}"'      if kpi_id else ""
    sub_id  = f' id="{kpi_id}-sub"'  if kpi_id else ""
    return f"""<div class="col-3">
      <div class="kpi-card">
        <p class="kpi-label">{title}</p>
        <h3 class="kpi-value"{val_id}>{value}</h3>
        <small class="kpi-sub"{sub_id}>{sub}</small>
      </div>
    </div>"""


def tab_btn(tab_id, label, active=False):
    cls = "nav-link active" if active else "nav-link"
    return (f'<button class="{cls}" id="btn-{tab_id}" data-bs-toggle="tab" '
            f'data-bs-target="#tab-{tab_id}" type="button" role="tab">{label}</button>')


def tab_pane(tab_id, content, active=False):
    cls = "tab-pane fade show active" if active else "tab-pane fade"
    return f'<div class="{cls}" id="tab-{tab_id}" role="tabpanel">{content}</div>'


def insight_card(title, body):
    return f"""<div class="col-4">
      <div class="insight-card">
        <strong class="insight-title">{title}</strong>
        <p class="insight-body">{body}</p>
      </div>
    </div>"""


# ── Pre-compute per-centro data for client-side filtering ─────────────────────

def build_centro_data(df: pl.DataFrame) -> str:
    """
    Returns a JSON string keyed by centro name (plus '__all__').
    Each entry has compact keys to keep the payload small:
      t  → trend   {y: years, n: nacional, e: extranjeros}
      mo → monthly {v: values for months 1-12}
      vt → visitor types {l: labels, v: values}
    """
    print("  Pre-computing trend data…")
    trend_pivot = (
        df.group_by("CENTRO DE TRABAJO", "year", "NACIONALIDAD")
        .agg(pl.col("NÚMERO DE VISITAS").sum())
        .pivot(values="NÚMERO DE VISITAS", index=["CENTRO DE TRABAJO", "year"],
               on="NACIONALIDAD")
        .fill_null(0)
        .sort("CENTRO DE TRABAJO", "year")
    )
    trend_grouped = (
        trend_pivot
        .group_by("CENTRO DE TRABAJO", maintain_order=True)
        .agg(pl.col("year"), pl.col("Nacional"), pl.col("Extranjeros"))
    )

    print("  Pre-computing monthly data…")
    monthly_grouped = (
        df.group_by("CENTRO DE TRABAJO", "month")
        .agg(pl.col("NÚMERO DE VISITAS").sum())
        .sort("CENTRO DE TRABAJO", "month")
        .group_by("CENTRO DE TRABAJO", maintain_order=True)
        .agg(pl.col("month"), pl.col("NÚMERO DE VISITAS"))
    )

    print("  Pre-computing visitor-type data…")
    vtypes_grouped = (
        df.group_by("CENTRO DE TRABAJO", "TIPO DE VISITANTES")
        .agg(pl.col("NÚMERO DE VISITAS").sum())
        .with_columns(pl.col("TIPO DE VISITANTES").replace(VISITOR_SHORT).alias("label"))
        .sort("NÚMERO DE VISITAS", descending=True)
        .group_by("CENTRO DE TRABAJO", maintain_order=True)
        .agg(pl.col("label"), pl.col("NÚMERO DE VISITAS"))
    )

    print("  Pre-computing paid-% data…")
    paid_cats = ["Boleto pagado", "Exposiciones temporales con costo adicional"]
    paid_grouped = (
        df.group_by("CENTRO DE TRABAJO", "year")
        .agg(
            pl.col("NÚMERO DE VISITAS").sum().alias("total"),
            pl.col("NÚMERO DE VISITAS")
              .filter(pl.col("TIPO DE VISITANTES").is_in(paid_cats))
              .sum().alias("paid"),
        )
        .with_columns((pl.col("paid") / pl.col("total") * 100).round(1).alias("pct"))
        .sort("CENTRO DE TRABAJO", "year")
        .group_by("CENTRO DE TRABAJO", maintain_order=True)
        .agg(pl.col("year"), pl.col("pct"))
    )

    print("  Indexing site types…")
    site_type_idx = {
        r["CENTRO DE TRABAJO"]: r["TIPO DE SITIO"]
        for r in (
            df.select("CENTRO DE TRABAJO", "TIPO DE SITIO")
            .group_by("CENTRO DE TRABAJO")
            .agg(pl.col("TIPO DE SITIO").first())
            .iter_rows(named=True)
        )
    }

    # Index by centro name for O(1) lookup
    monthly_idx = {r["CENTRO DE TRABAJO"]: r for r in monthly_grouped.iter_rows(named=True)}
    vtypes_idx  = {r["CENTRO DE TRABAJO"]: r for r in vtypes_grouped.iter_rows(named=True)}
    paid_idx    = {r["CENTRO DE TRABAJO"]: r for r in paid_grouped.iter_rows(named=True)}

    result = {}

    for row in trend_grouped.iter_rows(named=True):
        c = row["CENTRO DE TRABAJO"]
        years = row["year"]

        # monthly: fill all 12 slots with 0 if a month is missing
        mo_row = monthly_idx.get(c, {})
        month_map = dict(zip(mo_row.get("month", []),
                             mo_row.get("NÚMERO DE VISITAS", [])))
        monthly_vals = [month_map.get(m, 0) for m in range(1, 13)]

        vt_row = vtypes_idx.get(c, {})

        pd_row = paid_idx.get(c, {"year": [], "pct": []})
        result[c] = {
            "t":  {"y": years,
                   "n": row["Nacional"],
                   "e": row["Extranjeros"]},
            "mo": {"v": monthly_vals},
            "vt": {"l": vt_row.get("label", []),
                   "v": vt_row.get("NÚMERO DE VISITAS", [])},
            "si": site_type_idx.get(c, ""),
            "pd": {"y": pd_row["year"], "v": pd_row["pct"]},
        }

    # "__all__" entry — full-dataset aggregates
    all_trend = (
        df.group_by("year", "NACIONALIDAD")
        .agg(pl.col("NÚMERO DE VISITAS").sum())
        .pivot(values="NÚMERO DE VISITAS", index="year", on="NACIONALIDAD")
        .fill_null(0)
        .sort("year")
    )
    all_monthly = (
        df.group_by("month").agg(pl.col("NÚMERO DE VISITAS").sum()).sort("month")
    )
    all_vtypes = (
        df.group_by("TIPO DE VISITANTES")
        .agg(pl.col("NÚMERO DE VISITAS").sum())
        .with_columns(pl.col("TIPO DE VISITANTES").replace(VISITOR_SHORT).alias("label"))
        .sort("NÚMERO DE VISITAS", descending=True)
    )
    all_paid = (
        df.group_by("year")
        .agg(
            pl.col("NÚMERO DE VISITAS").sum().alias("total"),
            pl.col("NÚMERO DE VISITAS")
              .filter(pl.col("TIPO DE VISITANTES").is_in(paid_cats))
              .sum().alias("paid"),
        )
        .with_columns((pl.col("paid") / pl.col("total") * 100).round(1).alias("pct"))
        .sort("year")
    )
    mo_map = dict(zip(all_monthly["month"].to_list(),
                      all_monthly["NÚMERO DE VISITAS"].to_list()))
    result["__all__"] = {
        "t":  {"y": all_trend["year"].to_list(),
               "n": all_trend["Nacional"].to_list(),
               "e": all_trend["Extranjeros"].to_list()},
        "mo": {"v": [mo_map.get(m, 0) for m in range(1, 13)]},
        "vt": {"l": all_vtypes["label"].to_list(),
               "v": all_vtypes["NÚMERO DE VISITAS"].to_list()},
        "si": "",
        "pd": {"y": all_paid["year"].to_list(), "v": all_paid["pct"].to_list()},
    }

    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


# ── Build charts ──────────────────────────────────────────────────────────────

print("Rendering Plotly charts…")
chart_trend    = div(fig_annual_trend(d),        "chart-trend",       first=True)
chart_yoy      = div(fig_yoy_change(d),          "chart-yoy")
chart_site_don = div(fig_site_donut(d),          "chart-site-don")
chart_vtype_don= div(fig_visitor_type_donut(d),  "chart-vtype-don")
chart_map      = div(fig_states_map(d),          "chart-map")
chart_states   = div(fig_top_states(d),          "chart-states")
chart_fstate   = div(fig_foreign_ratio(d),       "chart-fstate")
chart_top      = div(fig_top_sites(d),           "chart-top")
chart_fst      = div(fig_foreign_by_site_type(d),"chart-fst")
chart_sbs      = div(fig_sites_by_state(d),      "chart-sbs")
chart_vtrend   = div(fig_visitor_trend(d),       "chart-vtrend")
chart_paid     = div(fig_paid_free_ratio(d),     "chart-paid")
chart_ftrend   = div(fig_foreign_trend(d),       "chart-ftrend")
chart_heat     = div(fig_seasonality_heatmap(d), "chart-heat")
chart_monthly  = div(fig_monthly_pattern(d),     "chart-monthly")

# ── Pre-compute JSON data for all centros ─────────────────────────────────────

print("Pre-computing per-centro data…")
centro_data_json = build_centro_data(df)

# ── Dropdown options grouped by state ─────────────────────────────────────────

centros_by_state = (
    df.select("ESTADO", "CENTRO DE TRABAJO")
    .unique()
    .sort("ESTADO", "CENTRO DE TRABAJO")
    .group_by("ESTADO", maintain_order=True)
    .agg(pl.col("CENTRO DE TRABAJO"))
)
optgroups = ""
for estado_row in centros_by_state.iter_rows(named=True):
    opts = "".join(
        f'<option value="{c.replace(chr(34), "&quot;")}">{c}</option>'
        for c in sorted(estado_row["CENTRO DE TRABAJO"])
    )
    optgroups += f'<optgroup label="{estado_row["ESTADO"]}">{opts}</optgroup>'

# ── Assemble HTML sections ────────────────────────────────────────────────────

kpis_html = "".join([
    kpi("Total de visitas",       f"{total/1e6:.0f}M",   f"{YEAR_MIN}–{YEAR_MAX}",          "kpi-total"),
    kpi("Sitios únicos",          f"{TOTAL_SITES:,}",    "museos y zonas arqueológicas",     "kpi-sites"),
    kpi("Visitantes extranjeros", f"{pct_foreign:.1f}%", "del total histórico",              "kpi-foreign"),
    kpi("Año pico",               str(peak_yr),          "mayor número de visitas",          "kpi-peak"),
])

tabs_nav = "".join([
    tab_btn("tendencias",     "📈 Tendencias",     active=True),
    tab_btn("entidades",      "🗾 Entidades"),
    tab_btn("geografia",      "🗺 Geografía"),
    tab_btn("sitios",         "🏛 Sitios"),
    tab_btn("visitantes",     "👥 Visitantes"),
    tab_btn("estacionalidad", "📅 Estacionalidad"),
])

tabs_content = "".join([
    tab_pane("tendencias", "".join([
        row((chart_trend,    12)),
        row((chart_yoy,     12)),
        row((chart_site_don, 6), (chart_vtype_don, 6)),
    ]), active=True),
    tab_pane("entidades",  row((chart_map,  12))),
    tab_pane("geografia",  "".join([
        row((chart_states, 12)),
        row((chart_fstate, 12)),
    ])),
    tab_pane("sitios",     "".join([
        row((chart_top, 12)),
        row((chart_fst, 12)),
        row((chart_sbs, 12)),
    ])),
    tab_pane("visitantes", "".join([
        row((chart_vtrend, 12)),
        row((chart_paid, 6), (chart_ftrend, 6)),
    ])),
    tab_pane("estacionalidad", "".join([
        row((chart_heat,    12)),
        row((chart_monthly, 12)),
    ])),
])

insights_html = "".join([
    insight_card("Teotihuacán lidera con 72M visitas",
        "Más del doble que el Museo Nacional de Antropología (58M). "
        "Las zonas arqueológicas concentran el 57% del total."),
    insight_card("Quintana Roo: 65% visitantes extranjeros",
        "Tulum y Chichén Itzá impulsan el turismo extranjero en el sureste. "
        "Las zonas arq. atraen 3× más extranjeros que los museos."),
    insight_card("COVID: caída del 73% en 2020",
        "Tras el mínimo de 7M en 2021, se recuperó a 21.4M en 2025, "
        "aún ~22% por debajo del pico de 2019 (27.4M)."),
    insight_card("Entrada dominical: 18.9% del total",
        "El acceso gratuito los domingos es la 2ª categoría más grande. "
        "Solo el 52% de las visitas son boleto pagado."),
    insight_card("Marzo es el mes pico (Semana Santa)",
        "Mayo y septiembre son los meses más bajos. "
        "El patrón estacional es estable y predecible."),
    insight_card("CDMX domina en volumen, Quintana Roo en turismo",
        "Ciudad de México concentra 26% de todas las visitas (150M), "
        "pero Quintana Roo tiene la mayor proporción de extranjeros (65.5%)."),
])

# ── JavaScript ────────────────────────────────────────────────────────────────

js = """
const MONTHS = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"];
const CFG    = {responsive: true, displayModeBar: false};

// ── Rebuild charts from scratch using CENTRO_DATA (avoids bdata issues) ────────
function updateCharts(centro) {
    const D = CENTRO_DATA[centro];
    if (!D) { console.error("No CENTRO_DATA entry for:", JSON.stringify(centro)); return; }

    const t      = D.t;
    const totals = t.y.map((_,i) => t.n[i] + t.e[i]);
    function layout(id) { return document.getElementById(id).layout; }

    // Annual trend — stacked area: Extranjeros + Nacional
    Plotly.newPlot("chart-trend", [
        { type:"scatter", mode:"lines", name:"Extranjeros", stackgroup:"1",
          x:t.y, y:t.e, line:{color:"#F4A261"}, fillcolor:"rgba(244,162,97,0.5)" },
        { type:"scatter", mode:"lines", name:"Nacional", stackgroup:"1",
          x:t.y, y:t.n, line:{color:"#3BB273"}, fillcolor:"rgba(59,178,115,0.5)" }
    ], layout("chart-trend"), CFG);

    // YoY change
    const yoyYears = t.y.slice(1);
    const yoyVals  = yoyYears.map((_,i) =>
        totals[i] > 0 ? ((totals[i+1] / totals[i]) - 1) * 100 : null);
    Plotly.newPlot("chart-yoy", [{
        type:"bar", x:yoyYears, y:yoyVals,
        marker:{color: yoyVals.map(v => v === null ? "#64748B" : v < 0 ? "#E84855" : "#3BB273")}
    }], layout("chart-yoy"), CFG);

    // Visitor-type donut
    Plotly.newPlot("chart-vtype-don", [{
        type:"pie", hole:0.5, labels:D.vt.l, values:D.vt.v,
        textinfo:"percent+label", textfont:{size:11}
    }], layout("chart-vtype-don"), CFG);

    // Monthly pattern
    Plotly.newPlot("chart-monthly", [{
        type:"bar", x:MONTHS, y:D.mo.v,
        marker:{color: MONTHS.map((_,i) => [0,2,11].includes(i) ? "#E84855" : "#2E86AB")}
    }], layout("chart-monthly"), CFG);

    // Foreign-% trend
    const fpct = t.y.map((_,i) => totals[i] > 0 ? (t.e[i] / totals[i] * 100) : 0);
    Plotly.newPlot("chart-ftrend", [{
        type:"scatter", mode:"lines+markers", x:t.y, y:fpct,
        line:{color:"#F4A261", width:2}, fill:"tozeroy", fillcolor:"rgba(244,162,97,0.15)"
    }], layout("chart-ftrend"), CFG);

    // Visitor-type trend (reuse nacional/foreign data for per-centro view)
    Plotly.newPlot("chart-vtrend", [
        { type:"scatter", mode:"lines", name:"Extranjeros", stackgroup:"1",
          x:t.y, y:t.e, line:{color:"#F4A261"}, fillcolor:"rgba(244,162,97,0.5)" },
        { type:"scatter", mode:"lines", name:"Nacional", stackgroup:"1",
          x:t.y, y:t.n, line:{color:"#3BB273"}, fillcolor:"rgba(59,178,115,0.5)" }
    ], layout("chart-vtrend"), CFG);

    // Paid-ticket % per year
    Plotly.newPlot("chart-paid", [{
        type:"scatter", mode:"lines+markers", x:D.pd.y, y:D.pd.v,
        line:{color:"#F4A261", width:2}, fill:"tozeroy", fillcolor:"rgba(244,162,97,0.15)",
        hovertemplate:"Año: %{x}<br>Pagaron: %{y:.1f}%<extra></extra>"
    }], layout("chart-paid"), CFG);

    // KPIs
    const sumN = t.n.reduce((a,b)=>a+b, 0);
    const sumE = t.e.reduce((a,b)=>a+b, 0);
    const sumAll = sumN + sumE;
    const pctForeign = sumAll > 0 ? (sumE / sumAll * 100) : 0;
    const peakIdx = totals.indexOf(Math.max(...totals));
    document.getElementById("kpi-total").textContent   = (sumAll/1e6).toFixed(1) + "M";
    document.getElementById("kpi-total-sub").textContent = t.y[0] + "–" + t.y[t.y.length-1];
    document.getElementById("kpi-sites").textContent   = "1 sitio";
    document.getElementById("kpi-sites-sub").textContent = D.si || "centro de trabajo";
    document.getElementById("kpi-foreign").textContent = pctForeign.toFixed(1) + "%";
    document.getElementById("kpi-foreign-sub").textContent = "del total del sitio";
    document.getElementById("kpi-peak").textContent    = String(t.y[peakIdx]);

    // Active-filter badge
    document.getElementById("filter-badge").textContent = centro;
    document.getElementById("filter-notice").style.display = "flex";
}

// ── Poll until Plotly has rendered chart-trend (SVG present), then update ──────
function applyWhenReady(centro, attempt) {
    attempt = attempt || 0;
    if (attempt > 100) { updateCharts(centro); return; }
    const el = document.getElementById("chart-trend");
    if (el && el.querySelector("svg.main-svg")) {
        updateCharts(centro);
    } else {
        setTimeout(() => applyWhenReady(centro, attempt + 1), 100);
    }
}

// ── Dropdown → reload page with ?centro=<value> ───────────────────────────────
document.getElementById("centro-select").addEventListener("change", function () {
    const url = new URL(window.location.href);
    if (this.value) {
        url.searchParams.set("centro", this.value);
    } else {
        url.searchParams.delete("centro");
    }
    window.location.href = url.toString();
});

// ── Clear button → remove param and reload (page reload resets everything) ───
document.getElementById("clear-filter").addEventListener("click", function () {
    window.location.href = window.location.pathname;
});


// ── On page load: read ?centro= and update charts once Plotly is ready ────────
window.addEventListener("load", function () {
    const centro = new URLSearchParams(window.location.search).get("centro");
    if (!centro) return;
    if (!CENTRO_DATA[centro]) { console.error("Not found in CENTRO_DATA:", JSON.stringify(centro)); return; }
    document.getElementById("centro-select").value = centro;
    applyWhenReady(centro);
});

// ── Resize charts when switching to a previously hidden tab ───────────────────
document.querySelectorAll("button[data-bs-toggle=tab]").forEach(btn => {
    btn.addEventListener("shown.bs.tab",
        () => window.dispatchEvent(new Event("resize")));
});
"""

# ── Full HTML ─────────────────────────────────────────────────────────────────

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>INAH · Museos y Zonas Arqueológicas</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
        rel="stylesheet" crossorigin="anonymous">
  <style>
    body          {{ background:#0F172A; color:#CBD5E1; padding:24px;
                    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}

    /* nav tabs */
    .nav-tabs       {{ border-bottom:1px solid #334155; }}
    .nav-link       {{ color:#94A3B8; background:transparent; border:none;
                      border-top:2px solid transparent; border-radius:0; padding:10px 16px; }}
    .nav-link:hover {{ color:#F8FAFC; background:#1E293B; }}
    .nav-link.active{{ color:#F8FAFC!important; background:#1E293B!important;
                      border-top:2px solid #2E86AB!important; font-weight:600; }}
    .tab-content    {{ background:#0F172A; padding-top:16px; }}

    /* kpi */
    .kpi-card  {{ background:#1E293B; border:1px solid #334155; border-radius:8px;
                  padding:16px; text-align:center; height:100%; }}
    .kpi-label {{ color:#94A3B8; font-size:.8rem; margin-bottom:4px; }}
    .kpi-value {{ color:#F8FAFC; font-weight:700; margin:0; }}
    .kpi-sub   {{ color:#64748B; }}

    /* filter box */
    .filter-box     {{ background:#1E293B; border:1px solid #334155; border-radius:8px;
                       padding:14px 18px; }}
    .filter-label   {{ color:#94A3B8; font-size:.82rem; margin-bottom:6px; }}
    .dark-select    {{ background-color:#1E293B!important; border-color:#334155!important;
                       color:#CBD5E1!important; }}
    .dark-select:focus {{ border-color:#2E86AB!important;
                          box-shadow:0 0 0 .2rem rgba(46,134,171,.25)!important; }}
    .dark-select option {{ background:#1E293B; color:#CBD5E1; }}
    .dark-select optgroup {{ color:#64748B; }}

    /* active-filter notice */
    #filter-notice  {{ display:none; align-items:center; gap:10px;
                       background:rgba(46,134,171,.12); border:1px solid #2E86AB;
                       border-radius:6px; padding:8px 14px; font-size:.83rem; color:#94A3B8; }}
    #filter-badge   {{ background:#2E86AB; color:#fff; border-radius:4px;
                       padding:2px 8px; font-size:.78rem; max-width:520px;
                       overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    #clear-filter   {{ cursor:pointer; color:#94A3B8; font-size:.75rem;
                       border:1px solid #334155; border-radius:4px; padding:2px 8px;
                       background:transparent; }}
    #clear-filter:hover {{ color:#F8FAFC; border-color:#64748B; }}

    /* insight */
    .insight-card  {{ background:#1E293B; border:1px solid #334155; border-radius:8px;
                      padding:16px; height:100%; }}
    .insight-title {{ color:#F8FAFC; display:block; margin-bottom:4px; }}
    .insight-body  {{ color:#94A3B8; font-size:.83rem; margin:0; }}

    h6.section-title {{ color:#94A3B8; font-weight:600; margin-bottom:12px; }}
  </style>
</head>
<body>
<div class="container-fluid">

  <!-- Header -->
  <div class="mb-4">
    <h2 style="color:#F8FAFC;font-weight:700;margin-bottom:4px;">
      Visitas a Museos y Zonas Arqueológicas · INAH
    </h2>
    <p style="color:#64748B;font-size:.9rem;">
      México · 1996–2025 · 418 mil registros · SIINAH
    </p>
  </div>

  <!-- KPIs -->
  <div class="row g-3 mb-3">{kpis_html}</div>

  <!-- Filter -->
  <div class="filter-box mb-3">
    <div class="row g-3 align-items-end">
      <div class="col-10 col-md-8">
        <p class="filter-label mb-1">Centro de trabajo
          <small style="color:#64748B"> — los gráficos de tendencias se actualizan con la selección</small>
        </p>
        <select id="centro-select" class="form-select dark-select">
          <option value="">Todos los centros de trabajo</option>
          {optgroups}
        </select>
      </div>
      <div class="col-2 col-md-4 d-flex align-items-end">
        <div id="filter-notice">
          <span id="filter-badge"></span>
          <button id="clear-filter">✕ limpiar</button>
        </div>
      </div>
    </div>
  </div>

  <!-- Tabs -->
  <ul class="nav nav-tabs" role="tablist">{tabs_nav}</ul>
  <div class="tab-content">{tabs_content}</div>

  <!-- Insights -->
  <div class="mt-4">
    <h6 class="section-title">Hallazgos clave</h6>
    <div class="row g-3">{insights_html}</div>
  </div>

</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"
        crossorigin="anonymous"></script>
<script>
const CENTRO_DATA = {centro_data_json};
{js}
</script>
</body>
</html>"""

out = "index.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(html)

size_kb = len(html.encode()) / 1024
print(f"\nWritten {out}  ({size_kb:.0f} KB)")
