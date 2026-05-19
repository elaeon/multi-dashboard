import json
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ── Constants ─────────────────────────────────────────────────────────────────

MONTHS = ['Enero','Febrero','Marzo','Abril','Mayo','Junio',
          'Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre']

STATE_ISO = {
    'Aguascalientes': 'MX-AGU', 'Baja California': 'MX-BCN',
    'Baja California Sur': 'MX-BCS', 'Campeche': 'MX-CAM',
    'Coahuila de Zaragoza': 'MX-COA', 'Colima': 'MX-COL',
    'Chiapas': 'MX-CHP', 'Chihuahua': 'MX-CHH',
    'Ciudad de México': 'MX-CMX', 'Durango': 'MX-DUR',
    'Guanajuato': 'MX-GUA', 'Guerrero': 'MX-GRO',
    'Hidalgo': 'MX-HID', 'Jalisco': 'MX-JAL',
    'México': 'MX-MEX', 'Michoacán de Ocampo': 'MX-MIC',
    'Morelos': 'MX-MOR', 'Nayarit': 'MX-NAY',
    'Nuevo León': 'MX-NLE', 'Oaxaca': 'MX-OAX',
    'Puebla': 'MX-PUE', 'Querétaro': 'MX-QUE',
    'Quintana Roo': 'MX-ROO', 'San Luis Potosí': 'MX-SLP',
    'Sinaloa': 'MX-SIN', 'Sonora': 'MX-SON',
    'Tabasco': 'MX-TAB', 'Tamaulipas': 'MX-TAM',
    'Tlaxcala': 'MX-TLA', 'Veracruz de Ignacio de la Llave': 'MX-VER',
    'Yucatán': 'MX-YUC', 'Zacatecas': 'MX-ZAC',
}

DELITOS_CLAVE = [
    'Homicidio doloso', 'Feminicidio', 'Secuestro', 'Extorsión',
    'Lesiones dolosas', 'Abuso sexual', 'Violencia familiar', 'Narcomenudeo',
    'Robo a negocio', 'Robo a casa habitación',
    'Robo de vehículo automotor', 'Robo en transporte público colectivo',
]

# Maps each row to one of the 12 headline crime labels (null if not in list)
_DELITO_CLAVE_EXPR = (
    pl.when(pl.col('Subtipo de delito') == 'Homicidio doloso').then(pl.lit('Homicidio doloso'))
    .when(pl.col('Subtipo de delito') == 'Feminicidio').then(pl.lit('Feminicidio'))
    .when(pl.col('Tipo de delito') == 'Secuestro').then(pl.lit('Secuestro'))
    .when(pl.col('Tipo de delito') == 'Extorsión').then(pl.lit('Extorsión'))
    .when(pl.col('Subtipo de delito') == 'Lesiones dolosas').then(pl.lit('Lesiones dolosas'))
    .when(pl.col('Subtipo de delito') == 'Abuso sexual').then(pl.lit('Abuso sexual'))
    .when(pl.col('Subtipo de delito') == 'Violencia familiar').then(pl.lit('Violencia familiar'))
    .when(pl.col('Tipo de delito') == 'Narcomenudeo').then(pl.lit('Narcomenudeo'))
    .when(pl.col('Subtipo de delito') == 'Robo a negocio').then(pl.lit('Robo a negocio'))
    .when(pl.col('Subtipo de delito') == 'Robo a casa habitación').then(pl.lit('Robo a casa habitación'))
    .when(pl.col('Subtipo de delito').str.starts_with('Robo de vehículo automotor')).then(pl.lit('Robo de vehículo automotor'))
    .when(pl.col('Subtipo de delito') == 'Robo en transporte público colectivo').then(pl.lit('Robo en transporte público colectivo'))
    .otherwise(pl.lit(None))
    .alias('Delito_Clave')
)

MES_NUM = {m: i + 1 for i, m in enumerate(
    ['Enero','Febrero','Marzo','Abril','Mayo','Junio',
     'Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre']
)}

BIEN_COLORS = {
    'El patrimonio': ('46,134,171', '#2E86AB'),
    'La vida y la Integridad corporal': ('232,72,85', '#E84855'),
    'La familia': ('244,162,97', '#F4A261'),
    'Otros bienes jurídicos afectados (del fuero común)': ('100,116,139', '#64748B'),
    'La libertad y la seguridad sexual': ('168,85,247', '#A855F7'),
    'Libertad personal': ('59,178,115', '#3BB273'),
    'La sociedad': ('245,158,11', '#F59E0B'),
}

# ── Data loading ──────────────────────────────────────────────────────────────

# lf_raw is reused by both the yearly and monthly aggregation paths
lf_raw = pl.scan_parquet("data/incidencia_delictiva_fuero_comun.parquet")

# sum_horizontal stays at 2.8M rows instead of unpivot's 33M — far less memory
lf = (
    lf_raw
    .with_columns(
        pl.sum_horizontal(*[pl.col(m).fill_null(0) for m in MONTHS]).alias('Casos')
    )
    .select(['Año', 'Entidad', 'Clave_Ent', 'Bien jurídico afectado', 'Tipo de delito', 'Casos'])
)

q_yr_bien    = lf.group_by(['Año', 'Bien jurídico afectado']).agg(pl.col('Casos').sum()).sort('Año')
q_yr_tipo    = lf.group_by(['Año', 'Tipo de delito', 'Bien jurídico afectado']).agg(pl.col('Casos').sum())
q_yr_ent_tipo = lf.group_by(['Año', 'Entidad', 'Clave_Ent', 'Tipo de delito']).agg(pl.col('Casos').sum())

# Monthly aggregation for 12 headline crimes by state — ~55k rows after unpivot, still small
q_monthly = (
    lf_raw
    .with_columns(_DELITO_CLAVE_EXPR)
    .filter(pl.col('Delito_Clave').is_not_null())
    .group_by(['Año', 'Entidad', 'Delito_Clave'])
    .agg([pl.col(m).sum().alias(m) for m in MONTHS])
)

agg_yr_bien, agg_yr_tipo, agg_yr_ent_tipo, agg_monthly = pl.collect_all(
    [q_yr_bien, q_yr_tipo, q_yr_ent_tipo, q_monthly]
)
agg_yr_ent = agg_yr_ent_tipo.group_by(['Año', 'Entidad', 'Clave_Ent']).agg(pl.col('Casos').sum())

# Unpivot the collected frame — safe since it's already aggregated
# cast MesNum to Int32 so sort is numeric, not lexicographic
agg_monthly_long = (
    agg_monthly
    .unpivot(on=MONTHS, index=['Año', 'Entidad', 'Delito_Clave'], variable_name='Mes', value_name='Casos')
    .with_columns(pl.col('Mes').replace(MES_NUM).cast(pl.Int32).alias('MesNum'))
    .sort(['Delito_Clave', 'Entidad', 'Año', 'MesNum'])
)

TIPOS = ['Todos los delitos'] + sorted(agg_yr_tipo['Tipo de delito'].unique().to_list())
AÑOS = sorted(agg_yr_tipo['Año'].cast(pl.Int32).unique().to_list())
MIN_AÑO, MAX_AÑO = AÑOS[0], AÑOS[-1]
ESTADOS = ['Nacional'] + sorted(agg_monthly_long['Entidad'].unique().to_list())

with open("data/mexico_states.geojson") as _f:
    GEO = json.load(_f)

# ── Theme ─────────────────────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#CBD5E1",
    xaxis=dict(gridcolor="#334155"),
    yaxis=dict(gridcolor="#334155"),
    margin=dict(t=40, b=40, l=10, r=10),
)
CARD_STYLE = {
    "background": "#1E293B", "border": "1px solid #334155",
    "borderRadius": "8px", "padding": "16px", "textAlign": "center",
}
TAB_STYLE = {"backgroundColor": "#0F172A", "color": "#94A3B8", "borderTop": "none",
             "borderBottom": "1px solid #334155", "padding": "10px 18px"}
TAB_SEL   = {"backgroundColor": "#1E293B", "color": "#F8FAFC",
             "borderTop": "2px solid #2E86AB", "borderBottom": "none",
             "fontWeight": "600", "padding": "10px 18px"}


def kpi_card(title: str, value: str, color: str = "#CBD5E1") -> dbc.Col:
    return dbc.Col(
        html.Div([
            html.P(title, style={"color": "#94A3B8", "fontSize": "12px", "margin": 0}),
            html.H3(value, style={"color": color, "margin": "4px 0 0", "fontSize": "22px"}),
        ], style=CARD_STYLE),
        xs=12, sm=6, md=3,
    )


# ── Figure factories ──────────────────────────────────────────────────────────

def fig_trend(d_yr: pl.DataFrame, tipo: str) -> go.Figure:
    fig = go.Figure()
    if tipo == 'Todos los delitos':
        bienes = d_yr.group_by('Bien jurídico afectado').agg(pl.col('Casos').sum()) \
                     .sort('Casos', descending=True)['Bien jurídico afectado'].to_list()
        for bien in bienes:
            sub = d_yr.filter(pl.col('Bien jurídico afectado') == bien).sort('Año')
            rgb, hex_col = BIEN_COLORS.get(bien, ('148,163,184', '#94A3B8'))
            fig.add_trace(go.Scatter(
                x=sub['Año'].to_list(), y=sub['Casos'].to_list(),
                name=bien, mode='lines', stackgroup='one',
                line=dict(width=0.5, color=hex_col),
                fillcolor=f'rgba({rgb},0.8)',
                hovertemplate="<b>%{x}</b><br>%{fullData.name}<br>Casos: %{y:,}<extra></extra>",
            ))
        title = "Delitos del fuero común por bien jurídico afectado"
    else:
        sub = d_yr.sort('Año')
        fig.add_trace(go.Scatter(
            x=sub['Año'].to_list(), y=sub['Casos'].to_list(),
            mode='lines+markers', fill='tozeroy',
            line=dict(color='#2E86AB', width=2),
            fillcolor='rgba(46,134,171,0.15)',
            hovertemplate="<b>%{x}</b><br>Casos: %{y:,}<extra></extra>",
        ))
        title = f"Tendencia: {tipo}"
    fig.update_layout(
        title=title, height=380,
        showlegend=(tipo == 'Todos los delitos'),
        legend=dict(orientation="h", y=-0.25, x=0, font=dict(size=10)),
        margin=dict(t=40, b=90 if tipo == 'Todos los delitos' else 40, l=10, r=10),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ('margin',)},
    )
    fig.update_layout(
        xaxis=dict(gridcolor="#334155", title="Año", dtick=1),
        yaxis=dict(gridcolor="#334155", title="Casos registrados"),
    )
    return fig


def fig_map(d_ent: pl.DataFrame) -> go.Figure:
    d = d_ent.with_columns(
        pl.col('Entidad').replace(STATE_ISO).alias('iso')
    )
    fig = px.choropleth_map(
        d,
        geojson=GEO,
        locations='iso',
        featureidkey='properties.id',
        color='Casos',
        color_continuous_scale='YlOrRd',
        hover_name='Entidad',
        map_style='carto-darkmatter',
        center={'lat': 23.5, 'lon': -102.5},
        zoom=3.8,
    )
    fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>Casos: %{z:,}<extra></extra>")
    fig.update_layout(
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
        margin=dict(t=0, b=0, l=0, r=0),
        coloraxis_colorbar=dict(
            title=dict(text="Casos", font=dict(color="#CBD5E1")),
            tickfont=dict(color="#CBD5E1"),
        ),
    )
    return fig


def fig_ranking(d_ent: pl.DataFrame) -> go.Figure:
    top = d_ent.sort('Casos', descending=True).head(15)
    n = len(top)
    fig = go.Figure(go.Bar(
        x=top['Casos'].to_list(),
        y=top['Entidad'].to_list(),
        orientation='h',
        marker_color='#2E86AB',
        hovertemplate="<b>%{y}</b><br>Casos: %{x:,}<extra></extra>",
    ))
    fig.update_layout({
        **CHART_LAYOUT,
        'title': "Top 15 estados",
        'height': max(300, n * 28 + 80),
        'yaxis': dict(autorange='reversed', gridcolor="#334155"),
        'xaxis': dict(gridcolor="#334155", title="Casos"),
    })
    return fig


def fig_tipos(d_tipo: pl.DataFrame, tipo: str) -> go.Figure:
    if tipo == 'Todos los delitos':
        top = d_tipo.sort('Casos', descending=True).head(15)
        label_col = 'Tipo de delito'
        title = "Top 15 tipos de delito"
    else:
        top = d_tipo.sort('Casos', descending=True).head(15)
        label_col = 'Entidad'
        title = f"Top 15 estados · {tipo}"
    n = len(top)
    fig = go.Figure(go.Bar(
        x=top['Casos'].to_list(),
        y=top[label_col].to_list(),
        orientation='h',
        marker_color='#F4A261',
        hovertemplate="<b>%{y}</b><br>Casos: %{x:,}<extra></extra>",
    ))
    fig.update_layout({
        **CHART_LAYOUT,
        'title': title,
        'height': max(300, n * 28 + 80),
        'yaxis': dict(autorange='reversed', gridcolor="#334155"),
        'xaxis': dict(gridcolor="#334155", title="Casos"),
    })
    return fig


def _fmt(n: float) -> str:
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000: return f"{n/1_000:.1f}k"
    return f"{n:,.0f}"


def fig_monthly_grid(d: pl.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=DELITOS_CLAVE,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    for i, delito in enumerate(DELITOS_CLAVE):
        r, c = divmod(i, 3)
        sub = d.filter(pl.col('Delito_Clave') == delito).sort(['Año', 'MesNum'])
        if sub.is_empty():
            continue

        pts = sub.iter_rows(named=True)
        data_pts = list(pts)

        x = [f"{pt['Mes'][:3]} '{str(pt['Año'])[2:]}" for pt in data_pts]
        y = [pt['Casos'] for pt in data_pts]

        # Lookup table for the reference line (same month, prior year)
        casos_map = {(pt['Año'], pt['MesNum']): pt['Casos'] for pt in data_pts}
        y_ref = [casos_map.get((pt['Año'] - 1, pt['MesNum'])) for pt in data_pts]
        has_ref = any(v is not None for v in y_ref)

        # Main trend line — yellow/orange, smooth spline
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='lines',
            line=dict(color='#F59E0B', width=2.5, shape='spline'),
            showlegend=False,
            hovertemplate="%{x}: %{y:,}<extra></extra>",
        ), row=r + 1, col=c + 1)

        # Reference line — same month from prior year, cyan, thin
        if has_ref:
            fig.add_trace(go.Scatter(
                x=x, y=y_ref, mode='lines',
                line=dict(color='#22D3EE', width=1.2, shape='spline'),
                connectgaps=False, showlegend=False,
                hovertemplate="%{x} (año ant.): %{y:,}<extra></extra>",
            ), row=r + 1, col=c + 1)

        # Endpoint marker + compact last-value label
        fig.add_trace(go.Scatter(
            x=[x[-1]], y=[y[-1]], mode='markers+text',
            marker=dict(color='#F59E0B', size=6),
            text=[_fmt(y[-1])], textposition='top left',
            textfont=dict(size=8, color='#F59E0B'),
            showlegend=False, hoverinfo='skip',
        ), row=r + 1, col=c + 1)

        # YoY delta annotation (colored box, bottom-left)
        last_pt = data_pts[-1]
        prior_val = casos_map.get((last_pt['Año'] - 1, last_pt['MesNum']))
        if prior_val and prior_val > 0:
            delta = y[-1] - prior_val
            pct = delta / prior_val * 100
            ann_text = f"{delta:+,.0f}<br>{pct:+.2f}%"
            ann_color = '#22D3EE' if delta < 0 else '#F59E0B'
        else:
            ann_text, ann_color = '', '#94A3B8'

        axis_idx = '' if i == 0 else str(i + 1)
        if ann_text:
            fig.add_annotation(
                text=ann_text,
                xref=f'x{axis_idx} domain', yref=f'y{axis_idx} domain',
                x=0.04, y=0.22, showarrow=False,
                font=dict(size=8, color=ann_color),
                bgcolor='#0F2744', bordercolor=ann_color, borderwidth=1, borderpad=4,
            )

        # X-axis: only label January of each year to avoid crowding
        ticktext = [
            f"Ene '{str(pt['Año'])[2:]}" if pt['MesNum'] == 1 else ''
            for pt in data_pts
        ]
        fig.update_xaxes(
            tickmode='array', tickvals=x, ticktext=ticktext,
            tickfont=dict(size=8, color='#94A3B8'),
            showgrid=False, zeroline=False, tickangle=0,
            row=r + 1, col=c + 1,
        )
        fig.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor='#1E3A5F',
            zeroline=False, tickformat='.2s',
            tickfont=dict(size=8, color='#94A3B8'),
            row=r + 1, col=c + 1,
        )

    fig.update_layout(
        height=950,
        paper_bgcolor='#0F172A', plot_bgcolor='#0A1628',
        font_color='#CBD5E1',
        margin=dict(t=50, b=30, l=50, r=20),
    )
    for ann in fig.layout.annotations:
        if ann.text in DELITOS_CLAVE:
            ann.font = dict(size=10, color='#CBD5E1')
    return fig


def compute_kpis(d_yr: pl.DataFrame, d_ent: pl.DataFrame, d_tipo_totals: pl.DataFrame,
                 tipo: str, yr_range: tuple) -> tuple:
    total = d_yr['Casos'].sum()

    # YoY: compare last complete year vs prior year
    last_yr = min(yr_range[1], 2025)
    prior_yr = last_yr - 1
    t_last = d_yr.filter(pl.col('Año') == last_yr)['Casos'].sum()
    t_prior = d_yr.filter(pl.col('Año') == prior_yr)['Casos'].sum()
    if t_prior > 0:
        pct = (t_last - t_prior) / t_prior * 100
        yoy = f"{pct:+.1f}% vs {prior_yr}"
        yoy_color = "#E84855" if pct > 0 else "#3BB273"
    else:
        yoy, yoy_color = "—", "#94A3B8"

    top_ent = d_ent.sort('Casos', descending=True).head(1)['Entidad'][0] if len(d_ent) > 0 else "—"

    if tipo == 'Todos los delitos':
        top_tipo = d_tipo_totals.sort('Casos', descending=True).head(1)['Tipo de delito'][0] \
            if len(d_tipo_totals) > 0 else "—"
    else:
        top_tipo = tipo

    return f"{total:,.0f}", yoy, yoy_color, top_ent, top_tipo


# ── Layout ────────────────────────────────────────────────────────────────────

app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.title = "Incidencia Delictiva · Fuero Común"

app.layout = html.Div(style={"backgroundColor": "#0F172A", "minHeight": "100vh",
                              "padding": "24px", "fontFamily": "sans-serif"}, children=[
    html.H2("Incidencia Delictiva del Fuero Común",
            style={"color": "#F8FAFC", "marginBottom": "4px"}),
    html.P("Carpetas de investigación por municipio · 2015–2026 · SESNSP",
           style={"color": "#64748B", "marginBottom": "20px"}),

    # Shared year slider
    dbc.Row([
        dbc.Col([
            html.Label("Rango de años", style={"color": "#94A3B8", "fontSize": "12px"}),
            dcc.RangeSlider(
                id="sl-años",
                min=MIN_AÑO, max=MAX_AÑO,
                value=[MIN_AÑO, 2025],
                marks={y: {"label": str(y), "style": {"color": "#94A3B8", "fontSize": "11px"}}
                       for y in AÑOS},
                step=1,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ], md=12),
    ], className="mb-3"),

    dcc.Tabs(id="tabs", value="tab-resumen", children=[

        # ── Tab 1: Resumen general ─────────────────────────────────────────────
        dcc.Tab(label="Resumen general", value="tab-resumen",
                style=TAB_STYLE, selected_style=TAB_SEL, children=[

            dbc.Row([
                dbc.Col([
                    html.Label("Tipo de delito", style={"color": "#94A3B8", "fontSize": "12px"}),
                    dcc.Dropdown(
                        id="dd-tipo",
                        options=[{"label": t, "value": t} for t in TIPOS],
                        value="Todos los delitos",
                        clearable=False,
                        style={"backgroundColor": "#1E293B", "color": "#0F172A"},
                    ),
                ], md=5),
            ], className="mb-3 mt-3"),

            dbc.Row(id="kpi-row", className="mb-3 g-2"),

            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-trend"), md=12),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(id="chart-map"),
                                 style={**CARD_STYLE, "padding": "8px"}), md=7),
                dbc.Col(html.Div(dcc.Graph(id="chart-ranking"),
                                 style={**CARD_STYLE, "padding": "8px"}), md=5),
            ], className="mb-3 g-2"),

            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(id="chart-tipos"),
                                 style={**CARD_STYLE, "padding": "8px"}), md=12),
            ], className="mb-3"),
        ]),

        # ── Tab 2: Alto impacto mensual ────────────────────────────────────────
        dcc.Tab(label="Alto impacto · mensual", value="tab-mensual",
                style=TAB_STYLE, selected_style=TAB_SEL, children=[

            dbc.Row([
                dbc.Col([
                    html.Label("Estado", style={"color": "#94A3B8", "fontSize": "12px"}),
                    dcc.Dropdown(
                        id="dd-estado",
                        options=[{"label": e, "value": e} for e in ESTADOS],
                        value="Nacional",
                        clearable=False,
                        style={"backgroundColor": "#1E293B", "color": "#0F172A"},
                    ),
                ], md=4),
            ], className="mb-3 mt-3"),

            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(id="chart-monthly-grid"),
                                 style={**CARD_STYLE, "padding": "12px"}), md=12),
            ], className="mb-3"),
        ]),

    ]),

    html.P("Fuente: Secretariado Ejecutivo del Sistema Nacional de Seguridad Pública (SESNSP)",
           style={"color": "#475569", "fontSize": "11px", "textAlign": "center", "marginTop": "16px"}),
])


# ── Callback ──────────────────────────────────────────────────────────────────

@app.callback(
    Output("kpi-row", "children"),
    Output("chart-trend", "figure"),
    Output("chart-map", "figure"),
    Output("chart-ranking", "figure"),
    Output("chart-tipos", "figure"),
    Input("dd-tipo", "value"),
    Input("sl-años", "value"),
)
def update_all(tipo: str, yr_range: list):
    yr0, yr1 = yr_range
    yr_mask = pl.col('Año').is_between(yr0, yr1)

    if tipo == 'Todos los delitos':
        d_yr = agg_yr_bien.filter(yr_mask)
        d_ent = agg_yr_ent.filter(yr_mask).group_by(['Entidad', 'Clave_Ent']).agg(pl.col('Casos').sum())
        d_tipo_totals = agg_yr_tipo.filter(yr_mask).group_by('Tipo de delito').agg(pl.col('Casos').sum())
        d_yr_kpi = agg_yr_bien.filter(yr_mask).group_by('Año').agg(pl.col('Casos').sum())
        d_breakdown = d_tipo_totals
    else:
        tipo_mask = pl.col('Tipo de delito') == tipo
        d_yr = agg_yr_tipo.filter(yr_mask & tipo_mask).group_by('Año').agg(pl.col('Casos').sum()).sort('Año')
        d_ent = agg_yr_ent_tipo.filter(yr_mask & tipo_mask).group_by(['Entidad', 'Clave_Ent']).agg(pl.col('Casos').sum())
        d_tipo_totals = agg_yr_tipo.filter(yr_mask & tipo_mask).group_by('Tipo de delito').agg(pl.col('Casos').sum())
        d_yr_kpi = d_yr
        d_breakdown = d_ent

    total, yoy, yoy_color, top_ent, top_tipo = compute_kpis(
        d_yr_kpi, d_ent, d_tipo_totals, tipo, (yr0, yr1)
    )

    kpis = dbc.Row([
        kpi_card("Total de casos registrados", total, "#CBD5E1"),
        kpi_card(f"Variación ({min(yr1,2025)} vs {min(yr1,2025)-1})", yoy, yoy_color),
        kpi_card("Estado con más casos", top_ent, "#F4A261"),
        kpi_card("Delito más frecuente" if tipo == 'Todos los delitos' else "Tipo seleccionado",
                 top_tipo if len(top_tipo) <= 30 else top_tipo[:28] + "…", "#2E86AB"),
    ], className="g-2").children

    return (
        kpis,
        fig_trend(d_yr, tipo),
        fig_map(d_ent),
        fig_ranking(d_ent),
        fig_tipos(d_breakdown, tipo),
    )


@app.callback(
    Output("chart-monthly-grid", "figure"),
    Input("sl-años", "value"),
    Input("dd-estado", "value"),
)
def update_monthly(yr_range: list, estado: str):
    yr0, yr1 = yr_range
    d = agg_monthly_long.filter(pl.col('Año').is_between(yr0, yr1))
    if estado and estado != 'Nacional':
        d = d.filter(pl.col('Entidad') == estado)
    else:
        d = d.group_by(['Año', 'Delito_Clave', 'Mes', 'MesNum']).agg(pl.col('Casos').sum())
    return fig_monthly_grid(d)


if __name__ == "__main__":
    app.run(debug=True, port=8060)
