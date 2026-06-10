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

MES_NUM = {m: i + 1 for i, m in enumerate(MONTHS)}

BIEN_COLORS = {
    'El patrimonio': ('46,134,171', '#2E86AB'),
    'La vida y la Integridad corporal': ('232,72,85', '#E84855'),
    'La familia': ('244,162,97', '#F4A261'),
    'Otros bienes jurídicos afectados (del fuero común)': ('100,116,139', '#64748B'),
    'La libertad y la seguridad sexual': ('168,85,247', '#A855F7'),
    'Libertad personal': ('59,178,115', '#3BB273'),
    'La sociedad': ('245,158,11', '#F59E0B'),
}

FOCUS, CONTEXT = "#2E86AB", "#475569"

# ── Data loading ──────────────────────────────────────────────────────────────

lf_raw = pl.scan_parquet("data/incidencia_delictiva_fuero_comun.parquet")

lf = (
    lf_raw
    .with_columns(pl.sum_horizontal(*[pl.col(m).fill_null(0) for m in MONTHS]).alias('Casos'))
    .select(['Año', 'Entidad', 'Clave_Ent', 'Bien jurídico afectado', 'Tipo de delito', 'Casos'])
)

q_yr_bien     = lf.group_by(['Año', 'Bien jurídico afectado']).agg(pl.col('Casos').sum()).sort('Año')
q_yr_tipo     = lf.group_by(['Año', 'Tipo de delito', 'Bien jurídico afectado']).agg(pl.col('Casos').sum())
q_yr_ent_tipo = lf.group_by(['Año', 'Entidad', 'Clave_Ent', 'Tipo de delito']).agg(pl.col('Casos').sum())
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

agg_monthly_long = (
    agg_monthly
    .unpivot(on=MONTHS, index=['Año', 'Entidad', 'Delito_Clave'], variable_name='Mes', value_name='Casos')
    .with_columns(pl.col('Mes').replace(MES_NUM).cast(pl.Int32).alias('MesNum'))
    .sort(['Delito_Clave', 'Entidad', 'Año', 'MesNum'])
)

# Yearly totals for Tab 3 — derived from already-collected agg_monthly, no extra scan
agg_yr_claims = (
    agg_monthly
    .filter(pl.col('Delito_Clave').is_in(['Homicidio doloso', 'Feminicidio', 'Violencia familiar']))
    .with_columns(pl.sum_horizontal(*[pl.col(m).fill_null(0) for m in MONTHS]).alias('Casos'))
    .group_by(['Año', 'Delito_Clave'])
    .agg(pl.col('Casos').sum())
    .sort('Año')
    .filter(pl.col('Año') <= 2025)  # 2026 is Jan–Apr only
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


def kpi_card(title: str, value: str, color: str = "#CBD5E1",
             delta: str = None, delta_color: str = None) -> dbc.Col:
    children = [
        html.P(title, style={"color": "#94A3B8", "fontSize": "12px", "margin": 0}),
        html.H3(value, style={"color": color, "margin": "4px 0 0", "fontSize": "22px"}),
    ]
    if delta:
        children.append(html.Span(delta, style={"color": delta_color or "#94A3B8", "fontSize": "11px"}))
    return dbc.Col(html.Div(children, style=CARD_STYLE), xs=12, sm=6, md=3)


# ── Figure factories ──────────────────────────────────────────────────────────

def fig_trend(d_yr: pl.DataFrame, tipo: str) -> go.Figure:
    fig = go.Figure()
    if tipo == 'Todos los delitos':
        bienes = (d_yr.group_by('Bien jurídico afectado').agg(pl.col('Casos').sum())
                  .sort('Casos', descending=True)['Bien jurídico afectado'].to_list())
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
        totals = d_yr.group_by('Año').agg(pl.col('Casos').sum()).sort('Casos', descending=True)
        peak_yr = int(totals['Año'][0])
        peak_val = int(totals['Casos'][0])
        title = (
            f"<b>Los delitos alcanzaron su máximo en {peak_yr} con {peak_val/1e6:.2f}M casos</b>"
            f"<br><sup style='color:#94A3B8'>Fuero común · casos por bien jurídico afectado</sup>"
        )
        show_legend = True
        b_margin = 90
    else:
        sub = d_yr.sort('Año')
        fig.add_trace(go.Scatter(
            x=sub['Año'].to_list(), y=sub['Casos'].to_list(),
            mode='lines+markers', fill='tozeroy',
            line=dict(color='#2E86AB', width=2),
            fillcolor='rgba(46,134,171,0.15)',
            hovertemplate="<b>%{x}</b><br>Casos: %{y:,}<extra></extra>",
        ))
        pk = sub.sort('Casos', descending=True)
        peak_yr = int(pk['Año'][0])
        peak_val = int(pk['Casos'][0])
        title = (
            f"<b>Tendencia: {tipo}</b>"
            f"<br><sup style='color:#94A3B8'>Pico: {peak_val:,} casos en {peak_yr}</sup>"
        )
        show_legend = False
        b_margin = 40

    years_in_data = d_yr['Año'].unique().to_list()
    if 2020 in years_in_data:
        fig.add_vrect(x0=2019.5, x1=2021.5, fillcolor="rgba(244,162,97,0.08)", line_width=0,
                      annotation_text="pandemia", annotation_font_color="#94A3B8",
                      annotation_position="top left")
    if peak_yr != max(years_in_data) and peak_yr in years_in_data:
        fig.add_vline(x=peak_yr, line_dash="dot", line_color="#64748B",
                      annotation_text=f"pico {peak_yr}",
                      annotation_font_color="#94A3B8", annotation_position="top right")
    if 2026 in years_in_data:
        fig.add_vrect(x0=2025.5, x1=2026.5, fillcolor="rgba(148,163,184,0.05)", line_width=0,
                      annotation_text="parcial", annotation_font_color="#64748B",
                      annotation_position="top right")

    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=380, showlegend=show_legend,
        legend=dict(orientation="h", y=-0.25, x=0, font=dict(size=10)),
        margin=dict(t=55, b=b_margin, l=10, r=10),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ('margin',)},
    )
    fig.update_layout(
        xaxis=dict(gridcolor="#334155", title="Año", dtick=1),
        yaxis=dict(gridcolor="#334155", title="Casos registrados"),
    )
    return fig


def fig_map(d_ent: pl.DataFrame) -> go.Figure:
    d = d_ent.with_columns(pl.col('Entidad').replace(STATE_ISO).alias('iso'))
    fig = px.choropleth_map(
        d, geojson=GEO, locations='iso', featureidkey='properties.id',
        color='Casos', color_continuous_scale='YlOrRd',
        hover_name='Entidad', map_style='carto-darkmatter',
        center={'lat': 23.5, 'lon': -102.5}, zoom=3.8,
    )
    fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>Casos: %{z:,}<extra></extra>")
    fig.update_layout(
        height=500, paper_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1",
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
    total_all = d_ent['Casos'].sum()
    top5_pct = top.head(5)['Casos'].sum() / total_all * 100 if total_all > 0 else 0
    colors_bar = [FOCUS if i < 5 else CONTEXT for i in range(n)]
    fig = go.Figure(go.Bar(
        x=top['Casos'].to_list(), y=top['Entidad'].to_list(),
        orientation='h', marker_color=colors_bar,
        hovertemplate="<b>%{y}</b><br>Casos: %{x:,}<extra></extra>",
    ))
    fig.update_layout({
        **CHART_LAYOUT,
        'title': dict(
            text=(f"<b>5 estados concentran el {top5_pct:.0f}% de todos los delitos</b>"
                  f"<br><sup style='color:#94A3B8'>Top 15 por casos registrados</sup>"),
            font=dict(size=13),
        ),
        'height': max(300, n * 28 + 80),
        'yaxis': dict(autorange='reversed', gridcolor="#334155"),
        'xaxis': dict(gridcolor="#334155", title="Casos"),
    })
    return fig


def fig_slope(d_breakdown: pl.DataFrame, tipo: str) -> go.Figure:
    """Slope chart: cases per state, first vs last year in selection."""
    years = sorted(d_breakdown['Año'].unique().cast(pl.Int32).to_list())
    if len(years) < 2:
        return go.Figure().update_layout(**CHART_LAYOUT, height=460)

    y_first = years[0]
    # Exclude partial 2026 from endpoint
    y_last = 2025 if years[-1] == 2026 else years[-1]
    if y_first == y_last:
        return go.Figure().update_layout(**CHART_LAYOUT, height=460)

    _ini = (d_breakdown.filter(pl.col('Año') == y_first)
            .group_by('Entidad').agg(pl.col('Casos').sum().alias('ini')))
    _fin = (d_breakdown.filter(pl.col('Año') == y_last)
            .group_by('Entidad').agg(pl.col('Casos').sum().alias('fin')))
    slope_d = (
        _fin.join(_ini, on='Entidad', how='inner')
        .with_columns(((pl.col('fin') - pl.col('ini')) / pl.col('ini') * 100).alias('pct'))
        .sort('fin', descending=True)
    )
    if slope_d.is_empty():
        return go.Figure().update_layout(**CHART_LAYOUT, height=460)

    n_up = slope_d.filter(pl.col('pct') > 0).height
    n_dn = slope_d.filter(pl.col('pct') <= 0).height
    n_total = len(slope_d)
    fig = go.Figure()

    for i, row in enumerate(slope_d.iter_rows(named=True)):
        worsened = row['pct'] > 0
        color = "#E84855" if worsened else "#3BB273"
        # Label only top-3 and bottom-3 by end-year volume
        show_label = i < 3 or i >= n_total - 3
        fig.add_trace(go.Scatter(
            x=[str(y_first), str(y_last)],
            y=[row['ini'], row['fin']],
            mode='lines+markers' + ('+text' if show_label else ''),
            line=dict(color=color, width=1.5),
            marker=dict(color=color, size=6),
            text=[None, row['Entidad']] if show_label else None,
            textposition='middle right',
            textfont=dict(size=8, color=color),
            showlegend=False,
            hovertemplate=(
                f"<b>{row['Entidad']}</b><br>"
                f"%{{x}}: %{{y:,}}  Δ {row['pct']:+.1f}%<extra></extra>"
            ),
        ))

    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines+markers',
                              line=dict(color='#E84855'), marker=dict(color='#E84855', size=7),
                              name=f'▲ Empeoró ({n_up} estados)'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines+markers',
                              line=dict(color='#3BB273'), marker=dict(color='#3BB273', size=7),
                              name=f'▼ Mejoró ({n_dn} estados)'))

    subtitle_tipo = 'Todos los delitos' if tipo == 'Todos los delitos' else tipo
    title = (
        f"<b>{n_up} de {n_total} estados registraron más delitos en {y_last} que en {y_first}</b>"
        f"<br><sup style='color:#94A3B8'>{subtitle_tipo} · comparativa {y_first} vs {y_last}</sup>"
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=480,
        xaxis=dict(type='category', gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="#334155", title="Casos registrados"),
        legend=dict(orientation="h", y=-0.08, x=0),
        margin=dict(t=55, b=60, l=10, r=160),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ('margin', 'xaxis', 'yaxis')},
    )
    return fig


def fig_claims(d_claims: pl.DataFrame) -> go.Figure:
    """3-panel trend chart for homicidio doloso, feminicidio, violencia familiar."""
    panels = [
        ('Homicidio doloso', '#E84855', 'rgba(232,72,85,0.12)', 2019, "Pico 2019"),
        ('Feminicidio',      '#A855F7', 'rgba(168,85,247,0.12)', 2022, "Pico 2022"),
        ('Violencia familiar', '#F4A261', 'rgba(244,162,97,0.12)', 2023, "Pico 2023"),
    ]
    subtitles = [
        "Homicidio doloso: cayó 32% desde el pico de 2019",
        "Feminicidios: más del doble entre 2015 y 2022",
        "Violencia familiar: +118% en una década",
    ]
    fig = make_subplots(rows=1, cols=3, subplot_titles=subtitles, horizontal_spacing=0.09)

    for col_i, (delito, color, fill, peak_yr, ann_label) in enumerate(panels, start=1):
        sub = d_claims.filter(pl.col('Delito_Clave') == delito).sort('Año')
        if sub.is_empty():
            continue
        fig.add_trace(go.Scatter(
            x=sub['Año'].to_list(), y=sub['Casos'].to_list(),
            mode='lines+markers', line=dict(color=color, width=2),
            fill='tozeroy', fillcolor=fill, showlegend=False,
            hovertemplate="<b>%{x}</b><br>Casos: %{y:,}<extra></extra>",
        ), row=1, col=col_i)

        pk = sub.filter(pl.col('Año') == peak_yr)
        if not pk.is_empty():
            pk_val = int(pk['Casos'][0])
            ax_x = 'x' if col_i == 1 else f'x{col_i}'
            ax_y = 'y' if col_i == 1 else f'y{col_i}'
            fig.add_annotation(
                x=peak_yr, y=pk_val,
                text=f"<b>{ann_label}</b><br>{pk_val:,}",
                font=dict(color=color, size=9),
                arrowcolor=color, arrowwidth=1, ax=28, ay=-35,
                xref=ax_x, yref=ax_y, showarrow=True,
                bgcolor="#0F172A", borderpad=3,
            )

    fig.update_layout(
        height=380,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#CBD5E1', margin=dict(t=55, b=40, l=10, r=10),
    )
    for col_i in range(1, 4):
        fig.update_xaxes(gridcolor="#334155", dtick=2, row=1, col=col_i)
        fig.update_yaxes(gridcolor="#334155", row=1, col=col_i)
    for ann in fig.layout.annotations:
        if ann.text in subtitles:
            ann.font = dict(size=11, color='#CBD5E1')
    return fig


def _fmt(n: float) -> str:
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000: return f"{n/1_000:.1f}k"
    return f"{n:,.0f}"


def fig_monthly_grid(d: pl.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=3, subplot_titles=DELITOS_CLAVE,
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )

    for i, delito in enumerate(DELITOS_CLAVE):
        r, c = divmod(i, 3)
        sub = d.filter(pl.col('Delito_Clave') == delito).sort(['Año', 'MesNum'])
        if sub.is_empty():
            continue

        data_pts = list(sub.iter_rows(named=True))
        x = [f"{pt['Mes'][:3]} '{str(pt['Año'])[2:]}" for pt in data_pts]
        y = [pt['Casos'] for pt in data_pts]
        casos_map = {(pt['Año'], pt['MesNum']): pt['Casos'] for pt in data_pts}
        y_ref = [casos_map.get((pt['Año'] - 1, pt['MesNum'])) for pt in data_pts]

        fig.add_trace(go.Scatter(
            x=x, y=y, mode='lines',
            line=dict(color='#F59E0B', width=2.5, shape='spline'),
            showlegend=False,
            hovertemplate="%{x}: %{y:,}<extra></extra>",
        ), row=r + 1, col=c + 1)

        if any(v is not None for v in y_ref):
            fig.add_trace(go.Scatter(
                x=x, y=y_ref, mode='lines',
                line=dict(color='#22D3EE', width=1.2, shape='spline'),
                connectgaps=False, showlegend=False,
                hovertemplate="%{x} (año ant.): %{y:,}<extra></extra>",
            ), row=r + 1, col=c + 1)

        fig.add_trace(go.Scatter(
            x=[x[-1]], y=[y[-1]], mode='markers+text',
            marker=dict(color='#F59E0B', size=6),
            text=[_fmt(y[-1])], textposition='top left',
            textfont=dict(size=8, color='#F59E0B'),
            showlegend=False, hoverinfo='skip',
        ), row=r + 1, col=c + 1)

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
        height=950, paper_bgcolor='#0F172A', plot_bgcolor='#0A1628',
        font_color='#CBD5E1', margin=dict(t=50, b=30, l=50, r=20),
    )
    for ann in fig.layout.annotations:
        if ann.text in DELITOS_CLAVE:
            ann.font = dict(size=10, color='#CBD5E1')
    return fig


def compute_kpis(d_yr: pl.DataFrame, d_ent: pl.DataFrame, d_tipo_totals: pl.DataFrame,
                 tipo: str, yr_range: tuple) -> tuple:
    total = int(d_yr['Casos'].sum())
    last_yr = min(yr_range[1], 2025)
    prior_yr = last_yr - 1
    t_last = int(d_yr.filter(pl.col('Año') == last_yr)['Casos'].sum())
    t_prior = int(d_yr.filter(pl.col('Año') == prior_yr)['Casos'].sum())
    if t_prior > 0:
        pct = (t_last - t_prior) / t_prior * 100
        arrow = "▲" if pct > 0 else "▼"
        yoy = f"{arrow} {abs(pct):.1f}% vs {prior_yr}"
        yoy_color = "#E84855" if pct > 0 else "#3BB273"
    else:
        yoy, yoy_color = "—", "#94A3B8"

    top_ent_row = d_ent.sort('Casos', descending=True).head(1)
    top_ent = top_ent_row['Entidad'][0] if len(d_ent) > 0 else "—"
    top_ent_n = f"{int(top_ent_row['Casos'][0]):,} casos" if len(d_ent) > 0 else ""

    if tipo == 'Todos los delitos':
        top_tipo_row = d_tipo_totals.sort('Casos', descending=True).head(1)
        top_tipo = top_tipo_row['Tipo de delito'][0] if len(d_tipo_totals) > 0 else "—"
        tipo_total = d_tipo_totals['Casos'].sum()
        tipo_pct = (f"{int(top_tipo_row['Casos'][0]/tipo_total*100)}% del total"
                    if len(d_tipo_totals) > 0 else "")
    else:
        top_tipo = tipo
        tipo_pct = ""

    return f"{total:,.0f}", _fmt(t_last), yoy, yoy_color, top_ent, top_ent_n, top_tipo, tipo_pct, last_yr


# ── Layout ────────────────────────────────────────────────────────────────────

app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.title = "Incidencia Delictiva · Fuero Común"

app.layout = html.Div(
    style={"backgroundColor": "#0F172A", "minHeight": "100vh",
           "padding": "24px", "fontFamily": "sans-serif"},
    children=[
        html.H2("Incidencia Delictiva del Fuero Común",
                style={"color": "#F8FAFC", "marginBottom": "4px"}),
        html.P("Carpetas de investigación por municipio · 2015–2026 · SESNSP",
               style={"color": "#64748B", "marginBottom": "20px"}),

        dbc.Row([
            dbc.Col([
                html.Label("Rango de años", style={"color": "#94A3B8", "fontSize": "12px"}),
                dcc.RangeSlider(
                    id="sl-años", min=MIN_AÑO, max=MAX_AÑO, value=[MIN_AÑO, 2025], step=1,
                    marks={y: {"label": str(y), "style": {"color": "#94A3B8", "fontSize": "11px"}}
                           for y in AÑOS},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ], md=12),
        ], className="mb-3"),

        dcc.Tabs(id="tabs", value="tab-resumen", children=[

            # ── Tab 1: Resumen general ─────────────────────────────────────────
            dcc.Tab(label="Resumen general", value="tab-resumen",
                    style=TAB_STYLE, selected_style=TAB_SEL, children=[

                dbc.Row([
                    dbc.Col([
                        html.Label("Tipo de delito", style={"color": "#94A3B8", "fontSize": "12px"}),
                        dcc.Dropdown(
                            id="dd-tipo",
                            options=[{"label": t, "value": t} for t in TIPOS],
                            value="Todos los delitos", clearable=False,
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
                    dbc.Col(html.Div(dcc.Graph(id="chart-slope"),
                                     style={**CARD_STYLE, "padding": "8px"}), md=12),
                ], className="mb-3"),
            ]),

            # ── Tab 2: Alto impacto mensual ────────────────────────────────────
            dcc.Tab(label="Alto impacto · mensual", value="tab-mensual",
                    style=TAB_STYLE, selected_style=TAB_SEL, children=[

                dbc.Row([
                    dbc.Col([
                        html.Label("Estado", style={"color": "#94A3B8", "fontSize": "12px"}),
                        dcc.Dropdown(
                            id="dd-estado",
                            options=[{"label": e, "value": e} for e in ESTADOS],
                            value="Nacional", clearable=False,
                            style={"backgroundColor": "#1E293B", "color": "#0F172A"},
                        ),
                    ], md=4),
                ], className="mb-3 mt-3"),

                dbc.Row([
                    dbc.Col(html.Div(dcc.Graph(id="chart-monthly-grid"),
                                     style={**CARD_STYLE, "padding": "12px"}), md=12),
                ], className="mb-3"),
            ]),

            # ── Tab 3: Delitos de alto impacto ────────────────────────────────
            dcc.Tab(label="Delitos de alto impacto", value="tab-claims",
                    style=TAB_STYLE, selected_style=TAB_SEL, children=[
                dbc.Row([
                    dbc.Col(html.Div(dcc.Graph(id="chart-claims"),
                                     style={**CARD_STYLE, "padding": "12px"}), md=12),
                ], className="mb-3 mt-3"),
            ]),

        ]),

        html.P("Fuente: Secretariado Ejecutivo del Sistema Nacional de Seguridad Pública (SESNSP)",
               style={"color": "#475569", "fontSize": "11px", "textAlign": "center",
                      "marginTop": "16px"}),
    ],
)

# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("kpi-row", "children"),
    Output("chart-trend", "figure"),
    Output("chart-map", "figure"),
    Output("chart-ranking", "figure"),
    Output("chart-slope", "figure"),
    Input("dd-tipo", "value"),
    Input("sl-años", "value"),
)
def update_all(tipo: str, yr_range: list):
    yr0, yr1 = yr_range
    yr_mask = pl.col('Año').is_between(yr0, yr1)

    if tipo == 'Todos los delitos':
        d_yr        = agg_yr_bien.filter(yr_mask)
        d_ent       = agg_yr_ent.filter(yr_mask).group_by(['Entidad', 'Clave_Ent']).agg(pl.col('Casos').sum())
        d_tipo_tots = agg_yr_tipo.filter(yr_mask).group_by('Tipo de delito').agg(pl.col('Casos').sum())
        d_yr_kpi    = agg_yr_bien.filter(yr_mask).group_by('Año').agg(pl.col('Casos').sum())
        d_breakdown = agg_yr_ent.filter(yr_mask)
    else:
        tipo_mask   = pl.col('Tipo de delito') == tipo
        d_yr        = agg_yr_tipo.filter(yr_mask & tipo_mask).group_by('Año').agg(pl.col('Casos').sum()).sort('Año')
        d_ent       = agg_yr_ent_tipo.filter(yr_mask & tipo_mask).group_by(['Entidad', 'Clave_Ent']).agg(pl.col('Casos').sum())
        d_tipo_tots = agg_yr_tipo.filter(yr_mask & tipo_mask).group_by('Tipo de delito').agg(pl.col('Casos').sum())
        d_yr_kpi    = d_yr
        d_breakdown = agg_yr_ent_tipo.filter(yr_mask & tipo_mask).group_by(['Año', 'Entidad']).agg(pl.col('Casos').sum())

    total, t_last_fmt, yoy, yoy_color, top_ent, top_ent_n, top_tipo, tipo_pct, last_yr = compute_kpis(
        d_yr_kpi, d_ent, d_tipo_tots, tipo, (yr0, yr1)
    )

    kpis = dbc.Row([
        kpi_card(f"Casos en {last_yr}", t_last_fmt, "#CBD5E1",
                 delta=yoy, delta_color=yoy_color),
        kpi_card("Acumulado del período", total, "#CBD5E1"),
        kpi_card("Estado con más casos", top_ent, "#F4A261",
                 delta=top_ent_n, delta_color="#64748B"),
        kpi_card(
            "Delito más frecuente" if tipo == 'Todos los delitos' else "Tipo seleccionado",
            top_tipo if len(top_tipo) <= 30 else top_tipo[:28] + "…",
            "#2E86AB",
            delta=tipo_pct or None, delta_color="#64748B",
        ),
    ], className="g-2").children

    return (
        kpis,
        fig_trend(d_yr, tipo),
        fig_map(d_ent),
        fig_ranking(d_ent),
        fig_slope(d_breakdown, tipo),
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


@app.callback(
    Output("chart-claims", "figure"),
    Input("sl-años", "value"),
)
def update_claims(yr_range: list):
    yr0, yr1 = yr_range
    d = agg_yr_claims.filter(pl.col('Año').is_between(yr0, min(yr1, 2025)))
    return fig_claims(d)


if __name__ == "__main__":
    app.run(debug=True, port=8060)
