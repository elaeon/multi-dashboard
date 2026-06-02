#!/usr/bin/env python3
"""FIFA World Cup 1930-2022 — match data + position matrix + consistency check."""

import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import numpy as np

# ── Style ─────────────────────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#CBD5E1",
)
CARD_STYLE = {
    "background": "#1E293B", "border": "1px solid #334155",
    "borderRadius": "8px", "padding": "16px", "textAlign": "center",
}
TAB_STYLE   = {"backgroundColor": "#0F172A", "color": "#94A3B8", "borderTop": "none"}
TAB_SEL     = {"backgroundColor": "#1E293B", "color": "#F8FAFC",
               "borderTop": "2px solid #2E86AB", "fontWeight": "600"}

# ── Mappings ──────────────────────────────────────────────────────────────────

# Match dataset → canonical name (same as positions CSV)
TEAM_NORM = {
    "Czechoslovakia":        "Czech Republic",
    "Dutch East Indies":     "Indonesia",
    "Serbia and Montenegro": "Serbia-Montenegro",
    "Soviet Union":          "Russia",
    "West Germany":          "Germany",
    "Zaire":                 "DR Congo",
    # 2002 RSSSF stored as "Ireland" instead of "Republic of Ireland"
    "Republic of Ireland":   "Republic of Ireland",
}

# Also normalise positions-side: "Ireland" 2002 should be "Republic of Ireland"
POS_NORM = {
    "Ireland": "Republic of Ireland",
}

# Years where group-stage eliminated teams get pos 9 (no R16)
PRE_R16_YEARS = {1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970, 1974, 1978}

STAGE_ORDER = {
    "group stage": 1, "final round": 2, "second group stage": 3,
    "round of 16": 4, "quarter-finals": 5, "semi-finals": 6,
    "third-place match": 7, "final": 8,
}

POS_COLOR = {1: "#FFD700", 2: "#A8A9AD", 3: "#CD7F32", 4: "#B87333",
             5: "#2E86AB",  9: "#3BB273", 17: "#475569"}
POS_LABEL = {1: "Champion", 2: "Runner-up", 3: "3rd Place", 4: "4th Place",
             5: "Quarter-final", 9: "Round of 16 / 1st KO", 17: "Group Stage"}

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    # ---- Matches ----
    df_raw = pl.read_csv(
        "data/fifa/FIFA World Cup 1930-2022 All Match Dataset.csv",
        encoding="latin1",
    )
    df_m = df_raw.with_columns([
        pl.col("Tournament Id").str.slice(3).cast(pl.Int32).alias("year"),
        pl.col("Home Team Name").replace(TEAM_NORM).alias("home"),
        pl.col("Away Team Name").replace(TEAM_NORM).alias("away"),
        pl.col("Stage Name").replace(STAGE_ORDER).cast(pl.Int32).alias("stage_rank"),
    ])

    # Long format: one row per team per match
    home_rows = df_m.select([
        "year",
        pl.col("home").alias("team"),
        "stage_rank",
        pl.col("Stage Name").alias("stage"),
        pl.col("Home Team Win").alias("won"),
        pl.col("Draw").alias("draw"),
        pl.col("Home Team Score").alias("gf"),
        pl.col("Away Team Score").alias("ga"),
    ])
    away_rows = df_m.select([
        "year",
        pl.col("away").alias("team"),
        "stage_rank",
        pl.col("Stage Name").alias("stage"),
        pl.col("Away Team Win").alias("won"),
        pl.col("Draw").alias("draw"),
        pl.col("Away Team Score").alias("gf"),
        pl.col("Home Team Score").alias("ga"),
    ])
    df_long = pl.concat([home_rows, away_rows])

    # Per-team-year stats
    team_year = df_long.group_by(["year", "team"]).agg([
        pl.len().alias("matches"),
        pl.col("won").sum().alias("wins"),
        pl.col("draw").sum().alias("draws"),
        pl.col("gf").sum().alias("goals_for"),
        pl.col("ga").sum().alias("goals_against"),
        pl.col("stage_rank").max().alias("max_stage_rank"),
    ]).with_columns(
        (pl.col("matches") - pl.col("wins") - pl.col("draws")).alias("losses")
    )

    # Attach final/3rd-place match results (to pin pos 1/2 and 3/4)
    decisive = df_long.filter(pl.col("stage").is_in(["final", "third-place match"]))
    team_year = (
        team_year
        .join(
            decisive.filter(pl.col("stage") == "final")
                    .select(["year", "team", pl.col("won").alias("won_final")]),
            on=["year", "team"], how="left",
        )
        .join(
            decisive.filter(pl.col("stage") == "third-place match")
                    .select(["year", "team", pl.col("won").alias("won_3rd")]),
            on=["year", "team"], how="left",
        )
    )

    # Inferred position from match data
    team_year = team_year.with_columns(
        pl.when(pl.col("max_stage_rank") == 8)
          .then(pl.when(pl.col("won_final") == 1).then(1).otherwise(2))
        .when(pl.col("max_stage_rank") == 7)
          .then(pl.when(pl.col("won_3rd") == 1).then(3).otherwise(4))
        .when(pl.col("max_stage_rank") == 6).then(None)   # semi loser (1930, no 3rd place)
        .when(pl.col("max_stage_rank") == 5).then(5)
        .when(pl.col("max_stage_rank") == 4).then(9)
        .when(pl.col("max_stage_rank") == 3).then(5)
        .when(pl.col("max_stage_rank") == 2).then(None)   # 1950 final round
        .when(pl.col("max_stage_rank") == 1)
          .then(pl.when(pl.col("year").is_in(list(PRE_R16_YEARS))).then(9).otherwise(17))
        .otherwise(None)
        .cast(pl.Int64)
        .alias("inferred_pos")
    )

    # ---- Positions (RSSSF) ----
    df_pos_wide = pl.read_csv("data/fifa/worldcup_positions.csv").with_columns(
        pl.col("country").replace(POS_NORM)
    )
    year_cols = [c for c in df_pos_wide.columns if c != "country"]
    pos_long = (
        df_pos_wide
        .unpivot(index="country", on=year_cols, variable_name="year_str", value_name="position")
        .with_columns(pl.col("year_str").cast(pl.Int32).alias("year"))
        .filter(pl.col("position").is_not_null())
    )

    # ---- Consistency join ----
    consistency = (
        team_year
        .join(pos_long.rename({"country": "team"}), on=["team", "year"], how="inner")
        .with_columns(
            pl.when(pl.col("inferred_pos").is_null())
              .then(pl.lit("undetermined"))
            .when(pl.col("inferred_pos") == pl.col("position"))
              .then(pl.lit("match"))
            # 1950 final round: position 1-4 is consistent with stage_rank 2
            .when((pl.col("max_stage_rank") == 2) & pl.col("position").is_in([1, 2, 3, 4]))
              .then(pl.lit("match"))
            .otherwise(pl.lit("mismatch"))
            .alias("consistency")
        )
    )

    return df_m, df_long, team_year, pos_long, consistency, df_pos_wide


df_m, df_long, team_year, pos_long, consistency, df_pos_wide = load_data()

# Pre-compute some global aggregates
years_list = sorted(df_m["year"].unique().to_list())
teams_list = sorted(
    set(df_long["team"].unique().to_list()) | set(pos_long["country"].unique().to_list())
)

goals_by_year = (
    df_m.with_columns((pl.col("Home Team Score") + pl.col("Away Team Score")).alias("goals"))
    .group_by("year").agg(
        pl.col("goals").sum().alias("total_goals"),
        pl.len().alias("matches"),
        (pl.col("goals").sum() / pl.len()).alias("avg_goals"),
    )
    .sort("year")
)

total_goals = int(df_m.select((pl.col("Home Team Score") + pl.col("Away Team Score")).sum())[0, 0])
n_teams = df_long["team"].n_unique()

# Position matrix for heatmap (wide → values array)
pos_countries = df_pos_wide["country"].to_list()
year_cols = [c for c in df_pos_wide.columns if c != "country"]

# ── Figure factories ──────────────────────────────────────────────────────────

def fig_goals_timeline(d: pl.DataFrame) -> go.Figure:
    """Total goals and avg goals/match per World Cup."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d["year"].to_list(), y=d["total_goals"].to_list(),
        name="Total Goals", marker_color="#2E86AB", opacity=0.7,
        hovertemplate="<b>%{x}</b><br>Total: %{y} goals<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=d["year"].to_list(), y=d["avg_goals"].to_list(),
        name="Avg/Match", mode="lines+markers",
        line=dict(color="#F4A261", width=2),
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Avg: %{y:.2f} goals/match<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title="Goals per World Cup",
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(title="Total Goals", gridcolor="#334155"),
        yaxis2=dict(title="Avg Goals/Match", overlaying="y", side="right",
                    gridcolor="rgba(0,0,0,0)", range=[0, 7]),
        legend=dict(orientation="h", y=1.1),
        height=360,
    )
    return fig


def fig_stage_counts(d: pl.DataFrame) -> go.Figure:
    """Matches played per stage per year (stacked bar)."""
    stage_year = (
        d.group_by(["year", "Stage Name"])
         .agg(pl.len().alias("count"))
         .sort(["year", "Stage Name"])
    )
    stage_order = [
        "group stage", "final round", "second group stage",
        "round of 16", "quarter-finals", "semi-finals",
        "third-place match", "final",
    ]
    colors = {
        "group stage": "#334155", "final round": "#475569",
        "second group stage": "#1E3A5F",
        "round of 16": "#1D4E89", "quarter-finals": "#2E86AB",
        "semi-finals": "#F4A261", "third-place match": "#CD7F32",
        "final": "#FFD700",
    }
    fig = go.Figure()
    for stage in stage_order:
        sub = stage_year.filter(pl.col("Stage Name") == stage)
        if len(sub) == 0:
            continue
        fig.add_trace(go.Bar(
            x=sub["year"].to_list(), y=sub["count"].to_list(),
            name=stage.title(), marker_color=colors.get(stage, "#64748B"),
            hovertemplate=f"<b>{stage}</b>: %{{y}} matches (%{{x}})<extra></extra>",
        ))
    fig.update_layout(
        **CHART_LAYOUT,
        title="Matches per Stage per Edition",
        barmode="stack",
        legend=dict(orientation="h", y=-0.25, x=0, font=dict(size=11)),
        margin=dict(t=40, b=90, l=10, r=10),
        height=400,
    )
    return fig


def fig_position_heatmap() -> go.Figure:
    """Country × year position matrix heatmap."""
    z = []
    text = []
    for c in pos_countries:
        row_z, row_t = [], []
        for yc in year_cols:
            val = df_pos_wide.filter(pl.col("country") == c)[yc][0]
            row_z.append(val)
            row_t.append(POS_LABEL.get(val, "—") if val is not None else "Did not participate")
        z.append(row_z)
        text.append(row_t)

    colorscale = [
        [0.00, "#FFD700"], [0.10, "#A8A9AD"], [0.25, "#CD7F32"],
        [0.40, "#B87333"], [0.55, "#2E86AB"], [0.75, "#3BB273"],
        [1.00, "#475569"],
    ]

    fig = go.Figure(go.Heatmap(
        z=z, x=year_cols, y=pos_countries,
        text=text, hovertemplate="<b>%{y}</b> %{x}<br>%{text}<extra></extra>",
        colorscale=colorscale,
        zmin=1, zmax=17,
        showscale=True,
        colorbar=dict(
            tickvals=[1, 2, 3, 4, 5, 9, 17],
            ticktext=["1st", "2nd", "3rd", "4th", "QF", "R16", "Group"],
            title=dict(text="Position", font=dict(color="#CBD5E1")),
            tickfont=dict(color="#CBD5E1"),
        ),
        xgap=1, ygap=1,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
        title="Country × Year Position Matrix (RSSSF data)",
        xaxis=dict(side="top", tickangle=-45),
        height=max(500, len(pos_countries) * 12 + 120),
        margin=dict(t=60, b=20, l=130, r=60),
    )
    return fig


def fig_consistency_summary() -> go.Figure:
    """Bar chart of consistency check results."""
    counts = (
        consistency
        .group_by("consistency")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    colors = {"match": "#3BB273", "mismatch": "#E84855", "undetermined": "#94A3B8"}
    fig = go.Figure(go.Bar(
        x=counts["consistency"].to_list(),
        y=counts["count"].to_list(),
        marker_color=[colors.get(c, "#64748B") for c in counts["consistency"].to_list()],
        hovertemplate="<b>%{x}</b>: %{y} team-editions<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title="Consistency: Match-data stage vs RSSSF position",
        height=300,
        xaxis_title="", yaxis_title="Team-Editions",
    )
    return fig


def fig_team_positions(country: str) -> go.Figure:
    """Position over time for a single country (lower = better)."""
    d = pos_long.filter(pl.col("country") == country).sort("year")
    if len(d) == 0:
        return go.Figure()

    y_vals = d["position"].to_list()
    x_vals = d["year"].to_list()
    colors = [POS_COLOR.get(p, "#64748B") for p in y_vals]
    labels = [POS_LABEL.get(p, str(p)) for p in y_vals]

    fig = go.Figure()
    # Grey line connecting all points
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals, mode="lines",
        line=dict(color="#334155", width=1.5), showlegend=False, hoverinfo="skip",
    ))
    # Colored dots per position tier
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals, mode="markers+text",
        marker=dict(color=colors, size=12, line=dict(color="#0F172A", width=1)),
        text=[str(p) for p in y_vals], textposition="top center",
        customdata=labels,
        hovertemplate="<b>%{x}</b>: %{customdata}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title=f"{country} — Position each World Cup",
        yaxis=dict(
            autorange="reversed", gridcolor="#334155",
            tickvals=[1, 2, 3, 4, 5, 9, 17],
            ticktext=["1st", "2nd", "3rd", "4th", "QF", "R16", "Group"],
            title="Finish",
        ),
        xaxis=dict(gridcolor="#334155", title="Year", tickmode="array",
                   tickvals=years_list, ticktext=[str(y) for y in years_list], tickangle=-45),
        height=380,
    )
    return fig


def fig_team_goals(country: str) -> go.Figure:
    """Goals for and against per World Cup edition for a single country."""
    d = team_year.filter(pl.col("team") == country).sort("year")
    if len(d) == 0:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d["year"].to_list(), y=d["goals_for"].to_list(),
        name="Goals For", marker_color="#3BB273",
        hovertemplate="<b>%{x}</b><br>Scored: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=d["year"].to_list(), y=(-d["goals_against"]).to_list(),
        name="Goals Against", marker_color="#E84855",
        hovertemplate="<b>%{x}</b><br>Conceded: %{customdata}<extra></extra>",
        customdata=d["goals_against"].to_list(),
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title=f"{country} — Goals Scored / Conceded",
        barmode="relative",
        yaxis=dict(gridcolor="#334155", zeroline=True, zerolinecolor="#64748B"),
        legend=dict(orientation="h", y=1.1),
        height=300,
    )
    return fig


def fig_team_wdl(country: str) -> go.Figure:
    """Win/draw/loss record per edition for a single country."""
    d = team_year.filter(pl.col("team") == country).sort("year")
    if len(d) == 0:
        return go.Figure()

    fig = go.Figure()
    for col, color, name in [("wins", "#3BB273", "Wins"), ("draws", "#F4A261", "Draws"), ("losses", "#E84855", "Losses")]:
        fig.add_trace(go.Bar(
            x=d["year"].to_list(), y=d[col].to_list(),
            name=name, marker_color=color,
            hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y}}<extra></extra>",
        ))
    fig.update_layout(
        **CHART_LAYOUT,
        title=f"{country} — Match Record",
        barmode="stack",
        legend=dict(orientation="h", y=1.1),
        height=280,
    )
    return fig


# ── App layout ────────────────────────────────────────────────────────────────

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="FIFA World Cup Dashboard",
)

def kpi(title, value, sub=""):
    return dbc.Col(html.Div([
        html.Div(str(value), style={"fontSize": "2rem", "fontWeight": "700", "color": "#F8FAFC"}),
        html.Div(title, style={"fontSize": "0.85rem", "color": "#94A3B8", "marginTop": "2px"}),
        html.Div(sub, style={"fontSize": "0.75rem", "color": "#64748B"}) if sub else None,
    ], style=CARD_STYLE), md=2)


def _consistency_table():
    """Render the mismatches as a DCC DataTable."""
    mismatches = consistency.filter(pl.col("consistency") == "mismatch").sort(["year", "team"])
    if len(mismatches) == 0:
        return html.P("✓ No mismatches found between datasets.",
                      style={"color": "#3BB273", "fontWeight": "600"})

    rows = mismatches.select([
        "year", "team", "max_stage_rank", "inferred_pos", "position",
    ]).rename({
        "year": "Year", "team": "Country",
        "max_stage_rank": "Stage Rank (match data)",
        "inferred_pos": "Inferred Pos",
        "position": "RSSSF Pos",
    }).to_dicts()

    return dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in rows[0].keys()],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#1E293B", "color": "#94A3B8", "fontWeight": "600"},
        style_cell={"backgroundColor": "#0F172A", "color": "#CBD5E1", "border": "1px solid #334155"},
        style_data_conditional=[
            {"if": {"column_id": "Inferred Pos"}, "color": "#F4A261"},
            {"if": {"column_id": "RSSSF Pos"},   "color": "#E84855"},
        ],
        page_size=20,
    )


# Consistency stats
n_match = int(consistency.filter(pl.col("consistency") == "match").shape[0])
n_mismatch = int(consistency.filter(pl.col("consistency") == "mismatch").shape[0])
n_total = int(consistency.shape[0])
pct_match = f"{n_match / n_total * 100:.1f}%"


app.layout = html.Div(style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "24px"}, children=[
    # Header
    html.H1("FIFA World Cup 1930–2022",
            style={"color": "#F8FAFC", "fontWeight": "700", "marginBottom": "4px"}),
    html.P("Match results × RSSSF final positions — cross-dataset analysis",
           style={"color": "#94A3B8", "marginBottom": "24px"}),

    # KPI row
    dbc.Row([
        kpi("Editions", 22),
        kpi("Matches", f"{len(df_m):,}"),
        kpi("Total Goals", f"{total_goals:,}", f"{total_goals/len(df_m):.2f} per match"),
        kpi("Countries", n_teams, "ever participated"),
        kpi("Consistent entries", pct_match, f"{n_match}/{n_total} team-editions"),
        kpi("Real mismatches", n_mismatch, "between datasets"),
    ], className="g-2 mb-4"),

    # Tabs
    dcc.Tabs(style={"marginBottom": "16px"}, children=[

        # ── Tab 1: Overview ───────────────────────────────────────────────────
        dcc.Tab(label="Overview", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_goals_timeline(goals_by_year), config={"displayModeBar": False}), md=6),
                dbc.Col(dcc.Graph(figure=fig_stage_counts(df_m), config={"displayModeBar": False}), md=6),
            ], className="mt-3"),
        ]),

        # ── Tab 2: Position Matrix ────────────────────────────────────────────
        dcc.Tab(label="Position Matrix", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            html.P(
                "Color = final position in each World Cup. "
                "Gold = champion · Silver = runner-up · Bronze = 3rd/4th · "
                "Blue = quarter-final · Green = round of 16 / first KO · "
                "Grey = group stage. Blank = did not participate.",
                style={"color": "#94A3B8", "marginTop": "16px", "fontSize": "0.85rem"},
            ),
            dcc.Graph(
                figure=fig_position_heatmap(),
                config={"displayModeBar": False},
                style={"overflowY": "auto"},
            ),
        ]),

        # ── Tab 3: Consistency Check ──────────────────────────────────────────
        dcc.Tab(label="Consistency Check", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            dbc.Row([
                dbc.Col([
                    html.H5("Method", style={"color": "#F8FAFC", "marginTop": "16px"}),
                    html.P([
                        "For each team × edition, the ",
                        html.B("furthest stage reached"), " in the match dataset is mapped "
                        "to an expected final position and compared to the RSSSF-derived "
                        "position. Pre-R16 eras (1930–1978) assign ",
                        html.B("pos 9"), " to group-stage eliminates; post-R16 eras assign ",
                        html.B("pos 17"), ". 1950 final-round and semi-final entries "
                        "without a third-place match are marked undetermined.",
                    ], style={"color": "#94A3B8", "fontSize": "0.9rem"}),
                ], md=6),
                dbc.Col(
                    dcc.Graph(figure=fig_consistency_summary(), config={"displayModeBar": False}),
                    md=6,
                ),
            ]),

            html.H5("Mismatches", style={"color": "#F8FAFC", "marginTop": "8px"}),
            _consistency_table(),
        ]),

        # ── Tab 4: Team Explorer ──────────────────────────────────────────────
        dcc.Tab(label="Team Explorer", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Select country:", style={"color": "#94A3B8", "marginTop": "16px"}),
                    dcc.Dropdown(
                        id="team-dropdown",
                        options=[{"label": t, "value": t} for t in teams_list],
                        value="Brazil",
                        style={"backgroundColor": "#1E293B", "color": "#F8FAFC"},
                        className="mb-3",
                    ),
                ], md=4),
                dbc.Col(html.Div(id="team-kpis"), md=8),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="team-positions", config={"displayModeBar": False}), md=12),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="team-goals", config={"displayModeBar": False}), md=6),
                dbc.Col(dcc.Graph(id="team-wdl",   config={"displayModeBar": False}), md=6),
            ]),
        ]),
    ]),
])


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("team-positions", "figure"),
    Output("team-goals",     "figure"),
    Output("team-wdl",       "figure"),
    Output("team-kpis",      "children"),
    Input("team-dropdown",   "value"),
)
def update_team(country: str):
    if not country:
        empty = go.Figure()
        return empty, empty, empty, []

    # KPIs for the selected country
    ty = team_year.filter(pl.col("team") == country)
    editions = len(ty)
    total_m  = int(ty["matches"].sum())
    total_w  = int(ty["wins"].sum())
    total_gf = int(ty["goals_for"].sum())
    total_ga = int(ty["goals_against"].sum())
    best_pos = pos_long.filter(pl.col("country") == country)["position"].min()
    best_label = POS_LABEL.get(best_pos, "—") if best_pos is not None else "—"

    kpis = dbc.Row([
        kpi("Editions", editions),
        kpi("Matches", total_m),
        kpi("Wins",    total_w, f"{total_w/total_m*100:.0f}% win rate" if total_m else ""),
        kpi("Goals For",     total_gf),
        kpi("Goals Against", total_ga),
        kpi("Best Finish",   best_label),
    ], className="g-2 mt-3")

    return (
        fig_team_positions(country),
        fig_team_goals(country),
        fig_team_wdl(country),
        kpis,
    )


if __name__ == "__main__":
    app.run(debug=True, port=8060)
