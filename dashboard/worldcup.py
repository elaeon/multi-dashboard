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

HOSTS = {
    1930: "Uruguay",      1934: "Italy",         1938: "France",
    1950: "Brazil",       1954: "Switzerland",   1958: "Sweden",
    1962: "Chile",        1966: "England",        1970: "Mexico",
    1974: "Germany",      1978: "Argentina",      1982: "Spain",
    1986: "Mexico",       1990: "Italy",          1994: "United States",
    1998: "France",       2002: "South Korea",    2006: "Germany",
    2010: "South Africa", 2014: "Brazil",         2018: "Russia",
    2022: "Qatar",
}

POINT_MAP = {1: 7, 2: 5, 3: 4, 4: 3, 5: 2, 9: 1, 17: 0}

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

# Match-based positions (exact per-team ranks, goals as tiebreaker within each tier)
_df_match_wide = pl.read_csv("data/fifa/worldcup_match_positions.csv")
_match_year_cols = [c for c in _df_match_wide.columns if c != "country"]
match_pos_long = (
    _df_match_wide
    .unpivot(index="country", on=_match_year_cols, variable_name="year_str", value_name="position")
    .with_columns(pl.col("year_str").cast(pl.Int32).alias("year"))
    .filter(pl.col("position").is_not_null())
)

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

# PPG per team per edition (for the timeline chart)
ppg_by_team_year = (
    team_year.with_columns(
        ((pl.col("wins") * 3 + pl.col("draws")) / pl.col("matches")).round(3).alias("ppg")
    )
    .select(["year", "team", "ppg", "matches", "wins", "draws", "losses",
             "goals_for", "goals_against"])
)

# Mean PPG for teams that exited at each knockout stage (reference lines)
_ppg_stage_raw = (
    ppg_by_team_year
    .join(pos_long.rename({"country": "team"}), on=["team", "year"], how="inner")
    .with_columns(
        pl.when(pl.col("position").is_in([1, 2])).then(pl.lit("Final"))
          .when(pl.col("position").is_in([3, 4])).then(pl.lit("Semi-final"))
          .when(pl.col("position") == 5).then(pl.lit("Quarter-final"))
          .when(pl.col("position") == 9).then(pl.lit("Round of 16"))
          .otherwise(None)
          .alias("stage_label")
    )
    .filter(pl.col("stage_label").is_not_null())
    .group_by("stage_label")
    .agg(pl.col("ppg").mean().round(2).alias("mean_ppg"))
)
stage_mean_ppg = {r["stage_label"]: r["mean_ppg"] for r in _ppg_stage_raw.iter_rows(named=True)}

# All-time country stats table
country_stats = (
    team_year.group_by("team").agg([
        pl.col("matches").sum(),
        pl.col("wins").sum(),
        pl.col("draws").sum(),
        pl.col("losses").sum(),
        pl.col("goals_for").sum().alias("goals"),
        pl.col("goals_against").sum().alias("against"),
    ])
    .with_columns(
        ((pl.col("wins") * 3 + pl.col("draws")) / pl.col("matches")).round(2).alias("ppg")
    )
    .sort("ppg", descending=True)
    .rename({
        "team": "Country", "matches": "Matches", "wins": "Wins",
        "draws": "Draws", "losses": "Losses",
        "goals": "Goals", "against": "Against",
    })
    .select(["Country", "Matches", "Wins", "Draws", "Losses", "Goals", "Against", "ppg"])
    .rename({"ppg": "PPG"})
)

# ── Insight aggregates (computed once at startup) ─────────────────────────────

# 1. Goals-per-match regression
_yr = goals_by_year["year"].cast(pl.Float64).to_numpy()
_gpm = goals_by_year["avg_goals"].to_numpy()
_slope, _intercept = np.polyfit(_yr, _gpm, 1)
_gpm_fit = (_slope * _yr + _intercept).tolist()
_r2 = float(np.corrcoef(_yr, _gpm)[0, 1] ** 2)

# 2. Champion consistency: median / best / worst position for every ever-champion
_champ_countries = pos_long.filter(pl.col("position") == 1)["country"].unique().to_list()
champion_stats = (
    pos_long.filter(pl.col("country").is_in(_champ_countries))
    .group_by("country").agg([
        pl.col("position").median().alias("median_pos"),
        pl.col("position").min().alias("best_pos"),
        pl.col("position").max().alias("worst_pos"),
        pl.col("position").count().alias("editions"),
        (pl.col("position") == 1).sum().alias("titles"),
    ])
    .sort("median_pos", descending=True)   # worst at bottom → best at top in chart
)

# 3. Edition points for top 8 nations (1st=7pts … group=0pts)
_pts_long = pos_long.with_columns(
    pl.when(pl.col("position") == 1).then(7)
    .when(pl.col("position") == 2).then(5)
    .when(pl.col("position") == 3).then(4)
    .when(pl.col("position") == 4).then(3)
    .when(pl.col("position") == 5).then(2)
    .when(pl.col("position") == 9).then(1)
    .otherwise(0)
    .alias("pts")
)
_top8 = (
    _pts_long.group_by("country").agg(pl.col("pts").sum().alias("total_pts"))
    .sort("total_pts", descending=True).head(8)["country"].to_list()
)
era_pts = _pts_long.filter(pl.col("country").is_in(_top8)).sort(["country", "year"])

# 4. Host nation final positions
_host_rows = []
for _y, _host in HOSTS.items():
    _r = pos_long.filter((pl.col("country") == _host) & (pl.col("year") == _y))
    _host_rows.append({
        "year": _y, "host": _host,
        "position": int(_r["position"][0]) if len(_r) > 0 else None,
    })
host_perf = pl.DataFrame(_host_rows)

# 5. Goals-per-match vs final position (per-match, not total — avoids games-played confound)
gpm_pos = (
    team_year
    .with_columns((pl.col("goals_for") / pl.col("matches")).alias("gpm"))
    .join(pos_long.rename({"country": "team"}), on=["team", "year"], how="inner")
    .select(["team", "year", "gpm", "position", "matches"])
    .filter(pl.col("position").is_not_null())
)

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


# Country list for the position tracker dropdown (from match-based positions CSV)
_pos_countries_list = sorted(match_pos_long["country"].unique().to_list())
_DEFAULT_NATIONS = ["Brazil", "Germany", "Argentina", "Italy", "France"]
_LINE_PALETTE = [
    "#FFD700", "#2E86AB", "#3BB273", "#F4A261", "#E84855",
    "#A8A9AD", "#CD7F32", "#7C3AED", "#0EA5E9", "#F97316",
    "#10B981", "#EC4899",
]


def fig_positions_multi(countries: list) -> go.Figure:
    """Position over time for multiple countries, one line each (exact numeric positions)."""
    fig = go.Figure()
    for i, country in enumerate(countries):
        d = match_pos_long.filter(pl.col("country") == country).sort("year")
        if len(d) == 0:
            continue
        color = _LINE_PALETTE[i % len(_LINE_PALETTE)]
        fig.add_trace(go.Scatter(
            x=d["year"].to_list(), y=d["position"].to_list(),
            mode="lines+markers", name=country,
            line=dict(color=color, width=2),
            marker=dict(size=9, color=color, line=dict(color="#0F172A", width=1)),
            hovertemplate=f"<b>{country}</b> %{{x}}<br>Position: %{{y}}<extra></extra>",
        ))
    fig.update_layout(
        **CHART_LAYOUT,
        title="Final Position per World Cup Edition",
        yaxis=dict(
            autorange="reversed", gridcolor="#334155",
            title="Final Position",
        ),
        xaxis=dict(
            gridcolor="#334155", title="Year",
            tickmode="array", tickvals=years_list,
            ticktext=[str(y) for y in years_list], tickangle=-45,
        ),
        legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=11)),
        margin=dict(b=90, t=40),
        height=520,
    )
    return fig


def fig_ppg_over_time(countries: list) -> go.Figure:
    """PPG per World Cup edition for selected countries."""
    fig = go.Figure()
    for i, country in enumerate(countries):
        d = ppg_by_team_year.filter(pl.col("team") == country).sort("year")
        if len(d) == 0:
            continue
        color = _LINE_PALETTE[i % len(_LINE_PALETTE)]
        fig.add_trace(go.Scatter(
            x=d["year"].to_list(),
            y=d["ppg"].to_list(),
            mode="lines+markers",
            name=country,
            line=dict(color=color, width=2),
            marker=dict(size=8, color=color, line=dict(color="#0F172A", width=1)),
            customdata=list(zip(
                d["wins"].to_list(), d["draws"].to_list(),
                d["losses"].to_list(), d["matches"].to_list(),
            )),
            hovertemplate=(
                f"<b>{country}</b> %{{x}}<br>"
                "PPG: %{y:.2f}<br>"
                "%{customdata[0]}W %{customdata[1]}D %{customdata[2]}L "
                "(%{customdata[3]} matches)<extra></extra>"
            ),
        ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="#475569",
                  annotation_text="1 pt/game", annotation_font_color="#64748B",
                  annotation_position="bottom right")
    _stage_lines = [
        ("Round of 16",   "#3BB273"),
        ("Quarter-final", "#F4A261"),
        ("Semi-final",    "#CD7F32"),
        ("Final",         "#FFD700"),
    ]
    for stage_label, color in _stage_lines:
        val = stage_mean_ppg.get(stage_label)
        if val is not None:
            fig.add_hline(
                y=val, line_dash="dash", line_color=color, line_width=1, opacity=0.6,
                annotation_text=f"{stage_label} ({val:.2f})",
                annotation_font_color=color, annotation_font_size=10,
                annotation_position="top right",
            )
    fig.update_layout(
        **CHART_LAYOUT,
        title="Points Per Game per World Cup Edition",
        xaxis=dict(
            gridcolor="#334155", title="Year",
            tickmode="array", tickvals=years_list,
            ticktext=[str(y) for y in years_list], tickangle=-45,
        ),
        yaxis=dict(gridcolor="#334155", title="PPG", range=[-0.1, 3.2]),
        legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=11)),
        margin=dict(b=90, t=40),
        height=440,
    )
    return fig


def fig_best5_ppg(countries: list) -> go.Figure:
    """Top-5 PPG World Cup editions per selected country, sorted globally by PPG."""
    rows = []
    for i, country in enumerate(countries):
        d = (
            ppg_by_team_year.filter(pl.col("team") == country)
            .sort("ppg", descending=True)
            .head(5)
        )
        for row in d.iter_rows(named=True):
            rows.append({
                "country": country,
                "label": f"{country} ({row['year']})",
                "ppg": row["ppg"],
                "wins": row["wins"],
                "draws": row["draws"],
                "losses": row["losses"],
                "matches": row["matches"],
            })

    if not rows:
        return go.Figure()

    result = pl.DataFrame(rows).sort("ppg", descending=False)
    country_colors = {c: _LINE_PALETTE[i % len(_LINE_PALETTE)] for i, c in enumerate(countries)}
    bar_colors = [country_colors[c] for c in result["country"].to_list()]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=result["ppg"].to_list(),
        y=result["label"].to_list(),
        orientation="h",
        marker_color=bar_colors,
        customdata=list(zip(
            result["wins"].to_list(), result["draws"].to_list(),
            result["losses"].to_list(), result["matches"].to_list(),
        )),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "PPG: %{x:.2f}<br>"
            "%{customdata[0]}W %{customdata[1]}D %{customdata[2]}L "
            "(%{customdata[3]} matches)<extra></extra>"
        ),
        showlegend=False,
    ))
    for country in countries:
        fig.add_trace(go.Bar(
            x=[None], y=[None], orientation="h",
            name=country, marker_color=country_colors[country],
        ))

    fig.update_layout(
        **CHART_LAYOUT,
        title="Best 5 PPG Editions per Country",
        xaxis=dict(gridcolor="#334155", title="PPG", range=[0, 3.5]),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", y=-0.18, x=0),
        margin=dict(t=40, b=70, l=10, r=10),
        height=max(300, len(rows) * 28 + 80),
    )
    return fig


def fig_h2h(matches: pl.DataFrame, team1: str, team2: str) -> go.Figure:
    """Goal-difference timeline of all World Cup meetings between two teams."""
    if len(matches) == 0:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT,
                          title=f"{team1} vs {team2} — No World Cup encounters",
                          height=300)
        return fig

    RESULT_COLORS = {"Win": "#3BB273", "Draw": "#F4A261", "Loss": "#E84855"}
    fig = go.Figure()
    for result, color in RESULT_COLORS.items():
        sub = matches.filter(pl.col("result") == result)
        if len(sub) == 0:
            continue
        scores = [f"{gf}–{ga}" for gf, ga in zip(sub["gf"].to_list(), sub["ga"].to_list())]
        fig.add_trace(go.Scatter(
            x=sub["year"].to_list(),
            y=sub["gd"].to_list(),
            mode="markers+text",
            name=result,
            marker=dict(color=color, size=16, line=dict(color="#0F172A", width=1)),
            text=scores,
            textposition="top center",
            textfont=dict(color=color, size=11),
            customdata=sub["Stage Name"].to_list(),
            hovertemplate=(
                "<b>%{x} · %{customdata}</b><br>"
                f"{team1} %{{text}} {team2}<br>"
                "Goal diff: %{y:+d}<extra></extra>"
            ),
        ))

    fig.add_hline(y=0, line_color="#475569", line_dash="dot")
    fig.update_layout(
        **CHART_LAYOUT,
        title=f"{team1} vs {team2} — World Cup Encounters",
        xaxis=dict(
            gridcolor="#334155", title="Year",
            tickmode="array", tickvals=years_list,
            ticktext=[str(y) for y in years_list], tickangle=-45,
        ),
        yaxis=dict(gridcolor="#334155", title=f"Goal Difference ({team1})"),
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=50, b=80),
        height=420,
    )
    return fig


# ── Insight figure factories ─────────────────────────────────────────────────

def fig_goals_regression() -> go.Figure:
    """Goals/match trend with linear regression line (Insights tab)."""
    yr_list = goals_by_year["year"].to_list()
    gpm_list = goals_by_year["avg_goals"].to_list()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yr_list, y=gpm_list, mode="lines+markers", name="Avg Goals/Match",
        line=dict(color="#2E86AB", width=2), marker=dict(size=8, color="#2E86AB"),
        hovertemplate="<b>%{x}</b>: %{y:.2f} goals/match<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=yr_list, y=_gpm_fit, mode="lines",
        name=f"Trend ({_slope*10:+.3f}/decade)",
        line=dict(color="#E84855", width=2, dash="dash"), hoverinfo="skip",
    ))
    fig.add_annotation(
        x=1954, y=5.38, text="1954: 5.38 g/m<br>(Hungary era)",
        showarrow=True, arrowhead=2, arrowcolor="#F4A261",
        font=dict(color="#F4A261", size=11),
        bgcolor="#1E293B", bordercolor="#F4A261", borderwidth=1,
    )
    fig.update_layout(
        **CHART_LAYOUT,
        title=f"Goals/Match Decline — slope {_slope*10:.3f}/decade · R²={_r2:.2f} · p<0.001",
        xaxis=dict(gridcolor="#334155", title="Year"),
        yaxis=dict(gridcolor="#334155", title="Avg Goals/Match", range=[1.5, 6.5]),
        legend=dict(orientation="h", y=1.12),
        height=380,
    )
    return fig


def fig_host_performance() -> go.Figure:
    """Host nation final position per edition."""
    d = host_perf.filter(pl.col("position").is_not_null()).sort("year")
    pos_vals = d["position"].to_list()
    colors = [POS_COLOR.get(p, "#64748B") for p in pos_vals]
    labels = [POS_LABEL.get(p, str(p)) for p in pos_vals]
    fig = go.Figure(go.Scatter(
        x=d["year"].to_list(), y=pos_vals,
        mode="markers+text",
        marker=dict(color=colors, size=18, line=dict(color="#0F172A", width=1)),
        text=d["host"].to_list(), textposition="top center",
        customdata=labels,
        hovertemplate="<b>%{text}</b> (%{x})<br>%{customdata}<extra></extra>",
    ))
    _mean_pos = float(d["position"].cast(pl.Float64).mean())
    fig.add_hline(y=_mean_pos, line_dash="dash", line_color="#94A3B8",
                  annotation_text=f"Mean pos {_mean_pos:.1f}",
                  annotation_font_color="#94A3B8")
    fig.update_layout(
        **CHART_LAYOUT,
        title="Host Nation Finish — 6 of 22 hosts won outright (median: 4th)",
        xaxis=dict(gridcolor="#334155", title="Year", tickmode="array",
                   tickvals=d["year"].to_list(), tickangle=-45),
        yaxis=dict(
            gridcolor="#334155", autorange="reversed",
            tickvals=[1, 2, 3, 4, 5, 9, 17],
            ticktext=["Champion", "Runner-up", "3rd", "4th", "QF", "R16", "Group"],
            title="Final Position",
        ),
        margin=dict(t=40, b=70),
        height=380,
    )
    return fig


def fig_champion_consistency() -> go.Figure:
    """Dumbbell: best → median → worst position for each ever-champion nation."""
    countries = champion_stats["country"].to_list()
    medians  = champion_stats["median_pos"].to_list()
    bests    = champion_stats["best_pos"].to_list()
    worsts   = champion_stats["worst_pos"].to_list()
    titles   = champion_stats["titles"].to_list()
    editions = champion_stats["editions"].to_list()

    x_lines, y_lines = [], []
    for b, w, c in zip(bests, worsts, countries):
        x_lines += [b, w, None]
        y_lines += [c, c, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_lines, y=y_lines, mode="lines",
        line=dict(color="#475569", width=2), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=bests, y=countries, mode="markers", name="Best finish",
        marker=dict(color="#FFD700", size=10, symbol="circle-open",
                    line=dict(color="#FFD700", width=2)),
        hovertemplate="<b>%{y}</b><br>Best: pos %{x}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=worsts, y=countries, mode="markers", name="Worst finish",
        marker=dict(color="#E84855", size=10),
        hovertemplate="<b>%{y}</b><br>Worst: pos %{x}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=medians, y=countries, mode="markers", name="Median finish",
        marker=dict(color="#2E86AB", size=14, symbol="diamond"),
        customdata=list(zip(titles, editions)),
        hovertemplate="<b>%{y}</b><br>Median: %{x:.1f} · %{customdata[0]} titles · %{customdata[1]} apps<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title="Champion Nations — Position Range Across All Appearances",
        xaxis=dict(
            gridcolor="#334155", title="Final Position",
            tickvals=[1, 2, 3, 4, 5, 9, 17],
            ticktext=["1st", "2nd", "3rd", "4th", "QF", "R16", "Group"],
            range=[0, 18],
        ),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", y=-0.18, x=0),
        margin=dict(b=70, t=40, l=100),
        height=max(300, len(countries) * 42 + 90),
    )
    return fig


def fig_era_dominance() -> go.Figure:
    """Edition points per year for top 8 nations (1st=7 … group=0)."""
    _palette = ["#FFD700", "#2E86AB", "#3BB273", "#F4A261",
                "#E84855", "#A8A9AD", "#CD7F32", "#B87333"]
    fig = go.Figure()
    for i, country in enumerate(_top8):
        d = era_pts.filter(pl.col("country") == country).sort("year")
        fig.add_trace(go.Scatter(
            x=d["year"].to_list(), y=d["pts"].to_list(),
            mode="lines+markers", name=country,
            line=dict(color=_palette[i % len(_palette)], width=2),
            marker=dict(size=6),
            hovertemplate=f"<b>{country}</b> %{{x}}: %{{y}} pts<extra></extra>",
        ))
    fig.update_layout(
        **CHART_LAYOUT,
        title="Edition Points — Top 8 Nations (Brazil peak 1994–2002; Germany peak 1982–1990)",
        xaxis=dict(gridcolor="#334155", tickmode="array",
                   tickvals=years_list, tickangle=-45, title="Year"),
        yaxis=dict(gridcolor="#334155", title="Points",
                   tickvals=[0, 1, 2, 3, 5, 7],
                   ticktext=["Group", "R16", "QF", "4th", "2nd", "Champion"]),
        legend=dict(orientation="h", y=-0.3, x=0, font=dict(size=11)),
        margin=dict(b=100, t=40),
        height=420,
    )
    return fig


def fig_gpm_vs_position() -> go.Figure:
    """Goals scored per match vs final position — avoids games-played confound."""
    _agg = (
        gpm_pos.group_by("position").agg(
            pl.col("gpm").median().alias("median_gpm"),
            pl.len().alias("n"),
        ).sort("position")
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gpm_pos["gpm"].to_list(), y=gpm_pos["position"].to_list(),
        mode="markers",
        marker=dict(color="#2E86AB", size=5, opacity=0.35),
        customdata=list(zip(gpm_pos["team"].to_list(), gpm_pos["year"].to_list())),
        hovertemplate="<b>%{customdata[0]}</b> %{customdata[1]}<br>%{x:.2f} gpm · pos %{y}<extra></extra>",
        showlegend=False, name="",
    ))
    fig.add_trace(go.Scatter(
        x=_agg["median_gpm"].to_list(), y=_agg["position"].to_list(),
        mode="markers+lines",
        marker=dict(color="#FFD700", size=12, symbol="diamond"),
        line=dict(color="#FFD700", width=2, dash="dot"),
        customdata=_agg["n"].to_list(),
        name="Median gpm per tier",
        hovertemplate="<b>Position %{y}</b><br>Median: %{x:.2f} gpm (n=%{customdata})<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title="Goals Scored per Match vs Final Position — Spearman ρ = −0.59 (p<0.001)",
        xaxis=dict(gridcolor="#334155", title="Goals Scored per Match"),
        yaxis=dict(
            gridcolor="#334155", autorange="reversed",
            tickvals=[1, 2, 3, 4, 5, 9, 17],
            ticktext=["Champion", "Runner-up", "3rd", "4th", "QF", "R16", "Group"],
            title="Final Position",
        ),
        legend=dict(orientation="h", y=1.1),
        height=400,
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

        # ── Tab 2: Position Tracker ───────────────────────────────────────────
        dcc.Tab(label="Position Tracker", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Select nations to compare:",
                               style={"color": "#94A3B8", "marginTop": "16px", "marginBottom": "6px",
                                      "display": "block"}),
                    dcc.Dropdown(
                        id="positions-country-dropdown",
                        options=[{"label": c, "value": c} for c in _pos_countries_list],
                        value=_DEFAULT_NATIONS,
                        multi=True,
                        placeholder="Choose one or more countries…",
                        style={"backgroundColor": "#1E293B", "color": "#0F172A"},
                        className="mb-2",
                    ),
                ], md=12),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="positions-multi-chart", config={"displayModeBar": False}), md=12),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="positions-ppg-chart", config={"displayModeBar": False}), md=12),
            ]),
        ]),

        # ── Tab 3: Team Explorer ─────────────────────────────────────────────
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

        # ── Tab 4: Head to Head ───────────────────────────────────────────────
        dcc.Tab(label="Head to Head", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Team 1:", style={"color": "#94A3B8", "marginTop": "16px"}),
                    dcc.Dropdown(
                        id="h2h-team1",
                        options=[{"label": t, "value": t} for t in teams_list],
                        value="Brazil",
                        style={"backgroundColor": "#1E293B", "color": "#0F172A"},
                        className="mb-3",
                    ),
                ], md=4),
                dbc.Col([
                    html.Label("Team 2:", style={"color": "#94A3B8", "marginTop": "16px"}),
                    dcc.Dropdown(
                        id="h2h-team2",
                        options=[{"label": t, "value": t} for t in teams_list],
                        value="Germany",
                        style={"backgroundColor": "#1E293B", "color": "#0F172A"},
                        className="mb-3",
                    ),
                ], md=4),
            ]),
            html.Div(id="h2h-kpis"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="h2h-chart", config={"displayModeBar": False}), md=12),
            ]),
        ]),

        # ── Tab 5: Stats ──────────────────────────────────────────────────────
        dcc.Tab(label="Stats", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            html.H5("Country Statistics", style={"color": "#F8FAFC", "marginTop": "16px", "marginBottom": "4px"}),
            html.P("Aggregated across selected World Cup editions. Click any column header to sort.",
                   style={"color": "#94A3B8", "fontSize": "0.85rem", "marginBottom": "16px"}),
            dbc.Row([
                dbc.Col([
                    html.Label("Edition range:", style={"color": "#94A3B8", "fontSize": "0.85rem", "marginBottom": "8px", "display": "block"}),
                    dcc.RangeSlider(
                        id="stats-year-slider",
                        min=years_list[0],
                        max=years_list[-1],
                        step=None,
                        marks={y: {"label": str(y), "style": {"color": "#94A3B8", "fontSize": "11px"}} for y in years_list},
                        value=[years_list[0], years_list[-1]],
                        allowCross=False,
                    ),
                ], md=12, style={"paddingBottom": "24px"}),
            ]),
            dash_table.DataTable(
                id="stats-table",
                data=country_stats.to_dicts(),
                columns=[{"name": c, "id": c} for c in country_stats.columns],
                sort_action="native",
                sort_by=[{"column_id": "PPG", "direction": "desc"}],
                style_table={"overflowX": "auto"},
                style_header={
                    "backgroundColor": "#1E293B", "color": "#94A3B8",
                    "fontWeight": "600", "border": "1px solid #334155",
                },
                style_cell={
                    "backgroundColor": "#0F172A", "color": "#CBD5E1",
                    "border": "1px solid #1E293B",
                    "textAlign": "center", "padding": "8px 14px",
                    "fontFamily": "monospace",
                },
                style_cell_conditional=[
                    {"if": {"column_id": "Country"}, "textAlign": "left",
                     "fontFamily": "inherit", "fontWeight": "500"},
                ],
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "#0D1B2A"},
                    {"if": {"filter_query": "{PPG} >= 2", "column_id": "PPG"},
                     "color": "#FFD700", "fontWeight": "700"},
                    {"if": {"filter_query": "{PPG} < 1", "column_id": "PPG"},
                     "color": "#E84855"},
                ],
                page_size=25,
            ),
            html.Hr(style={"borderColor": "#334155", "margin": "28px 0 20px"}),
            html.H6("Best 5 PPG Editions", style={"color": "#F8FAFC", "marginBottom": "8px"}),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id="best5-country-dropdown",
                        options=[{"label": t, "value": t} for t in sorted(ppg_by_team_year["team"].unique().to_list())],
                        value=_DEFAULT_NATIONS,
                        multi=True,
                        placeholder="Choose countries…",
                        style={"backgroundColor": "#1E293B", "color": "#0F172A"},
                        className="mb-2",
                    ),
                ], md=12),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="best5-ppg-chart", config={"displayModeBar": False}), md=12),
            ]),
        ]),

        # ── Tab 6: Insights ───────────────────────────────────────────────────
        dcc.Tab(label="Insights", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            html.H5("Key Statistical Findings", style={"color": "#F8FAFC", "marginTop": "16px", "marginBottom": "12px"}),
            dbc.Row([
                dbc.Col(html.Div([
                    html.Div("−0.24 g/m per decade", style={"fontSize": "1.3rem", "fontWeight": "700", "color": "#E84855"}),
                    html.Div("Scoring decline 1930–2022 · R²=0.57 · p<0.001", style={"fontSize": "0.78rem", "color": "#94A3B8"}),
                ], style=CARD_STYLE), md=3),
                dbc.Col(html.Div([
                    html.Div("6 / 22 hosts won", style={"fontSize": "1.3rem", "fontWeight": "700", "color": "#FFD700"}),
                    html.Div("Host nations: median finish 4th · mean 4.55", style={"fontSize": "0.78rem", "color": "#94A3B8"}),
                ], style=CARD_STYLE), md=3),
                dbc.Col(html.Div([
                    html.Div("Brazil: floor pos 9", style={"fontSize": "1.3rem", "fontWeight": "700", "color": "#3BB273"}),
                    html.Div("Only champion nation never group-stage eliminated", style={"fontSize": "0.78rem", "color": "#94A3B8"}),
                ], style=CARD_STYLE), md=3),
                dbc.Col(html.Div([
                    html.Div("ρ = 0.38 momentum", style={"fontSize": "1.3rem", "fontWeight": "700", "color": "#2E86AB"}),
                    html.Div("Prior finish predicts next, but regresses to mean", style={"fontSize": "0.78rem", "color": "#94A3B8"}),
                ], style=CARD_STYLE), md=3),
            ], className="g-2 mb-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_goals_regression(),      config={"displayModeBar": False}), md=6),
                dbc.Col(dcc.Graph(figure=fig_host_performance(),       config={"displayModeBar": False}), md=6),
            ], className="mt-2"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_champion_consistency(),   config={"displayModeBar": False}), md=5),
                dbc.Col(dcc.Graph(figure=fig_era_dominance(),          config={"displayModeBar": False}), md=7),
            ], className="mt-2"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_gpm_vs_position(),        config={"displayModeBar": False}), md=12),
            ], className="mt-2"),
        ]),
    ]),
])


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("best5-ppg-chart", "figure"),
    Input("best5-country-dropdown", "value"),
)
def update_best5_ppg(selected):
    if not selected:
        return go.Figure(layout=dict(**CHART_LAYOUT,
                                     xaxis=dict(visible=False), yaxis=dict(visible=False)))
    return fig_best5_ppg(selected)


@app.callback(
    Output("h2h-chart", "figure"),
    Output("h2h-kpis", "children"),
    Input("h2h-team1", "value"),
    Input("h2h-team2", "value"),
)
def update_h2h(team1, team2):
    if not team1 or not team2 or team1 == team2:
        empty = go.Figure(layout=dict(**CHART_LAYOUT,
                                      xaxis=dict(visible=False), yaxis=dict(visible=False)))
        return empty, []

    matches = df_m.filter(
        ((pl.col("home") == team1) & (pl.col("away") == team2)) |
        ((pl.col("home") == team2) & (pl.col("away") == team1))
    ).with_columns([
        pl.when(pl.col("home") == team1)
          .then(pl.col("Home Team Score"))
          .otherwise(pl.col("Away Team Score"))
          .alias("gf"),
        pl.when(pl.col("home") == team1)
          .then(pl.col("Away Team Score"))
          .otherwise(pl.col("Home Team Score"))
          .alias("ga"),
        pl.when(pl.col("home") == team1)
          .then(pl.col("Home Team Win"))
          .otherwise(pl.col("Away Team Win"))
          .alias("t1_won"),
    ]).with_columns([
        (pl.col("gf") - pl.col("ga")).alias("gd"),
        pl.when(pl.col("t1_won") == 1).then(pl.lit("Win"))
          .when(pl.col("Draw") == 1).then(pl.lit("Draw"))
          .otherwise(pl.lit("Loss"))
          .alias("result"),
    ]).sort("year")

    n_games = len(matches)
    t1_wins = int((matches["result"] == "Win").sum())
    draws   = int((matches["result"] == "Draw").sum())
    t2_wins = n_games - t1_wins - draws
    t1_goals = int(matches["gf"].sum())
    t2_goals = int(matches["ga"].sum())

    kpis_row = dbc.Row([
        kpi(f"{team1} Wins", t1_wins),
        kpi("Draws", draws),
        kpi(f"{team2} Wins", t2_wins),
        kpi("Matches", n_games),
        kpi("Goals", f"{t1_goals}–{t2_goals}"),
    ], className="g-2 mb-3")

    return fig_h2h(matches, team1, team2), kpis_row


@app.callback(
    Output("stats-table", "data"),
    Input("stats-year-slider", "value"),
)
def update_stats_table(year_range):
    lo, hi = year_range
    d = team_year.filter(pl.col("year").is_between(lo, hi))
    stats = (
        d.group_by("team").agg([
            pl.col("matches").sum(),
            pl.col("wins").sum(),
            pl.col("draws").sum(),
            pl.col("losses").sum(),
            pl.col("goals_for").sum().alias("goals"),
            pl.col("goals_against").sum().alias("against"),
        ])
        .with_columns(
            ((pl.col("wins") * 3 + pl.col("draws")) / pl.col("matches")).round(2).alias("ppg")
        )
        .sort("ppg", descending=True)
        .rename({
            "team": "Country", "matches": "Matches", "wins": "Wins",
            "draws": "Draws", "losses": "Losses",
            "goals": "Goals", "against": "Against", "ppg": "PPG",
        })
        .select(["Country", "Matches", "Wins", "Draws", "Losses", "Goals", "Against", "PPG"])
    )
    return stats.to_dicts()


@app.callback(
    Output("positions-multi-chart", "figure"),
    Input("positions-country-dropdown", "value"),
)
def update_positions_chart(selected):
    if not selected:
        return go.Figure(layout=dict(**CHART_LAYOUT,
                                     yaxis=dict(visible=False), xaxis=dict(visible=False)))
    return fig_positions_multi(selected)


@app.callback(
    Output("positions-ppg-chart", "figure"),
    Input("positions-country-dropdown", "value"),
)
def update_positions_ppg_chart(selected):
    if not selected:
        return go.Figure(layout=dict(**CHART_LAYOUT,
                                     xaxis=dict(visible=False), yaxis=dict(visible=False)))
    return fig_ppg_over_time(selected)


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
