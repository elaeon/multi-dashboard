#!/usr/bin/env python3
"""Build worldcup_match_positions.csv from the FIFA match dataset.

Positions 1–4 are exact (determined by final / 3rd-place match results).
Within every other elimination tier, teams are ranked by total goals scored
across the full tournament, using goals in favour as the tiebreaker so that
every position in the output is unique per edition.

Usage:
    uv run python scripts/build_match_positions.py
"""
import polars as pl
from pathlib import Path

ROOT      = Path(__file__).parent.parent
DATA_IN   = ROOT / "data/fifa/FIFA World Cup 1930-2022 All Match Dataset.csv"
DATA_OUT  = ROOT / "data/fifa/worldcup_match_positions.csv"
STATS_OUT = ROOT / "data/fifa/worldcup_country_stats.csv"

# Historical name → canonical name used in the output matrix
TEAM_NORM = {
    "West Germany":          "Germany",
    "Soviet Union":          "Russia",
    "Czechoslovakia":        "Czech Republic",
    "Dutch East Indies":     "Indonesia",
    "Serbia and Montenegro": "Serbia-Montenegro",
    "Zaire":                 "DR Congo",
}

# Stage name → numeric rank (higher = deeper in tournament)
STAGE_RANK = {
    "group stage":        1,
    "final round":        2,   # 1950 round-robin pool
    "second group stage": 3,   # 1974, 1978, 1982
    "round of 16":        4,   # 1934/38 first KO round + 1986–2022
    "quarter-finals":     5,   # 1954–1970, 1986–2022
    "semi-finals":        6,
    "third-place match":  7,
    "final":              8,
}

YEARS = [
    1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970,
    1974, 1978, 1982, 1986, 1990, 1994, 1998, 2002, 2006,
    2010, 2014, 2018, 2022,
]

# Expected number of participating teams per edition (for verification)
EXPECTED_PARTICIPANTS = {
    1930: 13, 1934: 16, 1938: 15, 1950: 13, 1954: 16, 1958: 16,
    1962: 16, 1966: 16, 1970: 16, 1974: 16, 1978: 16, 1982: 24,
    1986: 24, 1990: 24, 1994: 24, 1998: 32, 2002: 32, 2006: 32,
    2010: 32, 2014: 32, 2018: 32, 2022: 32,
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_long() -> pl.DataFrame:
    """Return long-format DataFrame: one row per team per match."""
    raw = pl.read_csv(DATA_IN, encoding="latin1")

    # Exclude voided matches (only applies to 1938 replays)
    raw = raw.filter(pl.col("Replayed") == 0)

    raw = raw.with_columns([
        pl.col("Tournament Id").str.slice(3).cast(pl.Int32).alias("year"),
        pl.col("Home Team Name").replace(TEAM_NORM).alias("home"),
        pl.col("Away Team Name").replace(TEAM_NORM).alias("away"),
    ])

    home = raw.select([
        "year",
        pl.col("home").alias("team"),
        pl.col("Stage Name").alias("stage"),
        pl.col("Home Team Score").alias("gf"),
        pl.col("Away Team Score").alias("ga"),
        pl.col("Home Team Win").alias("won"),
        pl.col("Draw").alias("draw"),
    ])
    away = raw.select([
        "year",
        pl.col("away").alias("team"),
        pl.col("Stage Name").alias("stage"),
        pl.col("Away Team Score").alias("gf"),
        pl.col("Home Team Score").alias("ga"),
        pl.col("Away Team Win").alias("won"),
        pl.col("Draw").alias("draw"),
    ])
    long = pl.concat([home, away])

    # Add numeric stage rank for easy max/comparison
    long = long.with_columns(
        pl.col("stage").map_elements(
            lambda s: STAGE_RANK.get(s, 0), return_dtype=pl.Int64
        ).alias("stage_rank")
    )
    return long


# ── Position assignment ───────────────────────────────────────────────────────

def assign_positions(year: int, yd: pl.DataFrame) -> dict[str, int]:
    """Return {team: position} for every team that played in this edition.

    Strategy:
      - Positions 1–4: derived from actual match results (final, 3rd-place match).
      - All other tiers: teams sorted by total goals scored (descending).
      - Tiers ordered best→worst: middle tier (QF / 2nd-group-stage losers),
        lower tier (R16 losers), group-stage tier.
    """
    stages = set(yd["stage"].unique().to_list())

    # Per-team totals across the full edition (goals used as tiebreaker)
    stats = yd.group_by("team").agg([
        pl.col("gf").sum().alias("goals"),
        pl.col("stage_rank").max().alias("max_stage"),
    ])

    positions: dict[str, int] = {}

    # ── 1950 special: no "final" match, positions 1–4 from final-round standings
    if "final round" in stages and "final" not in stages:
        fr_stats = (
            yd.filter(pl.col("stage") == "final round")
            .group_by("team").agg([
                (pl.col("won").sum() * 2 + pl.col("draw").sum()).alias("pts"),
                pl.col("gf").sum().alias("fr_gf"),
            ])
            .join(stats.select(["team", "goals"]), on="team")
            .sort(["pts", "fr_gf", "goals"], descending=True)
        )
        for rank, row in enumerate(fr_stats.iter_rows(named=True), 1):
            positions[row["team"]] = rank
        # Group-stage eliminated teams
        assigned = set(positions)
        nxt = 5
        for row in (
            stats.filter(~pl.col("team").is_in(list(assigned)))
            .sort("goals", descending=True)
            .iter_rows(named=True)
        ):
            positions[row["team"]] = nxt
            nxt += 1
        return positions

    # ── Positions 1 & 2: final winner / loser ────────────────────────────────
    for row in yd.filter(pl.col("stage") == "final").iter_rows(named=True):
        positions[row["team"]] = 1 if row["won"] == 1 else 2

    # ── Positions 3 & 4: 3rd-place match, or SF losers for 1930 ──────────────
    if "third-place match" in stages:
        for row in yd.filter(pl.col("stage") == "third-place match").iter_rows(named=True):
            positions[row["team"]] = 3 if row["won"] == 1 else 4
    else:
        # 1930: no 3rd-place match — semi-final losers ranked by goals
        sf_losers = (
            stats.filter(
                (pl.col("max_stage") == STAGE_RANK["semi-finals"]) &
                ~pl.col("team").is_in(list(positions))
            ).sort("goals", descending=True)
        )
        for rank, row in enumerate(sf_losers.iter_rows(named=True), 3):
            positions[row["team"]] = rank

    # ── Remaining tiers, assigned sequentially starting at position 5 ─────────
    assigned = set(positions)
    nxt = 5

    # Middle tier: QF losers (most eras) or 2nd-group-stage losers (1974/78/82)
    if "quarter-finals" in stages:
        mid_rank = STAGE_RANK["quarter-finals"]
    elif "second group stage" in stages:
        mid_rank = STAGE_RANK["second group stage"]
    else:
        mid_rank = None  # 1930: no middle tier

    if mid_rank is not None:
        for row in (
            stats.filter((pl.col("max_stage") == mid_rank) & ~pl.col("team").is_in(list(assigned)))
            .sort("goals", descending=True)
            .iter_rows(named=True)
        ):
            positions[row["team"]] = nxt
            nxt += 1
            assigned.add(row["team"])

    # Lower tier: R16 losers (1934/38 first KO round + 1986–2022)
    if "round of 16" in stages:
        for row in (
            stats.filter((pl.col("max_stage") == STAGE_RANK["round of 16"]) & ~pl.col("team").is_in(list(assigned)))
            .sort("goals", descending=True)
            .iter_rows(named=True)
        ):
            positions[row["team"]] = nxt
            nxt += 1
            assigned.add(row["team"])

    # Group-stage tier: all remaining teams
    for row in (
        stats.filter(~pl.col("team").is_in(list(assigned)))
        .sort("goals", descending=True)
        .iter_rows(named=True)
    ):
        positions[row["team"]] = nxt
        nxt += 1

    return positions


# ── Matrix construction ───────────────────────────────────────────────────────

def build_matrix(long: pl.DataFrame) -> pl.DataFrame:
    records: dict[str, dict[int, int]] = {}

    for year in YEARS:
        yd = long.filter(pl.col("year") == year)
        for team, pos in assign_positions(year, yd).items():
            records.setdefault(team, {})[year] = pos

    all_teams = sorted(records)
    rows = [
        {"country": t, **{str(y): records[t].get(y) for y in YEARS}}
        for t in all_teams
    ]
    return pl.DataFrame(rows)


# ── Verification ──────────────────────────────────────────────────────────────

def verify(df: pl.DataFrame) -> bool:
    year_cols = [c for c in df.columns if c != "country"]
    print("\n── Verification ────────────────────────────────────────")
    all_ok = True
    for yc in year_cols:
        col = df[yc].drop_nulls()
        n   = len(col)
        issues = []
        if not (col == 1).any():      issues.append("no champion")
        if not (col == 2).any():      issues.append("no runner-up")
        if col.n_unique() != n:       issues.append("duplicate positions")
        exp = EXPECTED_PARTICIPANTS.get(int(yc), n)
        if n != exp:                  issues.append(f"count {n} ≠ expected {exp}")
        mark = "✓" if not issues else "✗"
        if issues:
            all_ok = False
        suffix = f"  — {', '.join(issues)}" if issues else ""
        print(f"  {mark} {yc}: {n} teams{suffix}")

    # Spot-check: 1954 full standings
    print("\n── 1954 full standings (expected: GER 1, HUN 2, AUT 3, URU 4) ─")
    tbl = df.select(["country", "1954"]).drop_nulls().sort("1954")
    for row in tbl.iter_rows(named=True):
        print(f"  {row['1954']:>2}. {row['country']}")

    return all_ok


# ── Country stats ────────────────────────────────────────────────────────────

def build_stats(long: pl.DataFrame) -> pl.DataFrame:
    """Aggregate all-time stats per country across all World Cup editions."""
    return (
        long.group_by("team").agg([
            pl.len().alias("Matches"),
            pl.col("won").sum().alias("Wins"),
            pl.col("draw").sum().alias("Draws"),
            (pl.len() - pl.col("won").sum() - pl.col("draw").sum()).alias("Losses"),
            pl.col("gf").sum().alias("Goals"),
            pl.col("ga").sum().alias("Against"),
        ])
        .with_columns(
            ((pl.col("Wins") * 3 + pl.col("Draws")) / pl.col("Matches")).round(2).alias("PPG")
        )
        .sort("PPG", descending=True)
        .rename({"team": "Country"})
        .select(["Country", "Matches", "Wins", "Draws", "Losses", "Goals", "Against", "PPG"])
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Reading {DATA_IN.name} …")
    long = load_long()
    print(f"  {len(long)} team-match rows across {long['year'].n_unique()} editions")

    print("Assigning positions …")
    df = build_matrix(long)

    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(DATA_OUT)
    print(f"Saved → {DATA_OUT}  ({len(df)} countries × {len(df.columns) - 1} editions)")

    ok = verify(df)
    if not ok:
        raise SystemExit(1)
    print("\nAll checks passed.")

    print("\nBuilding country stats …")
    stats = build_stats(long)
    stats.write_csv(STATS_OUT)
    print(f"Saved → {STATS_OUT}  ({len(stats)} countries)")
    print("\nTop 10 by PPG:")
    for row in stats.head(10).iter_rows(named=True):
        print(f"  {row['PPG']:.2f}  {row['Country']:<25}  "
              f"{row['Wins']}W {row['Draws']}D {row['Losses']}L  "
              f"GF {row['Goals']}  GA {row['Against']}")


if __name__ == "__main__":
    main()
