"""
Zipf's law / power-law rank-size analysis for US and Mexican population data.

Two selectable fitting engines (--engine), both implementing the Clauset,
Shalizi & Newman (2009) method: continuous MLE of the power-law exponent
alpha, with xmin selected by minimizing the Kolmogorov-Smirnov distance
between data and fit. This replaces the OLS log-log regression used by the
original version of this script (and by dashboard/zipf_municipios.py's
zipf_slope()). Population counts are treated as continuous (standard
simplification at city/state scale).

  powerlaw (default)  Delegates entirely to the `powerlaw` library (Alstott,
                       Bullmore & Plenz 2014). No built-in Clauset-style
                       bootstrap GoF p-value or bootstrap CI -- alpha's CI
                       comes from the library's analytic standard error
                       (alpha +/- 1.96*sigma, asymptotic normal
                       approximation), and goodness-of-fit comes from
                       `distribution_compare` against a lognormal
                       alternative (R, p): R>0 & p<0.05 favors power-law,
                       R<0 & p<0.05 favors lognormal, p>=0.05 means the data
                       can't distinguish between them. Much faster (no
                       bootstrapping).
  custom               This script's own hand-rolled MLE/KS engine (used
                       before the `powerlaw` library was adopted), plus a
                       semi-parametric bootstrap goodness-of-fit p-value and
                       nonparametric bootstrap CIs for both alpha and xmin,
                       per Clauset et al. sec. 4.1 -- the CI/p-value method
                       `powerlaw` doesn't provide. Controlled by --gof-boot,
                       --ci-boot, --min-tail, --xmin-candidates, --seed.
                       Slower (bootstrapping).

Data sources:
  US cities: data/census/population/sub-est2025.csv (SUMLEV 162, place totals;
    SUMLEV 157 place-part rows used only to resolve --by-metro's place->CBSA join)
  US metro areas: data/census/population/cbsa-est2025-alldata.csv (CBSA-level rows)
  MX municipios: dashboard_data/conapo_pob_municipal.parquet
  MX metro areas (Zonas Metropolitanas): data/datos_gob/conapo/municipios_tipologia.csv
    (municipio -> metropoli crosswalk, joined to the CONAPO population parquet above --
    "Las Metropolis de Mexico 2020", SEDATU/CONAPO/INEGI)

Levels (all 4 pool nationally into one fit by default; --by-state/--by-metro break
the pool down into one fit per state/metro area instead):
  cities      US incorporated places
              --by-state: one fit PER US state
              --by-metro: one fit PER CBSA, using a population-weighted principal
              CBSA for the ~2% of places split across county lines into 2+ CBSAs;
              places with no CBSA-mapped county part are excluded (non-metro)
  municipios  MX municipios
              --by-state: one fit PER MX estado
              --by-metro: one fit PER Zona Metropolitana/Metropoli, using only the
              421 municipios that belong to one
  us-metro    the 387 US Metropolitan Statistical Areas
              --by-state: one fit PER state, using each metro's principal state
  mx-metro    the 92 Mexican Zonas Metropolitanas/Metropolis
              --by-state: one fit PER estado, using a population-weighted principal
              estado for the 8 zones spanning 2+ estados
              --list: also print all 92 zones ranked by population

Run:
  uv run python scripts/zipf/states_size.py
  uv run python scripts/zipf/states_size.py --level cities municipios --year 2020
  uv run python scripts/zipf/states_size.py --level cities --by-state --min-n 5
  uv run python scripts/zipf/states_size.py --level cities --by-metro --min-n 5
  uv run python scripts/zipf/states_size.py --level us-metro --by-state --min-n 5
  uv run python scripts/zipf/states_size.py --level mx-metro --by-state --min-n 5
  uv run python scripts/zipf/states_size.py --level mx-metro --list
  uv run python scripts/zipf/states_size.py --level municipios --by-metro --min-n 5
  uv run python scripts/zipf/states_size.py --engine custom --level mx-metro --gof-boot 50 --ci-boot 200
  uv run python scripts/zipf/states_size.py --output results.csv
"""
from __future__ import annotations

import argparse
import io
import re
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import powerlaw

US_CSV = Path("data/census/population/sub-est2025.csv")
US_CBSA_CSV = Path("data/census/population/cbsa-est2025-alldata.csv")
MX_PARQUET = Path("dashboard_data/conapo_pob_municipal.parquet")
MX_METRO_CSV = Path("data/datos_gob/conapo/municipios_tipologia.csv")

LEVELS = ["cities", "municipios", "us-metro", "mx-metro"]

STATE_ABBR = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}


def primary_state(name: str) -> str:
    """Principal state of a CBSA: the first state abbreviation listed in its official name
    (e.g. "Chicago-Naperville-Elgin, IL-IN" -> Illinois). Census orders constituent states
    by significance, so the first-listed state is the standard "principal state" convention."""
    suffix = re.search(r",\s*([A-Za-z-]+)$", name).group(1)
    abbr = suffix.split("-")[0]
    return STATE_ABBR.get(abbr, abbr)


# ── Statistical engine: powerlaw library ──────────────────────────────────

def analyze_powerlaw(x: np.ndarray, level: str, group: str) -> dict | None:
    """Fit a power law via `powerlaw.Fit` (continuous MLE + KS-minimizing xmin
    search) and compare it against a lognormal alternative."""
    if x.size < 5:
        print(f"  SKIP {level}/{group}: n={x.size}, too small to fit")
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = powerlaw.Fit(x, discrete=False, verbose=False)
        law = fit.power_law
        if not np.isfinite(law.alpha) or not np.isfinite(law.xmin):
            print(f"  SKIP {level}/{group}: n={x.size}, powerlaw fit failed")
            return None
        r, p = fit.distribution_compare("power_law", "lognormal", normalized_ratio=True)

    return {
        "Engine": "powerlaw", "Level": level, "Group": group, "N": int(x.size),
        "xmin": round(law.xmin, 1), "N_tail": int(np.sum(x >= law.xmin)),
        "alpha": round(law.alpha, 4), "KS_D": round(law.D, 4),
        "GoF_p_value": None,
        "LR_vs_lognormal": round(r, 3), "LR_p_value": round(p, 3),
        "Alpha_CI_low": round(law.alpha - 1.96 * law.sigma, 4),
        "Alpha_CI_high": round(law.alpha + 1.96 * law.sigma, 4),
        "Xmin_CI_low": None, "Xmin_CI_high": None,
    }


# ── Statistical engine: custom Clauset-Shalizi-Newman (2009) MLE/KS/bootstrap ─

def mle_alpha(x: np.ndarray, xmin: float) -> float:
    """Continuous MLE of the power-law exponent for the tail x >= xmin.
    Returns NaN if every tail value equals xmin (degenerate tail, e.g. a small
    bootstrap resample that drew the same value repeatedly) -- log(tail/xmin)
    is 0 for every point, so the MLE is undefined rather than a divide-by-zero."""
    tail = x[x >= xmin]
    log_sum = np.sum(np.log(tail / xmin))
    if log_sum <= 0:
        return float("nan")
    return 1.0 + tail.size / log_sum


def ks_distance(x: np.ndarray, xmin: float, alpha: float) -> float:
    """Max distance between the tail's empirical CDF and the fitted Pareto CDF."""
    tail = np.sort(x[x >= xmin])
    n = tail.size
    empirical_cdf = np.arange(1, n + 1) / n
    theoretical_cdf = 1.0 - (tail / xmin) ** (-(alpha - 1.0))
    return float(np.max(np.abs(empirical_cdf - theoretical_cdf)))


def fit_xmin(x: np.ndarray, min_tail: int = 20, max_candidates: int = 200) -> dict | None:
    """Grid-search xmin minimizing KS distance; returns best-fit dict or None."""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n < 5:
        return None
    effective_min_tail = max(5, min(min_tail, n // 2))

    uniq = np.unique(x)
    first_idx = np.searchsorted(x, uniq, side="left")
    n_tail_at = n - first_idx
    candidates = uniq[n_tail_at >= effective_min_tail]
    if candidates.size == 0:
        return None
    if candidates.size > max_candidates:
        idx = np.unique(np.linspace(0, candidates.size - 1, max_candidates).astype(int))
        candidates = candidates[idx]

    best = None
    for xmin in candidates:
        alpha = mle_alpha(x, xmin)
        if not np.isfinite(alpha) or alpha <= 1.0:
            continue
        d = ks_distance(x, xmin, alpha)
        if best is None or d < best["ks_d"]:
            best = {"xmin": float(xmin), "alpha": float(alpha), "ks_d": d, "n_tail": int(np.sum(x >= xmin))}
    return best


def gof_bootstrap_pvalue(
    x: np.ndarray, xmin: float, alpha: float, n_boot: int,
    min_tail: int, max_candidates: int, rng: np.random.Generator,
) -> float:
    """Semi-parametric bootstrap GoF test (Clauset et al. sec. 4.1). p > 0.1 -> power law plausible."""
    if n_boot <= 0:
        return float("nan")
    n = x.size
    below = x[x < xmin]
    p_tail = np.sum(x >= xmin) / n
    observed_d = ks_distance(x, xmin, alpha)

    count_ge = 0
    valid_reps = 0
    for _ in range(n_boot):
        n_synth_tail = rng.binomial(n, p_tail)
        n_synth_below = n - n_synth_tail
        u = rng.random(n_synth_tail)
        synth_tail = xmin * (1.0 - u) ** (-1.0 / (alpha - 1.0))
        synth_below = rng.choice(below, size=n_synth_below, replace=True) if below.size > 0 else np.array([])
        synth = np.concatenate([synth_below, synth_tail])

        fit = fit_xmin(synth, min_tail=min_tail, max_candidates=max_candidates)
        if fit is None:
            continue
        valid_reps += 1
        d_synth = ks_distance(synth, fit["xmin"], fit["alpha"])
        if d_synth >= observed_d:
            count_ge += 1
    return count_ge / valid_reps if valid_reps > 0 else float("nan")


def bootstrap_ci(
    x: np.ndarray, n_boot: int, min_tail: int, max_candidates: int,
    rng: np.random.Generator, level: float = 0.95,
) -> dict:
    """Nonparametric case-resampling CI for alpha and xmin."""
    n = x.size
    alphas, xmins = [], []
    for _ in range(n_boot):
        sample = rng.choice(x, size=n, replace=True)
        fit = fit_xmin(sample, min_tail=min_tail, max_candidates=max_candidates)
        if fit is None:
            continue
        alphas.append(fit["alpha"])
        xmins.append(fit["xmin"])

    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    if not alphas:
        return {"alpha_lo": float("nan"), "alpha_hi": float("nan"), "xmin_lo": float("nan"), "xmin_hi": float("nan")}
    return {
        "alpha_lo": float(np.quantile(alphas, lo_q)), "alpha_hi": float(np.quantile(alphas, hi_q)),
        "xmin_lo": float(np.quantile(xmins, lo_q)), "xmin_hi": float(np.quantile(xmins, hi_q)),
    }


def analyze_custom(
    x: np.ndarray, level: str, group: str,
    min_tail: int, max_candidates: int, gof_boot: int, ci_boot: int,
    rng: np.random.Generator,
) -> dict | None:
    fit = fit_xmin(x, min_tail=min_tail, max_candidates=max_candidates)
    if fit is None:
        print(f"  SKIP {level}/{group}: n={x.size}, no valid xmin fit")
        return None
    p_value = gof_bootstrap_pvalue(x, fit["xmin"], fit["alpha"], gof_boot, min_tail, max_candidates, rng)
    ci = bootstrap_ci(x, ci_boot, min_tail, max_candidates, rng)
    return {
        "Engine": "custom", "Level": level, "Group": group, "N": int(x.size),
        "xmin": round(fit["xmin"], 1), "N_tail": fit["n_tail"],
        "alpha": round(fit["alpha"], 4), "KS_D": round(fit["ks_d"], 4),
        "GoF_p_value": round(p_value, 3) if not np.isnan(p_value) else None,
        "LR_vs_lognormal": None, "LR_p_value": None,
        "Alpha_CI_low": round(ci["alpha_lo"], 4), "Alpha_CI_high": round(ci["alpha_hi"], 4),
        "Xmin_CI_low": round(ci["xmin_lo"], 1), "Xmin_CI_high": round(ci["xmin_hi"], 1),
    }


# ── Data loaders ───────────────────────────────────────────────────────────

def load_us(year: int) -> pl.DataFrame:
    raw = US_CSV.read_bytes().decode("latin-1").encode("utf-8")
    df = pl.read_csv(io.BytesIO(raw))
    pop_col = f"POPESTIMATE{year}"
    return (
        df.select(
            pl.col("SUMLEV"),
            pl.col("STNAME").alias("state"),
            pl.col("NAME").alias("name"),
            pl.col(pop_col).cast(pl.Float64).alias("population"),
        )
        .filter(pl.col("population") > 0)
    )


def _read_us_cbsa_csv() -> pl.DataFrame:
    raw = US_CBSA_CSV.read_bytes().decode("latin-1").encode("utf-8")
    return pl.read_csv(io.BytesIO(raw))


def load_us_metro(year: int) -> pl.DataFrame:
    df = _read_us_cbsa_csv()
    pop_col = f"POPESTIMATE{year}"
    return (
        df.filter(
            pl.col("STCOU").is_null() & pl.col("MDIV").is_null()
            & (pl.col("LSAD") == "Metropolitan Statistical Area")
        )
        .select(
            pl.col("CBSA").alias("cbsa"),
            pl.col("NAME").alias("name"),
            pl.col(pop_col).cast(pl.Float64).alias("population"),
        )
        .filter(pl.col("population") > 0)
    )


def load_us_place_cbsa(year: int) -> pl.DataFrame:
    """Maps each US incorporated place (SUMLEV=162) to its CBSA. Place-total rows
    carry no county FIPS (COUNTY is always "000"), so the join goes through
    SUMLEV=157 "place part in county" rows instead, which do. Places whose parts
    span 2+ CBSAs are assigned to whichever CBSA holds the largest population
    share (population-weighted argmax); places with no CBSA-mapped part at all
    are excluded (non-metro)."""
    raw = US_CSV.read_bytes().decode("latin-1").encode("utf-8")
    df = pl.read_csv(io.BytesIO(raw))
    pop_col = f"POPESTIMATE{year}"

    county_cbsa = (
        _read_us_cbsa_csv()
        .filter(pl.col("LSAD") == "County or equivalent")
        .select(pl.col("STCOU"), pl.col("CBSA"))
    )
    cbsa_names = load_us_metro(year).select(pl.col("cbsa").alias("CBSA"), pl.col("name").alias("cbsa_name"))

    place_cbsa = (
        df.filter(pl.col("SUMLEV") == 157)
        .with_columns((pl.col("STATE") * 1000 + pl.col("COUNTY")).alias("STCOU"))
        .join(county_cbsa, on="STCOU", how="inner")
        .join(cbsa_names, on="CBSA", how="inner")
        .group_by("STATE", "PLACE", "cbsa_name")
        .agg(pl.col(pop_col).cast(pl.Float64).sum().alias("part_pop"))
        .sort("part_pop", descending=True)
        .unique(subset=["STATE", "PLACE"], keep="first")
        .select("STATE", "PLACE", "cbsa_name")
    )
    places = df.filter(pl.col("SUMLEV") == 162).select(
        "STATE", "PLACE", pl.col(pop_col).cast(pl.Float64).alias("population"),
    )
    return places.join(place_cbsa, on=["STATE", "PLACE"], how="inner").filter(pl.col("population") > 0)


def load_mx(year: int) -> pl.DataFrame:
    df = pl.read_parquet(MX_PARQUET)
    return (
        df.filter(pl.col("AÑO") == year)
        .select(
            pl.col("NOM_ENT").alias("state"),
            pl.col("NOM_MUN").alias("name"),
            pl.col("POB_TOTAL").cast(pl.Float64).alias("population"),
        )
        .filter(pl.col("population") > 0)
    )


def _mx_municipios_with_metro(year: int) -> pl.DataFrame:
    """Municipio-level population joined to its Zona Metropolitana/Metropoli assignment
    (only the 421 municipios that belong to one, via the SEDATU/CONAPO/INEGI
    "Las Metropolis de Mexico 2020" municipio->metropoli crosswalk)."""
    pop = (
        pl.read_parquet(MX_PARQUET)
        .filter(pl.col("AÑO") == year)
        .select(pl.col("CLAVE"), pl.col("POB_TOTAL").cast(pl.Float64).alias("population"))
    )
    cross = pl.read_csv(MX_METRO_CSV).select(
        pl.col("clave_compuesta_municipio").alias("CLAVE"),
        pl.col("clave_metropoli"),
        pl.col("nombre"),
        pl.col("entidad"),
    )
    return pop.join(cross, on="CLAVE", how="inner").filter(pl.col("population") > 0)


def load_mx_metro(year: int) -> pl.DataFrame:
    """Aggregates CONAPO municipio populations up to Zona Metropolitana / Metropoli level."""
    return (
        _mx_municipios_with_metro(year)
        .group_by("clave_metropoli")
        .agg(pl.col("nombre").first(), pl.col("population").sum())
        .filter(pl.col("population") > 0)
    )


def mx_metro_principal_estado(year: int) -> pl.DataFrame:
    """Resolves each Zona Metropolitana/Metropoli to a single principal estado --
    trivial for the 84 single-estado zones; for the 8 zones spanning 2+ estados,
    picks whichever estado holds the largest total municipio population within
    that zone (a naive "most central-flagged municipios" rule gets the Ciudad de
    Mexico zone wrong: Mexico state has more central-flagged municipios than
    Ciudad de Mexico itself, despite the zone being population-centered on CDMX)."""
    entidad_pop = (
        _mx_municipios_with_metro(year)
        .group_by("clave_metropoli", "entidad")
        .agg(pl.col("population").sum().alias("entidad_pop"))
    )
    return (
        entidad_pop.sort("entidad_pop", descending=True)
        .unique(subset=["clave_metropoli"], keep="first")
        .select("clave_metropoli", pl.col("entidad").alias("principal_estado"))
    )


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Zipf/power-law rank-size analysis (Clauset-Shalizi-Newman MLE) "
                     "for US and Mexican population data.",
    )
    parser.add_argument("--level", nargs="+", choices=LEVELS, default=LEVELS)
    parser.add_argument("--year", type=int, choices=range(2020, 2026), default=2025)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--engine", choices=["powerlaw", "custom"], default="powerlaw",
                         help="powerlaw: delegate to the powerlaw library (fast). "
                              "custom: this script's own MLE/KS engine with bootstrap GoF p-value and CIs (slower).")
    parser.add_argument("--gof-boot", type=int, default=100, help="custom engine only")
    parser.add_argument("--ci-boot", type=int, default=1000, help="custom engine only")
    parser.add_argument("--min-tail", type=int, default=20, help="custom engine only")
    parser.add_argument("--xmin-candidates", type=int, default=200, help="custom engine only")
    parser.add_argument("--seed", type=int, default=42, help="custom engine only")
    parser.add_argument("--min-n", type=int, default=10, help="min entities to attempt a per-state/metro fit")
    parser.add_argument("--by-state", action="store_true", help="fit each state/estado separately instead of one pooled national fit (cities, municipios, us-metro, mx-metro)")
    parser.add_argument("--by-metro", action="store_true", help="fit each metro area separately instead of one pooled national fit (cities, municipios)")
    parser.add_argument("--list", action="store_true", help="mx-metro level only: also print all zones ranked by population alongside the pooled fit")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    levels = args.level
    rows = []

    def run(pop: np.ndarray, level: str, group: str):
        print(f"Fitting {level}/{group} (n={pop.size})...")
        if args.engine == "powerlaw":
            row = analyze_powerlaw(pop, level, group)
        else:
            row = analyze_custom(pop, level, group, args.min_tail, args.xmin_candidates, args.gof_boot, args.ci_boot, rng)
        if row:
            rows.append(row)

    def run_grouped(df: pl.DataFrame, group_col: str, level: str):
        for group in sorted(df[group_col].unique().to_list()):
            pop = df.filter(pl.col(group_col) == group)["population"].to_numpy()
            if pop.size < args.min_n:
                print(f"  SKIP {level}/{group}: n={pop.size} < min-n={args.min_n}")
                continue
            run(pop, level, group)

    if "cities" in levels:
        us_cities = load_us(args.year).filter(pl.col("SUMLEV") == 162)
        if args.by_state:
            run_grouped(us_cities, "state", "cities")
        elif args.by_metro:
            run_grouped(load_us_place_cbsa(args.year), "cbsa_name", "cities")
        else:
            run(us_cities["population"].to_numpy(), "cities", "ALL")

    if "municipios" in levels:
        if args.by_state:
            run_grouped(load_mx(args.year), "state", "municipios")
        elif args.by_metro:
            run_grouped(_mx_municipios_with_metro(args.year), "nombre", "municipios")
        else:
            mx = load_mx(args.year)
            run(mx["population"].to_numpy(), "municipios", "ALL")

    if "us-metro" in levels:
        metro = load_us_metro(args.year)
        if args.by_state:
            metro = metro.with_columns(pl.col("name").map_elements(primary_state, return_dtype=pl.String).alias("state"))
            run_grouped(metro, "state", "us-metro")
        else:
            run(metro["population"].to_numpy(), "us-metro", "ALL")

    if "mx-metro" in levels:
        mx_metro = load_mx_metro(args.year)
        if args.by_state:
            grouped = mx_metro.join(mx_metro_principal_estado(args.year), on="clave_metropoli")
            run_grouped(grouped, "principal_estado", "mx-metro")
        else:
            run(mx_metro["population"].to_numpy(), "mx-metro", "ALL")
        if args.list:
            ranked = mx_metro.sort("population", descending=True).with_row_index("rank", offset=1)
            with pl.Config(tbl_rows=-1):
                print(ranked.select("rank", "nombre", "population"))

    result = pl.DataFrame(rows)
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(result)

    if args.output:
        result.write_csv(args.output)
        print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
