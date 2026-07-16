"""
Prepara las tablas derivadas para el dashboard de PIB Estatal (INEGI PIBE).

Lee PIBE_2 (total), PIBE_14 (minería petrolera), PIBE_48 (composición
sectorial nacional) y los tabulados por entidad de los estados con
exposición petrolera (FOSSIL_STATES) de data/inegi/pibe/tabulados/,
deriva las tablas usadas por dashboard/pibe.py y las guarda como
parquet en dashboard_data/.
"""

import numpy as np
import polars as pl

PIBE_DIR = "data/inegi/pibe/tabulados"
OUT_DIR = "dashboard_data"

YEAR_COLS = [str(y) for y in range(2003, 2023)] + ["2023R", "2024R"]

NAME_MAP = {
    "Coahuila de Zaragoza": "Coahuila",
    "Michoacán de Ocampo": "Michoacán",
    "Veracruz de Ignacio de la Llave": "Veracruz",
}

# Estados con exposición petrolera relevante (oil_share_pct > 2% en algún año) — ver
# dashboard_data/pibe_oil_share_long.parquet. Tabulados por entidad (familia 2, PIBE_49-80).
FOSSIL_STATES = {
    "Campeche": "PIBE_52.xlsx",
    "Chiapas": "PIBE_55.xlsx",
    "Tabasco": "PIBE_75.xlsx",
    "Tamaulipas": "PIBE_76.xlsx",
    "Veracruz de Ignacio de la Llave": "PIBE_78.xlsx",
}


def _read_tabulado(filename: str) -> pl.DataFrame:
    return pl.read_excel(
        f"{PIBE_DIR}/{filename}", sheet_name="Tabulado",
        read_options={"header_row": 4},
    )


def _family1_block(filename: str, block: int) -> pl.DataFrame:
    """Familia 1: 35 filas/bloque (título + código SCN + 33 entidades)."""
    df = _read_tabulado(filename)
    return df.slice(block * 35 + 2, 33).with_columns(
        pl.col("Concepto").str.strip_chars().alias("entidad")
    )


def _family2_block(filename: str, block: int) -> pl.DataFrame:
    """Familia 2: 46 filas/bloque (título + 45 actividades)."""
    df = _read_tabulado(filename)
    return df.slice(block * 46 + 1, 45).with_columns(
        pl.col("Concepto").str.strip_chars().alias("actividad")
    )


def _to_long(d: pl.DataFrame, key_col: str, value_name: str) -> pl.DataFrame:
    return (
        d.unpivot(index=key_col, on=YEAR_COLS, variable_name="year_raw", value_name=value_name)
        .with_columns(
            pl.col("year_raw").str.strip_chars("R").cast(pl.Int32).alias("year"),
            pl.col("year_raw").str.ends_with("R").alias("revisado"),
        )
        .drop("year_raw")
        .filter(pl.col(value_name).is_not_null())
    )


def _add_nom_geo(d: pl.DataFrame, col: str = "entidad") -> pl.DataFrame:
    return d.with_columns(
        pl.col(col)
        .replace_strict(list(NAME_MAP.keys()), list(NAME_MAP.values()), default=None)
        .fill_null(pl.col(col))
        .alias("NOM_GEO")
    )


def _gini(values: np.ndarray) -> float:
    x = np.sort(values)
    n = len(x)
    cum = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


def main() -> None:
    gdp_levels = _family1_block("PIBE_2.xlsx", block=0)
    gdp_pct = _family1_block("PIBE_2.xlsx", block=1)
    oil_levels = _family1_block("PIBE_14.xlsx", block=0)
    sector_pct = _family2_block("PIBE_48.xlsx", block=2)

    # Ancla de validación (ver DATA_OVERVIEW.md): 2003, nacional = 17,899,317.915
    anchor = gdp_levels.filter(pl.col("entidad") == "Estados Unidos Mexicanos")["2003"][0]
    assert abs(anchor - 17899317.915) < 0.001, f"anchor mismatch: {anchor}"

    gdp_long = _add_nom_geo(_to_long(gdp_levels, "entidad", "pib_mdp"))
    gdp_pct_long = _to_long(gdp_pct, "entidad", "var_pct")
    oil_long = _to_long(oil_levels, "entidad", "oil_mdp")
    sector_long = _to_long(sector_pct, "actividad", "share_pct")

    states_only = gdp_long.filter(pl.col("entidad") != "Estados Unidos Mexicanos")
    national = gdp_long.filter(pl.col("entidad") == "Estados Unidos Mexicanos").select(
        "year", pl.col("pib_mdp").alias("pib_nacional")
    )

    # ── a) Gini + concentración por año ─────────────────────────────────────
    gini_rows = []
    for y in sorted(states_only["year"].unique().to_list()):
        vals = states_only.filter(pl.col("year") == y).sort("pib_mdp", descending=True)["pib_mdp"].to_numpy()
        gini_rows.append({
            "year": y,
            "gini": _gini(vals),
            "top5_share": float(vals[:5].sum() / vals.sum() * 100),
            "top10_share": float(vals[:10].sum() / vals.sum() * 100),
        })
    gini_by_year = pl.DataFrame(gini_rows)

    r2022 = gini_by_year.filter(pl.col("year") == 2022).to_dicts()[0]
    assert abs(r2022["gini"] - 0.4405) < 0.001, f"gini 2022 mismatch: {r2022['gini']}"
    assert abs(r2022["top5_share"] - 44.11) < 0.05, f"top5 2022 mismatch: {r2022['top5_share']}"

    # ── b) participación estatal en el PIB nacional por año ────────────────
    share_by_state_year = states_only.join(national, on="year").with_columns(
        (pl.col("pib_mdp") / pl.col("pib_nacional") * 100).alias("share_pct")
    )

    # ── c) CAGR por estado, 2003→2022 ───────────────────────────────────────
    cagr_by_state = gdp_levels.select("entidad", "2003", "2022").with_columns(
        ((pl.col("2022") / pl.col("2003")) ** (1 / 19) - 1).alias("cagr")
    ).with_columns((pl.col("cagr") * 100).alias("cagr_pct"))

    qroo = cagr_by_state.filter(pl.col("entidad") == "Quintana Roo")["cagr_pct"][0]
    campeche = cagr_by_state.filter(pl.col("entidad") == "Campeche")["cagr_pct"][0]
    assert abs(qroo - 3.19) < 0.05, f"Quintana Roo CAGR mismatch: {qroo}"
    assert abs(campeche - (-3.80)) < 0.05, f"Campeche CAGR mismatch: {campeche}"

    # ── d) participación del PIB petrolero en el PIB estatal ───────────────
    oil_share_long = oil_long.join(
        gdp_long.select("entidad", "year", "pib_mdp"), on=["entidad", "year"]
    ).with_columns((pl.col("oil_mdp") / pl.col("pib_mdp") * 100).alias("oil_share_pct"))

    camp_2003 = oil_share_long.filter((pl.col("entidad") == "Campeche") & (pl.col("year") == 2003))["oil_share_pct"][0]
    assert abs(camp_2003 - 87.3) < 0.5, f"Campeche oil-share 2003 mismatch: {camp_2003}"

    # ── e) impacto COVID 2020/2021 ──────────────────────────────────────────
    covid = gdp_pct_long.filter(pl.col("year").is_in([2020, 2021]))

    nat_2020 = covid.filter((pl.col("entidad") == "Estados Unidos Mexicanos") & (pl.col("year") == 2020))["var_pct"][0]
    assert abs(nat_2020 - (-8.354)) < 0.01, f"national COVID 2020 mismatch: {nat_2020}"

    # ── f) composición sectorial nacional ───────────────────────────────────
    sector_labels = [
        "Actividades primarias", "Actividades secundarias", "Actividades terciarias",
        "21 - Minería", "31-33 - Industrias manufactureras",
    ]
    sector_composition = sector_long.filter(pl.col("actividad").is_in(sector_labels))

    terc_2003 = sector_composition.filter(
        (pl.col("actividad") == "Actividades terciarias") & (pl.col("year") == 2003)
    )["share_pct"][0]
    assert abs(terc_2003 - 54.85) < 0.05, f"tertiary 2003 mismatch: {terc_2003}"

    # ── g) composición sectorial de los estados con exposición petrolera ────
    fossil_frames = [
        _family2_block(filename, block=2).with_columns(pl.lit(entidad).alias("entidad"))
        for entidad, filename in FOSSIL_STATES.items()
    ]
    fossil_sector_pct = pl.concat(fossil_frames)

    fossil_labels = [
        "Actividades primarias", "Actividades secundarias", "Actividades terciarias",
        "21-1 - Minería petrolera",
    ]
    fossil_long = _to_long(
        fossil_sector_pct.filter(pl.col("actividad").is_in(fossil_labels))
        .select("entidad", "actividad", *YEAR_COLS),
        ["entidad", "actividad"], "share_pct",
    )
    fossil_wide = fossil_long.pivot(on="actividad", index=["entidad", "year"], values="share_pct").with_columns(
        (pl.col("Actividades secundarias") - pl.col("21-1 - Minería petrolera")).alias("Secundario sin petróleo")
    )
    fossil_states_sector = fossil_wide.select(
        "entidad", "year",
        pl.col("Actividades primarias").alias("Primario"),
        pl.col("Secundario sin petróleo"),
        pl.col("21-1 - Minería petrolera").alias("Petróleo"),
        pl.col("Actividades terciarias").alias("Terciario"),
    ).unpivot(
        index=["entidad", "year"], on=["Primario", "Secundario sin petróleo", "Petróleo", "Terciario"],
        variable_name="categoria", value_name="share_pct",
    )

    camp_oil_2003 = fossil_states_sector.filter(
        (pl.col("entidad") == "Campeche") & (pl.col("year") == 2003) & (pl.col("categoria") == "Petróleo")
    )["share_pct"][0]
    assert abs(camp_oil_2003 - 87.33) < 0.05, f"fossil Campeche 2003 mismatch: {camp_oil_2003}"

    # ── guardar ──────────────────────────────────────────────────────────────
    outputs = {
        "pibe_gdp_long": gdp_long,
        "pibe_gini_by_year": gini_by_year,
        "pibe_share_by_state_year": share_by_state_year,
        "pibe_cagr_by_state": cagr_by_state,
        "pibe_oil_share_long": oil_share_long,
        "pibe_covid": covid,
        "pibe_sector_composition": sector_composition,
        "pibe_fossil_states_sector": fossil_states_sector,
    }
    for name, frame in outputs.items():
        path = f"{OUT_DIR}/{name}.parquet"
        frame.write_parquet(path)
        print(f"{path}: {frame.height} rows, {len(frame.columns)} cols")

    print("Todos los anclas de validación coinciden con DATA_OVERVIEW.md.")


if __name__ == "__main__":
    main()
