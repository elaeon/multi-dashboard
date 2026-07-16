"""
Prepara las tablas derivadas para el dashboard de Cuenta Pública (SHCP).

Lee los 15 años de Cuenta Pública (2011-2025), los 9 años de Anexos
Transversales (2017-2025) y los 4 años de Subsidios por Población
(2022-2025) de data/presupuesto_federacion/cuenta_publica/, normaliza
los esquemas cambiantes (ver DATA_OVERVIEW.md), une Subsidios con la
población CONAPO para calcular montos per cápita, y guarda las tablas
agregadas usadas por dashboard/cuenta_publica.py como parquet en
dashboard_data/.
"""

import io
import zipfile

import polars as pl

CP_DIR = "data/presupuesto_federacion/cuenta_publica/cuenta_publica"
AT_DIR = "data/presupuesto_federacion/cuenta_publica/anexos_transversales"
SB_DIR = "data/presupuesto_federacion/cuenta_publica/subsidios_poblacion"
CONAPO_POB = "data/conapo/proyecciones_poblacion/pob_estado_año.parquet"
OUT_DIR = "dashboard_data"

CP_FILES = {
    2011: "Cuenta_Publica_2011.xlsx",
    2012: "Cuenta_Publica_2012.xlsx",
    2013: "cuenta_publica_2013_ra_ecd.xlsx",
    2014: "cuenta_publica_2014_ra_ecd.xlsx",
    2015: "cuenta_publica_2015_ra_ecd_epe.xlsx",
    2016: "cuenta_publica_2016_gf_ecd_epe.xlsx",
    2017: "cuenta_publica_2017_gf_ecd_epe.xlsx",
    2018: "cuenta_publica_2018_gf_ecd_epe.xlsx",
    2019: "cuenta_publica_2019_gf_ecd_epe.xlsx",
    2020: "cuenta_publica_2020_gf_ecd_epe.xlsx",
    2021: "cuenta_publica_2021_gf_ecd_epe.xlsx",
    2022: "cuenta_publica_2022_gf_ecd_epe.xlsx",
    2023: "cuenta_publica_2023_gf_ecd_epe.xlsx",
    2024: "cuenta_publica_2024_gf_ecd_epe.xlsx",
    2025: "cuenta_publica_2025_gf_ecd_epe 1.xlsx",
}

AT_FILES = {
    2017: "BD_Transversales_CP 2017.xlsx",
    2018: "BD_Transversales_CP_2018.xlsx",
    2019: "BD_Transversales_CP_2019.xlsx",
    2020: "BD_Transversales_CP_2020.xlsx",
    2021: "BD_Transversales_CP_2021.xlsx",
    2022: "BD_Transversales_CP_2022.xlsx",
    2023: "BD_Transversales_CP_2023.xlsx",
    2024: "BD_Transversales_CP_2024.xlsx",
    2025: "BD_Transversales_CP_2025.xlsx",
}

SB_FILES = {
    2022: "Subsidios_CP2022.zip",
    2023: "Subsidios_CP2023.zip",
    2024: "Subsidios_CP2024.zip",
    2025: "Subsidios_CP2025.zip",
}

EJER_NAMES = ["MONTO_EJERCIDO", "MONTO_EJERCICIO", "EJERCICIO", "Ejercicio"]

# 2025 drops the ID_/DESC_ prefixing convention — restore it before selecting.
RENAME_2025 = {
    "R": "ID_RAMO", "TIPOGASTO": "ID_TIPOGASTO",
    "ENTIDAD_FEDERATIVA": "ID_ENTIDAD_FEDERATIVA",
    "DESC_ENTIDAD_FEDERATIVA": "ENTIDAD_FEDERATIVA",
}

# Subsidios' long INEGI names -> short accented names matching data/mexico_states.geojson
# (same crosswalk used in scripts/prepare_pibe.py).
NOM_GEO_MAP = {
    "Coahuila de Zaragoza": "Coahuila",
    "Michoacán de Ocampo": "Michoacán",
    "Veracruz de Ignacio de la Llave": "Veracruz",
}

# NOM_GEO (short accented) -> CONAPO's no-accent title-case "estado" names.
CONAPO_NAME_MAP = {
    "Ciudad de México": "Ciudad De Mexico",
    "México": "Mexico",
    "Michoacán": "Michoacan",
    "Nuevo León": "Nuevo Leon",
    "Querétaro": "Queretaro",
    "San Luis Potosí": "San Luis Potosi",
    "Yucatán": "Yucatan",
}

SINK_NO_DISTRIBUIBLE = "No distribuibles por entidad federativa"


CP_BASE_COLS = ["ID_RAMO", "DESC_RAMO", "ID_TIPOGASTO", "DESC_TIPOGASTO", "MONTO_APROBADO"]
CP_GEO_COLS = ["ID_ENTIDAD_FEDERATIVA", "ENTIDAD_FEDERATIVA"]
CP_2025_ALT = {"ID_RAMO": "R", "ID_TIPOGASTO": "TIPOGASTO", "ID_ENTIDAD_FEDERATIVA": "ENTIDAD_FEDERATIVA",
               "ENTIDAD_FEDERATIVA": "DESC_ENTIDAD_FEDERATIVA"}


def _load_cp_year(year: int) -> pl.DataFrame:
    path = f"{CP_DIR}/{CP_FILES[year]}"
    header = pl.read_excel(path, read_options={"n_rows": 0}).columns
    wanted = list(CP_BASE_COLS) + [c for c in CP_GEO_COLS if c in header or CP_2025_ALT.get(c) in header]
    wanted += [c for c in EJER_NAMES if c in header]
    if year == 2025:
        wanted = [CP_2025_ALT.get(c, c) for c in wanted]
    cols = [c for c in wanted if c in header]
    df = pl.read_excel(path, columns=cols)
    if year == 2025:
        df = df.rename({k: v for k, v in RENAME_2025.items() if k in df.columns})
    ecol = next(c for c in EJER_NAMES if c in df.columns)
    df = df.rename({ecol: "monto_ejercido"})

    has_geo = "ID_ENTIDAD_FEDERATIVA" in df.columns
    return df.select(
        pl.lit(year).alias("year"),
        pl.col("ID_RAMO").cast(pl.Int64, strict=False).alias("id_ramo"),
        pl.col("DESC_RAMO").str.strip_chars().alias("desc_ramo"),
        pl.col("ID_TIPOGASTO").cast(pl.Int64, strict=False).alias("id_tipogasto"),
        pl.col("DESC_TIPOGASTO").str.strip_chars().alias("desc_tipogasto"),
        (pl.col("ID_ENTIDAD_FEDERATIVA").cast(pl.Int64, strict=False) if has_geo
         else pl.lit(None, dtype=pl.Int64)).alias("id_entidad"),
        (pl.col("ENTIDAD_FEDERATIVA").str.strip_chars() if has_geo
         else pl.lit(None, dtype=pl.Utf8)).alias("entidad"),
        pl.col("MONTO_APROBADO").cast(pl.Float64, strict=False).alias("monto_aprobado"),
        pl.col("monto_ejercido").cast(pl.Float64, strict=False),
    )


# 2025 renames ID_TRANSVERSAL->CLAVE_TRANSVERSAL and drops MONTO_PAGADO entirely,
# replacing it with MONTO_EJERCIDO (closest available proxy for money actually spent).
AT_2025_RENAME = {"CLAVE_TRANSVERSAL": "ID_TRANSVERSAL", "MONTO_EJERCIDO": "MONTO_PAGADO"}


def _load_at_year(year: int) -> pl.DataFrame:
    path = f"{AT_DIR}/{AT_FILES[year]}"
    header = pl.read_excel(path, read_options={"n_rows": 0}).columns
    cols = ["ID_TRANSVERSAL", "TRANSVERSAL", "MONTO_APROBADO", "MONTO_PAGADO"]
    if year == 2025:
        inv = {v: k for k, v in AT_2025_RENAME.items()}
        cols = [inv.get(c, c) for c in cols]
    df = pl.read_excel(path, columns=[c for c in cols if c in header])
    if year == 2025:
        df = df.rename({k: v for k, v in AT_2025_RENAME.items() if k in df.columns})
    return df.select(
        pl.lit(year).alias("year"),
        pl.col("ID_TRANSVERSAL").cast(pl.Int64, strict=False).alias("id_transversal"),
        pl.col("TRANSVERSAL").str.strip_chars().alias("transversal"),
        pl.col("MONTO_APROBADO").cast(pl.Float64, strict=False).alias("aprobado"),
        pl.col("MONTO_PAGADO").cast(pl.Float64, strict=False).alias("pagado"),
    )


def _load_sb_year(year: int) -> pl.DataFrame:
    zpath = f"{SB_DIR}/{SB_FILES[year]}"
    inner = "Subsidios_CP2022/subsidios.csv" if year == 2022 else f"Subsidios_CP_{year}.csv"
    amt_col = "MONTO_PAGADO" if year == 2022 else "MONTO_EJERCIDO"
    with zipfile.ZipFile(zpath) as z, z.open(inner) as fh:
        raw = fh.read().decode("cp1252").encode("utf-8")
    df = pl.read_csv(io.BytesIO(raw), infer_schema_length=0).with_columns(
        pl.col("ENTIDAD_FEDERATIVA").str.strip_chars()
    )
    return (
        df.filter(pl.col("CICLO") == str(year))
        .filter(pl.col("ENTIDAD_FEDERATIVA") != SINK_NO_DISTRIBUIBLE)
        .select(
            pl.lit(year).alias("year"),
            pl.col("ENTIDAD_FEDERATIVA").alias("entidad"),
            pl.col(amt_col).cast(pl.Float64, strict=False).alias("monto"),
        )
    )


def main() -> None:
    cp = pl.concat([_load_cp_year(y) for y in CP_FILES])
    at = pl.concat([_load_at_year(y) for y in AT_FILES])
    sb = pl.concat([_load_sb_year(y) for y in SB_FILES])

    # ── a) totales y tasa de ejercicio por año ──────────────────────────────
    cp_totals_by_year = (
        cp.group_by("year")
        .agg(pl.col("monto_aprobado").sum().alias("aprobado"), pl.col("monto_ejercido").sum().alias("ejercido"))
        .with_columns((pl.col("ejercido") / pl.col("aprobado") * 100).alias("execution_rate"))
        .sort("year")
    )

    r2024 = cp_totals_by_year.filter(pl.col("year") == 2024).to_dicts()[0]
    assert abs(r2024["ejercido"] - 10.55e12) / 10.55e12 < 0.03, f"2024 ejercido mismatch: {r2024['ejercido']:.3e}"
    mean_rate = float(cp_totals_by_year.filter(pl.col("year") <= 2024)["execution_rate"].mean())
    assert abs(mean_rate - 104) < 3, f"mean execution_rate mismatch: {mean_rate:.1f}"
    r2016 = cp_totals_by_year.filter(pl.col("year") == 2016)["execution_rate"][0]
    assert abs(r2016 - 112.1) < 2, f"2016 execution_rate mismatch: {r2016:.1f}"

    # ── b) concentración por RAMO ────────────────────────────────────────────
    cp_ramo_year = (
        cp.group_by(["year", "id_ramo", "desc_ramo"])
        .agg(pl.col("monto_ejercido").sum().alias("ejercido"))
        .sort(["year", "ejercido"], descending=[False, True])
    )

    r2024_ramo = cp_ramo_year.filter(pl.col("year") == 2024).sort("ejercido", descending=True)
    total_2024 = float(r2024_ramo["ejercido"].sum())
    top10_share = float(r2024_ramo.head(10)["ejercido"].sum() / total_2024 * 100)
    assert abs(top10_share - 81.7) < 2, f"top10 ramo share mismatch: {top10_share:.1f}"

    # ── c) composición por tipo de gasto ─────────────────────────────────────
    cp_tipogasto_year = (
        cp.group_by(["year", "id_tipogasto", "desc_tipogasto"])
        .agg(pl.col("monto_ejercido").sum().alias("ejercido"))
        .sort(["year", "ejercido"], descending=[False, True])
    )

    r2024_tipo = cp_tipogasto_year.filter(pl.col("year") == 2024)
    total_tipo_2024 = float(r2024_tipo["ejercido"].sum())
    pensiones_pct = float(r2024_tipo.filter(pl.col("id_tipogasto") == 4)["ejercido"][0] / total_tipo_2024 * 100)
    corriente_pct = float(r2024_tipo.filter(pl.col("id_tipogasto") == 1)["ejercido"][0] / total_tipo_2024 * 100)
    assert abs(pensiones_pct - 23.2) < 2, f"pensiones share mismatch: {pensiones_pct:.1f}"
    assert abs(corriente_pct - 52.1) < 2, f"gasto corriente share mismatch: {corriente_pct:.1f}"

    # ── d) geografía (registro administrativo, ver Insight #6) ──────────────
    cp_geo_year = (
        cp.filter(pl.col("id_entidad").is_not_null())
        .group_by(["year", "id_entidad", "entidad"])
        .agg(pl.col("monto_ejercido").sum().alias("ejercido"))
        .sort(["year", "ejercido"], descending=[False, True])
    )

    r2024_geo = cp_geo_year.filter(pl.col("year") == 2024)
    total_geo_2024 = float(r2024_geo["ejercido"].sum())
    cdmx_pct = float(r2024_geo.filter(pl.col("id_entidad") == 9)["ejercido"][0] / total_geo_2024 * 100)
    nodist_pct = float(r2024_geo.filter(pl.col("id_entidad") == 34)["ejercido"][0] / total_geo_2024 * 100)
    assert abs(cdmx_pct - 39.6) < 2, f"CDMX share mismatch: {cdmx_pct:.1f}"
    assert abs(nodist_pct - 10.9) < 2, f"No Distribuible share mismatch: {nodist_pct:.1f}"

    # ── e) transferencias federalizadas por estado (RAMO 28 Participaciones +
    #      RAMO 33 Aportaciones) — tab Tabasco/Campeche: producción vs. retorno ──
    cp_transfers_state_year = (
        cp.filter(pl.col("id_ramo").is_in([28, 33]) & pl.col("id_entidad").is_between(1, 32))
        .group_by(["year", "id_entidad", "entidad", "id_ramo", "desc_ramo"])
        .agg(pl.col("monto_ejercido").sum().alias("ejercido"))
        .sort(["year", "id_entidad", "id_ramo"])
    )

    # Ancla cruzada contra informe_data/cp_estado_ramo.parquet (ya validado por
    # scripts/centralismo/cap1_riqueza_vs_retorno.py): Campeche(4)/Tabasco(27) 2024.
    def _t2024(id_entidad: int, id_ramo: int) -> float:
        return float(cp_transfers_state_year.filter(
            (pl.col("year") == 2024) & (pl.col("id_entidad") == id_entidad) & (pl.col("id_ramo") == id_ramo)
        )["ejercido"][0])

    for id_entidad, id_ramo, esperado in [(4, 28, 1.0309e10), (4, 33, 1.0706e10),
                                           (27, 28, 3.2208e10), (27, 33, 1.9971e10)]:
        obtenido = _t2024(id_entidad, id_ramo)
        assert abs(obtenido - esperado) / esperado < 0.02, (
            f"transferencia entidad={id_entidad} ramo={id_ramo} 2024 mismatch: {obtenido:.3e} vs {esperado:.3e}"
        )

    # ── f) anexos transversales ───────────────────────────────────────────────
    at_transversal_year = (
        at.group_by(["year", "id_transversal", "transversal"])
        .agg(pl.col("aprobado").sum(), pl.col("pagado").sum())
        .with_columns((pl.col("pagado") / pl.col("aprobado") * 100).alias("execution_ratio"))
        .sort(["year", "pagado"], descending=[False, True])
    )

    r2024_at = at_transversal_year.filter(pl.col("year") == 2024)
    total_pagado_2024 = float(r2024_at["pagado"].sum())
    children_pct = float(r2024_at.filter(pl.col("id_transversal") == 8)["pagado"][0] / total_pagado_2024 * 100)
    assert abs(children_pct - 25.0) < 2, f"children annex share mismatch: {children_pct:.1f}"

    # ── g) subsidios por entidad, con NOM_GEO para el mapa ──────────────────
    subsidios_state_year = (
        sb.group_by(["year", "entidad"])
        .agg(pl.col("monto").sum())
        .with_columns(
            pl.col("entidad").replace_strict(list(NOM_GEO_MAP.keys()), list(NOM_GEO_MAP.values()), default=None)
            .fill_null(pl.col("entidad")).alias("nom_geo")
        )
        .sort(["year", "monto"], descending=[False, True])
    )

    r2024_sb = subsidios_state_year.filter(pl.col("year") == 2024)
    total_sb_2024 = float(r2024_sb["monto"].sum())
    assert abs(total_sb_2024 - 0.76e12) / 0.76e12 < 0.05, f"2024 subsidios total mismatch: {total_sb_2024:.3e}"
    edomex_pct = float(r2024_sb.filter(pl.col("nom_geo") == "México")["monto"][0] / total_sb_2024 * 100)
    assert abs(edomex_pct - 10.4) < 2, f"Estado de México subsidios share mismatch: {edomex_pct:.1f}"

    # ── h) subsidios per cápita (join con población CONAPO) ──────────────────
    pob = pl.read_parquet(CONAPO_POB).select(
        pl.col("estado"), pl.col("año").alias("year"), pl.col("pob_total")
    )
    subsidios_percapita_year = (
        subsidios_state_year
        .with_columns(
            pl.col("nom_geo").replace_strict(
                list(CONAPO_NAME_MAP.keys()), list(CONAPO_NAME_MAP.values()), default=None
            ).fill_null(pl.col("nom_geo")).alias("entidad_conapo")
        )
        .join(pob, left_on=["entidad_conapo", "year"], right_on=["estado", "year"], how="left")
    )
    n_unmatched = subsidios_percapita_year.filter(pl.col("pob_total").is_null()).height
    assert n_unmatched == 0, f"{n_unmatched} filas sin coincidencia de población CONAPO"
    subsidios_percapita_year = subsidios_percapita_year.with_columns(
        (pl.col("monto") / pl.col("pob_total")).alias("monto_percapita")
    )

    # ── guardar ──────────────────────────────────────────────────────────────
    outputs = {
        "cp_totals_by_year": cp_totals_by_year,
        "cp_ramo_year": cp_ramo_year,
        "cp_tipogasto_year": cp_tipogasto_year,
        "cp_geo_year": cp_geo_year,
        "cp_transfers_state_year": cp_transfers_state_year,
        "at_transversal_year": at_transversal_year,
        "subsidios_state_year": subsidios_state_year,
        "subsidios_percapita_year": subsidios_percapita_year,
    }
    for name, frame in outputs.items():
        path = f"{OUT_DIR}/{name}.parquet"
        frame.write_parquet(path)
        print(f"{path}: {frame.height} rows, {len(frame.columns)} cols")

    print("Todos los anclas de validación coinciden con DATA_OVERVIEW.md.")


if __name__ == "__main__":
    main()
