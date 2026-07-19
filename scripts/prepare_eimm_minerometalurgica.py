"""
Prepara las tablas derivadas para el dashboard de Producción Minera (INEGI EIMM).

Lee conjunto_de_datos_eimm_municipio_csv.zip directamente (sin descomprimir a
disco), aplica la limpieza documentada en
data/inegi/industria_minerometalurgica/DATA_OVERVIEW.md (TAB/CRLF en catálogos,
punto final en ESTATUS, columnas constantes, año 2026 parcial/preliminar) y
guarda las tablas usadas por dashboard/produccion_minera.py como parquet en
dashboard_data/.
"""

import zipfile

import polars as pl

ZIP_PATH = "data/inegi/industria_minerometalurgica/conjunto_de_datos_eimm_municipio_csv.zip"
OUT_DIR = "dashboard_data"

# Mismo mapeo usado en scripts/prepare_pibe.py, para que los nombres de estado
# coincidan con data/mexico_states.geojson si se necesita en el futuro.
NAME_MAP = {
    "Coahuila de Zaragoza": "Coahuila",
    "Michoacán de Ocampo": "Michoacán",
    "Veracruz de Ignacio de la Llave": "Veracruz",
}


def _read_member(zf: zipfile.ZipFile, path: str, **kwargs) -> pl.DataFrame:
    with zf.open(path) as f:
        return pl.read_csv(f, encoding="utf8", **kwargs)


def load_catalogs(zf: zipfile.ZipFile) -> tuple[pl.DataFrame, pl.DataFrame]:
    entidad = _read_member(zf, "catalogos/tc_entidad.csv")
    entidad = entidad.with_columns([pl.col(c).str.strip_chars() for c in entidad.columns])
    municipio = _read_member(zf, "catalogos/tc_municipio.csv")
    municipio = municipio.with_columns([pl.col(c).str.strip_chars() for c in municipio.columns])
    return entidad, municipio


def load_annual(zf: zipfile.ZipFile, year: int) -> pl.DataFrame:
    path = f"conjunto_de_datos/eimm_municipio_mensual_tr_cifra_{year}.csv"
    df = _read_member(
        zf, path,
        schema_overrides={"MES": pl.Utf8, "ID_ENTIDAD": pl.Utf8, "ID_MUNICIPIO": pl.Utf8},
    )
    return df.drop(["PROD_EST", "COBERTURA"]).with_columns(
        pl.col("ESTATUS").str.strip_chars().str.strip_chars_end(".").alias("ESTATUS")
    )


def main() -> None:
    with zipfile.ZipFile(ZIP_PATH) as zf:
        entidad, municipio = load_catalogs(zf)
        # 2026 excluded: partial year (Jan-Apr only) with preliminary figures.
        frames = [load_annual(zf, year) for year in range(2001, 2026)]
        df = pl.concat(frames, how="vertical")

    entidad = entidad.with_columns(
        pl.col("NOMBRE_ENTIDAD")
        .replace_strict(list(NAME_MAP.keys()), list(NAME_MAP.values()), default=None)
        .fill_null(pl.col("NOMBRE_ENTIDAD"))
        .alias("estado")
    )
    df = (
        df.join(entidad.select("ID_ENTIDAD", "estado"), on="ID_ENTIDAD", how="left")
        .join(
            municipio.select("ID_ENTIDAD", "ID_MUNICIPIO", "NOMBRE_MUNICIPIO"),
            on=["ID_ENTIDAD", "ID_MUNICIPIO"], how="left",
        )
        .with_columns((pl.col("ID_ENTIDAD") + pl.col("ID_MUNICIPIO")).alias("cvegeo_mun"))
    )

    # ── a) volumen nacional anual por producto ──────────────────────────────
    nacional_anual = (
        df.group_by(["ANIO", "GRUPO_PRODUCTO", "PRODUCTO", "UNIDAD_MEDIDA"])
        .agg(pl.col("VOLUMEN").sum().alias("volumen"))
        .rename({
            "ANIO": "año", "GRUPO_PRODUCTO": "grupo_producto",
            "PRODUCTO": "producto", "UNIDAD_MEDIDA": "unidad_medida",
        })
        .sort("año")
    )

    # ── b) volumen anual por estado x producto (permite filtrar por rango de años
    # en el dashboard igual que el resto de las tablas; los hallazgos de
    # concentración usan el rango completo 2001-2025) ──────────────────────────
    estado_producto = (
        df.group_by(["ANIO", "estado", "PRODUCTO"])
        .agg(pl.col("VOLUMEN").sum().alias("volumen"))
        .rename({"ANIO": "año", "PRODUCTO": "producto"})
        .sort("año")
    )

    # ── c) municipios activos por año ────────────────────────────────────────
    municipios_activos = (
        df.group_by("ANIO").agg(pl.col("cvegeo_mun").n_unique().alias("n_municipios"))
        .rename({"ANIO": "año"}).sort("año")
    )

    # ── d) top-3 municipio por producto, 2025 ────────────────────────────────
    muni_vol_2025 = (
        df.filter(pl.col("ANIO") == 2025)
        .group_by(["PRODUCTO", "estado", "NOMBRE_MUNICIPIO"])
        .agg(pl.col("VOLUMEN").sum().alias("volumen"))
        .rename({"PRODUCTO": "producto", "NOMBRE_MUNICIPIO": "municipio"})
    )
    top_municipio_2025 = (
        muni_vol_2025
        .with_columns(
            pl.col("volumen").rank(method="ordinal", descending=True).over("producto").alias("rank")
        )
        .filter(pl.col("rank") <= 3)
        .sort(["producto", "rank"])
    )

    # ── anclas de validación (verificadas contra el ZIP fuente) ─────────────
    def _nat(prod: str, year: int) -> float:
        return float(nacional_anual.filter((pl.col("producto") == prod) & (pl.col("año") == year))["volumen"][0])

    assert abs(_nat("Oro", 2001) - 23542.8) < 1, "Oro 2001 mismatch"
    assert abs(_nat("Oro", 2022) - 148502.3) < 1, "Oro 2022 peak mismatch"
    assert abs(_nat("Oro", 2025) - 80389.9) < 1, "Oro 2025 mismatch"
    assert abs(_nat("Plata", 2001) - 2759985.0) < 1, "Plata 2001 mismatch"
    assert abs(_nat("Plata", 2022) - 6630389.0) < 1, "Plata 2022 peak mismatch"
    assert abs(_nat("Plata", 2025) - 5231912.0) < 1, "Plata 2025 mismatch"
    assert abs(_nat("Fierro en Extraccion", 2013) - 18839572.0) < 1, "Fierro 2013 peak mismatch"
    assert abs(_nat("Fierro en Extraccion", 2025) - 6020396.0) < 1, "Fierro 2025 mismatch"

    n_2013 = int(municipios_activos.filter(pl.col("año") == 2013)["n_municipios"][0])
    n_2025 = int(municipios_activos.filter(pl.col("año") == 2025)["n_municipios"][0])
    assert n_2013 == 138, f"municipios activos 2013 mismatch: {n_2013}"
    assert n_2025 == 78, f"municipios activos 2025 mismatch: {n_2025}"

    def _top_share(prod: str) -> tuple[str, float]:
        ranked = (
            estado_producto.filter(pl.col("producto") == prod)
            .group_by("estado").agg(pl.col("volumen").sum().alias("volumen"))
            .sort("volumen", descending=True)
        )
        total = ranked["volumen"].sum()
        top = ranked.row(0, named=True)
        return top["estado"], float(top["volumen"] / total * 100)

    for prod, exp_estado, exp_pct in [
        ("Cobre", "Sonora", 80.1), ("Fluorita", "San Luis Potosí", 90.8),
        ("Coque", "Coahuila", 92.1), ("Barita", "Nuevo León", 73.7),
    ]:
        est, pct = _top_share(prod)
        assert est == exp_estado and abs(pct - exp_pct) < 0.2, f"{prod} concentration mismatch: {est} {pct:.1f}%"

    for prod in ("Oro", "Plata", "Plomo", "Zinc"):
        top1 = top_municipio_2025.filter((pl.col("producto") == prod) & (pl.col("rank") == 1)).row(0, named=True)
        assert top1["municipio"] == "Mazapil" and top1["estado"] == "Zacatecas", (
            f"Mazapil top-1 mismatch for {prod}: {top1['municipio']}, {top1['estado']}"
        )
    cobre_top1 = top_municipio_2025.filter((pl.col("producto") == "Cobre") & (pl.col("rank") == 1)).row(0, named=True)
    assert cobre_top1["municipio"] == "Cananea" and cobre_top1["estado"] == "Sonora", (
        f"Cobre top-1 mismatch: {cobre_top1['municipio']}, {cobre_top1['estado']}"
    )

    outputs = {
        "eimm_nacional_anual": nacional_anual,
        "eimm_estado_producto": estado_producto,
        "eimm_municipios_activos": municipios_activos,
        "eimm_top_municipio_2025": top_municipio_2025,
    }
    for name, frame in outputs.items():
        path = f"{OUT_DIR}/{name}.parquet"
        frame.write_parquet(path)
        print(f"{path}: {frame.height} rows, {len(frame.columns)} cols")

    print("Todas las anclas de validación coinciden con DATA_OVERVIEW.md.")


if __name__ == "__main__":
    main()
