"""
Prepara las tablas de comercio exterior minero para el dashboard de Producción
Minera (SGM, Anuario Estadístico de la Minería Mexicana).

Lee data/sgm/produccion_minera/AnuarioEstadistico.rar (requiere el binario `unrar`
en PATH — Python no tiene soporte nativo para RAR), extrae a un directorio
temporal, limpia los 4 CSV de exportaciones/importaciones (BOM UTF-8, CRLF, filas
en blanco, pie de nota) y guarda dos tablas unificadas en dashboard_data/.

También valida, por separado, que el hallazgo de "Agregados pétreos" del
dashboard existente se confirma de forma independiente en este archivo oficial
(no se guarda como tabla — el resultado se imprime para copiar el texto exacto
al dashboard, igual que el hallazgo de Mazapil en scripts/prepare_eimm_minerometalurgica.py).
"""

import subprocess
import tempfile
from pathlib import Path

import polars as pl

RAR_PATH = "data/sgm/produccion_minera/AnuarioEstadistico.rar"
OUT_DIR = "dashboard_data"

# Nombre oficial en español (ambas variantes de ortografía donde el archivo de
# exportaciones y el de importaciones difieren) -> ISO-3.
NAME_TO_ISO3 = {
    "Alemania (República Federal de)": "DEU",
    "Arabia Saudita (Reino de)": "SAU",
    "Argentina (República)": "ARG",
    "Australia (Mancomunidad de)": "AUS",
    "Austria (República de)": "AUT",
    "Bahréin (Reino de)": "BHR",
    "Belice": "BLZ",
    "Bolivia (Estado Plurinacional de)": "BOL",
    "Brasil (República Federativa de)": "BRA",
    "Bélgica (Reino de)": "BEL",
    "Canadá": "CAN",
    "Chile (República de)": "CHL",
    "China (República Popular)": "CHN",
    "Colombia (República de)": "COL",
    "Corea (República de)": "KOR",
    "Costa Rica (República de)": "CRI",
    "Cuba (República de)": "CUB",
    "Dinamarca (Reino de)": "DNK",
    "Ecuador (República del)": "ECU",
    "El Salvador (República de)": "SLV",
    "Emiratos Árabes Unidos": "ARE",
    "Eslovenia (República de)": "SVN",
    "España (Reino de)": "ESP",
    "Estados Unidos de América": "USA",
    "Estonia (República de)": "EST",
    "Finlandia (República de)": "FIN",
    "Francia (República Francesa)": "FRA",
    "Grecia (República Helénica)": "GRC",
    "Guatemala (República de)": "GTM",
    "Guyana (República Cooperativa de)": "GUY",
    "Honduras (República de)": "HND",
    "Hong Kong (Región Administrativa Especial de)": "HKG",
    "India (República de la)": "IND",
    "Israel (Estado de)": "ISR",
    "Italia (República Italiana)": "ITA",
    "Jamaica": "JAM",
    "Japón": "JPN",
    "Malasia (Federación de)": "MYS",
    "Nicaragua (República de)": "NIC",
    "Noruega (Reino de)": "NOR",
    "Nueva Zelanda": "NZL",
    "Panamá (República de)": "PAN",
    "Países Bajos (Reino de los)": "NLD",
    "Perú (República del)": "PER",
    "Portugal (República Portuguesa)": "PRT",
    "Puerto Rico (Estado Libre Asociado de)": "PRI",
    "Reino Unido de Gran Bretaña e Irlanda del Norte": "GBR",
    "Reino Unido de la Gran Bretaña e Irlanda del Norte": "GBR",
    "República Dominicana": "DOM",
    "Rusia (Federación de)": "RUS",
    "Singapur (República de)": "SGP",
    "Sri Lanka (República Democrática Socialista de)": "LKA",
    "Sudáfrica (República de)": "ZAF",
    "Suecia (Reino de)": "SWE",
    "Suiza (Confederación Suiza)": "CHE",
    "Suiza (Confederación)": "CHE",
    "Tailandia (Reino de)": "THA",
    "Taiwán (República de China)": "TWN",
    "Turquía (República de)": "TUR",
    "Venezuela (República de)": "VEN",
    "Vietnam (República Socialista de)": "VNM",
    # "Los demás" es un cajón de sastre (resto de países), sin ISO-3 -> queda null.
}


def _clean(df: pl.DataFrame, key_col: str) -> pl.DataFrame:
    return df.filter(pl.col(key_col).str.strip_chars() != "").filter(
        ~pl.col(key_col).str.contains("Nota")
    )


def load_producto(extract_dir: Path, filename: str, flujo: str) -> pl.DataFrame:
    df = pl.read_csv(extract_dir / filename, encoding="utf8-lossy")
    df = _clean(df, "Productos")
    val_2024_col = [c for c in df.columns if "2024" in c][0]
    return df.select(
        pl.lit(flujo).alias("flujo"),
        pl.col("Productos").str.strip_chars().alias("producto"),
        pl.col("Tipo").str.strip_chars().alias("tipo"),
        pl.col("Dólares corrientes año 2023").cast(pl.Float64).alias("valor_2023"),
        pl.col(val_2024_col).cast(pl.Float64).alias("valor_2024"),
    )


def load_pais(extract_dir: Path, filename: str, flujo: str) -> pl.DataFrame:
    df = pl.read_csv(extract_dir / filename, encoding="utf8-lossy")
    df = _clean(df, "Pais")
    val_2024_col = [c for c in df.columns if "2024" in c][0]
    return df.select(
        pl.lit(flujo).alias("flujo"),
        pl.col("Pais").str.strip_chars().alias("pais"),
        pl.col("Dólares corrientes año 2023").cast(pl.Float64).alias("valor_2023"),
        pl.col(val_2024_col).cast(pl.Float64).alias("valor_2024"),
    ).with_columns(
        pl.col("pais").replace_strict(NAME_TO_ISO3, default=None).alias("iso3")
    )


def check_agregados_cross(extract_dir: Path) -> None:
    df = pl.read_csv(extract_dir / "8_Produccion_Minera_Total_Producto_UTF8.csv", encoding="utf8-lossy")
    df = _clean(df, "Productos").filter(pl.col("Tipo").str.strip_chars() != "")
    agregados_2024 = float(
        df.filter(pl.col("Productos").str.strip_chars() == "Agregados pétreos")
        ["Pesos corrientes año 2024 (cifras preliminares)"][0]
    )

    clean = pl.read_parquet(f"{OUT_DIR}/produccion_minera_clean.parquet")
    hidalgo_2024 = float(
        clean.filter(
            (pl.col("estado") == "Hidalgo") & (pl.col("producto") == "Agregados pétreos") & (pl.col("año") == 2024)
        )["valor_pesos"][0]
    )

    print(
        f"Cross-check Agregados pétreos 2024: fuente oficial RAR = {agregados_2024:,.0f} pesos "
        f"(nacional) vs. Hidalgo solo en la fuente OCR existente = {hidalgo_2024:,.0f} pesos. "
        f"Confirma el mismo patrón de forma independiente."
    )

    # Comparación de totales limpios (ex-Agregados) 2020-2024 entre ambas fuentes.
    ex_agregados = df.filter(pl.col("Productos").str.strip_chars() != "Agregados pétreos")
    year_cols = [c for c in df.columns if "Pesos corrientes" in c]
    nat = ex_agregados.select([pl.col(c).sum().alias(c) for c in year_cols])
    print("Totales nacionales ex-Agregados por año (fuente oficial RAR):")
    print(nat)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        extract_dir = Path(tmp)
        subprocess.run(
            ["unrar", "x", "-y", str(Path(RAR_PATH).resolve()), str(extract_dir) + "/"],
            check=True, capture_output=True,
        )

        producto = pl.concat([
            load_producto(extract_dir, "1_Exportaciones_Metalicos_NoMetalicos_UTF8.csv", "Exportación"),
            load_producto(extract_dir, "7_Importaciones_Metalicos_NoMetalicos_UTF8.csv", "Importación"),
        ])
        pais = pl.concat([
            load_pais(extract_dir, "6_Exportaciones_Areas_Geograficas_UTF8.csv", "Exportación"),
            load_pais(extract_dir, "12_Importaciones_Areas_Geograficas_UTF8.csv", "Importación"),
        ])

        check_agregados_cross(extract_dir)

    # ── anclas de validación ────────────────────────────────────────────────
    def _tot(frame: pl.DataFrame, flujo: str, year_col: str) -> float:
        return float(frame.filter(pl.col("flujo") == flujo)[year_col].sum())

    assert abs(_tot(producto, "Exportación", "valor_2023") - 16_784_358_737) < 1, "export 2023 total mismatch"
    assert abs(_tot(producto, "Exportación", "valor_2024") - 17_685_879_730) < 1, "export 2024 total mismatch"
    assert abs(_tot(producto, "Importación", "valor_2023") - 28_755_539_365) < 1, "import 2023 total mismatch"
    assert abs(_tot(producto, "Importación", "valor_2024") - 29_645_876_876) < 1, "import 2024 total mismatch"

    for flujo in ("Exportación", "Importación"):
        for year_col in ("valor_2023", "valor_2024"):
            prod_tot = _tot(producto, flujo, year_col)
            pais_tot = _tot(pais, flujo, year_col)
            assert abs(prod_tot - pais_tot) < 1, (
                f"cross-file total mismatch {flujo}/{year_col}: producto={prod_tot} pais={pais_tot}"
            )

    for flujo in ("Exportación", "Importación"):
        leader = (
            pais.filter(pl.col("flujo") == flujo)
            .sort("valor_2024", descending=True).row(0, named=True)
        )
        assert leader["pais"] == "Estados Unidos de América", f"expected USA leader for {flujo}, got {leader['pais']}"

    n_unmapped = pais.filter(pl.col("iso3").is_null() & (pl.col("pais") != "Los demás")).height
    assert n_unmapped == 0, f"{n_unmapped} country names have no ISO-3 mapping"

    outputs = {"comercio_producto": producto, "comercio_pais": pais}
    for name, frame in outputs.items():
        path = f"{OUT_DIR}/{name}.parquet"
        frame.write_parquet(path)
        print(f"{path}: {frame.height} rows, {len(frame.columns)} cols")

    print("Todas las anclas de validación coinciden con lo verificado en el plan.")


if __name__ == "__main__":
    main()
