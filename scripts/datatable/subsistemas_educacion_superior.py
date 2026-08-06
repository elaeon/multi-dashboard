"""
Peso de cada subsistema educativo público (estatal, federal, tecnológicas,
politécnicas, interculturales, Tecnológico Nacional, normales, otras IES
públicas, apoyo solidario) y de sostenimiento público/privado en la matrícula
de educación superior, por entidad federativa — ANUIES.

TecNM (Tecnológico Nacional de México) combina las unidades federales y
descentralizadas del catálogo SUBSISTEMA de ANUIES en una sola columna.

Por default usa solo licenciatura+TSU (LICENCIATURA UNIVERSITARIA Y
TECNOLÓGICA, LICENCIATURA EN EDUCACIÓN NORMAL, TÉCNICO SUPERIOR) — nunca
mezcla licenciatura con posgrado, porque tienen estructuras de oferta muy
distintas (ej. Centros de Investigación CONACYT solo aparece con --posgrado,
al no ofrecer licenciatura). --posgrado cambia a maestría+doctorado+
especialidad en su lugar.

--year elige el ciclo ANUIES por su año de FIN (ej. --year 2023 -> ciclo
2022-2023); default el ciclo más reciente disponible en
data/anuies/general/base_anuario_*_general.xlsx.

Por default solo imprime la tabla en terminal. --guardar escribe el parquet
en dashboard_data/ (nombre con sufijo _licenciatura o _posgrado).

Run: uv run python scripts/datatable/subsistemas_educacion_superior.py
Run: uv run python scripts/datatable/subsistemas_educacion_superior.py --year 2023
Run: uv run python scripts/datatable/subsistemas_educacion_superior.py --posgrado
"""

import argparse
from pathlib import Path

import polars as pl

RAIZ = Path(__file__).resolve().parents[2]
ANUIES_GENERAL_DIR = RAIZ / "data" / "anuies" / "general"
OUT_DIR = RAIZ / "dashboard_data"

NIVELES_LIC = {"LICENCIATURA UNIVERSITARIA Y TECNOLÓGICA", "LICENCIATURA EN EDUCACIÓN NORMAL", "TÉCNICO SUPERIOR"}
NIVELES_POS = {"MAESTRÍA", "DOCTORADO", "ESPECIALIDAD"}

# columna de salida -> valor(es) de SUBSISTEMA que agrupa
SUBSISTEMAS = {
    "estatal": ["UNIVERSIDADES PÚBLICAS ESTATALES"],
    "tecnologica": ["UNIVERSIDADES TECNOLÓGICAS"],
    "politecnica": ["UNIVERSIDADES POLITÉCNICAS"],
    "intercultural": ["UNIVERSIDADES INTERCULTURALES"],
    "federal": ["UNIVERSIDADES PÚBLICAS FEDERALES"],
    "otras_ies": ["OTRAS IES PÚBLICAS"],
    "normales": ["NORMALES PÚBLICAS"],
    "tecnm": ["UNIDADES FEDERALES DEL TECNOLÓGICO NACIONAL DE MÉXICO",
              "UNIDADES DESCENTRALIZADAS DEL TECNOLÓGICO NACIONAL DE MÉXICO"],
    "conacyt": ["CENTROS DE INVESTIGACIÓN CONACYT"],
    "apoyo_solidario": ["UNIVERSIDADES PÚBLICAS ESTATALES DE APOYO SOLIDARIO"],
}


def _año(valor):
    try:
        return int(valor)
    except ValueError:
        raise argparse.ArgumentTypeError(f"año inválido: {valor!r}")


def encontrar_anuies(year: int | None) -> tuple[Path, str]:
    """Ciclo ANUIES cuyo año de FIN coincide con --year (ej. 2023 -> "2022-2023");
    sin --year, el ciclo más reciente disponible."""
    archivos = sorted(ANUIES_GENERAL_DIR.glob("base_anuario_*_general.xlsx"))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron anuarios ANUIES (base_anuario_*_general.xlsx) en {ANUIES_GENERAL_DIR}")
    ciclos = {a.stem.removeprefix("base_anuario_").removesuffix("_general"): a for a in archivos}

    if year is None:
        ciclo = max(ciclos)  # "AAAA-AAAA" ordena correctamente como string
        return ciclos[ciclo], ciclo

    candidatos = [c for c in ciclos if c.split("-")[1] == str(year)]
    if not candidatos:
        disponibles = sorted(int(c.split("-")[1]) for c in ciclos)
        raise ValueError(f"--year {year}: no hay ciclo ANUIES que termine en {year}. Años disponibles: {disponibles}")
    ciclo = candidatos[0]
    return ciclos[ciclo], ciclo


def calcular_tabla(anuies_path: Path, posgrado: bool) -> pl.DataFrame:
    niveles = NIVELES_POS if posgrado else NIVELES_LIC
    anuies = pl.read_excel(anuies_path, sheet_name="Base de datos").filter(pl.col("NIVEL").is_in(niveles))
    if anuies.height == 0:
        nombre_nivel = "posgrado" if posgrado else "licenciatura"
        raise ValueError(f"Matrícula 0 en {nombre_nivel} en {anuies_path.name} — revisar NIVEL para este ciclo.")

    por_ent_sos = anuies.group_by("ENTIDAD", "SOSTENIMIENTO").agg(pl.sum("Matrícula Total").alias("MAT"))
    tabla = por_ent_sos.pivot(values="MAT", index="ENTIDAD", on="SOSTENIMIENTO").fill_null(0)
    for sost in ("PÚBLICO", "PARTICULAR"):
        if sost not in tabla.columns:
            tabla = tabla.with_columns(pl.lit(0).alias(sost))
    tabla = tabla.with_columns((pl.col("PÚBLICO") + pl.col("PARTICULAR")).alias("TOTAL"))

    for col, subsistemas in SUBSISTEMAS.items():
        sub = anuies.filter(pl.col("SUBSISTEMA").is_in(subsistemas)).group_by("ENTIDAD").agg(
            pl.sum("Matrícula Total").alias(col))
        tabla = tabla.join(sub, on="ENTIDAD", how="left").fill_null(0)

    pct_cols = {f"pct_{c}": (pl.col(c) / pl.col("TOTAL") * 100) for c in list(SUBSISTEMAS) + ["PÚBLICO", "PARTICULAR"]}
    tabla = tabla.with_columns(**pct_cols)
    return tabla.sort("pct_PARTICULAR", descending=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=_año, default=None,
                   help="Año de fin del ciclo ANUIES a usar (ej. --year 2023 -> ciclo 2022-2023); "
                        "default el ciclo más reciente disponible")
    p.add_argument("--posgrado", action="store_true",
                   help="Usa maestría+doctorado+especialidad en vez de licenciatura+TSU (default)")
    p.add_argument("--guardar", action="store_true",
                   help="Guarda el resultado como parquet en dashboard_data/ (por default solo se imprime)")
    a = p.parse_args()

    anuies_path, ciclo = encontrar_anuies(a.year)
    nombre_nivel = "posgrado" if a.posgrado else "licenciatura"
    tabla = calcular_tabla(anuies_path, a.posgrado)

    columnas_subsistema = [
        ("estatal", "Estat"), ("tecnologica", "Tecn"), ("politecnica", "Polit"),
        ("intercultural", "Interc"), ("federal", "Fed"), ("otras_ies", "OtIES"),
        ("normales", "Norm"), ("tecnm", "TecNM"), ("conacyt", "CONACYT"),
        ("apoyo_solidario", "ApoyoSol"),
    ]

    print(f"\n{'═' * 130}")
    print(f"  Matrícula de educación superior por sostenimiento y subsistema — {ciclo} ({nombre_nivel})")
    print("═" * 130)
    encabezado = f"  {'Entidad':<20} {'Total':>9} {'%Púb':>6} {'%Priv':>6}"
    for _, etiqueta in columnas_subsistema:
        encabezado += f" {'%' + etiqueta:>9}"
    print(encabezado)
    for r in tabla.iter_rows(named=True):
        fila = f"  {r['ENTIDAD']:<20} {r['TOTAL']:>9,.0f} {r['pct_PÚBLICO']:>5.1f}% {r['pct_PARTICULAR']:>5.1f}%"
        for col, _ in columnas_subsistema:
            fila += f" {r['pct_' + col]:>8.1f}%"
        print(fila)

    total = tabla["TOTAL"].sum()
    publico = tabla["PÚBLICO"].sum()
    particular = tabla["PARTICULAR"].sum()
    print(f"\n  Nacional: {total:,.0f} — público {publico/total*100:.1f}%, privado {particular/total*100:.1f}%")
    for col, etiqueta in columnas_subsistema:
        print(f"    {etiqueta}: {tabla[col].sum()/total*100:.1f}%", end="")
    print("\n")

    if a.guardar:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        ruta = OUT_DIR / f"subsistemas_educacion_superior_{ciclo}_{nombre_nivel}.parquet"
        tabla.write_parquet(ruta)
        print(f"Guardado → {ruta}  ({tabla.height:,} filas)\n")


if __name__ == "__main__":
    main()
