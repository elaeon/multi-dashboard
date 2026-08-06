"""
Matrícula "local" (excluyendo el nuevo ingreso proporcional de otras entidades)
de 18-24 años, como % de la población de esa edad — por entidad (ANUIES + CONAPO).

Motivo: comparar directo matrícula total / población total sobreestima a
entidades con universidades de alcance nacional (atraen foráneos) y con mucho
posgrado (estudiantes de 25+ años) — sobre todo Ciudad de México, que
concentra UNAM/IPN. Este script corrige ambos sesgos:

1. Usa matrícula por EDAD real (base_anuario_{ciclo}_edades.xlsx, columnas
   "Matrícula 18 años".."Matrícula 24 años") en vez de Matrícula Total.
2. La escala por el % "local" del nuevo ingreso (base_anuario_{ciclo}_procedencia.xlsx)
   — el % de nuevo ingreso cuya procedencia coincide con la entidad de la
   propia escuela — como aproximación de cuánta de esa matrícula 18-24 es de
   residentes de la entidad y no de estudiantes que vinieron de fuera. Es una
   aproximación: asume que la proporción local/foráneo del nuevo ingreso
   aplica igual a toda la matrícula 18-24 (no solo a primer ingreso).
3. Divide entre población 18-24 real (CONAPO, proyecciones_poblacion, banda
   quinquenal 15-19 al 40% + banda 20-24 completa — aproximación porque CONAPO
   no reporta edad simple, solo quinquenios de 5 años).

CAVEAT CONOCIDO (no resuelto por este script): el % local de instituciones
como UNAM está inflado porque su dato de procedencia reporta ~100% local de
forma degenerada en varios ciclos (no es geografía real — ver hallazgo en
conversación: UDEFA/Marina no tienen este problema, pero UNAM sí). Esto
sobreestima el % local — y por lo tanto el % final — de Ciudad de México
específicamente. No hay forma de corregirlo con los datos disponibles.

Fuentes:
- data/anuies/base_anuario_{ciclo}_edades.xlsx — matrícula por edad. Solo
  existe para un subconjunto de ciclos (se listan por glob).
- data/anuies/base_anuario_{ciclo}_procedencia.xlsx — nuevo ingreso por
  entidad de procedencia. Debe existir el mismo ciclo que edades.
- data/conapo/proyecciones_poblacion/00_Republica_mexicana.zip —
  1_Grupo_Quinq_00_RM.xlsx, población quinquenal por municipio, se agrega a
  entidad y años.

--lag elige el ciclo ANUIES: 0 = el más reciente con archivo de edades
disponible, 1 = uno atrás, etc. (mismo criterio que
prepare_comparacion_universidades.py, pero indexando ciclos con "edades").
--año elige el año de población CONAPO a usar (default: año esperado del
ciclo = año de fin de ciclo, ej. 2024-2025 -> 2025).

Por default solo imprime — pasa --save para guardar el parquet en
dashboard_data/.

Output (con --save):
  dashboard_data/matricula_local_18_24_por_entidad_{ciclo}_{año}.parquet
Run: uv run python scripts/prepare_matricula_local_18_24.py --lag 0 --save
"""

import argparse
import io
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

ANUIES_DIR = Path("data/anuies")
ANUIES_EDADES_DIR = ANUIES_DIR / "edades"
ANUIES_PROCEDENCIA_DIR = ANUIES_DIR / "procedencia"
CONAPO_ZIP = Path("data/conapo/proyecciones_poblacion/00_Republica_mexicana.zip")
CONAPO_MEMBER = "00_Republica_mexicana/1_Grupo_Quinq_00_RM.xlsx"
OUT_DIR = Path("dashboard_data")

COLS_18_24 = [f"Matrícula {e} años" for e in range(18, 25)]
COLS_EXTRANJERO = {
    "NUEVO INGRESO PROCEDENCIA ESTADOS UNIDOS", "NUEVO INGRESO PROCEDENCIA CANADÁ",
    "NUEVO INGRESO PROCEDENCIA CENTROAMÉRICA Y CARIBE", "NUEVO INGRESO PROCEDENCIA SUDAMÉRICA",
    "NUEVO INGRESO PROCEDENCIA ÁFRICA", "NUEVO INGRESO PROCEDENCIA ASIA",
    "NUEVO INGRESO PROCEDENCIA EUROPA", "NUEVO INGRESO PROCEDENCIA OCEANÍA",
}
SUFIJOS_OFICIALES = (" DE ZARAGOZA", " DE OCAMPO", " DE IGNACIO DE LA LLAVE")


def normalizar_entidad(s: str) -> str:
    s = s.upper()
    for suf in SUFIJOS_OFICIALES:
        s = s.replace(suf, "")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def encontrar_ciclo(lag: int) -> tuple[str, Path, Path, Path]:
    edades_files = sorted(ANUIES_EDADES_DIR.glob("base_anuario_*_edades.xlsx"), reverse=True)
    if not edades_files:
        raise FileNotFoundError(f"No se encontraron archivos de edades (base_anuario_*_edades.xlsx) en {ANUIES_EDADES_DIR}")
    if lag < 0 or lag >= len(edades_files):
        ciclos = [f.stem.removeprefix("base_anuario_").removesuffix("_edades") for f in edades_files]
        raise ValueError(f"--lag {lag} fuera de rango: solo hay {len(edades_files)} ciclo(s) con edades disponibles ({ciclos})")
    edades_path = edades_files[lag]
    ciclo = edades_path.stem.removeprefix("base_anuario_").removesuffix("_edades")

    procedencia_path = ANUIES_PROCEDENCIA_DIR / f"base_anuario_{ciclo}_procedencia.xlsx"
    if not procedencia_path.exists():
        raise FileNotFoundError(f"Falta {procedencia_path} — se requiere el mismo ciclo que edades ({ciclo})")

    año_fin_ciclo = int(ciclo.split("-")[1])
    return ciclo, edades_path, procedencia_path, año_fin_ciclo + 0  # año esperado = fin de ciclo


def matricula_18_24_por_entidad(edades_path: Path) -> pl.DataFrame:
    edades = pl.read_excel(edades_path, sheet_name="Base de datos")
    edades = edades.with_columns(pl.sum_horizontal(COLS_18_24).alias("mat_18_24"))
    return edades.group_by("ENTIDAD").agg(pl.sum("mat_18_24").alias("mat_18_24"))


def pct_local_por_entidad(procedencia_path: Path) -> pl.DataFrame:
    proc = pl.read_excel(procedencia_path, sheet_name="Base de datos")
    cols_estado_mx = [c for c in proc.columns if c.startswith("NUEVO INGRESO PROCEDENCIA") and c not in COLS_EXTRANJERO]

    proc = proc.with_columns(
        pl.when(pl.col("ENTIDAD") == "MÉXICO").then(pl.lit("NUEVO INGRESO PROCEDENCIA ESTADO DE MÉXICO"))
        .otherwise(pl.concat_str([pl.lit("NUEVO INGRESO PROCEDENCIA "), pl.col("ENTIDAD")]))
        .alias("col_local")
    )
    proc = proc.with_columns(pl.sum_horizontal(cols_estado_mx).alias("ni_nacional"))

    mat = proc.select(cols_estado_mx).to_numpy()
    col_index = {c: i for i, c in enumerate(cols_estado_mx)}
    locales = np.array([mat[i, col_index[c]] for i, c in enumerate(proc["col_local"].to_list())])
    proc = proc.with_columns(pl.Series("local", locales))

    por_ent = proc.group_by("ENTIDAD").agg(pl.sum("ni_nacional").alias("ni_nacional"), pl.sum("local").alias("ni_local"))
    return por_ent.with_columns((pl.col("ni_local") / pl.col("ni_nacional")).alias("pct_local"))


def poblacion_18_24_por_entidad(año: int) -> pl.DataFrame:
    with zipfile.ZipFile(CONAPO_ZIP) as zf:
        data = zf.read(CONAPO_MEMBER)
    quinq = pl.read_excel(
        io.BytesIO(data),
        columns=["NOM_ENT", "SEXO", "AÑO", "POB_15_19", "POB_20_24"],
    )
    quinq = quinq.filter(pl.col("AÑO") == año)
    if quinq.height == 0:
        raise ValueError(f"--año {año} no existe en {CONAPO_MEMBER} (rango típico: 1990-2040)")

    pob = quinq.group_by("NOM_ENT").agg(pl.sum("POB_15_19").alias("p15_19"), pl.sum("POB_20_24").alias("p20_24"))
    return pob.with_columns((pl.col("p15_19") * 0.4 + pl.col("p20_24")).alias("pob_18_24"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lag", type=int, default=0, help="Ciclo ANUIES a usar (por archivo de edades disponible): 0 = más reciente, 1 = uno atrás, etc.")
    parser.add_argument("--año", type=int, default=None, help="Año de población CONAPO a usar (default: año de fin del ciclo ANUIES)")
    parser.add_argument("--save", action="store_true", help="Guardar el resultado como parquet en dashboard_data/ (por default no se guarda, solo se imprime)")
    args = parser.parse_args()

    ciclo, edades_path, procedencia_path, año_esperado = encontrar_ciclo(args.lag)
    año = args.año if args.año is not None else año_esperado

    mat18 = matricula_18_24_por_entidad(edades_path)
    pct_local = pct_local_por_entidad(procedencia_path)
    pob = poblacion_18_24_por_entidad(año)
    pob = pob.with_columns(pl.col("NOM_ENT").map_elements(normalizar_entidad, return_dtype=pl.Utf8).alias("key"))

    tabla = mat18.join(pct_local.select("ENTIDAD", "pct_local"), on="ENTIDAD", how="left")
    tabla = tabla.with_columns(
        (pl.col("mat_18_24") * pl.col("pct_local")).alias("mat_18_24_local"),
        pl.col("ENTIDAD").map_elements(normalizar_entidad, return_dtype=pl.Utf8).alias("key"),
    )
    tabla = tabla.join(pob.select("key", "pob_18_24"), on="key", how="left")

    sin_pob = tabla.filter(pl.col("pob_18_24").is_null())
    if sin_pob.height:
        raise ValueError(f"Sin población 18-24 emparejada para: {sin_pob['ENTIDAD'].to_list()} (revisar normalización de nombre de entidad)")

    tabla = tabla.with_columns(
        (pl.col("mat_18_24_local") / pl.col("pob_18_24") * 100).alias("pct_final"),
        pl.lit(ciclo).alias("CICLO_ANUIES"),
        pl.lit(año).alias("AÑO_POBLACION"),
    ).sort("pct_final", descending=True).drop("key")

    print(f"Ciclo ANUIES: {ciclo} (edades: {edades_path.name}, procedencia: {procedencia_path.name})")
    print(f"Año de población CONAPO: {año}" + ("" if año == año_esperado else f" [aviso: distinto al año esperado del ciclo, {año_esperado}]"))
    print()
    print(f"{'Entidad':<20} {'Mat.18-24':>10} {'% local':>8} {'Mat.local':>10} {'Pob.18-24':>10} {'% FINAL':>8}")
    for r in tabla.iter_rows(named=True):
        print(f"{r['ENTIDAD']:<20} {r['mat_18_24']:>10,} {r['pct_local']*100:>7.1f}% {r['mat_18_24_local']:>10,.0f} {r['pob_18_24']:>10,.0f} {r['pct_final']:>7.2f}%")

    if args.save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"matricula_local_18_24_por_entidad_{ciclo}_{año}.parquet"
        tabla.write_parquet(out_path)
        print(f"\nGuardado → {out_path}")
    else:
        print("\n(no guardado — pasa --save para escribir el parquet)")


if __name__ == "__main__":
    main()
