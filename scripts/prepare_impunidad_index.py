"""
Índice de Impunidad Estatal.

Índice principal (panel 2020-2024, 32 entidades): 4 pilares, cada uno
percentilado dentro de su año e invertido donde "menos = peor" (menos
policía, menos judicialización, menos recomendaciones de derechos humanos
= más impunidad; más prisión preventiva = más impunidad directamente):

  - Capacidad policial: policías/1,000 hab (CNGSPSPE 2017-19 + CNSPE 2020-24,
    reusa fuerza_estatal() ya validada en cap12_capacidad_policial.py).
  - Denuncias sin resolución (fiscalía): CNPJE, "Ejercicio de la acción
    penal" ÷ total de determinaciones, por estado.
  - Derechos humanos sin sanción: CNDHE, recomendaciones formales (código
    de conclusión 300) ÷ expedientes concluidos, por estado.
  - Prisión preventiva: CNSIPEE, % de población privada de la libertad
    SIN sentencia.

Todas las tablas se localizan vía 0_indice/diccionario_de_datos por texto,
nunca por código de tabla o columna hardcodeado (el código cambia cada año
en las cuatro fuentes) — ver docstrings de cada `cargar_*`.

Columnas de contexto (fuera del índice compuesto, no afectan el score):
desconfianza institucional percibida (ENVIPE, "algo/mucha desconfianza" en
Policía Estatal, Jueces y Ministerio Público/Fiscalías Estatales, ponderado
por FAC_ELE). Se dejan separadas del índice porque son datos de encuesta
(con margen muestral) mientras los 4 pilares son datos administrativos
(censos) — mezclarlos en un solo score combinaría dos tipos de error
distintos. Ver `cargar_envipe_desconfianza()`.

Capa municipal (2 tablas SEPARADAS, NO fusionadas al índice estatal, cada
una un corte transversal único, no una serie):
  - Policía municipal (CNGMD, edición 2025 / dato 2024).
  - Enfrentamientos y bajas (CNGSPSPE m2pb, único año disponible: 2019).

No es un dato de dashboard: la carpeta de salida se especifica por línea de
comandos.

Run: uv run python scripts/prepare_impunidad_index.py --output-dir <ruta>
"""
import argparse
import glob
import io
import re
import sys
import zipfile
from pathlib import Path

import polars as pl
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts/centralismo"))
from comun import ESTADOS, cargar_poblacion          # noqa: E402
from cap12_capacidad_policial import fuerza_estatal, leer_zip_csv  # noqa: E402

ANIO_MIN, ANIO_MAX = 2020, 2024
SENTINELAS = ["NSS", "NA", "ND", "NP"]
PESOS = {"pct_capacidad": 0.25, "pct_judicializacion": 0.25,
         "pct_ddhh": 0.25, "pct_prision_preventiva": 0.25}
NIVELES = ["Bajo", "Medio", "Alto", "Extremo"]


# ---------------------------------------------------------------- utilidades genéricas

def _norm(n: str) -> str:
    return n.replace("\\", "/")


def _limpiar_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().replace("ï»¿", "").replace("﻿", "") for c in df.columns]
    return df


def _leer_miembro(z: zipfile.ZipFile, nombre_original: str, **kw) -> pd.DataFrame:
    with z.open(nombre_original) as f:
        raw = f.read()
    try:
        texto = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        texto = raw.decode("latin-1")
    # Algunas ediciones de ENVIPE (2020-2021) usan '\r' puro como terminador de
    # línea (estilo Mac clásico, sin '\n'); io.StringIO no lo traduce por
    # defecto y el '\r' queda pegado al último campo de cada fila.
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    return _limpiar_cols(pd.read_csv(io.StringIO(texto), **kw))


def _indice(zpath: Path) -> pd.DataFrame:
    """Lee 0_indice_*.csv (ARCHIVO, CONTENIDO) — mapa de tabla → descripción."""
    with zipfile.ZipFile(zpath) as z:
        entries = [_norm(n) for n in z.namelist()]
        orig = z.namelist()
        i = next(i for i, e in enumerate(entries) if "0_indice" in e.lower())
        return _leer_miembro(z, orig[i])


def _tabla_y_diccionario(zpath: Path, stem: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga la tabla de datos `stem` y su diccionario de columnas."""
    with zipfile.ZipFile(zpath) as z:
        entries = [_norm(n) for n in z.namelist()]
        orig = z.namelist()
        dato_i = next(i for i, e in enumerate(entries)
                      if e.endswith(".csv") and "conjunto_de_datos" in e and f"/{stem}_" in e)
        dic_i = next(i for i, e in enumerate(entries)
                     if e.endswith(".csv") and "diccionario" in e and stem in e)
        data = _leer_miembro(z, orig[dato_i])
        dic = _leer_miembro(z, orig[dic_i])
        return data, dic


def _col_ent(df: pd.DataFrame) -> str:
    return next(c for c in ("entidad_a", "cve_ent", "entidad") if c in df.columns)


def _a_numero(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(serie.replace(SENTINELAS, pd.NA), errors="coerce")


# ---------------------------------------------------------------- pilar 2: CNPJE

def cargar_cnpje(anio_dato: int) -> pl.DataFrame:
    """Estado × tasa_judicializacion = 'Ejercicio de la acción penal' / total
    de determinaciones (denuncias sin resolución, lado fiscalía)."""
    edicion = anio_dato + 1
    zpath = Path(glob.glob(str(RAIZ / f"data/inegi/cnpje/{edicion}/m2_*cierre*cnpje{edicion}*.zip"))[0])
    idx = _indice(zpath)
    fila = idx[idx["CONTENIDO"].str.startswith("Determinaciones y/o conclusiones", na=False)]
    stem = fila.iloc[0]["ARCHIVO"]
    data, dic = _tabla_y_diccionario(zpath, stem)

    total_col = dic[dic["DESCRIPCION"].str.strip() == "Total"]["COLUMNA"].iloc[0]
    accion_col = dic[dic["DESCRIPCION"].str.startswith("Ejercicio de la acción penal", na=False)
                      & ~dic["DESCRIPCION"].str.contains(" - ", na=False)]["COLUMNA"].iloc[0]
    ecol = _col_ent(data)

    data[total_col] = _a_numero(data[total_col])
    data[accion_col] = _a_numero(data[accion_col])
    data[ecol] = pd.to_numeric(data[ecol], errors="coerce")
    data = data[data[ecol].between(1, 32)]

    return pl.DataFrame({
        "cve_ent": data[ecol].astype(int).to_list(),
        "tasa_judicializacion": (data[accion_col] / data[total_col]).to_list(),
    }).with_columns(pl.lit(anio_dato).alias("anio"))


# ---------------------------------------------------------------- pilar 3: CNDHE

def cargar_cndhe(anio_dato: int) -> pl.DataFrame:
    """Estado × tasa_recomendacion = recomendaciones formales (código 300)
    / expedientes concluidos (derechos humanos sin sanción)."""
    edicion = anio_dato + 1
    zpath = Path(glob.glob(str(RAIZ / f"data/inegi/cndhe/{edicion}/*exped_conclu*cndhe{edicion}*.zip"))[0])
    idx = _indice(zpath)
    fila = idx[idx["CONTENIDO"].str.contains("tipo y grado de conclus", case=False, na=False)]
    stem = fila.iloc[0]["ARCHIVO"]

    with zipfile.ZipFile(zpath) as z:
        entries = [_norm(n) for n in z.namelist()]
        orig = z.namelist()
        dato_i = next(i for i, e in enumerate(entries)
                      if e.endswith(".csv") and "conjunto_de_datos" in e and f"/{stem}_" in e)
        cat_i = next(i for i, e in enumerate(entries) if "catalogos" in e and "tipoconc" in e)
        data = _leer_miembro(z, orig[dato_i])
        cat = _leer_miembro(z, orig[cat_i])

    tcol = next(c for c in data.columns if c.startswith("tipoconc"))
    ccol = next(c for c in cat.columns if c.startswith("tipoconc"))
    codigo_recomendacion = cat[cat["descrip"].str.contains("ecomendaci", na=False)
                               & ~cat["descrip"].str.contains(" - ", na=False)][ccol].iloc[0]

    ecol = _col_ent(data)
    data["exqucott"] = _a_numero(data["exqucott"])
    data[ecol] = pd.to_numeric(data[ecol], errors="coerce")
    data = data[data[ecol].between(1, 32)]

    total = data.groupby(ecol)["exqucott"].sum()
    rec = data[data[tcol] == codigo_recomendacion].groupby(ecol)["exqucott"].sum()
    tasa = (rec / total).reindex(total.index).fillna(0)

    return pl.DataFrame({
        "cve_ent": [int(v) for v in tasa.index],
        "tasa_recomendacion": tasa.to_list(),
    }).with_columns(pl.lit(anio_dato).alias("anio"))


# ---------------------------------------------------------------- pilar 4: CNSIPEE

def cargar_cnsipee(anio_dato: int) -> pl.DataFrame:
    """Estado × pct_sin_sentencia (prisión preventiva)."""
    edicion = anio_dato + 1
    zips = (glob.glob(str(RAIZ / f"data/inegi/cnsipee/{edicion}/m1_*poblacion_priv_libertad*cnsipee{edicion}*.zip"))
            + glob.glob(str(RAIZ / f"data/inegi/cnsipee/{edicion}/m1_pob_privad_libert_cnsipee{edicion}*.zip")))
    zpath = Path(zips[0])
    idx = _indice(zpath)
    fila = idx[idx["CONTENIDO"].str.startswith(
        "Personas privadas de la libertad en los centros penitenciarios, "
        "según estatus jurídico y sexo", na=False)]
    stem = fila.iloc[0]["ARCHIVO"]
    data, dic = _tabla_y_diccionario(zpath, stem)

    total_col = dic[dic["DESCRIPCION"].str.strip() == "Total"]["COLUMNA"].iloc[0]
    sin_cols = dic[dic["DESCRIPCION"].str.match(r"^Sin sentencia.*Subtotal$", na=False)]["COLUMNA"].tolist()
    ecol = _col_ent(data)

    for c in [total_col] + sin_cols:
        data[c] = _a_numero(data[c])
    data[ecol] = pd.to_numeric(data[ecol], errors="coerce")
    data = data[data[ecol].between(1, 32)]

    total = data.groupby(ecol)[total_col].sum()
    sin = data.groupby(ecol)[sin_cols].sum().sum(axis=1)
    pct = (sin / total).reindex(total.index)

    return pl.DataFrame({
        "cve_ent": [int(v) for v in pct.index],
        "pct_sin_sentencia": pct.to_list(),
    }).with_columns(pl.lit(anio_dato).alias("anio"))


# ---------------------------------------------------------------- contexto: ENVIPE

def _zip_envipe(anio: int) -> Path:
    candidatos = [p for p in (RAIZ / "data/inegi/envipe").glob("*.zip")
                  if re.search(rf"envipe_?{anio}", p.name, re.IGNORECASE)]
    return candidatos[0]


def _miembro_tper_vic1(z: zipfile.ZipFile, carpeta: str) -> str:
    """`carpeta`: 'conjunto_de_datos' o 'diccionario_de_datos' — el nombre de
    la carpeta raíz de la tabla cambia de capitalización cada año
    (tper_vic1_envipe2024/, conjunto_de_datos_TPer_Vic1_ENVIPE_2020/, ...),
    por eso se resuelve por substring de ruta con delimitadores de carpeta."""
    return next(e for e in z.namelist()
                if "tper_vic1" in e.lower() and f"/{carpeta}/" in e.lower() and e.endswith(".csv"))


def cargar_envipe_desconfianza(anio: int) -> pl.DataFrame:
    """cve_ent × tasa_desconfianza_policia/jueces/mp — % de respuestas "algo/
    mucha desconfianza" (códigos 3-4 de la escala AP5_4, 1-4) sobre
    respuestas válidas (excluye sentinela 9 "no sabe/no responde"), ponderado
    por FAC_ELE, entre entrevistas completas (RESUL_H == 'A').

    anio_envipe == anio, SIN desfase de un año (a diferencia de CNPJE/CNDHE/
    CNSIPEE): ENVIPE es una encuesta de percepción fielded a mitad del año de
    publicación, mide la confianza AL MOMENTO de la encuesta, no la de un año
    anterior como las tres fuentes censales.

    El código de columna AP5_4_NN se reasigna a una institución DISTINTA cada
    año (verificado: Jueces es AP5_4_11 en 2020 pero AP5_4_10 en 2021-2024;
    el texto de MP/Fiscalías Estatales también cambia de "MP, Procuradurías"
    a "Ministerio Público (MP) y Fiscalías Estatales" en 2022) — se resuelve
    siempre por texto en diccionario_de_datos, nunca por código ni posición.
    """
    zpath = _zip_envipe(anio)
    with zipfile.ZipFile(zpath) as z:
        dic = _leer_miembro(z, _miembro_tper_vic1(z, "diccionario_de_datos"))
        nemonico = dic["NEMONICO"].astype(str).str.strip()
        # Restringir al bloque AP5_4 ("Confianza en..."): AP5_3 ("Identifica a...")
        # y AP5_6 ("Percepción sobre desempeño de...") repiten los mismos nombres
        # de institución y darían falsos positivos si se buscara solo por texto.
        dic = dic[nemonico.str.match(r"^AP5_4_\d+$", na=False)]
        nombre = dic["NOMBRE_CAMPO"].astype(str).str.lower()

        pol_col = dic[nombre.str.contains("policía estatal", na=False)]["NEMONICO"].iloc[0]
        jue_col = dic[nombre.str.contains("jueces", na=False)]["NEMONICO"].iloc[0]
        mp_col = dic[(nombre.str.contains("procuradur", na=False) | nombre.str.contains("fiscal", na=False))
                     & ~nombre.str.contains("general de la rep", na=False)]["NEMONICO"].iloc[0]

        cols = ["CVE_ENT", "RESUL_H", "FAC_ELE", pol_col, jue_col, mp_col]
        data = _leer_miembro(z, _miembro_tper_vic1(z, "conjunto_de_datos"), usecols=cols)

    # Ediciones 2020-2021: cada valor de campo trae un '\n' pegado como parte
    # del contenido citado (p. ej. RESUL_H llega como "A\n") — artefacto del
    # export original, no un problema de delimitador de fila (las filas y
    # columnas ya se parsean bien). Sin este strip, '=="A"' y el cast numérico
    # de las columnas AP5_4 fallan en silencio (todo termina en NaN).
    for col in cols:
        if data[col].dtype == object or str(data[col].dtype).startswith("str"):
            data[col] = data[col].str.strip()

    data = data[data["RESUL_H"] == "A"].copy()
    data["FAC_ELE"] = pd.to_numeric(data["FAC_ELE"], errors="coerce")
    data["CVE_ENT"] = pd.to_numeric(data["CVE_ENT"], errors="coerce")
    for col in (pol_col, jue_col, mp_col):
        data[col] = pd.to_numeric(data[col], errors="coerce")

    resultado = pl.DataFrame({"cve_ent": list(range(1, 33))})
    for col, salida in [(pol_col, "tasa_desconfianza_policia"),
                         (jue_col, "tasa_desconfianza_jueces"),
                         (mp_col, "tasa_desconfianza_mp")]:
        d = data[data[col].isin([1, 2, 3, 4])].copy()
        d["ponderada"] = d[col].isin([3, 4]).astype(float) * d["FAC_ELE"]
        num = d.groupby("CVE_ENT")["ponderada"].sum()
        den = d.groupby("CVE_ENT")["FAC_ELE"].sum()
        tasa = num / den
        tabla = pl.DataFrame({"cve_ent": [int(v) for v in tasa.index], salida: tasa.to_list()},
                              schema={"cve_ent": pl.Int64, salida: pl.Float64})
        resultado = resultado.join(tabla, on="cve_ent", how="left")

    return resultado.with_columns(pl.lit(anio).alias("anio"))


# ---------------------------------------------------------------- índice estatal

def construir_indice_estatal() -> pl.DataFrame:
    pob = cargar_poblacion()
    fuerza = fuerza_estatal(pob).filter(pl.col("año").is_between(ANIO_MIN, ANIO_MAX)) \
        .rename({"año": "anio", "pc": "capacidad_policial_pc"}) \
        .select("cve_ent", "anio", "capacidad_policial_pc")

    cnpje = pl.concat([cargar_cnpje(a) for a in range(ANIO_MIN, ANIO_MAX + 1)])
    cndhe = pl.concat([cargar_cndhe(a) for a in range(ANIO_MIN, ANIO_MAX + 1)])
    cnsipee = pl.concat([cargar_cnsipee(a) for a in range(ANIO_MIN, ANIO_MAX + 1)])
    envipe = pl.concat([cargar_envipe_desconfianza(a) for a in range(ANIO_MIN, ANIO_MAX + 1)])

    df = (fuerza.join(cnpje, on=["cve_ent", "anio"], how="inner")
          .join(cndhe, on=["cve_ent", "anio"], how="inner")
          .join(cnsipee, on=["cve_ent", "anio"], how="inner")
          .join(envipe, on=["cve_ent", "anio"], how="left")
          .with_columns(pl.col("cve_ent").replace_strict({k: v[0] for k, v in ESTADOS.items()})
                        .alias("entidad")))

    for raw_col, pct_col, invertir in [
        ("capacidad_policial_pc", "pct_capacidad", True),
        ("tasa_judicializacion", "pct_judicializacion", True),
        ("tasa_recomendacion", "pct_ddhh", True),
        ("pct_sin_sentencia", "pct_prision_preventiva", False),
    ]:
        rank = pl.col(raw_col).rank(method="average").over("anio") / pl.len().over("anio") * 100
        df = df.with_columns((100 - rank if invertir else rank).alias(pct_col))

    df = df.with_columns(
        sum(pl.col(c) * w for c, w in PESOS.items()).alias("indice_impunidad")
    ).with_columns(
        pl.col("indice_impunidad").qcut([0.25, 0.5, 0.75], labels=NIVELES).over("anio").alias("nivel")
    )

    return df.select(
        "cve_ent", "entidad", "anio", "capacidad_policial_pc", "tasa_judicializacion",
        "tasa_recomendacion", "pct_sin_sentencia", "pct_capacidad", "pct_judicializacion",
        "pct_ddhh", "pct_prision_preventiva", "indice_impunidad", "nivel",
        "tasa_desconfianza_policia", "tasa_desconfianza_jueces", "tasa_desconfianza_mp",
    ).sort(["anio", "indice_impunidad"], descending=[False, True])


# ---------------------------------------------------------------- capa municipal (parcial)

def construir_policia_municipal_cngmd() -> pl.DataFrame:
    """Policía municipal, CNGMD edición 2025 (dato 2024) — corte único."""
    z = RAIZ / "data/inegi/cngmd/2025/datosabiertos_conjunto_de_datos_rec_huma_sp_cngmd2025_csv.zip"
    mun = (leer_zip_csv(z, "m3s1p22")
           .with_columns(pl.col("cvegeo").str.zfill(5).cast(pl.Int64).alias("cve_mun"),
                         pl.col("sexostt").cast(pl.Float64, strict=False))
           .group_by("cve_mun").agg(pl.sum("sexostt").alias("policia")))

    pob = (pl.read_parquet(RAIZ / "dashboard_data/conapo_pob_municipal.parquet")
           .filter(pl.col("AÑO") == 2024)
           .select(pl.col("CLAVE").alias("cve_mun"), pl.col("NOM_MUN").alias("municipio"),
                   pl.col("CLAVE_ENT").alias("cve_ent"), pl.col("POB_TOTAL").alias("poblacion")))

    df = (mun.join(pob, on="cve_mun", how="inner")
          .with_columns((pl.col("policia") / pl.col("poblacion") * 1e5).alias("policia_pc"))
          .with_columns((pl.col("policia_pc").rank(method="average") / pl.len() * 100).alias("percentil")))

    return df.with_columns(pl.lit(2024).alias("anio_dato")).select(
        "cve_mun", "municipio", "cve_ent", "anio_dato", "policia_pc", "percentil"
    ).sort("policia_pc", descending=True)


def construir_enfrentamientos_cngspspe_2019() -> pl.DataFrame:
    """Enfrentamientos y bajas, CNGSPSPE m2pb — único año disponible: 2019."""
    z = RAIZ / "data/inegi/cngspspe/2020/m2_enfrentamientos_cngspspe2020_csv.zip"
    df = (leer_zip_csv(z, "m2pb")
          .with_columns(
              pl.col("entidad_a").cast(pl.Int64, strict=False).alias("cve_ent"),
              pl.col("ubicageo_b").cast(pl.Int64, strict=False).alias("cve_mun"),
              pl.col("totalca1").cast(pl.Float64, strict=False),
              (pl.col("perads1").cast(pl.Float64, strict=False).fill_null(0)
               + pl.col("perads2").cast(pl.Float64, strict=False).fill_null(0)).alias("bajas_policia"),
              (pl.col("civarm1").cast(pl.Float64, strict=False).fill_null(0)
               + pl.col("civarm2").cast(pl.Float64, strict=False).fill_null(0)
               + pl.col("civarm3").cast(pl.Float64, strict=False).fill_null(0)).alias("bajas_civiles_armados"))
          .filter(pl.col("totalca1").is_not_null())
          .with_columns((pl.col("totalca1").rank(method="average") / pl.len() * 100).alias("percentil")))

    return df.with_columns(pl.lit(2019).alias("anio_dato")).select(
        "cve_mun", "cve_ent", "anio_dato", "totalca1", "bajas_policia",
        "bajas_civiles_armados", "percentil"
    ).sort("totalca1", descending=True)


# ---------------------------------------------------------------- validación

def validar(indice: pl.DataFrame) -> None:
    print("\nTasa de recomendación CNDHE — promedio simple entre los 32 estados por año")
    print("(no ponderado por volumen de expedientes; la referencia documentada ≈1.4% en")
    print("2024 es la tasa NACIONAL ponderada [suma de recomendaciones ÷ suma de expedientes],")
    print("una cifra distinta por construcción — aquí cada estado pesa igual, como en los")
    print("otros 3 pilares. Un estado con pocos expedientes puede mostrar una tasa extrema,")
    print("p. ej. Baja California Sur 2024 = 100% con solo 7 expedientes concluidos, todos")
    print("recomendación — ruido de denominador chico, no una señal de política real):")
    for anio in range(ANIO_MIN, ANIO_MAX + 1):
        d = indice.filter(pl.col("anio") == anio)
        print(f"  {anio}: promedio simple entre estados = {d['tasa_recomendacion'].mean() * 100:.2f}%")

    ultimo = ANIO_MAX
    d = indice.filter(pl.col("anio") == ultimo)
    print(f"\nDistribución de niveles ({ultimo}):")
    print(d["nivel"].value_counts().sort("nivel"))

    print(f"\nTop 10 estados con mayor impunidad ({ultimo}):")
    print(d.head(10).select("entidad", "capacidad_policial_pc", "tasa_judicializacion",
                             "tasa_recomendacion", "pct_sin_sentencia", "indice_impunidad", "nivel"))

    print(f"\nTop 10 estados con menor impunidad ({ultimo}):")
    print(d.tail(10).select("entidad", "capacidad_policial_pc", "tasa_judicializacion",
                             "tasa_recomendacion", "pct_sin_sentencia", "indice_impunidad", "nivel"))

    print("\nDesconfianza institucional (ENVIPE, contexto — no forma parte del índice):")
    print("correlación (Pearson) entre indice_impunidad y cada tasa_desconfianza_*, por año")
    print("(se espera signo positivo: más impunidad administrativa <-> más desconfianza")
    print("ciudadana; no se exige un umbral, solo revisar que el signo sea razonable):")
    for col in ("tasa_desconfianza_policia", "tasa_desconfianza_jueces", "tasa_desconfianza_mp"):
        for anio in range(ANIO_MIN, ANIO_MAX + 1):
            sub = indice.filter(pl.col("anio") == anio)
            corr = sub.select(pl.corr("indice_impunidad", col)).item()
            print(f"  {col} {anio}: r = {corr:.3f}" if corr is not None else f"  {col} {anio}: sin datos")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                         help="Carpeta donde escribir los parquet de salida "
                              "(este índice no es dato de dashboard, no tiene default).")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Construyendo Índice de Impunidad Estatal ({ANIO_MIN}-{ANIO_MAX})...")
    indice = construir_indice_estatal()
    out_a = args.output_dir / "impunidad_index_estatal.parquet"
    indice.write_parquet(out_a)
    print(f"Índice estatal: {indice.height:,} filas -> {out_a}")

    print("\nConstruyendo capa municipal parcial (policía CNGMD, dato 2024)...")
    policia_mun = construir_policia_municipal_cngmd()
    out_b = args.output_dir / "impunidad_municipal_policia_cngmd.parquet"
    policia_mun.write_parquet(out_b)
    print(f"Policía municipal: {policia_mun.height:,} filas -> {out_b}")

    print("\nConstruyendo capa municipal parcial (enfrentamientos CNGSPSPE, 2019)...")
    enfrent_mun = construir_enfrentamientos_cngspspe_2019()
    out_c = args.output_dir / "impunidad_municipal_enfrentamientos_2019.parquet"
    enfrent_mun.write_parquet(out_c)
    print(f"Enfrentamientos municipales: {enfrent_mun.height:,} filas -> {out_c}")

    validar(indice)


if __name__ == "__main__":
    main()
