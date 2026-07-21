"""Construye las tablas de presencia/intensidad/control de grupos criminales.

Salida en `informe_data/`:
  A. grupos_panel_ocved.parquet       municipio x anio x grupo   2007-2018
  B. grupos_snapshot_universal.parquet municipio x grupo          snapshot 2017-2022
  C. contexto_municipio_anio.parquet   municipio x anio           (sin grupo)
  D. contexto_estatal_anio.parquet     estado x anio              (sin grupo)
  +  grupos_crosswalk.csv              auditoría del vocabulario canónico

A y B NO se concatenan: cubren ventanas distintas y B no tiene dimensión
temporal real. C y D son CONTEXTO DEL TERRITORIO, no del grupo — unirlas a A/B
produce co-ocurrencia, nunca atribución de una actividad a un grupo.

Ver informe_data/README_grupos_criminales.md para el detalle de cada proxy.
"""

import sys
import unicodedata
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from normalizar_grupos import detectar_grupos  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
DATA = RAIZ / "data"
SALIDA = RAIZ / "informe_data"

# OCVED arranca en 2000, pero su propio overview documenta un ramp de cobertura
# noticiosa ~8.4x entre 2000-06 y 2007-18: los primeros años están subcontados y
# mezclarlos produce una "tendencia" que es artefacto del corpus de prensa.
ANIO_INICIO_OCVED = 2007

# Etiquetas que no son un actor nombrado (ver normalizar_grupos.SENTINELAS).
NO_ACTORES = {"SIN_REGISTRO", "NO_IDENTIFICADO", "OTRO"}


def _sin_acentos(s: pl.Expr) -> pl.Expr:
    """Normaliza texto de nombres geográficos para matching por nombre."""
    return (
        s.str.strip_chars()
        .str.to_uppercase()
        .str.replace_all("Á", "A")
        .str.replace_all("É", "E")
        .str.replace_all("Í", "I")
        .str.replace_all("Ó", "O")
        .str.replace_all("Ú", "U")
        .str.replace_all("Ñ", "N")
        .str.replace_all(r"\s+", " ")
    )


def _py_sin_acentos(t: str) -> str:
    if t is None:
        return ""
    d = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    return " ".join(d.upper().split())


# --------------------------------------------------------------------------
# Tabla A — panel OCVED
# --------------------------------------------------------------------------
def construir_tabla_a() -> pl.DataFrame:
    oc = pl.read_excel(DATA / "ocved/OCVED_2.0.xlsx", sheet_name="Criminals_v2 0")

    assert oc.height == 64_895, f"OCVED cambió de tamaño: {oc.height}"
    assert (oc["mun"] // 1000 == oc["state"]).all(), "invariante EEMMM rota"

    # month == 0 es centinela de fecha incompleta, no un mes real.
    oc = oc.filter((pl.col("month") != 0) & (pl.col("year") >= ANIO_INICIO_OCVED))

    oc = oc.with_columns(
        pl.col("state").cast(pl.Utf8).str.zfill(2).alias("cve_ent"),
        pl.col("mun").cast(pl.Utf8).str.zfill(5).alias("cve_mun"),
        pl.col("year").alias("anio"),
        pl.col("actor_main")
        .map_elements(lambda a: detectar_grupos(a)[0], return_dtype=pl.Utf8)
        .alias("grupo"),
    )

    # Denominador: cuántos eventos del municipio-año quedaron sin actor
    # nombrado. Se guarda aparte para poder decir "el grupo X explica el 40% de
    # los eventos ATRIBUIBLES", sin fingir que el resto no existe.
    tot = oc.group_by(["cve_mun", "anio"]).agg(
        pl.len().alias("eventos_totales"),
        pl.col("grupo").is_in(NO_ACTORES).sum().alias("eventos_sin_actor"),
    )

    nombrados = oc.filter(~pl.col("grupo").is_in(NO_ACTORES))

    panel = (
        nombrados.group_by(["cve_ent", "cve_mun", "anio", "grupo"])
        .agg(pl.len().alias("eventos"))
        .join(tot, on=["cve_mun", "anio"], how="left")
    )

    panel = panel.with_columns(
        pl.col("grupo").n_unique().over(["cve_mun", "anio"]).alias("n_grupos_mun_anio"),
        pl.col("eventos").sum().over(["cve_mun", "anio"]).alias("eventos_nombrados"),
    ).with_columns(
        # PROXY de control: hegemonía = un solo grupo con presencia detectada en
        # el municipio-año. No mide territorio ni dominio, solo cuántos actores
        # distintos aparecen en la prensa codificada.
        pl.when(pl.col("n_grupos_mun_anio") == 1)
        .then(pl.lit("hegemonico"))
        .otherwise(pl.lit("disputado"))
        .alias("control_proxy"),
        (pl.col("eventos") / pl.col("eventos_nombrados")).alias("share_grupo"),
        (pl.col("grupo") == "Huachicoleros").alias("es_huachicolero"),
    )

    return panel.sort(["cve_mun", "anio", "eventos"], descending=[False, False, True])


# --------------------------------------------------------------------------
# Tabla B — snapshot El Universal
# --------------------------------------------------------------------------
def construir_tabla_b() -> pl.DataFrame:
    un = pl.read_excel(DATA / "universal/narco/BaseCarteles.xlsx", sheet_name="BaseAbierta")

    assert un.height == 2_463, f"Universal cambió de tamaño: {un.height}"
    assert un["ClaveGeo"].n_unique() == 2_463, "ClaveGeo dejó de ser única"

    filas = []
    for geo, ent, cartel in zip(un["ClaveGeo"], un["ClaveEntidad"], un["Cartel"]):
        for g in detectar_grupos(cartel):
            filas.append(
                {"cve_mun": str(geo).zfill(5), "cve_ent": str(ent).zfill(2), "grupo": g}
            )
    df = pl.DataFrame(filas)

    # `Sin registro` (51.4%) es AUSENCIA DE DATO, no ausencia de cártel: 8
    # estados tienen 0% en blanco y Yucatán 96.2%. Es un mapa de cobertura
    # editorial. Se conserva la fila marcada; jamás se imputa como cero.
    df = df.with_columns(
        (pl.col("grupo") == "SIN_REGISTRO").alias("sin_registro"),
        pl.col("grupo")
        .is_in(NO_ACTORES)
        .not_()
        .sum()
        .over("cve_mun")
        .alias("n_grupos"),
    ).with_columns(
        pl.when(pl.col("n_grupos") == 0)
        .then(pl.lit(None, dtype=pl.Utf8))
        .when(pl.col("n_grupos") == 1)
        .then(pl.lit("hegemonico"))
        .otherwise(pl.lit("disputado"))
        .alias("control_proxy"),
    )

    return df.sort(["cve_mun", "grupo"])


# --------------------------------------------------------------------------
# Tabla C — contexto municipal
# --------------------------------------------------------------------------
DELITOS = {
    "Homicidio": "homicidio",
    "Feminicidio": "feminicidio",
    "Extorsión": "extorsion",
    "Secuestro": "secuestro",
    "Narcomenudeo": "narcomenudeo",
    "Robo": "robo",
}
MESES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]


def _incidencia() -> pl.DataFrame:
    lf = pl.scan_parquet(DATA / "incidencia_delictiva/incidencia_fuero_comun/incidencia_delictiva_fuero_comun.parquet")
    cols = lf.collect_schema().names()
    meses = [m for m in MESES if m in cols]

    # Se agrupa en `Tipo de delito` y NO en subtipo: los subtipos se renombran
    # entre 2015-16 y 2017+, así que sumar a nivel subtipo mezcla series.
    return (
        lf.filter(pl.col("Tipo de delito").is_in(list(DELITOS)) & (pl.col("Año") < 2026))
        .with_columns(pl.sum_horizontal([pl.col(m).fill_null(0) for m in meses]).alias("n"))
        .group_by(["Cve. Municipio", "Año", "Tipo de delito"])
        .agg(pl.col("n").sum())
        .collect()
        .with_columns(
            pl.col("Cve. Municipio").cast(pl.Utf8).str.zfill(5).alias("cve_mun"),
            pl.col("Año").alias("anio"),
            pl.col("Tipo de delito").replace_strict(DELITOS).alias("delito"),
        )
        .pivot(on="delito", index=["cve_mun", "anio"], values="n")
    )


def _incendios(catalogo: pl.DataFrame) -> tuple[pl.DataFrame, float]:
    """CONAFOR 2015-2024, resuelto por nombre de municipio dentro de estado.

    DOS TRAMPAS de `CVEGEO`, ninguna documentada en conafor/DATA_OVERVIEW.md
    (que instruye `CVEGEO.zfill(5)` y produce claves inexistentes como "00011"):

    1. CVEGEO está INVERTIDO respecto al formato INEGI: es MMM*100 + EE, no
       EE*1000 + MMM. Calvillo Ags -> 301 = mun 003 + ent 01 -> 01003.
       Morelia Mich -> 5316 = mun 053 + ent 16 -> 16053.
    2. En 2024 (y solo 2024) la codificación cambia otra vez y rompe hasta ese
       invariante en 3,073 filas (4.3% del archivo), sin regla derivable.

    Por eso se ignora `CVEGEO` y se resuelve por `CVE_ENT` + nombre, que es
    estable en todo el archivo.
    """
    # Solo las columnas necesarias: el archivo trae campos numéricos con dtypes
    # mixtos (p.ej. `Deteccion` mezcla int y float) que rompen el parseo.
    df = pl.read_csv(
        DATA / "conafor/incendios_forestales/estadisticasincendiosforestales2015-2024.csv",
        encoding="utf8-lossy",
        columns=["CVE_ENT", "Municipio", "anio"],
        schema_overrides={"CVE_ENT": pl.Int64, "anio": pl.Int64},
    ).with_columns(
        pl.col("CVE_ENT").cast(pl.Utf8).str.zfill(2).alias("cve_ent"),
        _sin_acentos(pl.col("Municipio")).alias("mun_norm"),
    )

    resuelto = df.join(catalogo, on=["cve_ent", "mun_norm"], how="left")
    tasa = resuelto["cve_mun"].is_not_null().mean()

    return (
        resuelto.filter(pl.col("cve_mun").is_not_null())
        .group_by(["cve_mun", "anio"])
        .agg(pl.len().alias("n_incendios")),
        tasa,
    )


def _tomas_clandestinas(catalogo: pl.DataFrame) -> tuple[pl.DataFrame, float]:
    """Cartocrítica 2008-2016. Resuelve nombre de municipio -> cve_mun.

    El archivo trae `Municipio` como texto ALL CAPS sin clave. El overview
    documenta 67.8% de match contra el catálogo nacional; aquí se resuelve
    DENTRO DE ESTADO (nombre + cve_ent), lo que reduce colisiones entre
    homónimos de estados distintos. Se devuelve la tasa de match para poder
    distinguir "cero tomas" de "municipio no resuelto".
    """
    sys.path.insert(0, str(RAIZ / "scripts/centralismo"))
    from comun import normalizar_estado

    df = pl.read_csv(
        DATA / "cartocritica/tomas_clandestinas_oleoductos.csv",
        encoding="utf8-lossy",  # el archivo trae BOM
    ).rename({"Year of Fecha": "anio", "Municipio": "mun_txt", "Estado": "est_txt"})

    df = df.filter(pl.col("mun_txt").is_not_null() & pl.col("est_txt").is_not_null())

    # "Distrito Federal" (pre-2016) lo resuelve normalizar_estado -> 09.
    df = df.with_columns(
        pl.col("est_txt")
        .map_elements(lambda e: normalizar_estado(e), return_dtype=pl.Int64)
        .alias("ent_num")
    ).filter(pl.col("ent_num").is_not_null())

    df = df.with_columns(
        pl.col("ent_num").cast(pl.Utf8).str.zfill(2).alias("cve_ent"),
        _sin_acentos(pl.col("mun_txt")).alias("mun_norm"),
    )

    resuelto = df.join(catalogo, on=["cve_ent", "mun_norm"], how="left")
    tasa = resuelto["cve_mun"].is_not_null().mean()

    tomas = (
        resuelto.filter(pl.col("cve_mun").is_not_null())
        .group_by(["cve_mun", "anio"])
        .agg(pl.len().alias("n_tomas"))
    )
    return tomas, tasa


def _catalogo_municipios() -> pl.DataFrame:
    """Catálogo nombre->cve_mun a partir de incidencia (cobertura nacional)."""
    lf = pl.scan_parquet(DATA / "incidencia_delictiva/incidencia_fuero_comun/incidencia_delictiva_fuero_comun.parquet")
    return (
        lf.select(["Cve. Municipio", "Municipio", "Clave_Ent"])
        .unique()
        .collect()
        .with_columns(
            pl.col("Cve. Municipio").cast(pl.Utf8).str.zfill(5).alias("cve_mun"),
            pl.col("Clave_Ent").cast(pl.Utf8).str.zfill(2).alias("cve_ent"),
            _sin_acentos(pl.col("Municipio")).alias("mun_norm"),
        )
        .select(["cve_mun", "cve_ent", "mun_norm"])
        .unique(subset=["cve_ent", "mun_norm"])
    )


def construir_tabla_c() -> tuple[pl.DataFrame, dict[str, float]]:
    catalogo = _catalogo_municipios()
    inc = _incidencia()
    inc_f, tasa_inc = _incendios(catalogo)
    tomas, tasa_tomas = _tomas_clandestinas(catalogo)

    ctx = inc.join(inc_f, on=["cve_mun", "anio"], how="full", coalesce=True).join(
        tomas, on=["cve_mun", "anio"], how="full", coalesce=True
    )
    return ctx.sort(["cve_mun", "anio"]), {"incendios": tasa_inc, "tomas": tasa_tomas}


# --------------------------------------------------------------------------
# Tabla D — contexto estatal
# --------------------------------------------------------------------------
def construir_tabla_d() -> pl.DataFrame:
    # ENVIPE es representativa a nivel ESTATAL, no municipal — por eso vive en
    # esta tabla y no en la C. Se reusa el panel ya procesado (con factores de
    # expansión aplicados) en vez de reprocesar 176 MB de microdatos.
    env = pl.read_parquet(DATA / "inegi/envipe/envipe_state_panel.parquet")
    env = env.rename({c: c.lower() for c in env.columns})
    col_ent = next(c for c in env.columns if "ent" in c)
    col_anio = next(c for c in env.columns if c in ("anio", "año", "year"))
    env = env.with_columns(
        pl.col(col_ent).cast(pl.Int64).cast(pl.Utf8).str.zfill(2).alias("cve_ent"),
        pl.col(col_anio).cast(pl.Int64).alias("anio"),
    ).select(["cve_ent", "anio", "vic_envipe", "inseg_envipe"])

    fed = pl.read_excel(
        DATA
        / "incidencia_delictiva/incidencia_fuero_federal"
        / "Incidencia_del_fuero_federal_2012-2026_mar2026.xlsx"
    )
    meses = [m.upper() for m in MESES]
    cols = fed.columns
    meses = [m for m in cols if m.upper() in meses]
    fed = (
        # INEGI nulo = filas "EXTRANJERO" (3%); 2026 llega solo a marzo.
        fed.filter(pl.col("INEGI").is_not_null() & (pl.col("AÑO") < 2026))
        .with_columns(
            pl.sum_horizontal([pl.col(m).fill_null(0) for m in meses]).alias("n"),
            pl.col("INEGI").cast(pl.Int64).cast(pl.Utf8).str.zfill(2).alias("cve_ent"),
            pl.col("AÑO").alias("anio"),
        )
    )

    por_concepto = (
        fed.group_by(["cve_ent", "anio", "CONCEPTO"])
        .agg(pl.col("n").sum())
        .pivot(on="CONCEPTO", index=["cve_ent", "anio"], values="n")
    )

    # Robo de hidrocarburos vive a nivel TIPO, no CONCEPTO. Es el único proxy
    # de huachicol que continúa después de 2016, donde termina Cartocrítica —
    # pero es ESTATAL y cuenta carpetas de investigación federales, no tomas
    # detectadas: las dos series miden cosas distintas y no son empalmables.
    #
    # La L.F.P.S.D.C.M.H. se promulgó en 2016: antes de 2017 la serie es cero
    # porque el tipo penal no existía, NO porque no hubiera robo. Se codifica
    # como nulo para que nadie lea un "arranque del huachicol en 2017" que es
    # puramente legislativo.
    hidro = (
        fed.filter(pl.col("TIPO").str.contains("HIDROCARBUROS"))
        .group_by(["cve_ent", "anio"])
        .agg(pl.col("n").sum().alias("delitos_hidrocarburos"))
        .with_columns(
            pl.when(pl.col("anio") >= 2017)
            .then(pl.col("delitos_hidrocarburos"))
            .otherwise(None)
            .alias("delitos_hidrocarburos")
        )
    )

    return (
        env.join(por_concepto, on=["cve_ent", "anio"], how="full", coalesce=True)
        .join(hidro, on=["cve_ent", "anio"], how="left")
        .sort(["cve_ent", "anio"])
    )


# --------------------------------------------------------------------------
def exportar_crosswalk() -> pl.DataFrame:
    """Artefacto auditable: cadena cruda -> grupos canónicos, por fuente."""
    filas = []
    oc = pl.read_excel(DATA / "ocved/OCVED_2.0.xlsx", sheet_name="Criminals_v2 0")
    for v in sorted(oc["actor_main"].unique().to_list()):
        filas.append({"fuente": "ocved", "nombre_raw": v, "grupos": "|".join(detectar_grupos(v))})
    un = pl.read_excel(DATA / "universal/narco/BaseCarteles.xlsx", sheet_name="BaseAbierta")
    for v in sorted(un["Cartel"].unique().to_list()):
        filas.append({"fuente": "universal", "nombre_raw": v, "grupos": "|".join(detectar_grupos(v))})
    return pl.DataFrame(filas)


def main() -> None:
    SALIDA.mkdir(exist_ok=True)

    print("=" * 70)
    a = construir_tabla_a()
    a.write_parquet(SALIDA / "grupos_panel_ocved.parquet")
    print(f"A  grupos_panel_ocved       {a.height:>7,} filas")
    print(f"   años {a['anio'].min()}-{a['anio'].max()}  "
          f"municipios {a['cve_mun'].n_unique():,}  grupos {a['grupo'].n_unique()}")
    print(f"   eventos nombrados totales: {a['eventos'].sum():,}")

    b = construir_tabla_b()
    b.write_parquet(SALIDA / "grupos_snapshot_universal.parquet")
    disputados = b.filter(pl.col("control_proxy") == "disputado")["cve_mun"].n_unique()
    print(f"\nB  grupos_snapshot_universal {b.height:>7,} filas")
    print(f"   municipios {b['cve_mun'].n_unique():,}  "
          f"sin_registro {b.filter('sin_registro').height:,}  disputados {disputados:,}")
    top = (
        b.filter(~pl.col("grupo").is_in(NO_ACTORES) & (pl.col("grupo") != "BANDA_LOCAL"))
        .group_by("grupo")
        .agg(pl.col("cve_mun").n_unique().alias("n"))
        .sort("n", descending=True)
        .head(4)
    )
    print(f"   top grupos: {dict(zip(top['grupo'], top['n']))}")

    c, tasas = construir_tabla_c()
    c.write_parquet(SALIDA / "contexto_municipio_anio.parquet")
    print(f"\nC  contexto_municipio_anio  {c.height:>7,} filas")
    print(f"   años {c['anio'].min()}-{c['anio'].max()}  municipios {c['cve_mun'].n_unique():,}")
    print(f"   match municipal — incendios {tasas['incendios']:.1%}  "
          f"tomas {tasas['tomas']:.1%} (baseline documentado 67.8%)")
    print(f"   total tomas resueltas: {c['n_tomas'].sum():,.0f}  "
          f"incendios: {c['n_incendios'].sum():,.0f}")

    d = construir_tabla_d()
    d.write_parquet(SALIDA / "contexto_estatal_anio.parquet")
    print(f"\nD  contexto_estatal_anio    {d.height:>7,} filas")
    print(f"   años {d['anio'].min()}-{d['anio'].max()}  columnas: {d.columns}")

    cw = exportar_crosswalk()
    cw.write_csv(SALIDA / "grupos_crosswalk.csv")
    print(f"\n   grupos_crosswalk.csv     {cw.height:>7,} filas")
    print("=" * 70)


if __name__ == "__main__":
    main()
