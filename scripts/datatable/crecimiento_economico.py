"""
Crecimiento económico por entidad: PIBE y recaudación imputada.

Fuentes:
  PIBE — data/inegi/pibe/tabulados/PIBE_2.xlsx vía comun.leer_pibe(PIBE_TOTAL,
  bloque="Millones de pesos a precios de 2018"): pesos CONSTANTES 2018, serie
  anual 2003-2024 sin huecos, 32 entidades.

  Recaudación imputada — informe_data/recaudacion_imputada_estatal.parquet
  (scripts/centralismo/preparar_recaudacion_imputada.py): recaudación SAT
  nacional repartida por entidad con la metodología de centralismo, pesos
  CORRIENTES (nominal, sin deflactar), serie anual 2018-2024 únicamente, y
  sólo conceptos tributarios (isr, iva, ieps_*, comercio_exterior,
  hidrocarburos, isan, accesorios_otros — las 3 cuotas IMSS/INFONAVIT/ISSTE
  no traen monto imputado).

Las dos series cubren rangos de años distintos y una es real (PIBE) y la otra
nominal (recaudación): NO son comparables en magnitud directa. Se muestran
una junto a la otra para contexto, no para comparar tasas de crecimiento.

Sin --entidad ni --regiones, muestra la tabla nacional de las 32 entidades
para un --año (default 2024, el último año con ambas fuentes disponibles).

--regiones K reusa el parquet que guarda scripts/datatable/crecimiento_poblacional.py
--clusters K --año Y --guardar (dashboard_data/crecimiento_clusters_<Y>_<K>.parquet)
para agregar PIBE y recaudación por región de estados vecinos en vez de por
entidad individual, y muestra la trayectoria completa por región (no una foto
de un año — --año no aplica aquí). --regiones-año (default 2020) debe
coincidir con el --año usado al generar ese parquet.

--periodo X (default 1) reduce las tablas de --entidad y --regiones a los
años múltiplos de X (--periodo 1 muestra todos los años, como hasta ahora).
El % de cambio se recalcula sobre el hueco real entre los años mostrados.

Run: uv run python scripts/datatable/crecimiento_economico.py --entidad Jalisco
     uv run python scripts/datatable/crecimiento_economico.py
     uv run python scripts/datatable/crecimiento_economico.py --año 2019
     uv run python scripts/datatable/crecimiento_economico.py --regiones 5
     uv run python scripts/datatable/crecimiento_economico.py --entidad Jalisco --periodo 5
"""

import argparse
import sys
from pathlib import Path

import polars as pl

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "scripts" / "centralismo"))
from comun import NOMBRE, PIBE_TOTAL, leer_pibe, normalizar_estado

DIR = RAIZ / "dashboard_data"
RECAUDACION_PATH = RAIZ / "informe_data" / "recaudacion_imputada_estatal.parquet"


def _entidad(valor):
    """Acepta clave (1-32) o nombre en cualquier régimen ('cdmx', 'Distrito Federal')."""
    if valor.isdigit() and 1 <= int(valor) <= 32:
        return int(valor)
    cve = normalizar_estado(valor)
    if cve is None:
        raise argparse.ArgumentTypeError(
            f"entidad no reconocida: {valor!r} (usa el nombre o la clave 1-32)"
        )
    return cve


def _año(valor):
    try:
        return int(valor)
    except ValueError:
        raise argparse.ArgumentTypeError(f"año inválido: {valor!r}")


def _k(valor):
    try:
        n = int(valor)
    except ValueError:
        raise argparse.ArgumentTypeError(f"clusters inválido: {valor!r}")
    if not (1 <= n <= 32):
        raise argparse.ArgumentTypeError(f"clusters inválido: {n} (debe estar entre 1 y 32)")
    return n


def _periodo(valor):
    try:
        n = int(valor)
    except ValueError:
        raise argparse.ArgumentTypeError(f"periodo inválido: {valor!r} (usa un entero, p. ej. 5)")
    if n < 1:
        raise argparse.ArgumentTypeError(f"periodo inválido: {n} (debe ser >= 1)")
    return n


def _fmt(v, decimales=0, signo=False, sufijo=""):
    if v is None:
        return "—"
    return f"{v:{'+' if signo else ''},.{decimales}f}{sufijo}"


def _entidades_df() -> pl.DataFrame:
    return pl.DataFrame({"cve_ent": list(NOMBRE.keys()), "entidad": list(NOMBRE.values())}
                        ).with_columns(pl.col("cve_ent").cast(pl.Int16))


def leer_pib() -> pl.DataFrame:
    return leer_pibe(PIBE_TOTAL, bloque="Millones de pesos a precios de 2018").rename({"valor": "pib"})


def leer_recaudacion() -> pl.DataFrame:
    if not RECAUDACION_PATH.exists():
        raise SystemExit(f"Falta {RECAUDACION_PATH}.\n"
                         "Corre primero: uv run python scripts/centralismo/preparar_recaudacion_imputada.py")
    return (
        pl.read_parquet(RECAUDACION_PATH)
        .filter(pl.col("monto_imputado_millones_pesos").is_not_null())
        .group_by("anio", "cve_ent")
        .agg(pl.sum("monto_imputado_millones_pesos").alias("recaudacion"))
        .rename({"anio": "año"})
        .with_columns(pl.col("cve_ent").cast(pl.Int16), pl.col("año").cast(pl.Int32))
    )


def combinar_economico() -> pl.DataFrame:
    """PIB (2003-2024, constante) + recaudación imputada (2018-2024, nominal),
    unidos por (cve_ent, año). Fuera de la ventana 2018-2024, recaudación
    queda null (no hay dato, no es cero)."""
    df = leer_pib().join(leer_recaudacion(), on=["cve_ent", "año"], how="full", coalesce=True)
    return df.join(_entidades_df(), on="cve_ent").sort(["cve_ent", "año"])


def _con_cambios(sub: pl.DataFrame) -> list:
    """A un DataFrame ya ordenado por año (y ya filtrado a los años a
    mostrar), agrega pib_cambio_pct/recaudacion_cambio_pct: % de cambio
    anualizado sobre el hueco real entre años CON DATO (TCMA) — con años
    consecutivos (--periodo 1) el hueco siempre es 1, así que coincide con el
    cambio simple año contra año. pib y recaudacion se rastrean por separado
    porque recaudación sólo tiene dato a partir de 2018."""
    filas = []
    año_prev_pib = pib_prev = None
    año_prev_rec = rec_prev = None
    for r in sub.iter_rows(named=True):
        fila = dict(r)
        fila["pib_cambio_pct"] = None
        fila["recaudacion_cambio_pct"] = None
        if r["pib"] is not None and pib_prev is not None:
            gap = r["año"] - año_prev_pib
            fila["pib_cambio_pct"] = ((r["pib"] / pib_prev) ** (1 / gap) - 1) * 100
        if r["recaudacion"] is not None and rec_prev is not None:
            gap = r["año"] - año_prev_rec
            fila["recaudacion_cambio_pct"] = ((r["recaudacion"] / rec_prev) ** (1 / gap) - 1) * 100
        filas.append(fila)
        if r["pib"] is not None:
            año_prev_pib, pib_prev = r["año"], r["pib"]
        if r["recaudacion"] is not None:
            año_prev_rec, rec_prev = r["año"], r["recaudacion"]
    return filas


def calcular(df, cve, periodo=1) -> pl.DataFrame:
    """Serie de una entidad, mostrando sólo los años múltiplos de `periodo`
    (periodo=1: todos los años, el comportamiento por default)."""
    sub = df.filter((pl.col("cve_ent") == cve) & (pl.col("año") % periodo == 0)).sort("año")
    return pl.DataFrame(_con_cambios(sub))


def tabla_nacional(df, año):
    actual = df.filter(pl.col("año") == año)
    total_pib = actual["pib"].sum()
    total_rec = actual["recaudacion"].sum()
    prev = df.filter(pl.col("año") == año - 1).select(
        ["cve_ent", pl.col("pib").alias("pib_prev"), pl.col("recaudacion").alias("recaudacion_prev")])
    tabla = (
        actual.join(prev, on="cve_ent", how="left")
        .with_columns(
            (pl.col("pib") / total_pib * 100).alias("pib_share_pct"),
            ((pl.col("pib") / pl.col("pib_prev") - 1) * 100).alias("pib_cambio_pct"),
            (pl.col("recaudacion") / total_rec * 100).alias("recaudacion_share_pct"),
            ((pl.col("recaudacion") / pl.col("recaudacion_prev") - 1) * 100).alias("recaudacion_cambio_pct"),
        )
        .sort("pib", descending=True)
    )
    return tabla, total_pib, total_rec


def regiones(df, k, año_regiones, periodo=1):
    """Trayectoria completa (todos los años disponibles — 2003-2024 para PIB,
    2018-2024 para recaudación) de PIB y recaudación por región, reusando el
    parquet ya guardado por crecimiento_poblacional.py --clusters. Sin
    restringir a un solo --año. `periodo`: sólo años múltiplos de periodo
    (periodo=1: todos)."""
    ruta = DIR / f"crecimiento_clusters_{año_regiones}_{k}.parquet"
    if not ruta.exists():
        raise SystemExit(
            f"Falta {ruta}.\nCorre primero: uv run python "
            f"scripts/datatable/crecimiento_poblacional.py --clusters {k} "
            f"--año {año_regiones} --guardar"
        )
    mapa = pl.read_parquet(ruta).select(["cluster_id", "cve_ent"])
    totales = df.group_by("año").agg(
        pl.sum("pib").alias("pib_total"), pl.sum("recaudacion").alias("recaudacion_total"))

    # sum() de un grupo enteramente null da 0.0 en polars, no null — para años
    # anteriores a 2018 (sin recaudación) hay que forzar null explícitamente,
    # no un total falso de $0.
    agregado = (
        df.join(mapa, on="cve_ent", how="inner")
        .group_by("cluster_id", "año")
        .agg(
            pl.sum("pib").alias("pib"),
            pl.when(pl.col("recaudacion").null_count() == pl.len()).then(None)
              .otherwise(pl.sum("recaudacion")).alias("recaudacion"),
        )
        .join(totales, on="año")
        .with_columns(
            (pl.col("pib") / pl.col("pib_total") * 100).alias("pib_share_pct"),
            (pl.col("recaudacion") / pl.col("recaudacion_total") * 100).alias("recaudacion_share_pct"),
        )
        .drop(["pib_total", "recaudacion_total"])
        .filter(pl.col("año") % periodo == 0)
        .sort(["cluster_id", "año"])
    )

    salida = []
    for cluster_id in dict.fromkeys(agregado["cluster_id"].to_list()):
        salida.extend(_con_cambios(agregado.filter(pl.col("cluster_id") == cluster_id)))
    tabla = pl.DataFrame(salida)

    entidades_por_cluster = (
        mapa.join(_entidades_df(), on="cve_ent")
        .group_by("cluster_id").agg(pl.col("entidad").sort().alias("entidades"))
    )
    return tabla, entidades_por_cluster


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entidad", type=_entidad, default=None,
                   help="Nombre o clave INEGI (1-32) de la entidad federativa. Si se omite, "
                        "se muestra la tabla nacional de las 32 entidades para --año")
    p.add_argument("--año", type=_año, default=2024,
                   help="Año a usar en la tabla nacional (sin --entidad ni --regiones); "
                        "default 2024, el último año con PIBE y recaudación disponibles. "
                        "--regiones ignora --año y siempre muestra la serie completa")
    p.add_argument("--regiones", type=_k, default=None,
                   help="Agrega PIB y recaudación por región de N estados vecinos, "
                        "reusando dashboard_data/crecimiento_clusters_<regiones-año>_<N>.parquet "
                        "(generado por crecimiento_poblacional.py --clusters). Incompatible "
                        "con --entidad")
    p.add_argument("--regiones-año", dest="regiones_año", type=_año, default=2020,
                   help="Año usado al generar el parquet de --clusters con "
                        "crecimiento_poblacional.py (default 2020)")
    p.add_argument("--periodo", type=_periodo, default=1,
                   help="Salto en años para las tablas de --entidad y --regiones (default 1: "
                        "todos los años, como hasta ahora). --periodo 5 sólo muestra los años "
                        "múltiplos de 5. No aplica a la tabla nacional (--año, un solo año)")
    p.add_argument("--guardar", action="store_true",
                   help="Escribe el resultado a dashboard_data/crecimiento_economico_*.parquet")
    a = p.parse_args()

    if a.entidad is not None and a.regiones is not None:
        p.error("--entidad es incompatible con --regiones (es un reporte por región)")

    df = combinar_economico()

    if a.regiones is not None:
        tabla, entidades_por_cluster = regiones(df, a.regiones, a.regiones_año, a.periodo)
        nombres = dict(zip(entidades_por_cluster["cluster_id"].to_list(),
                           entidades_por_cluster["entidades"].to_list()))

        print(f"\n{'═' * 88}")
        print(f"  Crecimiento económico por región ({a.regiones} clusters de {a.regiones_año})")
        print("═" * 88)
        for cluster_id in dict.fromkeys(tabla["cluster_id"].to_list()):
            sub = tabla.filter(pl.col("cluster_id") == cluster_id)
            print(f"\n  Cluster {cluster_id}: {', '.join(nombres[cluster_id])}")
            print(f"  {'Año':<6} {'PIB (M)':>13} {'Share':>7} {'Δ PIB %':>9}"
                  f" {'Recaudación (M)':>16} {'Share':>7} {'Δ Recaud. %':>12}")
            for r in sub.iter_rows(named=True):
                print(f"  {r['año']:<6} {_fmt(r['pib']):>13} {_fmt(r['pib_share_pct'], 1, sufijo='%'):>7}"
                      f" {_fmt(r['pib_cambio_pct'], 1, signo=True, sufijo='%'):>9}"
                      f" {_fmt(r['recaudacion']):>16} {_fmt(r['recaudacion_share_pct'], 1, sufijo='%'):>7}"
                      f" {_fmt(r['recaudacion_cambio_pct'], 1, signo=True, sufijo='%'):>12}")
        print("\nNota: PIB a precios constantes 2018 (2003-2024); recaudación a precios")
        print("corrientes/nominal, sólo disponible 2018-2024 (celdas anteriores en '—').\n")

        if a.guardar:
            salida = tabla.join(entidades_por_cluster, on="cluster_id").with_columns(
                pl.col("entidades").list.join(", "))
            ruta = DIR / f"crecimiento_economico_regiones_{a.regiones}.parquet"
            salida.write_parquet(ruta)
            print(f"Saved → {ruta}  ({salida.height:,} filas)\n")
        return

    if a.entidad is None:
        tabla, total_pib, total_rec = tabla_nacional(df, a.año)

        print(f"\n{'═' * 88}")
        print(f"  Crecimiento económico por entidad — {a.año} (vs. {a.año - 1})")
        print("═" * 88)
        print(f"  {'Entidad':<24} {'PIB (M)':>14} {'Share':>7} {'Δ PIB %':>9}"
              f" {'Recaudación (M)':>16} {'Share':>7} {'Δ Recaud. %':>12}")
        for r in tabla.iter_rows(named=True):
            print(f"  {r['entidad']:<24} {_fmt(r['pib']):>14} {_fmt(r['pib_share_pct'], 1, sufijo='%'):>7}"
                  f" {_fmt(r['pib_cambio_pct'], 1, signo=True, sufijo='%'):>9}"
                  f" {_fmt(r['recaudacion']):>16} {_fmt(r['recaudacion_share_pct'], 1, sufijo='%'):>7}"
                  f" {_fmt(r['recaudacion_cambio_pct'], 1, signo=True, sufijo='%'):>12}")
        print(f"  {'TOTAL NACIONAL':<24} {_fmt(total_pib):>14} {'100.0%':>7} {'':>9}"
              f" {_fmt(total_rec):>16} {'100.0%':>7}\n")
        print("Nota: PIB a precios constantes 2018; recaudación a precios corrientes (nominal).\n")

        if a.guardar:
            ruta = DIR / f"crecimiento_economico_nacional_{a.año}.parquet"
            tabla.write_parquet(ruta)
            print(f"Saved → {ruta}  ({tabla.height:,} filas)\n")
        return

    tabla = calcular(df, a.entidad, a.periodo)
    print(f"\n{'═' * 78}")
    print(f"  {NOMBRE[a.entidad]} — PIB y recaudación imputada")
    print("═" * 78)
    print(f"  {'Año':<6} {'PIB (M, 2018)':>15} {'Δ PIB %':>9} {'Recaudación (M, nominal)':>26} {'Δ Recaud. %':>12}")
    for r in tabla.iter_rows(named=True):
        print(f"  {r['año']:<6} {_fmt(r['pib']):>15} {_fmt(r['pib_cambio_pct'], 1, signo=True, sufijo='%'):>9}"
              f" {_fmt(r['recaudacion']):>26} {_fmt(r['recaudacion_cambio_pct'], 1, signo=True, sufijo='%'):>12}")
    print("\nNota: PIB a precios constantes 2018 (2003-2024); recaudación a precios")
    print("corrientes/nominal, sólo disponible 2018-2024 (celdas anteriores en '—').\n")

    if a.guardar:
        salida = tabla.select(["cve_ent", "entidad", "año", "pib", "pib_cambio_pct",
                               "recaudacion", "recaudacion_cambio_pct"])
        ruta = DIR / f"crecimiento_economico_{NOMBRE[a.entidad].lower().replace(' ', '_')}.parquet"
        salida.write_parquet(ruta)
        print(f"Saved → {ruta}  ({salida.height:,} filas)\n")


if __name__ == "__main__":
    main()
