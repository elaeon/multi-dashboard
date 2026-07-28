"""
Migración interestatal acumulada por LUGAR DE NACIMIENTO, censos 1950–2020.

Mide el stock de toda la vida: cuántos residentes de una entidad nacieron en otra, y
cuántos nacidos en ella residen hoy en otra. NO es comparable con
scripts/datatable/movilidad_estatal.py, que mide el flujo del quinquenio previo — en
2020, 3.8 M de personas cambiaron de entidad en 5 años, pero 21.6 M residen en una
entidad distinta a la de su nacimiento.

Fuente: dashboard_data/ccpv_nacimiento_estatal.parquet
        (generada por scripts/prepare_ccpv_nacimiento.py)

Run: uv run python scripts/datatable/migracion_nacimiento.py --entidad Jalisco --año 1970
     uv run python scripts/datatable/migracion_nacimiento.py --entidad cdmx --serie
     uv run python scripts/datatable/migracion_nacimiento.py --entidad Jalisco --año 1950-1980
"""

import argparse
import sys
from pathlib import Path

import polars as pl

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(RAIZ / "scripts"))
sys.path.insert(0, str(RAIZ / "scripts" / "centralismo"))
from comun import NOMBRE
from movilidad_estatal import _entidad, _tabla
from prepare_ccpv_nacimiento import TERRITORIOS

DIR = RAIZ / "dashboard_data"
CENSOS = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]


def _año_o_rango(valor):
    """Un censo suelto ('1970') o un rango inclusivo ('1950-1980') → int o tupla."""
    if "-" in valor:
        partes = valor.split("-")
        if len(partes) != 2:
            raise argparse.ArgumentTypeError(f"rango inválido: {valor!r} (usa 'AAAA-AAAA')")
        try:
            lo, hi = sorted(int(p) for p in partes)
        except ValueError:
            raise argparse.ArgumentTypeError(f"rango inválido: {valor!r} (usa 'AAAA-AAAA')")
        censos = tuple(c for c in CENSOS if lo <= c <= hi)
        if not censos:
            raise argparse.ArgumentTypeError(
                f"ningún censo cae en el rango {lo}-{hi} (censos disponibles: {CENSOS})")
        return censos
    try:
        censo = int(valor)
    except ValueError:
        raise argparse.ArgumentTypeError(f"año inválido: {valor!r}")
    if censo not in CENSOS:
        raise argparse.ArgumentTypeError(f"censo no disponible: {censo} (elige entre {CENSOS})")
    return censo


def cargar():
    ruta = DIR / "ccpv_nacimiento_estatal.parquet"
    if not ruta.exists():
        raise SystemExit(f"Falta {ruta}.\n"
                         "Corre primero: uv run python scripts/prepare_ccpv_nacimiento.py")
    return pl.read_parquet(ruta)


def flujos(df, cve, censo):
    """(inmigrantes, emigrantes, población total) de una entidad en un censo."""
    g = df.filter(pl.col("censo") == censo)
    desglose = g.filter(~pl.col("total_categoria"))
    inm = (desglose.filter(pl.col("cve_destino") == cve)
           .select(["origen", "personas"]).sort("personas", descending=True))
    emi = (desglose.filter((pl.col("cve_origen") == cve) & (pl.col("cve_destino") > 0))
           .select(["destino", "personas"]).sort("personas", descending=True))
    poblacion = g.filter(pl.col("total_categoria") & (pl.col("cve_destino") == cve)
                         & (pl.col("categoria") == "Total"))["personas"].sum()
    return inm, emi, poblacion


def detalle(df, cve, censo, top):
    inm, emi, poblacion = flujos(df, cve, censo)
    n_inm, n_emi = inm["personas"].sum(), emi["personas"].sum()

    print(f"\n{'═' * 72}")
    print(f"  {NOMBRE[cve]} — Censo {censo}")
    print("  Migración interestatal acumulada, por lugar de nacimiento")
    print(f"  Población total: {poblacion:,}")
    if cve in TERRITORIOS.get(censo, ()):
        print("  Nota: en este censo aún era territorio federal, no estado.")
    print("═" * 72)

    _tabla("INMIGRANTES — residen aquí pero nacieron en otra entidad",
           inm, "origen", n_inm, top)
    _tabla("EMIGRANTES — nacieron aquí pero residen en otra entidad",
           emi, "destino", n_emi, top)

    neto = n_inm - n_emi
    print(f"\nSALDO NETO ACUMULADO: {neto:+,}"
          f"  ({neto / poblacion * 1000:+.2f} por mil habitantes)")
    print(f"Residentes nacidos en otra entidad: {n_inm / poblacion * 100:.2f}% de la población")

    print(f"\n{'─' * 72}")
    otras = (df.filter((pl.col("censo") == censo) & (pl.col("cve_destino") == cve)
                       & pl.col("total_categoria")
                       & ~pl.col("categoria").is_in(["Total", "En la entidad", "En otra entidad"]))
             .sort("personas", descending=True))
    for r in otras.iter_rows(named=True):
        print(f"  {r['categoria']:<44} {r['personas']:>12,}")
    print()
    return inm, emi


def acumulado(df, cve, censos, top):
    """Reporte de un rango de censos, usando el CENSO DESTINO (el más reciente
    del rango) como cifra reportada — es un stock, ya consistente con la
    población real de ese año, así que sumarlo con censos anteriores duplicaría
    personas que nunca cambiaron de entidad. Los censos anteriores del rango se
    muestran como trayectoria de contexto, sin sumarse.
    """
    censo_destino = censos[-1]
    print(f"\n{'═' * 72}")
    print(f"  Rango solicitado: {censos[0]}–{censos[-1]} → usando censo destino {censo_destino}")
    print("  (el stock de migración por nacimiento es acumulado a cada fecha censal;")
    print("   sumar varios censos duplicaría personas. Se reporta el más reciente del")
    print("   rango, ya consistente con la población real de ese año.)")

    inm, emi = detalle(df, cve, censo_destino, top)

    if len(censos) > 1:
        print(f"{'─' * 72}")
        print("Trayectoria dentro del rango (censos anteriores, no se suman):")
        print(f"  {'Censo':<7} {'Población':>13} {'Inmigrantes':>13} {'Emigrantes':>13}"
              f" {'Saldo':>13} {'% nac. fuera':>13}")
        for censo in censos:
            i, e, poblacion = flujos(df, cve, censo)
            n_i, n_e = i["personas"].sum(), e["personas"].sum()
            nota = " *" if cve in TERRITORIOS.get(censo, ()) else ""
            print(f"  {censo:<7} {poblacion:>13,} {n_i:>13,} {n_e:>13,}"
                  f" {n_i - n_e:>+13,} {n_i / poblacion * 100:>12.2f}%{nota}")
        if any(cve in TERRITORIOS.get(c, ()) for c in censos):
            print("  * territorio federal en ese censo, no estado")
        print()

    return inm, emi


def serie(df, cve):
    print(f"\n{'═' * 78}")
    print(f"  {NOMBRE[cve]} — migración interestatal acumulada, 1950–2020")
    print("═" * 78)
    print(f"  {'Censo':<7} {'Población':>13} {'Inmigrantes':>13} {'Emigrantes':>13}"
          f" {'Saldo':>13} {'% nac. fuera':>13}")
    for censo in CENSOS:
        inm, emi, poblacion = flujos(df, cve, censo)
        n_inm, n_emi = inm["personas"].sum(), emi["personas"].sum()
        nota = " *" if cve in TERRITORIOS.get(censo, ()) else ""
        print(f"  {censo:<7} {poblacion:>13,} {n_inm:>13,} {n_emi:>13,}"
              f" {n_inm - n_emi:>+13,} {n_inm / poblacion * 100:>12.2f}%{nota}")
    if any(cve in TERRITORIOS.get(c, ()) for c in CENSOS):
        print("\n  * territorio federal en ese censo, no estado")
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entidad", type=_entidad, required=True,
                   help="Nombre o clave INEGI (1-32) de la entidad federativa")
    p.add_argument("--año", type=_año_o_rango,
                   help="Año censal, o rango 'AAAA-AAAA' para el acumulado de varios "
                        "censos (obligatorio salvo con --serie)")
    p.add_argument("--top", type=int, default=10, help="Filas por tabla (default 10)")
    p.add_argument("--serie", action="store_true",
                   help="Trayectoria de la entidad en los 8 censos, en vez del detalle")
    p.add_argument("--guardar", action="store_true",
                   help="Escribe el resultado a dashboard_data/nacimiento_<entidad>_<año>.parquet")
    a = p.parse_args()
    if not a.serie and a.año is None:
        p.error("se requiere --año (o usa --serie)")

    df, cve = cargar(), a.entidad
    if a.serie:
        serie(df, cve)
        return

    if isinstance(a.año, tuple):
        inm, emi = acumulado(df, cve, list(a.año), a.top)
        etiqueta_año = f"{a.año[0]}-{a.año[-1]}"
    else:
        inm, emi = detalle(df, cve, a.año, a.top)
        etiqueta_año = str(a.año)

    if a.guardar:
        salida = pl.concat([
            inm.rename({"origen": "contraparte"}).with_columns(pl.lit("inmigracion").alias("flujo")),
            emi.rename({"destino": "contraparte"}).with_columns(pl.lit("emigracion").alias("flujo")),
        ]).with_columns(
            pl.lit(etiqueta_año).alias("censo"),
            pl.lit(cve).cast(pl.Int16).alias("cve_ent"),
            pl.lit(NOMBRE[cve]).alias("entidad"),
        ).select(["censo", "cve_ent", "entidad", "flujo", "contraparte", "personas"])
        ruta = DIR / f"nacimiento_{NOMBRE[cve].lower().replace(' ', '_')}_{etiqueta_año}.parquet"
        salida.write_parquet(ruta)
        print(f"Saved → {ruta}  ({salida.height:,} filas)\n")


if __name__ == "__main__":
    main()
