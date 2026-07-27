"""
Movilidad de personas entre entidades federativas y extranjeros residentes,
para una entidad y un año censal.

Reporta, según el censo elegido (1990, 2000, 2010 o 2020):
  · Inmigrantes — quiénes llegaron desde otra entidad en el quinquenio previo.
  · Emigrantes  — quiénes salieron hacia otra entidad en el mismo quinquenio.
  · Saldo neto migratorio interestatal y tasa por mil habitantes.
  · Extranjeros — flujo (residían en otro país hace 5 años, con desglose por país
    en 2020) y stock (nacidos en el extranjero).

La movilidad es la declarada en el censo (residencia actual vs. residencia 5 años
antes), no una estimación indirecta como la de scripts/prepare_migracion_interna.py.

Fuente: dashboard_data/ccpv_migracion_estatal.parquet y
        dashboard_data/ccpv_extranjeros_pais_2020.parquet
        (generadas por scripts/prepare_ccpv_migracion.py)

Run: uv run python scripts/datatable/movilidad_estatal.py --entidad Jalisco --año 2020
"""

import argparse
import sys
from pathlib import Path

import polars as pl

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "scripts" / "centralismo"))
from comun import NOMBRE, normalizar_estado

DIR = RAIZ / "dashboard_data"
CENSOS = [1990, 2000, 2010, 2020]
VENTANA = {1990: "1985 → 1990", 2000: "enero 1995 → 2000",
           2010: "junio 2005 → 2010", 2020: "marzo 2015 → marzo 2020"}
EXTRANJERO = ["En los Estados Unidos de América", "En otro país"]


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


def _tabla(titulo, df, col, total, top):
    """Imprime top-N filas de (col, personas) con su participación en el total."""
    print(f"\n{titulo}: {total:,}")
    if total == 0:
        return
    print(f"  {'#':>3}  {'':<34} {'Personas':>12} {'%':>7}")
    for i, r in enumerate(df.head(top).iter_rows(named=True), start=1):
        print(f"  {i:>3}  {r[col]:<34} {r['personas']:>12,} {r['personas']/total*100:>6.1f}%")
    resto = df.slice(top)
    if resto.height:
        suma = resto["personas"].sum()
        print(f"  {'':>3}  {f'Resto ({resto.height})':<34} {suma:>12,} {suma/total*100:>6.1f}%")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entidad", type=_entidad, required=True,
                   help="Nombre o clave INEGI (1-32) de la entidad federativa")
    p.add_argument("--año", type=int, required=True, choices=CENSOS,
                   help="Año censal")
    p.add_argument("--top", type=int, default=10, help="Filas por tabla (default 10)")
    p.add_argument("--guardar", action="store_true",
                   help="Escribe el resultado a dashboard_data/movilidad_<entidad>_<año>.parquet")
    a = p.parse_args()
    cve, censo, top = a.entidad, a.año, a.top

    mig = pl.read_parquet(DIR / "ccpv_migracion_estatal.parquet").filter(pl.col("censo") == censo)
    flujo = mig.filter(pl.col("concepto") == "residencia_5a")
    stock = mig.filter(pl.col("concepto") == "nacimiento")

    def agregado(df, categoria, cve_ent=cve):
        f = df.filter(pl.col("total_categoria") & (pl.col("cve_destino") == cve_ent)
                      & (pl.col("categoria") == categoria))
        return f["personas"].sum()

    poblacion = agregado(flujo, "Total")
    print(f"\n{'═' * 72}")
    print(f"  {NOMBRE[cve]} — Censo {censo}")
    print(f"  Movilidad interestatal declarada, {VENTANA[censo]}")
    print(f"  Población de 5 años y más: {poblacion:,}")
    print("═" * 72)

    desglose = flujo.filter(~pl.col("total_categoria"))

    # Inmigrantes: incluye la fila 'Entidad no especificada' (sólo 1990) para que el
    # total cuadre con el agregado publicado de 'En otra entidad'. Esos casos no son
    # atribuibles a un origen, así que no aparecen del lado de los emigrantes.
    inm = (desglose.filter(pl.col("cve_destino") == cve)
           .select(["origen", "personas"]).sort("personas", descending=True))
    emi = (desglose.filter((pl.col("cve_origen") == cve) & (pl.col("cve_destino") > 0))
           .select(["destino", "personas"]).sort("personas", descending=True))
    n_inm, n_emi = inm["personas"].sum(), emi["personas"].sum()
    assert n_inm == agregado(flujo, "En otra entidad")

    _tabla("INMIGRANTES — llegaron desde otra entidad", inm, "origen", n_inm, top)
    _tabla("EMIGRANTES — salieron hacia otra entidad", emi, "destino", n_emi, top)

    neto = n_inm - n_emi
    print(f"\nSALDO NETO INTERESTATAL: {neto:+,}"
          f"  ({neto / poblacion * 1000:+.2f} por mil habitantes)")

    # ── Extranjeros ──────────────────────────────────────────────────────────
    print(f"\n{'─' * 72}\nEXTRANJEROS")

    f_ext = {c: agregado(flujo, c) for c in EXTRANJERO}
    print(f"\n  Flujo — residían en otro país en {VENTANA[censo].split(' → ')[0]}: "
          f"{sum(f_ext.values()):,}")
    if censo >= 2010:  # 1990 y 2000 no separan Estados Unidos del resto
        for c, v in f_ext.items():
            print(f"      {c:<38} {v:>12,}")

    paises = pl.DataFrame(schema={"pais": pl.Utf8, "personas": pl.Int64})
    if censo == 2020:
        paises = (
            pl.read_parquet(DIR / "ccpv_extranjeros_pais_2020.parquet")
            .filter((pl.col("cve_destino") == cve) & (pl.col("cod_pais") < 997))
            .select(["pais", "personas"]).sort("personas", descending=True)
        )
        _tabla("  Flujo por país de residencia en marzo de 2015", paises, "pais",
               paises["personas"].sum(), top)

    s_ext = {c: agregado(stock, c) for c in EXTRANJERO}
    s_tot = sum(s_ext.values())
    pob_total = agregado(stock, "Total")
    print(f"\n  Stock — nacidos en el extranjero: {s_tot:,}"
          f"  ({s_tot / pob_total * 100:.2f}% de la población total)")
    if censo >= 2010:
        for c, v in s_ext.items():
            print(f"      {c:<38} {v:>12,}")
    sexo = stock.filter(pl.col("total_categoria") & (pl.col("cve_destino") == cve)
                        & pl.col("categoria").is_in(EXTRANJERO))
    h, m = sexo["hombres"].sum(), sexo["mujeres"].sum()
    if h and m:
        print(f"      {'Hombres / Mujeres':<38} {h:>12,} / {m:,}")
    print()

    if a.guardar:
        salida = pl.concat([
            inm.rename({"origen": "contraparte"}).with_columns(pl.lit("inmigracion").alias("flujo")),
            emi.rename({"destino": "contraparte"}).with_columns(pl.lit("emigracion").alias("flujo")),
            paises.rename({"pais": "contraparte"}).with_columns(pl.lit("extranjero").alias("flujo")),
        ]).with_columns(
            pl.lit(censo).cast(pl.Int16).alias("censo"),
            pl.lit(cve).cast(pl.Int16).alias("cve_ent"),
            pl.lit(NOMBRE[cve]).alias("entidad"),
        ).select(["censo", "cve_ent", "entidad", "flujo", "contraparte", "personas"])
        ruta = DIR / f"movilidad_{NOMBRE[cve].lower().replace(' ', '_')}_{censo}.parquet"
        salida.write_parquet(ruta)
        print(f"Saved → {ruta}  ({salida.height:,} filas)\n")


if __name__ == "__main__":
    main()
