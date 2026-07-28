"""
Movilidad de personas entre entidades federativas y extranjeros residentes,
para una entidad y un año censal (o un rango de censos, acumulado).

Reporta, según el censo/encuesta elegido (1990, 2000, 2005, 2010, 2015 o 2020):
  · Inmigrantes — quiénes llegaron desde otra entidad en el quinquenio previo.
  · Emigrantes  — quiénes salieron hacia otra entidad en el mismo quinquenio.
  · Saldo neto migratorio interestatal y tasa por mil habitantes.
  · Extranjeros — flujo (residían en otro país hace 5 años, con desglose por país
    en 2020) y stock (nacidos en el extranjero).

2005 (II Conteo) y 2015 (Encuesta Intercensal) rellenan parcialmente los huecos
entre los 4 censos "completos". 2005 tiene la misma matriz origen-destino que los
censos; 2015 es una encuesta MUESTRAL cuya fuente pública sólo trae agregados
(población/inmigrantes/emigrantes/saldo) sin desglose por estado de origen ni
por país — para ese año, las tablas de top-estados y la sección de extranjeros
se muestran como "no disponible en esta fuente".

La movilidad es la declarada en el censo (residencia actual vs. residencia 5 años
antes), no una estimación indirecta como la de scripts/prepare_migracion_interna.py.

Fuente: dashboard_data/ccpv_migracion_estatal.parquet y
        dashboard_data/ccpv_extranjeros_pais_2020.parquet
        (generadas por scripts/prepare_ccpv_migracion.py)

Run: uv run python scripts/datatable/movilidad_estatal.py --entidad Jalisco --año 2020
     uv run python scripts/datatable/movilidad_estatal.py --entidad Jalisco --año 1990-2020
"""

import argparse
import sys
from pathlib import Path

import polars as pl

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "scripts" / "centralismo"))
sys.path.insert(0, str(RAIZ / "scripts"))
from comun import NOMBRE, normalizar_estado
from prepare_ccpv_migracion import EMIGRANTES_AGREGADO

DIR = RAIZ / "dashboard_data"
CENSOS = [1990, 2000, 2005, 2010, 2015, 2020]
VENTANA = {1990: "1985 → 1990", 2000: "enero 1995 → 2000",
           2005: "octubre 2000 → octubre 2005", 2010: "junio 2005 → 2010",
           2015: "marzo 2010 → marzo 2015", 2020: "marzo 2015 → marzo 2020"}
EXTRANJERO = ["En los Estados Unidos de América", "En otro país"]
SIN_DESGLOSE = "(sin desglose por estado en esta fuente)"


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


def _año_o_rango(valor):
    """Un censo suelto ('2010') o un rango inclusivo ('1990-2020') → int o tupla."""
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


def _tabla(titulo, df, col, total, top):
    """Imprime top-N filas de (col, personas) con su % y % acumulado del total."""
    print(f"\n{titulo}: {total:,}")
    if total == 0:
        return
    print(f"  {'#':>3}  {'':<34} {'Personas':>12} {'%':>7} {'% acum.':>8}")
    acumulado = 0
    for i, r in enumerate(df.head(top).iter_rows(named=True), start=1):
        acumulado += r["personas"]
        print(f"  {i:>3}  {r[col]:<34} {r['personas']:>12,} {r['personas']/total*100:>6.1f}%"
              f" {acumulado/total*100:>7.1f}%")
    resto = df.slice(top)
    if resto.height:
        suma = resto["personas"].sum()
        acumulado += suma
        print(f"  {'':>3}  {f'Resto ({resto.height})':<34} {suma:>12,} {suma/total*100:>6.1f}%"
              f" {acumulado/total*100:>7.1f}%")


def cargar():
    return pl.read_parquet(DIR / "ccpv_migracion_estatal.parquet")


def _agregado(df, cve_ent, categoria):
    f = df.filter(pl.col("total_categoria") & (pl.col("cve_destino") == cve_ent)
                  & (pl.col("categoria") == categoria))
    return f["personas"].sum()


def flujos(mig, cve, censo):
    """(inmigrantes, emigrantes, población, tabla flujo completa) de un censo."""
    flujo = mig.filter((pl.col("censo") == censo) & (pl.col("concepto") == "residencia_5a"))
    desglose = flujo.filter(~pl.col("total_categoria"))

    if desglose.filter(pl.col("cve_destino") == cve).height == 0:
        # Fuente sin matriz origen-destino (Encuesta Intercensal 2015): sólo hay
        # totales agregados. Se representa como una única fila "placeholder" para
        # que las tablas top-N y el total (personas.sum()) sigan funcionando sin
        # ramificar el resto del archivo.
        inm = pl.DataFrame({"origen": [SIN_DESGLOSE],
                            "personas": [_agregado(flujo, cve, "En otra entidad")]},
                           schema={"origen": pl.Utf8, "personas": pl.Int64})
        emi = pl.DataFrame({"destino": [SIN_DESGLOSE],
                            "personas": [_agregado(flujo, cve, EMIGRANTES_AGREGADO)]},
                           schema={"destino": pl.Utf8, "personas": pl.Int64})
    else:
        # Incluye la fila 'Entidad no especificada' (sólo 1990) para que el total
        # cuadre con el agregado publicado de 'En otra entidad'. Esos casos no son
        # atribuibles a un origen, así que no aparecen del lado de los emigrantes.
        inm = (desglose.filter(pl.col("cve_destino") == cve)
               .select(["origen", "personas"]).sort("personas", descending=True))
        emi = (desglose.filter((pl.col("cve_origen") == cve) & (pl.col("cve_destino") > 0))
               .select(["destino", "personas"]).sort("personas", descending=True))
        assert inm["personas"].sum() == _agregado(flujo, cve, "En otra entidad")

    poblacion = _agregado(flujo, cve, "Total")
    return inm, emi, poblacion, flujo


def paises_2020(cve, top):
    """Desglose por país del flujo extranjero de 2020 (único censo con este detalle)."""
    df = (
        pl.read_parquet(DIR / "ccpv_extranjeros_pais_2020.parquet")
        .filter((pl.col("cve_destino") == cve) & (pl.col("cod_pais") < 997))
        .select(["pais", "personas"]).sort("personas", descending=True)
    )
    _tabla("  Flujo por país de residencia en marzo de 2015", df, "pais",
           df["personas"].sum(), top)
    return df


def detalle(mig, cve, censo, top):
    inm, emi, poblacion, flujo = flujos(mig, cve, censo)
    n_inm, n_emi = inm["personas"].sum(), emi["personas"].sum()

    print(f"\n{'═' * 72}")
    print(f"  {NOMBRE[cve]} — Censo {censo}")
    print(f"  Movilidad interestatal declarada, {VENTANA[censo]}")
    print(f"  Población de 5 años y más: {poblacion:,}")
    print("═" * 72)

    _tabla("INMIGRANTES — llegaron desde otra entidad", inm, "origen", n_inm, top)
    _tabla("EMIGRANTES — salieron hacia otra entidad", emi, "destino", n_emi, top)

    neto = n_inm - n_emi
    print(f"\nSALDO NETO INTERESTATAL: {neto:+,}"
          f"  ({neto / poblacion * 1000:+.2f} por mil habitantes)")

    # ── Extranjeros ──────────────────────────────────────────────────────────
    print(f"\n{'─' * 72}\nEXTRANJEROS")

    if flujo.filter(pl.col("categoria").is_in(EXTRANJERO)).height == 0:
        print("\n  Flujo — no disponible en esta fuente (sólo migración interestatal).")
    else:
        f_ext = {c: _agregado(flujo, cve, c) for c in EXTRANJERO}
        print(f"\n  Flujo — residían en otro país en {VENTANA[censo].split(' → ')[0]}: "
              f"{sum(f_ext.values()):,}")
        if censo >= 2005:  # 1990 y 2000 no separan Estados Unidos del resto
            for c, v in f_ext.items():
                print(f"      {c:<38} {v:>12,}")

    paises = pl.DataFrame(schema={"pais": pl.Utf8, "personas": pl.Int64})
    if censo == 2020:
        paises = paises_2020(cve, top)

    stock = mig.filter((pl.col("censo") == censo) & (pl.col("concepto") == "nacimiento"))
    if stock.height == 0:
        print("\n  Stock — no disponible en esta fuente (no releva lugar de nacimiento).")
    else:
        s_ext = {c: _agregado(stock, cve, c) for c in EXTRANJERO}
        s_tot = sum(s_ext.values())
        pob_total = _agregado(stock, cve, "Total")
        print(f"\n  Stock — nacidos en el extranjero: {s_tot:,}"
              f"  ({s_tot / pob_total * 100:.2f}% de la población total)")
        if censo >= 2005:
            for c, v in s_ext.items():
                print(f"      {c:<38} {v:>12,}")
        sexo = stock.filter(pl.col("total_categoria") & (pl.col("cve_destino") == cve)
                            & pl.col("categoria").is_in(EXTRANJERO))
        h, m = sexo["hombres"].sum(), sexo["mujeres"].sum()
        if h and m:
            print(f"      {'Hombres / Mujeres':<38} {h:>12,} / {m:,}")
    print()
    return inm, emi, paises


def flujos_acumulados(mig, cve, censos):
    """Suma inmigrantes/emigrantes de una entidad a través de varios censos.

    Cada censo mide un quinquenio DISTINTO y no solapado (1985-1990, 1995-2000,
    2005-2010, 2015-2020), así que sumarlos no duplica el mismo movimiento dos
    veces — pero tampoco cubre los años intermedios (1990-1995, 2000-2005,
    2010-2015), que ningún censo captura.
    """
    inms, emis = [], []
    for censo in censos:
        inm, emi, _, _ = flujos(mig, cve, censo)
        inms.append(inm)
        emis.append(emi)
    inm = (pl.concat(inms).group_by("origen").agg(pl.col("personas").sum())
           .sort("personas", descending=True))
    emi = (pl.concat(emis).group_by("destino").agg(pl.col("personas").sum())
           .sort("personas", descending=True))
    return inm, emi


def acumulado(mig, cve, censos, top):
    inm, emi = flujos_acumulados(mig, cve, censos)
    n_inm, n_emi = inm["personas"].sum(), emi["personas"].sum()

    print(f"\n{'═' * 72}")
    print(f"  {NOMBRE[cve]} — Censos {censos[0]}–{censos[-1]} (acumulado)")
    print("  Movilidad interestatal declarada, suma de quinquenios no solapados")
    print(f"  Ventanas sumadas: {' · '.join(VENTANA[c] for c in censos)}")
    print("═" * 72)

    _tabla(f"INMIGRANTES — suma de {len(censos)} censos", inm, "origen", n_inm, top)
    _tabla(f"EMIGRANTES — suma de {len(censos)} censos", emi, "destino", n_emi, top)

    print(f"\nSALDO NETO INTERESTATAL (suma de los {len(censos)} censos): {n_inm - n_emi:+,}")
    print("\nNota: estos censos cubren quinquenios no solapados, pero dejan años sin medir")
    print("entre uno y otro (p. ej. 1990-1995, 2000-2005, 2010-2015). La suma es la")
    print("migración observada en las ventanas sumadas, no la migración continua del período.")

    print(f"\n{'─' * 72}")
    print("Balance por censo — migración, inmigración y población total:")
    print(f"  {'Censo':<7} {'Población':>13} {'Inmigrantes':>13} {'Emigrantes':>13}"
          f" {'Saldo neto':>13} {'Saldo ‰':>10}")
    for censo in censos:
        i, e, poblacion, _ = flujos(mig, cve, censo)
        n_i, n_e = i["personas"].sum(), e["personas"].sum()
        saldo = n_i - n_e
        print(f"  {censo:<7} {poblacion:>13,} {n_i:>13,} {n_e:>13,}"
              f" {saldo:>+13,} {saldo / poblacion * 1000:>9.2f}")

    # ── Extranjeros ──────────────────────────────────────────────────────────
    print(f"\n{'─' * 72}\nEXTRANJEROS")

    flujo = mig.filter(pl.col("censo").is_in(censos) & (pl.col("concepto") == "residencia_5a")
                       & pl.col("total_categoria") & (pl.col("cve_destino") == cve)
                       & pl.col("categoria").is_in(EXTRANJERO))
    ext = flujo.group_by("categoria").agg(pl.col("personas").sum())
    total_ext = ext["personas"].sum()
    print(f"\n  Flujo — residían en otro país, suma de {len(censos)} censos: {total_ext:,}")
    sin_flujo_ext = [c for c in censos
                     if mig.filter((pl.col("censo") == c) & (pl.col("concepto") == "residencia_5a")
                                   & pl.col("categoria").is_in(EXTRANJERO)).height == 0]
    if sin_flujo_ext:
        print(f"      (sin flujo de extranjero en esta fuente para: "
              f"{', '.join(str(c) for c in sin_flujo_ext)})")
    # 1990/2000 no separan EEUU del resto, así que desglosar mezclado con 2005+
    # atribuiría de más a 'En otro país'. Sólo se desglosa si todos los censos separan.
    if all(c >= 2005 for c in censos if c not in sin_flujo_ext):
        for r in ext.sort("personas", descending=True).iter_rows(named=True):
            print(f"      {r['categoria']:<38} {r['personas']:>12,}")

    paises = pl.DataFrame(schema={"pais": pl.Utf8, "personas": pl.Int64})
    if 2020 in censos:
        paises = paises_2020(cve, top)

    # El stock (nacidos en el extranjero) es un corte a cada censo, no un flujo: una
    # misma persona puede seguir ahí en el siguiente censo. Se muestra por censo, sin
    # sumar, para no fingir un conteo de personas únicas.
    print(f"\n  Stock — nacidos en el extranjero, por censo (no se suma entre censos):")
    for censo in censos:
        stock = mig.filter((pl.col("censo") == censo) & (pl.col("concepto") == "nacimiento"))
        if stock.height == 0:
            print(f"      {censo:<10} no disponible en esta fuente")
            continue
        s_tot = sum(_agregado(stock, cve, c) for c in EXTRANJERO)
        pob_total = _agregado(stock, cve, "Total")
        print(f"      {censo:<10} {s_tot:>12,}   ({s_tot / pob_total * 100:.2f}% de la población)")
    print()
    return inm, emi, paises


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entidad", type=_entidad, required=True,
                   help="Nombre o clave INEGI (1-32) de la entidad federativa")
    p.add_argument("--año", type=_año_o_rango, required=True,
                   help="Año censal, o rango 'AAAA-AAAA' para el acumulado de varios censos")
    p.add_argument("--top", type=int, default=10, help="Filas por tabla (default 10)")
    p.add_argument("--guardar", action="store_true",
                   help="Escribe el resultado a dashboard_data/movilidad_<entidad>_<año>.parquet")
    a = p.parse_args()
    cve, top = a.entidad, a.top

    mig = cargar()
    if isinstance(a.año, tuple):
        inm, emi, paises = acumulado(mig, cve, list(a.año), top)
        etiqueta_año = f"{a.año[0]}-{a.año[-1]}"
    else:
        inm, emi, paises = detalle(mig, cve, a.año, top)
        etiqueta_año = str(a.año)

    if a.guardar:
        salida = pl.concat([
            inm.rename({"origen": "contraparte"}).with_columns(pl.lit("inmigracion").alias("flujo")),
            emi.rename({"destino": "contraparte"}).with_columns(pl.lit("emigracion").alias("flujo")),
            paises.rename({"pais": "contraparte"}).with_columns(pl.lit("extranjero").alias("flujo")),
        ]).with_columns(
            pl.lit(etiqueta_año).alias("censo"),
            pl.lit(cve).cast(pl.Int16).alias("cve_ent"),
            pl.lit(NOMBRE[cve]).alias("entidad"),
        ).select(["censo", "cve_ent", "entidad", "flujo", "contraparte", "personas"])
        ruta = DIR / f"movilidad_{NOMBRE[cve].lower().replace(' ', '_')}_{etiqueta_año}.parquet"
        salida.write_parquet(ruta)
        print(f"Saved → {ruta}  ({salida.height:,} filas)\n")


if __name__ == "__main__":
    main()
