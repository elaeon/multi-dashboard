"""
Comparación de presupuesto federal (PEF) por alumno para tres organismos
públicos federales descentralizados que NO reciben el subsidio DGESUI que
usa scripts/prepare_comparacion_universidades.py: UPN (Universidad
Pedagógica Nacional), IPN (Instituto Politécnico Nacional) y TecNM
(Tecnológico Nacional de México). Sin campo de aportación estatal -- son
organismos 100% federales, no hay presupuesto estatal que agregar.

Cada organismo tiene su propia Unidad Responsable (UR) en el PEF, separada
de las dos URs de universidades estatales (UR_ESTATALES) que usa el script
de universidades -- no hay riesgo de mezclarlos. Pero cada uno requiere una
forma distinta de ubicar su matrícula en ANUIES (ver ORGANISMOS):
- IPN: nombre INSTITUCIÓN único y consistente ("INSTITUTO POLITÉCNICO
  NACIONAL"), igual que ya usa prepare_comparacion_universidades.py.
- UPN: el nombre en ANUIES está fragmentado en decenas de variantes por
  unidad ("Universidad Pedagógica Nacional Unidad 122", "UPN SEAD 211 Unidad
  Puebla", "U P N Unidad 26 A Hermosillo", etc.) -- se matchea por patrón
  regex, no por nombre exacto.
- TecNM: ~250 Institutos Tecnológicos con nombre propio cada uno -- se
  agrupan por SUBSISTEMA (unidades descentralizadas + federales del TecNM),
  no por nombre de INSTITUCIÓN.

UPN no tiene subfunción "Posgrado" separada en el PEF (su gasto de posgrado
está mezclado dentro de "Educación Superior", sin desglose) -- con --posdoc,
FEDERAL/COSTO_ALUMNO de UPN queda "s/d" (su matrícula de posgrado sí se
puede mostrar vía ANUIES, solo falta el presupuesto asociado). IPN y TecNM sí
separan Educación Superior y Posgrado, --posdoc funciona normal para ambos.

Por default (sin --states) imprime una tabla nacional, una fila por
organismo. Run: uv run python scripts/prepare_comparacion_organismos_descentralizados.py --year 2026

Con --states, desglosa por entidad federativa. En el PEF, UPN e IPN tienen
su presupuesto 100% atribuido a "Ciudad de México" (una sola fila, sin
desglose real, aunque su matrícula esté repartida en todo el país) --
mientras que TecNM sí trae desglose real por entidad (32 entidades
distintas en el PEF). Por eso --states PRORRATEA el FEDERAL nacional de
UPN/IPN entre entidades según la proporción de matrícula de cada una
(federal_entidad = federal_nacional × matricula_entidad /
matricula_nacional) -- marcado con ES_PRORRATEADO=True y "*" en la tabla --
mientras que TecNM usa su desglose real del PEF directamente
(ES_PRORRATEADO=False). Consecuencia matemática del prorrateo: como escala
proporcionalmente, COSTO_ALUMNO_FEDERAL de UPN e IPN sale IDÉNTICO en todas
sus entidades (siempre el promedio nacional) -- no refleja variación real,
a diferencia de TecNM donde sí varía entidad por entidad. La tabla es en
formato largo (una fila por combinación ORGANISMO×ENTIDAD) para no mezclar
en una sola cifra el prorrateo con el dato real; trae una fila TOTAL por
organismo (3 en total) para verificar que el desglose reconstruye el
nacional.
Run: uv run python scripts/prepare_comparacion_organismos_descentralizados.py --states --year 2026

Por default solo imprime la tabla en terminal. Pasa --save para guardar el
parquet en dashboard_data/.
"""

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_comparacion_universidades import (
    NIVELES_LIC,
    NIVELES_POS,
    OUT_DIR,
    SUBFUNCION_LIC,
    SUBFUNCION_POS,
    encontrar_anuies,
    encontrar_pef,
    nivel_activo,
)

ORGANISMOS = {
    "UPN": {
        "nombre": "Universidad Pedagógica Nacional",
        "desc_ur": "Universidad Pedagógica Nacional",
        "modo_matricula": "patron",
        # el nombre de INSTITUCIÓN en ANUIES está fragmentado por unidad --
        # no hay un solo nombre exacto que capture todas las unidades
        "patron_institucion": r"(?i)pedag[oó]gica\s*nacional|\bu\.?\s*p\.?\s*n\.?\b",
    },
    "IPN": {
        "nombre": "Instituto Politécnico Nacional",
        "desc_ur": "Instituto Politécnico Nacional",
        "modo_matricula": "exacto",
        "institucion_anuies": "INSTITUTO POLITÉCNICO NACIONAL",
    },
    "TecNM": {
        "nombre": "Tecnológico Nacional de México",
        "desc_ur": "Tecnológico Nacional de México",
        "modo_matricula": "subsistema",
        # ~250 Institutos Tecnológicos con nombre propio, agrupados por subsistema
        "subsistemas": {
            "UNIDADES DESCENTRALIZADAS DEL TECNOLÓGICO NACIONAL DE MÉXICO",
            "UNIDADES FEDERALES DEL TECNOLÓGICO NACIONAL DE MÉXICO",
        },
    },
}


def _filtro_institucion(config: dict) -> pl.Expr:
    """Expresión polars que selecciona las filas de ANUIES pertenecientes a
    este organismo, según config["modo_matricula"]."""
    modo = config["modo_matricula"]
    if modo == "patron":
        return pl.col("INSTITUCIÓN").str.contains(config["patron_institucion"])
    if modo == "exacto":
        return pl.col("INSTITUCIÓN") == config["institucion_anuies"]
    if modo == "subsistema":
        return pl.col("SUBSISTEMA").is_in(config["subsistemas"])
    raise ValueError(f"modo_matricula desconocido: {modo!r}")


def matricula_organismo(anuies: pl.DataFrame, config: dict, niveles: set[str], entidad: str | None = None) -> int | None:
    """Matrícula nacional (o de una sola entidad si se pasa `entidad`) de un
    organismo, para los niveles dados. None si no hay matrícula."""
    filtro = _filtro_institucion(config) & pl.col("NIVEL").is_in(niveles)
    if entidad is not None:
        filtro = filtro & (pl.col("ENTIDAD") == entidad)
    mat = anuies.filter(filtro)["Matrícula Total"].sum()
    return mat if mat else None


def matricula_organismo_por_entidad(anuies: pl.DataFrame, config: dict, niveles: set[str]) -> pl.DataFrame:
    """Matrícula y núm. de planteles de un organismo, agrupado por ENTIDAD.
    Excluye entidades con matrícula 0 en este nivel (ej. un plantel que solo
    ofrece posgrado no debe aparecer como fila de 0 alumnos al filtrar
    licenciatura -- produciría un COSTO_ALUMNO_FEDERAL indefinido)."""
    filtro = _filtro_institucion(config) & pl.col("NIVEL").is_in(niveles)
    sub = anuies.filter(filtro)
    tabla = sub.group_by("ENTIDAD").agg(
        pl.sum("Matrícula Total").alias("MATRICULA"),
        pl.struct(["MUNICIPIO", "ESCUELA/CAMPUS/PLANTEL"]).n_unique().alias("N_PLANTELES"),
    )
    return tabla.filter(pl.col("MATRICULA") > 0)


def presupuesto_organismo(pef: pl.DataFrame, col_monto: str, desc_ur: str, subfuncion: str, entidad: str | None = None) -> float | None:
    """Presupuesto federal nacional (o de una sola entidad si se pasa
    `entidad`) de un organismo, para la subfunción dada. None si no hay
    filas o la suma es 0 -- a diferencia de calcular_tabla_año() en
    prepare_comparacion_universidades.py, aquí NO se lanza error en 0,
    porque UPN + --posdoc es un caso legítimo sin dato (no un bug de
    nombres/esquema): UPN no tiene subfunción "Posgrado" separada."""
    filtro = (pl.col("DESC_UR") == desc_ur) & (pl.col("DESC_SUBFUNCION") == subfuncion)
    if entidad is not None:
        filtro = filtro & (pl.col("DESC_ENTIDAD_FEDERATIVA") == entidad)
    monto = pef.filter(filtro)[col_monto].sum()
    return float(monto) if monto else None


def presupuesto_organismo_por_entidad(pef: pl.DataFrame, col_monto: str, desc_ur: str, subfuncion: str) -> pl.DataFrame:
    """Presupuesto federal real de un organismo, agrupado por
    DESC_ENTIDAD_FEDERATIVA del PEF (solo tiene sentido para TecNM, que sí
    desglosa -- UPN/IPN van 100% a una sola fila 'Ciudad de México')."""
    sub = pef.filter((pl.col("DESC_UR") == desc_ur) & (pl.col("DESC_SUBFUNCION") == subfuncion))
    out = sub.group_by("DESC_ENTIDAD_FEDERATIVA").agg(pl.sum(col_monto).alias("FEDERAL"))
    return out.with_columns(
        pl.when(pl.col("DESC_ENTIDAD_FEDERATIVA") == "Estado de México")
        .then(pl.lit("MÉXICO"))
        .otherwise(pl.col("DESC_ENTIDAD_FEDERATIVA").str.to_uppercase())
        .alias("ENTIDAD")
    )


def _col_monto(pef: pl.DataFrame, pef_path: Path) -> str:
    try:
        return next(c for c in pef.columns if "MONTO" in c)
    except StopIteration:
        raise ValueError(f"{pef_path.name} no tiene columna MONTO_* -- no soportado.")


def calcular_tabla_nacional(year: int, lag: int, posdoc: bool) -> tuple[pl.DataFrame, Path, Path, str, int]:
    """Tabla nacional (sin --states): una fila por organismo con FEDERAL,
    MATRICULA y COSTO_ALUMNO_FEDERAL. Devuelve (tabla, pef_path, anuies_path,
    ciclo, año_esperado)."""
    nombre_nivel, niveles, subfuncion = nivel_activo(posdoc)
    anuies_path, ciclo, año_esperado = encontrar_anuies(lag)
    pef_path = encontrar_pef(year)
    pef = pl.read_excel(pef_path)
    col_monto = _col_monto(pef, pef_path)
    for columna in ("DESC_SUBFUNCION", "DESC_ENTIDAD_FEDERATIVA"):
        if columna not in pef.columns:
            raise ValueError(f"{pef_path.name} no tiene columna {columna}, requerida.")

    anuies = pl.read_excel(anuies_path, sheet_name="Base de datos")

    filas = []
    for sigla, config in ORGANISMOS.items():
        federal = presupuesto_organismo(pef, col_monto, config["desc_ur"], subfuncion)
        matricula = matricula_organismo(anuies, config, niveles)
        n_entidades = anuies.filter(_filtro_institucion(config) & pl.col("NIVEL").is_in(niveles))["ENTIDAD"].n_unique()
        n_planteles = (
            anuies.filter(_filtro_institucion(config) & pl.col("NIVEL").is_in(niveles))
            .select("MUNICIPIO", "ESCUELA/CAMPUS/PLANTEL")
            .unique()
            .height
        )
        costo = federal / matricula if federal is not None and matricula else None
        filas.append((sigla, config["nombre"], n_entidades, n_planteles, matricula, federal, costo))

    tabla = pl.DataFrame(
        filas, schema=["SIGLA", "INSTITUCION", "N_ENTIDADES", "N_PLANTELES", "MATRICULA", "FEDERAL", "COSTO_ALUMNO_FEDERAL"], orient="row"
    )
    return tabla, pef_path, anuies_path, ciclo, año_esperado


def calcular_tabla_entidades(year: int, lag: int, posdoc: bool) -> tuple[pl.DataFrame, Path, Path, str, int]:
    """Tabla --states, formato largo (una fila por ORGANISMO×ENTIDAD).
    UPN/IPN: FEDERAL prorrateado por matrícula (ES_PRORRATEADO=True).
    TecNM: FEDERAL real del PEF por entidad (ES_PRORRATEADO=False). Trae una
    fila TOTAL por organismo (3 en total). Devuelve (tabla, pef_path,
    anuies_path, ciclo, año_esperado)."""
    nombre_nivel, niveles, subfuncion = nivel_activo(posdoc)
    anuies_path, ciclo, año_esperado = encontrar_anuies(lag)
    pef_path = encontrar_pef(year)
    pef = pl.read_excel(pef_path)
    col_monto = _col_monto(pef, pef_path)
    for columna in ("DESC_SUBFUNCION", "DESC_ENTIDAD_FEDERATIVA"):
        if columna not in pef.columns:
            raise ValueError(f"{pef_path.name} no tiene columna {columna}, requerida.")

    anuies = pl.read_excel(anuies_path, sheet_name="Base de datos")

    bloques = []
    for sigla, config in ORGANISMOS.items():
        mat_ent = matricula_organismo_por_entidad(anuies, config, niveles)

        if sigla == "TecNM":
            fed_ent = presupuesto_organismo_por_entidad(pef, col_monto, config["desc_ur"], subfuncion)
            tabla = mat_ent.join(fed_ent.select("ENTIDAD", "FEDERAL"), on="ENTIDAD", how="left")
            tabla = tabla.with_columns(pl.col("FEDERAL").cast(pl.Float64), pl.lit(False).alias("ES_PRORRATEADO"))
        else:
            federal_nacional = presupuesto_organismo(pef, col_monto, config["desc_ur"], subfuncion)
            matricula_nacional = matricula_organismo(anuies, config, niveles)
            if federal_nacional is not None and matricula_nacional:
                tabla = mat_ent.with_columns(
                    (pl.col("MATRICULA") / matricula_nacional * federal_nacional).alias("FEDERAL")
                )
            else:
                tabla = mat_ent.with_columns(pl.lit(None, dtype=pl.Float64).alias("FEDERAL"))
            tabla = tabla.with_columns(pl.lit(True).alias("ES_PRORRATEADO"))

        tabla = tabla.with_columns(
            pl.lit(sigla).alias("ORGANISMO"),
            (pl.col("FEDERAL") / pl.col("MATRICULA")).alias("COSTO_ALUMNO_FEDERAL"),
        )
        tabla = tabla.sort("COSTO_ALUMNO_FEDERAL", descending=True, nulls_last=True)

        total = pl.DataFrame(
            {
                "ENTIDAD": ["TOTAL"],
                "MATRICULA": [tabla["MATRICULA"].sum()],
                "N_PLANTELES": [tabla["N_PLANTELES"].sum()],
                "FEDERAL": [tabla["FEDERAL"].sum() if tabla["FEDERAL"].null_count() < tabla.height else None],
                "ORGANISMO": [sigla],
                "ES_PRORRATEADO": [sigla != "TecNM"],
            }
        ).with_columns((pl.col("FEDERAL") / pl.col("MATRICULA")).alias("COSTO_ALUMNO_FEDERAL"))
        total = total.select([pl.col(c).cast(tabla.schema[c]) for c in tabla.columns])

        bloques.append(pl.concat([tabla, total]))

    tabla_final = pl.concat(bloques).select("ORGANISMO", "ENTIDAD", "N_PLANTELES", "MATRICULA", "FEDERAL", "COSTO_ALUMNO_FEDERAL", "ES_PRORRATEADO")
    return tabla_final, pef_path, anuies_path, ciclo, año_esperado


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--lag", type=int, default=0, help="Ciclo ANUIES a usar: 0 = más reciente disponible, 1 = un ciclo atrás, etc.")
    parser.add_argument("--states", action="store_true", help="Desglosa por entidad federativa en vez de la tabla nacional por organismo. UPN/IPN: FEDERAL prorrateado por matrícula (el PEF no los desglosa). TecNM: desglose real del PEF.")
    parser.add_argument("--posdoc", action="store_true", help="Usa posgrado (maestría+doctorado+especialidad) en vez de licenciatura+TSU (default). UPN no tiene subfunción Posgrado separada en el PEF -- su FEDERAL queda 's/d' con este flag.")
    parser.add_argument("--save", action="store_true", help="Guardar el resultado como parquet en dashboard_data/ (por default no se guarda, solo se imprime)")
    args = parser.parse_args()

    nombre_nivel, _, _ = nivel_activo(args.posdoc)
    year = args.year

    def fmt(v):
        return f"{'s/d':>16}" if v is None else f"{v:>16,.0f}"

    if args.states:
        tabla, pef_path, anuies_path, ciclo, año_esperado = calcular_tabla_entidades(year, args.lag, args.posdoc)

        print(f"Fuente presupuesto: {pef_path}")
        print(f"Fuente matrícula: {anuies_path} (ciclo {ciclo})")
        print(f"Nivel: {nombre_nivel}")
        if year != año_esperado:
            print(f"[aviso] --year {year} no coincide con el año esperado para el ciclo {ciclo} ({año_esperado}) — la comparación mezcla presupuesto y matrícula de años distintos.")
        print("[aviso] FEDERAL de UPN e IPN es una ESTIMACIÓN prorrateada por matrícula (el PEF no los desglosa por entidad, van 100% a Ciudad de México) -- marcado con '*'. COSTO_ALUMNO_FEDERAL de estos dos organismos sale igual en todas sus entidades por construcción del prorrateo, no refleja variación real. Solo TecNM usa desglose real del PEF por entidad.")
        if args.posdoc:
            print("[aviso] UPN no tiene subfunción 'Posgrado' separada en el PEF -- su FEDERAL/COSTO_ALUMNO queda 's/d' con --posdoc en todas las entidades.")
        print()

        print(f"{'Organismo':<8} {'Entidad':<20} {'Planteles':>10} {'Matrícula':>10} {'Federal':>16} {'Costo/al.':>16}")
        for r in tabla.iter_rows(named=True):
            marca = "*" if r["ES_PRORRATEADO"] and r["FEDERAL"] is not None else ""
            federal_str = fmt(r["FEDERAL"]) + marca if r["FEDERAL"] is not None else fmt(None)
            costo_str = fmt(r["COSTO_ALUMNO_FEDERAL"]) + marca if r["COSTO_ALUMNO_FEDERAL"] is not None else fmt(None)
            print(f"{r['ORGANISMO']:<8} {r['ENTIDAD']:<20} {r['N_PLANTELES']:>10,} {r['MATRICULA']:>10,} {federal_str:>17} {costo_str:>17}")

        if args.save:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            out_path = OUT_DIR / f"comparacion_organismos_entidades_{year}_{nombre_nivel}.parquet"
            tabla.write_parquet(out_path)
            print(f"\nGuardado → {out_path}")
        else:
            print("\n(no guardado — pasa --save para escribir el parquet)")
        return

    tabla, pef_path, anuies_path, ciclo, año_esperado = calcular_tabla_nacional(year, args.lag, args.posdoc)

    print(f"Fuente presupuesto: {pef_path}")
    print(f"Fuente matrícula: {anuies_path} (ciclo {ciclo})")
    print(f"Nivel: {nombre_nivel}")
    if year != año_esperado:
        print(f"[aviso] --year {year} no coincide con el año esperado para el ciclo {ciclo} ({año_esperado}) — la comparación mezcla presupuesto y matrícula de años distintos.")
    if args.posdoc:
        print("[aviso] UPN no tiene subfunción 'Posgrado' separada en el PEF -- su FEDERAL/COSTO_ALUMNO queda 's/d' con --posdoc.")
    print()

    print(f"{'Organismo':<8} {'Institución':<38} {'Entidades':>10} {'Planteles':>10} {'Matrícula':>10} {'Federal (M)':>13} {'Costo/alumno':>14}")
    for r in tabla.iter_rows(named=True):
        federal_m = f"{r['FEDERAL']/1e6:>13,.0f}" if r["FEDERAL"] is not None else f"{'s/d':>13}"
        costo = f"{r['COSTO_ALUMNO_FEDERAL']:>14,.0f}" if r["COSTO_ALUMNO_FEDERAL"] is not None else f"{'s/d':>14}"
        print(f"{r['SIGLA']:<8} {r['INSTITUCION']:<38} {r['N_ENTIDADES']:>10,} {r['N_PLANTELES']:>10,} {r['MATRICULA']:>10,} {federal_m} {costo}")

    if args.save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"comparacion_organismos_{year}_{nombre_nivel}.parquet"
        tabla.write_parquet(out_path)
        print(f"\nGuardado → {out_path}")
    else:
        print("\n(no guardado — pasa --save para escribir el parquet)")


if __name__ == "__main__":
    main()
