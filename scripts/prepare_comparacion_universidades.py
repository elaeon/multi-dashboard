"""
Comparación de presupuesto federal (PEF) por alumno entre UNAM, IPN, UAM y las
universidades públicas estatales (subsidio vía DGESUI), usando matrícula real
de ANUIES como denominador.

Motivo: UNAM e IPN administran, dentro de la misma Unidad Responsable, tanto su
nivel superior/posgrado como su bachillerato (CCH/prepas en UNAM, CECyT en IPN)
e investigación científica/cultura — mezclar todo eso en el presupuesto y
dividirlo entre la matrícula de educación superior (que es lo único que reporta
ANUIES) infla artificialmente el gasto por alumno. Lo mismo aplica al lado de
las universidades estatales: la bolsa de DGESUI "Subsidios para organismos
descentralizados estatales" también trae mezclado el bachillerato de algunas
universidades (ej. SEMS de la Universidad de Guadalajara).

Este script filtra por DESC_SUBFUNCION en {'Educación Superior', 'Posgrado'} en
ambos lados (excluye Educación Media Superior, Investigación Científica,
Cultura y otras subfunciones) para que numerador y denominador midan lo mismo:
docencia de nivel superior.

Fuente presupuesto: data/presupuesto_federacion/presupuesto/egresos_federacion/
PEF_{año}.xlsx (mismos patrones de nombre que prepare_ramos_sector_presupuesto.py;
requiere columna DESC_SUBFUNCION, disponible desde el esquema 2018+).

Fuente matrícula: data/anuies/base_anuario_{ciclo}_general.xlsx (Anuario
Estadístico ANUIES). --lag elige qué ciclo usar: 0 (default) = el más reciente
disponible en data/anuies/, 1 = un ciclo atrás, etc. — útil porque el anuario
del ciclo más reciente suele tardar en publicarse, o porque se quiere comparar
un PEF viejo con la matrícula de su propio ciclo escolar en vez de la más
reciente. Si el --year del PEF no coincide con el año esperado para ese ciclo
(fin_de_ciclo + 1), se imprime un aviso porque la comparación mezcla años.

Los nombres de DESC_UR/DESC_PP cambian entre años — la UR de las universidades
estatales se llamó "Dirección General de Educación Superior Universitaria"
hasta 2021 y luego le agregaron "e Intercultural" (2022 en adelante); ambos
nombres están cubiertos en UR_ESTATALES. Verificado 2019-2026 (2018 hacia
atrás no tiene ciclo ANUIES disponible localmente, ver --lag). Si en el
futuro vuelve a cambiar el nombre y no está en UR_ESTATALES, el script
levanta el error de presupuesto-en-cero de abajo en vez de reportar una
cifra silenciosamente equivocada.

Por default solo imprime la tabla en terminal — no escribe nada a disco. Pasa
--save para guardar el parquet en dashboard_data/.

Output (con --save): dashboard_data/comparacion_universidades_{año}.parquet
Run: uv run python scripts/prepare_comparacion_universidades.py --year 2026 --lag 0 --save

Con --historico INICIO-FIN (ej. --historico 2019-2026), en vez de un solo año
corre un año por cada uno del rango, encadenando --lag automáticamente (lag 0
para el año más reciente cuyo ciclo ANUIES exista, +1 por cada año hacia
atrás — misma relación fin_de_ciclo+1=año_PEF que usa --lag suelto). Los años
sin ciclo ANUIES disponible o con error de datos (DESC_UR/DESC_PP no
reconocido) se omiten con un aviso, no truenan la corrida completa. El
resultado se deflacta con el índice INPC real (cargar_indice_inpc, de
scripts/datatable/poder_adquisitivo_nacional.py) a pesos constantes del
último año del rango, y se imprime la evolución (nominal, real, % acumulado
vs. el primer año, CAGR real) por institución, más el ratio 3 federales vs.
estatales por año.
Run: uv run python scripts/prepare_comparacion_universidades.py --historico 2019-2026 --save
"""

import argparse
import sys
from pathlib import Path

import polars as pl

PEF_DIR = Path("data/presupuesto_federacion/presupuesto/egresos_federacion")
ANUIES_DIR = Path("data/anuies")
OUT_DIR = Path("dashboard_data")

sys.path.insert(0, str(Path(__file__).resolve().parent / "datatable"))

INSTITUCIONES_FEDERALES = {
    "UNAM": "Universidad Nacional Autónoma de México",
    "IPN": "Instituto Politécnico Nacional",
    "UAM": "Universidad Autónoma Metropolitana",
}
# ANUIES reporta INSTITUCIÓN en mayúsculas (DESC_UR del PEF, en cambio, va en Title Case)
INSTITUCIONES_FEDERALES_ANUIES = {sigla: nombre.upper() for sigla, nombre in INSTITUCIONES_FEDERALES.items()}
SUBFUNCIONES_EDUCATIVAS = {"Educación Superior", "Posgrado"}
# La UR se renombró en algún año entre 2021 y 2022 (le agregaron "e Intercultural")
UR_ESTATALES = {
    "Dirección General de Educación Superior Universitaria e Intercultural",
    "Dirección General de Educación Superior Universitaria",
}
PP_ESTATALES = "Subsidios para organismos descentralizados estatales"
SUBSISTEMA_ESTATALES = "UNIVERSIDADES PÚBLICAS ESTATALES"


def encontrar_pef(year: int) -> Path:
    candidatos = [
        f"PEF_{year}.xlsx",
        f"PEF{year}_AC01.xlsx",
        f"pef_{year}.xlsx",
        f"pef_ac01_{year}.xlsx",
    ]
    for nombre in candidatos:
        ruta = PEF_DIR / nombre
        if ruta.exists():
            return ruta
    raise FileNotFoundError(f"No se encontró el PEF de {year} en {PEF_DIR} (probé: {candidatos})")


def encontrar_anuies(lag: int) -> tuple[Path, str, int]:
    archivos = sorted(ANUIES_DIR.glob("base_anuario_*_general.xlsx"), reverse=True)
    if not archivos:
        raise FileNotFoundError(f"No se encontraron anuarios ANUIES (base_anuario_*_general.xlsx) en {ANUIES_DIR}")
    if lag < 0 or lag >= len(archivos):
        ciclos = [a.stem.removeprefix("base_anuario_").removesuffix("_general") for a in archivos]
        raise ValueError(f"--lag {lag} fuera de rango: solo hay {len(archivos)} ciclo(s) disponible(s) en {ANUIES_DIR} ({ciclos})")
    ruta = archivos[lag]
    ciclo = ruta.stem.removeprefix("base_anuario_").removesuffix("_general")
    año_fin_ciclo = int(ciclo.split("-")[1])
    return ruta, ciclo, año_fin_ciclo + 1


def presupuesto_educativo(pef: pl.DataFrame, col_monto: str, desc_ur: str) -> float:
    sub = pef.filter(
        (pl.col("DESC_UR") == desc_ur) & (pl.col("DESC_SUBFUNCION").is_in(SUBFUNCIONES_EDUCATIVAS))
    )
    return sub[col_monto].sum()


def cargar_matricula(anuies_path: Path) -> dict[str, int]:
    anuies = pl.read_excel(anuies_path, sheet_name="Base de datos")
    matricula = {
        sigla: anuies.filter(pl.col("INSTITUCIÓN") == nombre)["Matrícula Total"].sum()
        for sigla, nombre in INSTITUCIONES_FEDERALES_ANUIES.items()
    }
    matricula["ESTATALES"] = anuies.filter(pl.col("SUBSISTEMA") == SUBSISTEMA_ESTATALES)["Matrícula Total"].sum()
    sin_matricula = [k for k, v in matricula.items() if not v]
    if sin_matricula:
        raise ValueError(f"Matrícula 0 para {sin_matricula} — revisar nombres de INSTITUCIÓN/SUBSISTEMA en {anuies_path}")
    return matricula


def calcular_tabla_año(year: int, lag: int) -> tuple[pl.DataFrame, Path, Path, str, int]:
    """Presupuesto (Educ. Superior + Posgrado) y matrícula ANUIES para un solo
    año. Devuelve (tabla, pef_path, anuies_path, ciclo, año_esperado)."""
    anuies_path, ciclo, año_esperado = encontrar_anuies(lag)
    pef_path = encontrar_pef(year)
    raw = pl.read_excel(pef_path)
    try:
        col_monto = next(c for c in raw.columns if "MONTO" in c)
    except StopIteration:
        raise ValueError(
            f"{pef_path.name} no tiene columna MONTO_* (columnas: {raw.columns}). "
            "Los PEF 2008-2017 en formato AC01 no siguen este esquema y no están soportados."
        )
    if "DESC_SUBFUNCION" not in raw.columns:
        raise ValueError(f"{pef_path.name} no tiene columna DESC_SUBFUNCION, requerida para separar bachillerato/investigación/cultura.")

    matricula = cargar_matricula(anuies_path)

    filas = [
        (sigla, nombre, presupuesto_educativo(raw, col_monto, nombre), matricula[sigla])
        for sigla, nombre in INSTITUCIONES_FEDERALES.items()
    ]

    estatales_raw = raw.filter(
        (pl.col("DESC_UR").is_in(UR_ESTATALES))
        & (pl.col("DESC_PP") == PP_ESTATALES)
        & (pl.col("DESC_SUBFUNCION").is_in(SUBFUNCIONES_EDUCATIVAS))
    )
    filas.append(("ESTATALES", "Universidades Públicas Estatales (35, vía DGESUI)", estatales_raw[col_monto].sum(), matricula["ESTATALES"]))

    sin_presupuesto = [nombre for _, nombre, presupuesto, _ in filas if not presupuesto]
    if sin_presupuesto:
        raise ValueError(
            f"Presupuesto 0 para {sin_presupuesto} en {pef_path.name} — probablemente DESC_UR/DESC_PP "
            "cambiaron de nombre en este año (el script solo está verificado para 2019-2026)."
        )

    tabla = pl.DataFrame(filas, schema=["SIGLA", "INSTITUCION", "PRESUPUESTO", "MATRICULA"], orient="row")
    tabla = tabla.with_columns(
        (pl.col("PRESUPUESTO") / pl.col("MATRICULA")).alias("GASTO_POR_ALUMNO"),
        pl.lit(year).alias("AÑO_PEF"),
    )
    return tabla, pef_path, anuies_path, ciclo, año_esperado


def parsear_rango(historico: str) -> tuple[int, int]:
    try:
        inicio, fin = historico.split("-")
        inicio, fin = int(inicio), int(fin)
    except ValueError:
        raise ValueError(f"--historico debe tener el formato INICIO-FIN (ej. 2019-2026), recibí {historico!r}")
    if inicio > fin:
        raise ValueError(f"--historico {historico}: el año de inicio no puede ser mayor al de fin")
    return inicio, fin


def correr_historico(historico: str, save: bool):
    from poder_adquisitivo_nacional import cargar_indice_inpc

    inicio, fin = parsear_rango(historico)
    _, _, año_esperado_lag0 = encontrar_anuies(0)

    tablas = []
    for year in range(inicio, fin + 1):
        lag = año_esperado_lag0 - year
        if lag < 0:
            print(f"[omitido] {year}: no hay ciclo ANUIES tan reciente disponible localmente.")
            continue
        try:
            tabla, pef_path, anuies_path, ciclo, año_esperado = calcular_tabla_año(year, lag)
        except (FileNotFoundError, ValueError) as e:
            print(f"[omitido] {year} (lag {lag}): {e}")
            continue
        tablas.append(tabla)

    if not tablas:
        raise RuntimeError(f"Ningún año del rango {historico} pudo calcularse (ver avisos [omitido] arriba).")

    combinada = pl.concat(tablas)
    años_calculados = sorted(combinada["AÑO_PEF"].unique().to_list())
    año_base = años_calculados[-1]

    inpc = pl.from_pandas(cargar_indice_inpc()[["año", "indice"]])
    indice_base = inpc.filter(pl.col("año") == año_base)["indice"][0]

    combinada = combinada.join(inpc, left_on="AÑO_PEF", right_on="año", how="left")
    assert combinada["indice"].null_count() == 0, "Falta índice INPC para alguno de los años calculados"
    combinada = combinada.with_columns(
        (pl.col("GASTO_POR_ALUMNO") * indice_base / pl.col("indice")).alias("GASTO_REAL"),
        pl.lit(año_base).alias("AÑO_BASE"),
    )

    print(f"\nAños calculados: {años_calculados} (pesos constantes de {año_base})\n")
    for sigla in list(INSTITUCIONES_FEDERALES) + ["ESTATALES"]:
        sub = combinada.filter(pl.col("SIGLA") == sigla).sort("AÑO_PEF")
        if sub.height == 0:
            continue
        nombre = sub["INSTITUCION"][0]
        primero = sub["GASTO_REAL"][0]
        print(f"--- {nombre} ---")
        print(f"{'Año':>5} {'Nominal':>12} {'Real':>12} {'% acum. real':>14}")
        for r in sub.iter_rows(named=True):
            acum = (r["GASTO_REAL"] / primero - 1) * 100
            print(f"{r['AÑO_PEF']:>5} {r['GASTO_POR_ALUMNO']:>12,.0f} {r['GASTO_REAL']:>12,.0f} {acum:>+13.1f}%")
        n = sub.height - 1
        if n > 0:
            cagr = ((sub["GASTO_REAL"][-1] / primero) ** (1 / n) - 1) * 100
            print(f"  CAGR real {sub['AÑO_PEF'][0]}→{sub['AÑO_PEF'][-1]}: {cagr:+.2f}%/año")
        print()

    fed = combinada.filter(pl.col("SIGLA") != "ESTATALES").group_by("AÑO_PEF").agg(
        pl.sum("PRESUPUESTO").alias("P"), pl.sum("MATRICULA").alias("M"), pl.first("indice").alias("indice")
    ).with_columns((pl.col("P") / pl.col("M") * indice_base / pl.col("indice")).alias("REAL_3FED"))
    est = combinada.filter(pl.col("SIGLA") == "ESTATALES").select("AÑO_PEF", pl.col("GASTO_REAL").alias("REAL_EST"))
    ratio = fed.join(est, on="AÑO_PEF").with_columns((pl.col("REAL_3FED") / pl.col("REAL_EST")).alias("RATIO")).sort("AÑO_PEF")

    print("--- Ratio 3 Federales (UNAM+IPN+UAM) vs. Estatales (real) ---")
    print(f"{'Año':>5} {'3 Federales':>14} {'Estatales':>12} {'Ratio':>7}")
    for r in ratio.iter_rows(named=True):
        print(f"{r['AÑO_PEF']:>5} {r['REAL_3FED']:>14,.0f} {r['REAL_EST']:>12,.0f} {r['RATIO']:>6.2f}x")

    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"comparacion_universidades_historico_{inicio}_{fin}.parquet"
        combinada.write_parquet(out_path)
        print(f"\nGuardado → {out_path}")
    else:
        print("\n(no guardado — pasa --save para escribir el parquet)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--lag", type=int, default=0, help="Ciclo ANUIES a usar: 0 = más reciente disponible, 1 = un ciclo atrás, etc.")
    parser.add_argument("--historico", type=str, default=None, metavar="INICIO-FIN", help="Corre un rango de años (ej. 2019-2026) encadenando --lag automáticamente, y muestra la evolución real (deflactada con INPC) en vez de la tabla de un solo año.")
    parser.add_argument("--save", action="store_true", help="Guardar el resultado como parquet en dashboard_data/ (por default no se guarda, solo se imprime)")
    args = parser.parse_args()

    if args.historico:
        correr_historico(args.historico, args.save)
        return

    year = args.year
    tabla, pef_path, anuies_path, ciclo, año_esperado = calcular_tabla_año(year, args.lag)

    print(f"Fuente presupuesto: {pef_path}")
    print(f"Fuente matrícula: {anuies_path} (ciclo {ciclo})")
    if year != año_esperado:
        print(f"[aviso] --year {year} no coincide con el año esperado para el ciclo {ciclo} ({año_esperado}) — la comparación mezcla presupuesto y matrícula de años distintos.")
    print()
    print(f"{'Institución':<52} {'Educ.Sup+Posgrado (M)':>22} {'Matrícula':>10} {'Gasto/alumno':>14}")
    for r in tabla.iter_rows(named=True):
        print(f"{r['INSTITUCION']:<52} {r['PRESUPUESTO']/1e6:>22,.0f} {r['MATRICULA']:>10,} {r['GASTO_POR_ALUMNO']:>14,.0f}")

    if args.save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"comparacion_universidades_{year}.parquet"
        tabla.write_parquet(out_path)
        print(f"\nGuardado → {out_path}")
    else:
        print("\n(no guardado — pasa --save para escribir el parquet)")


if __name__ == "__main__":
    main()
