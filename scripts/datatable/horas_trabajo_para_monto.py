"""
Horas de trabajo necesarias para ganar X pesos, nacional y por entidad
federativa, en los percentiles 50/75/95/99 del salario por hora (ENOE).

salario_hora = ingocup / (hrsocup * 4.345)  (ingreso mensual / horas mensuales)
horas_necesarias = monto / salario_hora_percentil

Fuente: data/inegi/enoe/ (sdem, 2005–2026). Ver Loading Recipe en
data/inegi/enoe/DATA_OVERVIEW.md — se reproduce aquí sin hardcodear el patrón
de nombre de ZIP (4 variantes a través de los años).

2 rupturas de esquema manejadas vía fallback: fac→fac_tri (2020Q3),
ent→cve_ent (2025Q3). 2020Q2 no tiene ZIP (hueco COVID).

Output: dashboard_data/horas_trabajo_por_entidad_<año>t<trimestre>.parquet
Run: uv run python scripts/datatable/horas_trabajo_para_monto.py --monto 1000 --año 2026 --trimestre 1
    o --todos-trimestres en vez de --trimestre, para juntar los trimestres
    disponibles del año y ganar precisión (más muestra por entidad).
"""

import argparse
import glob
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "centralismo"))
from comun import NOMBRE

BASE = Path(__file__).resolve().parents[2] / "data" / "inegi" / "enoe"

ENOEN_WINDOW = {(2020, 3), (2020, 4), (2021, 1), (2021, 2), (2021, 3), (2021, 4),
                (2022, 1), (2022, 2), (2022, 3), (2022, 4)}

# Columnas sdem realmente usadas (por este script y por poder_adquisitivo_nacional.py,
# que importa read_enoe_table). fac/fac_tri y ent/cve_ent varían por trimestre (ver
# rupturas de esquema arriba) — se incluyen ambos nombres, un callable en usecols no
# falla si alguno no está presente en un trimestre dado (a diferencia de pasar una lista).
USECOLS_SDEM = {"clase2", "ingocup", "hrsocup", "fac", "fac_tri", "ent", "cve_ent", "scian", "c_ocu11c"}

# Catálogos ENOE sdem (codigo 0="No aplica" y el código "No especificado" excluidos:
# no son opciones de filtro válidas). Fuente: catalogos/scian.csv y catalogos/c_ocu11c.csv
# dentro del ZIP de cada trimestre.
SCIAN = {
    1: "Agricultura, ganadería, aprovechamiento forestal, pesca y caza",
    2: "Minería",
    3: "Generación y distribución de electricidad, suministro de agua y gas",
    4: "Construcción",
    5: "Industrias manufactureras",
    6: "Comercio al por mayor",
    7: "Comercio al por menor",
    8: "Transportes, correos y almacenamiento",
    9: "Información en medios masivos",
    10: "Servicios financieros y de seguros",
    11: "Servicios inmobiliarios y de alquiler de bienes",
    12: "Servicios profesionales, científicos y técnicos",
    13: "Corporativos",
    14: "Servicios de apoyo a los negocios y manejo de desechos",
    15: "Servicios educativos",
    16: "Servicios de salud y de asistencia social",
    17: "Servicios de esparcimiento, culturales y deportivos",
    18: "Servicios de hospedaje y de preparación de alimentos y bebidas",
    19: "Otros servicios, excepto actividades gubernamentales",
    20: "Actividades gubernamentales y de organismos internacionales",
}

C_OCU11C = {
    1: "Profesionales, técnicos y trabajadores del arte",
    2: "Trabajadores de la educación",
    3: "Funcionarios y directivos",
    4: "Oficinistas",
    5: "Trabajadores industriales, artesanos y ayudantes",
    6: "Comerciantes",
    7: "Operadores de transporte",
    8: "Trabajadores en servicios personales",
    9: "Trabajadores en protección y vigilancia",
    10: "Trabajadores agropecuarios",
}


def _catalogo_texto(nombre, catalogo):
    return f"{nombre}:\n" + "\n".join(f"  {c:>2}  {l}" for c, l in catalogo.items())


EPILOG = (
    _catalogo_texto("--sector (SCIAN)", SCIAN) + "\n\n" +
    _catalogo_texto("--ocupacion (c_ocu11c)", C_OCU11C)
)


def read_enoe_table(year: int, quarter: int, table: str) -> pd.DataFrame:
    zips = glob.glob(f"{BASE}/{year}/*.zip")
    match = None
    for zpath in zips:
        with zipfile.ZipFile(zpath) as z:
            names = z.namelist()
            if any(f"_{year}_{quarter}t/" in n or f"_{year}_{quarter}t.csv" in n for n in names):
                match = zpath
                break
    if match is None:
        raise FileNotFoundError(f"No se encontró ZIP para {year}Q{quarter}")
    with zipfile.ZipFile(match) as z:
        candidates = [n for n in z.namelist()
                      if f"conjunto_de_datos_{table}_" in n
                      and n.endswith(".csv")
                      and "/conjunto_de_datos/" in n
                      and "bitacora" not in n]
        if not candidates:
            raise FileNotFoundError(f"{table} no encontrado en {match}")
        with z.open(candidates[0]) as f:
            data = f.read()
    return pd.read_csv(io.BytesIO(data), encoding="latin-1", low_memory=False,
                        usecols=lambda c: c in USECOLS_SDEM)


def cargar_enoe_año_completo(año: int):
    """Lee y concatena todos los trimestres sdem disponibles de un año.

    Resuelve peso_col/ent_col POR TRIMESTRE antes de concatenar — necesario
    porque 2020 (fac→fac_tri en Q3) y 2025 (ent→cve_ent en Q3) cruzan una
    ruptura de esquema a media año, no en el límite del año.
    """
    partes, trimestres_usados = [], []
    for q in (1, 2, 3, 4):
        try:
            df_q = read_enoe_table(año, q, "sdem")
        except FileNotFoundError:
            continue
        peso_col = "fac_tri" if "fac_tri" in df_q.columns else "fac"
        ent_col = "cve_ent" if "cve_ent" in df_q.columns else "ent"
        partes.append(df_q.rename(columns={peso_col: "_peso", ent_col: "_cve_ent"}))
        trimestres_usados.append(q)
    if not partes:
        raise FileNotFoundError(f"No se encontró ningún trimestre para {año}")
    return pd.concat(partes, ignore_index=True), trimestres_usados


def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cum = np.cumsum(weights) - 0.5 * weights
    cum /= weights.sum()
    return np.interp(q, cum, values)


def weighted_mad(values, weights):
    """Desviación absoluta mediana ponderada — robusta ante colas largas (a diferencia de la std)."""
    mediana = weighted_quantile(values, weights, 0.5)
    return weighted_quantile(np.abs(values - mediana), weights, 0.5)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--monto", type=float, default=1000.0, help="Monto X en pesos (default 1000)")
    parser.add_argument("--año", type=int, default=2026, choices=range(2005, 2027), metavar="[2005-2026]")
    parser.add_argument("--guardar", action="store_true", help="Guardar resultado en dashboard_data/ (opcional)")
    grupo_t = parser.add_mutually_exclusive_group()
    grupo_t.add_argument("--trimestre", type=int, default=None, choices=[1, 2, 3, 4])
    grupo_t.add_argument("--todos-trimestres", action="store_true", dest="todos_trimestres",
                         help="Usar todos los trimestres disponibles del año (más muestra, más precisión)")
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--sector", type=int, choices=SCIAN.keys(), metavar="[1-20]",
                        help="Filtrar por sector económico SCIAN (ver catálogo abajo)")
    grupo.add_argument("--ocupacion", type=int, choices=C_OCU11C.keys(), metavar="[1-10]",
                        help="Filtrar por tipo de ocupación c_ocu11c (ver catálogo abajo)")
    args = parser.parse_args()

    if args.todos_trimestres:
        df, trimestres_usados = cargar_enoe_año_completo(args.año)
        peso_col, ent_col = "_peso", "_cve_ent"
    else:
        trimestre = args.trimestre if args.trimestre is not None else 1
        if (args.año, trimestre) == (2020, 2):
            parser.error("2020Q2 no existe (hueco COVID, ver ENOE DATA_OVERVIEW.md)")
        df = read_enoe_table(args.año, trimestre, "sdem")
        peso_col = "fac_tri" if "fac_tri" in df.columns else "fac"
        ent_col = "cve_ent" if "cve_ent" in df.columns else "ent"
        trimestres_usados = [trimestre]

    filtro = (df["clase2"] == 1) & (df["ingocup"] > 0) & (df["hrsocup"] > 0) & (df[peso_col] > 0)
    etiqueta_filtro = None
    if args.sector is not None:
        filtro &= df["scian"] == args.sector
        etiqueta_filtro = f"Sector: {SCIAN[args.sector]}"
    elif args.ocupacion is not None:
        filtro &= df["c_ocu11c"] == args.ocupacion
        etiqueta_filtro = f"Ocupación: {C_OCU11C[args.ocupacion]}"

    d = df[filtro].copy()
    d["ingreso_hora"] = d["ingocup"] / (d["hrsocup"] * 4.345)

    filas = []
    for cve, g in d.groupby(ent_col):
        valores, pesos = g["ingreso_hora"].to_numpy(), g[peso_col].to_numpy()
        p25, p50, p75, p95, p99 = weighted_quantile(valores, pesos, [0.25, 0.5, 0.75, 0.95, 0.99])
        filas.append({"cve_ent": int(cve), "entidad": NOMBRE.get(int(cve), str(cve)), "n": len(g),
                      "ingreso_hora_p50": p50, "ingreso_hora_p75": p75, "ingreso_hora_p95": p95,
                      "ingreso_hora_p99": p99, "ingreso_hora_iqr": p75 - p25,
                      "ingreso_hora_mad": weighted_mad(valores, pesos)})
    valores, pesos = d["ingreso_hora"].to_numpy(), d[peso_col].to_numpy()
    p25, p50, p75, p95, p99 = weighted_quantile(valores, pesos, [0.25, 0.5, 0.75, 0.95, 0.99])
    filas.append({"cve_ent": 0, "entidad": "Nacional", "n": len(d),
                  "ingreso_hora_p50": p50, "ingreso_hora_p75": p75, "ingreso_hora_p95": p95,
                  "ingreso_hora_p99": p99, "ingreso_hora_iqr": p75 - p25,
                  "ingreso_hora_mad": weighted_mad(valores, pesos)})

    tabla = pd.DataFrame(filas)
    tabla["ingreso_hora_cv"] = tabla["ingreso_hora_mad"] / tabla["ingreso_hora_p50"]
    tabla["alta_dispersion"] = tabla["ingreso_hora_cv"] > 1  # mad no puede escapar tanto a colas largas, umbral se mantiene
    for p in ("p50", "p75", "p95", "p99"):
        tabla[f"horas_{p}"] = args.monto / tabla[f"ingreso_hora_{p}"]
        # propagación de error: horas = monto/ingreso_hora -> mad(horas) ≈ |d horas/d ingreso_hora| * mad(ingreso_hora)
        tabla[f"horas_{p}_mad"] = args.monto * tabla["ingreso_hora_mad"] / tabla[f"ingreso_hora_{p}"] ** 2

    assert (tabla["ingreso_hora_p50"] <= tabla["ingreso_hora_p75"]).all()
    assert (tabla["ingreso_hora_p75"] <= tabla["ingreso_hora_p95"]).all()
    assert (tabla["ingreso_hora_p95"] <= tabla["ingreso_hora_p99"]).all()
    if etiqueta_filtro is None:
        assert tabla["cve_ent"].nunique() == 33  # 32 entidades + Nacional
    elif tabla["cve_ent"].nunique() < 33:
        faltantes = sorted(set(NOMBRE.values()) - set(tabla["entidad"]))
        print(f"[aviso] {len(faltantes)} entidad(es) sin muestra suficiente para este filtro: {faltantes}\n")

    tabla = tabla.sort_values("horas_p50", ascending=False)

    if args.todos_trimestres:
        qs = ",".join(str(q) for q in trimestres_usados)
        print(f"{args.año} (trimestres {qs}) — monto X = ${args.monto:,.2f}")
        if len(trimestres_usados) < 4:
            print(f"[aviso] Solo {len(trimestres_usados)}/4 trimestres disponibles para {args.año}.\n")
    else:
        print(f"{args.año}Q{trimestres_usados[0]} — monto X = ${args.monto:,.2f}")
    if etiqueta_filtro:
        print(etiqueta_filtro)

    en_ventana = [q for q in trimestres_usados if (args.año, q) in ENOEN_WINDOW]
    fuera_ventana = [q for q in trimestres_usados if (args.año, q) not in ENOEN_WINDOW]
    if en_ventana and fuera_ventana:
        print("[aviso] Se mezclan trimestres ENOE estándar y ENOEN (metodología de encuesta distinta, "
              "cara-a-cara vs telefónica) dentro del mismo año — no es solo más muestra, es mezclar dos "
              "formas de medir.\n")
    elif en_ventana:
        print("[aviso] Trimestre(s) en ventana ENOEN (2020Q3-2022Q4): no comparable directamente con otras eras.\n")

    if args.guardar:
        sufijo_filtro = f"_scian{args.sector}" if args.sector is not None else (
            f"_ocup{args.ocupacion}" if args.ocupacion is not None else "")
        sufijo_año = "_anual" if args.todos_trimestres else f"t{trimestres_usados[0]}"
        out_path = Path(f"dashboard_data/horas_trabajo_por_entidad_{args.año}{sufijo_año}{sufijo_filtro}.parquet")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tabla.to_parquet(out_path, index=False)
        print(f"Guardado → {out_path}\n")

    def fmt(valor, mad, ancho):
        return f"{valor:.1f}/±{mad:.1f}".rjust(ancho)

    n_disp = int(tabla["alta_dispersion"].sum())
    if n_disp:
        print(f"[aviso] {n_disp} entidad(es) con dispersión alta (cv=mad/mediana>1, "
              f"marcadas con ⚠)\n")

    print(f"{'Entidad':<22} {'n':>6} {'IQR':>8} {'$/h p50':>13} {'$/h p75':>13} {'$/h p95':>13} {'$/h p99':>13} "
          f"{'h p50':>12} {'h p75':>12} {'h p95':>12} {'h p99':>12}")
    for r in tabla.itertuples():
        nombre = r.entidad + (" ⚠" if r.alta_dispersion else "")
        print(f"{nombre:<22} {r.n:>6} {r.ingreso_hora_iqr:>8.1f} "
              f"{fmt(r.ingreso_hora_p50, r.ingreso_hora_mad, 13)} "
              f"{fmt(r.ingreso_hora_p75, r.ingreso_hora_mad, 13)} "
              f"{fmt(r.ingreso_hora_p95, r.ingreso_hora_mad, 13)} "
              f"{fmt(r.ingreso_hora_p99, r.ingreso_hora_mad, 13)} "
              f"{fmt(r.horas_p50, r.horas_p50_mad, 12)} "
              f"{fmt(r.horas_p75, r.horas_p75_mad, 12)} "
              f"{fmt(r.horas_p95, r.horas_p95_mad, 12)} "
              f"{fmt(r.horas_p99, r.horas_p99_mad, 12)}")


if __name__ == "__main__":
    main()
