"""
Poder adquisitivo nacional a través del tiempo: salario/hora ENOE (nacional,
percentiles 50/75/95/99) deflactado por INPC, expresado en pesos constantes
de un año base.

ingreso_hora = ingocup / (hrsocup * 4.345)  (igual que horas_trabajo_para_monto.py)
valor_real[año] = valor_nominal[año] * indice_inpc[año_base] / indice_inpc[año]

Fuente de precios: data/inegi/inpc/Indicadores20260719173842.xls — INPC
nacional (única área geográfica disponible; no hay INPC por ciudad ni
CONEVAL Línea de Bienestar por entidad en este repo, de ahí que esta serie
sea SOLO nacional, sin comparación entre estados). El archivo trae
únicamente "Inflación mensual interanual (Variación Porcentual)", no el
nivel del índice — se reconstruye un índice encadenado promediando la
inflación interanual mensual disponible por año. Esto es una APROXIMACIÓN
(no un encadenamiento mes a mes del nivel real del INPC).

Fuente de salarios: data/inegi/enoe/ (sdem, 2005-2026), reutilizando
cargar_enoe_año_completo/weighted_quantile de horas_trabajo_para_monto.py.
El rango de años queda acotado por la cobertura INPC (2008 en adelante).

Output: dashboard_data/poder_adquisitivo_nacional.parquet
Run: uv run python scripts/datatable/poder_adquisitivo_nacional.py [--año-base 2026] [--guardar]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from horas_trabajo_para_monto import ENOEN_WINDOW, cargar_enoe_año_completo, weighted_quantile

INPC_PATH = Path(__file__).resolve().parents[2] / "data" / "inegi" / "inpc" / "Indicadores20260719173842.xls"
AÑO_INPC_INICIO = 2008
AÑO_ENOE_FIN = 2026


def cargar_indice_inpc() -> pd.DataFrame:
    """Reconstruye un índice de precios anual (base=primer año=100) a partir
    de la inflación interanual mensual del INPC (única serie disponible en
    el archivo). Devuelve columnas: año, meses_disponibles, indice."""
    raw = pd.read_excel(INPC_PATH, header=None, skiprows=6)
    raw.columns = ["periodo", "area", "interanual_pct"]
    raw = raw[raw["area"] == "00 Estados Unidos Mexicanos"].copy()
    raw["año"] = raw["periodo"].str.split("/").str[0].astype(int)

    anual = raw.groupby("año")["interanual_pct"].agg(["mean", "count"]).reset_index()
    anual = anual.rename(columns={"mean": "promedio_interanual_pct", "count": "meses_disponibles"})
    anual = anual.sort_values("año").reset_index(drop=True)

    indices = [100.0]
    for i in range(1, len(anual)):
        indices.append(indices[-1] * (1 + anual.loc[i, "promedio_interanual_pct"] / 100))
    anual["indice"] = indices
    return anual


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--año-base", type=int, default=None, dest="año_base",
                        help="Año en cuyos pesos se expresan los valores reales (default: último año disponible)")
    parser.add_argument("--guardar", action="store_true", help="Guardar resultado en dashboard_data/ (opcional)")
    args = parser.parse_args()

    inpc = cargar_indice_inpc()
    años_inpc = set(inpc["año"])

    filas_enoe = []
    for año in range(AÑO_INPC_INICIO, AÑO_ENOE_FIN + 1):
        if año not in años_inpc:
            continue
        try:
            df, trimestres_usados = cargar_enoe_año_completo(año)
        except FileNotFoundError:
            continue
        filtro = (df["clase2"] == 1) & (df["ingocup"] > 0) & (df["hrsocup"] > 0) & (df["_peso"] > 0)
        d = df[filtro].copy()
        d["ingreso_hora"] = d["ingocup"] / (d["hrsocup"] * 4.345)

        valores, pesos = d["ingreso_hora"].to_numpy(), d["_peso"].to_numpy()
        p50, p75, p95, p99 = weighted_quantile(valores, pesos, [0.5, 0.75, 0.95, 0.99])
        en_ventana_enoen = any((año, q) in ENOEN_WINDOW for q in trimestres_usados)
        fuera_ventana_enoen = any((año, q) not in ENOEN_WINDOW for q in trimestres_usados)
        filas_enoe.append({
            "año": año, "n": len(d), "trimestres_usados": len(trimestres_usados),
            "ingreso_hora_p50": p50, "ingreso_hora_p75": p75, "ingreso_hora_p95": p95, "ingreso_hora_p99": p99,
            "metodologia_mixta": en_ventana_enoen and fuera_ventana_enoen,
        })

    if not filas_enoe:
        raise FileNotFoundError(f"No se encontró ningún año ENOE en el rango {AÑO_INPC_INICIO}-{AÑO_ENOE_FIN}")

    tabla = pd.DataFrame(filas_enoe).merge(inpc[["año", "meses_disponibles", "indice"]], on="año", how="left")
    assert tabla["indice"].notna().all()

    año_base = args.año_base if args.año_base is not None else int(tabla["año"].max())
    if año_base not in set(tabla["año"]):
        parser.error(f"--año-base {año_base} no está en el rango disponible ({tabla['año'].min()}-{tabla['año'].max()})")
    indice_base = float(tabla.loc[tabla["año"] == año_base, "indice"].iloc[0])

    for p in ("p50", "p75", "p95", "p99"):
        tabla[f"ingreso_hora_{p}_real"] = tabla[f"ingreso_hora_{p}"] * indice_base / tabla["indice"]

    tabla = tabla.sort_values("año").reset_index(drop=True)
    assert (tabla["ingreso_hora_p50"] <= tabla["ingreso_hora_p75"]).all()
    assert (tabla["ingreso_hora_p75"] <= tabla["ingreso_hora_p95"]).all()
    assert (tabla["ingreso_hora_p95"] <= tabla["ingreso_hora_p99"]).all()

    real_p50 = tabla["ingreso_hora_p50_real"]
    tabla["var_pct_interanual_real_p50"] = real_p50.pct_change() * 100
    tabla["var_pct_acumulada_real_p50"] = (real_p50 / real_p50.iloc[0] - 1) * 100

    print(f"Poder adquisitivo nacional del salario/hora (ENOE, p50/p75/p95/p99), pesos constantes de {año_base}")
    print("[aviso] índice INPC reconstruido: promedio de inflación interanual mensual por año "
          "(aproximación, no encadenamiento mes a mes del nivel real del INPC).\n")

    parcial = tabla[tabla["meses_disponibles"] < 12]
    if not parcial.empty:
        for r in parcial.itertuples():
            print(f"[aviso] {r.año}: solo {r.meses_disponibles}/12 meses de INPC disponibles (año parcial).")
    mixta = tabla[tabla["metodologia_mixta"]]
    if not mixta.empty:
        print(f"[aviso] {sorted(mixta['año'].tolist())}: mezcla trimestres ENOE estándar y ENOEN "
              "dentro del mismo año (metodología de encuesta distinta).")
    print()

    print(f"{'Año':>6} {'n':>7} {'trim.':>5} {'$/h p50 nom':>12} {'$/h p50 real':>13} "
          f"{'var % a/a':>10} {'var % acum.':>11}")
    for r in tabla.itertuples():
        var_aa = "" if pd.isna(r.var_pct_interanual_real_p50) else f"{r.var_pct_interanual_real_p50:>+9.1f}%"
        print(f"{r.año:>6} {r.n:>7} {r.trimestres_usados:>5} {r.ingreso_hora_p50:>12.2f} "
              f"{r.ingreso_hora_p50_real:>13.2f} {var_aa:>10} {r.var_pct_acumulada_real_p50:>+10.1f}%")

    acumulado_total = tabla["var_pct_acumulada_real_p50"].iloc[-1]
    caidas = int((tabla["var_pct_interanual_real_p50"] < 0).sum())
    print(f"\nCambio real acumulado del salario/hora mediano, {tabla['año'].iloc[0]}→{tabla['año'].iloc[-1]}: "
          f"{acumulado_total:+.1f}%")
    print(f"Años con caída real año-a-año: {caidas}/{len(tabla) - 1}")

    if args.guardar:
        out_path = Path("dashboard_data/poder_adquisitivo_nacional.parquet")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tabla.to_parquet(out_path, index=False)
        print(f"\nGuardado → {out_path}")


if __name__ == "__main__":
    main()
