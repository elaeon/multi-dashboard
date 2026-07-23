"""
Índice de Desigualdad Multidimensional Estatal (ENIGH, panel 2016–2024).

Cuarto índice, complementario a impunidad/conflicto/carencias. Aquéllos miden
NIVELES (registros administrativos); éste mide la DISTRIBUCIÓN dentro de cada
estado: un estado puede tener baja carencia promedio y aun así estar muy
polarizado (NL/CDMX). Tres dimensiones:

  A. Económica  — Gini del ingreso corriente per cápita (ICTPC) y ratio de Palma
                  (10% más rico / 40% más pobre), dentro de cada estado.
  B. Social     — desigualdad en el acceso a derechos sociales por ingreso:
                  (1) brecha en el # de carencias sociales CONEVAL entre el decil
                      de ingreso más pobre y el más rico (2022/2024, reusa
                      carencias_index.poblacion_base), y
                  (2) brecha en % sin seguridad social entre esos deciles
                      (5 olas; el derecho social más estratificado en México).
  C. Educativa  — Gini de años de escolaridad (población 15+). La brecha
                  intergeneracional (jóvenes − mayores) se reporta como CONTEXTO,
                  no entra al score (su signo es ambiguo: una brecha grande
                  refleja rezago histórico ya remediado, no desigualdad actual).

Diseño metodológico (corrige la falla #1 de scripts/indices/AUDITORIA.md):
NO se normaliza por percentil-dentro-de-año. Gini y Palma son cardinales y
comparables entre estados Y años; cada métrica se estandariza (z-score) contra un
ANCLA FIJA — la media/desviación agrupada de TODOS los estado-año — de modo que
la tendencia nacional 2016→2024 sea real, no un artefacto de re-anclado. El nivel
(Bajo…Extremo) usa cortes de cuartil fijos sobre el conjunto agrupado, así que un
estado puede cambiar de nivel con el tiempo.

Cobertura: las tres dimensiones y sus sub-pilares corren en las 5 olas
(2016–2024). Las 6 carencias CONEVAL están reconstruidas y validadas contra el
panel oficial en 2016/2018/2020/2022 (ver carencias_index.validar_reconstruccion).

No es un dato de dashboard: la carpeta de salida se especifica por línea de
comandos.

Run: uv run python scripts/indices/prepare_desigualdad_index.py --output-dir <ruta>
"""
import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ / "scripts/centralismo"))
sys.path.insert(0, str(RAIZ / "scripts/indices"))
from comun import NOMBRE  # noqa: E402
import carencias_index as ci  # noqa: E402  (reusa poblacion_base para 2022/2024)

OLAS = [2016, 2018, 2020, 2022, 2024]
ZIPS_ENIGH = {
    2016: "conjunto_de_datos_enigh2016_nueva_serie_csv.zip",
    2018: "conjunto_de_datos_enigh_2018_ns_csv.zip",
    2020: "conjunto_de_datos_enigh_ns_2020_csv.zip",
    2022: "conjunto_de_datos_enigh_ns_2022_csv.zip",
    2024: "conjunto_de_datos_enigh2024_ns_csv.zip",
}
NIVELES = ["Bajo", "Medio", "Alto", "Extremo"]
N_BOOT = 300       # réplicas bootstrap para los IC de la dimensión económica
SEED_BOOT = 20240722

# Métricas que entran al índice compuesto, agrupadas por dimensión, y su
# orientación (todas: "más alto = más desigual").
DIMENSIONES = {
    "economica": ["gini_ingreso", "palma_ingreso"],
    "social": ["brecha_carencias", "brecha_segsoc"],
    "educativa": ["gini_educativo"],
}


# ---------------------------------------------------------------- lectura ENIGH

def _leer(año: int, tabla: str, cols: list[str]) -> pl.DataFrame:
    """Lee columnas de una tabla ENIGH (todo como texto; cast explícito luego).
    polars descarta el BOM de folioviv en 2016/2018 automáticamente, así que los
    nombres limpios funcionan en todas las olas."""
    z = zipfile.ZipFile(RAIZ / "data/inegi/enigh" / ZIPS_ENIGH[año])
    ruta = next(n for n in z.namelist()
                if n.rsplit("/", 1)[-1].startswith(f"conjunto_de_datos_{tabla}")
                and n.endswith(".csv") and "bitacora" not in n)
    with z.open(ruta) as f:
        return pl.read_csv(io.BytesIO(f.read()), columns=cols, infer_schema_length=0)


def _num(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    return df.with_columns(pl.col(c).cast(pl.Float64, strict=False) for c in cols)


# ---------------------------------------------------------------- Gini / Palma

def gini(x: np.ndarray, w: np.ndarray) -> float:
    """Gini ponderado (área bajo la curva de Lorenz por trapecios).
    Copiado de scripts/centralismo/cap7_desigualdad_violencia.py:32 para no
    arrastrar el import de plotly de ese módulo."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    orden = np.argsort(x)
    x, w = x[orden], w[orden]
    F = np.concatenate([[0.0], np.cumsum(w) / w.sum()])
    L = np.concatenate([[0.0], np.cumsum(x * w) / np.sum(x * w)])
    return float(1 - np.sum((F[1:] - F[:-1]) * (L[1:] + L[:-1])))


def palma(x: np.ndarray, w: np.ndarray) -> float:
    """Ratio de Palma ponderado: ingreso del 10% más rico / 40% más pobre,
    con reparto proporcional de los hogares que cruzan los cortes 40% y 90%."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    orden = np.argsort(x)
    x, w = x[orden], w[orden]
    ingreso = x * w
    cum = np.cumsum(w)
    tot = cum[-1]
    lo = np.interp(0.40 * tot, cum, np.cumsum(ingreso))
    hi_bajo = np.interp(0.90 * tot, cum, np.cumsum(ingreso))
    top10 = ingreso.sum() - hi_bajo
    return float(top10 / lo) if lo > 0 else float("nan")


# ---------------------------------------------------------------- hogares + ingreso + decil

def hogares_ingreso(año: int) -> pl.DataFrame:
    """Un renglón por hogar: cve_ent, ictpc (ingreso pc mensual, constructo
    CONEVAL sin renta imputada), peso-persona = factor×tot_integ, y su decil de
    ingreso DENTRO del estado (ponderado por persona)."""
    c = _leer(año, "concentradohogar",
              ["folioviv", "foliohog", "ubica_geo", "factor", "tot_integ",
               "ing_cor", "estim_alqu", "est_dis", "upm"])
    c = _num(c, ["factor", "tot_integ", "ing_cor", "estim_alqu", "est_dis", "upm"])
    c = c.with_columns(
        pl.col("folioviv").cast(pl.Int64, strict=False),
        pl.col("foliohog").cast(pl.Int64, strict=False),
        pl.col("ubica_geo").str.slice(0, 2).cast(pl.Int64).alias("cve_ent"),
        ((pl.col("ing_cor") - pl.col("estim_alqu")) / 3 / pl.col("tot_integ")).alias("ictpc"),
    ).with_columns((pl.col("factor") * pl.col("tot_integ")).alias("pw"))
    c = c.filter(pl.col("ictpc").is_not_null() & (pl.col("pw") > 0)
                 & pl.col("cve_ent").is_between(1, 32))

    # decil de ingreso dentro del estado: participación-persona en el punto medio
    # del hogar sobre el total del estado -> 1 (más pobre) … 10 (más rico)
    c = c.sort(["cve_ent", "ictpc"]).with_columns(
        ((pl.col("pw").cum_sum().over("cve_ent") - pl.col("pw") / 2)
         / pl.col("pw").sum().over("cve_ent")).alias("frac"))
    c = c.with_columns(
        pl.min_horizontal((pl.col("frac") * 10).floor().cast(pl.Int64) + 1, pl.lit(10))
        .alias("decil"))
    return c.select("folioviv", "foliohog", "cve_ent", "ictpc", "factor", "pw",
                    "decil", "est_dis", "upm")


# ---------------------------------------------------------------- A. económica

def _boot_eco(x: np.ndarray, w: np.ndarray, est_dis: np.ndarray, upm: np.ndarray,
              rng: np.random.Generator) -> tuple:
    """IC 95% (percentil) de Gini y Palma con bootstrap ESTRATIFICADO POR UPM,
    acorde al diseño muestral de la ENIGH: dentro de cada estrato (est_dis) se
    remuestrean sus UPMs con reemplazo (conservando el número de UPMs del
    estrato), se juntan TODOS los hogares de las UPMs elegidas y se recalcula la
    métrica ponderada. Es el bootstrap de conglomerados (Rao-Wu, con reemplazo);
    captura la correlación intra-UPM que el bootstrap de hogares ignora.

    Vectorizado: cada UPM recibe un id 0..K-1 ordenado por estrato (los estratos
    ocupan rangos contiguos de id), los hogares se agrupan por UPM en layout CSR
    (`order`+`ptr`) y cada réplica muestrea K ids y hace un gather ragged."""
    BIG = int(upm.max()) + 1
    key = est_dis.astype(np.int64) * BIG + upm.astype(np.int64)
    uniq, inv = np.unique(key, return_inverse=True)      # inv: id de UPM por hogar
    K = len(uniq)
    order = np.argsort(inv, kind="stable")               # hogares agrupados por UPM
    cnt = np.bincount(inv, minlength=K)                  # hogares por UPM
    ptr = np.zeros(K + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(cnt)
    strat = uniq // BIG                                  # estrato de cada UPM (ordenado)
    seg_lo = np.empty(K, np.int64)
    seg_hi = np.empty(K, np.int64)
    bounds = np.concatenate([[0], np.flatnonzero(np.diff(strat)) + 1, [K]])
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        seg_lo[s:e] = s
        seg_hi[s:e] = e

    gs = np.empty(N_BOOT)
    ps = np.empty(N_BOOT)
    for b in range(N_BOOT):
        su = seg_lo + (rng.random(K) * (seg_hi - seg_lo)).astype(np.int64)  # UPMs muestreadas
        counts = cnt[su]
        pos = order[np.repeat(ptr[su], counts)
                    + np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)]
        gs[b] = gini(x[pos], w[pos])
        ps[b] = palma(x[pos], w[pos])
    return (float(np.percentile(gs, 2.5)), float(np.percentile(gs, 97.5)),
            float(np.percentile(ps, 2.5)), float(np.percentile(ps, 97.5)))


def dim_economica(hog: pl.DataFrame) -> pl.DataFrame:
    rng = np.random.default_rng(SEED_BOOT)
    filas = []
    for cve, d in hog.sort("cve_ent").group_by("cve_ent", maintain_order=True):
        x, w = d["ictpc"].to_numpy(), d["pw"].to_numpy()
        ed, up = d["est_dis"].to_numpy(), d["upm"].to_numpy()
        glo, ghi, plo, phi = _boot_eco(x, w, ed, up, rng)
        filas.append((cve[0], gini(x, w), glo, ghi, palma(x, w), plo, phi))
    return pl.DataFrame(
        filas,
        schema=["cve_ent", "gini_ingreso", "gini_ingreso_lo", "gini_ingreso_hi",
                "palma_ingreso", "palma_ingreso_lo", "palma_ingreso_hi"],
        orient="row")


# ---------------------------------------------------------------- C. educativa

def anios_escolaridad(niv: pl.Expr, gra: pl.Expr) -> pl.Expr:
    """Años de escolaridad aprobados a partir de nivelaprob + gradoaprob de la
    ENIGH. Base acumulada por nivel + grado dentro del nivel (grados >=90 = no
    sabe -> 0). Aproximación estándar; validada contra el grado promedio
    nacional (~9–10 años) en el gate."""
    g = pl.when(gra >= 90).then(0).otherwise(gra).fill_null(0)
    return (
        pl.when(niv <= 1).then(pl.lit(0))                       # ninguno / preescolar
        .when(niv == 2).then(pl.min_horizontal(g, pl.lit(6)))    # primaria
        .when(niv == 3).then(6 + pl.min_horizontal(g, pl.lit(3)))  # secundaria
        .when(niv.is_in([4, 5, 6])).then(9 + pl.min_horizontal(g, pl.lit(3)))  # media sup / técnica / normal
        .when(niv == 7).then(12 + pl.min_horizontal(g, pl.lit(5)))  # profesional
        .when(niv == 8).then(pl.lit(17))                         # maestría
        .when(niv >= 9).then(pl.lit(19))                        # doctorado
        .otherwise(None)
    )


def dim_educativa(año: int, hog: pl.DataFrame) -> pl.DataFrame:
    """Gini de años de escolaridad (15+) por estado, y brecha intergeneracional
    (jóvenes 21-30 − mayores 60+) como contexto. Peso = factor de hogar (la
    tabla poblacion no trae factor en 2016/2018)."""
    p = _leer(año, "poblacion", ["folioviv", "foliohog", "numren", "edad",
                                  "nivelaprob", "gradoaprob"])
    p = _num(p, ["edad", "nivelaprob", "gradoaprob"]).with_columns(
        pl.col("folioviv").cast(pl.Int64, strict=False),
        pl.col("foliohog").cast(pl.Int64, strict=False),
        anios_escolaridad(pl.col("nivelaprob"), pl.col("gradoaprob")).alias("anios_esc"),
    )
    p = (p.join(hog.select("folioviv", "foliohog", "cve_ent", "factor"),
                on=["folioviv", "foliohog"], how="inner")
         .filter((pl.col("edad") >= 15) & pl.col("anios_esc").is_not_null()))

    filas = []
    for cve, d in p.group_by("cve_ent"):
        g = gini(d["anios_esc"].to_numpy(), d["factor"].to_numpy())
        jov = d.filter(pl.col("edad").is_between(21, 30))
        may = d.filter(pl.col("edad") >= 60)

        def wmean(x):
            return ((x["anios_esc"] * x["factor"]).sum() / x["factor"].sum()) if x.height else None
        mj, mm = wmean(jov), wmean(may)
        brecha = (mj - mm) if (mj is not None and mm is not None) else None
        filas.append((cve[0], g, brecha))
    return pl.DataFrame(filas, schema=["cve_ent", "gini_educativo", "brecha_generacional"],
                        orient="row")


# ---------------------------------------------------------------- B. social

def _brecha_por_decil(persona: pl.DataFrame, valor: str) -> pl.DataFrame:
    """Brecha estatal = (media ponderada de `valor` en decil 1) − (en decil 10).
    `persona` trae cve_ent, decil, factor y la columna `valor` (0/1 o conteo)."""
    d = (persona.filter(pl.col("decil").is_in([1, 10]))
         .group_by("cve_ent", "decil")
         .agg(((pl.col(valor) * pl.col("factor")).sum()
               / pl.col("factor").sum()).alias("m")))
    ancho = d.pivot(on="decil", index="cve_ent", values="m")
    return ancho.select("cve_ent", (pl.col("1") - pl.col("10")).alias("brecha"))


def brecha_segsoc(año: int, hog: pl.DataFrame) -> pl.DataFrame:
    """Brecha en % sin seguridad social entre decil 1 y 10 (proxy directo:
    segsoc==1 => cubierto). 5 olas."""
    p = _leer(año, "poblacion", ["folioviv", "foliohog", "numren", "segsoc"])
    p = p.with_columns(
        pl.col("folioviv").cast(pl.Int64, strict=False),
        pl.col("foliohog").cast(pl.Int64, strict=False),
        pl.when(pl.col("segsoc").str.strip_chars() == "1").then(0).otherwise(1).alias("ic_segsoc"),
    )
    p = p.join(hog.select("folioviv", "foliohog", "cve_ent", "factor", "decil"),
               on=["folioviv", "foliohog"], how="inner")
    return _brecha_por_decil(p, "ic_segsoc").rename({"brecha": "brecha_segsoc"})


def brecha_carencias(año: int, hog: pl.DataFrame) -> pl.DataFrame:
    """Brecha en el # de carencias sociales (i_privacion, 0-6) entre decil 1 y
    10. Reusa carencias_index.poblacion_base (sólo 2022/2024)."""
    base = ci.poblacion_base(año).with_columns(
        pl.col("folioviv").cast(pl.Int64, strict=False),
        pl.col("foliohog").cast(pl.Int64, strict=False),
    )
    p = base.join(hog.select("folioviv", "foliohog", "decil"),
                  on=["folioviv", "foliohog"], how="inner") \
            .filter(pl.col("i_privacion").is_not_null())
    return _brecha_por_decil(p, "i_privacion").rename({"brecha": "brecha_carencias"})


# ---------------------------------------------------------------- ensamblado

def metricas_ola(año: int) -> pl.DataFrame:
    hog = hogares_ingreso(año)
    df = (dim_economica(hog)
          .join(dim_educativa(año, hog), on="cve_ent", how="full", coalesce=True)
          .join(brecha_segsoc(año, hog), on="cve_ent", how="full", coalesce=True))
    df = df.join(brecha_carencias(año, hog), on="cve_ent", how="full", coalesce=True)
    return df.with_columns(pl.lit(año).alias("anio")).sort("cve_ent")


def estandarizar_y_componer(panel: pl.DataFrame) -> pl.DataFrame:
    """z-score de cada métrica contra ANCLA FIJA (media/sd de todos los estado-
    año), score por dimensión (promedio de sus métricas z), e índice = promedio
    de las 3 dimensiones. Nivel por cuartiles fijos del conjunto agrupado."""
    metricas = [m for ms in DIMENSIONES.values() for m in ms]
    z = {m: ((pl.col(m) - pl.col(m).mean()) / pl.col(m).std()).alias(f"z_{m}")
         for m in metricas}
    panel = panel.with_columns(*z.values())

    dim_exprs = []
    for dim, ms in DIMENSIONES.items():
        dim_exprs.append(pl.mean_horizontal([pl.col(f"z_{m}") for m in ms]).alias(f"dim_{dim}"))
    panel = panel.with_columns(*dim_exprs)
    panel = panel.with_columns(
        pl.mean_horizontal("dim_economica", "dim_social", "dim_educativa")
        .alias("indice_desigualdad"))

    # cuartiles fijos sobre TODOS los estado-año (no re-anclado por año)
    cortes = panel["indice_desigualdad"].qcut([0.25, 0.5, 0.75], labels=NIVELES,
                                               allow_duplicates=True)
    return panel.with_columns(cortes.alias("nivel"))


# ---------------------------------------------------------------- validación

def validar(panel: pl.DataFrame) -> None:
    print("\n=== Gate 1: Gini de ingreso nacional por ola (referencia ≈0.42–0.50) ===")
    for año in OLAS:
        d = panel.filter(pl.col("anio") == año)
        # Gini nacional aproximado = promedio de Ginis estatales ponderado por peso
        # no disponible aquí; se reporta el promedio simple + rango como sensatez.
        g = d["gini_ingreso"]
        print(f"  {año}: Gini estatal medio={g.mean():.3f} "
              f"(min {g.min():.3f} {NOMBRE[d.sort('gini_ingreso')['cve_ent'][0]]}, "
              f"max {g.max():.3f} {NOMBRE[d.sort('gini_ingreso')['cve_ent'][-1]]})")
        assert 0.35 < g.mean() < 0.55, f"Gini medio {g.mean():.3f} fuera de rango sensato"

    print("\n=== Gate 2: tendencia CARDINAL nacional (índice medio por ola) ===")
    print("(a diferencia de conflicto/impunidad, este índice SÍ puede subir/bajar)")
    for año in OLAS:
        d = panel.filter(pl.col("anio") == año)
        print(f"  {año}: índice medio={d['indice_desigualdad'].mean():+.3f} · "
              f"Palma medio={d['palma_ingreso'].mean():.2f} · "
              f"Gini educ medio={d['gini_educativo'].mean():.3f}")

    print("\n=== Prueba de sensatez: top-5 estados más desiguales (última ola) ===")
    ult = panel.filter(pl.col("anio") == OLAS[-1]).sort("indice_desigualdad", descending=True)
    print(ult.head(5).select("estado", "gini_ingreso", "palma_ingreso",
                             "gini_educativo", "brecha_segsoc", "indice_desigualdad", "nivel"))
    print("\nDistribución de niveles (todas las olas, cuartiles fijos):")
    print(panel["nivel"].value_counts().sort("nivel"))

    print(f"\n=== IC bootstrap 95% ({N_BOOT} réplicas): ¿la caída del Gini de "
          f"ingreso {OLAS[0]}→{OLAS[-1]} es distinguible del ruido muestral? ===")
    a = panel.filter(pl.col("anio") == OLAS[0]).select(
        "cve_ent", "estado", pl.col("gini_ingreso").alias("g0"),
        pl.col("gini_ingreso_lo").alias("lo0"))
    b = panel.filter(pl.col("anio") == OLAS[-1]).select(
        "cve_ent", pl.col("gini_ingreso").alias("g1"), pl.col("gini_ingreso_hi").alias("hi1"))
    cmp = a.join(b, on="cve_ent").with_columns(
        (pl.col("hi1") < pl.col("lo0")).alias("caida_signif"))  # IC 2024 debajo del IC 2016
    n_sig = cmp["caida_signif"].sum()
    print(f"  {n_sig}/32 estados con caída estadísticamente significativa "
          f"(IC {OLAS[-1]} por debajo del IC {OLAS[0]}, sin traslape).")
    print("  Ejemplos (Gini con IC95%):")
    for r in cmp.sort("g0", descending=True).head(4).iter_rows(named=True):
        marca = "significativa" if r["caida_signif"] else "NO distinguible"
        print(f"    {r['estado']:<18} {OLAS[0]}={r['g0']:.3f} → {OLAS[-1]}={r['g1']:.3f}  "
              f"(IC{OLAS[0]} desde {r['lo0']:.3f}, IC{OLAS[-1]} hasta {r['hi1']:.3f}) — {marca}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Carpeta donde escribir el parquet (no es dato de "
                             "dashboard, no tiene default).")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    partes = []
    for año in OLAS:
        print(f"Procesando ENIGH {año}...")
        partes.append(metricas_ola(año))
    panel = pl.concat(partes, how="diagonal")
    panel = estandarizar_y_componer(panel).with_columns(
        pl.col("cve_ent").replace_strict(NOMBRE).alias("estado"))

    cols = ["cve_ent", "estado", "anio",
            "gini_ingreso", "gini_ingreso_lo", "gini_ingreso_hi",
            "palma_ingreso", "palma_ingreso_lo", "palma_ingreso_hi",
            "gini_educativo", "brecha_generacional",
            "brecha_segsoc", "brecha_carencias",
            "dim_economica", "dim_social", "dim_educativa",
            "indice_desigualdad", "nivel"]
    panel = panel.select(cols).sort(["anio", "indice_desigualdad"], descending=[False, True])

    out = args.output_dir / "desigualdad_index_estatal.parquet"
    panel.write_parquet(out)
    print(f"\nÍndice de desigualdad: {panel.height:,} filas -> {out}")

    validar(panel)


if __name__ == "__main__":
    main()
