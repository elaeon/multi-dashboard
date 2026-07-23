"""
Índice de Conflicto Municipal (rediseño de constructo + método, 2015-2024).

Corrige las siete debilidades de scripts/indices/AUDITORIA.md §2. El Índice A ya no
depende de una sola fuente (SESNSP) ni normaliza por percentil-dentro-de-año.
Estructura por dimensiones (compuesto = promedio de los z-scores de dimensión):

  - letalidad   : homicidio INEGI / 100k          (Deadliness)
  - dano_civil  : feminicidio, secuestro, desaparecidos — TRES indicadores
                  SEPARADOS (§2.5), agrupados para no triple-pesar el daño a civiles
  - coercion    : extorsión SESNSP / 100k          (proxy; cifra negra >98%, §2.1)
  - persistencia: % de meses del año con >=1 homicidio INEGI (Diffusion, §2.3 —
                  ya no suma todos los componentes, ya no es proxy de población)

Fuentes y correcciones:
  - HOMICIDIO por registro de mortalidad INEGI (defunciones, agresiones CIE-10
    X85–Y09, por municipio de OCURRENCIA), NO SESNSP: el registro de mortalidad es
    independiente de la fiscalía y robusto a la reclasificación que sí manipula a
    SESNSP (§2.2). La brecha INEGI↔SESNSP por estado se reporta como validación.
  - FEMINICIDIO, SECUESTRO y EXTORSIÓN sí de SESNSP (INEGI no los distingue), cada
    uno como indicador propio (§2.5).
  - DESAPARECIDOS del RNPDNO (§2.4) — la firma del conflicto ausente en el índice
    viejo. Cubre el 55% de los registros colocable a municipio-año (45% con
    municipio/fecha redactados como 'CONFIDENCIAL' o clave 999); se documenta como
    cobertura parcial, no se imputa. ADVERTENCIA: la redacción es DESIGUAL por
    estado (Jalisco 68%, Nayarit 71%, CDMX 63% vs Chihuahua 9%, Colima 9%), así que
    este pilar es un PISO que subcuenta MÁS a varios de los estados con más
    desapariciones — sesgo sistemático, no ruido. Ver el diagnóstico en validar().

Método (espejo de prepare_impunidad_index.py, corrige §2.6/§2.7):
  - Tasas con ENCOGIMIENTO BAYES-EMPÍRICO (Marshall 1991) hacia la media del año,
    con fuerza ∝ población, sobre conteos suavizados a 3 años — así los municipios
    diminutos con un evento dejan de dominar el ranking.
  - z-score de ANCLA FIJA (media/sd agrupada de todos los municipio-año), no
    percentil-dentro-de-año: la escalada nacional 2015–2024 se vuelve visible.
    Nivel por cuartiles fijos del conjunto agrupado.

Capas de CONTEXTO (parquets separados, NO fusionadas ni comparables al índice
municipal — su grano es estatal/nacional): desplazamiento forzado (IDMC GIDD,
nacional por año) y posición de México en el índice global de ACLED.

Sub-índice B (2007-2018, fragmentación OCVED): tabla SEPARADA, sin cambio (aislada,
el audit no la objeta). Ver informe_data/README_grupos_criminales.md.

No es un dato de dashboard: la carpeta de salida se especifica por línea de comandos.

Run: uv run python scripts/indices/prepare_conflict_index.py --output-dir <ruta>
"""
import argparse
import glob
import io
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

RAIZ = Path(__file__).resolve().parent.parent.parent

INCIDENCIA = (RAIZ / "data/incidencia_delictiva/incidencia_fuero_comun/"
              "incidencia_delictiva_fuero_comun.parquet")
CONAPO_ZIP = RAIZ / "data/conapo/proyecciones_poblacion/00_Republica_mexicana.zip"
CONAPO_INNER = "00_Republica_mexicana/3_Indicadores_Dem_00_RM.xlsx"
GRUPOS_OCVED = RAIZ / "informe_data/grupos_panel_ocved.parquet"
DEFUN_DIR = RAIZ / "data/inegi/defunciones"
DESAP_CSV = RAIZ / "data/datamx/desaparecidos/desaparecidos.csv"
IDMC_TS = RAIZ / "data/idmc/IDMC_Internal_Displacement_Conflict-Violence_Disasters.xlsx"
ACLED_XLSX = RAIZ / "data/acled/conflic_index/ACLED_Conflict_Index_2025.xlsx"

MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
         "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]

YEAR_MIN, YEAR_MAX = 2015, 2024
POB_FLOOR = 5000       # municipios por debajo se marcan (pob_baja); EB ya los encoge

# Indicadores de TASA (conteo/pob, encogidos EB); persistencia es proporción aparte.
TASAS = ["homicidio", "feminicidio", "secuestro", "desaparecidos", "extorsion"]
DIMENSIONES = {
    "letalidad": ["homicidio"],
    "dano_civil": ["feminicidio", "secuestro", "desaparecidos"],
    "coercion": ["extorsion"],
    "persistencia": ["persistencia"],
}
NIVELES = ["Bajo", "Medio", "Alto", "Extremo"]


# ---------------------------------------------------------------- población / catálogo

def cargar_poblacion() -> pl.DataFrame:
    with zipfile.ZipFile(CONAPO_ZIP) as z, z.open(CONAPO_INNER) as f:
        buf = io.BytesIO(f.read())
    raw = pl.read_excel(buf)
    return (raw.filter(pl.col("AÑO").is_between(YEAR_MIN, YEAR_MAX))
            .select(pl.col("CLAVE").cast(pl.Int64).alias("cve_mun"),
                    pl.col("AÑO").alias("anio"),
                    pl.col("POB_MIT_MUN").cast(pl.Float64).alias("poblacion")))


def cargar_catalogo_municipios() -> pl.DataFrame:
    return (pl.scan_parquet(INCIDENCIA)
            .select(pl.col("Cve. Municipio").cast(pl.Int64).alias("cve_mun"),
                    pl.col("Municipio").alias("municipio"),
                    pl.col("Entidad").alias("entidad"))
            .unique(subset="cve_mun")
            .collect())


# ---------------------------------------------------------------- homicidio INEGI (patrón oro)

def _defun_zip(anio: int) -> Path:
    for patron in (f"*base_datos_{anio}_csv.zip", f"*generales_{anio}_csv.zip",
                   f"*registradas_{anio}_csv.zip", f"*edr{anio}_csv.zip"):
        hits = glob.glob(str(DEFUN_DIR / patron))
        if hits:
            return Path(hits[0])
    raise FileNotFoundError(f"defunciones {anio}")


def homicidio_inegi(anio: int) -> pl.DataFrame:
    """cve_mun × (homicidio, persistencia) desde defunciones INEGI: agresiones
    CIE-10 X85–Y09, por municipio de OCURRENCIA, ocurridas en el año. persistencia
    = nº de meses distintos con >=1 homicidio / 12. Excluye mun_ocurr 999."""
    zpath = _defun_zip(anio)
    with zipfile.ZipFile(zpath) as z:
        miembro = next(n for n in z.namelist()
                       if n.upper().endswith(".CSV") and "conjunto_de_datos/" in n.replace("\\", "/").lower()
                       and "diccion" not in n.lower() and "catalog" not in n.lower())
        raw = z.read(miembro)
    df = pl.read_csv(io.BytesIO(raw),
                     columns=["causa_def", "ent_ocurr", "mun_ocurr", "anio_ocur", "mes_ocurr"],
                     infer_schema_length=0)
    c = pl.col("causa_def").str.to_uppercase()
    letra, num = c.str.slice(0, 1), c.str.slice(1, 2).cast(pl.Int64, strict=False)
    homicidio = ((letra == "X") & num.is_between(85, 99)) | ((letra == "Y") & num.is_between(0, 9))
    # Los archivos base_datos (2015-2016) traen un tab pegado en las claves
    # ('05\t'); strip antes del cast o se pierden todas las filas.
    df = (df.with_columns(pl.col(k).str.strip_chars().cast(pl.Int64, strict=False)
                          for k in ("ent_ocurr", "mun_ocurr", "anio_ocur", "mes_ocurr"))
          .filter(homicidio & (pl.col("anio_ocur") == anio)
                  & pl.col("ent_ocurr").is_between(1, 32)
                  & (pl.col("mun_ocurr") != 999) & pl.col("mun_ocurr").is_not_null())
          .with_columns((pl.col("ent_ocurr") * 1000 + pl.col("mun_ocurr")).alias("cve_mun")))
    return (df.group_by("cve_mun").agg(
                pl.len().alias("homicidio"),
                (pl.col("mes_ocurr").filter(pl.col("mes_ocurr").is_between(1, 12)).n_unique() / 12)
                .alias("persistencia"))
            .with_columns(pl.lit(anio).alias("anio")))


# ---------------------------------------------------------------- SESNSP (femin/secuestro/extorsión + homicidio para brecha)

def sesnsp_componentes() -> pl.DataFrame:
    """cve_mun × anio × (feminicidio, secuestro, extorsion, homicidio_sesnsp) —
    conteos anuales SESNSP. homicidio_sesnsp sólo para la brecha vs INEGI."""
    lf = pl.scan_parquet(INCIDENCIA).filter(pl.col("Año").is_between(YEAR_MIN, YEAR_MAX))
    comp = (pl.when(pl.col("Subtipo de delito") == "Homicidio doloso").then(pl.lit("homicidio_sesnsp"))
            .when(pl.col("Subtipo de delito") == "Feminicidio").then(pl.lit("feminicidio"))
            .when(pl.col("Tipo de delito") == "Secuestro").then(pl.lit("secuestro"))
            .when(pl.col("Tipo de delito") == "Extorsión").then(pl.lit("extorsion"))
            .otherwise(None).alias("componente"))
    anual = (lf.with_columns(comp).filter(pl.col("componente").is_not_null())
             .with_columns(pl.sum_horizontal([pl.col(m).cast(pl.Float64, strict=False) for m in MESES]).alias("casos"))
             .group_by(["Año", "Cve. Municipio", "componente"])
             .agg(pl.col("casos").sum())
             .collect()
             .pivot(on="componente", index=["Año", "Cve. Municipio"], values="casos")
             .rename({"Año": "anio", "Cve. Municipio": "cve_mun"}))
    return anual.with_columns(pl.col("cve_mun").cast(pl.Int64))


# ---------------------------------------------------------------- desaparecidos (RNPDNO)

def desaparecidos_municipal() -> pl.DataFrame:
    """cve_mun × anio × desaparecidos (RNPDNO). Sólo registros COLOCABLES: municipio
    y fecha no redactados ('CONFIDENCIAL') ni clave 999. Es cobertura parcial (~55%),
    NO se imputa el resto."""
    d = pl.read_csv(DESAP_CSV, infer_schema_length=0)
    d = (d.filter((pl.col("CVE_MUN") != "999") & (pl.col("CVE_MUN") != "CONFIDENCIAL")
                  & (pl.col("FECHA_DESAPARICION") != "CONFIDENCIAL")
                  & pl.col("FECHA_DESAPARICION").is_not_null())
         .with_columns(pl.col("FECHA_DESAPARICION").str.slice(0, 4).cast(pl.Int64, strict=False).alias("anio"),
                       (pl.col("CVE_ENT").cast(pl.Int64, strict=False) * 1000
                        + pl.col("CVE_MUN").cast(pl.Int64, strict=False)).alias("cve_mun"))
         .filter(pl.col("anio").is_between(YEAR_MIN, YEAR_MAX) & pl.col("cve_mun").is_not_null()))
    return d.group_by(["cve_mun", "anio"]).agg(pl.len().alias("desaparecidos"))


# ---------------------------------------------------------------- Bayes empírico + suavizado

def _eb_marshall(O: np.ndarray, pop: np.ndarray) -> np.ndarray:
    """Tasa encogida Bayes-empírica (Marshall 1991) por municipio, para UN año:
    r_i^EB = m + (r_i − m)·A/(A + m/pop_i), con m = tasa global, A = varianza
    entre-áreas (momentos). Encoge fuerte donde pop es chica (§2.7)."""
    tot_pop = pop.sum()
    if tot_pop <= 0:
        return np.zeros_like(O, dtype=float)
    m = O.sum() / tot_pop                      # tasa global (α/β)
    if m <= 0:                                 # indicador sin eventos ese año
        return np.zeros_like(O, dtype=float)
    r = O / pop                                # pop>0 garantizado (CONAPO)
    pbar = pop.mean()
    s2 = (pop * (r - m) ** 2).sum() / tot_pop
    A = max(s2 - m / pbar, 0.0)                # varianza entre-áreas
    if A == 0:                                 # sin señal geográfica -> todo a la media
        return np.full_like(O, m * 1e5, dtype=float)
    C = A / (A + m / pop)                       # encogimiento ∝ pop
    return (m + C * (r - m)) * 1e5             # por 100k


def construir_panel() -> pl.DataFrame:
    """Panel municipio×año con conteos suavizados a 3 años, tasas EB por 100k y
    persistencia suavizada. La rejilla base = todos los municipios con población."""
    poblacion = cargar_poblacion()
    inegi = pl.concat([homicidio_inegi(a) for a in range(YEAR_MIN, YEAR_MAX + 1)])
    sesnsp = sesnsp_componentes()
    desap = desaparecidos_municipal()

    p = (poblacion
         .join(inegi, on=["cve_mun", "anio"], how="left")
         .join(sesnsp, on=["cve_mun", "anio"], how="left")
         .join(desap, on=["cve_mun", "anio"], how="left"))
    cuentas = ["homicidio", "feminicidio", "secuestro", "extorsion", "desaparecidos", "homicidio_sesnsp"]
    p = p.with_columns([pl.col(c).fill_null(0.0) for c in cuentas]
                       + [pl.col("persistencia").fill_null(0.0)])

    # suavizado 3 años (t-1,t,t+1) de conteos y población, por municipio
    p = p.sort(["cve_mun", "anio"])
    roll = {c: pl.col(c).rolling_sum(window_size=3, min_samples=1, center=True).over("cve_mun").alias(f"{c}_s")
            for c in TASAS + ["homicidio_sesnsp"]}
    p = p.with_columns(
        *roll.values(),
        pl.col("poblacion").rolling_sum(window_size=3, min_samples=1, center=True).over("cve_mun").alias("pop_s"),
        pl.col("persistencia").rolling_mean(window_size=3, min_samples=1, center=True).over("cve_mun").alias("persistencia_s"),
    )

    # tasa EB por indicador y año (numpy por año)
    for ind in TASAS:
        col = np.full(p.height, np.nan)
        for anio in range(YEAR_MIN, YEAR_MAX + 1):
            idx = (p["anio"] == anio).to_numpy().nonzero()[0]
            O = p[f"{ind}_s"].to_numpy()[idx]
            pop = p["pop_s"].to_numpy()[idx]
            col[idx] = _eb_marshall(O, pop)
        p = p.with_columns(pl.Series(f"tasa_{ind}", col))

    return p.with_columns((pl.col("poblacion") < POB_FLOOR).alias("pob_baja"))


# ---------------------------------------------------------------- estandarización ancla fija

def estandarizar(panel: pl.DataFrame) -> pl.DataFrame:
    """z-score de cada indicador contra ANCLA FIJA (todos los municipio-año), score
    por dimensión (promedio de sus z), compuesto = promedio de las 4 dimensiones.
    Nivel por cuartiles fijos del conjunto agrupado."""
    metricas = {ind: f"tasa_{ind}" for ind in TASAS}
    metricas["persistencia"] = "persistencia_s"
    z = {ind: ((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(f"z_{ind}")
         for ind, c in metricas.items()}
    panel = panel.with_columns(*z.values())
    dim_exprs = [pl.mean_horizontal([pl.col(f"z_{i}") for i in inds]).alias(f"dim_{d}")
                 for d, inds in DIMENSIONES.items()]
    panel = panel.with_columns(*dim_exprs)
    panel = panel.with_columns(
        pl.mean_horizontal([pl.col(f"dim_{d}") for d in DIMENSIONES]).alias("indice_conflicto"))
    cortes = panel["indice_conflicto"].qcut([0.25, 0.5, 0.75], labels=NIVELES, allow_duplicates=True)
    return panel.with_columns(cortes.alias("nivel"))


# ---------------------------------------------------------------- contexto (no fusionado)

def contexto_desplazamiento() -> pl.DataFrame:
    """Desplazamiento forzado nacional por año (IDMC, serie país-año). Se usa el
    FLUJO anual 'Conflict Internal Displacements' (desplazamientos nuevos), no el
    stock de IDPs (trampa de sobreconteo)."""
    g = pl.read_excel(IDMC_TS, sheet_name="1_Displacement_data")
    nombre = next(c for c in g.columns if c in ("Name", "Country"))
    flujo = next(c for c in g.columns
                 if c.startswith("Conflict Internal Displacements") and "raw" not in c.lower())
    return (g.filter(pl.col(nombre) == "Mexico")
            .with_columns(pl.col("Year").cast(pl.Int64, strict=False),
                          pl.col(flujo).cast(pl.Float64, strict=False).alias("desplazados"))
            .filter(pl.col("Year").is_between(YEAR_MIN, YEAR_MAX))
            .select("Year", "desplazados").sort("Year"))


def contexto_acled() -> pl.DataFrame:
    """Posición de México en el índice global de ACLED (contexto nacional, no
    subnacional)."""
    r = pl.read_excel(ACLED_XLSX, sheet_name="Results")
    col0 = r.columns[0]
    r = r.with_row_index("rank_global", offset=1)
    return r.filter(pl.col(col0) == "Mexico")


# ---------------------------------------------------------------- sub-índice B (sin cambio)

def construir_subindice_fragmentacion(catalogo: pl.DataFrame) -> pl.DataFrame:
    df = pl.read_parquet(GRUPOS_OCVED)
    base = (df.group_by(["cve_ent", "cve_mun", "anio"])
            .agg(pl.col("n_grupos_mun_anio").first().alias("n_grupos"),
                 pl.col("eventos_totales").first().alias("eventos"),
                 pl.col("control_proxy").first()))
    base = base.with_columns(
        (pl.col("n_grupos").rank(method="average").over("anio")
         / pl.len().over("anio") * 100).alias("pct_fragmentacion")
    ).with_columns(pl.col("cve_mun").cast(pl.Int64))
    return (base.join(catalogo, on="cve_mun", how="left")
            .select("cve_ent", "cve_mun", "municipio", "anio", "n_grupos",
                    "control_proxy", "eventos", "pct_fragmentacion")
            .sort(["anio", "pct_fragmentacion"], descending=[False, True]))


# ---------------------------------------------------------------- validación

def validar(indice: pl.DataFrame, desap: pl.DataFrame) -> None:
    print("\n=== Gate 1: brecha homicidio INEGI vs SESNSP (nacional por año) ===")
    print("(INEGI = registro de mortalidad, patrón oro; SESNSP suele subcontar)")
    for anio in range(YEAR_MIN, YEAR_MAX + 1):
        d = indice.filter(pl.col("anio") == anio)
        ine, ses = d["homicidio"].sum(), d["homicidio_sesnsp"].sum()
        print(f"  {anio}: INEGI={ine:>6,.0f}  SESNSP={ses:>6,.0f}  SESNSP/INEGI={ses/ine*100:5.1f}%")

    print("\n=== Gate 2: tendencia CARDINAL nacional (índice medio ponderado por pob) ===")
    print("(con ancla fija la escalada 2015→2024 SÍ es visible; imposible con percentil)")
    for anio in range(YEAR_MIN, YEAR_MAX + 1):
        d = indice.filter(pl.col("anio") == anio)
        w = d["poblacion"].to_numpy()
        im = np.average(d["indice_conflicto"].to_numpy(), weights=w)
        hr = d["homicidio"].sum() / d["poblacion"].sum() * 1e5
        print(f"  {anio}: índice medio={im:+.3f} · tasa homicidio nacional={hr:5.1f}/100k")

    print("\n=== Gate 3: efecto del encogimiento EB (top-10 letalidad 2024) ===")
    print("(sin EB, los primeros lugares serían municipios diminutos con 1-2 casos)")
    u = indice.filter(pl.col("anio") == YEAR_MAX)
    cruda = (pl.col("homicidio") / pl.col("poblacion") * 1e5)
    print("  Por tasa CRUDA (lo que hacía el índice viejo):")
    for r in u.with_columns(cruda.alias("tc")).sort("tc", descending=True).head(5).iter_rows(named=True):
        print(f"    {r['municipio']:<24}{r['entidad']:<16} pob={r['poblacion']:>10,.0f} tasa_cruda={r['tc']:7.1f} hom={r['homicidio']:.0f}")
    print("  Por índice con EB (rediseño):")
    for r in u.sort("indice_conflicto", descending=True).head(5).iter_rows(named=True):
        print(f"    {r['municipio']:<24}{r['entidad']:<16} pob={r['poblacion']:>10,.0f} índice={r['indice_conflicto']:+.2f} hom={r['homicidio']:.0f}")

    print(f"\n=== Sensatez: top-12 municipios más conflictivos ({YEAR_MAX}) ===")
    print(u.sort("indice_conflicto", descending=True).head(12)
          .select("municipio", "entidad", "tasa_homicidio", "tasa_extorsion",
                  "tasa_desaparecidos", "indice_conflicto", "nivel"))

    print("\nDistribución de niveles (todo el panel, cuartiles fijos):")
    print(indice["nivel"].value_counts().sort("nivel"))

    print("\n=== Cobertura desaparecidos (RNPDNO colocable) por año ===")
    cob = desap.group_by("anio").agg(pl.col("desaparecidos").sum()).sort("anio")
    for r in cob.iter_rows(named=True):
        print(f"  {r['anio']}: {r['desaparecidos']:>6,.0f} colocables a municipio")

    print("\n=== ALERTA: sesgo de redacción del pilar desaparecidos por estado ===")
    print("(el municipio/fecha se redacta como CONFIDENCIAL o 999 de forma DESIGUAL por")
    print(" estado; el pilar es un PISO que subcuenta más a los estados de arriba)")
    raw = pl.read_csv(DESAP_CSV, infer_schema_length=0).with_columns(
        pl.col("CVE_ENT").cast(pl.Int64, strict=False))
    redact = ((pl.col("CVE_MUN") == "999") | (pl.col("CVE_MUN") == "CONFIDENCIAL")
              | (pl.col("FECHA_DESAPARICION") == "CONFIDENCIAL"))
    diag = (raw.group_by("CVE_ENT").agg(pl.len().alias("total"), redact.sum().alias("redact"))
            .with_columns((pl.col("redact") / pl.col("total") * 100).alias("pct"))
            .filter(pl.col("CVE_ENT").is_between(1, 32)).sort("pct", descending=True))
    for r in diag.head(5).iter_rows(named=True):
        print(f"  edo {r['CVE_ENT']:>2}: {r['total']:>7,} registros, {r['pct']:>4.0f}% redactado")
    print(f"  ... (mediana estatal de redacción: {diag['pct'].median():.0f}%)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando catálogo de municipios...")
    catalogo = cargar_catalogo_municipios()

    print(f"Construyendo panel municipal ({YEAR_MIN}-{YEAR_MAX}): homicidio INEGI, "
          "SESNSP, desaparecidos; suavizado 3 años + tasas EB...")
    panel = construir_panel()
    indice = estandarizar(panel).join(catalogo, on="cve_mun", how="left").with_columns(
        (pl.col("cve_mun") // 1000).alias("cve_ent"))

    cols = ["cve_ent", "cve_mun", "municipio", "entidad", "anio", "poblacion", "pob_baja",
            "homicidio", "homicidio_sesnsp", "persistencia_s",
            "tasa_homicidio", "tasa_feminicidio", "tasa_secuestro",
            "tasa_desaparecidos", "tasa_extorsion",
            "z_homicidio", "z_feminicidio", "z_secuestro", "z_desaparecidos",
            "z_extorsion", "z_persistencia",
            "dim_letalidad", "dim_dano_civil", "dim_coercion", "dim_persistencia",
            "indice_conflicto", "nivel"]
    indice = indice.select(cols).sort(["anio", "indice_conflicto"], descending=[False, True])
    out_a = args.output_dir / "conflict_index_municipio_anio.parquet"
    indice.write_parquet(out_a)
    print(f"Índice A: {indice.height:,} filas -> {out_a}")

    print("\nConstruyendo Sub-índice B (fragmentación OCVED, 2007-2018)...")
    frag = construir_subindice_fragmentacion(catalogo)
    frag.write_parquet(args.output_dir / "conflict_index_fragmentacion_2007_2018.parquet")
    print(f"Sub-índice B: {frag.height:,} filas")

    print("\nCapas de contexto (NO fusionadas)...")
    desp = contexto_desplazamiento()
    desp.write_parquet(args.output_dir / "conflict_contexto_desplazamiento_idmc.parquet")
    acled = contexto_acled()
    acled.write_parquet(args.output_dir / "conflict_contexto_acled.parquet")
    print(f"  desplazamiento IDMC: {desp.height} años · ACLED México: rank global "
          f"{acled['rank_global'][0] if acled.height else 'n/d'}")

    desap = desaparecidos_municipal()
    validar(indice, desap)


if __name__ == "__main__":
    main()
