"""
Índice de Impunidad Estatal (rediseño de constructo + método, panel 2020-2024).

Corrige las fallas centrales de scripts/indices/AUDITORIA.md §1: los pilares ya no
miden PROCESO (densidad policial, cocientes de trámite) sino RESULTADO — crimen
que queda impune. El compuesto se arma con TRES pilares de resultado, todos
orientados "más alto = más impunidad":

  1. Cifra negra (ENVIPE, encuesta de victimización). % de delitos que NO se
     denuncian, o que denunciados NO derivan en carpeta de investigación
     (BP1_20 ≠ Sí, o BP1_20 = Sí ∧ BP1_24 ≠ Sí), ponderado por FAC_DEL. Es el
     insumo central de los índices de referencia (IGI-MEX/UDLAP, México Evalúa)
     y el frente del embudo. Con IC 95% por bootstrap estratificado por UPM
     (diseño muestral de la ENVIPE), porque es dato de encuesta.

  2. Impunidad de homicidio (embudo duro). 1 − (condenas por homicidio doloso /
     defunciones por homicidio). Numerador: CNIJE, personas/delitos con resolución
     "Doloso - Condenatoria" en homicidio+feminicidio (edición = año+1).
     Denominador: defunciones INEGI por agresión (CIE-10 X85–Y09), por entidad de
     ocurrencia — registro de mortalidad INDEPENDIENTE de la fiscalía, robusto a la
     reclasificación que sí contamina a SESNSP. Es un cociente de flujo (condenas
     de un año contra muertes del mismo año): las causas tardan, así que sub/sobre-
     estima en los extremos; se documenta como proxy, no como tasa exacta de
     esclarecimiento.

  3. No-judicialización (CNPJE, fiscalías). 1 − (ejercicio de la acción penal /
     carpetas de investigación INICIADAS). Corrige §1.2: el denominador ya no son
     las determinaciones (sólo lo que la fiscalía cerró) sino las carpetas ABIERTAS
     en el año (m2s3p4), así que una fiscalía que no determina nada ya no se ve
     mejor.

Metodología (espejo de prepare_desigualdad_index.py, corrige §1.6/§1.7): NO se
normaliza por percentil-dentro-de-año. Cada pilar se estandariza (z-score) contra
un ANCLA FIJA — media/desviación agrupada de TODOS los estado-año — de modo que la
impunidad nacional SÍ puede subir o bajar 2020→2024. El nivel (Bajo…Extremo) usa
cuartiles fijos del conjunto agrupado. Se reporta PCA + alfa de Cronbach (¿los 3
pilares miden un constructo latente común?) y sensibilidad de pesos.

Columnas de CONTEXTO (no entran al compuesto, quedan para diagnóstico): capacidad
policial (estructural, ≠ efectividad, §1.5), % en prisión preventiva (de doble
filo, §1.4), tasa de recomendación DDHH (signo ambiguo + ruido de denominador
chico, §1.3 — se anula donde hay < MIN_EXP_DDHH expedientes concluidos) y
desconfianza institucional percibida (ENVIPE).

Capa municipal (2 tablas SEPARADAS, NO fusionadas al índice estatal — impunidad es
un constructo estatal por diseño institucional: fiscalías, comisiones de DDHH y
sistema penitenciario existen sólo a nivel estado): policía municipal (CNGMD 2024)
y enfrentamientos (CNGSPSPE 2019). Cada una un corte único, no una serie.

No es un dato de dashboard: la carpeta de salida se especifica por línea de comandos.

Run: uv run python scripts/indices/prepare_impunidad_index.py --output-dir <ruta>
"""
import argparse
import glob
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ / "scripts/centralismo"))
from comun import ESTADOS, cargar_poblacion          # noqa: E402
from cap12_capacidad_policial import fuerza_estatal, leer_zip_csv  # noqa: E402

ANIO_MIN, ANIO_MAX = 2020, 2024
SENTINELAS = ["NSS", "NA", "ND", "NP"]

# Pilares de RESULTADO que forman el compuesto (todos: "más = más impunidad").
PILARES = ["cifra_negra", "impunidad_homicidio", "no_judicializacion"]
# Pesos iguales — supuesto documentado y ajustable (ver sensibilidad_pesos()).
PESOS = {p: 1 / 3 for p in PILARES}
NIVELES = ["Bajo", "Medio", "Alto", "Extremo"]

MIN_EXP_DDHH = 30      # supresión de denominador chico en el pilar de contexto DDHH
N_BOOT = 300           # réplicas bootstrap para el IC de la cifra negra
SEED_BOOT = 20240722


# ---------------------------------------------------------------- utilidades genéricas

def _norm(n: str) -> str:
    return n.replace("\\", "/")


def _limpiar_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().replace("ï»¿", "").replace("﻿", "") for c in df.columns]
    return df


def _leer_miembro(z: zipfile.ZipFile, nombre_original: str, **kw) -> pd.DataFrame:
    with z.open(nombre_original) as f:
        raw = f.read()
    try:
        texto = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        texto = raw.decode("latin-1")
    # Algunas ediciones de ENVIPE (2020-2021) usan '\r' puro como terminador de
    # línea (estilo Mac clásico, sin '\n'); io.StringIO no lo traduce por
    # defecto y el '\r' queda pegado al último campo de cada fila.
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    return _limpiar_cols(pd.read_csv(io.StringIO(texto), **kw))


def _indice(zpath: Path) -> pd.DataFrame:
    """Lee 0_indice_*.csv (ARCHIVO, CONTENIDO) — mapa de tabla → descripción."""
    with zipfile.ZipFile(zpath) as z:
        entries = [_norm(n) for n in z.namelist()]
        orig = z.namelist()
        i = next(i for i, e in enumerate(entries) if "0_indice" in e.lower())
        return _leer_miembro(z, orig[i])


def _tabla_y_diccionario(zpath: Path, stem: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga la tabla de datos `stem` y su diccionario de columnas."""
    with zipfile.ZipFile(zpath) as z:
        entries = [_norm(n) for n in z.namelist()]
        orig = z.namelist()
        dato_i = next(i for i, e in enumerate(entries)
                      if e.endswith(".csv") and "conjunto_de_datos" in e and f"/{stem}_" in e)
        dic_i = next(i for i, e in enumerate(entries)
                     if e.endswith(".csv") and "diccionario" in e and stem in e)
        data = _leer_miembro(z, orig[dato_i])
        dic = _leer_miembro(z, orig[dic_i])
        return data, dic


def _catalogo(zpath: Path, substr: str) -> pd.DataFrame:
    """Lee un catálogo (catalogos/<substr>...csv) del ZIP."""
    with zipfile.ZipFile(zpath) as z:
        entries = [_norm(n) for n in z.namelist()]
        orig = z.namelist()
        i = next(i for i, e in enumerate(entries)
                 if "catalogos" in e.lower() and substr in e.lower() and e.endswith(".csv"))
        return _leer_miembro(z, orig[i])


def _col_ent(df: pd.DataFrame) -> str:
    return next(c for c in ("entidad_a", "cve_ent", "entidad") if c in df.columns)


def _a_numero(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(serie.replace(SENTINELAS, pd.NA), errors="coerce")


# ================================================================ PILAR 1: cifra negra

def _boot_media_upm(ind: np.ndarray, w: np.ndarray, est_dis: np.ndarray,
                    upm: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """IC 95% (percentil) de una media ponderada Σw·ind/Σw con bootstrap
    ESTRATIFICADO POR UPM (Rao-Wu, con reemplazo), acorde al diseño de la ENVIPE:
    dentro de cada estrato (est_dis) se remuestrean sus UPMs conservando su número,
    se juntan todos los registros de las UPMs elegidas y se recalcula la media.
    Vectorizado igual que prepare_desigualdad_index._boot_eco (layout CSR + gather
    ragged)."""
    BIG = int(upm.max()) + 1
    key = est_dis.astype(np.int64) * BIG + upm.astype(np.int64)
    uniq, inv = np.unique(key, return_inverse=True)
    K = len(uniq)
    order = np.argsort(inv, kind="stable")
    cnt = np.bincount(inv, minlength=K)
    ptr = np.zeros(K + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(cnt)
    strat = uniq // BIG
    seg_lo = np.empty(K, np.int64)
    seg_hi = np.empty(K, np.int64)
    bounds = np.concatenate([[0], np.flatnonzero(np.diff(strat)) + 1, [K]])
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        seg_lo[s:e] = s
        seg_hi[s:e] = e

    wi = w * ind
    ms = np.empty(N_BOOT)
    for b in range(N_BOOT):
        su = seg_lo + (rng.random(K) * (seg_hi - seg_lo)).astype(np.int64)
        counts = cnt[su]
        pos = order[np.repeat(ptr[su], counts)
                    + np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)]
        ms[b] = wi[pos].sum() / w[pos].sum()
    return float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


def _miembro_tmod_vic(z: zipfile.ZipFile) -> str:
    return next(e for e in z.namelist()
                if "tmod_vic" in e.lower() and "/conjunto_de_datos/" in e.lower()
                and e.endswith(".csv"))


def _zip_envipe(anio: int) -> Path:
    candidatos = [p for p in (RAIZ / "data/inegi/envipe").glob("*.zip")
                  if re.search(rf"envipe_?{anio}", p.name, re.IGNORECASE)]
    return candidatos[0]


def cargar_cifra_negra(anio: int) -> pl.DataFrame:
    """cve_ent × cifra_negra = % (ponderado FAC_DEL) de delitos sin denuncia o
    denunciados sin carpeta de investigación iniciada, con IC 95% bootstrap-UPM.

    BP1_20 (¿denunció ante MP/Fiscalía? 1=Sí, 2=No) y BP1_24 (¿se inició carpeta?
    1=Sí, 2=No, 9=NS/NR). Un delito NO es cifra negra sólo si BP1_20==1 ∧ BP1_24==1
    (denunciado E investigado); todo lo demás es cifra negra. Definición de INEGI.
    Los códigos BP1_* son estables entre ediciones; aun así se leen por nombre."""
    zpath = _zip_envipe(anio)
    with zipfile.ZipFile(zpath) as z:
        # tmod_vic no trae CVE_ENT; la entidad son los 2 primeros dígitos de ID_VIV
        # (verificado contra tvivienda.CVE_ENT).
        cols = ["ID_VIV", "FAC_DEL", "BP1_20", "BP1_24", "EST_DIS", "UPM"]
        data = _leer_miembro(z, _miembro_tmod_vic(z), usecols=cols, dtype={"ID_VIV": str})

    for c in cols:
        if data[c].dtype == object or str(data[c].dtype).startswith("str"):
            data[c] = data[c].str.strip()
    data["CVE_ENT"] = pd.to_numeric(data["ID_VIV"].str[:2], errors="coerce")
    for c in ["FAC_DEL", "BP1_20", "BP1_24", "EST_DIS", "UPM"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna(subset=["CVE_ENT", "FAC_DEL", "EST_DIS", "UPM"])
    data = data[data["CVE_ENT"].between(1, 32) & (data["FAC_DEL"] > 0)]

    # 1 = cifra negra (impune); 0 = denunciado e investigado
    data["dark"] = ~((data["BP1_20"] == 1) & (data["BP1_24"] == 1))

    rng = np.random.default_rng(SEED_BOOT + anio)
    filas = []
    for cve, d in data.groupby("CVE_ENT"):
        w = d["FAC_DEL"].to_numpy(float)
        ind = d["dark"].to_numpy(float)
        cn = float((w * ind).sum() / w.sum())
        lo, hi = _boot_media_upm(ind, w, d["EST_DIS"].to_numpy(), d["UPM"].to_numpy(), rng)
        filas.append((int(cve), cn, lo, hi))
    return pl.DataFrame(filas, schema=["cve_ent", "cifra_negra", "cifra_negra_lo",
                                       "cifra_negra_hi"], orient="row") \
        .with_columns(pl.lit(anio).alias("anio"))


# ================================================================ PILAR 2: impunidad homicidio

def _defunciones_homicidio(anio: int) -> pl.DataFrame:
    """cve_ent × homicidios = defunciones por agresión (CIE-10 X85–Y09) ocurridas
    en el año, por entidad de OCURRENCIA. Identificación por causa_def (CIE-10), NO
    por presunto/tipo_defun: el código 3 de esa variable es SUICIDIO, no homicidio
    (el DATA_OVERVIEW del dataset trae ese mapeo equivocado); X85–Y09 reproduce las
    cifras oficiales de INEGI (~33k/año)."""
    cands = (glob.glob(str(RAIZ / f"data/inegi/defunciones/*registradas_{anio}_csv.zip"))
             + glob.glob(str(RAIZ / f"data/inegi/defunciones/*edr{anio}_csv.zip")))
    zpath = Path(cands[0])
    with zipfile.ZipFile(zpath) as z:
        miembro = next(n for n in z.namelist()
                       if (n.lower().endswith(".csv")) and "conjunto_de_datos/" in _norm(n).lower()
                       and "diccion" not in n.lower() and "catalog" not in n.lower())
        raw = z.read(miembro)
    df = pl.read_csv(io.BytesIO(raw), columns=["causa_def", "ent_ocurr", "anio_ocur"],
                     infer_schema_length=0)
    c = pl.col("causa_def").str.to_uppercase()
    letra = c.str.slice(0, 1)
    num = c.str.slice(1, 2).cast(pl.Int64, strict=False)
    es_homicidio = ((letra == "X") & num.is_between(85, 99)) | ((letra == "Y") & num.is_between(0, 9))
    return (df.with_columns(pl.col("ent_ocurr").cast(pl.Int64, strict=False),
                            pl.col("anio_ocur").cast(pl.Int64, strict=False))
            .filter(es_homicidio & (pl.col("anio_ocur") == anio)
                    & pl.col("ent_ocurr").is_between(1, 32))
            .group_by("ent_ocurr").len()
            .rename({"ent_ocurr": "cve_ent", "len": "homicidios"}))


def _condenas_homicidio(anio: int) -> pl.DataFrame:
    """cve_ent × condenas = personas/delitos con resolución 'Doloso - Condenatoria'
    en homicidio y feminicidio, sumando las tablas de sentencias por tipo de delito
    y forma de comisión del Sistema Penal Acusatorio (y el residual del Sistema
    Tradicional). Edición CNIJE = año del dato + 1. Tabla y columna se resuelven por
    TEXTO (la sección cambia de número cada año: m2s4p25 en 2021, m2s3p25a/b en
    2025)."""
    edicion = anio + 1
    zips = glob.glob(str(RAIZ / f"data/inegi/cnije/{edicion}/m2_*senten_regis_causa_penal_*cnije{edicion}*.zip"))
    total = None
    for zp in zips:
        zpath = Path(zp)
        idx = _indice(zpath)
        stems = idx[idx["CONTENIDO"].str.contains("forma de comisión", case=False, na=False)
                    & idx["CONTENIDO"].str.contains("tipo de resolución", case=False, na=False)]["ARCHIVO"]
        for stem in stems:
            data, dic = _tabla_y_diccionario(zpath, str(stem))
            # Columna de delito por diccionario (tipdelit_e en 2025, tipdelit_d en
            # 2021). Se suma el desglose por sexo (Hombres+Mujeres+No identificado):
            # el 'Subtotal' no existe en ediciones viejas, el desglose sí.
            delito_col = dic[dic["DESCRIPCION"].str.strip() == "Tipo de delito"]["COLUMNA"]
            cols = dic[dic["DESCRIPCION"].str.match(
                r"^Doloso - Condenatoria - (Hombres|Mujeres|No identificado)$", na=False)]["COLUMNA"].tolist()
            if delito_col.empty or not cols:
                continue
            delito_col = delito_col.iloc[0]
            codigos = _codigos_homicidio(zpath, delito_col)
            ecol = _col_ent(data)
            data[ecol] = pd.to_numeric(data[ecol], errors="coerce")
            data[delito_col] = pd.to_numeric(data[delito_col], errors="coerce")
            for c in cols:
                data[c] = _a_numero(data[c])
            sel = data[data[ecol].between(1, 32) & data[delito_col].isin(codigos)]
            g = sel.groupby(ecol)[cols].sum().sum(axis=1)
            s = pl.DataFrame({"cve_ent": [int(v) for v in g.index], "condenas": g.to_list()})
            total = s if total is None else pl.concat([total, s])
    return (total.group_by("cve_ent").agg(pl.col("condenas").sum())
            if total is not None else
            pl.DataFrame({"cve_ent": [], "condenas": []},
                         schema={"cve_ent": pl.Int64, "condenas": pl.Float64}))


def _codigos_homicidio(zpath: Path, delito_col: str) -> set[int]:
    """Códigos de homicidio+feminicidio del catálogo que corresponde a la columna de
    delito de la tabla (tipdelit_e o tipdelit_d). El registro de mortalidad cuenta
    toda muerte por agresión, incluidas víctimas mujeres que en lo penal se tipifican
    como feminicidio; se incluyen ambos para que numerador y denominador cubran el
    mismo universo."""
    cat = _catalogo(zpath, delito_col.lower())
    ccol = next(c for c in cat.columns if c.lower().startswith("tipdeli"))
    dcol = next(c for c in cat.columns if "descrip" in c.lower())
    m = cat[dcol].str.contains("homicidio", case=False, na=False) \
        | cat[dcol].str.contains("feminicidio", case=False, na=False)
    return set(pd.to_numeric(cat[m][ccol], errors="coerce").dropna().astype(int))


def cargar_impunidad_homicidio(anio: int) -> pl.DataFrame:
    """cve_ent × impunidad_homicidio = 1 − condenas/defunciones (recortado a [0,1]).
    Cociente de flujo (condenas y muertes del mismo año); un valor cercano a 1
    significa que casi ningún homicidio termina en condena ese año."""
    condenas = _condenas_homicidio(anio)
    muertes = _defunciones_homicidio(anio)
    df = muertes.join(condenas, on="cve_ent", how="left").with_columns(
        pl.col("condenas").fill_null(0))
    return df.with_columns(
        (1 - pl.col("condenas") / pl.col("homicidios")).clip(0, 1).alias("impunidad_homicidio")
    ).select("cve_ent", "impunidad_homicidio").with_columns(pl.lit(anio).alias("anio"))


# ================================================================ PILAR 3: no-judicialización

def _carpetas_iniciadas(anio: int) -> pd.Series:
    """cve_ent → carpetas de investigación abiertas en el año (SPA). El total puro
    ('durante el año') existe desde ~2023; en ediciones previas sólo está el
    desglose 'según agencia y/o fiscalía' — ambos son partición a nivel carpeta, así
    que sumar `totalca1` por entidad da el mismo total estatal. Se resuelve por
    texto (la sección m2s3p4 es estable pero el sufijo del CONTENIDO cambia)."""
    edicion = anio + 1
    zpath = Path(glob.glob(str(RAIZ / f"data/inegi/cnpje/{edicion}/m2_*inicio_inv*cnpje{edicion}*.zip"))[0])
    idx = _indice(zpath)
    base = idx["CONTENIDO"].str.startswith(
        "Carpetas de investigación abiertas por el Ministerio Público del fuero común", na=False)
    fila = idx[base & (idx["CONTENIDO"].str.contains("durante el año", na=False)
                       | idx["CONTENIDO"].str.contains("según agencia", na=False))]
    stem = fila.iloc[0]["ARCHIVO"]
    data, dic = _tabla_y_diccionario(zpath, stem)
    total_col = ("totalca1" if "totalca1" in data.columns
                 else dic[dic["DESCRIPCION"].str.strip() == "Total"]["COLUMNA"].iloc[0])
    ecol = _col_ent(data)
    data[ecol] = pd.to_numeric(data[ecol], errors="coerce")
    data[total_col] = _a_numero(data[total_col])
    data = data[data[ecol].between(1, 32)]
    # min_count=1: si un estado reporta TODO NSS (p. ej. Chihuahua, CDMX, QRoo,
    # Gto, Zac en la edición 2021), el total queda NaN (dato faltante), no 0.
    return data.groupby(ecol)[total_col].sum(min_count=1)


def cargar_no_judicializacion(anio: int) -> pl.DataFrame:
    """cve_ent × no_judicializacion = 1 − (ejercicio de la acción penal / carpetas
    de investigación iniciadas). Denominador corregido (§1.2): carpetas ABIERTAS
    (no determinaciones). Numerador: determinaciones de 'Ejercicio de la acción
    penal' (subtotal, excluye los desgloses ' - ...')."""
    edicion = anio + 1
    zpath = Path(glob.glob(str(RAIZ / f"data/inegi/cnpje/{edicion}/m2_*cierre*cnpje{edicion}*.zip"))[0])
    idx = _indice(zpath)
    fila = idx[idx["CONTENIDO"].str.startswith("Determinaciones y/o conclusiones", na=False)]
    stem = fila.iloc[0]["ARCHIVO"]
    data, dic = _tabla_y_diccionario(zpath, stem)

    accion_col = dic[dic["DESCRIPCION"].str.startswith("Ejercicio de la acción penal", na=False)
                     & ~dic["DESCRIPCION"].str.contains(" - ", na=False)]["COLUMNA"].iloc[0]
    ecol = _col_ent(data)
    data[ecol] = pd.to_numeric(data[ecol], errors="coerce")
    data[accion_col] = _a_numero(data[accion_col])
    data = data[data[ecol].between(1, 32)]
    accion = data.groupby(ecol)[accion_col].sum(min_count=1)

    carpetas = _carpetas_iniciadas(anio)
    # Un estado con miles de carpetas y 0 (o NSS) ejercicios de la acción penal —
    # o 0/NaN carpetas — es un HUECO DE REPORTE, no un 0%/100% real (p. ej. Yucatán
    # 2024 y Hidalgo 2020 reportan 0 acción penal con >4k carpetas; Chihuahua/CDMX/
    # QRoo/Gto/Zac/Camp reportan carpetas NSS en 2020). Numerador y denominador
    # inválidos => pilar NaN; esos estado-año quedan fuera del compuesto.
    tasa = (1 - (accion.where(accion > 0) / carpetas.where(carpetas > 0))).reindex(carpetas.index).clip(0, 1)
    tasa = tasa.dropna()
    return pl.DataFrame({
        "cve_ent": [int(v) for v in tasa.index],
        "no_judicializacion": tasa.to_list(),
    }).with_columns(pl.lit(anio).alias("anio"))


# ================================================================ CONTEXTO (fuera del compuesto)

def cargar_cndhe(anio_dato: int) -> pl.DataFrame:
    """cve_ent × tasa_recomendacion (recomendaciones formales cód. 300 / expedientes
    concluidos) + exp_concluidos (denominador, para supresión de N chico)."""
    edicion = anio_dato + 1
    zpath = Path(glob.glob(str(RAIZ / f"data/inegi/cndhe/{edicion}/*exped_conclu*cndhe{edicion}*.zip"))[0])
    idx = _indice(zpath)
    fila = idx[idx["CONTENIDO"].str.contains("tipo y grado de conclus", case=False, na=False)]
    stem = fila.iloc[0]["ARCHIVO"]

    with zipfile.ZipFile(zpath) as z:
        entries = [_norm(n) for n in z.namelist()]
        orig = z.namelist()
        dato_i = next(i for i, e in enumerate(entries)
                      if e.endswith(".csv") and "conjunto_de_datos" in e and f"/{stem}_" in e)
        cat_i = next(i for i, e in enumerate(entries) if "catalogos" in e and "tipoconc" in e)
        data = _leer_miembro(z, orig[dato_i])
        cat = _leer_miembro(z, orig[cat_i])

    tcol = next(c for c in data.columns if c.startswith("tipoconc"))
    ccol = next(c for c in cat.columns if c.startswith("tipoconc"))
    codigo_recomendacion = cat[cat["descrip"].str.contains("ecomendaci", na=False)
                               & ~cat["descrip"].str.contains(" - ", na=False)][ccol].iloc[0]

    ecol = _col_ent(data)
    data["exqucott"] = _a_numero(data["exqucott"])
    data[ecol] = pd.to_numeric(data[ecol], errors="coerce")
    data = data[data[ecol].between(1, 32)]

    total = data.groupby(ecol)["exqucott"].sum()
    rec = data[data[tcol] == codigo_recomendacion].groupby(ecol)["exqucott"].sum()
    tasa = (rec / total).reindex(total.index).fillna(0)

    return pl.DataFrame({
        "cve_ent": [int(v) for v in tasa.index],
        "tasa_recomendacion": tasa.to_list(),
        "exp_concluidos": [float(total[v]) for v in tasa.index],
    }).with_columns(pl.lit(anio_dato).alias("anio"))


def cargar_cnsipee(anio_dato: int) -> pl.DataFrame:
    """cve_ent × pct_sin_sentencia (prisión preventiva)."""
    edicion = anio_dato + 1
    zips = (glob.glob(str(RAIZ / f"data/inegi/cnsipee/{edicion}/m1_*poblacion_priv_libertad*cnsipee{edicion}*.zip"))
            + glob.glob(str(RAIZ / f"data/inegi/cnsipee/{edicion}/m1_pob_privad_libert_cnsipee{edicion}*.zip")))
    zpath = Path(zips[0])
    idx = _indice(zpath)
    fila = idx[idx["CONTENIDO"].str.startswith(
        "Personas privadas de la libertad en los centros penitenciarios, "
        "según estatus jurídico y sexo", na=False)]
    stem = fila.iloc[0]["ARCHIVO"]
    data, dic = _tabla_y_diccionario(zpath, stem)

    total_col = dic[dic["DESCRIPCION"].str.strip() == "Total"]["COLUMNA"].iloc[0]
    sin_cols = dic[dic["DESCRIPCION"].str.match(r"^Sin sentencia.*Subtotal$", na=False)]["COLUMNA"].tolist()
    ecol = _col_ent(data)

    for c in [total_col] + sin_cols:
        data[c] = _a_numero(data[c])
    data[ecol] = pd.to_numeric(data[ecol], errors="coerce")
    data = data[data[ecol].between(1, 32)]

    total = data.groupby(ecol)[total_col].sum()
    sin = data.groupby(ecol)[sin_cols].sum().sum(axis=1)
    pct = (sin / total).reindex(total.index)

    return pl.DataFrame({
        "cve_ent": [int(v) for v in pct.index],
        "pct_sin_sentencia": pct.to_list(),
    }).with_columns(pl.lit(anio_dato).alias("anio"))


def _miembro_tper_vic1(z: zipfile.ZipFile, carpeta: str) -> str:
    return next(e for e in z.namelist()
                if "tper_vic1" in e.lower() and f"/{carpeta}/" in e.lower() and e.endswith(".csv"))


def cargar_envipe_desconfianza(anio: int) -> pl.DataFrame:
    """cve_ent × tasa_desconfianza_policia/jueces/mp — % "algo/mucha desconfianza"
    (códigos 3-4 de AP5_4) ponderado por FAC_ELE, entre entrevistas completas.
    Contexto; se resuelve por texto de diccionario (el código AP5_4_NN cambia de
    institución cada año)."""
    zpath = _zip_envipe(anio)
    with zipfile.ZipFile(zpath) as z:
        dic = _leer_miembro(z, _miembro_tper_vic1(z, "diccionario_de_datos"))
        nemonico = dic["NEMONICO"].astype(str).str.strip()
        dic = dic[nemonico.str.match(r"^AP5_4_\d+$", na=False)]
        nombre = dic["NOMBRE_CAMPO"].astype(str).str.lower()

        pol_col = dic[nombre.str.contains("policía estatal", na=False)]["NEMONICO"].iloc[0]
        jue_col = dic[nombre.str.contains("jueces", na=False)]["NEMONICO"].iloc[0]
        mp_col = dic[(nombre.str.contains("procuradur", na=False) | nombre.str.contains("fiscal", na=False))
                     & ~nombre.str.contains("general de la rep", na=False)]["NEMONICO"].iloc[0]

        cols = ["CVE_ENT", "RESUL_H", "FAC_ELE", pol_col, jue_col, mp_col]
        data = _leer_miembro(z, _miembro_tper_vic1(z, "conjunto_de_datos"), usecols=cols)

    for col in cols:
        if data[col].dtype == object or str(data[col].dtype).startswith("str"):
            data[col] = data[col].str.strip()

    data = data[data["RESUL_H"] == "A"].copy()
    data["FAC_ELE"] = pd.to_numeric(data["FAC_ELE"], errors="coerce")
    data["CVE_ENT"] = pd.to_numeric(data["CVE_ENT"], errors="coerce")
    for col in (pol_col, jue_col, mp_col):
        data[col] = pd.to_numeric(data[col], errors="coerce")

    resultado = pl.DataFrame({"cve_ent": list(range(1, 33))})
    for col, salida in [(pol_col, "tasa_desconfianza_policia"),
                        (jue_col, "tasa_desconfianza_jueces"),
                        (mp_col, "tasa_desconfianza_mp")]:
        d = data[data[col].isin([1, 2, 3, 4])].copy()
        d["ponderada"] = d[col].isin([3, 4]).astype(float) * d["FAC_ELE"]
        num = d.groupby("CVE_ENT")["ponderada"].sum()
        den = d.groupby("CVE_ENT")["FAC_ELE"].sum()
        tasa = num / den
        tabla = pl.DataFrame({"cve_ent": [int(v) for v in tasa.index], salida: tasa.to_list()},
                             schema={"cve_ent": pl.Int64, salida: pl.Float64})
        resultado = resultado.join(tabla, on="cve_ent", how="left")

    return resultado.with_columns(pl.lit(anio).alias("anio"))


# ================================================================ ensamblado + estandarización

def estandarizar(panel: pl.DataFrame) -> pl.DataFrame:
    """z-score de cada pilar contra ANCLA FIJA (media/sd de todos los estado-año),
    compuesto = suma ponderada de z, nivel por cuartiles fijos del conjunto
    agrupado. Ningún re-anclado por año => el nivel nacional puede moverse."""
    z = {p: ((pl.col(p) - pl.col(p).mean()) / pl.col(p).std()).alias(f"z_{p}") for p in PILARES}
    panel = panel.with_columns(*z.values())
    panel = panel.with_columns(
        sum(pl.col(f"z_{p}") * w for p, w in PESOS.items()).alias("indice_impunidad"))
    cortes = panel["indice_impunidad"].qcut([0.25, 0.5, 0.75], labels=NIVELES,
                                            allow_duplicates=True)
    return panel.with_columns(cortes.alias("nivel"))


def construir_indice_estatal() -> pl.DataFrame:
    anios = range(ANIO_MIN, ANIO_MAX + 1)
    pob = cargar_poblacion()
    fuerza = fuerza_estatal(pob).filter(pl.col("año").is_between(ANIO_MIN, ANIO_MAX)) \
        .rename({"año": "anio", "pc": "capacidad_policial_pc"}) \
        .select("cve_ent", "anio", "capacidad_policial_pc")

    cifra = pl.concat([cargar_cifra_negra(a) for a in anios])
    homic = pl.concat([cargar_impunidad_homicidio(a) for a in anios])
    nojud = pl.concat([cargar_no_judicializacion(a) for a in anios])
    cndhe = pl.concat([cargar_cndhe(a) for a in anios])
    cnsipee = pl.concat([cargar_cnsipee(a) for a in anios])
    envipe = pl.concat([cargar_envipe_desconfianza(a) for a in anios])

    df = (cifra
          .join(homic, on=["cve_ent", "anio"], how="inner")
          .join(nojud, on=["cve_ent", "anio"], how="inner")
          .join(fuerza, on=["cve_ent", "anio"], how="left")
          .join(cndhe, on=["cve_ent", "anio"], how="left")
          .join(cnsipee, on=["cve_ent", "anio"], how="left")
          .join(envipe, on=["cve_ent", "anio"], how="left"))

    # supresión de denominador chico en el pilar de contexto DDHH (§1.3)
    df = df.with_columns(
        pl.when(pl.col("exp_concluidos") >= MIN_EXP_DDHH)
        .then(pl.col("tasa_recomendacion")).otherwise(None).alias("tasa_recomendacion"))

    df = df.with_columns(pl.col("cve_ent").replace_strict({k: v[0] for k, v in ESTADOS.items()})
                         .alias("entidad"))
    return estandarizar(df).sort(["anio", "indice_impunidad"], descending=[False, True])


# ================================================================ diagnóstico del constructo

def pca_cronbach(panel: pl.DataFrame) -> None:
    """¿Los 3 pilares miden un constructo latente común? PCA sobre la matriz de
    correlación de los z-scores (share del PC1) + alfa de Cronbach."""
    X = panel.select([f"z_{p}" for p in PILARES]).drop_nulls().to_numpy()
    R = np.corrcoef(X, rowvar=False)
    vals, vecs = np.linalg.eigh(R)
    orden = np.argsort(vals)[::-1]
    vals, vecs = vals[orden], vecs[:, orden]
    print("\n=== PCA / consistencia del constructo (z-scores de los 3 pilares) ===")
    print("Matriz de correlación entre pilares:")
    for i, p in enumerate(PILARES):
        print(f"  {p:<20} " + "  ".join(f"{R[i, j]:+.2f}" for j in range(len(PILARES))))
    print(f"Varianza explicada por PC1: {vals[0] / vals.sum() * 100:.1f}%  "
          f"(eigenvalores: {', '.join(f'{v:.2f}' for v in vals)})")
    print("Cargas del PC1: " + ", ".join(f"{p}={vecs[i, 0]:+.2f}" for i, p in enumerate(PILARES)))
    k = len(PILARES)
    var_i = X.var(axis=0, ddof=1).sum()
    var_t = X.sum(axis=1).var(ddof=1)
    alpha = k / (k - 1) * (1 - var_i / var_t)
    print(f"Alfa de Cronbach: {alpha:.2f}  "
          f"({'consistencia aceptable' if alpha >= 0.6 else 'baja consistencia — los pilares miden facetas distintas'})")


def sensibilidad_pesos(panel: pl.DataFrame) -> None:
    """Correlación de rangos (Spearman) del compuesto de pesos iguales contra
    esquemas alternativos: dejar-uno-fuera y pesos del PC1."""
    ult = panel.filter(pl.col("anio") == ANIO_MAX)
    base = ult["indice_impunidad"].to_numpy()

    def spearman(a, b):
        ra = pd.Series(a).rank().to_numpy()
        rb = pd.Series(b).rank().to_numpy()
        return float(np.corrcoef(ra, rb)[0, 1])

    print(f"\n=== Sensibilidad de pesos ({ANIO_MAX}, ρ de Spearman vs pesos iguales) ===")
    Z = {p: ult[f"z_{p}"].to_numpy() for p in PILARES}
    for fuera in PILARES:
        usar = [p for p in PILARES if p != fuera]
        alt = np.mean([Z[p] for p in usar], axis=0)
        print(f"  sin '{fuera}': ρ={spearman(base, alt):.3f}")


# ================================================================ capa municipal (parcial)

def construir_policia_municipal_cngmd() -> pl.DataFrame:
    """Policía municipal, CNGMD edición 2025 (dato 2024) — corte único."""
    z = RAIZ / "data/inegi/cngmd/2025/datosabiertos_conjunto_de_datos_rec_huma_sp_cngmd2025_csv.zip"
    mun = (leer_zip_csv(z, "m3s1p22")
           .with_columns(pl.col("cvegeo").str.zfill(5).cast(pl.Int64).alias("cve_mun"),
                         pl.col("sexostt").cast(pl.Float64, strict=False))
           .group_by("cve_mun").agg(pl.sum("sexostt").alias("policia")))

    pob = (pl.read_parquet(RAIZ / "dashboard_data/conapo_pob_municipal.parquet")
           .filter(pl.col("AÑO") == 2024)
           .select(pl.col("CLAVE").alias("cve_mun"), pl.col("NOM_MUN").alias("municipio"),
                   pl.col("CLAVE_ENT").alias("cve_ent"), pl.col("POB_TOTAL").alias("poblacion")))

    df = (mun.join(pob, on="cve_mun", how="inner")
          .with_columns((pl.col("policia") / pl.col("poblacion") * 1e5).alias("policia_pc"))
          .with_columns((pl.col("policia_pc").rank(method="average") / pl.len() * 100).alias("percentil")))

    return df.with_columns(pl.lit(2024).alias("anio_dato")).select(
        "cve_mun", "municipio", "cve_ent", "anio_dato", "policia_pc", "percentil"
    ).sort("policia_pc", descending=True)


def construir_enfrentamientos_cngspspe_2019() -> pl.DataFrame:
    """Enfrentamientos y bajas, CNGSPSPE m2pb — único año disponible: 2019."""
    z = RAIZ / "data/inegi/cngspspe/2020/m2_enfrentamientos_cngspspe2020_csv.zip"
    df = (leer_zip_csv(z, "m2pb")
          .with_columns(
              pl.col("entidad_a").cast(pl.Int64, strict=False).alias("cve_ent"),
              pl.col("ubicageo_b").cast(pl.Int64, strict=False).alias("cve_mun"),
              pl.col("totalca1").cast(pl.Float64, strict=False),
              (pl.col("perads1").cast(pl.Float64, strict=False).fill_null(0)
               + pl.col("perads2").cast(pl.Float64, strict=False).fill_null(0)).alias("bajas_policia"),
              (pl.col("civarm1").cast(pl.Float64, strict=False).fill_null(0)
               + pl.col("civarm2").cast(pl.Float64, strict=False).fill_null(0)
               + pl.col("civarm3").cast(pl.Float64, strict=False).fill_null(0)).alias("bajas_civiles_armados"))
          .filter(pl.col("totalca1").is_not_null())
          .with_columns((pl.col("totalca1").rank(method="average") / pl.len() * 100).alias("percentil")))

    return df.with_columns(pl.lit(2019).alias("anio_dato")).select(
        "cve_mun", "cve_ent", "anio_dato", "totalca1", "bajas_policia",
        "bajas_civiles_armados", "percentil"
    ).sort("totalca1", descending=True)


# ================================================================ validación

def validar(panel: pl.DataFrame) -> None:
    print("\n=== Gate 1: cifra negra nacional por año (referencia INEGI ≈ 92–94%) ===")
    for anio in range(ANIO_MIN, ANIO_MAX + 1):
        d = panel.filter(pl.col("anio") == anio)
        m = d["cifra_negra"].mean()
        print(f"  {anio}: cifra negra estatal media={m * 100:.1f}%  "
              f"(min {d['cifra_negra'].min() * 100:.1f}, max {d['cifra_negra'].max() * 100:.1f})")
        assert 0.80 < m < 0.98, f"cifra negra media {m:.3f} fuera de rango sensato"

    print("\n=== Gate 2: tendencia CARDINAL nacional (índice medio por año) ===")
    print("(a diferencia del índice viejo por percentil, éste SÍ puede subir/bajar)")
    for anio in range(ANIO_MIN, ANIO_MAX + 1):
        d = panel.filter(pl.col("anio") == anio)
        print(f"  {anio}: índice medio={d['indice_impunidad'].mean():+.3f} · "
              f"cifra negra={d['cifra_negra'].mean() * 100:.1f}% · "
              f"imp. homicidio={d['impunidad_homicidio'].mean() * 100:.1f}% · "
              f"no-judicializ.={d['no_judicializacion'].mean() * 100:.1f}%")

    pca_cronbach(panel)
    sensibilidad_pesos(panel)

    print(f"\n=== Sensatez: top-10 estados más impunes ({ANIO_MAX}) ===")
    ult = panel.filter(pl.col("anio") == ANIO_MAX)
    print(ult.head(10).select("entidad", "cifra_negra", "impunidad_homicidio",
                              "no_judicializacion", "indice_impunidad", "nivel"))
    print(f"\nDistribución de niveles ({ANIO_MAX}, cuartiles fijos):")
    print(ult["nivel"].value_counts().sort("nivel"))

    print("\n=== Validez convergente: r(compuesto, desconfianza) por año ===")
    print("(se espera signo +: más impunidad administrativa <-> más desconfianza)")
    for col in ("tasa_desconfianza_policia", "tasa_desconfianza_jueces", "tasa_desconfianza_mp"):
        rs = []
        for anio in range(ANIO_MIN, ANIO_MAX + 1):
            sub = panel.filter(pl.col("anio") == anio)
            r = sub.select(pl.corr("indice_impunidad", col)).item()
            rs.append(f"{anio}:{r:+.2f}" if r is not None else f"{anio}:—")
        print(f"  {col:<28} " + "  ".join(rs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Carpeta donde escribir los parquet de salida "
                             "(este índice no es dato de dashboard, no tiene default).")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Construyendo Índice de Impunidad Estatal ({ANIO_MIN}-{ANIO_MAX})...")
    indice = construir_indice_estatal()
    cols = ["cve_ent", "entidad", "anio",
            "cifra_negra", "cifra_negra_lo", "cifra_negra_hi",
            "impunidad_homicidio", "no_judicializacion",
            "z_cifra_negra", "z_impunidad_homicidio", "z_no_judicializacion",
            "indice_impunidad", "nivel",
            "capacidad_policial_pc", "pct_sin_sentencia", "tasa_recomendacion",
            "tasa_desconfianza_policia", "tasa_desconfianza_jueces", "tasa_desconfianza_mp"]
    indice = indice.select(cols)
    out_a = args.output_dir / "impunidad_index_estatal.parquet"
    indice.write_parquet(out_a)
    print(f"Índice estatal: {indice.height:,} filas -> {out_a}")

    print("\nConstruyendo capa municipal parcial (policía CNGMD, dato 2024)...")
    policia_mun = construir_policia_municipal_cngmd()
    out_b = args.output_dir / "impunidad_municipal_policia_cngmd.parquet"
    policia_mun.write_parquet(out_b)
    print(f"Policía municipal: {policia_mun.height:,} filas -> {out_b}")

    print("\nConstruyendo capa municipal parcial (enfrentamientos CNGSPSPE, 2019)...")
    enfrent_mun = construir_enfrentamientos_cngspspe_2019()
    out_c = args.output_dir / "impunidad_municipal_enfrentamientos_2019.parquet"
    enfrent_mun.write_parquet(out_c)
    print(f"Enfrentamientos municipales: {enfrent_mun.height:,} filas -> {out_c}")

    validar(indice)


if __name__ == "__main__":
    main()
