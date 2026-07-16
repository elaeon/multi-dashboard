"""
Imputación de la recaudación fiscal federal por entidad federativa, 2018-2024,
replicando el método de incidencia económica de Ríos & Saucedo (2025),
"La Recaudación Fiscal de los Estados Mexicanos desde un Enfoque Económico"
(centralismo/articulos/Rios_y_Saucedo_7_1.pdf, Cuadro 10).

Alcance: solo ingresos TRIBUTARIOS (ISR, IVA, IEPS, ISAN, comercio exterior,
hidrocarburos, accesorios y otros) más cuotas de seguridad social
(IMSS/INFONAVIT/ISSSTE) -- ~96%+ del ingreso federal 2020 del artículo.
Quedan FUERA los no tributarios federales (derechos, productos,
aprovechamientos, contribuciones de mejoras, otras contribuciones): no existe
localmente una base estatal de recaudación FEDERAL para esos conceptos (ni en
data/sat/, ni en presupuesto_federacion/, ni en EFIPEM -- que mide finanzas
PROPIAS de los gobiernos estatales, un universo distinto).

Salida: informe_data/recaudacion_imputada_estatal.parquet
  grano: anio x cve_ent x concepto
  columnas: participacion_pct (siempre), monto_imputado_miles_pesos (solo
  donde hay total nacional confiable: todos los tributarios; null para
  IMSS/INFONAVIT/ISSSTE por falta de un total nacional de cuotas confiable
  en los datos locales).

SUPUESTOS Y DESVIACIONES DEL MÉTODO ORIGINAL (documentadas explícitamente):
  1. ISR personas físicas vs morales: no existe el desglose en pesos para
     2021+ (SAT solo publica el ISR total agregado). Se fija la proporción
     que encontró el artículo para 2020: 49.75% PF / 50.25% PM.
  2. La porción de ISRPF correspondiente a nómina pública estatal/municipal
     (Fondo del Impuesto Sobre la Renta, participaciones EFIPEM -- 100% de la
     retención de ISR a empleados estatales/municipales, Art. 3-A LCF) se
     resta primero del ISRPF nacional y se reparte con ese peso REAL; el
     resto de ISRPF se reparte con ingreso total de los hogares (ENIGH).
  3. Años sin encuesta ENIGH (2019, 2021, 2023) usan la ola más cercana (2018,
     2020, 2022 respectivamente; empate 2019 resuelto hacia la ola anterior).
  4. Censo Económico (factor trabajo/capital de ISRPM) es quinquenal: se usa
     la edición 2019 (año base 2018) para 2021-2023 y la edición 2024 (año
     base 2023) para 2024.
  5. IEPS gasolinas/diésel y exploración/extracción de hidrocarburos: en vez
     de reconstruir ventas de PEMEX + vehículos + permisos (método del
     artículo), se usa el peso REAL de las participaciones federales EFIPEM
     ("IEPS Gasolinas", "Fondo de Extracción de Hidrocarburos") -- series
     fiscales anuales ya calculadas con fórmulas de reparto basadas en
     consumo/producción real, una mejora sobre reconstruir un proxy.
  6. ISAN (vehículos nuevos): no hay recaudación efectiva por entidad; se usa
     como proxy el parque vehicular privado registrado (INEGI VMRC, nivel de
     stock, no altas nuevas ni recaudación real).
  7. Accesorios y otros (categoría tributaria residual, <2% del total
     nacional): sin base estatal disponible; se reparte por población
     (CONAPO) -- única excepción al alcance "solo tributarios".
  8. Cuotas ISSSTE: no hay afiliados ISSSTE por entidad; se usa el gasto en
     Capítulo 1000 "Servicios Personales" por entidad (Cuenta Pública, Anexos
     Transversales) como proxy de nómina pública federal. Esta variable
     refleja la unidad administrativa que REGISTRA el gasto (~40% se
     contabiliza en CDMX por sedes centrales), no necesariamente dónde
     trabaja el empleado -- sesgo análogo al que el propio artículo señala
     para los datos oficiales que intenta corregir.
  9. IEPS "otros" (juegos y sorteos, telecomunicaciones, alimentos alta
     densidad calórica, plaguicidas, carbono -- residual pequeño) se reparte
     con el mismo criterio que IVA (consumo total de los hogares, ENIGH).

Run: uv run python scripts/centralismo/preparar_recaudacion_imputada.py
"""

import io
import zipfile
from pathlib import Path

import fastexcel
import pandas as pd
import polars as pl

RAIZ = Path(__file__).resolve().parents[2]
SALIDA = RAIZ / "informe_data/recaudacion_imputada_estatal.parquet"

ANIOS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

# ola de ENIGH más cercana a cada año fiscal (supuesto #3); empate -> ola anterior
ENIGH_OLA = {2018: 2018, 2019: 2018, 2020: 2020,
             2021: 2020, 2022: 2022, 2023: 2022, 2024: 2024}
ENIGH_ZIP = {
    2018: "conjunto_de_datos_enigh_2018_ns_csv.zip",
    2020: "conjunto_de_datos_enigh_ns_2020_csv.zip",
    2022: "conjunto_de_datos_enigh_ns_2022_csv.zip",
    2024: "conjunto_de_datos_enigh2024_ns_csv.zip",
}

# edición del Censo Económico más cercana a cada año fiscal (supuesto #4)
CE_EDICION = {2018: 2019, 2019: 2019, 2020: 2019,
             2021: 2019, 2022: 2019, 2023: 2019, 2024: 2024}
CE_ABREV = [
    "ags", "bc", "bcs", "camp", "cdmx", "chih", "chis", "coah", "col", "dgo",
    "gro", "gto", "hgo", "jal", "mex", "mich", "mor", "nay", "nl", "oax",
    "pue", "qro", "qroo", "sin", "slp", "son", "tab", "tamps", "tlax", "ver",
    "yuc", "zac",
]

ISR_PF_SHARE = 0.4975   # supuesto #1 (hallazgo 2020 del artículo, pág. 69)
ISR_PM_SHARE = 0.5025

CVE_ENT_ALL = pl.DataFrame({"cve_ent": list(range(1, 33))})


def normalizar(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Completa las 32 entidades (rellena 0.0 donde falta) y ordena por cve_ent."""
    faltan = [c for c in df.columns if c != "cve_ent"]
    df = df.with_columns(pl.col("cve_ent").cast(pl.Int64))
    out = CVE_ENT_ALL.join(df, on="cve_ent", how="left").fill_null(0.0)
    return out.select("cve_ent", *faltan)


# --------------------------------------------------------------- totales nacionales

def cargar_totales_nacionales() -> pl.DataFrame:
    """SAT Ingresos Tributarios (nacional, mensual) -> anual, por concepto."""
    df = pd.read_csv(RAIZ / "data/sat/recaudacion_federal/ingresostributarios.csv",
                      encoding="latin-1")
    df = df.rename(columns={df.columns[0]: "ejercicio"})
    anual = df.groupby("ejercicio").sum(numeric_only=True)

    filas = []
    for anio in ANIOS:
        r = anual.loc[anio]
        filas.append((anio, "isr", float(r["impuesto_renta"])))
        filas.append((anio, "iva", float(r["impuesto_al_valor_agregado"])))
        filas.append((anio, "ieps_tabaco", float(r["ieps_tabacos_labrados"])))
        bebidas = sum(float(r.get(c, 0.0) or 0.0) for c in
                      ["ieps_alcohol", "ieps_cerveza_y_bebidas_refrescantes",
                       "ieps_bebidas_energetizantes", "ieps_bebidas_saborizadas"])
        filas.append((anio, "ieps_bebidas", bebidas))
        filas.append((anio, "ieps_gasolinas", float(r["ieps_gasolinas_y_diesel"])))
        otros_ieps = sum(float(r.get(c, 0.0) or 0.0) for c in
                         ["ieps_juegos_y_sorteos", "ieps_telecomunicaciones",
                          "ieps_alimentosnobasicos_conalta_densidad_calorica",
                          "ieps_plaguicidas", "ieps_carbono"])
        filas.append((anio, "ieps_otros", otros_ieps))
        filas.append((anio, "comercio_exterior",
                      float(r["importaciones"]) + float(r["exportaciones"])))
        filas.append((anio, "isan", float(r["impuesto_automoviles_nuevos"])))
        filas.append((anio, "hidrocarburos",
                      float(r["impuesto_porlaactividadde_exploracionyextraccion_dehidrocarburos"])))
        filas.append((anio, "accesorios_otros", float(r["otros_ingresos_tributarios"])))

    out = pl.DataFrame(filas, schema=["anio", "concepto", "monto_nacional_miles"],
                        orient="row")
    print(f"[nacional] SAT ingresos tributarios cargados {ANIOS[0]}-{ANIOS[-1]}")
    return out


# --------------------------------------------------------------- ENIGH

def _cargar_concentradohogar(ola: int) -> pd.DataFrame:
    ruta = RAIZ / "data/inegi/enigh" / ENIGH_ZIP[ola]
    with zipfile.ZipFile(ruta) as zf:
        nombre = next(n for n in zf.namelist()
                      if "conjunto_de_datos/conjunto_de_datos_concentradohogar" in n)
        with zf.open(nombre) as f:
            df = pd.read_csv(io.BytesIO(f.read()), low_memory=False)
    col_geo = "ubica_geo" if "ubica_geo" in df.columns else "ubic_geo"
    df["cve_ent"] = df[col_geo] // 1000
    return df


def pesos_enigh(anio_fiscal: int) -> pl.DataFrame:
    ola = ENIGH_OLA[anio_fiscal]
    df = _cargar_concentradohogar(ola)
    agg = (df.assign(
        ing_cor_w=df["ing_cor"] * df["factor"],
        gasto_mon_w=df["gasto_mon"] * df["factor"],
        tabaco_w=df["tabaco"] * df["factor"],
        bebidas_w=df["bebidas"] * df["factor"],
    ).groupby("cve_ent")[["ing_cor_w", "gasto_mon_w", "tabaco_w", "bebidas_w"]]
     .sum().reset_index())
    return pl.from_pandas(agg).with_columns(pl.col("cve_ent").cast(pl.Int64))


# --------------------------------------------------------------- Censo Económico

def pesos_censo_economico(anio_fiscal: int) -> pl.DataFrame:
    edicion = CE_EDICION[anio_fiscal]
    filas = []
    for geo in CE_ABREV:
        zipname = f"datosabiertos_conjunto_de_datos_ce_{geo}_{edicion}_csv.zip"
        ruta = RAIZ / "data/inegi/censo_economico/2024" / zipname
        with zipfile.ZipFile(ruta) as zf:
            csvname = f"conjunto_de_datos/tr_ce_{geo}_{edicion}.csv"
            with zf.open(csvname) as f:
                df = pd.read_csv(f, encoding="latin-1", low_memory=False)
        df["CODIGO"] = df["CODIGO"].astype(str).str.strip()
        fila = df[(df["CODIGO"] == "TOTAL DE SECTOR") &
                  df["ID_ESTRATO"].isnull() &
                  df["E04"].isnull()]
        if fila.empty:
            continue
        r = fila.iloc[0]
        cve_ent = int(r["E03"])
        j000a = float(r["J000A"])
        capital = float(r["A131A"]) - j000a
        filas.append((cve_ent, j000a, capital))
    return pl.DataFrame(filas, schema=["cve_ent", "trabajo_w", "capital_w"], orient="row")


# --------------------------------------------------------------- EFIPEM (participaciones)

_EFIPEM_CONCEPTOS = ["IEPS Gasolinas", "Fondo de Extracción de Hidrocarburos",
                     "Fondo del Impuesto Sobre la Renta"]


def _cargar_efipem_anio(anio: int) -> pd.DataFrame:
    partes = []
    for zipname in ("conjunto_de_datos_efipem_estatal_csv.zip",
                    "conjunto_de_datos_efipem_cdmx_csv.zip"):
        ruta = RAIZ / "data/inegi/efipem" / zipname
        with zipfile.ZipFile(ruta) as zf:
            nombre = next(n for n in zf.namelist()
                          if "conjunto_de_datos" in n and str(anio) in n and n.endswith(".csv"))
            with zf.open(nombre) as f:
                partes.append(pd.read_csv(io.BytesIO(f.read()), encoding="utf-8", low_memory=False))
    return pd.concat(partes, ignore_index=True)


def pesos_efipem(anio_fiscal: int) -> pl.DataFrame:
    df = _cargar_efipem_anio(anio_fiscal)
    df = df[(df["TEMA"] == "Ingresos") &
            df["DESCRIPCION_CATEGORIA"].isin(_EFIPEM_CONCEPTOS)]
    piv = df.pivot_table(index="CVE_ENT", columns="DESCRIPCION_CATEGORIA",
                         values="VALOR", aggfunc="sum").reset_index()
    piv = piv.rename(columns={
        "IEPS Gasolinas": "ieps_gasolinas_w",
        "Fondo de Extracción de Hidrocarburos": "hidrocarburos_w",
        "Fondo del Impuesto Sobre la Renta": "isr_nomina_w",
        "CVE_ENT": "cve_ent",
    })
    for c in ("ieps_gasolinas_w", "hidrocarburos_w", "isr_nomina_w"):
        if c not in piv.columns:
            piv[c] = 0.0
    piv = piv.fillna(0.0)
    return pl.from_pandas(piv).with_columns(pl.col("cve_ent").cast(pl.Int64))


def total_nacional_efipem(anio_fiscal: int, concepto: str) -> float:
    """Suma nacional (32 estados) de una partida de participaciones EFIPEM en un año."""
    df = _cargar_efipem_anio(anio_fiscal)
    return float(df[(df["TEMA"] == "Ingresos") &
                    (df["DESCRIPCION_CATEGORIA"] == concepto)]["VALOR"].sum())


# --------------------------------------------------------------- IMSS empleo formal

def pesos_imss(anio_fiscal: int) -> pl.DataFrame:
    lf = pl.scan_csv(RAIZ / "data/datamx/empleo_formal/empleo_formal.csv")
    agg = (lf.filter(pl.col("PERIODO").str.starts_with(str(anio_fiscal)))
             .group_by("CVE_ENT")
             .agg(pl.col("MASA_SALARIAL_TOTAL").sum().alias("masa_salarial_w"))
             .rename({"CVE_ENT": "cve_ent"})
             .collect())
    return agg


# --------------------------------------------------------------- INEGI Vehículos (ISAN proxy)

def pesos_vehiculos(anio_fiscal: int) -> pl.DataFrame:
    with zipfile.ZipFile(RAIZ / "data/inegi/vehiculos/conjunto_de_datos_vmrc_anual_csv.zip") as zf:
        nombre = f"conjunto_de_datos/vmrc_anual_tr_cifra_{anio_fiscal}.csv"
        with zf.open(nombre) as f:
            df = pd.read_csv(f)
    df["ID_ENTIDAD"] = df["ID_ENTIDAD"].astype(str).str.strip().astype(int)
    agg = df.groupby("ID_ENTIDAD")["AUTO_PARTICULAR"].sum().reset_index()
    agg = agg.rename(columns={"ID_ENTIDAD": "cve_ent", "AUTO_PARTICULAR": "autos_w"})
    return pl.from_pandas(agg)


# --------------------------------------------------------------- Capítulo 1000 (ISSSTE proxy)

_CP_ANEXOS = {
    2018: "BD_Transversales_CP_2018.xlsx",
    2019: "BD_Transversales_CP_2019.xlsx",
    2020: "BD_Transversales_CP_2020.xlsx",
    2021: "BD_Transversales_CP_2021.xlsx",
    2022: "BD_Transversales_CP_2022.xlsx",
    2023: "BD_Transversales_CP_2023.xlsx",
    2024: "BD_Transversales_CP_2024.xlsx",
}


def pesos_capitulo1000(anio_fiscal: int) -> pl.DataFrame:
    ruta = RAIZ / "data/presupuesto_federacion/cuenta_publica/anexos_transversales" / _CP_ANEXOS[anio_fiscal]
    lector = fastexcel.read_excel(ruta)
    df = lector.load_sheet(0, use_columns=["ID_CAPITULO", "ID_ENTIDAD_FEDERATIVA",
                                           "MONTO_PAGADO"]).to_polars()
    agg = (df.filter((pl.col("ID_CAPITULO") == 1000) &
                     pl.col("ID_ENTIDAD_FEDERATIVA").is_between(1, 32))
             .group_by("ID_ENTIDAD_FEDERATIVA")
             .agg(pl.col("MONTO_PAGADO").sum().alias("nomina_publica_w"))
             .rename({"ID_ENTIDAD_FEDERATIVA": "cve_ent"}))
    return agg


# --------------------------------------------------------------- Población CONAPO

def pesos_poblacion(anio_fiscal: int) -> pl.DataFrame:
    import sys
    sys.path.insert(0, str(RAIZ / "scripts/centralismo"))
    import comun
    pob = comun.cargar_poblacion()
    return (pob.filter(pl.col("año") == anio_fiscal)
               .select("cve_ent", pl.col("pob_total").alias("pob_w")))


# --------------------------------------------------------------- orquestación

def compartir(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Normaliza una columna de pesos a participación % sobre las 32 entidades."""
    df = normalizar(df.select("cve_ent", col), col)
    total = df[col].sum()
    pct = (df[col] / total * 100) if total else (df[col] * 0)
    return df.select("cve_ent", pct.alias("participacion_pct"))


def construir_anio(anio: int, nacional: pl.DataFrame) -> pl.DataFrame:
    print(f"[{anio}] cargando pesos por estado...")
    enigh = pesos_enigh(anio)
    censo = pesos_censo_economico(anio)
    efipem = pesos_efipem(anio)
    imss = pesos_imss(anio)
    vehic = pesos_vehiculos(anio)
    cap1000 = pesos_capitulo1000(anio)
    pob = pesos_poblacion(anio)

    def monto(concepto: str) -> float:
        row = nacional.filter((pl.col("anio") == anio) & (pl.col("concepto") == concepto))
        return float(row["monto_nacional_miles"][0]) if row.height else 0.0

    partes = []

    # --- ISR: nómina pública (peso real EFIPEM) + resto (ENIGH ingreso hogares)
    isr_total = monto("isr")
    isrpf_total = isr_total * ISR_PF_SHARE
    isrpm_total = isr_total * ISR_PM_SHARE
    isr_nomina_nacional = total_nacional_efipem(anio, "Fondo del Impuesto Sobre la Renta")
    isr_nomina_nacional = min(isr_nomina_nacional, isrpf_total)  # no puede exceder el ISRPF
    isrpf_resto_total = isrpf_total - isr_nomina_nacional

    pct_isr_nomina = compartir(efipem, "isr_nomina_w")
    pct_isr_resto = compartir(enigh, "ing_cor_w")
    monto_isr = (
        pct_isr_nomina.rename({"participacion_pct": "p1"})
        .join(pct_isr_resto.rename({"participacion_pct": "p2"}), on="cve_ent")
        .with_columns(
            (pl.col("p1") / 100 * isr_nomina_nacional +
             pl.col("p2") / 100 * isrpf_resto_total).alias("monto_isrpf")
        )
    )

    # ISRPM: 1/3 consumo (ENIGH gasto), 1/3 trabajo (censo J000A), 1/3 capital (censo margen)
    pct_consumo = compartir(enigh, "gasto_mon_w").rename({"participacion_pct": "p_consumo"})
    pct_trabajo = compartir(censo, "trabajo_w").rename({"participacion_pct": "p_trabajo"})
    pct_capital = compartir(censo, "capital_w").rename({"participacion_pct": "p_capital"})
    monto_isrpm = (pct_consumo.join(pct_trabajo, on="cve_ent").join(pct_capital, on="cve_ent")
                   .with_columns(
                       (isrpm_total / 3 * (pl.col("p_consumo") + pl.col("p_trabajo") +
                                           pl.col("p_capital")) / 100).alias("monto_isrpm")
                   ))

    isr_final = (monto_isr.join(monto_isrpm, on="cve_ent")
                 .with_columns((pl.col("monto_isrpf") + pl.col("monto_isrpm")).alias("monto"))
                 .with_columns((pl.col("monto") / isr_total * 100).alias("participacion_pct"))
                 .with_columns(pl.lit(anio).alias("anio"), pl.lit("isr").alias("concepto"))
                 .select("anio", "concepto", "cve_ent", "participacion_pct",
                         pl.col("monto").alias("monto_imputado_miles_pesos")))
    partes.append(isr_final)

    # --- conceptos simples: (peso, columna, nacional-conocido)
    simples = [
        ("iva", enigh, "gasto_mon_w", monto("iva")),
        ("ieps_tabaco", enigh, "tabaco_w", monto("ieps_tabaco")),
        ("ieps_bebidas", enigh, "bebidas_w", monto("ieps_bebidas")),
        ("ieps_otros", enigh, "gasto_mon_w", monto("ieps_otros")),
        ("comercio_exterior", enigh, "gasto_mon_w", monto("comercio_exterior")),
        ("ieps_gasolinas", efipem, "ieps_gasolinas_w", monto("ieps_gasolinas")),
        ("hidrocarburos", efipem, "hidrocarburos_w", monto("hidrocarburos")),
        ("isan", vehic, "autos_w", monto("isan")),
        ("accesorios_otros", pob, "pob_w", monto("accesorios_otros")),
    ]
    for concepto, fuente, col, monto_nal in simples:
        pct = compartir(fuente, col)
        partes.append(
            pct.with_columns(
                pl.lit(anio).alias("anio"), pl.lit(concepto).alias("concepto"),
                (pl.col("participacion_pct") / 100 * monto_nal).alias("monto_imputado_miles_pesos"),
            ).select("anio", "concepto", "cve_ent", "participacion_pct", "monto_imputado_miles_pesos")
        )

    # --- cuotas de seguridad social: solo participación %, sin total nacional confiable local
    for concepto, fuente, col in [("imss_cuota", imss, "masa_salarial_w"),
                                  ("infonavit_cuota", imss, "masa_salarial_w"),
                                  ("isste_cuota", cap1000, "nomina_publica_w")]:
        pct = compartir(fuente, col)
        partes.append(
            pct.with_columns(
                pl.lit(anio).alias("anio"), pl.lit(concepto).alias("concepto"),
                pl.lit(None, dtype=pl.Float64).alias("monto_imputado_miles_pesos"),
            ).select("anio", "concepto", "cve_ent", "participacion_pct", "monto_imputado_miles_pesos")
        )

    return pl.concat(partes)


def main():
    nacional = cargar_totales_nacionales()
    resultado = pl.concat([construir_anio(anio, nacional) for anio in ANIOS])

    # sanity: participación suma ~100 por (anio, concepto)
    sumas = resultado.group_by("anio", "concepto").agg(pl.col("participacion_pct").sum().alias("s"))
    fuera_rango = sumas.filter((pl.col("s") < 99.0) | (pl.col("s") > 101.0))
    assert fuera_rango.is_empty(), f"participación no suma ~100%:\n{fuera_rango}"

    # sanity: reconciliación de montos tributarios contra el total SAT
    tributarios = ["isr", "iva", "ieps_tabaco", "ieps_bebidas", "ieps_otros", "ieps_gasolinas",
                  "comercio_exterior", "isan", "hidrocarburos", "accesorios_otros"]
    for anio in ANIOS:
        for concepto in tributarios:
            suma_estados = resultado.filter(
                (pl.col("anio") == anio) & (pl.col("concepto") == concepto)
            )["monto_imputado_miles_pesos"].sum()
            fila = nacional.filter((pl.col("anio") == anio) & (pl.col("concepto") == concepto))
            nal = float(fila["monto_nacional_miles"][0]) if fila.height else 0.0
            if nal:
                assert abs(suma_estados / nal - 1) < 0.01, (anio, concepto, suma_estados, nal)

    # sanity: hidrocarburos solo en estados productores (peso EFIPEM > 0 en pocos estados)
    prod2024 = resultado.filter(
        (pl.col("anio") == 2024) & (pl.col("concepto") == "hidrocarburos") &
        (pl.col("participacion_pct") > 0)
    )
    assert 4 <= prod2024.height <= 12, f"estados productores fuera de rango: {prod2024.height}"

    resultado.write_parquet(SALIDA)
    print(f"\nOK -> {SALIDA.relative_to(RAIZ)} ({resultado.height:,} filas)")
    print(resultado.filter(pl.col("anio") == 2024).sort("participacion_pct", descending=True)
          .filter(pl.col("concepto") == "isr").head(5))


if __name__ == "__main__":
    main()
