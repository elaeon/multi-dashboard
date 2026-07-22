"""
Carencias sociales por estado (CONEVAL 2022 oficial, metodología replicada para 2024
desde ENIGH cruda) + PIB per cápita 2022/2024.

CONEVAL dejó de publicar la Medición Multidimensional de Pobreza tras 2022 (la
función pasó a INEGI en julio 2025) y no existe todavía una tabla de carencias
sociales 2024 lista para usar. Este script reconstruye las 6 carencias sociales de
CONEVAL para 2022 y 2024 a partir de los microdatos crudos de la ENIGH, replicando el
algoritmo oficial de CONEVAL (data/coneval/Python_MMP_2022.zip →
"Programa de cálculo/Pobreza 2022 CONEVAL.py") para rezago educativo, calidad y
espacios de la vivienda, servicios básicos de la vivienda y alimentación (EMSA + lca).

Salud (ic_asalud) y seguridad social (ic_segsoc) usan un PROXY DE ACCESO DIRECTO
simplificado (sin la propagación por parentesco del algoritmo oficial, que requiere
las tablas trabajos/ingresos completas) — se aplica el MISMO proxy en 2022 y 2024
para que ambos años sean comparables entre sí, y el sesgo frente al valor oficial de
CONEVAL 2022 se mide explícitamente en el gate de validación antes de confiar en 2024.

Alcance: NO se calcula pct_pobreza (pobreza multidimensional por ingreso) para 2024 —
requeriría reconstruir las líneas de pobreza LPI/LPEI de CONEVAL, fuera de alcance.

También agrega recaudación fiscal federal imputada por estado (metodología Ríos &
Saucedo 2025, informe_data/recaudacion_imputada_estatal.parquet). OJO: esto NO es
PIB — es recaudación tributaria imputada por incidencia económica. El propio
preparar_recaudacion_imputada.py lo advierte: "correlaciona con el PIB por
construcción" (los pesos de reparto son consumo/ingreso ENIGH y factores del Censo
Económico), pero no es una medición de PIB. Se incluye con nombres honestos
(recaudacion_imputada_pc, share_recaudacion_imputada), no como PIB.

Run: uv run python scripts/indices/carencias_index.py
       uv run python scripts/indices/carencias_index.py --indice   # vista condensada
"""

import argparse
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "centralismo"))
from comun import RAIZ, NOMBRE, PIBE_TOTAL, cargar_pobreza, cargar_poblacion, leer_pibe  # noqa: E402
from cap1_recaudacion_imputada import cargar_aporte_imputado, cargar_transferencias  # noqa: E402

ZIPS_ENIGH = {
    2016: "conjunto_de_datos_enigh2016_nueva_serie_csv.zip",
    2018: "conjunto_de_datos_enigh_2018_ns_csv.zip",
    2020: "conjunto_de_datos_enigh_ns_2020_csv.zip",
    2022: "conjunto_de_datos_enigh_ns_2022_csv.zip",
    2024: "conjunto_de_datos_enigh2024_ns_csv.zip",
}
# ENOE: 2022 se publicó bajo el nombre "ENOEN" (variante COVID), vigente hasta
# que la ENOE regular se reanudó en 2023. El layout de columnas de sdem no
# cambia entre variantes.
ENOE_VARIANTE = {2022: "enoen", 2024: "enoe"}
SALIDA = Path("/tmp/carencias_pib_2024.csv")


def leer_tabla(año: int, tabla: str, columnas: list[str]) -> pl.DataFrame:
    z = zipfile.ZipFile(RAIZ / "data/inegi/enigh" / ZIPS_ENIGH[año])
    ruta = next(n for n in z.namelist()
                if n.rsplit("/", 1)[-1].startswith(f"conjunto_de_datos_{tabla}")
                and n.endswith(".csv") and "bitacora" not in n)
    with z.open(ruta) as f:
        return pl.read_csv(io.BytesIO(f.read()), columns=columnas, infer_schema_length=0)


def _num(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    return df.with_columns(pl.col(c).cast(pl.Float64, strict=False) for c in cols)


# ---------------------------------------------------------------- informalidad (ENOE)

def leer_sdem_trimestre(año: int, t: int) -> pl.DataFrame:
    """Tabla sdem de la ENOE para un trimestre: cve_ent, clase2, emp_ppal, fac_tri."""
    variante = ENOE_VARIANTE[año]
    stem = f"{variante}_{año}_{t}t"
    zip_path = RAIZ / f"data/inegi/enoe/{año}/{año}_conjunto_de_datos_{variante}_{año}_{t}t_csv.zip"
    inner = f"conjunto_de_datos_sdem_{stem}/conjunto_de_datos/conjunto_de_datos_sdem_{stem}.csv"
    with zipfile.ZipFile(zip_path) as z, z.open(inner) as f:
        df = pd.read_csv(f, encoding="latin-1",
                          usecols=["ent", "clase2", "emp_ppal", "fac_tri"])
    return (pl.from_pandas(df)
              .rename({"ent": "cve_ent"})
              .with_columns(pl.col("cve_ent", "clase2", "emp_ppal").cast(pl.Int64),
                            pl.col("fac_tri").cast(pl.Float64))
              .filter(pl.col("fac_tri") > 0))


def informalidad(año: int) -> pl.DataFrame:
    """Tasa de informalidad laboral por estado (emp_ppal==1, ponderada por
    fac_tri, agrupando las 4 trimestrales del año). emp_ppal: 1=Informal, 2=Formal."""
    sdem = pl.concat([leer_sdem_trimestre(año, t) for t in (1, 2, 3, 4)])
    ocupados = sdem.filter(pl.col("clase2") == 1)
    out = (ocupados.group_by("cve_ent")
           .agg((pl.col("fac_tri").filter(pl.col("emp_ppal") == 1).sum()
                 / pl.col("fac_tri").sum() * 100).alias("pct_informal"))
           .sort("cve_ent"))
    assert out.height == 32
    return out


def excluir_huespedes(df: pl.DataFrame) -> pl.DataFrame:
    """Población objetivo CONEVAL: excluye huéspedes y trabajadores domésticos."""
    return df.filter(
        ~(((pl.col("parentesco") >= 400) & (pl.col("parentesco") < 500)) |
          ((pl.col("parentesco") >= 700) & (pl.col("parentesco") < 800)))
    )


# ---------------------------------------------------------------- 1. rezago educativo

def rezago_educativo(año: int) -> pl.DataFrame:
    cols = ["folioviv", "foliohog", "numren", "parentesco", "edad", "asis_esc",
            "nivelaprob", "gradoaprob", "antec_esc"]
    p = leer_tabla(año, "poblacion", cols)
    p = _num(p, ["parentesco", "edad", "asis_esc", "nivelaprob", "gradoaprob", "antec_esc"])
    p = excluir_huespedes(p)
    p = p.with_columns((pl.lit(año) - pl.col("edad")).alias("anac_e"))

    niv_ed = (
        pl.when((pl.col("nivelaprob") < 2) |
                ((pl.col("nivelaprob") == 2) & (pl.col("gradoaprob") < 6))).then(0)
        .when(((pl.col("nivelaprob") == 2) & (pl.col("gradoaprob") == 6)) |
              ((pl.col("nivelaprob") == 3) & (pl.col("gradoaprob") < 3)) |
              (pl.col("nivelaprob").is_in([5, 6]) & (pl.col("gradoaprob") < 3) &
               (pl.col("antec_esc") == 1))).then(1)
        .when(((pl.col("nivelaprob") == 3) & (pl.col("gradoaprob") == 3)) |
              ((pl.col("nivelaprob") == 4) & (pl.col("gradoaprob") < 3)) |
              (pl.col("nivelaprob").is_in([5, 6]) & (pl.col("antec_esc") == 1) &
               (pl.col("gradoaprob") >= 3)) |
              (pl.col("nivelaprob").is_in([5, 6]) & (pl.col("antec_esc") == 2) &
               (pl.col("gradoaprob") < 3))).then(2)
        .when(((pl.col("nivelaprob") == 4) & (pl.col("gradoaprob") == 3)) |
              (pl.col("nivelaprob").is_in([5, 6]) & (pl.col("antec_esc") == 2) &
               (pl.col("gradoaprob") >= 3)) |
              (pl.col("nivelaprob").is_in([5, 6]) & (pl.col("antec_esc") > 2)) |
              (pl.col("nivelaprob") >= 7)).then(3)
        .otherwise(None)
    )
    p = p.with_columns(niv_ed.alias("niv_ed"))

    ic_rezedu = (
        pl.when((pl.col("anac_e") >= 1998) & pl.col("edad").is_between(3, 21) &
                (pl.col("asis_esc") == 2) & (pl.col("niv_ed") < 3)).then(1)
        .when((pl.col("anac_e") >= 1982) & (pl.col("anac_e") <= 1997) &
              (pl.col("edad") >= 16) & (pl.col("niv_ed") < 2)).then(1)
        .when((pl.col("anac_e") <= 1981) & (pl.col("edad") >= 16) &
              (pl.col("niv_ed") == 0)).then(1)
        .when((pl.col("anac_e") >= 1998) & (pl.col("edad") >= 22) &
              (pl.col("niv_ed") < 3)).then(1)
        .when(pl.col("edad").is_between(0, 2)).then(0)
        .when((pl.col("anac_e") >= 1998) & pl.col("edad").is_between(3, 21) &
              (pl.col("asis_esc") == 1)).then(0)
        .when(pl.col("niv_ed") == 3).then(0)
        .when((pl.col("anac_e") >= 1982) & (pl.col("anac_e") <= 1997) &
              (pl.col("edad") >= 16) & (pl.col("niv_ed") >= 2)).then(0)
        .when((pl.col("anac_e") <= 1981) & (pl.col("edad") >= 16) &
              (pl.col("niv_ed") >= 1)).then(0)
        .otherwise(None)
    )
    return p.with_columns(ic_rezedu.alias("ic_rezedu")).select(
        "folioviv", "foliohog", "numren", "ic_rezedu")


# ---------------------------------------------------------------- 2/3. salud, seg. social (proxy)

def acceso_salud(año: int) -> pl.DataFrame:
    """Proxy 'sin afiliación', sin propagación por parentesco (ver docstring del
    módulo). La variable de afiliación cambia de nombre por ola: segpop (Seguro
    Popular, 2016/2018), pop_insabi (INSABI, 2020/2022) e inst_9 (IMSS-Bienestar
    multiselect, 2024). Misma lógica que cap4_h4_fuera_muestra.py:carencia_salud."""
    extra = "segpop" if año <= 2018 else "pop_insabi" if año <= 2022 else "inst_9"
    p = leer_tabla(año, "poblacion", ["folioviv", "foliohog", "numren", "atemed", extra])
    if año <= 2022:
        ic = pl.when((pl.col(extra).str.strip_chars() != "1") &
                      (pl.col("atemed").str.strip_chars() != "1")).then(1).otherwise(0)
    else:
        ic = pl.when(pl.col("inst_9").str.strip_chars() == "9").then(1).otherwise(0)
    return p.with_columns(ic.alias("ic_asalud")).select(
        "folioviv", "foliohog", "numren", "ic_asalud")


def seguridad_social(año: int) -> pl.DataFrame:
    """Proxy directo vía 'segsoc' (contribución a la seguridad social), sin propagación."""
    p = leer_tabla(año, "poblacion", ["folioviv", "foliohog", "numren", "segsoc"])
    ic = pl.when(pl.col("segsoc").str.strip_chars() == "1").then(0).otherwise(1)
    return p.with_columns(ic.alias("ic_segsoc")).select(
        "folioviv", "foliohog", "numren", "ic_segsoc")


# ---------------------------------------------------------------- 4. calidad y espacios vivienda

def calidad_vivienda(año: int) -> pl.DataFrame:
    v = leer_tabla(año, "viviendas",
                    ["folioviv", "mat_pisos", "mat_techos", "mat_pared", "tot_resid", "num_cuarto"])
    v = _num(v, ["mat_pisos", "mat_techos", "mat_pared", "tot_resid", "num_cuarto"])
    c = leer_tabla(año, "concentradohogar", ["folioviv", "foliohog"])
    cev = c.join(v, on="folioviv", how="left")

    icv_pisos = pl.when(pl.col("mat_pisos") == 1).then(1).when(pl.col("mat_pisos") >= 2).then(0).otherwise(None)
    icv_techos = pl.when(pl.col("mat_techos") <= 2).then(1).when(pl.col("mat_techos") >= 3).then(0).otherwise(None)
    icv_muros = pl.when(pl.col("mat_pared") <= 5).then(1).when(pl.col("mat_pared") >= 6).then(0).otherwise(None)
    cv_hac = pl.col("tot_resid") / pl.col("num_cuarto")
    icv_hac = pl.when(cv_hac > 2.5).then(1).when(cv_hac <= 2.5).then(0).otherwise(None)

    cev = cev.with_columns(icv_pisos.alias("icv_pisos"), icv_techos.alias("icv_techos"),
                            icv_muros.alias("icv_muros"), icv_hac.alias("icv_hac"))

    ic_cv = (
        pl.when(pl.col("icv_pisos").is_null() | pl.col("icv_techos").is_null() |
                pl.col("icv_muros").is_null() | pl.col("icv_hac").is_null()).then(None)
        .when((pl.col("icv_pisos") == 1) | (pl.col("icv_techos") == 1) |
              (pl.col("icv_muros") == 1) | (pl.col("icv_hac") == 1)).then(1)
        .when((pl.col("icv_pisos") == 0) & (pl.col("icv_techos") == 0) &
              (pl.col("icv_muros") == 0) & (pl.col("icv_hac") == 0)).then(0)
        .otherwise(None)
    )
    return cev.with_columns(ic_cv.alias("ic_cv")).select("folioviv", "foliohog", "ic_cv")


# ---------------------------------------------------------------- 5. servicios básicos vivienda

def servicios_vivienda(año: int) -> pl.DataFrame:
    if año <= 2022:
        cols_v = ["folioviv", "disp_agua", "drenaje", "disp_elect",
                  "combustible", "estufa_chi"]
        # procaptar (captación de lluvia) no existe en 2016; sí en 2018/2020/2022
        if año != 2016:
            cols_v.insert(1, "procaptar")
    else:
        cols_v = ["folioviv", "agua_ent", "drenaje", "disp_elect", "combus", "fogon_chi"]
    v = leer_tabla(año, "viviendas", cols_v)
    v = _num(v, [c for c in cols_v if c != "folioviv"])
    c = leer_tabla(año, "concentradohogar", ["folioviv", "foliohog"])
    sbv = c.join(v, on="folioviv", how="left")

    if año <= 2022:
        if año == 2016:
            # 2016 no trae procaptar: sin la excepción de captación de lluvia
            isb_agua = (
                pl.when(pl.col("disp_agua") >= 3).then(1)
                .when(pl.col("disp_agua") <= 2).then(0)
                .otherwise(None)
            )
        else:
            isb_agua = (
                pl.when((pl.col("procaptar") == 1) & (pl.col("disp_agua") == 4)).then(0)
                .when(pl.col("disp_agua") >= 3).then(1)
                .when(pl.col("disp_agua") <= 2).then(0)
                .otherwise(None)
            )
        isb_combus = (
            pl.when(pl.col("combustible").is_in([1, 2]) & (pl.col("estufa_chi") == 2)).then(1)
            .when(pl.col("combustible").is_in([1, 2]) & (pl.col("estufa_chi") == 1)).then(0)
            .when(pl.col("combustible").is_between(3, 6)).then(0)
            .otherwise(None)
        )
    else:
        # Catálogo agua_ent 2024 colapsado a 3 códigos (1=dentro,2=patio,3=no tiene):
        # no hay equivalente a la excepción PROCAPTAR de 2022 (procaptar cambió de
        # significado en 2024), se documenta como desviación.
        isb_agua = (
            pl.when(pl.col("agua_ent") == 3).then(1)
            .when(pl.col("agua_ent").is_in([1, 2])).then(0)
            .otherwise(None)
        )
        isb_combus = (
            pl.when(pl.col("combus").is_in([1, 2]) & (pl.col("fogon_chi") == 2)).then(1)
            .when(pl.col("combus").is_in([1, 2]) & (pl.col("fogon_chi") == 1)).then(0)
            .when(pl.col("combus").is_between(3, 7)).then(0)  # 2024 agrega 7=no cocinan
            .otherwise(None)
        )

    isb_dren = pl.when(pl.col("drenaje") >= 3).then(1).when(pl.col("drenaje") <= 2).then(0).otherwise(None)
    isb_luz = pl.when(pl.col("disp_elect") >= 5).then(1).when(pl.col("disp_elect") <= 4).then(0).otherwise(None)

    sbv = sbv.with_columns(isb_agua.alias("isb_agua"), isb_dren.alias("isb_dren"),
                            isb_luz.alias("isb_luz"), isb_combus.alias("isb_combus"))

    ic_sbv = (
        pl.when(pl.col("isb_agua").is_null() | pl.col("isb_dren").is_null() |
                pl.col("isb_luz").is_null() | pl.col("isb_combus").is_null()).then(None)
        .when((pl.col("isb_agua") == 1) | (pl.col("isb_dren") == 1) |
              (pl.col("isb_luz") == 1) | (pl.col("isb_combus") == 1)).then(1)
        .when((pl.col("isb_agua") == 0) & (pl.col("isb_dren") == 0) &
              (pl.col("isb_luz") == 0) & (pl.col("isb_combus") == 0)).then(0)
        .otherwise(None)
    )
    return sbv.with_columns(ic_sbv.alias("ic_sbv")).select("folioviv", "foliohog", "ic_sbv")


# ---------------------------------------------------------------- 6. alimentación

def alimentacion(año: int) -> pl.DataFrame:
    p = leer_tabla(año, "poblacion", ["folioviv", "foliohog", "numren", "parentesco", "edad"])
    p = _num(p, ["parentesco", "edad"])
    p = excluir_huespedes(p)
    menores = (
        p.with_columns(pl.col("edad").is_between(0, 17).cast(pl.Int8).alias("men"))
         .group_by(["folioviv", "foliohog"]).agg(pl.col("men").sum().alias("men_sum"))
         .with_columns(pl.when(pl.col("men_sum") >= 1).then(1).otherwise(0).alias("id_men"))
         .select("folioviv", "foliohog", "id_men")
    )

    cols_hog = ["folioviv", "foliohog",
                "acc_alim2", "acc_alim4", "acc_alim5", "acc_alim6", "acc_alim7", "acc_alim8",
                "acc_alim11", "acc_alim12", "acc_alim13", "acc_alim14", "acc_alim15", "acc_alim16",
                "alim17_1", "alim17_2", "alim17_3", "alim17_4", "alim17_5", "alim17_6", "alim17_7",
                "alim17_8", "alim17_9", "alim17_10", "alim17_11", "alim17_12"]
    hog = leer_tabla(año, "hogares", cols_hog)
    hog = _num(hog, [c for c in cols_hog if c not in ("folioviv", "foliohog")])
    ali = hog.join(menores, on=["folioviv", "foliohog"], how="left")

    ali = ali.with_columns(
        (pl.col("acc_alim4") == 1).cast(pl.Int8).alias("ia_1ad"),
        (pl.col("acc_alim5") == 1).cast(pl.Int8).alias("ia_2ad"),
        (pl.col("acc_alim6") == 1).cast(pl.Int8).alias("ia_3ad"),
        (pl.col("acc_alim2") == 1).cast(pl.Int8).alias("ia_4ad"),
        (pl.col("acc_alim7") == 1).cast(pl.Int8).alias("ia_5ad"),
        (pl.col("acc_alim8") == 1).cast(pl.Int8).alias("ia_6ad"),
        (pl.col("acc_alim11") == 1).cast(pl.Int8).alias("ia_7men"),
        (pl.col("acc_alim12") == 1).cast(pl.Int8).alias("ia_8men"),
        (pl.col("acc_alim13") == 1).cast(pl.Int8).alias("ia_9men"),
        (pl.col("acc_alim14") == 1).cast(pl.Int8).alias("ia_10men"),
        (pl.col("acc_alim15") == 1).cast(pl.Int8).alias("ia_11men"),
        (pl.col("acc_alim16") == 1).cast(pl.Int8).alias("ia_12men"),
    )

    tot_iaad = pl.sum_horizontal("ia_1ad", "ia_2ad", "ia_3ad", "ia_4ad", "ia_5ad", "ia_6ad")
    tot_iamen = pl.sum_horizontal("ia_1ad", "ia_2ad", "ia_3ad", "ia_4ad", "ia_5ad", "ia_6ad",
                                   "ia_7men", "ia_8men", "ia_9men", "ia_10men", "ia_11men", "ia_12men")
    ali = ali.with_columns(
        pl.when(pl.col("id_men") == 0).then(tot_iaad).otherwise(None).alias("tot_iaad"),
        pl.when(pl.col("id_men") == 1).then(tot_iamen).otherwise(None).alias("tot_iamen"),
    )

    ins_ali = (
        pl.when((pl.col("tot_iaad") == 0) | (pl.col("tot_iamen") == 0)).then(0)
        .when(pl.col("tot_iaad").is_in([1, 2]) | pl.col("tot_iamen").is_in([1, 2, 3])).then(1)
        .when(pl.col("tot_iaad").is_in([3, 4]) | pl.col("tot_iamen").is_in([4, 5, 6, 7])).then(2)
        .when(pl.col("tot_iaad").is_in([5, 6]) | (pl.col("tot_iamen") >= 8)).then(3)
        .otherwise(None)
    )
    ali = ali.with_columns(ins_ali.alias("ins_ali"))
    ic_ali = (pl.when(pl.col("ins_ali").is_in([2, 3])).then(1)
                .when(pl.col("ins_ali").is_in([0, 1])).then(0).otherwise(None))
    ali = ali.with_columns(ic_ali.alias("ic_ali"))

    # Ponderador PMA: grupo1=max(alim17_1,2)*2, grupo3=*1, grupo4=*1,
    # grupo5=max(5,6,7)*4, grupo8=*3, grupo9=*4, grupo10=*0.5, grupo11=*0.5, grupo12=*0
    tot_cpond = (
        pl.max_horizontal("alim17_1", "alim17_2") * 2
        + pl.col("alim17_3")
        + pl.col("alim17_4")
        + pl.max_horizontal("alim17_5", "alim17_6", "alim17_7") * 4
        + pl.col("alim17_8") * 3
        + pl.col("alim17_9") * 4
        + pl.col("alim17_10") * 0.5
        + pl.col("alim17_11") * 0.5
    )
    ali = ali.with_columns(tot_cpond.alias("tot_cpond"))
    dch = (
        pl.when(pl.col("tot_cpond").is_between(0, 28)).then(1)
        .when((pl.col("tot_cpond") > 28) & (pl.col("tot_cpond") <= 42)).then(2)
        .when(pl.col("tot_cpond") > 42).then(3)
        .otherwise(None)
    )
    ali = ali.with_columns(dch.alias("dch"))
    lca = pl.when(pl.col("dch").is_in([1, 2])).then(1).when(pl.col("dch") == 3).then(0).otherwise(None)
    ali = ali.with_columns(lca.alias("lca"))

    ic_ali_nc = (
        pl.when((pl.col("ic_ali") == 0) & (pl.col("lca") == 0)).then(0)
        .when((pl.col("ic_ali") == 1) |
              ((pl.col("lca") == 1) & pl.col("ic_ali").is_not_null() & pl.col("lca").is_not_null())).then(1)
        .otherwise(None)
    )
    return ali.with_columns(ic_ali_nc.alias("ic_ali_nc")).select("folioviv", "foliohog", "ic_ali_nc")


# ---------------------------------------------------------------- ensamblado y agregación

def _ki(df: pl.DataFrame) -> pl.DataFrame:
    """Normaliza las llaves de join a Int64. En 2016/2018 `folioviv` pierde los
    ceros a la izquierda en poblacion/hogares pero no en concentradohogar/
    viviendas (match string 68%, Int64 100%); castear a entero une bien todas
    las olas."""
    keys = [pl.col(k).cast(pl.Int64, strict=False) for k in ("folioviv", "foliohog")
            if k in df.columns]
    return df.with_columns(keys)


def poblacion_base(año: int) -> pl.DataFrame:
    if año <= 2020:
        # poblacion no trae entidad ni factor en 2016/2018/2020: se toman de
        # concentradohogar (cve_ent = 2 primeros dígitos de ubica_geo).
        p = leer_tabla(año, "poblacion", ["folioviv", "foliohog", "numren", "parentesco"])
        p = _ki(p).with_columns(pl.col("parentesco").cast(pl.Float64, strict=False))
        con = _ki(leer_tabla(año, "concentradohogar",
                             ["folioviv", "foliohog", "ubica_geo", "factor"]))
        con = con.with_columns(
            pl.col("factor").cast(pl.Float64, strict=False),
            pl.col("ubica_geo").str.slice(0, 2).cast(pl.Int64).alias("cve_ent"),
        ).select("folioviv", "foliohog", "cve_ent", "factor")
        p = p.join(con, on=["folioviv", "foliohog"], how="left")
    else:
        p = leer_tabla(año, "poblacion",
                       ["folioviv", "foliohog", "numren", "parentesco", "entidad", "factor"])
        p = _ki(p).with_columns(
            pl.col("parentesco").cast(pl.Float64, strict=False),
            pl.col("factor").cast(pl.Float64, strict=False),
            pl.col("entidad").cast(pl.Int64, strict=False).alias("cve_ent"),
        )
    p = excluir_huespedes(p)

    base = (
        p.join(_ki(rezago_educativo(año)), on=["folioviv", "foliohog", "numren"], how="left")
         .join(_ki(acceso_salud(año)), on=["folioviv", "foliohog", "numren"], how="left")
         .join(_ki(seguridad_social(año)), on=["folioviv", "foliohog", "numren"], how="left")
         .join(_ki(calidad_vivienda(año)), on=["folioviv", "foliohog"], how="left")
         .join(_ki(servicios_vivienda(año)), on=["folioviv", "foliohog"], how="left")
         .join(_ki(alimentacion(año)), on=["folioviv", "foliohog"], how="left")
    )
    return base.with_columns(
        (pl.col("ic_rezedu") + pl.col("ic_asalud") + pl.col("ic_segsoc") +
         pl.col("ic_cv") + pl.col("ic_sbv") + pl.col("ic_ali_nc")).alias("i_privacion")
    )


def agregar_estado(base: pl.DataFrame, año: int) -> pl.DataFrame:
    def wavg(col: str) -> pl.Expr:
        w = pl.col("factor")
        v = pl.col(col)
        return (v * w).sum() / w.filter(v.is_not_null()).sum()

    out = base.group_by("cve_ent").agg(
        (wavg("ic_rezedu") * 100).alias("pct_ic_rezedu"),
        (wavg("ic_asalud") * 100).alias("pct_ic_asalud"),
        (wavg("ic_segsoc") * 100).alias("pct_ic_segsoc"),
        (wavg("ic_cv") * 100).alias("pct_ic_cv"),
        (wavg("ic_sbv") * 100).alias("pct_ic_sbv"),
        (wavg("ic_ali_nc") * 100).alias("pct_ic_ali"),
        wavg("i_privacion").alias("promedio_carencias"),
        (((pl.col("i_privacion") >= 3).cast(pl.Int8) * pl.col("factor")).sum()
         / pl.col("factor").filter(pl.col("i_privacion").is_not_null()).sum() * 100
         ).alias("pct_carencias3"),
    ).with_columns(pl.lit(año).alias("año"))
    return out.sort("cve_ent")


def validar_reconstruccion(recon: pl.DataFrame, año: int) -> None:
    """Compara la reconstrucción de una ola contra el panel CONEVAL oficial
    (cargar_pobreza cubre 2016/2018/2020/2022). Las 4 carencias exactas se
    bloquean si el error medio supera 2pp; los 2 proxies (salud, seg. social)
    sólo reportan su sesgo."""
    oficial = cargar_pobreza().filter(pl.col("año") == año)
    comp = recon.join(oficial, on="cve_ent", suffix="_oficial")

    print(f"\n=== Validación {año}: reconstrucción vs. CONEVAL oficial ===")
    exactas = [("pct_ic_rezedu", "rezago educativo"), ("pct_ic_cv", "calidad vivienda"),
               ("pct_ic_sbv", "servicios básicos"), ("pct_ic_ali", "alimentación")]
    for col, nombre in exactas:
        err = (pl.col(col) - pl.col(f"{col}_oficial")).abs()
        e = comp.select(err.mean().alias("m"), err.max().alias("x")).row(0, named=True)
        print(f"  {nombre:20s}: error abs medio {e['m']:.2f}pp (máx {e['x']:.2f}pp)")
        assert e["m"] < 2.0, f"{año} {nombre}: error medio {e['m']:.2f}pp supera el umbral de 2pp"

    proxies = [("pct_ic_asalud", "salud (proxy)"), ("pct_ic_segsoc", "seg. social (proxy)")]
    for col, nombre in proxies:
        sesgo = pl.col(col) - pl.col(f"{col}_oficial")
        s = comp.select(sesgo.mean().alias("m"), sesgo.min().alias("lo"), sesgo.max().alias("hi")).row(0, named=True)
        print(f"  {nombre:20s}: sesgo nacional medio {s['m']:+.2f}pp "
              f"(rango {s['lo']:+.2f} a {s['hi']:+.2f}pp) — proxy, no bloquea")
    print(f"  → gate de validación {año} OK")


def tabla_resumen(ancho: pl.DataFrame) -> pl.DataFrame:
    """Versión legible de la tabla ancha: encabezados cortos, valores redondeados,
    ordenada de más a menos carencias en 2024 (promedio_carencias_2024 descendente).
    La tabla de precisión completa se conserva sin cambios en SALIDA (CSV)."""
    pct = ["pct_ic_rezedu", "pct_ic_asalud", "pct_ic_segsoc", "pct_ic_cv", "pct_ic_sbv",
           "pct_ic_ali", "pct_carencias3", "pct_informal"]
    cortos = ["Rezedu", "Salud", "SegSoc", "VivCal", "VivServ", "Alim", "3+Car", "Informal"]

    exprs = [pl.col("estado").alias("Estado")]
    for base, corto in zip(pct, cortos):
        for año in (2022, 2024):
            exprs.append(pl.col(f"{base}_{año}").round(1).alias(f"{corto}{año}"))
    for año in (2022, 2024):
        exprs.append(pl.col(f"promedio_carencias_{año}").round(2).alias(f"ProCar{año}"))
    for año in (2022, 2024):
        exprs.append((pl.col(f"pib_pc_{año}") / 1000).round(0).cast(pl.Int64)
                     .cast(pl.String).add("k").alias(f"PIBpc{año}"))
    for año in (2022, 2024):
        exprs.append((pl.col(f"recaudacion_imputada_pc_{año}") / 1000).round(1)
                     .cast(pl.String).add("k").alias(f"RecImpPC{año}"))
    for año in (2022, 2024):
        exprs.append(pl.col(f"share_recaudacion_imputada_{año}").round(2)
                     .cast(pl.String).add("%").alias(f"Share{año}"))

    return ancho.sort("promedio_carencias_2024", descending=True).select(exprs)


COLUMNAS_INDICE = {
    "Estado": "estado",
    "ProCar2022": "promedio_carencias_2022",
    "ProCar2024": "promedio_carencias_2024",
    "RecImpPC2024": "recaudacion_imputada_pc_2024",
    "Share2024": "share_recaudacion_imputada_2024",
    "TransfPC2024": "transferencias_pc_2024",
    "Informal2022": "pct_informal_2022",
    "Informal2024": "pct_informal_2024",
}


def tabla_indice(ancho: pl.DataFrame, ordenar_por: str = "ProCar2024",
                  ascendente: bool = False, ranking: bool = False) -> pl.DataFrame:
    """Índice condensado: promedio de carencias de cada año, y recaudación
    imputada, share y transferencias federales (Ramo 28+33) per cápita del
    último año (2024) solamente. Ordenable por cualquier columna del índice
    (ordenar_por, uno de COLUMNAS_INDICE) vía el flag --ordenar-por.

    Si ranking=True (--ranking), cada columna numérica muestra el lugar del
    estado (1-32) dentro de ESA categoría en vez del valor — el lugar 1 es el
    valor más alto si ascendente=False (default), o el más bajo si ascendente=True."""
    if ranking:
        exprs = [pl.col("estado").alias("Estado")]
        for corto in ("ProCar2022", "ProCar2024", "RecImpPC2024", "Share2024", "TransfPC2024",
                      "Informal2022", "Informal2024"):
            exprs.append(pl.col(COLUMNAS_INDICE[corto])
                         .rank(method="ordinal", descending=not ascendente)
                         .cast(pl.Int64).alias(corto))
    else:
        exprs = [
            pl.col("estado").alias("Estado"),
            pl.col("promedio_carencias_2022").round(2).alias("ProCar2022"),
            pl.col("promedio_carencias_2024").round(2).alias("ProCar2024"),
            (pl.col("recaudacion_imputada_pc_2024") / 1000).round(1).cast(pl.String).add("k").alias("RecImpPC2024"),
            pl.col("share_recaudacion_imputada_2024").round(2).cast(pl.String).add("%").alias("Share2024"),
            (pl.col("transferencias_pc_2024") / 1000).round(1).cast(pl.String).add("k").alias("TransfPC2024"),
            pl.col("pct_informal_2022").round(1).cast(pl.String).add("%").alias("Informal2022"),
            pl.col("pct_informal_2024").round(1).cast(pl.String).add("%").alias("Informal2024"),
        ]
    # ordena por la columna cruda (numérica) ANTES de convertir a string/ranking,
    # para no terminar con un orden alfabético en vez de numérico
    columna_cruda = COLUMNAS_INDICE[ordenar_por]
    return ancho.sort(columna_cruda, descending=not ascendente).select(exprs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indice", action="store_true",
                         help="Muestra un índice condensado (promedio de carencias de "
                              "cada año + recaudación imputada y share del último año) "
                              "en vez de la tabla resumen completa.")
    parser.add_argument("--ordenar-por", default="ProCar2024", choices=list(COLUMNAS_INDICE),
                         help="Columna por la que ordenar el índice (--indice). "
                              "Default: ProCar2024.")
    parser.add_argument("--ascendente", action="store_true",
                         help="Ordena el índice ascendente en vez de descendente "
                              "(default: descendente). También define el sentido del "
                              "ranking con --ranking (lugar 1 = valor más alto por "
                              "default, o más bajo con --ascendente).")
    parser.add_argument("--ranking", action="store_true",
                         help="En vez del valor, muestra el lugar (1-32) de cada estado "
                              "dentro de cada categoría del índice (--indice).")
    args = parser.parse_args()

    print("=== 1/6 Reconstrucción 2022 y validación contra CONEVAL oficial ===")
    car22 = agregar_estado(poblacion_base(2022), 2022)
    validar_reconstruccion(car22, 2022)

    print("\n=== 2/6 Reconstrucción 2024 (ENIGH cruda) ===")
    car24 = agregar_estado(poblacion_base(2024), 2024)

    cols_carencias = ["cve_ent", "año", "pct_ic_rezedu", "pct_ic_asalud", "pct_ic_segsoc",
                       "pct_ic_cv", "pct_ic_sbv", "pct_ic_ali", "pct_carencias3", "promedio_carencias"]
    carencias = pl.concat([car22.select(cols_carencias), car24.select(cols_carencias)])

    print("\n=== 3/6 PIB per cápita 2022/2024 ===")
    pob = cargar_poblacion().filter(pl.col("año").is_in([2022, 2024]))
    pib = leer_pibe(PIBE_TOTAL, bloque="Millones de pesos a precios de 2018").filter(
        pl.col("año").is_in([2022, 2024]))
    pib_pc = (pib.join(pob, on=["cve_ent", "año"])
                 .with_columns((pl.col("valor") * 1_000_000 / pl.col("pob_total")).alias("pib_pc"))
                 .select("cve_ent", "año", "pib_pc"))

    print("\n=== 4/6 Recaudación fiscal federal imputada 2022/2024 (NO es PIB, ver docstring) ===")
    aporte = cargar_aporte_imputado().filter(pl.col("año").is_in([2022, 2024]))
    recaudacion = (
        aporte.join(pob, on=["cve_ent", "año"])
              .with_columns(
                  # 'recaudacion' = suma de monto_imputado_millones_pesos (informe_data/
                  # recaudacion_imputada_estatal.parquet) -> millones de pesos, de ahí ×1e6.
                  (pl.col("recaudacion") * 1_000_000 / pl.col("pob_total")).alias("recaudacion_imputada_pc"),
                  (pl.col("share_recaudacion") * 100).alias("share_recaudacion_imputada"),
              )
              .select("cve_ent", "año", "recaudacion_imputada_pc", "share_recaudacion_imputada")
    )
    for año in (2022, 2024):
        s = recaudacion.filter(pl.col("año") == año)["share_recaudacion_imputada"].sum()
        print(f"  {año}: suma de shares = {s:.1f}% (debe rondar 100%)")

    print("\n=== Transferencias federales (participaciones Ramo 28 + aportaciones Ramo 33), 2024 ===")
    transf_2024 = (
        cargar_transferencias()
        .rename({"ciclo": "año"})
        .with_columns(pl.col("año").cast(pl.Int64))
        .filter(pl.col("año") == 2024)
        .join(pob.filter(pl.col("año") == 2024), on=["cve_ent", "año"])
        .with_columns((pl.col("transfer") / pl.col("pob_total")).alias("transferencias_pc_2024"))
        .select("cve_ent", "transferencias_pc_2024")
    )

    print("\n=== 5/6 Informalidad laboral (ENOE, emp_ppal) 2022/2024 ===")
    informal = pl.concat([
        informalidad(año).with_columns(pl.lit(año).alias("año")) for año in (2022, 2024)
    ])
    for año in (2022, 2024):
        i = informal.filter(pl.col("año") == año).join(pob.filter(pl.col("año") == año),
                                                         on="cve_ent")
        nac = (i["pct_informal"] * i["pob_total"]).sum() / i["pob_total"].sum()
        print(f"  {año}: informalidad nacional (ponderada por población) ≈ {nac:.1f}%")

    tabla = (carencias.join(pib_pc, on=["cve_ent", "año"], how="left")
             .join(recaudacion, on=["cve_ent", "año"], how="left")
             .join(informal, on=["cve_ent", "año"], how="left")
             .with_columns(pl.col("cve_ent").replace_strict(NOMBRE).alias("estado")))

    print("\n=== 6/6 Prueba de sensatez: top-5 / bottom-5 por promedio_carencias ===")
    for año in (2022, 2024):
        t = tabla.filter(pl.col("año") == año).sort("promedio_carencias", descending=True)
        print(f"  {año} más carencias: {t.head(5)['estado'].to_list()}")
        print(f"  {año} menos carencias: {t.tail(5)['estado'].to_list()}")

    ancho = (
        tabla.sort(["cve_ent", "año"])
             .pivot(on="año", index=["cve_ent", "estado"],
                    values=["pct_ic_rezedu", "pct_ic_asalud", "pct_ic_segsoc", "pct_ic_cv",
                            "pct_ic_sbv", "pct_ic_ali", "pct_carencias3", "promedio_carencias", "pib_pc",
                            "recaudacion_imputada_pc", "share_recaudacion_imputada", "pct_informal"])
             .sort("cve_ent")
             .join(transf_2024, on="cve_ent", how="left")
    )
    ancho.write_csv(SALIDA)
    print(f"\nTabla final (32 estados, precisión completa) guardada en {SALIDA}")
    if args.indice:
        direccion = "ascendente" if args.ascendente else "descendente"
        modo = "ranking" if args.ranking else "valores"
        print(f"\nÍndice condensado ({modo}, ordenado por {args.ordenar_por}, {direccion}):")
        tabla_mostrar = tabla_indice(ancho, ordenar_por=args.ordenar_por,
                                      ascendente=args.ascendente, ranking=args.ranking)
    else:
        print("\nTabla resumen (ordenada de más a menos carencias en 2024):")
        tabla_mostrar = tabla_resumen(ancho)
    with pl.Config(tbl_cols=-1, tbl_rows=-1, fmt_str_lengths=60, tbl_width_chars=280):
        print(tabla_mostrar)


if __name__ == "__main__":
    main()
