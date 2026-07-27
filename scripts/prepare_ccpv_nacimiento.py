"""
Migración interestatal por LUGAR DE NACIMIENTO, censos 1950–2020.

Concepto distinto al de scripts/prepare_ccpv_migracion.py: aquí la matriz es
"residentes de la entidad X nacidos en la entidad Y" — migración acumulada de toda
la vida (stock), no el flujo del quinquenio previo. Es el único concepto de
movimiento entre estados disponible en 1950–1980, y existe en los 8 censos.

Fuentes 1950–1980 (data/inegi/ccpv/migracion/):
  · 1950  CGP50_Nal_Mig1  agregados   · CGP50_Nal_Mig2  matriz (incluye diagonal)
  · 1960  CGP60_Nal_Mig3  agregados   · CGP60_Nal_Mig2  matriz (incluye diagonal)
  · 1970  CGP70_Nal_Mig1  agregados y matriz en la misma hoja
  · 1980  CPyV80_Nal_Mig1 agregados y matriz en la misma hoja (columna compartida)
1990–2020: se reutilizan las filas concepto='nacimiento' que ya produjo
scripts/prepare_ccpv_migracion.py (córrelo antes que este script).

Salida: dashboard_data/ccpv_nacimiento_estatal.parquet

Run: uv run python scripts/prepare_ccpv_nacimiento.py
"""

import re
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "centralismo"))
from comun import NOMBRE, normalizar_estado
from prepare_ccpv_migracion import DIR, OUT_DIR, _entero, _filas

PARQUET_5A = OUT_DIR / "ccpv_migracion_estatal.parquet"
SALIDA = OUT_DIR / "ccpv_nacimiento_estatal.parquet"

# Etiquetas históricas que normalizar_estado() no resuelve (o resuelve mal).
# Ojo con 1970: 'Baja California' es el estado (02) y 'Baja California, Territorio'
# es Baja California Sur (03) — sólo los distingue la coma.
HISTORICAS = {
    "Baja California Territorio Norte": 2,
    "Baja California Territorio Sur": 3,
    "Baja California, Territorio": 3,
    "Quintana Roo, Territorio": 23,
    "San Luís Potosí": 24,
}

# Entidades que en ese censo aún eran territorio federal (se elevaron en 1974).
TERRITORIOS = {1950: {2, 3, 23}, 1960: {3, 23}, 1970: {3, 23}}

CATEGORIA = {
    "Total": "Total",
    "Nacidos en la entidad": "En la entidad",
    "En la entidad": "En la entidad",
    "Nacidos en otra entidad": "En otra entidad",
    "En otra entidad": "En otra entidad",
    "Nacidos en otro país": "En otro país",
    "En país extranjero": "En otro país",
    "No especificado": "No especificado",
    # 1970 y 1980 la publican como categoría hermana de 'En otra entidad'; 1990 en
    # cambio la mete DENTRO del desglose de esa categoría (ver prepare_ccpv_migracion).
    "Entidad insuficientemente especificada": "Entidad de nacimiento no especificada",
}
NO_ESPECIFICADA = "Entidad no especificada"
ORIGEN_NO_ENTIDAD = {"Entidad no indicada"}  # 1960: va dentro del desglose
# Pseudo-entidades de la columna de residencia que no son entidades federativas.
PSEUDO = {"Complementarios /a", "No indicado"}


def _limpia(valor):
    """Celda → texto con espacios normalizados ('01  Aguascalientes' → '01 Aguascalientes')."""
    return re.sub(r"\s+", " ", str(valor)).strip() if valor is not None else ""


def _cve(etiqueta):
    """Etiqueta de entidad → CVE_ENT 1-32. 'Total' → 0. Cualquier otra cosa → None."""
    s = _limpia(etiqueta)
    if s == "Total":
        return 0
    if s in HISTORICAS:
        return HISTORICAS[s]
    if len(s) > 2 and s[:2].isdigit():  # 1980 trae el código pegado al nombre
        s = s[3:] if s[2] == " " else s
    return HISTORICAS.get(s) or normalizar_estado(s)


def _fila(censo, cve_destino, categoria, cve_origen, origen, valores):
    personas, hombres, mujeres = (_entero(v) for v in valores)
    return {
        "censo": censo,
        "cve_destino": cve_destino,
        "destino": NOMBRE.get(cve_destino, "Nacional"),
        "categoria": categoria,
        "total_categoria": origen is None,
        "cve_origen": cve_origen,
        "origen": origen,
        "personas": personas,
        "hombres": hombres,
        "mujeres": mujeres,
    }


def agregados_por_sexo(censo, archivo, hoja, col_ent, col_sexo, cols, filtro_censo=None):
    """Hojas de 1950/1960: las filas son entidad × sexo (× censo). Devuelve una fila
    por (entidad, categoría) con hombres y mujeres pivotados a columnas."""
    acum = {}
    for fila in _filas(archivo, hoja):
        if filtro_censo and _limpia(fila[filtro_censo[0]]) != filtro_censo[1]:
            continue
        if all(_entero(fila[c]) is None for c in cols.values()):
            continue
        cve = _cve(fila[col_ent])
        if cve is None:
            continue
        sexo = _limpia(fila[col_sexo])
        if sexo not in ("Total", "Hombres", "Mujeres"):
            continue
        for etiqueta, col in cols.items():
            acum.setdefault((cve, etiqueta), {})[sexo] = _entero(fila[col])

    return [
        _fila(censo, cve, etiqueta, None, None,
              (v.get("Total"), v.get("Hombres"), v.get("Mujeres")))
        for (cve, etiqueta), v in acum.items()
    ]


def matriz_con_diagonal(censo, archivo, hoja, col_ent, col_org, cols, filtro_censo=None):
    """Hojas de 1950/1960: la matriz incluye la fila del propio estado como un origen
    más. Se descarta la diagonal (ya viene como categoría 'En la entidad' en la hoja
    de agregados) y el resto queda como desglose de 'En otra entidad'.

    El bloque nacional de estas hojas NO es una matriz de migración —es la
    distribución de toda la población por entidad de nacimiento— así que se descarta
    y se reconstruye sumando las 32 entidades de residencia.
    """
    filas = []
    for fila in _filas(archivo, hoja):
        if filtro_censo and _limpia(fila[filtro_censo[0]]) != filtro_censo[1]:
            continue
        if _entero(fila[cols[0]]) is None:
            continue
        cve_destino, etiq_origen = _cve(fila[col_ent]), _limpia(fila[col_org])
        cve_origen = _cve(etiq_origen)
        if not cve_destino or cve_origen == cve_destino or cve_origen == 0:
            continue
        if cve_origen is None and etiq_origen not in ORIGEN_NO_ENTIDAD:
            continue
        filas.append(_fila(
            censo, cve_destino, "En otra entidad", cve_origen,
            NOMBRE[cve_origen] if cve_origen else NO_ESPECIFICADA,
            [fila[c] for c in cols],
        ))
    return filas + _nacional_desde_estados(filas)


def _nacional_desde_estados(filas):
    """Suma el desglose de las 32 entidades para producir el bloque nacional."""
    if not filas:
        return []
    acum = {}
    for f in filas:
        clave = (f["cve_origen"], f["origen"])
        a = acum.setdefault(clave, {"personas": 0, "hombres": 0, "mujeres": 0})
        for k in a:
            a[k] += f[k] or 0
    plantilla = filas[0]
    return [
        {**plantilla, "cve_destino": 0, "destino": "Nacional",
         "cve_origen": cve, "origen": nombre, **valores}
        for (cve, nombre), valores in acum.items()
    ]


def hoja_unica(censo, archivo, hoja, col_ent, col_cat, col_org, cols, insesp_dentro):
    """Hojas de 1970/1980: agregados y desglose conviven. En 1980 la columna de
    categoría es la misma que la de origen — un valor con prefijo 'NN ' es un estado
    bajo la categoría implícita 'En otra entidad'.

    insesp_dentro: si 'Entidad insuficientemente especificada' forma parte de
    'En otra entidad' (1980, igual que 1990) o es una categoría hermana (1970).
    """
    filas = []
    for fila in _filas(archivo, hoja):
        if _entero(fila[cols[0]]) is None:
            continue
        cve_destino = _cve(fila[col_ent])
        if cve_destino is None:
            continue
        etiq_cat = _limpia(fila[col_cat])
        etiq_org = _limpia(fila[col_org]) if col_org != col_cat else etiq_cat

        es_estado = bool(re.match(r"^\d{2}\s", etiq_org)) or (
            col_org != col_cat and etiq_cat == "Nacidos en otra entidad"
            and etiq_org != "Total"
        )
        if es_estado:
            cve_origen = _cve(etiq_org)
            if cve_origen is None:
                raise ValueError(f"{archivo}/{hoja}: origen no reconocido {etiq_org!r}")
            filas.append(_fila(censo, cve_destino, "En otra entidad", cve_origen,
                               NOMBRE[cve_origen], [fila[c] for c in cols]))
        elif insesp_dentro and etiq_cat == "Entidad insuficientemente especificada":
            filas.append(_fila(censo, cve_destino, "En otra entidad", None,
                               NO_ESPECIFICADA, [fila[c] for c in cols]))
        elif etiq_cat in CATEGORIA:
            filas.append(_fila(censo, cve_destino, CATEGORIA[etiq_cat], None, None,
                               [fila[c] for c in cols]))
        elif etiq_cat not in PSEUDO:
            raise ValueError(f"{archivo}/{hoja}: categoría no reconocida {etiq_cat!r}")
    return filas


def historicos():
    """Las 4 ediciones 1950-1980, cada una con su layout."""
    f50 = "cgp50_nal_migracion.xlsx"
    f60 = "CGP60_nal_Migracion.xlsx"
    censo60 = (3, "1960")

    bloques = {
        1950: (
            agregados_por_sexo(1950, f50, "CGP50_Nal_Mig1", 1, 2,
                               {"Total": 3, "En la entidad": 4,
                                "En otra entidad": 5, "En otro país": 6})
            + matriz_con_diagonal(1950, f50, "CGP50_Nal_Mig2", 1, 2, [3, 4, 5])
        ),
        1960: (
            agregados_por_sexo(1960, f60, "CGP60_Nal_Mig3", 1, 2,
                               {"Total": 4, "En la entidad": 5,
                                "En otra entidad": 6, "En otro país": 7},
                               filtro_censo=censo60)
            + matriz_con_diagonal(1960, f60, "CGP60_Nal_Mig2", 1, 2, [4, 5, 6],
                                  filtro_censo=censo60)
        ),
        1970: hoja_unica(1970, "cgp70_nal_migracion.xlsx", "CGP70_Nal_Mig1",
                         1, 2, 3, [4, 5, 6], insesp_dentro=False),
        1980: hoja_unica(1980, "cpyv80_nal_migracion.xlsx", "CPyV80_Nal_Mig1",
                         1, 2, 2, [3, 4, 5], insesp_dentro=True),
    }
    for censo, filas in bloques.items():
        print(f"  {censo}  {len(filas):>6,} filas")
    return pl.DataFrame([f for filas in bloques.values() for f in filas], schema=ESQUEMA)


ESQUEMA = {
    "censo": pl.Int16, "cve_destino": pl.Int16, "destino": pl.Utf8,
    "categoria": pl.Utf8, "total_categoria": pl.Boolean,
    "cve_origen": pl.Int16, "origen": pl.Utf8,
    "personas": pl.Int64, "hombres": pl.Int64, "mujeres": pl.Int64,
}


def modernos():
    """1990-2020: ya parseados por prepare_ccpv_migracion.py."""
    if not PARQUET_5A.exists():
        raise SystemExit(
            f"Falta {PARQUET_5A}.\n"
            "Corre primero: uv run python scripts/prepare_ccpv_migracion.py"
        )
    df = (
        pl.read_parquet(PARQUET_5A)
        .filter(pl.col("concepto") == "nacimiento")
        .select(list(ESQUEMA))
    )
    for censo, g in df.group_by("censo", maintain_order=True):
        print(f"  {censo[0]}  {g.height:>6,} filas")
    return df


def validar(df):
    for (censo,), g in df.group_by("censo", maintain_order=True):
        estados = g.filter(pl.col("cve_destino") > 0)
        assert estados["cve_destino"].n_unique() == 32, \
            f"{censo}: {estados['cve_destino'].n_unique()} entidades destino"

        agregado = g.filter(pl.col("total_categoria"))
        desglose = g.filter(~pl.col("total_categoria"))

        # a) Las categorías suman la fila 'Total' de cada entidad.
        partes = (agregado.filter(pl.col("categoria") != "Total")
                  .group_by("cve_destino").agg(pl.col("personas").sum().alias("suma")))
        total = agregado.filter(pl.col("categoria") == "Total").select(
            ["cve_destino", pl.col("personas").alias("total")])
        d = partes.join(total, on="cve_destino").filter(pl.col("suma") != pl.col("total"))
        assert d.height == 0, f"{censo}: categorías no suman\n{d}"

        # b) El desglose por origen suma el agregado de 'En otra entidad'. En 1950 y
        #    1960 esto cruza DOS tabulados independientes (la matriz contra la hoja de
        #    agregados), que es la validación más fuerte de este script.
        partes = desglose.group_by("cve_destino").agg(pl.col("personas").sum().alias("suma"))
        total = agregado.filter(pl.col("categoria") == "En otra entidad").select(
            ["cve_destino", pl.col("personas").alias("total")])
        cuadre = partes.join(total, on="cve_destino")
        d = cuadre.filter((pl.col("cve_destino") > 0) & (pl.col("suma") != pl.col("total")))
        assert d.height == 0, f"{censo}: desglose no suma el agregado\n{d}"

        # El agregado nacional publicado en 1950 incluye el bloque 'Complementarios'
        # (residencia temporal en el extranjero, cuerpos diplomáticos), que no es una
        # entidad federativa y por tanto no aparece en la matriz. La nota de
        # CGP60_Nal_Mig3 lo cuantifica: 8 914 personas en la columna 'otras entidades'.
        nal = cuadre.filter(pl.col("cve_destino") == 0)
        hueco = (nal["total"] - nal["suma"]).sum()
        assert hueco == (8_914 if censo == 1950 else 0), \
            f"{censo}: el bloque nacional difiere del desglose en {hueco:,}"

        # c) Transponer la matriz reproduce el bloque nacional por origen.
        emig = (desglose.filter((pl.col("cve_destino") > 0) & pl.col("cve_origen").is_not_null())
                .group_by("cve_origen").agg(pl.col("personas").sum().alias("suma")))
        nal = desglose.filter((pl.col("cve_destino") == 0) & pl.col("cve_origen").is_not_null()
                              ).select(["cve_origen", pl.col("personas").alias("total")])
        d = emig.join(nal, on="cve_origen").filter(pl.col("suma") != pl.col("total"))
        assert d.height == 0 and emig.height == 32, \
            f"{censo}: matriz interestatal descuadrada\n{d}"

    def nacional(censo, categoria):
        return df.filter((pl.col("censo") == censo) & (pl.col("cve_destino") == 0)
                         & (pl.col("categoria") == categoria)
                         & pl.col("total_categoria"))["personas"].sum()

    esperado = {
        (1950, "Total"): 25_791_017, (1950, "En otra entidad"): 3_314_631,
        (1970, "Total"): 48_225_238, (1970, "En otra entidad"): 6_984_483,
        (1980, "Total"): 66_846_833, (1980, "En otra entidad"): 11_501_316,
        (1990, "En otra entidad"): 13_976_176,
        (2000, "En otra entidad"): 17_220_424,
        (2010, "En otra entidad"): 19_747_511,
        (2020, "Total"): 126_014_024, (2020, "En otra entidad"): 21_611_963,
    }
    for (censo, cat), val in esperado.items():
        obt = nacional(censo, cat)
        assert obt == val, f"{censo}/{cat}: {obt:,} != {val:,}"
    print(f"  {len(esperado)} cifras nacionales de INEGI verificadas")


def main():
    print("Censos históricos (1950-1980):")
    hist = historicos()
    print("Censos modernos (1990-2020), desde ccpv_migracion_estatal.parquet:")
    df = pl.concat([hist, modernos()]).sort(["censo", "cve_destino", "categoria"])

    print("Validación:")
    validar(df)

    df.write_parquet(SALIDA)
    print(f"Saved → {SALIDA}  ({df.height:,} filas, {df['censo'].n_unique()} censos)")


if __name__ == "__main__":
    main()
