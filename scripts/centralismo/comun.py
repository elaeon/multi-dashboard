"""
Módulo común del informe centralismo (centralismo/informe/).

- Catálogo canónico de las 32 entidades federativas (CVE_ENT INEGI 1-32).
- normalizar_estado(): resuelve los ~6 regímenes de nombres del repo a CVE_ENT.
- Loaders de paneles ya preparados (población CONAPO, pobreza CONEVAL, ENVIPE).
- Parser de tabulados PIBE (data/inegi/pibe/) por bloque.
- guardar_fig(): export PNG homogéneo a centralismo/informe/figuras/.

Self-test: uv run python scripts/centralismo/comun.py
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd
import polars as pl

RAIZ = Path(__file__).resolve().parents[2]
DIR_FIGURAS = RAIZ / "centralismo" / "informe" / "figuras"

# ---------------------------------------------------------------- catálogo

# CVE_ENT INEGI → (nombre canónico de despliegue, nombre corto ASCII)
ESTADOS = {
    1: ("Aguascalientes", "Aguascalientes"),
    2: ("Baja California", "Baja California"),
    3: ("Baja California Sur", "Baja California Sur"),
    4: ("Campeche", "Campeche"),
    5: ("Coahuila", "Coahuila"),
    6: ("Colima", "Colima"),
    7: ("Chiapas", "Chiapas"),
    8: ("Chihuahua", "Chihuahua"),
    9: ("Ciudad de México", "CDMX"),
    10: ("Durango", "Durango"),
    11: ("Guanajuato", "Guanajuato"),
    12: ("Guerrero", "Guerrero"),
    13: ("Hidalgo", "Hidalgo"),
    14: ("Jalisco", "Jalisco"),
    15: ("Estado de México", "Edomex"),
    16: ("Michoacán", "Michoacán"),
    17: ("Morelos", "Morelos"),
    18: ("Nayarit", "Nayarit"),
    19: ("Nuevo León", "Nuevo León"),
    20: ("Oaxaca", "Oaxaca"),
    21: ("Puebla", "Puebla"),
    22: ("Querétaro", "Querétaro"),
    23: ("Quintana Roo", "Quintana Roo"),
    24: ("San Luis Potosí", "San Luis Potosí"),
    25: ("Sinaloa", "Sinaloa"),
    26: ("Sonora", "Sonora"),
    27: ("Tabasco", "Tabasco"),
    28: ("Tamaulipas", "Tamaulipas"),
    29: ("Tlaxcala", "Tlaxcala"),
    30: ("Veracruz", "Veracruz"),
    31: ("Yucatán", "Yucatán"),
    32: ("Zacatecas", "Zacatecas"),
}

NOMBRE = {cve: n for cve, (n, _) in ESTADOS.items()}
CORTO = {cve: c for cve, (_, c) in ESTADOS.items()}

# Regiones (convención analítica del informe; ver INFORME.md)
REGION = {
    "Noroeste": [2, 3, 8, 25, 26],
    "Noreste": [5, 10, 19, 28, 32],
    "Occidente": [1, 6, 11, 14, 16, 18],
    "Centro": [9, 13, 15, 17, 21, 22, 29],
    "Sur-Sureste": [4, 7, 12, 20, 23, 24, 27, 30, 31],
}
REGION_DE = {cve: reg for reg, cves in REGION.items() for cve in cves}


def _clave(s: str) -> str:
    """'Veracruz de Ignacio de la Llave ' → 'veracruz de ignacio de la llave'."""
    s = unicodedata.normalize("NFD", str(s))
    s = s.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


# Alias observados en los datasets del repo (claves ya pasadas por _clave)
_ALIAS = {}
for cve, (nombre, corto) in ESTADOS.items():
    _ALIAS[_clave(nombre)] = cve
    _ALIAS[_clave(corto)] = cve
_ALIAS.update({
    # nombres oficiales largos (INEGI/SNII/Subsidios)
    "coahuila de zaragoza": 5,
    "michoacan de ocampo": 16,
    "veracruz de ignacio de la llave": 30,
    "queretaro de arteaga": 22,
    "mexico": 15,                # PIBE/SEMARNAT: 'México' = Estado de México
    "estado de mexico": 15,
    "distrito federal": 9,       # pre-2016 (SIE, SGM)
    "df": 9,
    "cdmx": 9,
    "ciudad de mexico": 9,
    # slugs (tollbooths)
    "edomx": 15,
    "edomex": 15,
    "bcs": 3,
    "bc": 2,
    "nl": 19,
    "slp": 24,
    "qroo": 23,
})

# etiquetas de agregados que NO son estados
_NACIONAL = {
    "estados unidos mexicanos", "nacional", "total", "total nacional",
    "republica mexicana", "promedio nacional", "extranjero", "en el extranjero",
    "no distribuible geograficamente", "no especificado", "no aplica", "sin dato",
}


def normalizar_estado(nombre, estricto: bool = False):
    """Nombre de estado en cualquier régimen del repo → CVE_ENT (int 1-32).

    Devuelve None para agregados nacionales/extranjero/no-distribuible.
    Con estricto=True lanza ValueError si el nombre no se reconoce.
    """
    if nombre is None:
        return None
    k = _clave(nombre)
    if not k or k in _NACIONAL:
        return None
    if k in _ALIAS:
        return _ALIAS[k]
    if estricto:
        raise ValueError(f"Estado no reconocido: {nombre!r} (clave {k!r})")
    return None


def con_cve(df: pl.DataFrame, col: str, estricto: bool = True) -> pl.DataFrame:
    """Agrega columna 'cve_ent' a un DataFrame polars a partir de la columna de nombres."""
    mapa = {v: normalizar_estado(v) for v in df[col].unique().to_list()}
    no_rec = [v for v, c in mapa.items() if c is None and _clave(v or "") not in _NACIONAL]
    if estricto and no_rec:
        raise ValueError(f"Nombres de estado no reconocidos en {col!r}: {no_rec}")
    return df.with_columns(pl.col(col).replace_strict(mapa, default=None).alias("cve_ent"))


# ---------------------------------------------------------------- loaders

def cargar_poblacion() -> pl.DataFrame:
    """CONAPO: cve_ent, año, pob_total (1990-2040)."""
    pob = pl.read_parquet(RAIZ / "data/conapo/proyecciones_poblacion/pob_estado_año.parquet")
    pob = con_cve(pob, "estado")
    assert pob["cve_ent"].n_unique() == 32
    return pob.select("cve_ent", "año", "pob_total")


def cargar_pobreza() -> pl.DataFrame:
    """CONEVAL: panel estatal 2016/2018/2020/2022 con cve_ent."""
    dep = pl.read_parquet(RAIZ / "data/coneval/deprivacion_panel.parquet")
    dep = con_cve(dep, "estado")
    assert dep["cve_ent"].n_unique() == 32 and dep.height == 128
    return dep


def cargar_envipe() -> pl.DataFrame:
    """ENVIPE: panel estatal 2017-2025 (CLAVE_ENT ya es CVE_ENT)."""
    env = pl.read_parquet(RAIZ / "data/inegi/envipe/envipe_state_panel.parquet")
    return env.rename({"CLAVE_ENT": "cve_ent"})


# ---------------------------------------------------------------- PIBE

DIR_PIBE = RAIZ / "data/inegi/pibe/tabulados"

# tabulados clave (ver data/inegi/pibe/manifest.csv)
PIBE_TOTAL = 2
PIBE_MINERIA = 13
PIBE_MINERIA_PETROLERA = 14
PIBE_MINERIA_NO_PETROLERA = 15
PIBE_EDUCACION = 41
PIBE_SALUD = 42
PIBE_GOBIERNO = 46

BLOQUES_PIBE = [
    "Millones de pesos a precios de 2018",
    "Variación porcentual anual en valores constantes",
    "Estructura porcentual en valores constantes",
    "Contribución a la variación nacional en valores constantes",
    "Índice de volumen físico, base 2018=100",
    "Millones de pesos",
    "Variación porcentual anual en valores corrientes",
    "Estructura porcentual en valores corrientes",
    "Contribución a la variación nacional en valores corrientes",
    "Índice de precios implícitos base 2018=100",
]


def leer_pibe(n: int, bloque: str = "Millones de pesos a precios de 2018",
              incluir_nacional: bool = False) -> pl.DataFrame:
    """Tabulado PIBE_<n> → largo: cve_ent, año, valor (para el bloque pedido).

    Solo aplica a PIBE_2-46 (una actividad × 33 filas de entidades).
    """
    if bloque not in BLOQUES_PIBE:
        raise ValueError(f"Bloque desconocido: {bloque!r}")
    raw = pd.read_excel(DIR_PIBE / f"PIBE_{n}.xlsx", sheet_name="Tabulado", header=None)
    col_a = raw[0].astype(str).str.strip()

    inicio = col_a[col_a == bloque].index
    if len(inicio) != 1:
        raise ValueError(f"PIBE_{n}: bloque {bloque!r} hallado {len(inicio)} veces")
    i0 = inicio[0]
    resto = [i for i in col_a[col_a.isin(BLOQUES_PIBE)].index if i > i0]
    i1 = resto[0] if resto else len(raw)

    años = pd.to_numeric(raw.iloc[4, 1:].astype(str).str.replace("R", "", regex=False),
                         errors="coerce")
    filas = []
    for _, row in raw.iloc[i0 + 1: i1].iterrows():
        etiqueta = str(row[0]).strip()
        cve = normalizar_estado(etiqueta)
        es_nacional = _clave(etiqueta) in _NACIONAL and "mexicanos" in _clave(etiqueta)
        if cve is None and not (incluir_nacional and es_nacional):
            continue
        for j, año in años.items():
            if pd.isna(año):
                continue
            val = pd.to_numeric(row[j], errors="coerce")
            if pd.notna(val):
                filas.append((0 if cve is None else cve, int(año), float(val)))

    df = pl.DataFrame(filas, schema=["cve_ent", "año", "valor"], orient="row")
    n_est = df.filter(pl.col("cve_ent") > 0)["cve_ent"].n_unique()
    assert n_est == 32, f"PIBE_{n}/{bloque}: {n_est} estados (esperaba 32)"
    return df


# ---------------------------------------------------------------- figuras

TEMA = dict(template="plotly_white", font=dict(family="Lato, Arial", size=13))


def guardar_fig(fig, nombre: str, ancho: int = 1000, alto: int = 600) -> Path:
    """Exporta la figura a centralismo/informe/figuras/<nombre>.png y devuelve la ruta."""
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.update_layout(**TEMA)
    destino = DIR_FIGURAS / f"{nombre}.png"
    fig.write_image(str(destino), width=ancho, height=alto, scale=2)
    print(f"[fig] {destino.relative_to(RAIZ)}")
    return destino


# ---------------------------------------------------------------- self-test

if __name__ == "__main__":
    casos = {
        # (entrada observada en datasets del repo) → cve esperado
        "Veracruz de Ignacio de la Llave": 30,   # PIBE/SNII/Subsidios
        "VERACRUZ": 30,                          # SECIHTI 2025
        "Ciudad De Mexico": 9,                   # pob_estado_año.parquet
        "Distrito Federal": 9,                   # SIE pre-2016
        "cdmx": 9,                               # tollbooths slug
        "edomx": 15,                             # tollbooths slug
        "México": 15,                            # PIBE/SEMARNAT
        "Estado de México": 15,                  # CONAFOR 2015-2024
        "Coahuila de Zaragoza": 5,
        "MICHOACÁN DE OCAMPO": 16,
        "Michoacan": 16,                         # panels ASCII
        "Nuevo Leon": 19,
        "san luis potosi": 24,
    }
    for entrada, esperado in casos.items():
        obtenido = normalizar_estado(entrada, estricto=True)
        assert obtenido == esperado, f"{entrada!r}: {obtenido} != {esperado}"
    for agg in ["Estados Unidos Mexicanos", "Total", "No distribuible geográficamente"]:
        assert normalizar_estado(agg) is None, agg
    print(f"normalizar_estado: {len(casos)} casos OK")

    pob = cargar_poblacion()
    p24 = pob.filter(pl.col("año") == 2024)["pob_total"].sum()
    assert 125_000_000 < p24 < 135_000_000, p24
    print(f"población 2024: {p24/1e6:.1f} M OK")

    dep = cargar_pobreza()
    print(f"pobreza CONEVAL: {dep.height} filas OK")

    pib = leer_pibe(PIBE_TOTAL, incluir_nacional=True)
    nac2024 = pib.filter((pl.col("cve_ent") == 0) & (pl.col("año") == 2024))["valor"][0]
    suma2024 = pib.filter((pl.col("cve_ent") > 0) & (pl.col("año") == 2024))["valor"].sum()
    assert abs(suma2024 / nac2024 - 1) < 0.01, (suma2024, nac2024)
    print(f"PIBE 2024: nacional {nac2024/1e6:.2f} B (2018 MXN); suma estados cuadra OK")

    corr = leer_pibe(PIBE_TOTAL, bloque="Millones de pesos")
    print(f"PIBE corrientes: {corr.height} filas OK")
    print("comun.py: self-test completo")
