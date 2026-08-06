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

Este script filtra siempre por un solo nivel a la vez — licenciatura+TSU
(DESC_SUBFUNCION="Educación Superior", NIVEL en licenciatura universitaria y
tecnológica/licenciatura en educación normal/técnico superior) por default, o
posgrado (DESC_SUBFUNCION="Posgrado", NIVEL en maestría/doctorado/especialidad)
con --posdoc — para que numerador y denominador midan lo mismo y no se mezclen
niveles con estructuras de costo muy distintas. Antes de agregar --posdoc el
script combinaba ambos niveles en un solo cálculo; --posdoc reemplaza ese
comportamiento por defecto (ya no hay opción de combinarlos).

Fuente presupuesto: data/presupuesto_federacion/presupuesto/egresos_federacion/
PEF_{año}.xlsx (mismos patrones de nombre que prepare_ramos_sector_presupuesto.py;
requiere columna DESC_SUBFUNCION, disponible desde el esquema 2018+).

Fuente matrícula: data/anuies/general/base_anuario_{ciclo}_general.xlsx (Anuario
Estadístico ANUIES). --lag elige qué ciclo usar: 0 (default) = el más reciente
disponible en data/anuies/general/, 1 = un ciclo atrás, etc. — útil porque el
anuario del ciclo más reciente suele tardar en publicarse, o porque se quiere
comparar un PEF viejo con la matrícula de su propio ciclo escolar en vez de la
más reciente. Si el --year del PEF no coincide con el año esperado para ese
ciclo (fin_de_ciclo + 1), se imprime un aviso porque la comparación mezcla años.

Los nombres de DESC_UR/DESC_PP cambian entre años — la UR de las universidades
estatales se llamó "Dirección General de Educación Superior Universitaria"
hasta 2021 y luego le agregaron "e Intercultural" (2022 en adelante); ambos
nombres están cubiertos en UR_ESTATALES. Verificado 2019-2026 (2018 hacia
atrás no tiene ciclo ANUIES disponible localmente, ver --lag). Si en el
futuro vuelve a cambiar el nombre y no está en UR_ESTATALES, el script
levanta el error de presupuesto-en-cero de abajo en vez de reportar una
cifra silenciosamente equivocada.

Por default solo imprime la tabla en terminal — no escribe nada a disco. Pasa
--save para guardar el parquet en dashboard_data/. El nombre del parquet lleva
sufijo _licenciatura o _posgrado según --posdoc.

Output (con --save): dashboard_data/comparacion_universidades_{año}_{nivel}.parquet
Run: uv run python scripts/prepare_comparacion_universidades.py --year 2026 --lag 0 --save
Run (posgrado): uv run python scripts/prepare_comparacion_universidades.py --year 2026 --posdoc

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
estatales por año. Respeta --posdoc igual que el camino de un solo año.
Run: uv run python scripts/prepare_comparacion_universidades.py --historico 2019-2026 --save

Con --states, en vez de la tabla UNAM/IPN/UAM/Estatales, muestra el costo por
alumno del subsidio DGESUI desglosado por entidad federativa (las 35
instituciones del subsistema ANUIES "UNIVERSIDADES PÚBLICAS ESTATALES"), usando
la columna DESC_ENTIDAD_FEDERATIVA del PEF para geolocalizar el gasto. También
respeta --posdoc (columnas de licenciatura o de posgrado, nunca ambas a la vez).
No es compatible con --historico. Caveat: en las 4 entidades con 2 instituciones
de este subsistema (Campeche, Chihuahua, Sinaloa, Sonora) el PEF no desglosa el
gasto por institución dentro del estado — el costo/alumno mostrado es el
promedio combinado de ambas, no el de cada una por separado.

--states también agrega, cuando está disponible, la APORTACIÓN ESTATAL (el
presupuesto propio del gobierno de la entidad hacia sus universidades
públicas, además del federal) y el TOTAL/costo por alumno combinado. La
fuente es el egresos.xlsx del portal de transparencia fiscal de cada estado,
clasificación "Administrativa" — ver APORTACION_ESTATAL_CONFIG. Hoy tiene
entrada verificada para **SONORA** (columna "Universidad de Sonora" en
data/entidades_federativas/sonora/egresos.xlsx, un solo archivo con años
2015-2023 como columnas — modo "columnas_por_año"; tiene un defecto de
unidades detectado y corregido: la hoja declara "miles de pesos" para toda la
tabla pero 2021-2023 ya vienen en pesos completos, se distingue por magnitud,
valor > 1e8 ⇒ ya está en pesos), para **CHIHUAHUA** (Universidad Autónoma de
Chihuahua + Universidad Autónoma de Cd. Juárez, columna DEVENGADO de
data/entidades_federativas/chihuahua/{año}/18*.xlsx — un archivo por año, con
nombre de archivo distinto cada año, localizado por glob; modo
"carpeta_por_año"; 2023 no lleva acentos en los nombres de institución, se
corrige emparejando sin acentos), y para **SINALOA** (Universidad Autónoma de
Sinaloa + Universidad Autónoma de Occidente, columna "Monto" de
data/entidades_federativas/sinaloa/{año}/*nexo 10.csv — mismo modo
"carpeta_por_año", pero en CSV con encoding iso-8859-1 en vez de xlsx; 2023
trae el nombre de archivo en minúscula ("anexo 10.csv") vs. 2024-2025 en
mayúscula, cubierto por el glob con comodín). Para agregar otra entidad, basta
con añadir una entrada a APORTACION_ESTATAL_CONFIG con el modo que corresponda
a cómo esa entidad publica su cuenta pública; las entidades ausentes del
diccionario muestran "s/d" en ESTATAL/TOTAL, no truenan.

Dos caveats de ESTATAL/TOTAL que no se pueden resolver con los datos
disponibles: (1) puede ser una cifra PARCIAL dentro de la entidad — en Sonora
solo "Universidad de Sonora" tiene línea propia, Instituto Tecnológico de
Sonora (la otra institución del subsistema en esa entidad) no aparece en el
archivo estatal, así que el ESTATAL de Sonora no cubre a ITSON (Chihuahua y
Sinaloa sí cubren a sus dos instituciones del subsistema). (2) ESTATAL no distingue nivel
(licenciatura vs. posgrado) — la fuente estatal reporta un solo monto por
universidad, todos los niveles mezclados — mientras que FEDERAL sí está
filtrado por nivel vía --posdoc. TOTAL, por lo tanto, siempre mezcla un
FEDERAL de un solo nivel con un ESTATAL de todos los niveles.
Run: uv run python scripts/prepare_comparacion_universidades.py --states --posdoc
"""

import argparse
import sys
import unicodedata
from pathlib import Path

import polars as pl

PEF_DIR = Path("data/presupuesto_federacion/presupuesto/egresos_federacion")
ANUIES_DIR = Path("data/anuies")
ANUIES_GENERAL_DIR = ANUIES_DIR / "general"
ENTIDADES_DIR = Path("data/entidades_federativas")
OUT_DIR = Path("dashboard_data")

# Configuración por entidad para leer su aportación estatal a universidades
# públicas estatales, desde el egresos.xlsx de su propio portal de
# transparencia fiscal (clasificación Administrativa). Solo se completa
# conforme se verifica cada entidad — las ausentes de este dict quedan sin
# dato ("s/d") en --states, no truenan.
APORTACION_ESTATAL_CONFIG = {
    "SONORA": {
        "modo": "columnas_por_año",  # un solo archivo, un año por columna
        "ruta": ENTIDADES_DIR / "sonora" / "egresos.xlsx",
        "hoja": "Administrativa",
        "ancla_header": "Capítulo / Concepto devengado",
        "instituciones": ["Universidad de Sonora"],  # ITSON no tiene línea propia en este archivo
        # La hoja declara "(miles de pesos)" para toda la tabla, pero 2021-2023
        # ya vienen en pesos completos (defecto detectado por magnitud).
        "corregir_unidad": lambda v: v if v is None or v > 1e8 else v * 1000,
    },
    "CHIHUAHUA": {
        "modo": "carpeta_por_año",  # un archivo distinto por año, adentro de ENTIDADES_DIR/chihuahua/{año}/
        "carpeta": ENTIDADES_DIR / "chihuahua",
        # el nombre del archivo cambia cada año (portada distinta), pero siempre
        # empieza con "18" (Clasificación Administrativa Entidades Paraestatales)
        "glob_por_año": "18*.xlsx",
        "ancla_header": "DEVENGADO",  # se usa DEVENGADO (no "Presupuesto Aprobado") para ser consistente con Sonora, que también reporta devengado
        # 2023 no lleva acentos en los nombres de institución (defecto de la
        # fuente); se corrige emparejando sin acentos, no hace falta listar
        # ambas variantes.
        "instituciones": ["UNIVERSIDAD AUTÓNOMA DE CHIHUAHUA", "UNIVERSIDAD AUTÓNOMA DE CD. JUÁREZ"],
        "corregir_unidad": lambda v: v,  # ya viene en pesos completos
    },
    "SINALOA": {
        "modo": "carpeta_por_año",  # un archivo distinto por año, adentro de ENTIDADES_DIR/sinaloa/{año}/
        "carpeta": ENTIDADES_DIR / "sinaloa",
        # "Anexo 10.csv" en 2024-2025, "anexo 10.csv" (minúscula) en 2023
        "glob_por_año": "*nexo 10.csv",
        "ancla_header": "Monto",
        "instituciones": ["Universidad Autónoma de Sinaloa", "Universidad Autónoma de Occidente"],
        "corregir_unidad": lambda v: v,  # ya viene en pesos completos
        "encoding": "iso-8859-1",  # CSV del portal de Sinaloa no viene en UTF-8
    },
}

sys.path.insert(0, str(Path(__file__).resolve().parent / "datatable"))

INSTITUCIONES_FEDERALES = {
    "UNAM": "Universidad Nacional Autónoma de México",
    "IPN": "Instituto Politécnico Nacional",
    "UAM": "Universidad Autónoma Metropolitana",
}
# ANUIES reporta INSTITUCIÓN en mayúsculas (DESC_UR del PEF, en cambio, va en Title Case)
INSTITUCIONES_FEDERALES_ANUIES = {sigla: nombre.upper() for sigla, nombre in INSTITUCIONES_FEDERALES.items()}
# La UR se renombró en algún año entre 2021 y 2022 (le agregaron "e Intercultural")
UR_ESTATALES = {
    "Dirección General de Educación Superior Universitaria e Intercultural",
    "Dirección General de Educación Superior Universitaria",
}
PP_ESTATALES = "Subsidios para organismos descentralizados estatales"
SUBSISTEMA_ESTATALES = "UNIVERSIDADES PÚBLICAS ESTATALES"

NIVELES_LIC = {"LICENCIATURA UNIVERSITARIA Y TECNOLÓGICA", "LICENCIATURA EN EDUCACIÓN NORMAL", "TÉCNICO SUPERIOR"}
NIVELES_POS = {"MAESTRÍA", "DOCTORADO", "ESPECIALIDAD"}
SUBFUNCION_LIC = "Educación Superior"
SUBFUNCION_POS = "Posgrado"


def nivel_activo(posdoc: bool) -> tuple[str, set[str], str]:
    """(nombre_nivel, niveles_anuies, subfuncion_pef) según --posdoc."""
    if posdoc:
        return "posgrado", NIVELES_POS, SUBFUNCION_POS
    return "licenciatura", NIVELES_LIC, SUBFUNCION_LIC


def _normalizar_institucion(s: str) -> str:
    """Mayúsculas + sin acentos, para emparejar nombres de institución que
    varían de acentuación entre años en la misma fuente estatal (ej. Chihuahua
    2023 sin acentos vs. 2024-2025 con acentos)."""
    s = s.strip().upper()
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _parsear_monto(valor) -> float | None:
    """'1,721,195,634\\xa0 ' -> 1721195634.0 (Chihuahua trae los montos como
    texto con separador de miles y NBSP); '$70,004,102,181' -> 70004102181.0
    (Sinaloa, signo de pesos en algunos renglones); si ya viene numérico
    (Sonora), lo devuelve tal cual."""
    if valor is None:
        return None
    if isinstance(valor, (int, float)):
        return float(valor)
    texto = str(valor).replace("\xa0", "").replace(",", "").replace("$", "").strip()
    return float(texto) if texto else None


def cargar_aportacion_estatal(entidad: str, year: int) -> float | None:
    """Aportación estatal (egresos.xlsx/archivo análogo del portal de hacienda
    de la propia entidad, clasificación Administrativa) a las instituciones
    mapeadas en APORTACION_ESTATAL_CONFIG para esa entidad, o None si la
    entidad no tiene fuente mapeada o el año no está disponible. Puede ser una
    cifra PARCIAL si la entidad tiene más de una institución en el subsistema
    ANUIES y solo algunas están mapeadas (ver caveat en el docstring del
    módulo). No mezcla "sin dato" con "aportación cero": si nada se
    encuentra, devuelve None.

    Dos modos de fuente, según config["modo"]:
    - "columnas_por_año": un solo archivo con un año por columna (Sonora).
    - "carpeta_por_año": un archivo distinto por año dentro de
      config["carpeta"]/{año}/, localizado por config["glob_por_año"] porque
      el nombre exacto del archivo cambia de un año a otro (xlsx en Chihuahua,
      csv con encoding variable — config["encoding"] — en Sinaloa)."""
    config = APORTACION_ESTATAL_CONFIG.get(entidad)
    if config is None:
        return None
    instituciones_norm = {_normalizar_institucion(i) for i in config["instituciones"]}
    modo = config.get("modo", "columnas_por_año")

    if modo == "columnas_por_año":
        ruta = config["ruta"]
        if not ruta.exists():
            return None
        filas = pl.read_excel(ruta, sheet_name=config["hoja"], has_header=False).rows()
        try:
            header = next(f for f in filas if f[0] == config["ancla_header"])
        except StopIteration:
            return None
        años = [int(float(a)) if a is not None else None for a in header[1:]]
        if year not in años:
            return None
        col_idx = años.index(year) + 1
    elif modo == "carpeta_por_año":
        carpeta_año = config["carpeta"] / str(year)
        if not carpeta_año.exists():
            return None
        candidatos = sorted(carpeta_año.glob(config["glob_por_año"]))
        if not candidatos:
            return None
        ruta = candidatos[0]
        if ruta.suffix.lower() == ".csv":
            filas = pl.read_csv(ruta, encoding=config.get("encoding", "utf8"), has_header=False).rows()
        else:
            filas = pl.read_excel(ruta, sheet_id=1, has_header=False).rows()
        try:
            header = next(f for f in filas if config["ancla_header"] in f)
        except StopIteration:
            return None
        col_idx = header.index(config["ancla_header"])
    else:
        raise ValueError(f"modo desconocido en APORTACION_ESTATAL_CONFIG[{entidad!r}]: {modo!r}")

    total = 0.0
    encontrado = False
    for fila in filas:
        if not isinstance(fila[0], str) or _normalizar_institucion(fila[0]) not in instituciones_norm:
            continue
        valor = _parsear_monto(fila[col_idx])
        if valor is None:
            continue
        total += config["corregir_unidad"](valor)
        encontrado = True
    return total if encontrado else None


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
    archivos = sorted(ANUIES_GENERAL_DIR.glob("base_anuario_*_general.xlsx"), reverse=True)
    if not archivos:
        raise FileNotFoundError(f"No se encontraron anuarios ANUIES (base_anuario_*_general.xlsx) en {ANUIES_GENERAL_DIR}")
    if lag < 0 or lag >= len(archivos):
        ciclos = [a.stem.removeprefix("base_anuario_").removesuffix("_general") for a in archivos]
        raise ValueError(f"--lag {lag} fuera de rango: solo hay {len(archivos)} ciclo(s) disponible(s) en {ANUIES_GENERAL_DIR} ({ciclos})")
    ruta = archivos[lag]
    ciclo = ruta.stem.removeprefix("base_anuario_").removesuffix("_general")
    año_fin_ciclo = int(ciclo.split("-")[1])
    return ruta, ciclo, año_fin_ciclo + 1


def presupuesto_educativo(pef: pl.DataFrame, col_monto: str, desc_ur: str, posdoc: bool) -> float:
    _, _, subfuncion = nivel_activo(posdoc)
    sub = pef.filter((pl.col("DESC_UR") == desc_ur) & (pl.col("DESC_SUBFUNCION") == subfuncion))
    return sub[col_monto].sum()


def cargar_matricula(anuies_path: Path, posdoc: bool) -> dict[str, int]:
    nombre_nivel, niveles, _ = nivel_activo(posdoc)
    anuies = pl.read_excel(anuies_path, sheet_name="Base de datos")
    anuies = anuies.filter(pl.col("NIVEL").is_in(niveles))
    matricula = {
        sigla: anuies.filter(pl.col("INSTITUCIÓN") == nombre)["Matrícula Total"].sum()
        for sigla, nombre in INSTITUCIONES_FEDERALES_ANUIES.items()
    }
    matricula["ESTATALES"] = anuies.filter(pl.col("SUBSISTEMA") == SUBSISTEMA_ESTATALES)["Matrícula Total"].sum()
    sin_matricula = [k for k, v in matricula.items() if not v]
    if sin_matricula:
        raise ValueError(f"Matrícula 0 en {nombre_nivel} para {sin_matricula} — revisar nombres de INSTITUCIÓN/SUBSISTEMA en {anuies_path}")
    return matricula


def calcular_tabla_año(year: int, lag: int, posdoc: bool) -> tuple[pl.DataFrame, Path, Path, str, int]:
    """Presupuesto y matrícula ANUIES de un solo nivel (licenciatura+TSU, o
    posgrado con posdoc=True) para un solo año. Devuelve (tabla, pef_path,
    anuies_path, ciclo, año_esperado)."""
    nombre_nivel, _, subfuncion = nivel_activo(posdoc)
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

    matricula = cargar_matricula(anuies_path, posdoc)

    filas = [
        (sigla, nombre, presupuesto_educativo(raw, col_monto, nombre, posdoc), matricula[sigla])
        for sigla, nombre in INSTITUCIONES_FEDERALES.items()
    ]

    estatales_raw = raw.filter(
        (pl.col("DESC_UR").is_in(UR_ESTATALES))
        & (pl.col("DESC_PP") == PP_ESTATALES)
        & (pl.col("DESC_SUBFUNCION") == subfuncion)
    )
    filas.append(("ESTATALES", "Universidades Públicas Estatales (35, vía DGESUI)", estatales_raw[col_monto].sum(), matricula["ESTATALES"]))

    sin_presupuesto = [nombre for _, nombre, presupuesto, _ in filas if not presupuesto]
    if sin_presupuesto:
        raise ValueError(
            f"Presupuesto 0 en {nombre_nivel} para {sin_presupuesto} en {pef_path.name} — probablemente DESC_UR/DESC_PP "
            "cambiaron de nombre en este año (el script solo está verificado para 2019-2026)."
        )

    tabla = pl.DataFrame(filas, schema=["SIGLA", "INSTITUCION", "PRESUPUESTO", "MATRICULA"], orient="row")
    tabla = tabla.with_columns(
        (pl.col("PRESUPUESTO") / pl.col("MATRICULA")).alias("GASTO_POR_ALUMNO"),
        pl.lit(year).alias("AÑO_PEF"),
    )
    return tabla, pef_path, anuies_path, ciclo, año_esperado


def calcular_tabla_entidades(year: int, lag: int, posdoc: bool) -> tuple[pl.DataFrame, Path, Path, str, int]:
    """Costo por alumno del subsidio DGESUI (universidades públicas estatales)
    desglosado por entidad federativa, para un solo nivel (licenciatura+TSU, o
    posgrado con posdoc=True). Devuelve (tabla, pef_path, anuies_path, ciclo,
    año_esperado); tabla trae una fila TOTAL con las sumas nacionales."""
    _, niveles, subfuncion = nivel_activo(posdoc)
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
    for columna in ("DESC_SUBFUNCION", "DESC_ENTIDAD_FEDERATIVA"):
        if columna not in raw.columns:
            raise ValueError(f"{pef_path.name} no tiene columna {columna}, requerida para --states.")

    sub = raw.filter((pl.col("DESC_UR").is_in(UR_ESTATALES)) & (pl.col("DESC_SUBFUNCION") == subfuncion))
    por_ent_pef = sub.group_by("DESC_ENTIDAD_FEDERATIVA").agg(pl.sum(col_monto).alias("PRESUPUESTO"))
    por_ent_pef = por_ent_pef.with_columns(
        pl.when(pl.col("DESC_ENTIDAD_FEDERATIVA") == "Estado de México")
        .then(pl.lit("MÉXICO"))
        .otherwise(pl.col("DESC_ENTIDAD_FEDERATIVA").str.to_uppercase())
        .alias("ENTIDAD")
    )
    if por_ent_pef["PRESUPUESTO"].sum() == 0:
        raise ValueError(f"Presupuesto 0 para --states en {pef_path.name} — revisar UR_ESTATALES/subfunción para este año.")

    anuies = pl.read_excel(anuies_path, sheet_name="Base de datos")
    estatales = anuies.filter((pl.col("SUBSISTEMA") == SUBSISTEMA_ESTATALES) & (pl.col("NIVEL").is_in(niveles)))
    mat_ent = estatales.group_by("ENTIDAD").agg(
        pl.sum("Matrícula Total").alias("MATRICULA"),
        pl.col("INSTITUCIÓN").n_unique().alias("N_INSTITUCIONES"),
        pl.col("ESCUELA/CAMPUS/PLANTEL").n_unique().alias("N_PLANTELES"),
    )
    if mat_ent["MATRICULA"].sum() == 0:
        raise ValueError(f"Matrícula 0 para --states en {anuies_path} — revisar SUBSISTEMA_ESTATALES/nivel para este ciclo.")

    tabla = mat_ent.join(por_ent_pef.select("ENTIDAD", "PRESUPUESTO"), on="ENTIDAD", how="left").fill_null(0)
    tabla = tabla.rename({"PRESUPUESTO": "FEDERAL"})
    tabla = tabla.with_columns(
        pl.col("ENTIDAD").map_elements(lambda e: cargar_aportacion_estatal(e, year), return_dtype=pl.Float64).alias("ESTATAL")
    )
    tabla = tabla.with_columns(
        (pl.col("FEDERAL") / pl.col("MATRICULA")).alias("COSTO_ALUMNO_FEDERAL"),
        (pl.col("FEDERAL") + pl.col("ESTATAL")).alias("TOTAL"),
    )
    tabla = tabla.with_columns((pl.col("TOTAL") / pl.col("MATRICULA")).alias("COSTO_ALUMNO_TOTAL"))
    tabla = tabla.sort("COSTO_ALUMNO_FEDERAL", descending=True)

    entidades_con_estatal = tabla.filter(pl.col("ESTATAL").is_not_null())
    estatal_nacional = entidades_con_estatal["ESTATAL"].sum() if entidades_con_estatal.height > 0 else None

    total = pl.DataFrame(
        {
            "ENTIDAD": ["TOTAL"],
            "MATRICULA": [tabla["MATRICULA"].sum()],
            "N_INSTITUCIONES": [tabla["N_INSTITUCIONES"].sum()],
            "N_PLANTELES": [tabla["N_PLANTELES"].sum()],
            "FEDERAL": [tabla["FEDERAL"].sum()],
            "ESTATAL": [estatal_nacional],  # parcial: solo suma entidades con dato (ver aviso en main()); None si ninguna entidad tiene dato para este año
        }
    ).with_columns(
        (pl.col("FEDERAL") / pl.col("MATRICULA")).alias("COSTO_ALUMNO_FEDERAL"),
        (pl.col("FEDERAL") + pl.col("ESTATAL")).alias("TOTAL"),
    )
    total = total.with_columns((pl.col("TOTAL") / pl.col("MATRICULA")).alias("COSTO_ALUMNO_TOTAL"))
    total = total.select([pl.col(c).cast(tabla.schema[c]) for c in tabla.columns])
    tabla = pl.concat([tabla, total])

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


def correr_historico(historico: str, save: bool, posdoc: bool):
    from poder_adquisitivo_nacional import cargar_indice_inpc

    nombre_nivel, _, _ = nivel_activo(posdoc)
    inicio, fin = parsear_rango(historico)
    _, _, año_esperado_lag0 = encontrar_anuies(0)

    tablas = []
    for year in range(inicio, fin + 1):
        lag = año_esperado_lag0 - year
        if lag < 0:
            print(f"[omitido] {year}: no hay ciclo ANUIES tan reciente disponible localmente.")
            continue
        try:
            tabla, pef_path, anuies_path, ciclo, año_esperado = calcular_tabla_año(year, lag, posdoc)
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

    print(f"\nNivel: {nombre_nivel}")
    print(f"Años calculados: {años_calculados} (pesos constantes de {año_base})\n")
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
        out_path = OUT_DIR / f"comparacion_universidades_historico_{inicio}_{fin}_{nombre_nivel}.parquet"
        combinada.write_parquet(out_path)
        print(f"\nGuardado → {out_path}")
    else:
        print("\n(no guardado — pasa --save para escribir el parquet)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--lag", type=int, default=0, help="Ciclo ANUIES a usar: 0 = más reciente disponible, 1 = un ciclo atrás, etc.")
    parser.add_argument("--historico", type=str, default=None, metavar="INICIO-FIN", help="Corre un rango de años (ej. 2019-2026) encadenando --lag automáticamente, y muestra la evolución real (deflactada con INPC) en vez de la tabla de un solo año.")
    parser.add_argument("--states", action="store_true", help="Muestra el costo por alumno por entidad federativa (subsidio DGESUI vía universidades públicas estatales) en vez de la tabla UNAM/IPN/UAM/Estatales. No compatible con --historico.")
    parser.add_argument("--posdoc", action="store_true", help="Usa posgrado (maestría+doctorado+especialidad, DESC_SUBFUNCION=Posgrado) en vez de licenciatura+TSU (default) en todos los cálculos.")
    parser.add_argument("--save", action="store_true", help="Guardar el resultado como parquet en dashboard_data/ (por default no se guarda, solo se imprime)")
    args = parser.parse_args()

    if args.historico and args.states:
        parser.error("--historico y --states no se pueden combinar")

    if args.historico:
        correr_historico(args.historico, args.save, args.posdoc)
        return

    nombre_nivel, _, _ = nivel_activo(args.posdoc)
    year = args.year

    if args.states:
        tabla, pef_path, anuies_path, ciclo, año_esperado = calcular_tabla_entidades(year, args.lag, args.posdoc)

        print(f"Fuente presupuesto: {pef_path}")
        print(f"Fuente matrícula: {anuies_path} (ciclo {ciclo})")
        print(f"Nivel: {nombre_nivel}")
        if year != año_esperado:
            print(f"[aviso] --year {year} no coincide con el año esperado para el ciclo {ciclo} ({año_esperado}) — la comparación mezcla presupuesto y matrícula de años distintos.")
        con_dato = sorted(APORTACION_ESTATAL_CONFIG.keys())
        print(f"[aviso] Aportación estatal (ESTATAL/TOTAL) solo disponible para: {con_dato} — el resto muestra 's/d'. La fila TOTAL nacional de ESTATAL/TOTAL es parcial (solo suma las entidades con dato).")
        print("[aviso] ESTATAL no distingue nivel (licenciatura/posgrado) — la fuente estatal reporta un solo monto por universidad, mientras FEDERAL sí está filtrado por --posdoc. TOTAL mezcla ambos criterios.")
        print()

        def fmt(v):
            return f"{v:>15,.0f}" if v is not None else f"{'s/d':>15}"

        print(f"{'Entidad':<20} {'Insts':>6} {'Planteles':>10} {'Matrícula':>10} {'Federal':>15} {'Estatal':>15} {'Total':>15} {'Costo/al.Fed':>13} {'Costo/al.Total':>15}")
        for r in tabla.iter_rows(named=True):
            print(
                f"{r['ENTIDAD']:<20} {r['N_INSTITUCIONES']:>6,} {r['N_PLANTELES']:>10,} {r['MATRICULA']:>10,} "
                f"{r['FEDERAL']:>15,.0f} {fmt(r['ESTATAL'])} {fmt(r['TOTAL'])} {r['COSTO_ALUMNO_FEDERAL']:>13,.0f} "
                f"{fmt(r['COSTO_ALUMNO_TOTAL'])}"
            )

        if args.save:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            out_path = OUT_DIR / f"comparacion_universidades_entidades_{year}_{nombre_nivel}.parquet"
            tabla.write_parquet(out_path)
            print(f"\nGuardado → {out_path}")
        else:
            print("\n(no guardado — pasa --save para escribir el parquet)")
        return

    tabla, pef_path, anuies_path, ciclo, año_esperado = calcular_tabla_año(year, args.lag, args.posdoc)

    print(f"Fuente presupuesto: {pef_path}")
    print(f"Fuente matrícula: {anuies_path} (ciclo {ciclo})")
    print(f"Nivel: {nombre_nivel}")
    if year != año_esperado:
        print(f"[aviso] --year {year} no coincide con el año esperado para el ciclo {ciclo} ({año_esperado}) — la comparación mezcla presupuesto y matrícula de años distintos.")
    print()
    print(f"{'Institución':<52} {'Presupuesto (M)':>16} {'Matrícula':>10} {'Gasto/alumno':>14}")
    for r in tabla.iter_rows(named=True):
        print(f"{r['INSTITUCION']:<52} {r['PRESUPUESTO']/1e6:>16,.0f} {r['MATRICULA']:>10,} {r['GASTO_POR_ALUMNO']:>14,.0f}")

    if args.save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"comparacion_universidades_{year}_{nombre_nivel}.parquet"
        tabla.write_parquet(out_path)
        print(f"\nGuardado → {out_path}")
    else:
        print("\n(no guardado — pasa --save para escribir el parquet)")


if __name__ == "__main__":
    main()
