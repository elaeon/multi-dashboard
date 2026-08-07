"""
Comparación de presupuesto federal (PEF) por alumno entre UNAM, IPN, UAM y las
universidades del subsidio DGESUI (públicas estatales + interculturales + de
apoyo solidario, ver UNIVERSIDADES_DGESUI_2025), usando matrícula real de
ANUIES como denominador.

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
alumno del subsidio DGESUI desglosado por entidad federativa, usando la
columna DESC_ENTIDAD_FEDERATIVA del PEF para geolocalizar el gasto. La
matrícula (denominador) suma los tres subsistemas ANUIES "UNIVERSIDADES
PÚBLICAS ESTATALES", "UNIVERSIDADES INTERCULTURALES" y "UNIVERSIDADES
PÚBLICAS ESTATALES DE APOYO SOLIDARIO" (SUBSISTEMAS_DGESUI) porque el PEF no
desglosa el subsidio DGESUI por institución dentro de una entidad — la misma
cifra financia a las tres por igual, así que contarlas por separado inflaría
artificialmente el costo/alumno de las entidades con universidad intercultural
o de apoyo solidario. Cada institución se verifica además contra
UNIVERSIDADES_DGESUI_2025 (lista oficial de las 78 universidades del programa
"Subsidio Ordinario" DGESUI 2025) antes de entrar al cálculo; las que no
matchean se excluyen y se listan en un aviso (ver no_verificadas). También
respeta --posdoc (columnas de licenciatura o de posgrado, nunca ambas a la
vez). No es compatible con --historico. Caveat: en las entidades con más de
una institución bajo SUBSISTEMAS_DGESUI (ver columnas
N_INSTITUCIONES/N_PLANTELES impresas en la tabla) el PEF no desglosa el gasto
por institución dentro del estado — el costo/alumno mostrado es el promedio
combinado de todas ellas, no el de cada una por separado.

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
Chihuahua + Universidad Autónoma de Cd. Juárez + El Colegio de Chihuahua
(apoyo solidario, monto pequeño), columna DEVENGADO de
data/entidades_federativas/chihuahua/{año}/18*.xlsx — un archivo por año, con
nombre de archivo distinto cada año, localizado por glob; modo
"carpeta_por_año"; 2023 no lleva acentos en los nombres de institución, se
corrige emparejando sin acentos), y para **SINALOA** (Universidad Autónoma de
Sinaloa + Universidad Autónoma de Occidente + Universidad Autónoma Indígena
de México (Intercultural), columna "Monto" de
data/entidades_federativas/sinaloa/{año}/*nexo 10.csv — mismo modo
"carpeta_por_año", pero en CSV con encoding iso-8859-1 en vez de xlsx; 2023
trae el nombre de archivo en minúscula ("anexo 10.csv") vs. 2024-2025 en
mayúscula, cubierto por el glob con comodín), y para **BAJA CALIFORNIA**
(Universidad Autónoma de Baja California, única institución del subsistema en
esa entidad, columna "Asignación Presupuestal" de
data/entidades_federativas/baja_california/{año}/Clasificación
[Aa]dministrativ*.xlsx — modo "carpeta_por_año"; a diferencia de las otras
tres entidades el nombre de la institución va en la columna B, no la A
(config["col_institucion"] = 1), y lleva un prefijo de clave presupuestal
delante del nombre), y para **BAJA CALIFORNIA SUR** (Universidad Autónoma de
Baja California Sur, única institución del subsistema en esa entidad, columna
"MONTO" de data/entidades_federativas/baja_california_sur/{año}/ANEXO I-15
SECTOR EDUCATIVO.xlsx, sección "II.- Capítulo 4000" — modo "carpeta_por_año";
solo verificada para 2025, el archivo de 2024 tiene otra estructura y no
incluye a la UABCS como dependencia propia; la UABCS aparece 3 veces en el
archivo con montos distintos — estatal, federal Ramo 33, y total combinado —
por lo que se usa config["desde_ancla"]/["hasta_ancla"] para acotar la
búsqueda solo a la sección de financiamiento estatal). Para agregar otra entidad, basta
con añadir una entrada a APORTACION_ESTATAL_CONFIG con el modo que corresponda
a cómo esa entidad publica su cuenta pública; las entidades ausentes del
diccionario muestran "s/d" en ESTATAL/TOTAL, no truenan.

Tres caveats de ESTATAL/TOTAL que no se pueden resolver con los datos
disponibles: (1) puede ser una cifra PARCIAL dentro de la entidad — en Sonora
solo "Universidad de Sonora" tiene línea propia, ni Instituto Tecnológico de
Sonora, ni Universidad del Pueblo Yaqui (Intercultural), ni El Colegio de
Sonora/Universidad de la Sierra/Universidad Estatal de Sonora (apoyo
solidario) -- las otras cinco instituciones DGESUI de esa entidad -- aparecen
en el archivo estatal, así que el ESTATAL de Sonora no las cubre. En Baja
California, Universidad Intercultural de Baja California sí tiene línea
propia en un archivo local distinto ("Presupuesto <año>-Baja
California-Datos abiertos.xlsx") pero con estructura inconsistente entre
años (filas jerárquicas en 2023/2025, columnas planas en otra hoja en 2024
porque la hoja normal viene rota) que no encaja en el mecanismo genérico de
APORTACION_ESTATAL_CONFIG sin un parser ad-hoc — queda pendiente de
integrar, así que el ESTATAL de Baja California hoy solo cubre a UABC.
(Sinaloa, Chihuahua y Baja California Sur sí cubren a todas sus instituciones
DGESUI: Sinaloa incluye a Universidad Autónoma Indígena de México, Chihuahua
incluye a El Colegio de Chihuahua -- su única institución de apoyo solidario,
monto pequeño --, y Baja California Sur no tiene ninguna institución
intercultural ni de apoyo solidario registrada en ANUIES). (2) ESTATAL no
distingue nivel
(licenciatura vs. posgrado) — la fuente estatal reporta un solo monto por
universidad, todos los niveles mezclados — mientras que FEDERAL sí está
filtrado por nivel vía --posdoc. TOTAL, por lo tanto, siempre mezcla un
FEDERAL de un solo nivel con un ESTATAL de todos los niveles. (3) Sonora y
Chihuahua reportan DEVENGADO (gasto ejercido), pero Baja California solo
publica presupuesto ASIGNADO/aprobado (no hay archivo de Cuenta Pública con
devengado en su carpeta) — comparar el ESTATAL de Baja California contra el
de Sonora/Chihuahua mezcla dos conceptos presupuestales distintos.

Con --proyectar-estatal, --states rellena con una PROYECCIÓN el ESTATAL/TOTAL
de cualquier entidad cuyo --year pedido sea posterior al último año con dato
REAL en su fuente (hoy solo dispara para Sonora, cuyo egresos.xlsx llega solo
hasta 2023). La fórmula es aportación_real(último_año_real) ×
matrícula_entidad(year) / matrícula_entidad(último_año_real) — escala el
último dato real conocido por el cambio de matrícula del subsistema estatal
en esa entidad, asumiendo implícitamente que el costo estatal por alumno se
mantuvo constante desde entonces (no captura recortes/aumentos reales de
presupuesto, inflación, ni cambios de política estatal — es una extrapolación
mecánica, no una predicción informada). Nunca encadena una proyección sobre
otra: siempre parte del último año con dato REAL, nunca de un año ya
proyectado. Por default el flag está apagado y esos años muestran "s/d" como
siempre; con el flag, las celdas proyectadas se marcan con "*" en la tabla de
texto y con la columna `ES_PROYECTADO=True` en el parquet (--save).
Run: uv run python scripts/prepare_comparacion_universidades.py --states --posdoc

Con --dgesui, en vez de PEF+ANUIES, recrea la tabla por entidad usando
EXCLUSIVAMENTE los datos que expone el propio portal DGESUI
(data/dgesui/{year}/{SIGLA}_{year}.json, uno por universidad -- ver
scripts/download_dgesui_montos.py para descargarlos). Agrupa por entidad vía
UNIVERSIDADES_DGESUI_2025; las universidades sin archivo descargado para ese
año se omiten con un aviso, no truenan. FEDERAL/ESTATAL/TOTAL salen
directamente de "Monto Federal"/"Monto Estatal"/"Monto Público" de cada JSON,
y MATRICULA de "Matrícula Educación Superior Total" (licenciatura+TSU -- el
portal no desglosa por nivel, así que --posdoc se ignora en este modo).
Comparación hecha para 2025 (ver conversación): FEDERAL de DGESUI sale
sistemáticamente más alto que el de --states (nacional +13%, hasta +22% en
algunas entidades) -- consistente con que PEF es presupuesto APROBADO y
DGESUI reporta presupuesto EJERCIDO/ampliado. MATRICULA de DGESUI también más
alta (nacional +5%) -- consistente con un autorreporte más reciente que el
Anuario ANUIES publicado. La diferencia más importante es en ESTATAL: en
Sinaloa y Chihuahua, --states da una cifra 2.7-3.1x más alta que --dgesui,
porque el "Monto Estatal" de DGESUI es solo la contraparte FORMULAICA del
convenio de coparticipación (ver "Porcentaje de Participación del Estado" en
cada JSON, casi siempre una proporción fija tipo 2:1 o 1:1), mientras
--states (vía APORTACION_ESTATAL_CONFIG) captura el presupuesto estatal
COMPLETO de operación de la universidad desde su propia cuenta pública. En
Baja California ambas cifras salen casi idénticas -- son métricas distintas,
ninguna de las dos está "mal".
Run: uv run python scripts/prepare_comparacion_universidades.py --dgesui --year 2025
"""

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import polars as pl

PEF_DIR = Path("data/presupuesto_federacion/presupuesto/egresos_federacion")
ANUIES_DIR = Path("data/anuies")
ANUIES_GENERAL_DIR = ANUIES_DIR / "general"
ENTIDADES_DIR = Path("data/entidades_federativas")
DGESUI_DIR = Path("data/dgesui")
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
        "instituciones": [
            "UNIVERSIDAD AUTÓNOMA DE CHIHUAHUA",
            "UNIVERSIDAD AUTÓNOMA DE CD. JUÁREZ",
            "EL COLEGIO DE CHIHUAHUA",  # apoyo solidario, monto pequeño frente a las otras dos
        ],
        "corregir_unidad": lambda v: v,  # ya viene en pesos completos
    },
    "SINALOA": {
        "modo": "carpeta_por_año",  # un archivo distinto por año, adentro de ENTIDADES_DIR/sinaloa/{año}/
        "carpeta": ENTIDADES_DIR / "sinaloa",
        # "Anexo 10.csv" en 2024-2025, "anexo 10.csv" (minúscula) en 2023
        "glob_por_año": "*nexo 10.csv",
        "ancla_header": "Monto",
        "instituciones": [
            "Universidad Autónoma de Sinaloa",
            "Universidad Autónoma de Occidente",
            "Universidad Autónoma Indígena de México",
        ],
        "corregir_unidad": lambda v: v,  # ya viene en pesos completos
        "encoding": "iso-8859-1",  # CSV del portal de Sinaloa no viene en UTF-8
    },
    "BAJA CALIFORNIA": {
        "modo": "carpeta_por_año",  # un archivo distinto por año, adentro de ENTIDADES_DIR/baja_california/{año}/
        "carpeta": ENTIDADES_DIR / "baja_california",
        # el nombre del archivo cambia cada año (con/sin acento, may/minúscula,
        # con/sin año en el nombre): "Clasificacion Administrativa 2023.xlsx",
        # "Clasificación administrativa 2024.xlsx", "Clasificación administrativa.xlsx" (2025)
        "glob_por_año": "Clasificaci?n [Aa]dministrativ*.xlsx",
        "ancla_header": "Asignación Presupuestal",
        # a diferencia de Sonora/Chihuahua/Sinaloa, el nombre de la institución
        # va en la columna B (índice 1), no en la A
        "col_institucion": 1,
        # el nombre en el archivo lleva un prefijo de clave presupuestal
        # delante, idéntico en 2023-2025 -- se hardcodea completo en vez de
        # generalizar el emparejamiento a substring
        "instituciones": ["2.1.1.1.4.1 UNIVERSIDAD AUTONOMA DE BAJA CALIFORNIA"],
        "corregir_unidad": lambda v: v,  # ya viene en pesos completos
    },
    "BAJA CALIFORNIA SUR": {
        "modo": "carpeta_por_año",  # un archivo distinto por año, adentro de ENTIDADES_DIR/baja_california_sur/{año}/
        "carpeta": ENTIDADES_DIR / "baja_california_sur",
        "glob_por_año": "ANEXO I-15*.xlsx",
        # la UABCS aparece 3 veces en el archivo (estatal, federal Ramo 33, y
        # total combinado) -- se acota a la sección "II.- Capítulo 4000"
        # (Financiamiento Estatal) para no sumar las otras dos
        "desde_ancla": "II.- Capítulo 4000 Transferencias, Asignaciones, Subsidios y Otras Ayudas:",
        "hasta_ancla": "III.- Participaciones y Aportaciones:",
        "ancla_header": "MONTO",
        "instituciones": ["Universidad Autónoma de Baja California Sur"],
        "corregir_unidad": lambda v: v,  # ya viene en pesos completos
    },
}

# Nombre completo + entidad de las 78 universidades del programa "Subsidio
# Ordinario" DGESUI 2025 (https://dgesui.ses.sep.gob.mx/sep.subsidioentransparencia.mx/2025/subsidio-ordinario,
# ficha de cada universidad -- dirección física + gobernador/a en turno usados
# para resolver la entidad). Es una lista MÁS AMPLIA que SUBSISTEMA_ESTATALES
# de ANUIES: incluye Universidades Interculturales, "El Colegio de X", etc.,
# no solo las 35 "Universidades Públicas Estatales" que usa el resto del
# script (ej. Sonora tiene aquí 6 instituciones -- Colson, ITSON, UES,
# UNISIERRA, UNISON, UPY -- pero solo 2 caen en SUBSISTEMA_ESTATALES). Se usa
# como lista de verificación: solo la matrícula de instituciones que aparecen
# aquí (por nombre, ver _es_universidad_dgesui) entra al cálculo de
# MATRICULA -- así una institución que ANUIES reclasifique bajo
# SUBSISTEMA_ESTATALES/SUBSISTEMA_INTERCULTURALES pero que no esté confirmada
# como receptora del subsidio DGESUI no se cuela silenciosamente.
UNIVERSIDADES_DGESUI_2025 = {
    "BUAP": ("Benemérita Universidad Autónoma de Puebla", "Puebla"),
    "COLECH": ("El Colegio de Chihuahua", "Chihuahua"),
    "COLMOR": ("El Colegio de Morelos", "Morelos"),
    "Colson": ("El Colegio de Sonora", "Sonora"),
    "IC": ("Instituto Campechano", "Campeche"),
    "ITSON": ("Instituto Tecnológico de Sonora", "Sonora"),
    "UAA": ("Universidad Autónoma de Aguascalientes", "Aguascalientes"),
    "UABC": ("Universidad Autónoma de Baja California", "Baja California"),
    "UABCS": ("Universidad Autónoma de Baja California Sur", "Baja California Sur"),
    "UABJO": ("Universidad Autónoma Benito Juárez de Oaxaca", "Oaxaca"),
    "UACAM": ("Universidad Autónoma de Campeche", "Campeche"),
    "UACH": ("Universidad Autónoma de Chihuahua", "Chihuahua"),
    "UACJ": ("Universidad Autónoma de Ciudad Juárez", "Chihuahua"),
    "UACO": ("Universidad Autónoma Comunal de Oaxaca", "Oaxaca"),
    "UAdeC": ("Universidad Autónoma de Coahuila", "Coahuila"),
    "UAdeO": ("Universidad Autónoma de Occidente", "Sinaloa"),
    "UADY": ("Universidad Autónoma de Yucatán", "Yucatán"),
    "UAEH": ("Universidad Autónoma del Estado de Hidalgo", "Hidalgo"),
    "UAEMéx": ("Universidad Autónoma del Estado de México", "México"),
    "UAEM": ("Universidad Autónoma del Estado de Morelos", "Morelos"),
    "UAEQROO": ("Universidad Autónoma del Estado de Quintana Roo", "Quintana Roo"),
    "UAGro": ("Universidad Autónoma de Guerrero", "Guerrero"),
    "UAIM": ("Universidad Autónoma Indígena de México", "Sinaloa"),
    "UAN": ("Universidad Autónoma de Nayarit", "Nayarit"),
    "UANL": ("Universidad Autónoma de Nuevo León", "Nuevo León"),
    "UAQ": ("Universidad Autónoma de Querétaro", "Querétaro"),
    "UAS": ("Universidad Autónoma de Sinaloa", "Sinaloa"),
    "UASLP": ("Universidad Autónoma de San Luis Potosí", "San Luis Potosí"),
    "UAT": ("Universidad Autónoma de Tamaulipas", "Tamaulipas"),
    "UATx": ("Universidad Autónoma de Tlaxcala", "Tlaxcala"),
    "UAZ": ('Universidad Autónoma De Zacatecas "Francisco García Salinas"', "Zacatecas"),
    "UCEMICH": ("Universidad de la Ciénega del Estado de Michoacán de Ocampo", "Michoacán"),
    "UCOL": ("Universidad de Colima", "Colima"),
    "UdeG": ("Universidad de Guadalajara", "Jalisco"),
    "UES": ("Universidad Estatal de Sonora", "Sonora"),
    "UG": ("Universidad de Guanajuato", "Guanajuato"),
    "UIBC": ("Universidad Intercultural de Baja California", "Baja California"),
    "UIC": ("Universidad Intercultural de Colima", "Colima"),
    "UICAM": ("Universidad Intercultural de Campeche", "Campeche"),
    "UICEH": ("Universidad Intercultural del Estado de Hidalgo", "Hidalgo"),
    "UICH": ("Universidad Interserrana del Estado de Puebla-Chilchotla", "Puebla"),
    "UICSLP": ("Universidad Intercultural de San Luis Potosí", "San Luis Potosí"),
    "UIEG": ("Universidad Intercultural del Estado de Guerrero", "Guerrero"),
    "UIEM": ("Universidad Intercultural del Estado de México", "México"),
    "UIEP": ("Universidad Intercultural del Estado de Puebla", "Puebla"),
    "UIEPA": ("Universidad Interserrana del Estado de Puebla-Ahuacatlán", "Puebla"),
    "UIET": ("Universidad Intercultural del Estado de Tabasco", "Tabasco"),
    "UIG": ("Universidad Intercultural del Estado de Guanajuato", "Guanajuato"),
    "UIIM": ("Universidad Intercultural Indígena de Michoacán", "Michoacán"),
    "UIJ": ("Universidad Intercultural de Jalisco", "Jalisco"),
    "UIMQROO": ("Universidad Intercultural Maya de Quintana Roo", "Quintana Roo"),
    "UIP": ("Universidad Intercultural del Pueblo", "Oaxaca"),
    "UIT": ("Universidad Intercultural de Tlaxcala", "Tlaxcala"),
    "UJAT": ("Universidad Juárez Autónoma de Tabasco", "Tabasco"),
    "UJED": ("Universidad Juárez del Estado de Durango", "Durango"),
    "UMAR": ("Universidad del Mar", "Oaxaca"),
    "UMB": ("Universidad Mexiquense del Bicentenario", "México"),
    "UMSNH": ("Universidad Michoacana de San Nicolás de Hidalgo", "Michoacán"),
    "UNACAR": ("Universidad Autónoma del Carmen", "Campeche"),
    "UNACH": ("Universidad Autónoma de Chiapas", "Chiapas"),
    "UNCA": ("Universidad de la Cañada", "Oaxaca"),
    "UNEVE": ("Universidad Estatal del Valle de Ecatepec", "México"),
    "UNEVT": ("Universidad Estatal del Valle de Toluca", "México"),
    "UNICACH": ("Universidad Autónoma de Ciencias y Artes de Chiapas", "Chiapas"),
    "UNICARIBE": ("Universidad del Caribe", "Quintana Roo"),
    "UNICH": ("Universidad Intercultural de Chiapas", "Chiapas"),
    "UNISIERRA": ("Universidad de la Sierra", "Sonora"),
    "UNISON": ("Universidad de Sonora", "Sonora"),
    "UNISTMO": ("Universidad del Istmo", "Oaxaca"),
    "UNITI": ("Universidad Intercultural para la Igualdad", "Aguascalientes"),
    "UNO": ("Universidad de Oriente", "Yucatán"),
    "UNPA": ("Universidad del Papaloapan", "Oaxaca"),
    "UNSIJ": ("Universidad de la Sierra Juárez", "Oaxaca"),
    "UNSIS": ("Universidad de la Sierra Sur", "Oaxaca"),
    "UPCH": ("Universidad Popular de la Chontalpa", "Tabasco"),
    "UPY": ("Universidad del Pueblo Yaqui", "Sonora"),
    "UTM": ("Universidad Tecnológica de la Mixteca", "Oaxaca"),
    "UV": ("Universidad Veracruzana", "Veracruz"),
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
SUBSISTEMA_INTERCULTURALES = "UNIVERSIDADES INTERCULTURALES"
SUBSISTEMA_APOYO_SOLIDARIO = "UNIVERSIDADES PÚBLICAS ESTATALES DE APOYO SOLIDARIO"
# El PEF no desglosa el gasto por institución dentro de la UR DGESUI: la
# misma cifra financia a las universidades públicas estatales, interculturales,
# y "de apoyo solidario" de cada entidad por igual (las 78 de
# UNIVERSIDADES_DGESUI_2025 se reparten entre estos tres subsistemas ANUIES).
# La matrícula usada como denominador debe sumar los tres para medir el mismo
# universo que el numerador.
SUBSISTEMAS_DGESUI = {SUBSISTEMA_ESTATALES, SUBSISTEMA_INTERCULTURALES, SUBSISTEMA_APOYO_SOLIDARIO}

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


# Casos donde ANUIES usa un nombre distinto (más corto, o con espaciado de
# guion diferente) que el oficial DGESUI para la misma institución (misma
# entidad, sin ambigüedad) -- se mapean a mano en vez de intentar un
# emparejamiento difuso genérico.
_ALIAS_ANUIES_DGESUI = {
    "UNIVERSIDAD INTERCULTURAL DE GUANAJUATO",  # = DGESUI UIG, sin "del Estado de"
    "UNIVERSIDAD AUTONOMA DE ZACATECAS",  # = DGESUI UAZ, sin el honorífico "Francisco García Salinas"
    "UNIVERSIDAD DE CIENCIAS Y ARTES DE CHIAPAS",  # = DGESUI UNICACH, sin "Autónoma"
    "UNIVERSIDAD INTERSERRANA DEL ESTADO DE PUEBLA - AHUACATLAN",  # = DGESUI UIEPA, espacios alrededor del guion
    "UNIVERSIDAD INTERSERRANA DEL ESTADO DE PUEBLA - CHILCHOTLA",  # = DGESUI UICH, espacios alrededor del guion
}
_NOMBRES_DGESUI_NORM = {_normalizar_institucion(nombre) for nombre, _ in UNIVERSIDADES_DGESUI_2025.values()} | _ALIAS_ANUIES_DGESUI


def _es_universidad_dgesui(institucion: str) -> bool:
    """True si `institucion` (tal como aparece en INSTITUCIÓN de ANUIES)
    corresponde a alguna de las 78 universidades verificadas en
    UNIVERSIDADES_DGESUI_2025 -- para no incluir en MATRICULA ninguna
    institución que ANUIES clasifique bajo SUBSISTEMA_ESTATALES o
    SUBSISTEMA_INTERCULTURALES pero que no esté confirmada como receptora del
    subsidio DGESUI Subsidio Ordinario."""
    return _normalizar_institucion(institucion) in _NOMBRES_DGESUI_NORM


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
      csv con encoding variable — config["encoding"] — en Sinaloa).

    config["col_institucion"] (default 0, columna A) indica en qué columna de
    cada fila está el nombre de la institución -- en Baja California va en la
    columna B (índice 1), no A como en el resto de las entidades.

    config["desde_ancla"]/config["hasta_ancla"] (opcionales, solo
    "carpeta_por_año") acotan la búsqueda a las filas entre esas dos anclas
    (ambas deben ser el valor EXACTO de una celda) -- necesario cuando la
    misma institución aparece más de una vez en el archivo en secciones
    distintas (ej. Baja California Sur: la UABCS aparece con su aportación
    ESTATAL, con su aportación FEDERAL vía Ramo 33, y con el total combinado
    de ambas -- sin acotar, se sumarían las tres)."""
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
        if "desde_ancla" in config:
            try:
                i0 = next(i for i, f in enumerate(filas) if config["desde_ancla"] in f)
            except StopIteration:
                return None
            i1 = len(filas)
            if "hasta_ancla" in config:
                try:
                    i1 = next(i for i, f in enumerate(filas) if i > i0 and config["hasta_ancla"] in f)
                except StopIteration:
                    pass
            filas = filas[i0:i1]
        try:
            header = next(f for f in filas if config["ancla_header"] in f)
        except StopIteration:
            return None
        col_idx = header.index(config["ancla_header"])
    else:
        raise ValueError(f"modo desconocido en APORTACION_ESTATAL_CONFIG[{entidad!r}]: {modo!r}")

    col_institucion = config.get("col_institucion", 0)
    total = 0.0
    encontrado = False
    for fila in filas:
        if not isinstance(fila[col_institucion], str) or _normalizar_institucion(fila[col_institucion]) not in instituciones_norm:
            continue
        valor = _parsear_monto(fila[col_idx])
        if valor is None:
            continue
        total += config["corregir_unidad"](valor)
        encontrado = True
    return total if encontrado else None


def años_disponibles_aportacion_estatal(entidad: str) -> list[int]:
    """Años con dato REAL (no proyectado) de aportación estatal disponibles
    para esta entidad en su fuente, según config["modo"]. Lista vacía si la
    entidad no está mapeada o su fuente no existe."""
    config = APORTACION_ESTATAL_CONFIG.get(entidad)
    if config is None:
        return []
    modo = config.get("modo", "columnas_por_año")

    if modo == "columnas_por_año":
        ruta = config["ruta"]
        if not ruta.exists():
            return []
        filas = pl.read_excel(ruta, sheet_name=config["hoja"], has_header=False).rows()
        try:
            header = next(f for f in filas if f[0] == config["ancla_header"])
        except StopIteration:
            return []
        return sorted({int(float(a)) for a in header[1:] if a is not None})
    elif modo == "carpeta_por_año":
        if not config["carpeta"].exists():
            return []
        return sorted(
            int(sub.name)
            for sub in config["carpeta"].iterdir()
            if sub.is_dir() and sub.name.isdigit() and any(sub.glob(config["glob_por_año"]))
        )
    else:
        raise ValueError(f"modo desconocido en APORTACION_ESTATAL_CONFIG[{entidad!r}]: {modo!r}")


def _matricula_entidad_estatales(entidad: str, year: int, niveles: set[str]) -> int | None:
    """Matrícula DGESUI (subsistemas estatales + interculturales, ver
    SUBSISTEMAS_DGESUI, verificada contra UNIVERSIDADES_DGESUI_2025) para una
    sola entidad y año, usando el ciclo ANUIES cuyo año esperado
    (fin_de_ciclo + 1) coincida con `year` (mismo truco que correr_historico
    para mapear year -> lag). None si el año es muy reciente para tener
    ciclo, o si no hay matrícula para esa entidad/nivel."""
    _, _, año_esperado_lag0 = encontrar_anuies(0)
    lag = año_esperado_lag0 - year
    if lag < 0:
        return None
    try:
        anuies_path, _, _ = encontrar_anuies(lag)
    except (FileNotFoundError, ValueError):
        return None
    anuies = pl.read_excel(anuies_path, sheet_name="Base de datos")
    mat = anuies.filter(
        (pl.col("SUBSISTEMA").is_in(SUBSISTEMAS_DGESUI))
        & (pl.col("INSTITUCIÓN").map_elements(_es_universidad_dgesui, return_dtype=pl.Boolean))
        & (pl.col("NIVEL").is_in(niveles))
        & (pl.col("ENTIDAD") == entidad)
    )["Matrícula Total"].sum()
    return mat if mat else None


def proyectar_aportacion_estatal(entidad: str, year: int, niveles: set[str]) -> tuple[float | None, int | None]:
    """Proyecta ESTATAL para `year` cuando esa entidad no tiene dato real ahí,
    escalando el último año REAL disponible (siempre el último real, nunca un
    año ya proyectado, para no encadenar error) por el cambio de matrícula
    del subsistema estatal entre ese año base y `year`. Devuelve
    (valor_proyectado, año_base), o (None, None) si no hay año real anterior
    a `year` o falta cualquier insumo (aportación o matrícula base/actual)."""
    años_reales_previos = [a for a in años_disponibles_aportacion_estatal(entidad) if a < year]
    if not años_reales_previos:
        return None, None
    año_base = max(años_reales_previos)
    aportacion_base = cargar_aportacion_estatal(entidad, año_base)
    if aportacion_base is None:
        return None, None
    matricula_base = _matricula_entidad_estatales(entidad, año_base, niveles)
    matricula_actual = _matricula_entidad_estatales(entidad, year, niveles)
    if not matricula_base or not matricula_actual:
        return None, None
    return aportacion_base * matricula_actual / matricula_base, año_base


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
    matricula["ESTATALES"] = anuies.filter(
        (pl.col("SUBSISTEMA").is_in(SUBSISTEMAS_DGESUI))
        & (pl.col("INSTITUCIÓN").map_elements(_es_universidad_dgesui, return_dtype=pl.Boolean))
    )["Matrícula Total"].sum()
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
    filas.append(("ESTATALES", "Universidades del subsidio DGESUI (estatales + interculturales + apoyo solidario)", estatales_raw[col_monto].sum(), matricula["ESTATALES"]))

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


def calcular_tabla_entidades(
    year: int, lag: int, posdoc: bool, proyectar: bool = False
) -> tuple[pl.DataFrame, Path, Path, str, int, list[str]]:
    """Costo por alumno del subsidio DGESUI (universidades públicas estatales
    + interculturales, ver SUBSISTEMAS_DGESUI, verificadas contra
    UNIVERSIDADES_DGESUI_2025) desglosado por entidad federativa, para un
    solo nivel (licenciatura+TSU, o posgrado con posdoc=True). Devuelve
    (tabla, pef_path, anuies_path, ciclo, año_esperado, no_verificadas);
    tabla trae una fila TOTAL con las sumas nacionales. no_verificadas es la
    lista de instituciones que ANUIES clasifica bajo SUBSISTEMAS_DGESUI pero
    que no matchean ninguna de las 78 en UNIVERSIDADES_DGESUI_2025 (excluidas
    del cálculo; normalmente vacía -- ver aviso en main()).

    proyectar=True rellena ESTATAL/TOTAL con una proyección (ver
    proyectar_aportacion_estatal) para las entidades cuyo `year` sea
    posterior al último año con dato real -- marcadas con ES_PROYECTADO=True.
    Con proyectar=False (default), esas entidades quedan en None ("s/d")
    igual que antes de agregar este flag."""
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
    dgesui_pool = anuies.filter((pl.col("SUBSISTEMA").is_in(SUBSISTEMAS_DGESUI)) & (pl.col("NIVEL").is_in(niveles)))
    no_verificadas = sorted(
        dgesui_pool.filter(~pl.col("INSTITUCIÓN").map_elements(_es_universidad_dgesui, return_dtype=pl.Boolean))["INSTITUCIÓN"]
        .unique()
        .to_list()
    )
    estatales = dgesui_pool.filter(pl.col("INSTITUCIÓN").map_elements(_es_universidad_dgesui, return_dtype=pl.Boolean))
    mat_ent = estatales.group_by("ENTIDAD").agg(
        pl.sum("Matrícula Total").alias("MATRICULA"),
        pl.col("INSTITUCIÓN").n_unique().alias("N_INSTITUCIONES"),
        pl.col("ESCUELA/CAMPUS/PLANTEL").n_unique().alias("N_PLANTELES"),
    )
    if mat_ent["MATRICULA"].sum() == 0:
        raise ValueError(f"Matrícula 0 para --states en {anuies_path} — revisar SUBSISTEMAS_DGESUI/nivel para este ciclo.")

    tabla = mat_ent.join(por_ent_pef.select("ENTIDAD", "PRESUPUESTO"), on="ENTIDAD", how="left").fill_null(0)
    tabla = tabla.rename({"PRESUPUESTO": "FEDERAL"})
    tabla = tabla.with_columns(
        pl.col("ENTIDAD").map_elements(lambda e: cargar_aportacion_estatal(e, year), return_dtype=pl.Float64).alias("ESTATAL")
    )

    if proyectar:
        proyecciones = {}
        for ent in tabla.filter(pl.col("ESTATAL").is_null())["ENTIDAD"]:
            if ent not in APORTACION_ESTATAL_CONFIG:
                continue
            valor, _año_base = proyectar_aportacion_estatal(ent, year, niveles)
            if valor is not None:
                proyecciones[ent] = valor
        tabla = tabla.with_columns(
            pl.col("ENTIDAD").map_elements(lambda e: e in proyecciones, return_dtype=pl.Boolean).alias("ES_PROYECTADO"),
            pl.when(pl.col("ESTATAL").is_null())
              .then(pl.col("ENTIDAD").map_elements(lambda e: proyecciones.get(e), return_dtype=pl.Float64))
              .otherwise(pl.col("ESTATAL"))
              .alias("ESTATAL"),
        )
    else:
        tabla = tabla.with_columns(pl.lit(False).alias("ES_PROYECTADO"))

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
            "ES_PROYECTADO": [False],  # es un agregado, no un valor proyectado en sí mismo
        }
    ).with_columns(
        (pl.col("FEDERAL") / pl.col("MATRICULA")).alias("COSTO_ALUMNO_FEDERAL"),
        (pl.col("FEDERAL") + pl.col("ESTATAL")).alias("TOTAL"),
    )
    total = total.with_columns((pl.col("TOTAL") / pl.col("MATRICULA")).alias("COSTO_ALUMNO_TOTAL"))
    total = total.select([pl.col(c).cast(tabla.schema[c]) for c in tabla.columns])
    tabla = pl.concat([tabla, total])

    return tabla, pef_path, anuies_path, ciclo, año_esperado, no_verificadas


def calcular_tabla_dgesui(year: int) -> tuple[pl.DataFrame, Path, list[str]]:
    """Costo por alumno usando EXCLUSIVAMENTE los datos que expone el propio
    portal DGESUI (data/dgesui/{year}/{SIGLA}_{year}.json, ver
    scripts/download_dgesui_montos.py), agrupados por entidad vía
    UNIVERSIDADES_DGESUI_2025 -- sin tocar PEF ni ANUIES. Devuelve (tabla,
    directorio, faltantes); tabla trae una fila TOTAL con las sumas
    nacionales; faltantes es la lista de siglas de UNIVERSIDADES_DGESUI_2025
    sin archivo descargado para ese año (se omiten del cálculo, no truenan).

    ESTATAL aquí es el "Monto Estatal" que reporta el propio DGESUI, que en
    la práctica suele ser solo la contraparte FORMULAICA del convenio de
    coparticipación (ver "Porcentaje de Participación del Estado" en cada
    JSON) -- no el presupuesto estatal completo de la universidad, que sí
    mide --states vía APORTACION_ESTATAL_CONFIG. Pueden diferir 2-3x en
    entidades con universidades grandes (comparación hecha para 2025:
    Sinaloa y Chihuahua salen ~2.7-3.1x más altas en --states que en
    --dgesui; Baja California prácticamente idéntico). MATRICULA es
    "Matrícula Educación Superior Total" (alcance licenciatura+TSU) -- el
    portal no desglosa por nivel, así que --posdoc no aplica en este modo."""
    directorio = DGESUI_DIR / str(year)
    if not directorio.exists():
        raise FileNotFoundError(
            f"No se encontraron datos DGESUI para {year} en {directorio} -- corre primero: "
            f"uv run python scripts/download_dgesui_montos.py --year {year} --output {directorio}"
        )

    filas = []
    faltantes = []
    for sigla, (_nombre, entidad) in UNIVERSIDADES_DGESUI_2025.items():
        ruta = directorio / f"{sigla}_{year}.json"
        if not ruta.exists():
            faltantes.append(sigla)
            continue
        data = json.loads(ruta.read_text(encoding="utf-8"))
        montos = data.get("Montos", {})
        numeralia = data.get("Numeralia", {})
        federal = _parsear_monto(montos.get("Monto Federal", {}).get("Número"))
        estatal = _parsear_monto(montos.get("Monto Estatal", {}).get("Número"))
        clave_matricula = next((k for k in numeralia if k.startswith("Matrícula Educación Superior Total")), None)
        matricula = _parsear_monto(numeralia.get(clave_matricula)) if clave_matricula else None
        filas.append((sigla, entidad.upper(), federal, estatal, int(matricula) if matricula is not None else None))

    if not filas:
        raise ValueError(f"Ningún archivo DGESUI encontrado en {directorio} -- revisa que el año {year} se haya descargado.")

    df = pl.DataFrame(filas, schema=["SIGLA", "ENTIDAD", "FEDERAL", "ESTATAL", "MATRICULA"], orient="row")
    tabla = df.group_by("ENTIDAD").agg(
        pl.sum("FEDERAL").alias("FEDERAL"),
        pl.sum("ESTATAL").alias("ESTATAL"),
        pl.sum("MATRICULA").alias("MATRICULA"),
        pl.len().alias("N_INSTITUCIONES"),
    )
    tabla = tabla.with_columns(
        (pl.col("FEDERAL") + pl.col("ESTATAL")).alias("TOTAL"),
        (pl.col("FEDERAL") / pl.col("MATRICULA")).alias("COSTO_ALUMNO_FEDERAL"),
    )
    tabla = tabla.with_columns((pl.col("TOTAL") / pl.col("MATRICULA")).alias("COSTO_ALUMNO_TOTAL"))
    tabla = tabla.sort("COSTO_ALUMNO_FEDERAL", descending=True)

    total = pl.DataFrame(
        {
            "ENTIDAD": ["TOTAL"],
            "MATRICULA": [tabla["MATRICULA"].sum()],
            "N_INSTITUCIONES": [tabla["N_INSTITUCIONES"].sum()],
            "FEDERAL": [tabla["FEDERAL"].sum()],
            "ESTATAL": [tabla["ESTATAL"].sum()],
        }
    ).with_columns(
        (pl.col("FEDERAL") + pl.col("ESTATAL")).alias("TOTAL"),
        (pl.col("FEDERAL") / pl.col("MATRICULA")).alias("COSTO_ALUMNO_FEDERAL"),
    )
    total = total.with_columns((pl.col("TOTAL") / pl.col("MATRICULA")).alias("COSTO_ALUMNO_TOTAL"))
    total = total.select([pl.col(c).cast(tabla.schema[c]) for c in tabla.columns])
    tabla = pl.concat([tabla, total])

    return tabla, directorio, faltantes


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
    parser.add_argument("--proyectar-estatal", dest="proyectar_estatal", action="store_true", help="Con --states: si --year es posterior al último año con dato real de aportación estatal de una entidad, proyecta el valor escalando el último año real por el cambio de matrícula (asume costo estatal por alumno constante). Por default no se proyecta -- se muestra 's/d'. Las celdas proyectadas se marcan con '*'.")
    parser.add_argument("--dgesui", action="store_true", help="Recrea la tabla por entidad usando EXCLUSIVAMENTE datos del portal DGESUI (data/dgesui/{year}/, ver scripts/download_dgesui_montos.py) en vez de PEF+ANUIES. FEDERAL/ESTATAL miden algo distinto a --states -- ver docstring de calcular_tabla_dgesui. No compatible con --historico ni --states; ignora --posdoc (el portal no desglosa matrícula por nivel).")
    args = parser.parse_args()

    if args.historico and args.states:
        parser.error("--historico y --states no se pueden combinar")
    if args.dgesui and (args.historico or args.states):
        parser.error("--dgesui no se puede combinar con --historico ni --states")

    if args.historico:
        correr_historico(args.historico, args.save, args.posdoc)
        return

    nombre_nivel, _, _ = nivel_activo(args.posdoc)
    year = args.year

    if args.dgesui:
        if args.posdoc:
            print("[aviso] --posdoc se ignora en --dgesui: el portal no desglosa matrícula por nivel (Numeralia solo trae 'Matrícula Educación Superior Total', alcance licenciatura+TSU).")
        tabla, directorio, faltantes = calcular_tabla_dgesui(year)

        print(f"Fuente: {directorio}/ (portal DGESUI, ver scripts/download_dgesui_montos.py)")
        if faltantes:
            print(f"[aviso] Sin archivo descargado para {year}, excluidas del cálculo: {faltantes}")
        print("[aviso] ESTATAL aquí es el 'Monto Estatal' que reporta el propio DGESUI -- en la práctica suele ser solo la contraparte FORMULAICA del convenio de coparticipación, no el presupuesto estatal completo de la universidad (que sí mide --states vía APORTACION_ESTATAL_CONFIG; pueden diferir 2-3x en entidades con universidades grandes).")
        print()

        print(f"{'Entidad':<20} {'Insts':>6} {'Matrícula':>10} {'Federal':>15} {'Estatal':>15} {'Total':>15} {'Costo/al.Fed':>13} {'Costo/al.Total':>15}")
        for r in tabla.iter_rows(named=True):
            print(
                f"{r['ENTIDAD']:<20} {r['N_INSTITUCIONES']:>6,} {r['MATRICULA']:>10,} "
                f"{r['FEDERAL']:>15,.0f} {r['ESTATAL']:>15,.0f} {r['TOTAL']:>15,.0f} {r['COSTO_ALUMNO_FEDERAL']:>13,.0f} "
                f"{r['COSTO_ALUMNO_TOTAL']:>15,.0f}"
            )

        if args.save:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            out_path = OUT_DIR / f"comparacion_universidades_dgesui_{year}.parquet"
            tabla.write_parquet(out_path)
            print(f"\nGuardado → {out_path}")
        else:
            print("\n(no guardado — pasa --save para escribir el parquet)")
        return

    if args.states:
        tabla, pef_path, anuies_path, ciclo, año_esperado, no_verificadas = calcular_tabla_entidades(
            year, args.lag, args.posdoc, proyectar=args.proyectar_estatal
        )

        print(f"Fuente presupuesto: {pef_path}")
        print(f"Fuente matrícula: {anuies_path} (ciclo {ciclo})")
        print(f"Nivel: {nombre_nivel}")
        if year != año_esperado:
            print(f"[aviso] --year {year} no coincide con el año esperado para el ciclo {ciclo} ({año_esperado}) — la comparación mezcla presupuesto y matrícula de años distintos.")
        if no_verificadas:
            print(f"[aviso] Excluidas de MATRICULA por no estar en UNIVERSIDADES_DGESUI_2025 (revisar si deben agregarse): {no_verificadas}")
        con_dato = sorted(APORTACION_ESTATAL_CONFIG.keys())
        print(f"[aviso] Aportación estatal (ESTATAL/TOTAL) solo disponible para: {con_dato} — el resto muestra 's/d'. La fila TOTAL nacional de ESTATAL/TOTAL es parcial (solo suma las entidades con dato).")
        print("[aviso] ESTATAL no distingue nivel (licenciatura/posgrado) — la fuente estatal reporta un solo monto por universidad, mientras FEDERAL sí está filtrado por --posdoc. TOTAL mezcla ambos criterios.")
        if args.proyectar_estatal:
            print("[aviso] --proyectar-estatal activo: las celdas marcadas con '*' son una PROYECCIÓN (no dato real), calculada escalando el último año real por el cambio de matrícula -- asume costo estatal por alumno constante desde entonces.")
        print()

        def fmt(v, proyectado=False):
            if v is None:
                return f"{'s/d':>15}"
            return f"{f'{v:,.0f}' + ('*' if proyectado else ''):>15}"

        print(f"{'Entidad':<20} {'Insts':>6} {'Planteles':>10} {'Matrícula':>10} {'Federal':>15} {'Estatal':>15} {'Total':>15} {'Costo/al.Fed':>13} {'Costo/al.Total':>15}")
        for r in tabla.iter_rows(named=True):
            print(
                f"{r['ENTIDAD']:<20} {r['N_INSTITUCIONES']:>6,} {r['N_PLANTELES']:>10,} {r['MATRICULA']:>10,} "
                f"{r['FEDERAL']:>15,.0f} {fmt(r['ESTATAL'], r['ES_PROYECTADO'])} {fmt(r['TOTAL'], r['ES_PROYECTADO'])} {r['COSTO_ALUMNO_FEDERAL']:>13,.0f} "
                f"{fmt(r['COSTO_ALUMNO_TOTAL'], r['ES_PROYECTADO'])}"
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
