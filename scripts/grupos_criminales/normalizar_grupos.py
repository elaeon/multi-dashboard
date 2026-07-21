"""Normalización de nombres de grupos criminales a un vocabulario canónico.

Las tres fuentes con nombre de grupo (OCVED, El Universal, Animal Político) no
comparten vocabulario. `DATA_HAWK_INDEX.md` prohíbe el match automático por
string y exige un crosswalk curado.

Este módulo implementa ese crosswalk como **detección por patrón sobre la cadena
completa**, no como split por separadores. Razón: en `universal/narco` el campo
`Cartel` mezcla al menos cinco separadores (`,`, ` y `, ` vs `, `/`, `;`) y
`y/o` NO es conjunción sino **alias** — `"Cártel de Sinaloa y/o Pacífico"` es un
solo grupo con dos nombres, igual que `"Nuevo Cártel de Juárez y/o La Línea"`.
Un split los habría contado como dos. Además `", facción X"` modifica al grupo
anterior y `"(bandas delictivas)"` es anotación editorial, no un actor.

Buscar patrones y deduplicar resuelve los tres casos sin reglas de separador.
"""

import re
import unicodedata

# Vocabulario canónico. El orden importa: los patrones más específicos van
# primero para que "Nuevo Cártel de Juárez" no caiga en el patrón genérico de
# Juárez antes de tiempo (aquí ambos mapean igual, pero la regla se sostiene
# si el vocabulario crece).
#
# Cada entrada: grupo_canonico -> lista de regex (sobre texto ya normalizado,
# minúsculas sin acentos).
PATRONES = {
    "CJNG": [r"\bcjng\b", r"jalisco nueva generacion"],
    "Cartel de Sinaloa": [
        r"cartel de sinaloa",
        r"cartel del pacifico",
        r"\bpacifico\b",
        r"mayo zambada",
        r"los menores",
        r"los antrax",
    ],
    "Cartel del Golfo": [r"cartel del golfo", r"cartel el golfo", r"\bc\.?d\.?g\.?\b"],
    "Los Zetas": [r"\bzetas?\b", r"\bzve\b", r"sangre nueva zeta"],
    "Cartel del Noreste": [r"cartel del noreste", r"\bcdn\b"],
    "Beltran Leyva": [r"beltran leyva"],
    "La Familia Michoacana": [r"familia michoacana", r"familia michoacama", r"\bla familia\b"],
    "Caballeros Templarios": [r"caballeros templarios"],
    "Cartel de Juarez": [r"cartel de juarez", r"la linea", r"carillo fuentes", r"carrillo fuentes"],
    "Cartel de Tijuana": [r"arellano felix", r"cartel de tijuana"],
    "Cartel Santa Rosa de Lima": [r"santa rosa de lima", r"\bcsrl\b"],
    "Union Tepito": [r"union tepito"],
    "Fuerza Anti Union": [r"fuerza anti ?union"],
    "Guerreros Unidos": [r"guerreros unidos"],
    "Los Rojos": [r"\blos rojos\b", r"\bc\.?d\.?g\.? - rojos\b"],
    "Los Viagras": [r"\blos viagras\b"],
    "Los Ardillos": [r"\blos ardillos\b"],
    "Carteles Unidos": [r"carteles unidos"],
    "Cartel de Tlahuac": [r"cartel de tlahuac"],
    "CSLPNG": [r"\bcslpng\b", r"san luis potosi nueva generacion"],
    "Los Talibanes": [r"\btalibanes\b"],
    "Gente Nueva": [r"gente nueva"],
    "Los Rodolfos": [r"\blos rodolfos?\b"],
    "Mara Salvatrucha": [r"maras? salvatrucha"],
    "Huachicoleros": [r"\bhuachicoler"],
    "La Barbie": [r"\bla barbie\b"],
    # Bandas locales recurrentes. No son cárteles nacionales, pero son grupos
    # nombrados con presencia municipal repetida, así que merecen entrada propia
    # en vez de caer en el saco de "no catalogado". Incluyen los typos de la
    # fuente: "Granadso" -> Granados, "Los correa" -> Los Correa.
    "Los Sierra": [r"\blos sierra\b"],
    "Los Granados": [r"\blos granados\b", r"\blos granadso\b", r"^granados$"],
    "Los Rojas Romero": [r"\bl[ao]s roja?s romero\b"],
    "Los Trinis": [r"\blos trinis\b"],
    "Los Tellez": [r"\blos tellez\b", r"el loco tellez"],
    "Los Yglesias": [r"\blos yglesias\b"],
    "Los Correa": [r"\blos correa\b"],
    "Los Hades": [r"\blos hades\b"],
    "Columna Armada Pedro J. Mendez": [r"columna armada pedro"],
    "Cartel de la Sierra": [r"cartel de la sierra"],
    "Cartel Unido de la Huasteca": [r"cartel unido de la huasteca"],
    "Guardia Guerrerense": [r"guardia guerrerense"],
    "Los Molina": [r"\blos molina\b"],
    "Nueva Plaza": [r"\bnueva plaza\b"],
}

# Etiquetas que NO son un grupo: buckets de "desconocido" y anotaciones
# editoriales. Se preservan como categoría propia, nunca como actor nombrado.
SENTINELAS = {
    "sin registro": "SIN_REGISTRO",
    "no identificado": "NO_IDENTIFICADO",
    "independiente": "NO_IDENTIFICADO",
    "unidentified criminal group": "NO_IDENTIFICADO",
    "other": "OTRO",
}


def _norm(texto: str) -> str:
    """Minúsculas sin acentos, para que los patrones no dependan de tildes."""
    sin_acentos = "".join(
        c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"\s+", " ", sin_acentos.lower()).strip()


def detectar_grupos(texto: str) -> list[str]:
    """Cadena cruda -> lista ordenada de grupos canónicos detectados.

    Cuatro resultados posibles, deliberadamente distintos entre sí:
      - `["SIN_REGISTRO"]`  — la fuente no registró nada (≠ "no hay grupo")
      - `["NO_IDENTIFICADO"]` — hay actividad, sin actor identificado
      - `["BANDA_LOCAL"]`   — hay un grupo nombrado, pero es una banda local o
        un individuo sin equivalente en el vocabulario canónico (p. ej.
        "Don José", "El América"). No se fuerza a un cártel nacional.
      - lista de grupos canónicos
    """
    if texto is None:
        return []
    t = _norm(texto)

    if t in SENTINELAS:
        return [SENTINELAS[t]]

    encontrados = []
    for canonico, patrones in PATRONES.items():
        if any(re.search(p, t) for p in patrones):
            encontrados.append(canonico)
    return sorted(set(encontrados)) if encontrados else ["BANDA_LOCAL"]
