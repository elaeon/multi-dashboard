#!/usr/bin/env python3
"""Normalize the CGPE *gasto programable* CSVs into one clean, tagged, unit-consistent file.

Reads ``data/cgpe/gasto_programable_YYYY.csv`` (2012-2026) and writes to
``data/cgpe_normalized/``:

* ``gasto_programable_normalizado.csv`` — all 15 years stacked, with extraction
  junk removed, ramo names canonicalized, every row tagged by ``clasificacion`` / ``grupo``
  / ``tipo`` (replacing the unreliable source ``nivel``), and all money expressed in
  ``millones_pesos`` (2012-2023 were in *miles de millones* and are rescaled x1000).
* ``ramo_mapping.csv`` — every distinct ``(año, ramo_original) -> ramo, clasificacion,
  grupo, tipo`` for review of the canonicalization and tagging.

The raw CSVs and the extractor are left untouched; this is post-hoc cleanup of the CSVs
as they exist.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import polars as pl

DATA_DIR = Path("data/cgpe")
OUT_DIR = Path("data/cgpe_normalized")
YEARS = list(range(2012, 2027))  # 2017 included (clasificación administrativa, pág. 81)

# Source schema (10 cols). Money cols are rescaled; pct cols are left as-is.
MONEY_COLS = ["ppef_anterior", "pef_anterior", "proyectado", "var_abs_ppef", "var_abs_pef"]
PCT_COLS = ["var_real_ppef_pct", "var_real_pef_pct"]

# --------------------------------------------------------------------------- #
# Junk / footnote-row detection
# --------------------------------------------------------------------------- #
# Footnote paragraphs the PDF extractor captured as data rows. They either start with
# one of these markers, or (in 2020/2021) carry extra fields beyond the 10-col schema.
JUNK_PREFIXES = (
    "p= Proyecto",
    "a= Aprobado",
    "En términos",
    "CRITERIOS GENERALES",
    "PERSPECTIVAS",
    "productivas De acuerdo",
    "directo y empresas productivas",
    "Para las empresas",
)


def is_junk(label: str, nfields: int) -> bool:
    return nfields > 10 or label.startswith(JUNK_PREFIXES)


# --------------------------------------------------------------------------- #
# Label cleaning (produces ramo_original) — strips OCR artifacts, not real text
# --------------------------------------------------------------------------- #
def clean_label(label: str) -> str:
    s = label.strip()
    s = re.sub(r"\s+p= Proyecto\.?\s*$", "", s)  # OCR suffix on real rows
    s = re.sub(r"\s+a= Aprobado\.?\s*$", "", s)
    s = re.sub(r"\s+\d{1,3}$", "", s)             # trailing footnote digit, e.g. " 62"
    s = re.sub(r"\s+", " ", s).strip()
    return s


# --------------------------------------------------------------------------- #
# Canonical name map: cleaned source label -> canonical label.
# Covers abbreviations, capitalization, mis-split/truncated names, and genuine
# institutional renames (mapped to the latest official name). ramo_original keeps the
# source label so every mapping stays auditable/reversible.
# --------------------------------------------------------------------------- #
RAMO_MAP = {
    # --- Ramos Autónomos -------------------------------------------------- #
    "Legislativo": "Poder Legislativo",
    "Judicial": "Poder Judicial",
    "Instituto Federal Electoral": "Instituto Nacional Electoral",
    "CNDH": "Comisión Nacional de los Derechos Humanos",
    "Comisión Nacional de los Derechos": "Comisión Nacional de los Derechos Humanos",
    "Comisión Federal de Competencia": "Comisión Federal de Competencia Económica",
    "Institóuto Nacional para la Evaluación de la Educación": "Instituto Nacional para la Evaluación de la Educación",
    "INAI": "Instituto Nacional de Transparencia, Acceso a la Información y Protección de Datos Personales",
    "Instituto Nacional de Transparencia,": "Instituto Nacional de Transparencia, Acceso a la Información y Protección de Datos Personales",
    "Inst. Nal. de Transparencia, Acceso a la Info. y Protección de Datos Personales": "Instituto Nacional de Transparencia, Acceso a la Información y Protección de Datos Personales",
    "Instituto Federal de Acceso a la Información y Protección de Datos": "Instituto Nacional de Transparencia, Acceso a la Información y Protección de Datos Personales",
    "PGR": "Fiscalía General de la República",
    "Procuraduría General de la República": "Fiscalía General de la República",
    "Acceso a la Información y Protección de Datos Personales Fiscalía General de la República": "Fiscalía General de la República",
    "INEGI": "Información Nacional Estadística y Geográfica",
    "Acceso a la Información y Protección de Datos Personales Información Nacional Estadística y Geográfica": "Información Nacional Estadística y Geográfica",
    "TFJFA": "Tribunal Federal de Justicia Administrativa",
    "Tribunal Federal de Justicia Fiscal y Administrativa": "Tribunal Federal de Justicia Administrativa",
    "Tribunal Federal de Justicia": "Tribunal Federal de Justicia Administrativa",
    "Poderes y Entes Autónomos": "Ramos Autónomos",
    "Ramos administrativos": "Ramos Administrativos",
    # --- Ramos Administrativos ------------------------------------------- #
    "Presidencia de la República": "Oficina de la Presidencia de la República",
    "Sagarpa": "Agricultura y Desarrollo Rural",
    "Agricultura, Ganadería, Desarrollo Rural, Pesca y Alimentación": "Agricultura y Desarrollo Rural",
    "Agricultura, Ganadería, Desarrollo Rural,": "Agricultura y Desarrollo Rural",
    "Comunicaciones y Transportes": "Infraestructura, Comunicaciones y Transportes",
    "Pesca y Alimentación Comunicaciones y Transportes": "Infraestructura, Comunicaciones y Transportes",
    "Reforma Agraria": "Desarrollo Agrario, Territorial y Urbano",
    "Semarnat": "Medio Ambiente y Recursos Naturales",
    "Función Pública": "Anticorrupción y Buen Gobierno",
    "Seguridad Pública": "Seguridad y Protección Ciudadana",
    "Consejería Jurídica": "Consejería Jurídica del Ejecutivo Federal",
    "CONACYT": "Ciencia, Humanidades, Tecnología e Innovación",
    "Consejo Nacional de Ciencia y Tecnología": "Ciencia, Humanidades, Tecnología e Innovación",
    "Consejo Nacional de Ciencia y Tecnología e Innovación": "Ciencia, Humanidades, Tecnología e Innovación",
    "Agencia de Transf. Digital y Telecomunic.": "Agencia de Transformación Digital y Telecomunicaciones",
    # --- Ramos Generales / entidades ------------------------------------- #
    "Entidades Control Directo": "Entidades de Control Directo",
    "IMSS": "Instituto Mexicano del Seguro Social",
    "ISSSTE": "Instituto de Seguridad y Servicios Sociales de los Trabajadores del Estado",
    "Instituto de Seguridad y Servicios Sociales": "Instituto de Seguridad y Servicios Sociales de los Trabajadores del Estado",
    "Instituto de Seguridad y Servicios": "Instituto de Seguridad y Servicios Sociales de los Trabajadores del Estado",
    "Pemex": "Petróleos Mexicanos",
    "CFE": "Comisión Federal de Electricidad",
    "Empresas productivas del Estado": "Empresas Productivas del Estado",
    "Empresas Públicas del Estado": "Empresas Productivas del Estado",
    "de los Trabajadores del Estado Empresas Productivas del Estado": "Empresas Productivas del Estado",
    "Sociales de los Trabajadores del Estado Empresas Productivas del Estado": "Empresas Productivas del Estado",
    "(-)Aportaciones, subsidios y transferencias": "Aportaciones, subsidios y transferencias",
    "Aportaciones ISSSTE, subsidios,": "Aportaciones, subsidios y transferencias",
    "Aportaciones ISSSTE, subsidios, transferencias y apoyos fiscales a entidades de control": "Aportaciones, subsidios y transferencias",
    # --- económica ------------------------------------------------------- #
    "Gasto Corriente": "Gasto corriente",
    "Gasto de Capital": "Gasto de capital",
    "Gasto de Inversión": "Gasto de capital",
    "Gastos de inversión": "Gasto de capital",
    "Inversión Física": "Inversión física",
    "Inversión financiera y otros": "Inversión financiera",
    "Inversión Financiera y Otros": "Inversión financiera",
    # --- funcional ------------------------------------------------------- #
    "Poderes, Entes Autónomos, INEG y TFJA": "Poderes, órganos autónomos, INEGI y TFJFA",
    "autónomos, Inegi y TFJFA": "Poderes, órganos autónomos, INEGI y TFJFA",
}


def canonical(clean: str, clasif: str | None) -> str:
    # "Desarrollo Social" is a ministry (administrativa, -> Bienestar) AND a macro-function
    # (funcional, kept as-is); disambiguate by section.
    if clean == "Desarrollo Social":
        return "Bienestar" if clasif == "administrativa" else "Desarrollo Social"
    return RAMO_MAP.get(clean, clean)


# --------------------------------------------------------------------------- #
# Section / group / type tagging (anchor-driven, order within a file is reliable)
# --------------------------------------------------------------------------- #
ECON_CORRIENTE = {"Gasto corriente", "Gasto Corriente"}
ECON_CAPITAL = {"Gasto de capital", "Gasto de Capital", "Gasto de Inversión", "Gastos de inversión"}
ADMIN_AUTONOMOS = {"Ramos Autónomos", "Ramos autónomos", "Poderes y Entes Autónomos"}
ADMIN_ADMIN = {"Ramos Administrativos", "Ramos administrativos"}
ADMIN_GENERALES = {"Ramos Generales"}
FUNC_ANCHORS = {
    "Administración Pública Federal",
    "Poderes, órganos autónomos, INEGI y TFJFA",
    "Poderes, Entes Autónomos, INEG y TFJA",
    "autónomos, Inegi y TFJFA",
}
FED_LABELS = {"Participaciones", "Aportaciones", "Otros Conceptos"}
# Sub-subtotals that don't switch section/grupo but should be flagged as subtotals.
SUBTOTAL_EXTRA = {
    "Organismos y empresas",
    "Entidades de Control Directo",
    "Entidades Control Directo",
    "Empresas Productivas del Estado",
    "Empresas Públicas del Estado",
}


def parse_file(path: Path) -> list[dict]:
    """Read one yearly CSV, drop junk rows, return cleaned (uncanonicalized) records."""
    out: list[dict] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)  # header
        for row in reader:
            if not row:
                continue
            label = row[0].strip()
            if is_junk(label, len(row)):
                continue

            def num(i: int) -> float | None:
                try:
                    return float(row[i])
                except (ValueError, IndexError):
                    return None

            out.append(
                {
                    "clean": clean_label(label),
                    "ppef_anterior": num(2),
                    "pef_anterior": num(3),
                    "proyectado": num(4),
                    "var_abs_ppef": num(5),
                    "var_abs_pef": num(6),
                    "var_real_ppef_pct": num(7),
                    "var_real_pef_pct": num(8),
                }
            )
    return out


def tag_file(rows: list[dict]) -> list[dict]:
    """Assign clasificacion / grupo / tipo to each row of one file (in order)."""
    section: str | None = None
    grupo: str | None = None
    recs: list[dict] = []
    for d in rows:
        c = d["clean"]
        if c == "Total":
            recs.append({**d, "clasificacion": None, "grupo": None, "tipo": "total"})
            continue

        tipo = "detalle"
        if c in ECON_CORRIENTE:
            section, grupo, tipo = "economica", "Corriente", "subtotal"
        elif c in ECON_CAPITAL:
            section, grupo, tipo = "economica", "Capital", "subtotal"
        elif c in ADMIN_AUTONOMOS:
            section, grupo, tipo = "administrativa", "Autónomos", "subtotal"
        elif c in ADMIN_ADMIN:
            section, grupo, tipo = "administrativa", "Administrativos", "subtotal"
        elif c in ADMIN_GENERALES:
            section, grupo, tipo = "administrativa", "Generales", "subtotal"
        elif c in FUNC_ANCHORS:
            section, grupo, tipo = "funcional", None, "subtotal"
        elif c in FED_LABELS:
            section, grupo, tipo = "federalizado", None, "detalle"
        elif c in SUBTOTAL_EXTRA:
            tipo = "subtotal"

        recs.append({**d, "clasificacion": section, "grupo": grupo, "tipo": tipo})

    # The grand Total heads the whole table; tag it administrativa (or federalizado for the
    # 2021 stray gasto-federalizado mini-table that precedes everything).
    first_section = next((r["clasificacion"] for r in recs if r["tipo"] != "total"), "administrativa")
    total_clasif = "federalizado" if first_section == "federalizado" else "administrativa"
    for r in recs:
        if r["tipo"] == "total":
            r["clasificacion"] = total_clasif
    return recs


# --------------------------------------------------------------------------- #
# Build the combined frame
# --------------------------------------------------------------------------- #
def build() -> tuple[pl.DataFrame, dict[int, int]]:
    records: list[dict] = []
    dropped: dict[int, int] = {}
    for year in YEARS:
        path = DATA_DIR / f"gasto_programable_{year}.csv"
        raw_rows = parse_file(path)
        with path.open(newline="", encoding="utf-8") as fh:
            n_in = sum(1 for _ in fh) - 1  # minus header
        dropped[year] = n_in - len(raw_rows)

        factor = 1000.0 if year <= 2023 else 1.0
        for r in tag_file(raw_rows):
            ramo_original = r["clean"]
            ramo = canonical(ramo_original, r["clasificacion"])
            rec = {
                "año": year,
                "clasificacion": r["clasificacion"],
                "grupo": r["grupo"],
                "tipo": r["tipo"],
                "ramo": ramo,
                "ramo_original": ramo_original,
            }
            for col in MONEY_COLS:
                v = r[col]
                rec[col] = round(v * factor, 1) if v is not None else None
            for col in PCT_COLS:
                rec[col] = r[col]
            rec["unidad"] = "millones_pesos"
            records.append(rec)

    df = pl.DataFrame(records)
    return df, dropped


# --------------------------------------------------------------------------- #
# Validation report
# --------------------------------------------------------------------------- #
def validate(df: pl.DataFrame, dropped: dict[int, int]) -> None:
    print("\n=== Drop counts (junk rows removed per year) ===")
    for year in YEARS:
        n = df.filter(pl.col("año") == year).height
        print(f"  {year}: kept {n:>3}  dropped {dropped[year]}")

    years_present = sorted(df["año"].unique().to_list())
    print(f"\nYears present: {years_present}")
    assert years_present == list(range(2012, 2027)), "expected 15 years 2012-2026"

    print("\n=== Tag sanity ===")
    assert df["clasificacion"].null_count() == 0, "clasificacion has nulls"
    assert set(df["clasificacion"].unique()) <= {
        "economica", "administrativa", "funcional", "federalizado",
    }, "unexpected clasificacion value"
    bad_grupo = df.filter(
        pl.col("clasificacion").is_in(["economica", "administrativa"])
        & (pl.col("tipo") != "total")
        & pl.col("grupo").is_null()
    )
    assert bad_grupo.height == 0, f"missing grupo on {bad_grupo.height} econ/admin detail rows"
    assert set(df["tipo"].unique()) == {"total", "subtotal", "detalle"}, "unexpected tipo"

    print("  clasificacion:", df["clasificacion"].value_counts().sort("clasificacion").to_dicts())
    print("  tipo:", df["tipo"].value_counts().sort("tipo").to_dicts())

    print("\n=== No junk text leaked into names ===")
    leaked = df.filter(
        pl.col("ramo").str.contains("Proyecto|Aprobado|por ciento|clasificación")
        | pl.col("ramo_original").str.contains("Proyecto|Aprobado|por ciento|clasificación")
    )
    assert leaked.height == 0, f"junk text in {leaked.height} names:\n{leaked['ramo'].to_list()}"
    print("  ok")

    print("\n=== Duplicate (año, clasificacion, grupo, ramo) ===")
    dups = (
        df.group_by(["año", "clasificacion", "grupo", "ramo"])
        .len()
        .filter(pl.col("len") > 1)
        .sort(["año", "ramo"])
    )
    if dups.height:
        print(dups)
    assert dups.height == 0, "duplicate rows after canonicalization (likely a mis-mapping)"
    print("  none")

    print("\n=== Spot checks ===")
    def total(year):
        return df.filter((pl.col("año") == year) & (pl.col("tipo") == "total"))["proyectado"][0]
    print(f"  2012 Total proyectado (x1000): {total(2012):,.1f}  (expect 2,800,200.0)")
    print(f"  2023 Total proyectado (x1000): {total(2023):,.1f}  (expect 5,958,300.0)")
    print(f"  2024 Total proyectado (x1):    {total(2024):,.1f}  (expect 6,490,404.6)")

    ds = df.filter((pl.col("año") == 2019) & (pl.col("ramo").is_in(["Bienestar", "Desarrollo Social"])))
    print("\n  2019 Bienestar vs Desarrollo Social (disambiguation):")
    print(ds.select(["clasificacion", "ramo", "ramo_original", "proyectado"]))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, dropped = build()

    out_csv = OUT_DIR / "gasto_programable_normalizado.csv"
    df.write_csv(out_csv)

    # Reviewable canonicalization summary: each distinct source label -> canonical, with
    # the years it covers. Rows where ramo != ramo_original are the renames/fixes to audit.
    mapping = (
        df.group_by(["ramo", "ramo_original", "clasificacion", "grupo"])
        .agg(
            pl.col("año").sort().cast(pl.Utf8).str.join(";").alias("años"),
            pl.len().alias("n"),
        )
        .with_columns((pl.col("ramo") != pl.col("ramo_original")).alias("modificado"))
        .sort(["ramo", "clasificacion", "ramo_original"])
        .select(["ramo", "ramo_original", "modificado", "clasificacion", "grupo", "n", "años"])
    )
    mapping.write_csv(OUT_DIR / "ramo_mapping.csv")

    validate(df, dropped)
    print(f"\nWrote {df.height} rows -> {out_csv}")
    print(f"Wrote {mapping.height} mapping rows -> {OUT_DIR / 'ramo_mapping.csv'}")


if __name__ == "__main__":
    main()
