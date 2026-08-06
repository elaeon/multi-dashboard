#!/usr/bin/env python3
"""
Build a firearms/narcotics seizure table (by state or municipality) from INEGI's
CNSPE (state prosecutors), CNPJF (federal prosecutors, FGR), or CNSPF (Guardia
Nacional) censuses.

All three censuses publish an "aseguramientos" (seizures) module with a weapons
table and a narcotics table. Table numbers (m2s3pN / m2s8pN / m2bs4pN) and key
column names (entidad_a / cve_ent / cvegeo / entifed1, narmed* / narcgn*) drift
from year to year AND between institutions, so this script never hardcodes a
table number or a fixed narcotics column set: it reads each year's own
0_indice_*.csv to find the right table by its Spanish description, and detects
the narcotics column "profile" (narmed* for CNSPE/CNPJF vs. narcgn* for CNSPF)
from whichever columns are actually present.

CNSPE (state census) has one row per state in its base weapons/narcotics table.
CNPJF and CNSPF (federal censuses) only have ONE national row in their base
table -- there is no federal "by state" table, so state-level output for them
is a rollup of the municipality-level table instead. This is automatic
(detected by row count / missing entity column), not a flag.

Usage:
  uv run python scripts/datatable/inegi_seizures_table.py                          # CNPJF (federal), state-level, latest year
  uv run python scripts/datatable/inegi_seizures_table.py --source state           # CNSPE (state), state-level, latest year
  uv run python scripts/datatable/inegi_seizures_table.py --source gn              # CNSPF (Guardia Nacional), state-level, latest year
  uv run python scripts/datatable/inegi_seizures_table.py --source state --year 2023  # CNSPE, year 2023
  uv run python scripts/datatable/inegi_seizures_table.py --municipality           # CNPJF, municipality-level
  uv run python scripts/datatable/inegi_seizures_table.py --source gn --municipality --year 2024
"""
import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DATASETS = {
    "cnspe": REPO_ROOT / "data" / "inegi" / "cnspe",
    "cnpjf": REPO_ROOT / "data" / "inegi" / "cnpjf",
    "cnspf": REPO_ROOT / "data" / "inegi" / "cnspf",
}

SENTINELS = {"NA", "NSS", "ND", "NP", "NULL"}

FIRE_COLS = ["armasa1", "armasa2", "armasa3"]

# Same drug taxonomy, different column-name prefix/numbering per institution.
NARCO_PROFILES = {
    "narmed": {  # CNSPE, CNPJF
        "groups": {
            "Cocaína": ["narmed1"],
            "Cannabis/Mariguana": ["narmed3", "narmed4", "narmed5", "narmed6", "narmed7"],
            "Amapola/Opiáceos": ["narmed10", "narmed11", "narmed12", "narmed13", "narmed14", "narmed15"],
            "LSD": ["narmed17"],
            "MDA": ["narmed18"],
            "MDMA": ["narmed20"],
            "Metanfetamina": ["narmed22"],
            "Fentanilo": ["narmed25"],
        },
        "pastillas_col": "narmed26",
    },
    "narcgn": {  # CNSPF (Guardia Nacional)
        "groups": {
            "Cocaína": ["narcgn1"],
            "Cannabis/Mariguana": ["narcgn2", "narcgn3", "narcgn4", "narcgn5", "narcgn6"],
            "Amapola/Opiáceos": ["narcgn7", "narcgn8", "narcgn9", "narcgn10", "narcgn11", "narcgn12"],
            "LSD": ["narcgn13"],
            "MDA": ["narcgn14"],
            "MDMA": ["narcgn16"],
            "Metanfetamina": ["narcgn18"],
            "Fentanilo": ["narcgn21"],
        },
        "pastillas_col": "narcgn22",
    },
}
ALL_NARCO_KG_COLS = sorted({c for p in NARCO_PROFILES.values() for cols in p["groups"].values() for c in cols})

# 2025+ schema for CNSPE/CNSPF (narmed*/narcgn* above still applies to CNPJF):
# one row per (entity[, municipality], narcotic type) instead of one wide
# column per narcotic subtype, with the amount split across unit-of-measure
# columns per the official diccionario de datos: ucnarc1=miligramos,
# ucnarc2=gramos, ucnarc3=kilogramos, ucnarc4=toneladas, ucnarc5=litros,
# ucnarc6=metros, ucnarc7=piezas o unidades, ucnarc8=tableta/pastilla/cápsula,
# ucnarc9=caja o cajetilla, ucnarc10=otra unidad, ucnarc11=no identificado.
# Only the weight columns convert to kg -- liquids/pieces/pills aren't a
# weight, same convention as before (fentanilo pastillas were already
# tracked separately from fentanilo kg in the old narmed26/narcgn22 column).
UCNARC_KG_FACTORS = {"ucnarc1": 1e-6, "ucnarc2": 1e-3, "ucnarc3": 1.0, "ucnarc4": 1000.0}
UCNARC_PASTILLAS_COL = "ucnarc8"
UCNARC_FENTANILO_CODE = 16
# narcotic_a catalog codes -> same group taxonomy as NARCO_PROFILES. MDMA no
# longer has its own code in this catalog (folded into "Psicotrópicos"/"Otro
# tipo", codes 15/17) so it's dropped instead of force-mapped somewhere else.
NARCOTIC_A_GROUPS = {
    "Cocaína": [1],
    "Cannabis/Mariguana": [2, 3, 4, 5],
    "Amapola/Opiáceos": [6, 7, 8, 9, 10, 11],
    "LSD": [12],
    "MDA": [13],
    "Metanfetamina": [14],
    "Fentanilo": [16],
}

# Phrases are stable across years even when table numbers/suffixes aren't.
WEAPONS_BASE_RE = re.compile(r"armas asegurad.*seg[uú]n tipo,\s*durante", re.IGNORECASE)
WEAPONS_MUNI_RE = re.compile(r"armas asegurad.*municipio.*tipo", re.IGNORECASE)
NARCO_BASE_RE = re.compile(
    r"narc[oó]ticos asegurad.*tipo de narc[oó]tico y unidad de medida", re.IGNORECASE
)
NARCO_MUNI_RE = re.compile(
    r"narc[oó]ticos asegurad.*municipio.*tipo de narc[oó]tico", re.IGNORECASE
)


def find_zip(dataset: str, year: int) -> Path:
    year_dir = DATASETS[dataset] / str(year)
    if not year_dir.is_dir():
        raise SystemExit(f"No {dataset.upper()} directory for {year}: {year_dir}")
    candidates = [p for p in year_dir.glob("*.zip") if "asegu" in p.name.lower()]
    if not candidates:
        raise SystemExit(f"No seizures ('asegu*') ZIP found in {year_dir}")
    return candidates[0]


def latest_year(dataset: str) -> int:
    years = []
    for year_dir in DATASETS[dataset].iterdir():
        if year_dir.is_dir() and year_dir.name.isdigit():
            if any("asegu" in p.name.lower() for p in year_dir.glob("*.zip")):
                years.append(int(year_dir.name))
    if not years:
        raise SystemExit(f"No seizures ZIPs found under {DATASETS[dataset]}")
    return max(years)


def load_index(zf: zipfile.ZipFile) -> pd.DataFrame:
    (name,) = [n for n in zf.namelist() if re.search(r"0_indice.*\.csv$", n)]
    raw = zf.read(name)
    try:
        return pd.read_csv(io.BytesIO(raw), encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(raw), encoding="latin-1")


def resolve_table(index_df: pd.DataFrame, pattern: re.Pattern) -> str | None:
    hits = index_df[index_df["CONTENIDO"].str.contains(pattern, na=False, regex=True)]
    if hits.empty:
        return None
    return hits.iloc[0]["ARCHIVO"]


def load_table(zf: zipfile.ZipFile, archivo: str) -> pd.DataFrame:
    (name,) = [
        n
        for n in zf.namelist()
        if n.startswith("conjunto_de_datos/") and n.split("/")[-1].startswith(archivo + "_")
    ]
    return pd.read_csv(io.BytesIO(zf.read(name)), encoding="utf-8-sig", dtype=str)


def to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = 0
            continue
        s = df[c].astype(str).str.strip()
        df[c] = pd.to_numeric(s, errors="coerce").where(~s.isin(SENTINELS), 0).fillna(0)
    return df


def load_entity_catalog(zf: zipfile.ZipFile, code_col: str) -> dict:
    matches = [n for n in zf.namelist() if n.startswith(f"catalogos/{code_col}_")]
    if not matches:
        return {}
    raw = zf.read(matches[0])
    try:
        cat = pd.read_csv(io.BytesIO(raw), encoding="utf-8")
    except UnicodeDecodeError:
        cat = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
    cat.columns = [c.strip() for c in cat.columns]
    name_col = "descrip" if "descrip" in cat.columns else "nom_ent"
    return dict(zip(cat[code_col].astype(int), cat[name_col]))


ENTITY_KEY_CANDIDATES = ("entifed1", "entidad_a", "cve_ent", "cvegeo")


class NoEntityColumn(Exception):
    pass


def has_entity_col(df: pd.DataFrame) -> bool:
    return any(c in df.columns for c in ENTITY_KEY_CANDIDATES)


def add_state_column(df: pd.DataFrame, zf: zipfile.ZipFile) -> pd.DataFrame:
    if "entifed1" in df.columns:
        df["state"] = df["entifed1"].str.title()
        return df
    for code_col in ("entidad_a", "cve_ent", "cvegeo"):
        if code_col in df.columns:
            mapping = load_entity_catalog(zf, code_col)
            if mapping:
                df["state"] = pd.to_numeric(df[code_col], errors="coerce").map(mapping)
                return df
    raise NoEntityColumn(f"Could not resolve a state column among: {list(df.columns)}")


def check_schema(df: pd.DataFrame, expected_cols: list[str], label: str, archivo: str):
    if not any(c in df.columns for c in expected_cols):
        raise SystemExit(
            f"{label} table '{archivo}' does not use the expected column schema "
            f"(none of {expected_cols} present; found {list(df.columns)}). "
            "This year's table layout has likely changed and isn't supported yet."
        )


def assert_one_row_per_state(df: pd.DataFrame, label: str):
    if df["state"].duplicated().any():
        raise SystemExit(
            f"{label}: expected one row per state but found duplicates "
            f"({df['state'].duplicated().sum()} dup rows). The source table's shape "
            "has likely changed for this year and isn't supported yet."
        )


def resolve_narco_profile(df: pd.DataFrame) -> dict | None:
    for profile in NARCO_PROFILES.values():
        first_col = next(iter(profile["groups"].values()))[0]
        if first_col in df.columns:
            return profile
    return None


def top_narcotic(row: pd.Series, group_names: list[str]) -> tuple[str, float]:
    vals = row[group_names]
    if vals.sum() == 0:
        return ("—", 0.0)
    top = vals.idxmax()
    return (top, round(float(vals[top]), 1))


def _weapons_muni_rollup(zf: zipfile.ZipFile, index_df: pd.DataFrame) -> pd.DataFrame | None:
    muni_archivo = resolve_table(index_df, WEAPONS_MUNI_RE)
    if muni_archivo is None:
        return None
    muni = load_table(zf, muni_archivo)
    check_schema(muni, FIRE_COLS, "Weapons (municipality)", muni_archivo)
    muni = to_num(muni, FIRE_COLS)
    muni = add_state_column(muni, zf)
    out = muni.groupby("state", as_index=False)[FIRE_COLS].sum()
    out["firearms"] = out[FIRE_COLS].sum(axis=1).astype(int)
    return out[["state", "firearms"]]


def build_weapons(zf: zipfile.ZipFile, index_df: pd.DataFrame, municipality: bool) -> pd.DataFrame | None:
    if municipality:
        archivo = resolve_table(index_df, WEAPONS_MUNI_RE)
    else:
        archivo = resolve_table(index_df, WEAPONS_BASE_RE)
    if archivo is None:
        return None
    df = load_table(zf, archivo)
    check_schema(df, FIRE_COLS, "Weapons", archivo)
    df = to_num(df, FIRE_COLS)
    df["firearms"] = df[FIRE_COLS].sum(axis=1).astype(int)

    if not municipality:
        # CNPJF-style national single-row (or columnless) base table: no
        # per-state grain here, roll up the municipality table instead.
        if not has_entity_col(df):
            return _weapons_muni_rollup(zf, index_df)
        df = add_state_column(df, zf)
        if df["state"].nunique() <= 1:
            return _weapons_muni_rollup(zf, index_df)
        out = df[["state", "firearms"]]
        assert_one_row_per_state(out, "Weapons")
        return out

    df = add_state_column(df, zf)
    muni_name_col = "municip1" if "municip1" in df.columns else None
    cols = (["state", muni_name_col] if muni_name_col else ["state"]) + ["firearms"]
    return df[cols]


def _finish_narcotics(df: pd.DataFrame, profile: dict) -> pd.DataFrame:
    groups = profile["groups"]
    kg_cols = [c for cols in groups.values() for c in cols]
    pastillas_col = profile["pastillas_col"]
    for g, cols in groups.items():
        df[g] = df[cols].sum(axis=1)
    df["narcotics_kg"] = df[kg_cols].sum(axis=1).round(1)
    df[["top_drug", "top_drug_kg"]] = df.apply(
        lambda r: pd.Series(top_narcotic(r, list(groups.keys()))), axis=1
    )
    fentanilo_kg_col = groups["Fentanilo"][0]
    df = df.rename(columns={fentanilo_kg_col: "fentanilo_kg", pastillas_col: "fentanilo_pastillas"})
    df["fentanilo_pastillas"] = df["fentanilo_pastillas"].astype(int)
    return df


def _is_long_narco_schema(df: pd.DataFrame) -> bool:
    return "narcotic_a" in df.columns


def _ucnarc_long_to_groups(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    df = to_num(df, list(UCNARC_KG_FACTORS) + [UCNARC_PASTILLAS_COL])
    df["narcotic_a"] = pd.to_numeric(df["narcotic_a"], errors="coerce")
    df["kg"] = sum(df[c] * factor for c, factor in UCNARC_KG_FACTORS.items())

    code_to_group = {code: g for g, codes in NARCOTIC_A_GROUPS.items() for code in codes}
    df["grupo"] = df["narcotic_a"].map(code_to_group)

    group_names = list(NARCOTIC_A_GROUPS)
    kg_wide = (
        df.dropna(subset=["grupo"])
        .groupby(group_cols + ["grupo"])["kg"].sum()
        .unstack("grupo", fill_value=0.0)
        .reindex(columns=group_names, fill_value=0.0)
        .reset_index()
    )

    fent = (
        df[df["narcotic_a"] == UCNARC_FENTANILO_CODE]
        .groupby(group_cols)[UCNARC_PASTILLAS_COL].sum()
        .rename("fentanilo_pastillas")
        .reset_index()
    )
    out = kg_wide.merge(fent, on=group_cols, how="left")
    out["fentanilo_pastillas"] = out["fentanilo_pastillas"].fillna(0).astype(int)

    out["narcotics_kg"] = out[group_names].sum(axis=1).round(1)
    out[["top_drug", "top_drug_kg"]] = out.apply(lambda r: pd.Series(top_narcotic(r, group_names)), axis=1)
    out = out.rename(columns={"Fentanilo": "fentanilo_kg"})
    out["fentanilo_kg"] = out["fentanilo_kg"].round(1)
    return out[group_cols + ["narcotics_kg", "top_drug", "top_drug_kg", "fentanilo_kg", "fentanilo_pastillas"]]


def _build_narcotics_long(zf: zipfile.ZipFile, index_df: pd.DataFrame, df: pd.DataFrame, municipality: bool) -> pd.DataFrame | None:
    """Camino de cálculo para el esquema narcotic_a/ucnarc* (2025+, formato
    largo) -- espejo de build_narcotics/_narcotics_muni_rollup para el
    esquema narmed*/narcgn* anterior."""
    if municipality:
        df = add_state_column(df, zf)
        group_cols = ["state", "municip1"] if "municip1" in df.columns else ["state"]
        return _ucnarc_long_to_groups(df, group_cols)

    use_muni_rollup = not has_entity_col(df)
    if not use_muni_rollup:
        df = add_state_column(df, zf)
        use_muni_rollup = df["state"].nunique() <= 1

    if use_muni_rollup:
        muni_archivo = resolve_table(index_df, NARCO_MUNI_RE)
        if muni_archivo is None:
            return None
        df = add_state_column(load_table(zf, muni_archivo), zf)

    return _ucnarc_long_to_groups(df, ["state"])


def _narcotics_muni_rollup(zf: zipfile.ZipFile, index_df: pd.DataFrame) -> pd.DataFrame | None:
    muni_archivo = resolve_table(index_df, NARCO_MUNI_RE)
    if muni_archivo is None:
        return None
    muni = load_table(zf, muni_archivo)
    check_schema(muni, ALL_NARCO_KG_COLS, "Narcotics (municipality)", muni_archivo)
    profile = resolve_narco_profile(muni)
    kg_cols = [c for cols in profile["groups"].values() for c in cols]
    pastillas_col = profile["pastillas_col"]
    muni = to_num(muni, kg_cols + [pastillas_col])
    muni = add_state_column(muni, zf)
    df = muni.groupby("state", as_index=False)[kg_cols + [pastillas_col]].sum()
    df = _finish_narcotics(df, profile)
    return df[["state", "narcotics_kg", "top_drug", "top_drug_kg", "fentanilo_kg", "fentanilo_pastillas"]]


def build_narcotics(zf: zipfile.ZipFile, index_df: pd.DataFrame, municipality: bool) -> pd.DataFrame | None:
    if municipality:
        archivo = resolve_table(index_df, NARCO_MUNI_RE)
    else:
        archivo = resolve_table(index_df, NARCO_BASE_RE)
    if archivo is None:
        return None
    df = load_table(zf, archivo)

    if _is_long_narco_schema(df):
        return _build_narcotics_long(zf, index_df, df, municipality)

    check_schema(df, ALL_NARCO_KG_COLS, "Narcotics", archivo)
    profile = resolve_narco_profile(df)
    if profile is None:
        raise SystemExit(f"Narcotics table '{archivo}' matched no known column profile.")
    kg_cols = [c for cols in profile["groups"].values() for c in cols]
    pastillas_col = profile["pastillas_col"]
    df = to_num(df, kg_cols + [pastillas_col])

    if not municipality:
        if not has_entity_col(df):
            return _narcotics_muni_rollup(zf, index_df)
        df = add_state_column(df, zf)
        if df["state"].nunique() <= 1:
            return _narcotics_muni_rollup(zf, index_df)
        assert_one_row_per_state(df, "Narcotics")
        df = _finish_narcotics(df, profile)
        return df[["state", "narcotics_kg", "top_drug", "top_drug_kg", "fentanilo_kg", "fentanilo_pastillas"]]

    df = add_state_column(df, zf)
    df = _finish_narcotics(df, profile)
    cols = ["state"]
    if "municip1" in df.columns:
        cols.append("municip1")
    cols += ["narcotics_kg", "top_drug", "top_drug_kg", "fentanilo_kg", "fentanilo_pastillas"]
    return df[cols]


def _fmt(v, decimales=0, sufijo=""):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v:,.{decimales}f}{sufijo}"


DATASET_LABELS = {
    "cnspe": "CNSPE (Fiscalías estatales)",
    "cnpjf": "CNPJF (FGR)",
    "cnspf": "CNSPF (Guardia Nacional)",
}


def print_table(merged: pd.DataFrame, dataset: str, year: int, municipality: bool):
    has_muni = "municip1" in merged.columns
    has_weapons = "firearms" in merged.columns
    has_narco = "narcotics_kg" in merged.columns
    grano = "municipio" if municipality else "entidad"
    ancho = 128 if has_narco else 40

    print(f"\n{'═' * ancho}")
    print(f"  Aseguramientos de armas y narcóticos — {DATASET_LABELS[dataset]}, {year} (por {grano})")
    print("═" * ancho)

    encabezado = f"  {'Entidad':<22}"
    if has_muni:
        encabezado += f" {'Municipio':<24}"
    if has_weapons:
        encabezado += f" {'Armas':>8}"
    if has_narco:
        encabezado += (f" {'Narcóticos (kg)':>16} {'Droga principal':<20} {'kg principal':>13}"
                       f" {'Fentanilo (kg)':>15} {'Fent. (pastillas)':>18}")
    print(encabezado)

    for _, r in merged.iterrows():
        fila = f"  {str(r['state']):<22}"
        if has_muni:
            fila += f" {str(r.get('municip1') or '—'):<24}"
        if has_weapons:
            fila += f" {_fmt(r.get('firearms')):>8}"
        if has_narco:
            fila += (f" {_fmt(r.get('narcotics_kg'), 1):>16} {str(r.get('top_drug') or '—'):<20}"
                     f" {_fmt(r.get('top_drug_kg'), 1):>13} {_fmt(r.get('fentanilo_kg'), 1):>15}"
                     f" {_fmt(r.get('fentanilo_pastillas')):>18}")
        print(fila)

    print()
    if has_weapons:
        print(f"  Total armas: {_fmt(merged['firearms'].sum())}")
    if has_narco:
        print(f"  Total narcóticos: {_fmt(merged['narcotics_kg'].sum(), 1)} kg"
              f" | Fentanilo: {_fmt(merged['fentanilo_kg'].sum(), 1)} kg, {_fmt(merged['fentanilo_pastillas'].sum())} pastillas")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        choices=["state", "fed", "gn"],
        default="fed",
        help="state=CNSPE (state prosecutors), fed=CNPJF (federal prosecutors/FGR, default), gn=CNSPF (Guardia Nacional)",
    )
    parser.add_argument("--municipality", action="store_true", help="Roll down to municipality level instead of state level")
    parser.add_argument("--year", type=int, default=None, help="Census reference year (default: latest available)")
    args = parser.parse_args()

    dataset = {"state": "cnspe", "fed": "cnpjf", "gn": "cnspf"}[args.source]
    year = args.year or latest_year(dataset)
    zip_path = find_zip(dataset, year)

    print(f"# dataset={dataset.upper()} year={year} source={zip_path.relative_to(REPO_ROOT)}", file=sys.stderr)

    with zipfile.ZipFile(zip_path) as zf:
        index_df = load_index(zf)
        weapons = build_weapons(zf, index_df, args.municipality)
        narcotics = build_narcotics(zf, index_df, args.municipality)

    if weapons is None and narcotics is None:
        raise SystemExit("Neither a weapons nor a narcotics table was found in this ZIP for this year.")

    join_cols = ["state", "municip1"] if (args.municipality and weapons is not None and "municip1" in weapons.columns) else ["state"]

    if weapons is not None and narcotics is not None:
        merged = weapons.merge(narcotics, on=join_cols, how="outer")
    elif weapons is not None:
        print("# warning: no narcotics table found for this year", file=sys.stderr)
        merged = weapons
    else:
        print("# warning: no weapons table found for this year", file=sys.stderr)
        merged = narcotics

    sort_col = "firearms" if "firearms" in merged.columns else "narcotics_kg"
    merged = merged.sort_values(sort_col, ascending=False)
    print_table(merged, dataset, year, args.municipality)


if __name__ == "__main__":
    main()
