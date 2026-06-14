"""Build state × year ENVIPE panel with victimization prevalence and insecurity perception.

Sources: data/inegi/envipe/ — 9 ZIP files (2017–2025)
Output:  data/inegi/envipe/envipe_state_panel.parquet

Columns:
  CLAVE_ENT    Int64   — 2-digit state code (matches _build_panel_annual join key)
  año          Int64   — ENVIPE survey year (measures crimes from prior 12 months)
  vic_envipe   Float64 — weighted % of persons who were crime victims (from tmod_vic)
  inseg_envipe Float64 — weighted % feeling insecure in their municipality (AP4_3_2 == 1)

Methodology:
  · tmod_vic: one row per crime incident; stream to collect unique victim person IDs
  · tper_vic1: one row per respondent; provides CVE_ENT, FAC_ELE, AP4_3_2
    - victim if ID_PER appears in tmod_vic victim set
    - insecure if AP4_3_2 == "1" (coded 1=Inseguro, 2=Seguro, 9=DK in ENVIPE)
  · Filter tper_vic1 to RESUL_H == "B" (completed household interview)
  · Weighted mean by FAC_ELE per state
  · ENVIPE year N surveys the prior 12 months → inherent 1-year lag vs annual migration data

Memory strategy:
  · tmod_vic: only ID_PER column loaded → victim set fits in ~1 MB
  · tper_vic1: streamed row-by-row; peak RAM = O(32 states × 4 floats)
"""

import csv
import io
import os
import re
import zipfile

import polars as pl

ENVIPE_DIR = "data/inegi/envipe"
OUTPUT     = "data/inegi/envipe/envipe_state_panel.parquet"


def extract_year(filename: str) -> int | None:
    m = re.search(r"(20\d{2})", filename)
    return int(m.group(1)) if m else None


def find_table(z: zipfile.ZipFile, table_substr: str) -> str | None:
    """Return path to a table CSV inside the zip (case-insensitive match)."""
    names = z.namelist()
    candidates = [
        n for n in names
        if table_substr in n.lower()
        and n.lower().endswith(".csv")
        and "diccionario" not in n.lower()
        and "catalogo"    not in n.lower()
        and "metadato"    not in n.lower()
    ]
    data = [c for c in candidates if "conjunto_de_datos" in c.lower()]
    return (data or candidates or [None])[0]


def collect_victim_ids(z: zipfile.ZipFile, path: str) -> set[str]:
    """Stream tmod_vic and collect unique ID_PER values (one per victim).

    tmod_vic only contains crime incidents — every row is a victim record.
    RESUL_H in tmod_vic is "A" (not "B"), so we skip the household-result filter.
    """
    victims: set[str] = set()
    with z.open(path) as raw_f:
        f = io.TextIOWrapper(raw_f, encoding="latin-1")
        reader = csv.reader(f)
        headers = [h.strip().strip('"').upper() for h in next(reader)]
        try:
            i_per = headers.index("ID_PER")
        except ValueError:
            return victims
        for row in reader:
            if not row or i_per >= len(row):
                continue
            victims.add(row[i_per].strip().strip('"'))
    return victims


def aggregate_tper_vic1(
    z: zipfile.ZipFile, path: str, victims: set[str], year: int
) -> pl.DataFrame | None:
    """Stream tper_vic1, compute per-state weighted victimization and insecurity rates."""

    state_agg: dict[str, dict[str, float]] = {}
    n_rows = 0

    with z.open(path) as raw_f:
        f = io.TextIOWrapper(raw_f, encoding="latin-1")
        reader = csv.reader(f)
        headers = [h.strip().strip('"').upper() for h in next(reader)]

        def idx(col: str) -> int | None:
            try:
                return headers.index(col)
            except ValueError:
                return None

        i_per    = idx("ID_PER")
        i_state  = idx("CVE_ENT")
        i_result = idx("RESUL_H")
        i_weight = idx("FAC_ELE")
        i_inseg  = idx("AP4_3_2")   # Percepción inseguridad municipio (1=Inseguro, 2=Seguro)

        if i_state is None or i_weight is None:
            print(f"  ERROR: missing CVE_ENT or FAC_ELE in {year}")
            return None

        for row in reader:
            if not row:
                continue
            # RESUL_H in tper_vic1: 'B' = general questionnaire, 'A' = victimization module.
            # Both are valid respondents — do NOT filter by RESUL_H; use FAC_ELE > 0 instead.
            try:
                weight = float(row[i_weight].strip().strip('"').replace(",", ""))
            except (ValueError, IndexError):
                continue
            if weight <= 0:
                continue

            state = row[i_state].strip().strip('"').zfill(2)
            is_victim = (
                i_per is not None
                and i_per < len(row)
                and row[i_per].strip().strip('"') in victims
            )

            inseg_flag: float | None = None
            if i_inseg is not None and i_inseg < len(row):
                v = row[i_inseg].strip().strip('"')
                if v in ("1", "2"):
                    inseg_flag = 1.0 if v == "1" else 0.0

            if state not in state_agg:
                state_agg[state] = {
                    "vic_num": 0.0, "w_total": 0.0,
                    "inseg_num": 0.0, "w_inseg": 0.0,
                }
            a = state_agg[state]
            a["vic_num"]   += weight * (1.0 if is_victim else 0.0)
            a["w_total"]   += weight
            if inseg_flag is not None:
                a["inseg_num"] += weight * inseg_flag
                a["w_inseg"]   += weight

            n_rows += 1

    if not state_agg:
        return None

    rows = []
    for state_str, a in sorted(state_agg.items()):
        try:
            clave = int(state_str)
        except ValueError:
            continue
        if not (1 <= clave <= 32):
            continue
        vic   = a["vic_num"]  / a["w_total"]  if a["w_total"]  > 0 else None
        inseg = a["inseg_num"] / a["w_inseg"] if a["w_inseg"] > 0 else None
        rows.append({
            "CLAVE_ENT": clave, "año": year,
            "vic_envipe": vic, "inseg_envipe": inseg,
        })

    result = pl.DataFrame(rows, schema={
        "CLAVE_ENT":    pl.Int64,
        "año":          pl.Int64,
        "vic_envipe":   pl.Float64,
        "inseg_envipe": pl.Float64,
    }).sort("CLAVE_ENT")

    n_states = result.height
    vic_nat  = float(result["vic_envipe"].drop_nulls().mean() or 0) * 100
    ins_nat  = float(result["inseg_envipe"].drop_nulls().mean() or 0) * 100
    print(
        f"  {year}: n={n_rows:,} respondents  {len(victims):,} unique victims  "
        f"{n_states} states  vic={vic_nat:.1f}%  inseg={ins_nat:.1f}%"
    )
    return result


def process_year(zip_path: str, year: int) -> pl.DataFrame | None:
    try:
        z = zipfile.ZipFile(zip_path)
    except Exception as e:
        print(f"  ERROR opening zip: {e}")
        return None

    mod_path = find_table(z, "mod_vic")
    per_path = find_table(z, "per_vic1")

    if mod_path is None or per_path is None:
        print(f"  ERROR: missing tmod_vic or tper_vic1 in {os.path.basename(zip_path)}")
        return None

    victims = collect_victim_ids(z, mod_path)
    return aggregate_tper_vic1(z, per_path, victims, year)


def main() -> None:
    zips = sorted(
        [os.path.join(ENVIPE_DIR, f) for f in os.listdir(ENVIPE_DIR) if f.endswith(".zip")],
        key=lambda p: extract_year(p) or 0,
    )
    print(f"Found {len(zips)} zip files")

    frames = []
    for zip_path in zips:
        year = extract_year(os.path.basename(zip_path))
        if year is None:
            print(f"  SKIP: could not parse year from {zip_path}")
            continue
        print(f"Processing {year} ← {os.path.basename(zip_path)}")
        frame = process_year(zip_path, year)
        if frame is not None:
            frames.append(frame)

    if not frames:
        print("ERROR: no data processed")
        return

    panel = pl.concat(frames).sort(["CLAVE_ENT", "año"])
    print(f"\nPanel: {panel.height} rows × {panel.width} cols")
    print(panel)

    panel.write_parquet(OUTPUT)
    print(f"\nSaved → {OUTPUT}")


if __name__ == "__main__":
    main()
