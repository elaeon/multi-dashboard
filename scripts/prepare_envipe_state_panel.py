"""Build state × year ENVIPE panel with victimization, insecurity, reporting rate,
and institutional trust.

Sources: data/inegi/envipe/ — 9 ZIP files (2017–2025)
Output:  data/inegi/envipe/envipe_state_panel.parquet

Columns:
  CLAVE_ENT        Int64   — 2-digit state code (matches _build_panel_annual join key)
  año              Int64   — ENVIPE survey year (measures crimes from prior 12 months)
  vic_envipe       Float64 — weighted % of persons who were crime victims (from tmod_vic)
  inseg_envipe     Float64 — weighted % feeling insecure in their municipality (AP4_3_2 == 1)
  rep_rate_envipe  Float64 — weighted % of crimes reported to authorities (BP1_20, FAC_DEL)
  trust_inst_envipe Float64 — composite institutional trust index [0,1] (1=max trust)
                              mean of AP5_4_01..AP5_4_11, inverted: (5 - raw) / 3
                              raw scale: 1=mucha confianza … 4=nada de confianza; 9=NS/NR excluded

Methodology:
  · tmod_vic: one row per crime; provides unique victim IDs (for vic_envipe) and
    BP1_20/FAC_DEL (for rep_rate_envipe)
  · tper_vic1: one row per respondent; provides CVE_ENT, FAC_ELE, AP4_3_2 (inseg),
    and AP5_4_* (trust)
  · Filter tper_vic1 to weight > 0 (both result codes 'A' and 'B' are valid)
  · Weighted mean by FAC_ELE (persons) or FAC_DEL (crimes) per state
  · ENVIPE year N surveys the prior 12 months → inherent 1-year lag vs annual migration data

Memory strategy:
  · tmod_vic: two passes — victim ID set (~1 MB) then state-level reporting aggregates
  · tper_vic1: single streaming pass; peak RAM = O(32 states × ~20 floats)
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


def collect_victim_ids_and_reporting(
    z: zipfile.ZipFile, path: str
) -> tuple[set[str], dict[str, dict[str, float]]]:
    """Stream tmod_vic once: collect victim IDs and aggregate reporting rate per state.

    Returns:
      victims   — set of ID_PER strings (one per unique victim)
      state_rep — {state_str: {"rep_num": float, "w_total": float}}
                  rep_num = sum(FAC_DEL) where BP1_20 == 1
                  w_total = sum(FAC_DEL) where BP1_20 in {1, 2}
    """
    victims: set[str] = set()
    state_rep: dict[str, dict[str, float]] = {}

    with z.open(path) as raw_f:
        f = io.TextIOWrapper(raw_f, encoding="latin-1")
        reader = csv.reader(f)
        headers = [h.strip().strip('"').upper() for h in next(reader)]

        def idx(col: str) -> int | None:
            try:
                return headers.index(col)
            except ValueError:
                return None

        i_per   = idx("ID_PER")
        i_upm   = idx("UPM")       # first 2 chars = state code (no CVE_ENT in tmod_vic)
        i_fac   = idx("FAC_DEL")
        i_bp120 = idx("BP1_20")

        for row in reader:
            if not row:
                continue

            # Victim ID (for vic_envipe)
            if i_per is not None and i_per < len(row):
                victims.add(row[i_per].strip().strip('"'))

            # Reporting rate — skip if required columns absent
            if i_upm is None or i_fac is None or i_bp120 is None:
                continue
            try:
                weight = float(row[i_fac].strip().strip('"').replace(",", ""))
            except (ValueError, IndexError):
                continue
            if weight <= 0:
                continue

            bp120 = row[i_bp120].strip().strip('"') if i_bp120 < len(row) else ""
            if bp120 not in ("1", "2"):
                continue  # only count crimes with a valid yes/no on reporting

            # Extract state from UPM: first 2 characters (INEGI convention)
            upm = row[i_upm].strip().strip('"') if i_upm < len(row) else ""
            state = upm[:2].zfill(2) if len(upm) >= 2 else ""
            if not state:
                continue
            try:
                clave = int(state)
            except ValueError:
                continue
            if not (1 <= clave <= 32):
                continue

            if state not in state_rep:
                state_rep[state] = {"rep_num": 0.0, "w_total": 0.0}
            state_rep[state]["rep_num"] += weight * (1.0 if bp120 == "1" else 0.0)
            state_rep[state]["w_total"] += weight

    return victims, state_rep


def aggregate_tper_vic1(
    z: zipfile.ZipFile,
    path: str,
    victims: set[str],
    state_rep: dict[str, dict[str, float]],
    year: int,
) -> pl.DataFrame | None:
    """Stream tper_vic1: compute per-state victimization rate, insecurity, and trust."""

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
        i_weight = idx("FAC_ELE")
        i_inseg  = idx("AP4_3_2")

        # Discover all AP5_4_NN trust columns present in this year's file
        trust_cols = sorted(
            [h for h in headers if re.match(r"AP5_4_\d+$", h)],
            key=lambda c: int(re.search(r"\d+$", c).group()),
        )
        trust_idxs = [headers.index(c) for c in trust_cols]

        if i_state is None or i_weight is None:
            print(f"  ERROR: missing CVE_ENT or FAC_ELE in {year}")
            return None

        for row in reader:
            if not row:
                continue
            try:
                weight = float(row[i_weight].strip().strip('"').replace(",", ""))
            except (ValueError, IndexError):
                continue
            if weight <= 0:
                continue

            state = row[i_state].strip().strip('"').zfill(2) if i_state < len(row) else ""
            if not state:
                continue

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

            # Composite institutional trust: mean of valid AP5_4_* (1–4 scale, 9=exclude)
            # Invert: (5 - raw) / 3  →  maps [1,4] to [1.0, 0.33]; raw=1 = máxima confianza
            trust_sum = 0.0
            trust_n   = 0
            for ti in trust_idxs:
                if ti >= len(row):
                    continue
                try:
                    v_int = int(row[ti].strip().strip('"'))
                except ValueError:
                    continue
                if 1 <= v_int <= 4:
                    trust_sum += (5 - v_int) / 3
                    trust_n   += 1

            if state not in state_agg:
                state_agg[state] = {
                    "vic_num": 0.0, "w_total": 0.0,
                    "inseg_num": 0.0, "w_inseg": 0.0,
                    "trust_num": 0.0, "w_trust": 0.0,
                }
            a = state_agg[state]
            a["vic_num"]  += weight * (1.0 if is_victim else 0.0)
            a["w_total"]  += weight
            if inseg_flag is not None:
                a["inseg_num"] += weight * inseg_flag
                a["w_inseg"]   += weight
            if trust_n > 0:
                a["trust_num"] += weight * (trust_sum / trust_n)
                a["w_trust"]   += weight

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

        vic   = a["vic_num"]   / a["w_total"]  if a["w_total"]  > 0 else None
        inseg = a["inseg_num"] / a["w_inseg"]  if a["w_inseg"]  > 0 else None
        trust = a["trust_num"] / a["w_trust"]  if a["w_trust"]  > 0 else None

        rep_data = state_rep.get(state_str)
        rep = (rep_data["rep_num"] / rep_data["w_total"]
               if rep_data and rep_data["w_total"] > 0 else None)

        rows.append({
            "CLAVE_ENT":         clave,
            "año":               year,
            "vic_envipe":        vic,
            "inseg_envipe":      inseg,
            "rep_rate_envipe":   rep,
            "trust_inst_envipe": trust,
        })

    result = pl.DataFrame(rows, schema={
        "CLAVE_ENT":          pl.Int64,
        "año":                pl.Int64,
        "vic_envipe":         pl.Float64,
        "inseg_envipe":       pl.Float64,
        "rep_rate_envipe":    pl.Float64,
        "trust_inst_envipe":  pl.Float64,
    }).sort("CLAVE_ENT")

    n_states  = result.height
    vic_nat   = float(result["vic_envipe"].drop_nulls().mean()        or 0) * 100
    ins_nat   = float(result["inseg_envipe"].drop_nulls().mean()      or 0) * 100
    rep_nat   = float(result["rep_rate_envipe"].drop_nulls().mean()   or 0) * 100
    trust_nat = float(result["trust_inst_envipe"].drop_nulls().mean() or 0)
    n_trust   = len(trust_cols)
    print(
        f"  {year}: n={n_rows:,} respondents  {len(victims):,} unique victims  "
        f"{n_states} states  "
        f"vic={vic_nat:.1f}%  inseg={ins_nat:.1f}%  "
        f"rep={rep_nat:.1f}%  trust={trust_nat:.3f} ({n_trust} inst)"
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

    victims, state_rep = collect_victim_ids_and_reporting(z, mod_path)
    return aggregate_tper_vic1(z, per_path, victims, state_rep, year)


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
