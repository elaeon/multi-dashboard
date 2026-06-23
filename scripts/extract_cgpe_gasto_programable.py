"""Extract 'Gasto programable del Sector Público' table from CGPE PDFs to CSV.

Source: data/cgpe/cgpe_YYYY.pdf (2012-2026, except 2025 which is scanned)
Output: data/cgpe/gasto_programable_YYYY.csv (one file per year)

Table: Gasto programable del Sector Público Presupuestario por clasificación
administrativa — found in all text-based PDFs.

Units vary by year:
  2012-2023: Miles de millones de pesos
  2024, 2026: Millones de pesos
"""

import argparse
import csv
import re
from pathlib import Path

import pdfplumber

PDF_DIR = Path("data/cgpe/raw")
OUT_DIR = Path("data/cgpe")

# Anchors that identify the right table page.
# Both must appear within ANCHOR_PROXIMITY characters of each other.
ANCHOR_A = "gasto programable del sector p"   # matches "sector público/publico"
ANCHOR_B = "clasificación administrativa"
ANCHOR_UNITS = ["millones de pesos", "miles de millones"]
ANCHOR_EXCLUDE = ["por ciento del pib", "porcentaje del pib"]
ANCHOR_PROXIMITY = 250  # title + subtitle are always close together

NUM_RE = re.compile(r"^-?\d{1,3}(?:,\d{3})*(?:\.\d+)?$")  # for matching individual words
_NUM_COUNT_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?")  # for counting in page text
_FOOTNOTE_RE = re.compile(r"^(p/|a/|e/|\d+[_/]|fuente\s*:|nota\s*:)", re.IGNORECASE)

OUTPUT_COLS = (
    "ramo",
    "nivel",
    "ppef_anterior",
    "pef_anterior",
    "proyectado",
    "var_abs_ppef",
    "var_abs_pef",
    "var_real_ppef_pct",
    "var_real_pef_pct",
    "unidad",
)


# ---------------------------------------------------------------------------
# Page detection
# ---------------------------------------------------------------------------

def _anchors_are_proximate(text):
    """Return True if ANCHOR_A and ANCHOR_B appear within ANCHOR_PROXIMITY chars."""
    tl = text.lower()
    for a, b in [(ANCHOR_A, ANCHOR_B), (ANCHOR_B, ANCHOR_A)]:
        i = tl.find(a)
        while i >= 0:
            j = tl.find(b, i)
            if 0 <= j - i <= ANCHOR_PROXIMITY:
                return True
            i = tl.find(a, i + 1)
    return False


def _find_table_pages(pdf):
    """Return list of 0-based page indices that contain the target table."""
    candidates = []
    for i, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        tl = text.lower()
        if ANCHOR_A not in tl:
            continue
        if not any(u in tl for u in ANCHOR_UNITS):
            continue
        if any(e in tl for e in ANCHOR_EXCLUDE):
            continue
        num_count = len(_NUM_COUNT_RE.findall(text))
        candidates.append((i, num_count))
    if not candidates:
        return []
    # Keep pages with substantial numeric content (the data page, not TOC)
    max_nums = max(n for _, n in candidates)
    return [i for i, n in candidates if n >= max_nums * 0.5]


def _detect_unit(page):
    text = (page.extract_text() or "").lower()
    if "miles de millones" in text:
        return "miles_de_millones_pesos"
    return "millones_pesos"


# ---------------------------------------------------------------------------
# Word-level row parsing
# ---------------------------------------------------------------------------

def _group_words_into_rows(words, y_tolerance=3):
    """Group words by y-coordinate within tolerance. Returns list of row word lists."""
    rows = {}
    for w in words:
        y_key = round(w["top"] / y_tolerance) * y_tolerance
        rows.setdefault(y_key, []).append(w)
    return [sorted(ws, key=lambda w: w["x0"]) for ws in sorted(rows.values(), key=lambda ws: ws[0]["top"])]


def _join_name_parts(parts):
    """Join name words, merging letter-spaced text like 'T o t a l'."""
    result = []
    i = 0
    while i < len(parts):
        p = parts[i]
        # Detect a run of single-character tokens (letter-spaced word)
        if len(p) == 1 and i + 1 < len(parts) and len(parts[i + 1]) == 1:
            run = []
            while i < len(parts) and len(parts[i]) <= 1 and not parts[i].isdigit():
                run.append(parts[i])
                i += 1
            result.append("".join(run))
        else:
            result.append(p)
            i += 1
    return " ".join(result).strip()


def _clean_number(s):
    """Parse a Mexican-format number string to float, or None if unparseable."""
    s = re.sub(r"[^\d.,-]", "", (s or "").strip())
    if not s or s in ("-", ""):
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Table extraction
# ---------------------------------------------------------------------------

_DATA_NUM_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+\.\d+")


def _parse_data_line(line):
    """Return (name_text, nums_list) for a budget data line, else (None, [])."""
    nums = _DATA_NUM_RE.findall(line)
    if len(nums) < 5:
        return None, []
    m = _DATA_NUM_RE.search(line)
    name = line[:m.start()].strip() if m else ""
    name = re.sub(r"\s*\d+_?/?\s*$", "", name).strip()
    name = re.sub(r"^\s*\d+_?/?\s*", "", name).strip()
    return name, nums[:7]


def _extract_rows_from_pages(pdf, page_indices):
    """Extract raw data rows from the given pages.

    Uses extract_text(layout=True) for entity names — this correctly handles
    multi-line names where text wraps around numbers.  Uses extract_words()
    to get x0 positions for hierarchy detection.

    Returns (list_of_raw_row_dicts, x_split) or ([], 0) on failure.
    """
    # --- Step 1: word-level extraction for x_split and hierarchy x0 ---
    all_words = []
    for pg_idx in page_indices:
        all_words.extend(pdf.pages[pg_idx].extract_words(keep_blank_chars=False, x_tolerance=2))

    if not all_words:
        return [], 0

    y_rows = _group_words_into_rows(all_words, y_tolerance=4)

    x_split = None
    y_start = None
    for row_words in y_rows:
        nums = [w for w in row_words if NUM_RE.match(w["text"])]
        if len(nums) >= 5:
            x_split = min(w["x0"] for w in nums) - 5
            y_start = row_words[0]["top"] - 1
            break

    if x_split is None:
        return [], 0

    # Collect (y, x0_first_name_word) for each data row, in page order.
    # For pure-number data rows (no name words), fall back to the x0 of the
    # most recent preceding name-only row so hierarchy is correctly detected.
    data_row_x0 = []
    last_name_x0 = None
    for row_words in y_rows:
        if row_words[0]["top"] < y_start:
            continue
        nums = [w for w in row_words if w["x0"] >= x_split and NUM_RE.match(w["text"])]
        name_ws = [w for w in row_words if w["x0"] < x_split]
        if name_ws:
            last_name_x0 = min(w["x0"] for w in name_ws)
        if len(nums) >= 5:
            x0 = min((w["x0"] for w in name_ws), default=None) or last_name_x0
            data_row_x0.append(x0)

    # --- Step 2: parse entity names from layout text lines ---
    all_lines = []
    for pg_idx in page_indices:
        text = pdf.pages[pg_idx].extract_text(layout=True) or ""
        all_lines.extend(text.splitlines())

    # Skip header lines; find the first data line.
    data_lines = []
    in_data = False
    for line in all_lines:
        stripped = line.strip()
        if not in_data and _parse_data_line(stripped)[1]:
            in_data = True
        if in_data:
            data_lines.append(stripped)

    if not data_lines:
        return [], 0

    # Parse entity rows: accumulate pending_name before data lines.
    # After each data line, check if the next non-empty line is a short
    # "trailing word" (≤2 words) belonging to the just-emitted entity.
    result_names = []  # list of [name_str, nums_list]
    pending_name = ""
    i = 0

    while i < len(data_lines):
        line = data_lines[i]
        if not line:
            i += 1
            continue

        name_part, nums = _parse_data_line(line)

        if nums:  # data line
            full_name = " ".join(filter(None, [pending_name, name_part]))
            full_name = re.sub(r"\s+", " ", full_name).strip()
            full_name = _join_name_parts(full_name.split())

            if full_name:
                result_names.append([full_name, nums])

            pending_name = ""

            # Look ahead: if next non-empty line is a "trailing word" belonging
            # to the entity just emitted, absorb it.
            #
            # When the data line had NO name (pure-numbers line), the entity's
            # name came entirely from pending_name + trailing words, so a larger
            # trailing threshold (≤8 words) is used.  When the data line already
            # had a name, only very short tails (≤2 words) are trailing; longer
            # lines are the start of the next entity.
            trailing_threshold = 8 if not name_part else 2

            j = i + 1
            while j < len(data_lines) and not data_lines[j].strip():
                j += 1
            if j < len(data_lines) and result_names:
                nxt = data_lines[j].strip()
                _, nxt_nums = _parse_data_line(nxt)
                if not nxt_nums and not _FOOTNOTE_RE.match(nxt):
                    if len(nxt.split()) <= trailing_threshold:
                        result_names[-1][0] = (result_names[-1][0] + " " + nxt).strip()
                        i = j + 1
                        continue
        else:
            # Name-only line — add to pending (skip footnotes and very short lines)
            if not _FOOTNOTE_RE.match(line) and len(line) > 1:
                pending_name = " ".join(filter(None, [pending_name, line]))

        i += 1

    # --- Step 3: pair names with x0 and build result ---
    # data_row_x0 was built in Step 1 in the same page order.
    n_names = len(result_names)
    n_x0 = len(data_row_x0)
    if n_names != n_x0:
        # Count mismatch; pad or trim x0 list so it aligns
        data_row_x0 = data_row_x0[:n_names] + [None] * max(0, n_names - n_x0)

    result = []
    for (name, nums_raw), x0 in zip(result_names, data_row_x0):
        nums = [_clean_number(n) for n in nums_raw[:7]]
        nums += [None] * (7 - len(nums))
        result.append({"_x0": x0 or 0, "_ramo": name, "_nums": nums})

    return result, x_split


def _assign_nivel(raw_rows):
    """Assign hierarchy level (0/1/2) based on x0 clustering."""
    if not raw_rows:
        return []

    x0_values = [r["_x0"] for r in raw_rows]
    # Cluster x0 values: sort unique values and identify 2-3 distinct bands
    unique_xs = sorted(set(round(x, 0) for x in x0_values))
    # Group into bands where gaps > 5pt indicate a level boundary
    bands = []
    for x in unique_xs:
        if not bands or x - bands[-1][-1] > 5:
            bands.append([x])
        else:
            bands[-1].append(x)
    band_min = [min(b) for b in bands]

    def x_to_nivel(x):
        closest = min(range(len(band_min)), key=lambda i: abs(band_min[i] - x))
        return closest

    result = []
    for i, r in enumerate(raw_rows):
        nivel = x_to_nivel(r["_x0"])
        # First row is always Total (nivel 0)
        if i == 0:
            nivel = 0
        result.append({
            "ramo": r["_ramo"],
            "nivel": nivel,
            "ppef_anterior": r["_nums"][0],
            "pef_anterior": r["_nums"][1],
            "proyectado": r["_nums"][2],
            "var_abs_ppef": r["_nums"][3],
            "var_abs_pef": r["_nums"][4],
            "var_real_ppef_pct": r["_nums"][5],
            "var_real_pef_pct": r["_nums"][6],
        })
    return result


# ---------------------------------------------------------------------------
# Per-year entry point
# ---------------------------------------------------------------------------

def extract_year(year):
    """Extract table for a single year. Returns list of row dicts."""
    pdf_path = PDF_DIR / f"cgpe_{year}.pdf"
    if not pdf_path.exists():
        print(f"  {year}: PDF not found, skipping")
        return []

    with pdfplumber.open(pdf_path) as pdf:
        page_indices = _find_table_pages(pdf)
        if not page_indices:
            print(f"  {year}: table page not found")
            return []

        print(f"  {year}: table on page(s) {[i+1 for i in page_indices]}")

        unit = _detect_unit(pdf.pages[page_indices[0]])

        result = _extract_rows_from_pages(pdf, page_indices)
        if not result:
            print(f"  {year}: no data rows extracted")
            return []

        raw_rows, _ = result
        rows = _assign_nivel(raw_rows)

        for row in rows:
            row["unidad"] = unit

        return rows


def write_csv(rows, year):
    out_path = OUT_DIR / f"gasto_programable_{year}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in OUTPUT_COLS})
    print(f"  {year}: {len(rows)} rows → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Extract CGPE gasto programable table to CSV")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--year", type=int, metavar="YYYY", help="Extract single year")
    group.add_argument("--all", action="store_true", help="Extract all available years (2012-2026)")
    return p.parse_args()


def main():
    args = parse_args()
    years = list(range(2012, 2027)) if args.all else [args.year]

    for year in years:
        rows = extract_year(year)
        if rows:
            write_csv(rows, year)


if __name__ == "__main__":
    main()
