#!/usr/bin/env python3
"""
Build FIFA World Cup (1930-2022) position matrix from RSSSF data.

Position encoding:
  1  Champion
  2  Runner-up
  3  Third place
  4  Fourth place
  5  Quarterfinalist / second-round-group loser (equivalent stage)
  9  Round-of-16 loser / first-round KO loser
  17 Group-stage eliminate
  None  Did not participate
"""

import re
import time
import urllib.request
from html.parser import HTMLParser
import polars as pl

# ── URLs ──────────────────────────────────────────────────────────────────────
YEAR_URLS = {
    1930: "https://www.rsssf.org/tables/30full.html",
    1934: "https://www.rsssf.org/tables/34full.html",
    1938: "https://www.rsssf.org/tables/38full.html",
    1950: "https://www.rsssf.org/tables/50full.html",
    1954: "https://www.rsssf.org/tables/54full.html",
    1958: "https://www.rsssf.org/tables/58full.html",
    1962: "https://www.rsssf.org/tables/62full.html",
    1966: "https://www.rsssf.org/tables/66full.html",
    1970: "https://www.rsssf.org/tables/70full.html",
    1974: "https://www.rsssf.org/tables/74full.html",
    1978: "https://www.rsssf.org/tables/78full.html",
    1982: "https://www.rsssf.org/tables/82full.html",
    1986: "https://www.rsssf.org/tables/86full.html",
    1990: "https://www.rsssf.org/tables/90full.html",
    1994: "https://www.rsssf.org/tables/94full.html",
    1998: "https://www.rsssf.org/tables/98full.html",
    2002: "https://www.rsssf.org/tables/2002full.html",
    2006: "https://www.rsssf.org/tables/2006full.html",
    2010: "https://www.rsssf.org/tables/2010full.html",
    2014: "https://www.rsssf.org/tables/2014full.html",
    2018: "https://www.rsssf.org/tables/2018full.html",
    2022: "https://www.rsssf.org/tables/2022f.html",
}

# Ground truth for positions 1-4 (verified)
KNOWN_TOP4 = {
    1930: {1: "Uruguay",      2: "Argentina",      3: "Yugoslavia",     4: "United States"},
    1934: {1: "Italy",        2: "Czech Republic", 3: "Germany",        4: "Austria"},
    1938: {1: "Italy",        2: "Hungary",        3: "Brazil",         4: "Sweden"},
    1950: {1: "Uruguay",      2: "Brazil",         3: "Sweden",         4: "Spain"},
    1954: {1: "Germany",      2: "Hungary",        3: "Austria",        4: "Uruguay"},
    1958: {1: "Brazil",       2: "Sweden",         3: "France",         4: "Germany"},
    1962: {1: "Brazil",       2: "Czech Republic", 3: "Chile",          4: "Yugoslavia"},
    1966: {1: "England",      2: "Germany",        3: "Portugal",       4: "Russia"},
    1970: {1: "Brazil",       2: "Italy",          3: "Germany",        4: "Uruguay"},
    1974: {1: "Germany",      2: "Netherlands",    3: "Poland",         4: "Brazil"},
    1978: {1: "Argentina",    2: "Netherlands",    3: "Brazil",         4: "Italy"},
    1982: {1: "Italy",        2: "Germany",        3: "Poland",         4: "France"},
    1986: {1: "Argentina",    2: "Germany",        3: "France",         4: "Belgium"},
    1990: {1: "Germany",      2: "Argentina",      3: "Italy",          4: "England"},
    1994: {1: "Brazil",       2: "Italy",          3: "Sweden",         4: "Bulgaria"},
    1998: {1: "France",       2: "Brazil",         3: "Croatia",        4: "Netherlands"},
    2002: {1: "Brazil",       2: "Germany",        3: "Turkey",         4: "South Korea"},
    2006: {1: "Italy",        2: "France",         3: "Germany",        4: "Portugal"},
    2010: {1: "Spain",        2: "Netherlands",    3: "Germany",        4: "Uruguay"},
    2014: {1: "Germany",      2: "Argentina",      3: "Netherlands",    4: "Brazil"},
    2018: {1: "France",       2: "Croatia",        3: "Belgium",        4: "England"},
    2022: {1: "Argentina",    2: "France",         3: "Croatia",        4: "Morocco"},
}

# RSSSF 3-letter codes → modern country names (for 1930-1998 pages)
CODE_TO_NAME = {
    "ALG": "Algeria",          "ANG": "Angola",           "ARG": "Argentina",
    "AUS": "Australia",        "AUT": "Austria",          "BEL": "Belgium",
    "BOL": "Bolivia",          "BRA": "Brazil",           "BUL": "Bulgaria",
    "CAM": "Cameroon",         "CAN": "Canada",           "CHI": "Chile",
    "CMR": "Cameroon",         "COL": "Colombia",         "CRC": "Costa Rica",
    "CRI": "Costa Rica",       "CRO": "Croatia",          "CUB": "Cuba",
    "DDR": "East Germany",     "DEN": "Denmark",          "ECU": "Ecuador",
    "EGY": "Egypt",            "ENG": "England",          "ESP": "Spain",
    "FRA": "France",           "GER": "Germany",          "GHA": "Ghana",
    "GRE": "Greece",           "GUA": "Guatemala",        "HAI": "Haiti",
    "HOL": "Netherlands",      "HON": "Honduras",         "HUN": "Hungary",
    "INA": "Indonesia",        "IRA": "Iran",             "IRN": "Iran",
    "IRL": "Republic of Ireland", "IRQ": "Iraq",          "ISL": "Iceland",
    "ISR": "Israel",           "ITA": "Italy",            "IVC": "Ivory Coast",
    "JAM": "Jamaica",          "JPN": "Japan",            "JUG": "Yugoslavia",
    "KLD": "North Korea",      "KOR": "South Korea",      "KSA": "Saudi Arabia",
    "KUW": "Kuwait",           "MAR": "Morocco",          "MEX": "Mexico",
    "MLI": "Mali",             "NED": "Netherlands",      "NGA": "Nigeria",
    "NOR": "Norway",           "NZL": "New Zealand",      "PAR": "Paraguay",
    "PER": "Peru",             "POL": "Poland",           "POR": "Portugal",
    "PRK": "North Korea",      "ROM": "Romania",          "ROU": "Romania",
    "RSA": "South Africa",     "RUS": "Russia",           "SAL": "El Salvador",
    "SAU": "Saudi Arabia",     "SCO": "Scotland",         "SEN": "Senegal",
    "SLO": "Slovenia",         "SUI": "Switzerland",      "SWE": "Sweden",
    "TCH": "Czech Republic",   "TOG": "Togo",             "TRI": "Trinidad and Tobago",
    "TUN": "Tunisia",          "TUR": "Turkey",           "UKR": "Ukraine",
    "URU": "Uruguay",          "USA": "United States",    "WAL": "Wales",
    "YUG": "Yugoslavia",       "ZAI": "DR Congo",         "ZSR": "Russia",
    # Alternate codes found in specific RSSSF pages
    "CZE": "Czech Republic",   # 1938-1990 (used instead of TCH)
    "NIR": "Northern Ireland", # 1958, 1982, 1986, 1990
    "IHO": "Indonesia",        # Dutch East Indies, 1938 only
    "DAN": "Denmark",          # 1986, 1998 (instead of DEN)
    "IRK": "Iraq",             # 1986 (instead of IRQ)
    "COS": "Costa Rica",       # 1990 (instead of CRC)
    "EMI": "United Arab Emirates",  # 1990
    "JAP": "Japan",            # 1998 (instead of JPN)
    "SAF": "South Africa",     # 1998 (instead of RSA)
    "ARS": "Saudi Arabia",     # 1994, 1998 (instead of KSA)
    "RPA": "South Africa",     # 1998 group tables (instead of SAF)
}

# Full-name normalization (apply after stripping/uppercasing input)
NAME_NORMALIZE = {
    "NETHERLANDS": "Netherlands",
    "WEST GERMANY": "Germany",
    "SOVIET UNION": "Russia",
    "CZECHOSLOVAKIA": "Czech Republic",
    "YUGOSLAVIA": "Yugoslavia",
    "EAST GERMANY": "East Germany",
    "ZAIRE": "DR Congo",
    "COTE D'IVOIRE": "Ivory Coast",
    "CÔTE D'IVOIRE": "Ivory Coast",
    "IVORY COAST": "Ivory Coast",
    "TRINIDAD AND TOBAGO": "Trinidad and Tobago",
    "TRINIDAD & TOBAGO": "Trinidad and Tobago",
    "SAUDI ARABIA": "Saudi Arabia",
    "COSTA RICA": "Costa Rica",
    "SOUTH KOREA": "South Korea",
    "NORTH KOREA": "North Korea",
    "SOUTH AFRICA": "South Africa",
    "NEW ZEALAND": "New Zealand",
    "UNITED STATES": "United States",
    "CZECH REPUBLIC": "Czech Republic",
    "REPUBLIC OF IRELAND": "Republic of Ireland",
    "BOSNIA AND HERZEGOVINA": "Bosnia and Herzegovina",
    "BURKINA FASO": "Burkina Faso",
    "EL SALVADOR": "El Salvador",
    "DR CONGO": "DR Congo",
    "DEMOCRATIC REPUBLIC OF CONGO": "DR Congo",
    "CAPE VERDE": "Cape Verde",
    "EQUATORIAL GUINEA": "Equatorial Guinea",
    "CENTRAL AFRICAN REPUBLIC": "Central African Republic",
    "USA": "United States",
    "KOREA DPR": "North Korea",
    "KOREA REPUBLIC": "South Korea",
}

# ── HTML stripping ────────────────────────────────────────────────────────────

class _Stripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True
        elif tag in ("br", "p", "div", "tr", "h1", "h2", "h3", "h4", "li"):
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
        ct = r.headers.get("Content-Type", "")
    charset = "iso-8859-1"
    if "charset=" in ct:
        charset = ct.split("charset=")[-1].strip()
    html = raw.decode(charset, errors="replace")
    s = _Stripper()
    s.feed(html)
    return s.get_text()


# ── Name normalization ────────────────────────────────────────────────────────

def _norm_code(code: str) -> str | None:
    """3-letter RSSSF code → canonical country name."""
    return CODE_TO_NAME.get(code.upper())


def _norm_name(name: str) -> str:
    """Full country name (any case) → canonical name."""
    key = name.upper().strip()
    if key in NAME_NORMALIZE:
        return NAME_NORMALIZE[key]
    # Title-case as fallback
    return name.strip().title()


# ── Section splitting ─────────────────────────────────────────────────────────

# Headers that delimit sections in RSSSF pages
_HEADER_RE = re.compile(
    r"^[ \t]*(?:"
    r"(?:THE\s+)?Finals?"
    r"|Third.{0,6}place(?:.{0,12}match)?"
    r"|Semi.{0,2}finals?"
    r"|Quarter.{0,2}finals?|1/4.{0,6}finals?"
    r"|1/8.{0,6}finals?|Eightfinals?|Round of sixteen|Round of 16|ROUND OF 16"
    r"|Second\s+round|Second\s+Phase"
    r"|First\s+round|First\s+Phase"
    r"|Group [A-H]|Group [0-9]|FIRST ROUND"
    r")[ \t]*$",
    re.I | re.MULTILINE,
)


def _sections(text: str) -> list[tuple[str, str]]:
    """Return [(raw_header, section_body), ...] split on section headers."""
    boundaries = [(m.start(), m.end(), m.group(0).strip()) for m in _HEADER_RE.finditer(text)]
    if not boundaries:
        return [("", text)]
    result = []
    for i, (start, end, hdr) in enumerate(boundaries):
        body_end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        result.append((hdr, text[end:body_end]))
    return result


def _section_body(text: str, *keywords: str) -> str:
    """Return concatenated bodies of all sections matching any keyword.

    Handles pages with a TOC at the top (header appears twice: once in TOC with
    no content, once before actual match results).
    """
    kw_pats = [re.compile(kw, re.I) for kw in keywords]
    parts = [body for hdr, body in _sections(text) if any(p.search(hdr) for p in kw_pats)]
    return "\n".join(parts)


# ── Match-team extraction ─────────────────────────────────────────────────────

# Old format: CODE - CODE score:score
_OLD_MATCH_RE = re.compile(
    r"\b([A-Z]{2,4})\s*-\s*([A-Z]{2,4})\s+\d+\s*[:.]\s*\d+"
)

# New format (2002+): TEAMNAME  score  (one team per line)
_NEW_MATCH_LINE_RE = re.compile(
    r"^([A-Z][A-Za-zÀ-ž \t'\".()&/-]{0,40}?)\s{2,}\d+",
    re.MULTILINE,
)


def _old_section_teams(body: str) -> set[str]:
    """Extract normalized names from old-format match lines in a section body."""
    teams: set[str] = set()
    for m in _OLD_MATCH_RE.finditer(body):
        for code in (m.group(1), m.group(2)):
            name = _norm_code(code)
            if name:
                teams.add(name)
    return teams


def _new_section_teams(body: str) -> set[str]:
    """Extract normalized names from new-format knockout match lines.

    In 2002-2022 pages, team names in knockout match lines are in ALL CAPS.
    Title-case entries (e.g. penalty-shootout kicker names) are filtered out.
    """
    teams: set[str] = set()
    for m in _NEW_MATCH_LINE_RE.finditer(body):
        raw = m.group(1).strip()
        # Skip rank-prefixed group table entries
        if re.match(r"^\d+\.", raw):
            continue
        # Only accept ALL CAPS names (knockout match format; skip scorers, kickers, refs)
        if not re.match(r"^[A-ZÀ-Ž][A-ZÀ-Ž\s'-]{1,}$", raw):
            continue
        teams.add(_norm_name(raw))
    return teams


# ── Group-table extraction ────────────────────────────────────────────────────

# Old format: "  1. GER^  3 1 2 0  4 5-1"
_OLD_GROUP_LINE_RE = re.compile(
    r"^\s*(\d+)[.]\s+([A-Z]{2,4})(\^?)\s",
    re.MULTILINE,
)

# New format: " 1.GERMANY  3  3  0  0  8  2  9"
# Note: advanced = ALL CAPS, eliminated = Title Case
_NEW_GROUP_LINE_RE = re.compile(
    r"^\s*(\d+)[.]\s*([A-Z][A-Za-zÀ-ž \t'\".()&/-]{0,40}?)\s{2,}\d",
    re.MULTILINE,
)


def _digit_group_teams(text: str) -> set[str]:
    """Extract all teams from sections with digit-labeled group headers (second-round groups).

    1974/1978/1982 use 'Group 1', 'Group 2', ... for second-round groups.
    """
    teams: set[str] = set()
    for hdr, body in _sections(text):
        if re.match(r"^Group\s+[0-9]", hdr, re.I):
            teams |= _old_section_teams(body)
            adv, elim = _old_group_teams(body)
            teams |= adv | elim
    return teams


def _old_group_teams(text: str) -> tuple[set[str], set[str]]:
    """Return (advanced, eliminated) sets from old-format group tables."""
    advanced: set[str] = set()
    eliminated: set[str] = set()
    for m in _OLD_GROUP_LINE_RE.finditer(text):
        code = m.group(2)
        name = _norm_code(code)
        if not name:
            continue
        if m.group(3) == "^":
            advanced.add(name)
        else:
            eliminated.add(name)
    return advanced, eliminated


def _new_group_teams(text: str) -> tuple[set[str], set[str]]:
    """Return (advanced, eliminated) from new-format group tables.

    Convention on RSSSF new-format pages: teams that advanced to knockout
    rounds appear in ALL CAPS in the group table; eliminated teams are
    in Title Case.
    """
    advanced: set[str] = set()
    eliminated: set[str] = set()
    for m in _NEW_GROUP_LINE_RE.finditer(text):
        raw = m.group(2).strip()
        name = _norm_name(raw)
        if raw.isupper():
            advanced.add(name)
        else:
            eliminated.add(name)
    return advanced, eliminated


# ── Per-year position builder ─────────────────────────────────────────────────

def build_year(year: int, text: str) -> dict[str, int]:
    old = year <= 1998
    positions: dict[str, int] = {}

    # Apply ground-truth top 4
    top4 = KNOWN_TOP4[year]
    top4_names = set(top4.values())
    for pos, name in top4.items():
        positions[name] = pos

    # Helper: extract teams from a named section
    def sec_teams(text: str, *kws: str) -> set[str]:
        body = _section_body(text, *kws)
        if not body:
            return set()
        return _old_section_teams(body) if old else _new_section_teams(body)

    # ── Special-era handling ──────────────────────────────────────────────────

    if year in {1974, 1978}:
        # Format: 4 first-round groups → 2 second-round groups (of 4) → Final+3rd
        # 1974 uses a "Second Phase" header; 1978 has inline "Group 1 - CODE CODE CODE" listings
        sr_body = _section_body(text, r"Second\s+phase", r"Second\s+round")
        if sr_body:
            sr_adv, sr_elim = _old_group_teams(sr_body)
            sr_all = sr_adv | sr_elim
        else:
            sr_all = _digit_group_teams(text)
        if not sr_all:
            # 1978: second-round groups listed inline as "Group 1 - CODE CODE CODE CODE"
            for m in re.finditer(r'^Group\s+[0-9]+\s*[-]\s*([A-Z]{3}(?:\s+[A-Z]{3})+)',
                                 text, re.MULTILINE):
                for code in m.group(1).split():
                    name = _norm_code(code)
                    if name:
                        sr_all.add(name)
        first_adv, first_elim = _old_group_teams(text)
        all_teams = first_adv | first_elim | sr_all | top4_names
        for name in all_teams:
            if name in positions:
                continue
            if name in sr_all:
                if name not in top4_names:
                    positions[name] = 5
            else:
                positions[name] = 9
        return positions

    if year == 1982:
        # Format: 6 first-round groups (24 teams) → 4 second-round groups (12 teams) → SF → Final
        # 1982 uses "Second  phase" (double-space) header, falling back to digit groups
        sr_body = _section_body(text, r"Second\s+phase", r"Second\s+round")
        if sr_body:
            sr_adv, sr_elim = _old_group_teams(sr_body)
            second_round_all = sr_adv | sr_elim
        else:
            second_round_all = _digit_group_teams(text)
        first_adv, first_elim = _old_group_teams(text)
        all_teams = first_adv | first_elim | second_round_all | top4_names
        for name in all_teams:
            if name in positions:
                continue
            if name in second_round_all:
                positions[name] = 5
            else:
                positions[name] = 17
        return positions

    if year in {1934, 1938}:
        # Pure knockout: R1 (8 matches) → QF → SF → 3rd + Final
        qf_teams = sec_teams(text, r"Quarter")
        r1_teams = sec_teams(text, r"1/8|Eightfinal|First round|Round 1|Round of 16")
        # Also grab from semifinals to make sure SF teams aren't labeled as QF losers
        sf_teams = sec_teams(text, r"Semi")
        final_teams = sec_teams(text, r"^Final$", r"^THE FINAL$")
        all_ko = r1_teams | qf_teams | sf_teams | final_teams | top4_names
        for name in all_ko:
            if name in positions:
                continue
            if name in qf_teams:
                positions[name] = 5
            else:
                positions[name] = 9
        return positions

    if year == 1950:
        # 4 groups → 4-team final round-robin. Group eliminates = pos 9.
        _adv, elim = _old_group_teams(text)
        for name in elim:
            if name not in positions:
                positions[name] = 9
        return positions

    if year == 1930:
        # 4 groups → Semis → Final (no 3rd place match played officially)
        # Group eliminates = pos 9; semi losers already in KNOWN_TOP4
        _adv, elim = _old_group_teams(text)
        for name in elim:
            if name not in positions:
                positions[name] = 9
        return positions

    # ── Standard format (1954-1970, 1986-2022) ───────────────────────────────
    # Groups → (R16 from 1986) → QF → SF → 3rd + Final

    qf_teams = sec_teams(text, r"Quarter")
    r16_teams = sec_teams(text, r"1/8|Eightfinal|Round of 16|Round of sixteen|Second\s+round")
    sf_teams = sec_teams(text, r"Semi")

    if old:
        # Old format (1930-1998): R16 section exists in pages that have it (1986+)
        grp_adv, grp_elim = _old_group_teams(text)
        all_group = grp_adv | grp_elim
        has_r16 = bool(r16_teams)
        for name in all_group | r16_teams | qf_teams | sf_teams | top4_names:
            if name in positions:
                continue
            if name in qf_teams:
                positions[name] = 5
            elif has_r16 and name in r16_teams:
                positions[name] = 9
            elif name in all_group:
                positions[name] = 9 if not has_r16 else 17
    else:
        # New format (2002+): R16 section header is missing on many pages (2010+).
        # Infer R16 losers from group advancement instead.
        group_text = "\n".join(
            body for hdr, body in _sections(text)
            if re.match(r"^Group\s+[A-H]$", hdr, re.I)
        )
        grp_adv, grp_elim = _new_group_teams(group_text)
        # Some pages (2006, 2022) use formats where ALL CAPS extraction fails.
        # Try three score-adjacent patterns to avoid false positives (referee nationalities etc):
        #   A) "TeamName Score"  at line start (2006 format)
        #   B) "  TeamName  Score"  padded (2022 left-side team)
        #   C) "score-score TeamName"  (2022 right-side team)
        if not qf_teams and grp_adv:
            qf_body = _section_body(text, r"Quarter")
            for name in grp_adv | top4_names:
                esc = re.escape(name)
                if (re.search(r'^' + esc + r'\s+\d', qf_body, re.MULTILINE) or
                        re.search(r'\s{2,}' + esc + r'\s+\d', qf_body) or
                        re.search(r'\d-\d\s+' + esc + r'\b', qf_body)):
                    qf_teams.add(name)
        for name in grp_elim:
            if name not in positions:
                positions[name] = 17
        for name in grp_adv:
            if name in positions:
                continue
            positions[name] = 5 if name in qf_teams else 9

    return positions


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    all_results: dict[int, dict[str, int]] = {}

    for year, url in sorted(YEAR_URLS.items()):
        print(f"  {year} ...", end=" ", flush=True)
        try:
            text = _fetch(url)
            result = build_year(year, text)
            all_results[year] = result
            n = len(result)
            champion = result.get(KNOWN_TOP4[year][1], "?")
            print(f"{n} teams  (champion: {KNOWN_TOP4[year][1]})")
        except Exception as exc:
            print(f"ERROR: {exc}")
            all_results[year] = KNOWN_TOP4[year]
        time.sleep(0.8)

    # Build wide-format matrix
    all_countries = sorted({c for r in all_results.values() for c in r})
    years = sorted(all_results.keys())
    year_strs = [str(y) for y in years]

    data: dict[str, list] = {"country": all_countries}
    for y, ys in zip(years, year_strs):
        yr = all_results[y]
        data[ys] = [yr.get(c) for c in all_countries]

    df = pl.DataFrame(
        data,
        schema={"country": pl.String, **{ys: pl.Int8 for ys in year_strs}},
    )
    out = "data/worldcup_positions.csv"
    df.write_csv(out)
    print(f"\nSaved {out}: {len(all_countries)} countries × {len(years)} editions")

    # Quick sanity check
    _sanity(df, year_strs)


def _sanity(df: pl.DataFrame, year_strs: list[str]) -> None:
    print("\nSanity check — champions per edition:")
    for ys in year_strs:
        champs = df.filter(pl.col(ys) == 1)["country"].to_list()
        expected = KNOWN_TOP4[int(ys)][1]
        ok = "✓" if champs == [expected] else f"✗ expected {expected}, got {champs}"
        print(f"  {ys}: {champs[0] if champs else '??'}  {ok}")


if __name__ == "__main__":
    main()
