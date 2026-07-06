# /// script
# requires-python = ">=3.11"
# dependencies = ["polars"]
# ///
"""
Parse places.xml and prices.xml from the gasolina dataset and join them
into a single enriched CSV: place_id, name, cre_id, lon, lat, regular, premium, diesel.

Usage:
    uv run scripts/enrich_gasolina.py [prices_xml] [places_xml] [output_csv]
"""

import sys
import xml.etree.ElementTree as ET
import polars as pl


def parse_places(path: str) -> pl.DataFrame:
    root = ET.parse(path).getroot()
    rows = []
    for place in root.findall("place"):
        loc = place.find("location")
        rows.append({
            "place_id": int(place.get("place_id")),
            "name": (place.findtext("name") or "").strip(),
            "cre_id": (place.findtext("cre_id") or "").strip(),
            "lon": float(loc.findtext("x")) if loc is not None and loc.findtext("x") else None,
            "lat": float(loc.findtext("y")) if loc is not None and loc.findtext("y") else None,
        })
    return pl.DataFrame(rows)


def parse_prices(path: str) -> pl.DataFrame:
    root = ET.parse(path).getroot()
    rows = []
    for place in root.findall("place"):
        pid = int(place.get("place_id"))
        for gp in place.findall("gas_price"):
            fuel_type = gp.get("type")
            price = float(gp.text) if gp.text else None
            rows.append({"place_id": pid, "fuel_type": fuel_type, "price": price})
    df = pl.DataFrame(rows)
    # Deduplicate in case place_id appears in multiple <place> elements for the same fuel type
    df = df.unique(subset=["place_id", "fuel_type"])
    return df.pivot(on="fuel_type", index="place_id", values="price")


def main():
    args = sys.argv[1:]
    from datetime import date
    today = date.today().strftime("%Y%m%d")
    prices_path = args[0] if len(args) > 0 else f"data/gasolina/{today}/prices.xml"
    places_path = args[1] if len(args) > 1 else f"data/gasolina/{today}/places.xml"
    output_path = args[2] if len(args) > 2 else f"data/gasolina/{today}/gasolina_enriched.csv"

    places = parse_places(places_path)
    prices = parse_prices(prices_path)

    result = prices.join(places, on="place_id", how="left")

    # Ensure consistent column order; fill missing fuel columns with null
    fuel_cols = ["regular", "premium", "diesel"]
    for col in fuel_cols:
        if col not in result.columns:
            result = result.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    result = result.select(["place_id", "name", "cre_id", "lon", "lat"] + fuel_cols)
    result = result.sort("place_id")
    result.write_csv(output_path)
    print(f"Wrote {result.height} rows x {result.width} columns → {output_path}")


if __name__ == "__main__":
    main()
