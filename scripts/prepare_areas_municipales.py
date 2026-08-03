"""
Superficie municipal (km²) a partir del Marco Geoestadístico Nacional 2025 (INEGI).

Lee los 32 zips estatales en data/inegi/marco_geoestadistico/2025/ (shapefile
`<cve_ent>mun.shp`, polígono de límites municipales en proyección Lambert
Conformal Conic, EPSG:6372, metros — área calculable directo sin reproyectar).

Output: un CSV por entidad en data/inegi/marco_geoestadistico/areas_municipales/,
mismo nombre base que el zip fuente, columnas: cvegeo, municipio, area (km²).

Run: uv run --with pyshp --with shapely python scripts/prepare_areas_municipales.py
"""

import io
import zipfile
from pathlib import Path

import shapefile
from shapely.geometry import shape

DIR_ZIPS = Path("data/inegi/marco_geoestadistico/2025")
DIR_OUT = Path("data/inegi/marco_geoestadistico/areas_municipales")


def procesar_zip(zip_path: Path) -> list[tuple[str, str, float]]:
    cve_ent = zip_path.stem.split("_")[0]
    member_prefix = f"conjunto_de_datos/{cve_ent}mun"

    with zipfile.ZipFile(zip_path) as zf:
        shp = io.BytesIO(zf.read(f"{member_prefix}.shp"))
        shx = io.BytesIO(zf.read(f"{member_prefix}.shx"))
        dbf = io.BytesIO(zf.read(f"{member_prefix}.dbf"))

    sf = shapefile.Reader(shp=shp, shx=shx, dbf=dbf, encoding="latin-1")

    filas = []
    for sr in sf.iterShapeRecords():
        geom = shape(sr.shape.__geo_interface__)
        rec = sr.record.as_dict()
        filas.append((rec["CVEGEO"], rec["NOMGEO"].strip(), geom.area / 1e6))
    filas.sort(key=lambda r: r[0])
    return filas


def main():
    DIR_OUT.mkdir(parents=True, exist_ok=True)
    zips = sorted(DIR_ZIPS.glob("*.zip"))
    assert zips, f"No se encontraron zips en {DIR_ZIPS}"

    total_municipios = 0
    for zip_path in zips:
        filas = procesar_zip(zip_path)
        out_path = DIR_OUT / f"{zip_path.stem}.csv"
        with out_path.open("w", encoding="utf-8") as f:
            f.write("cvegeo,municipio,area\n")
            for cvegeo, municipio, area in filas:
                municipio_csv = municipio.replace('"', '""')
                if "," in municipio_csv:
                    municipio_csv = f'"{municipio_csv}"'
                f.write(f"{cvegeo},{municipio_csv},{area:.4f}\n")
        total_municipios += len(filas)
        print(f"{zip_path.name}: {len(filas)} municipios -> {out_path}")

    print(f"\nTotal: {len(zips)} entidades, {total_municipios} municipios")


if __name__ == "__main__":
    main()
